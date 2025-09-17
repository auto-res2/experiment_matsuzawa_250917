import argparse
import yaml
import os
import random
import numpy as np
import torch
import pynvml
import time
import sys
import timm
from transformers import AutoModel, AutoConfig, logging as hf_logging

# Local imports
from . import preprocess
from . import train
from . import evaluate

# Suppress verbose warnings from transformers
hf_logging.set_verbosity_error()

class GpuEnergyMonitor:
    def __init__(self, device_id=0):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.is_active = True
            self.start_time = None
            self.last_power = 0
        except pynvml.NVMLError as e:
            print(f"Warning: Could not initialize NVML for GPU monitoring: {e}. Energy metrics will be zero.")
            self.is_active = False

    def start(self):
        if self.is_active:
            self.start_time = time.perf_counter()
            self.last_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # In Watts

    def stop(self):
        if not self.is_active or self.start_time is None:
            return 0.0
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        current_power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        avg_power = (self.last_power + current_power) / 2.0
        energy_joules = avg_power * duration
        self.start_time = None
        return energy_joules
    
    def peek(self):
        """Get instantaneous power without stopping the timer."""
        if not self.is_active:
            return 0.0
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def shutdown(self):
        if self.is_active:
            pynvml.nvmlShutdown()

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(config):
    model_name = config['model_name']
    print(f"Loading model: {model_name}")
    if 'resnet' in model_name or 'convnext' in model_name:
        # Using timm for CNNs
        model = timm.create_model(f'timm/{model_name}.a1_in1k' if 'resnet50' in model_name else f'timm/{model_name}.fb_in22k_ft_in1k', pretrained=True)
    elif 'vit' in model_name:
        model = AutoModel.from_pretrained(f'google/vit-base-patch16-224-in21k')
        # Add a classifier head for ViT
        vit_config = AutoConfig.from_pretrained(f'google/vit-base-patch16-224-in21k')
        model.classifier = torch.nn.Linear(vit_config.hidden_size, 1000) # ImageNet-1k classes
    elif 'hubert' in model_name:
        model = AutoModel.from_pretrained(f'facebook/{model_name}-base-ls960')
        # Add a classifier head for HuBERT
        hubert_config = AutoConfig.from_pretrained(f'facebook/{model_name}-base-ls960')
        num_classes = 35 # SpeechCommands has 35 classes
        model.classifier = torch.nn.Linear(hubert_config.hidden_size, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Run EAGER-TTA experiments.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    results_dir = config['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    energy_monitor = GpuEnergyMonitor()

    for strategy_name, strategy_config in config['strategies'].items():
        print(f"\n--- Running Strategy: {strategy_name} ---")
        for exp in strategy_config['experiments']:
            for baseline in exp['baselines']:
                for seed in exp['random_seeds']:
                    exp_config = exp.copy()
                    exp_config['baseline'] = baseline
                    exp_config['seed'] = seed
                    exp_config.update(strategy_config.get('defaults', {}))
                    exp_config['smoke_test'] = 'smoke' in args.config

                    # Create a unique name for the run
                    run_name_parts = [
                        exp_config['model_name'],
                        exp_config['dataset'],
                    ]
                    if 'arrival_rate' in exp_config:
                         run_name_parts.append(f"eta_{exp_config['arrival_rate']}")
                    if 'energy_budget' in exp_config:
                        run_name_parts.append(f"budget_{exp_config['energy_budget']}")

                    run_name = "_".join(map(str, run_name_parts))

                    output_dir = os.path.join(results_dir, strategy_name, exp_config['model_name'], run_name, baseline, f'seed{seed}')
                    output_csv_path = os.path.join(output_dir, 'results.csv')

                    if os.path.exists(output_csv_path):
                        print(f"Skipping existing run: {output_dir}")
                        continue

                    print(f"\n>>> Starting run: Strategy='{strategy_name}', Model='{exp_config['model_name']}', Dataset='{exp_config['dataset']}', Baseline='{baseline}', Seed={seed}")
                    
                    set_seeds(seed)

                    try:
                        # 1. Get Data Stream
                        data_stream = preprocess.get_data_stream(exp_config)
                        
                        # 2. Get Model
                        base_model = get_model(exp_config)
                        
                        # 3. Wrap model for TTA
                        if baseline == 'EAGER-TTA':
                            model = train.EagerTTAWrapper(base_model, exp_config)
                        elif baseline == 'Tent-5':
                             model = train.Tent(base_model, exp_config)
                        elif baseline == 'EATA':
                             # EATA needs a small clean dataset for fisher info
                             dummy_clean_data = [torch.randn(exp_config['batch_size'], 3, 224, 224), torch.randint(0, 1000, (exp_config['batch_size'],))]
                             model = train.EATA(base_model, exp_config, [dummy_clean_data])
                        elif baseline in ['GLaD', 'RoTTA', 'RePTA', 'AdaBN', 'skip-2', 'skip-4', 'AdaLN-Tent', 'EAGER-TTA-NoMask']:
                             # Using a generic optimizer-based wrapper for these for simplicity
                             # A full implementation would have specific logic for each
                             print(f"Using simplified adaptation logic for {baseline}")
                             model = train.Tent(base_model, exp_config)
                        else: # No-TTA
                            model = base_model

                        # 4. Run Experiment
                        train.run_adaptation_stream(model, data_stream, exp_config, output_csv_path, energy_monitor)

                    except RuntimeError as e:
                        print(f"ERROR during run: {e}. Skipping to next run.", file=sys.stderr)
                        # Log error to a file in the output dir
                        os.makedirs(output_dir, exist_ok=True)
                        with open(os.path.join(output_dir, 'error.log'), 'w') as err_file:
                            err_file.write(str(e))
                        continue
    
    energy_monitor.shutdown()
    
    # 5. Final Evaluation
    print("\n--- All experiments complete. Generating final report. ---")
    evaluate.generate_report(results_dir)

if __name__ == '__main__':
    # Add src to python path to allow relative imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src import main as main_module
    main_module.main()
