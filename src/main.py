import argparse
import yaml
import os
import logging
import copy
from itertools import product
import torch.multiprocessing as mp

from .preprocess import load_and_preprocess_data
from .train import run_training
from .evaluate import generate_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_config_path():
    parser = argparse.ArgumentParser(description="Run GNN experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')
    args = parser.parse_args()

    if args.smoke_test:
        return 'config/smoke_test.yaml', 'smoke_test'
    elif args.full_experiment:
        return 'config/full_experiment.yaml', 'full_experiment'

def run_experiment(exp_config, global_config, exp_key):
    world_size = len(global_config.get('gpus', [0]))

    # Training phase
    for model_config in exp_config['models']:
        for dataset in exp_config['datasets']:
            for seed in global_config['seeds']:
                trial_config = copy.deepcopy(global_config)
                trial_config.update(exp_config)
                trial_config['model'] = model_config
                trial_config['dataset'] = dataset
                trial_config['globals']['seeds'] = [seed] # Use one seed per run
                
                model_name = model_config['name']
                exp_name = f"{exp_key}_{model_name}_{dataset.replace('-', '_')}_{seed}"
                trial_config['experiment_name'] = exp_name

                logging.info(f"--- Starting Trial: {exp_name} ---")
                
                # 1. Preprocess data
                data_path = load_and_preprocess_data(dataset)
                
                # 2. Run training
                if world_size > 1:
                    mp.spawn(run_training, args=(world_size, trial_config, data_path), nprocs=world_size, join=True)
                else:
                    run_training(0, 1, trial_config, data_path)
                logging.info(f"--- Finished Trial: {exp_name} ---")

def main():
    config_path, run_type = get_config_path()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    global_config = config.get('globals', {})
    os.makedirs(global_config.get('output_dir', '.research/iteration1'), exist_ok=True)

    if run_type == 'smoke_test':
        logging.info("========== Running Smoke Test ==========")
        exp_config = config['smoke_test_exp']
        run_experiment(exp_config, global_config, 'smoke_test')
        logging.info("========== Smoke Test Passed ==========")
        # In a real CI/CD, you might proceed to the full experiment here.
        # For this script, we just run what's requested.
    
    elif run_type == 'full_experiment':
        for exp_key in ['experiment_1', 'experiment_3']: # Exp2 is placeholder
            if exp_key in config:
                logging.info(f"========== Running {exp_key} ==========")
                exp_config = config[exp_key]
                run_experiment(exp_config, global_config, exp_key)
                logging.info(f"========== Finished {exp_key} ==========")
                
                # Evaluation phase after each experiment block
                logging.info(f"========== Evaluating {exp_key} ==========")
                eval_config = {'globals': global_config, exp_key: exp_config}
                generate_results(eval_config)
                logging.info(f"========== Evaluation for {exp_key} complete ==========")

if __name__ == '__main__':
    main()
