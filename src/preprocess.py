import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import numpy as np
from datasets import load_dataset, Audio
import time
import random
import torchaudio

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Vision Preprocessing ---
def get_vision_transform(size=224):
    return transforms.Compose([
        transforms.Resize(size if isinstance(size, int) else size[0] * 8 // 7), # Resize shorter edge
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# --- Audio Preprocessing ---
def get_audio_transform():
    return transforms.Compose([
        torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000), # Ensure 16kHz
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128, n_fft=400, hop_length=160),
        torchaudio.transforms.AmplitudeToDB(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # To 3 channels for some models
        transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    ])

class CorruptedDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Stream Generators ---

def imagenet_c_stream_generator(config):
    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    severities = [1, 2, 3, 4, 5]
    transform = get_vision_transform()
    
    random.seed(config['seed'])
    stream_order = []
    for _ in range(int(np.ceil(config['stream_length_steps'] / (5 * len(corruption_types))))):
        random.shuffle(corruption_types)
        for corruption in corruption_types:
            random.shuffle(severities)
            for severity in severities:
                 stream_order.append((corruption, severity))
    
    step = 0
    for corruption, severity in stream_order:
        if step >= config['stream_length_steps']:
            break
        try:
            dataset_name = f'imagenet-c--{corruption}-{severity}'
            dset = load_dataset('hendrycks/imagenet-c', name=corruption, split=f'test_severity_{severity}', streaming=True)
            dset = dset.with_transform(lambda examples: {'image': [transform(img.convert('RGB')) for img in examples['image']], 'label': examples['label']})
        except Exception as e:
            raise RuntimeError(f"Dataset ang9867/ImageNet-C with corruption {corruption} unavailable. Aborting experiment. Error: {e}")

        loader = DataLoader(dset, batch_size=config['batch_size'])
        for x, y in loader:
            if step >= config['stream_length_steps']:
                return
            yield x, y
            if 'arrival_rate' in config:
                time.sleep(config['arrival_rate'])
            step += 1

def realstream_1m_generator(config):
    print("Initializing RealStream-1M generator...")
    transform = get_vision_transform()
    batch_size = config['batch_size']

    try:
        # Clean data from imagenet-1k validation set
        clean_dset = load_dataset('imagenet-1k', split='validation', use_auth_token=True).with_transform(lambda e: {'image': [transform(img.convert('RGB')) for img in e['image']], 'label': e['label']})
        clean_loader = iter(DataLoader(clean_dset, batch_size=batch_size, shuffle=True))

        # Corrupted data
        mild_dsets = [load_dataset('hendrycks/imagenet-c', name=c, split=f'test_severity_{s}', streaming=True).with_transform(lambda e: {'image': [transform(img.convert('RGB')) for img in e['image']], 'label': e['label']}) for c in ['gaussian_noise', 'shot_noise'] for s in [1, 2]]
        severe_dsets = [load_dataset('hendrycks/imagenet-c', name=c, split=f'test_severity_5', streaming=True).with_transform(lambda e: {'image': [transform(img.convert('RGB')) for img in e['image']], 'label': e['label']}) for c in ['contrast', 'motion_blur']]
        mild_loaders = [iter(DataLoader(d, batch_size=batch_size)) for d in mild_dsets]
        severe_loaders = [iter(DataLoader(d, batch_size=batch_size)) for d in severe_dsets]
    except Exception as e:
        raise RuntimeError(f"Could not load datasets for RealStream-1M. Aborting. Error: {e}")

    stream_phase = ['clean', 'mild', 'severe', 'clean']
    phase_idx = 0
    steps_in_phase = 0
    dwell_time = random.randint(400, 800)

    for step in range(config['stream_length_steps']):
        current_phase = stream_phase[phase_idx]
        try:
            if current_phase == 'clean':
                x, y = next(clean_loader)
            elif current_phase == 'mild':
                x, y = next(random.choice(mild_loaders))
            else: # severe
                x, y = next(random.choice(severe_loaders))
        except StopIteration:
            # Reset iterators if exhausted
            clean_loader = iter(DataLoader(clean_dset, batch_size=batch_size, shuffle=True))
            mild_loaders = [iter(DataLoader(d, batch_size=batch_size)) for d in mild_dsets]
            severe_loaders = [iter(DataLoader(d, batch_size=batch_size)) for d in severe_dsets]
            # Retry getting batch
            if current_phase == 'clean': x, y = next(clean_loader)
            elif current_phase == 'mild': x, y = next(random.choice(mild_loaders))
            else: x, y = next(random.choice(severe_loaders))

        yield x, y

        steps_in_phase += 1
        if steps_in_phase >= dwell_time:
            phase_idx = (phase_idx + 1) % len(stream_phase)
            steps_in_phase = 0
            dwell_time = random.randint(400, 800)
            print(f"\nStep {step}: Switching to phase '{stream_phase[phase_idx]}'")

def get_data_stream(config):
    dataset_name = config['dataset']
    print(f"Loading dataset: {dataset_name}")

    if dataset_name == 'ImageNet-C':
        return imagenet_c_stream_generator(config)
    elif dataset_name == 'RealStream-1M':
        return realstream_1m_generator(config)
    
    # Dataloader-based datasets
    try:
        if dataset_name == 'SpeechCommands-C':
            # Create corrupted version by adding noise
            dset = load_dataset('google/speech_commands', 'v0.02', split='validation')
            dset = dset.cast_column("audio", Audio(sampling_rate=16_000))
            transform = get_audio_transform()
            
            def add_noise(batch):
                audio_arrays = [x["array"] for x in batch["audio"]]
                # Add moderate Gaussian noise to simulate corruption
                noisy_audio = [torch.from_numpy(arr) + 0.01 * torch.randn(len(arr)) for arr in audio_arrays]
                processed_audio = [transform(a.unsqueeze(0)) for a in noisy_audio]
                return {"input_values": processed_audio, "labels": batch["label"]}
            
            dset.set_transform(add_noise)
            return DataLoader(dset, batch_size=config['batch_size'], collate_fn=lambda b: {"input_values": torch.stack([x['input_values'] for x in b]), "labels": torch.tensor([x['labels'] for x in b])})

        elif dataset_name == 'Camelyon17-WILDS':
            dset = load_dataset('wltjr1007/Camelyon17-WILDS', split='test')
            transform = get_vision_transform(size=96)
            dset = CorruptedDataset(dset, transform)
            return DataLoader(dset, batch_size=config['batch_size'], shuffle=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        raise RuntimeError(f"Dataset {dataset_name} unavailable. Aborting experiment per policy. Error: {e}")
