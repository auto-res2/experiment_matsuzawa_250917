import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoConfig
import numpy as np
import time
import os
from collections import deque, defaultdict
import math

# --- EAGER-TTA Components ---

class KalmanFilter:
    """A simple, non-learnable Kalman Filter for 6D statistics."""
    def __init__(self, process_noise=1e-3, measurement_noise=1e-3, device='cuda'):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_dim = 6
        self.device = device
        self.reset()

    def reset(self):
        self.x = torch.zeros(self.state_dim, device=self.device)
        self.P = torch.eye(self.state_dim, device=self.device)

    def predict(self):
        self.x = self.x  # No state transition model (assume static)
        self.P = self.P + self.process_noise * torch.eye(self.state_dim, device=self.device)

    def update(self, z):
        K = self.P / (self.P + self.measurement_noise)
        self.x = self.x + torch.diag(K) * (z - self.x)
        self.P = (torch.eye(self.state_dim, device=self.device) - K) * self.P

    def get_log_likelihood_ratio(self, z):
        # Simplified log-likelihood ratio for change-point detection
        residual = z - self.x
        # Using diagonal of P for simplicity
        innovation_covariance = torch.diag(self.P) + self.measurement_noise
        log_likelihood_p1 = -0.5 * torch.sum(torch.log(2 * torch.pi * innovation_covariance) + (residual ** 2) / innovation_covariance)
        # Likelihood under None hypothesis (z is from the same distribution as x)
        log_likelihood_p0 = -0.5 * torch.sum(torch.log(2 * torch.pi * self.measurement_noise) + ((z - self.x) ** 2) / self.measurement_noise)
        # Avoid nan for p0 when z=x
        log_likelihood_p0 = torch.nan_to_num(log_likelihood_p0, nan=-1e9)
        return log_likelihood_p1 - log_likelihood_p0

class TinyGRU(nn.Module):
    """TinyGRU gate for cross-time evidence pooling."""
    def __init__(self, input_dim=1, hidden_dim=12, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.hidden = None

    def forward(self, llr_sequence):
        # llr_sequence shape: (batch_size, window_size, 1)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()
        output, self.hidden = self.gru(llr_sequence, self.hidden)
        utility_score = torch.sigmoid(self.fc(output[:, -1, :]))
        return utility_score

class MaskedHyperNetwork(nn.Module):
    """Generates masked low-rank updates for normalization layers."""
    def __init__(self, input_dim=6, norm_layer_dim=64, rank=8, mask_sparsity=0.15):
        super().__init__()
        self.rank = rank
        self.norm_layer_dim = norm_layer_dim
        self.mask_sparsity = mask_sparsity

        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * norm_layer_dim * rank)
        )
        self.mask_logits = nn.Parameter(torch.randn(norm_layer_dim))

    def forward(self, stats):
        params = self.hypernet(stats)
        u = params[:, :self.norm_layer_dim * self.rank].view(-1, self.norm_layer_dim, self.rank)
        v = params[:, self.norm_layer_dim * self.rank:].view(-1, self.norm_layer_dim, self.rank)
        
        if self.training:
            mask = F.gumbel_softmax(self.mask_logits.unsqueeze(0).expand(stats.size(0), -1), tau=1, hard=True, dim=-1)
        else:
            # STE for inference
            mask = (self.mask_logits > 0).float().unsqueeze(0).expand(stats.size(0), -1)

        return u, v, mask

class EnergyAwareScheduler:
    """Makes adaptation decisions based on utility and energy budget."""
    def __init__(self, budget_factor=1.1, ema_alpha=0.1):
        self.base_energy = None
        self.energy_budget = None
        self.budget_factor = budget_factor
        self.ema_energy = None
        self.ema_alpha = ema_alpha

    def set_base_energy(self, base_energy):
        self.base_energy = base_energy
        self.energy_budget = self.base_energy * self.budget_factor
        self.ema_energy = self.base_energy

    def decide(self, utility_score, current_energy):
        if self.base_energy is None:
            return True # Adapt during warmup

        self.ema_energy = (1 - self.ema_alpha) * self.ema_energy + self.ema_alpha * current_energy
        delta_e = max(0, self.ema_energy - self.base_energy)
        
        # Shadow price lambda: increases as we approach the budget
        shadow_price = torch.exp(5.0 * (self.ema_energy / self.energy_budget - 0.95))
        
        should_adapt = (utility_score - shadow_price * delta_e) > 0
        return should_adapt.item()


def get_6d_stats(x):
    """Extracts 6D statistics from a tensor (B, C, H, W) or (B, N, C)."""
    if x.dim() == 4: # CNN
        x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
    elif x.dim() == 3: # Transformer
        x = x.reshape(-1, x.size(2))
    
    mean = x.mean(0)
    std = x.std(0)
    skew = torch.mean(((x - mean) / (std + 1e-5)) ** 3, dim=0)
    kurt = torch.mean(((x - mean) / (std + 1e-5)) ** 4, dim=0) - 3.0
    # Use min/max as simpler proxies for range
    min_val, _ = x.min(0)
    max_val, _ = x.max(0)
    stats = torch.stack([mean, std, skew, kurt, min_val, max_val], dim=1)
    return stats.mean(0) # Average over channels

# --- Model Wrappers ---

class EagerTTAWrapper(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.norm_layers = []
        self.hooks = []
        self.layer_stats = {}

        self.kalman_filters = {}
        self.gru_gates = {}
        self.hypernetworks = {}
        self.llr_history = defaultdict(lambda: deque(maxlen=config['eager_params']['gru_window']))
        self.scheduler = EnergyAwareScheduler(budget_factor=config.get('energy_budget', 1.1))
        
        self._find_and_instrument_norm_layers()
        self._initialize_eager_components()

        # Dummy initialization for meta-trained components
        # In a real scenario, these would be loaded from a checkpoint
        self.run_meta_training(None) # Creates dummy weights if not present

    def _find_and_instrument_norm_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self.norm_layers.append((name, module))
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
    
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.layer_stats[name] = get_6d_stats(input[0].detach())
        return hook

    def _initialize_eager_components(self):
        for name, norm_layer in self.norm_layers:
            if isinstance(norm_layer, (nn.BatchNorm2d, nn.GroupNorm)):
                norm_dim = norm_layer.num_features
            elif isinstance(norm_layer, nn.LayerNorm):
                norm_dim = norm_layer.normalized_shape[0]

            self.kalman_filters[name] = KalmanFilter(device=self.device, **self.config['eager_params']['kalman'])
            self.gru_gates[name] = TinyGRU(window_size=self.config['eager_params']['gru_window']).to(self.device)
            self.hypernetworks[name] = MaskedHyperNetwork(norm_layer_dim=norm_dim, **self.config['eager_params']['hypernet']).to(self.device)
        self.gru_gates = nn.ModuleDict(self.gru_gates)
        self.hypernetworks = nn.ModuleDict(self.hypernetworks)
    
    def forward(self, x, adapt=False, current_energy=0.0):
        self.layer_stats = {} # Reset stats
        
        if not adapt:
            return self.model(x)

        # 1. Forward pass to collect stats
        _ = self.model(x)
        
        # 2. EAGER-TTA decision logic
        adapt_decision = False
        if self.layer_stats: # If hooks were triggered
            total_utility = 0.0
            for name, norm_layer in self.norm_layers:
                stats = self.layer_stats[name]
                kf = self.kalman_filters[name]
                
                kf.predict()
                llr = kf.get_log_likelihood_ratio(stats)
                kf.update(stats)

                self.llr_history[name].append(llr.item())
                if len(self.llr_history[name]) == self.config['eager_params']['gru_window']:
                    history_tensor = torch.tensor(self.llr_history[name], device=self.device).view(1, -1, 1)
                    utility = self.gru_gates[name](history_tensor)
                    total_utility += utility.item()
            
            avg_utility = total_utility / len(self.norm_layers) if self.norm_layers else 0
            adapt_decision = self.scheduler.decide(avg_utility, current_energy)

        # 3. Apply updates if decision is to adapt
        if adapt_decision:
            for name, norm_layer in self.norm_layers:
                stats = self.layer_stats[name]
                u, v, mask = self.hypernetworks[name](stats.unsqueeze(0))
                delta_w = (u @ v.transpose(-1, -2)).squeeze(0)
                masked_delta_w = delta_w * mask.squeeze(0).unsqueeze(1)

                if hasattr(norm_layer, 'weight') and norm_layer.weight is not None:
                    # Assuming update is for affine weight parameter
                    norm_layer.weight.data += self.config['optimizer']['lr'] * masked_delta_w.mean(dim=0).view_as(norm_layer.weight.data)

        # 4. Final forward pass with potentially updated model
        return self.model(x), adapt_decision

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def run_meta_training(self, config):
        """Dummymeta-training function to save initial random weights."""
        if not os.path.exists('meta_weights'):
            os.makedirs('meta_weights')
        
        model_name = self.config.get('model_name', 'default')
        weights_path = f'meta_weights/{model_name}_eager_weights.pth'
        
        if os.path.exists(weights_path):
            print(f'Loading existing meta-weights from {weights_path}')
            self.load_state_dict(torch.load(weights_path), strict=False)
        else:
            print(f'No meta-weights found. Saving randomly initialized weights to {weights_path}')
            torch.save(self.state_dict(), weights_path)

# --- Baseline Implementations ---

class Tent(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['optimizer']['lr'])
        self.steps = config['tta_steps']

    def forward(self, x):
        for _ in range(self.steps):
            outputs = self.model(x)
            loss = -(outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.model(x)

class EATA(nn.Module):
    """Simplified EATA implementation"""
    def __init__(self, model, config, fisher_dataset):
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['optimizer']['lr'])
        self.fisher_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.ewc_lambda = 0.1 # Simplified EWC lambda
        # Compute Fisher Information Matrix (diagonal) on a small clean dataset
        # This is a placeholder; a real implementation would use a proper dataset.
        self._compute_fisher(fisher_dataset)
    
    def _compute_fisher(self, dataset):
        self.model.eval()
        for x, y in dataset:
            x = x.cuda()
            outputs = self.model(x)
            log_probs = F.log_softmax(outputs, dim=1)
            label = torch.argmax(log_probs, dim=1)
            loss = F.nll_loss(log_probs, label)
            loss.backward()
            break # Use a single batch for approximation
        self.fisher_info = {n: p.grad.clone().detach()**2 for n, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad()

    def forward(self, x):
        outputs = self.model(x)
        entropy = -(outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1)
        # Sample selection based on entropy
        if entropy.mean() < 0.5 * math.log(outputs.size(1)):
            loss = entropy.mean()
            # EWC regularization
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fisher_info:
                    ewc_loss += (self.fisher_info[name] * (param - self.fisher_params[name])**2).sum()
            loss += self.ewc_lambda * ewc_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self.model(x)

# --- Main Training Loop ---

def run_adaptation_stream(model, data_loader, config, output_csv_path, energy_monitor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Setup writer
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    writer = open(output_csv_path, 'w')
    writer.write('step,accuracy,latency_ms,energy_joules,adapted\n')

    correct, total = 0, 0
    total_energy = 0
    is_eager = isinstance(model, EagerTTAWrapper)
    if is_eager:
        # Warmup for scheduler
        print("Running scheduler warmup...")
        base_energies = []
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            energy_monitor.start()
            with torch.no_grad():
                _ = model.model(x)
            energy = energy_monitor.stop()
            base_energies.append(energy)
            if i >= 10: # 10 batches for warmup
                break
        model.scheduler.set_base_energy(np.mean(base_energies) if base_energies else 0.01)
        print(f"Base energy set to: {model.scheduler.base_energy:.4f} J")

    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        torch.cuda.synchronize()
        tic = time.perf_counter()
        energy_monitor.start()
        
        adapted = 0
        if config['baseline'] == 'No-TTA':
            with torch.no_grad():
                outputs = model(x)
        elif is_eager:
            energy_so_far = energy_monitor.peek()
            with torch.no_grad(): # EAGER updates internally
                 outputs, adapted = model(x, adapt=True, current_energy=energy_so_far)
            adapted = 1 if adapted else 0
        elif config['baseline'] in ['Tent-5', 'EATA']:
            # These models handle their own updates
            outputs = model(x)
            adapted = 1
        else: # Other baselines - simplified loop
            # This is a simplified adaptation for other baselines like RoTTA, GLaD, etc.
            # A full implementation would be more complex.
            optimizer = optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
            outputs = model(x)
            loss = -(outputs.softmax(1) * F.log_softmax(outputs, 1)).sum(1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adapted = 1

        energy_joules = energy_monitor.stop()
        torch.cuda.synchronize()
        toc = time.perf_counter()

        latency_ms = (toc - tic) * 1000
        total_energy += energy_joules

        _, predicted = torch.max(outputs.data, 1)
        batch_total = y.size(0)
        batch_correct = (predicted == y).sum().item()
        
        correct += batch_correct
        total += batch_total
        accuracy = (batch_correct / batch_total) * 100

        writer.write(f'{i},{accuracy:.4f},{latency_ms:.4f},{energy_joules:.4f},{adapted}\n')
        if i % 100 == 0:
            print(f"Step {i}: Acc={accuracy:.2f}%, Latency={latency_ms:.2f}ms, Energy={energy_joules:.4f}J, Adapted={adapted}")
        
        # For smoke test, run for fewer steps
        if config.get('smoke_test', False) and i >= config.get('smoke_test_steps', 100):
            break

    writer.close()
    if is_eager:
        model.remove_hooks()
    print(f"Finished stream. Final avg accuracy: {100 * correct / total:.2f}%")
