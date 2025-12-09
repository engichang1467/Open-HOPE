import torch
import torch.nn as nn
from tqdm import tqdm
from src.optimizers.deep_opt import DeepMomentumOptimizer

class DeepMomentumWrapper(torch.optim.Optimizer):
    """
    Wrapper to make DeepMomentumOptimizer compatible with torch.optim.Optimizer.
    Applies DeepMomentumOptimizer coordinate-wise (param_dim=1).
    """
    def __init__(self, params, hidden_dim=64, alpha=0.9, eta=1e-3, device='cpu'):
        defaults = dict(hidden_dim=hidden_dim, alpha=alpha, eta=eta, device=device)
        super().__init__(params, defaults)
        
        # Initialize the Deep Optimizer module
        # We use a single shared optimizer module for all parameters (coordinate-wise)
        self.deep_opt = DeepMomentumOptimizer(param_dim=1, hidden_dim=hidden_dim, alpha=alpha, eta=eta)
        self.deep_opt.to(device)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Get momentum state
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)
                    
                m_prev = state['momentum']
                
                # Prepare inputs for Deep Optimizer
                # Flatten to [N, 1] for coordinate-wise processing
                grad_flat = grad.view(-1, 1)
                m_prev_flat = m_prev.view(-1, 1)
                
                # Deep Optimizer Step
                # We use no_grad for the outer update application, 
                # but DeepMomentumOptimizer handles its own autograd internally if needed.
                # However, since we are updating model weights, we usually don't need to track 
                # gradients of the update step itself unless we are meta-learning.
                # The DeepMomentumOptimizer.forward uses enable_grad() internally for its energy calculation.
                update_flat, m_next_flat = self.deep_opt(grad_flat, m_prev_flat)
                
                # Update State
                state['momentum'] = m_next_flat.view_as(p.data)
                
                # Update Parameters
                # W_{i+1} = W_i + m_{i+1}
                p.data.add_(state['momentum'])
                
        return loss

class MultiFrequencyTrainer:
    """
    Trainer that handles Multi-Time Scale Updates for Nested Learning Paradigm.
    
    Parameters are grouped by their update frequency (not chunk_size).
    Each group updates only when global_step % frequency == 0.
    
    For example:
    - frequency=1: Updates every step (high frequency)
    - frequency=16: Updates every 16 steps (mid frequency) 
    - frequency=1000: Updates every 1000 steps (low frequency)
    
    This implements the strict multi-time scale update logic where:
    θ^(f_ℓ) is updated only when i ≡ 0 (mod f_ℓ)
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Group parameters by UPDATE FREQUENCY (not chunk_size)
        # This is critical for Multi-Time Scale Updates
        self.optimizers = {}
        self.param_groups = self._group_parameters()
        
        # Initialize optimizers for each frequency group
        # DMGD Hyperparameters
        alpha = 0.9
        eta = float(config['training']['learning_rate'])
        
        for frequency, params in self.param_groups.items():
            if params:
                # Create optimizer for this frequency group
                self.optimizers[frequency] = DeepMomentumWrapper(
                    params, 
                    alpha=alpha, 
                    eta=eta, 
                    device=device
                )
                
        # Initialize global step counter
        self.global_step = 0
                
    def _group_parameters(self):
        """
        Groups parameters based on UPDATE FREQUENCY (not chunk_size).
        
        Returns:
            dict: {frequency: [parameters]} mapping
                  e.g., {1: [...], 16: [...], 1000: [...]}
        """
        groups = {}
        
        # 1. CMS Parameters - group by FREQUENCY
        cms_levels = self.config['cms']['levels']
        
        # Track assigned parameters to avoid duplicates
        assigned_params = set()
        
        # Initialize groups for all defined frequencies
        for level_config in cms_levels:
            frequency = level_config['frequency']  # USE FREQUENCY, not chunk_size
            if frequency not in groups:
                groups[frequency] = []
                
        # Default frequency for non-CMS parameters (e.g. Titans, Embeddings, Norms)
        # These are high frequency components that update every step
        if 1 not in groups:
            groups[1] = []
            
        # Iterate through model layers
        for layer in self.model.layers:
            # CMS - assign each level's parameters to its frequency group
            cms_module = layer['cms']
            for i, level_config in enumerate(cms_levels):
                frequency = level_config['frequency']  # USE FREQUENCY
                # The CMS module has a list of layers corresponding to levels
                # cms.layers[i] corresponds to levels[i]
                if i < len(cms_module.layers):
                    mlp = cms_module.layers[i]
                    for p in mlp.parameters():
                        groups[frequency].append(p)
                        assigned_params.add(id(p))
                        
            # Titans & Norms - always frequency 1 (update every step)
            for name, submodule in layer.named_children():
                if name != 'cms':
                    for p in submodule.parameters():
                        if id(p) not in assigned_params:
                            groups[1].append(p)
                            assigned_params.add(id(p))
                            
        # Global parameters (Embeddings, Final Norm, Head) - frequency 1
        for p in self.model.embedding.parameters():
            if id(p) not in assigned_params:
                groups[1].append(p)
                assigned_params.add(id(p))
                
        for p in self.model.final_norm.parameters():
            if id(p) not in assigned_params:
                groups[1].append(p)
                assigned_params.add(id(p))
                
        for p in self.model.lm_head.parameters():
            if id(p) not in assigned_params:
                groups[1].append(p)
                assigned_params.add(id(p))
                
        return groups

    def train(self, dataloader, num_epochs):
        self.model.train()

        # 1. Get accumulation steps from config
        accum_steps = self.config['training']['gradient_accumulation_steps']
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            # 2. Use enumerate to track batch index
            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(self.device)
                
                # Forward pass
                # Labels are same as input for causal LM (shifted internally or by loss)
                # Standard HF GPT2 model calculates loss if labels provided.
                # Our HOPE model returns logits.
                logits = self.model(batch)
                
                # Shift for loss
                # logits: [B, S, V], batch: [B, S]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # 3. Normalize loss
                loss = loss / accum_steps
                
                # Backward
                loss.backward()
                
                # 4. Step only after accumulation
                if (batch_idx + 1) % accum_steps == 0:
                    # Increment global step for multi-time scale tracking
                    self.global_step += 1
                    
                    # STRICT MULTI-TIME SCALE UPDATE LOGIC
                    # Update parameters based on their UPDATE FREQUENCY
                    # θ^(f_ℓ) updates only when global_step ≡ 0 (mod f_ℓ)
                    for frequency, optimizer in self.optimizers.items():
                        # Synchronization Point Check: Does this frequency update now?
                        if frequency > 0 and self.global_step % frequency == 0:
                            # YES: Apply accumulated gradients and update parameters
                            optimizer.step()
                            optimizer.zero_grad()
                        # else: NO update - gradients continue to accumulate
                        # This is intentional for multi-time scale learning
                        
                progress_bar.set_postfix({'loss': loss.item() * accum_steps})
