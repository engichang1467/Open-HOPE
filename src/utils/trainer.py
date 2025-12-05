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
    Trainer that handles Multi-Time Scale Updates.
    Parameters are grouped by their update frequency.
    """
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Group parameters by frequency
        self.optimizers = {}
        self.param_groups = self._group_parameters()
        
        # Initialize optimizers for each group
        # lr = float(config['training']['learning_rate']) # Not used by DMGD directly in this formulation, but could be eta
        
        # DMGD Hyperparameters
        # We can pull these from config if available, otherwise defaults
        alpha = 0.9
        eta = float(config['training']['learning_rate']) # Use LR as eta for the energy gradient step
        
        for chunk_size, params in self.param_groups.items():
            if params:
                # Replace AdamW with DeepMomentumWrapper
                self.optimizers[chunk_size] = DeepMomentumWrapper(
                    params, 
                    alpha=alpha, 
                    eta=eta, 
                    device=device
                )
                
        # Initialize global step counter
        self.global_step = 0
                
    def _group_parameters(self):
        """
        Groups parameters based on the CMS levels and other components.
        """
        groups = {}
        
        # 1. CMS Parameters (Explicit frequencies)
        cms_levels = self.config['cms']['levels']
        
        # We need to traverse the model and find which parameters belong to which CMS level.
        # This is tricky because parameters are just tensors.
        # We can rely on the model structure.
        
        # Let's iterate over modules.
        # We'll maintain a set of parameter IDs to avoid duplicates (though we shouldn't have any).
        assigned_params = set()
        
        for level_config in cms_levels:
            chunk_size = level_config['chunk_size']
            if chunk_size not in groups:
                groups[chunk_size] = []
                
        # Default chunk_size for non-CMS parameters (e.g. Titans, Embeddings, Norms)
        # Usually these are high frequency (chunk_size=1).
        if 1 not in groups:
            groups[1] = []
            
        # Iterate through model layers
        for layer in self.model.layers:
            # CMS
            cms_module = layer['cms']
            for i, level_config in enumerate(cms_levels):
                chunk_size = level_config['chunk_size']
                # The CMS module has a list of layers corresponding to levels
                # We assume cms.layers[i] corresponds to levels[i]
                if i < len(cms_module.layers):
                    mlp = cms_module.layers[i]
                    for p in mlp.parameters():
                        groups[chunk_size].append(p)
                        assigned_params.add(id(p))
                        
            # Titans & Norms (Freq 1 / Chunk Size 1)
            # Add everything else in the layer to Chunk Size 1
            for name, submodule in layer.named_children():
                if name != 'cms':
                    for p in submodule.parameters():
                        if id(p) not in assigned_params:
                            groups[1].append(p)
                            assigned_params.add(id(p))
                            
        # Global parameters (Embeddings, Final Norm, Head)
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
                    # Update parameters based on chunk size
                    self.global_step += 1
                    
                    for chunk_size, optimizer in self.optimizers.items():
                        # Case 1: Update Occurs (Synchronization Point)
                        # If chunk_size > 0 and global_step is a multiple of chunk_size
                        if chunk_size > 0 and self.global_step % chunk_size == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                        # Case 2: No Update Occurs (Between Synchronization Points)
                        # Gradients accumulate.
                        else:
                            pass
                        
                progress_bar.set_postfix({'loss': loss.item() * accum_steps})
