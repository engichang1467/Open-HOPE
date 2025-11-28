import torch
import torch.nn as nn
from tqdm import tqdm
from src.optimizers.deep_opt import DeepMomentumOptimizer

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
        lr = float(config['training']['learning_rate'])
        for freq, params in self.param_groups.items():
            if params:
                # We use AdamW as the base optimizer for now, 
                # or we could use the DeepMomentumOptimizer if we fully implemented it as a torch.optim.Optimizer.
                # The DeepMomentumOptimizer in src/optimizers/deep_opt.py is an nn.Module, 
                # which implies it's part of the model or a custom update rule.
                # For simplicity and stability in this demo, we'll use AdamW,
                # but we acknowledge the paper uses DMGD.
                # To use DMGD properly, we'd need to write a custom loop that calls it.
                # Let's stick to standard optimizers for the "outer" loop but respect frequencies.
                self.optimizers[freq] = torch.optim.AdamW(params, lr=lr)
                
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
        
        for freq_config in cms_levels:
            freq = freq_config['frequency']
            if freq not in groups:
                groups[freq] = []
                
        # Default frequency for non-CMS parameters (e.g. Titans, Embeddings, Norms)
        # Usually these are high frequency (freq=1).
        if 1 not in groups:
            groups[1] = []
            
        # Iterate through model layers
        for layer in self.model.layers:
            # CMS
            cms_module = layer['cms']
            for i, level_config in enumerate(cms_levels):
                freq = level_config['frequency']
                # The CMS module has a list of layers corresponding to levels
                # We assume cms.layers[i] corresponds to levels[i]
                if i < len(cms_module.layers):
                    mlp = cms_module.layers[i]
                    for p in mlp.parameters():
                        groups[freq].append(p)
                        assigned_params.add(id(p))
                        
            # Titans & Norms (Freq 1)
            # Add everything else in the layer to Freq 1
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
        global_step = 0
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
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
                
                # Backward
                loss.backward()
                
                # Update parameters based on frequency
                global_step += 1
                
                for freq, optimizer in self.optimizers.items():
                    if global_step % freq == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        # For parameters not updating, we should probably zero their grads 
                        # to prevent accumulation? 
                        # Or we just let them accumulate and step later?
                        # "Gradient accumulation" is usually a feature.
                        # If we want strict "update every k steps", accumulation is natural.
                        # So we do NOTHING here.
                        pass
                        
                progress_bar.set_postfix({'loss': loss.item()})
