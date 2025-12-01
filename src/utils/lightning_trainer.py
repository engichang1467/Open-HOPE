import torch
import torch.nn as nn
import lightning as L
from src.models.hope import HOPE


class HOPELightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = HOPE(config)
        
        # Manual optimization is needed for multi-frequency updates
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Access optimizers
        optimizers = self.optimizers()
        
        # Determine accumulation steps
        accum_steps = self.config['training']['gradient_accumulation_steps']
        
        # Forward pass
        logits = self(batch)
        
        # Calculate loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Normalize loss for accumulation
        self.manual_backward(loss / accum_steps)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Step optimizers only after accumulating gradients
        if (batch_idx + 1) % accum_steps == 0:
            # Multi-frequency optimizer stepping
            for opt_idx, optimizer in enumerate(optimizers):
                freq = optimizer.defaults.get('frequency', 1)
                if (self.global_step // accum_steps) % freq == 0:
                    optimizer.step()

            # Zero gradients after stepping
            for optimizer in optimizers:
                optimizer.zero_grad()

    def configure_optimizers(self):
        param_groups = self._group_parameters()
        optimizers = []
        lr = float(self.config['training']['learning_rate'])
        
        # Create an optimizer for each frequency group
        sorted_freqs = sorted(param_groups.keys())
        for freq in sorted_freqs:
            params = param_groups[freq]
            if params:
                optimizer = torch.optim.AdamW(params, lr=lr)
                # Store frequency in the optimizer for access in training_step
                optimizer.defaults['frequency'] = freq
                optimizers.append(optimizer)
        
        return optimizers

    def _group_parameters(self):
        """
        Groups parameters based on the CMS levels and other components.
        This logic is copied from the original MultiFrequencyTrainer.
        """
        groups = {}
        assigned_params = set()
        
        # CMS Parameters with explicit frequencies
        if 'cms' in self.config and 'levels' in self.config['cms']:
            cms_levels = self.config['cms']['levels']
            for level_config in cms_levels:
                freq = level_config['frequency']
                if freq not in groups:
                    groups[freq] = []
        
        # Default frequency for non-CMS parameters (freq=1)
        if 1 not in groups:
            groups[1] = []
            
        # Iterate through model layers to assign parameters to frequency groups
        for layer in self.model.layers:
            # Assign CMS parameters
            if 'cms' in layer:
                cms_module = layer['cms']
                cms_levels = self.config['cms']['levels']
                for i, level_config in enumerate(cms_levels):
                    if i < len(cms_module.layers):
                        freq = level_config['frequency']
                        mlp = cms_module.layers[i]
                        for p in mlp.parameters():
                            if id(p) not in assigned_params:
                                groups[freq].append(p)
                                assigned_params.add(id(p))
            
            # Assign other layer parameters to frequency 1
            for name, submodule in layer.items():
                if name != 'cms':
                    for p in submodule.parameters():
                        if id(p) not in assigned_params:
                            groups[1].append(p)
                            assigned_params.add(id(p))
                            
        # Assign global parameters (Embeddings, Final Norm, Head) to frequency 1
        for p_list in [self.model.embedding.parameters(), self.model.final_norm.parameters(), self.model.lm_head.parameters()]:
            for p in p_list:
                if id(p) not in assigned_params:
                    groups[1].append(p)
                    assigned_params.add(id(p))
                    
        return groups

