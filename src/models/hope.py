import torch
import torch.nn as nn
from src.models.titan import SelfModifyingTitans
from src.memory.cms import CMS

class HOPE(nn.Module):
    """
    HOPE: High-order Optimization & Perception Engine.
    Combines Self-Modifying Titans with Continuum Memory System (CMS).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config['model']['dim']
        self.depth = config['model']['depth']
        self.vocab_size = config['model']['vocab_size']
        
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            # Each layer consists of Titans (Attention-like) and CMS (FFN-like)
            self.layers.append(nn.ModuleDict({
                'titans': SelfModifyingTitans(self.dim, self.dim // config['model']['num_heads']), # Simplified head dim
                'cms': CMS(self.dim, config['cms']['levels']),
                'norm1': nn.LayerNorm(self.dim),
                'norm2': nn.LayerNorm(self.dim)
            }))
            
        self.final_norm = nn.LayerNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Titans Block
            residual = x
            x = layer['norm1'](x)
            x_titans, _ = layer['titans'](x)
            x = residual + x_titans
            
            # CMS Block
            residual = x
            x = layer['norm2'](x)
            x_cms = layer['cms'](x)
            x = residual + x_cms
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
    
    def get_cms_parameters_by_frequency(self, current_step):
        """
        Collects parameters from all CMS modules that need update.
        """
        params = []
        for layer in self.layers:
            params.extend(layer['cms'].get_parameters_by_frequency(current_step))
        return params
