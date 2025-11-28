import torch
import torch.nn as nn

class CMS(nn.Module):
    """
    Continuum Memory System (CMS).
    A chain of MLP blocks where the l-th MLP is updated every C^{(l)} steps.
    y_t = MLP^{(f_k)}(...MLP^{(f_1)}(x_t))
    """
    def __init__(self, dim, levels_config):
        """
        Args:
            dim: Input/Output dimension
            levels_config: List of dicts defining levels (frequency, chunk_size, etc.)
        """
        super().__init__()
        self.dim = dim
        self.levels_config = levels_config
        self.layers = nn.ModuleList()
        
        for level in levels_config:
            # Simple MLP block: Linear -> GELU -> Linear
            # We can make it more complex if needed, but standard FFN is usually 2 layers.
            mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
            self.layers.append(mlp)
            
    def forward(self, x):
        """
        Pass through all MLPs in sequence.
        """
        # Residual connection is usually applied around FFN in Transformers.
        # Here, the CMS *replaces* the FFN.
        # The paper says "chain of MLP blocks". 
        # We will assume sequential application: x -> MLP1 -> MLP2 -> ...
        # Or is it nested? "y_t = MLP^{(f_k)}(...MLP^{(f_1)}(x_t))" implies composition.
        
        out = x
        for layer in self.layers:
            # Usually we want residual connections between these layers too to avoid vanishing gradients?
            # The formula implies direct composition.
            # But standard Transformer FFN has a residual around it.
            # Let's assume standard composition for the internal structure.
            out = layer(out) + out # Adding residual for stability
            
        return out

    def get_parameters_by_frequency(self, current_step):
        """
        Returns parameters that should be updated at the current step.
        """
        params_to_update = []
        for i, level in enumerate(self.levels_config):
            freq = level['frequency']
            if current_step % freq == 0:
                params_to_update.extend(self.layers[i].parameters())
        return params_to_update
