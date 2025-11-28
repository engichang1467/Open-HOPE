import torch
import torch.nn as nn

class DynamicProjections(nn.Module):
    """
    Dynamic Q, K, V projections.
    Part of the optimization flow.
    """
    def __init__(self, dim, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, head_dim)
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, head_dim)
        
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return q, k, v
