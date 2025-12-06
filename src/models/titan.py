import torch
import torch.nn as nn
import torch.nn.functional as F
from src.optimizers.internal_opt import InternalOptimizer
from src.layers.projections import DynamicProjections

class SelfModifyingTitans(nn.Module):
    """
    Self-Modifying Titans module.
    Uses an internal optimizer to update its own weights during the forward pass.
    
    This module combines:
    1. Dynamic Q, K, V projections (updated via Equation 28)
    2. Memory state W that stores key-value associations
    3. Internal optimizer for memory updates
    """
    def __init__(self, dim, head_dim, use_dynamic=True, proj_lr=0.01):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.use_dynamic = use_dynamic
        
        # Dynamic projections with Equation 28 updates
        self.projections = DynamicProjections(
            dim=dim, 
            head_dim=head_dim,
            learning_rate=proj_lr,
            use_dynamic=use_dynamic,
            share_weights=False
        )
        
        # Internal optimizer for memory state updates
        self.internal_opt = InternalOptimizer(head_dim)
        
        # Output projection
        self.o_proj = nn.Linear(head_dim, dim, bias=False)

    def forward(self, x, state=None, proj_states=None):
        """
        Forward pass with dynamic projections and memory updates.
        
        Args:
            x: Input sequence [Batch, SeqLen, Dim]
            state: Previous memory matrix W_{t-1} [Batch, HeadDim, HeadDim]
            proj_states: Previous projection weight states (W_q, W_k, W_v)
        
        Returns:
            outputs: Output sequence [Batch, SeqLen, Dim]
            new_state: Updated memory matrix [Batch, HeadDim, HeadDim]
            new_proj_states: Updated projection states (W_q, W_k, W_v)
        """
        b, s, d = x.shape
        
        # Initialize memory state if not provided
        if state is None:
            state = torch.zeros(b, self.head_dim, self.head_dim, device=x.device)
        
        if not self.use_dynamic:
            # Static mode: traditional linear attention
            q, k, v = self.projections(x)
            
            # Simple linear attention update
            outputs = []
            current_W = state
            
            for t in range(s):
                q_t = q[:, t, :].unsqueeze(-1)  # [B, H, 1]
                k_t = k[:, t, :].unsqueeze(-1)  # [B, H, 1]
                v_t = v[:, t, :].unsqueeze(-1)  # [B, H, 1]
                
                # Retrieve from memory
                y_t = torch.bmm(current_W, q_t)  # [B, H, 1]
                
                # Update memory with internal optimizer
                pred_val = torch.bmm(current_W, k_t)
                error = pred_val - v_t
                current_W = self.internal_opt(current_W, k_t, error)
                
                # Output projection
                out_t = self.o_proj(y_t.transpose(1, 2))  # [B, 1, D]
                outputs.append(out_t)
            
            outputs = torch.cat(outputs, dim=1)
            return outputs, current_W
        
        else:
            # Dynamic mode: projections update themselves via Equation 28
            q, k, v, new_proj_states = self.projections(x, states=proj_states)
            
            # Process memory updates
            outputs = []
            current_W = state
            
            for t in range(s):
                # Use dynamically computed q, k, v
                q_t = q[:, t, :].unsqueeze(-1)  # [B, H, 1]
                k_t = k[:, t, :].unsqueeze(-1)  # [B, H, 1]
                v_t = v[:, t, :].unsqueeze(-1)  # [B, H, 1]
                
                # Retrieve from memory using query
                y_t = torch.bmm(current_W, q_t)  # [B, H, 1]
                
                # Update memory state using internal optimizer
                # Predict value from key
                pred_val = torch.bmm(current_W, k_t)
                error = pred_val - v_t
                
                # Apply Equation 28 to memory state
                current_W = self.internal_opt(current_W, k_t, error)
                
                # Output projection
                out_t = self.o_proj(y_t.transpose(1, 2))  # [B, 1, D]
                outputs.append(out_t)
            
            outputs = torch.cat(outputs, dim=1)
            return outputs, current_W, new_proj_states
