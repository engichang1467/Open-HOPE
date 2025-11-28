import torch
import torch.nn as nn
import torch.nn.functional as F
from src.optimizers.internal_opt import InternalOptimizer

class SelfModifyingTitans(nn.Module):
    """
    Self-Modifying Titans module.
    Uses an internal optimizer to update its own weights during the forward pass (or conceptually).
    Actually, Titans usually refers to a memory module that updates its state.
    Here, we implement the "Neural Learning Module" that learns to modify itself.
    """
    def __init__(self, dim, head_dim):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        
        # The weight matrix W_t that is updated.
        # In a real implementation, this might be the "memory" state.
        # We'll initialize it as a parameter but it will be updated dynamically.
        # Actually, if it's updated per step, it's a state, not a static parameter.
        # So we won't register it as a parameter to be optimized by the global optimizer directly,
        # or maybe we do, but it changes during inference?
        # The prompt says: "The model uses a variant of gradient descent... as its internal forward pass mechanism."
        
        self.internal_opt = InternalOptimizer(head_dim)
        
        # Projections for the update rule
        self.q_proj = nn.Linear(dim, head_dim)
        self.k_proj = nn.Linear(dim, head_dim)
        self.v_proj = nn.Linear(dim, head_dim)
        self.o_proj = nn.Linear(head_dim, dim)

    def forward(self, x, state=None):
        """
        Args:
            x: Input sequence [Batch, SeqLen, Dim]
            state: Previous weight matrix W_{t-1} [Batch, HeadDim, HeadDim]
        """
        b, s, d = x.shape
        
        if state is None:
            # Initialize W_0 as zero or learned init
            state = torch.zeros(b, self.head_dim, self.head_dim, device=x.device)
            
        outputs = []
        current_W = state
        
        # We process the sequence step by step to apply the update rule
        # This is slow in Python loop, but necessary for the recurrence unless we use a parallel scan (like Linear Attention / Mamba).
        # The update rule is non-linear (involves W_t in the update of W_t?), wait.
        # Eq 28: W_{t+1} = W_t(...) - ...
        # It depends on W_t.
        
        for t in range(s):
            xt = x[:, t, :].unsqueeze(1) # [B, 1, D]
            
            # Project to subspace
            q = self.q_proj(xt).transpose(1, 2) # [B, H, 1]
            k = self.k_proj(xt).transpose(1, 2) # [B, H, 1]
            v = self.v_proj(xt).transpose(1, 2) # [B, H, 1]
            
            # Prediction y_t = W_t * x_t (conceptually)
            # Here we use q, k, v.
            # Maybe W_t acts as the value memory?
            # In linear attention: y = W * q
            # Let's assume W_t stores the key-value associations.
            
            # Let's follow the "Neural Learning Module" idea.
            # We want to predict something and minimize loss.
            # Usually in Titans/Linear Transformers:
            # W_t is the memory.
            # Retrieval: y_t = W_t * q_t
            
            y_t = torch.bmm(current_W, q) # [B, H, 1]
            
            # Update step
            # We need a "target" or "loss" to drive the update.
            # In self-supervised sequence modeling, what is the target?
            # Usually we reconstruct v_t.
            # Loss = || W_t * k_t - v_t ||^2
            # Gradient w.r.t W_t is (W_t * k_t - v_t) * k_t^T
            
            # Let's map Eq 28 terms to this.
            # grad_L = (Prediction - Target)
            # If we use k_t as input to W_t to predict v_t.
            # Prediction = W_t * k_t
            # Error = Prediction - v_t
            # grad_L w.r.t Prediction is Error.
            # grad_L w.r.t W_t is Error * k_t^T.
            
            # Eq 28: W_{t+1} = W_t(I - x x^T) - eta * grad \otimes x
            # Here x is k_t.
            # So W_{t+1} = W_t - W_t k k^T - eta * (W_t k - v) k^T
            # This looks like a standard linear attention update / delta rule.
            
            # We will use the InternalOptimizer module.
            # We need to define what 'x' and 'grad_L' are passed to it.
            # x -> k
            # grad_L -> (W_t * k - v)
            
            pred_val = torch.bmm(current_W, k)
            error = pred_val - v
            
            # Update W
            # Note: The paper Eq 28 has a decay term W_t(I - x x^T).
            # Our InternalOptimizer implements exactly Eq 28.
            # We pass k as x.
            
            current_W = self.internal_opt(current_W, k, error)
            
            # Output projection
            # We use the retrieved value y_t (based on q)
            out_t = self.o_proj(y_t.transpose(1, 2)) # [B, 1, D]
            outputs.append(out_t)
            
        outputs = torch.cat(outputs, dim=1)
        return outputs, current_W
