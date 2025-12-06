import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicProjections(nn.Module):
    """
    Dynamic Q, K, V projections for Self-Modifying Titans.
    
    Implements Equation 28: W_{t+1} = W_t(I - x_t x_t^T) - eta * grad_L ⊗ x_t
    
    The projection weights are updated at every time step based on the input,
    making them "fast weights" that adapt dynamically to the context.
    
    Args:
        dim: Input dimension
        head_dim: Output dimension (per head)
        learning_rate: Learning rate eta for weight updates (default: 0.01)
        use_dynamic: If True, use dynamic weight updates; if False, use static projections
        share_weights: If True, use shared weight matrix for Q, K, V
    """
    def __init__(self, dim, head_dim, learning_rate=0.01, use_dynamic=True, share_weights=False):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.learning_rate = learning_rate
        self.use_dynamic = use_dynamic
        self.share_weights = share_weights
        
        if not use_dynamic:
            # Static projections (baseline)
            self.q_proj = nn.Linear(dim, head_dim, bias=False)
            self.k_proj = nn.Linear(dim, head_dim, bias=False)
            self.v_proj = nn.Linear(dim, head_dim, bias=False)
        else:
            # Initialize weight parameters that serve as W_0 (initial state)
            # These are the "slow weights" that can be updated by backprop
            if share_weights:
                self.init_weight = nn.Parameter(torch.randn(head_dim, dim) * 0.02)
            else:
                self.init_weight_q = nn.Parameter(torch.randn(head_dim, dim) * 0.02)
                self.init_weight_k = nn.Parameter(torch.randn(head_dim, dim) * 0.02)
                self.init_weight_v = nn.Parameter(torch.randn(head_dim, dim) * 0.02)
    
    def reset_weights(self, batch_size, device):
        """
        Reset dynamic weight states to initial values.
        Called at the beginning of each sequence or batch.
        
        Returns:
            Tuple of (W_q, W_k, W_v) weight matrices [batch, head_dim, dim]
        """
        if self.share_weights:
            W = self.init_weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
            return W, W, W
        else:
            W_q = self.init_weight_q.unsqueeze(0).expand(batch_size, -1, -1).clone()
            W_k = self.init_weight_k.unsqueeze(0).expand(batch_size, -1, -1).clone()
            W_v = self.init_weight_v.unsqueeze(0).expand(batch_size, -1, -1).clone()
            return W_q, W_k, W_v
    
    def update_weight(self, W, x, lss, eta):
        """
        Apply Equation 28 to update weight matrix W.
        
        W_{t+1} = W_t(I - x_t x_t^T) - eta * lss ⊗ x_t
        
        Args:
            W: Current weight matrix [batch, head_dim, dim]
            x: Input vector [batch, dim]
            lss: Local Surprise Signal (gradient) [batch, head_dim]
            eta: Learning rate
            
        Returns:
            W_next: Updated weight matrix [batch, head_dim, dim]
        """
        batch_size = x.size(0)
        
        # Reshape x to [batch, dim, 1] for outer product
        x_col = x.unsqueeze(-1)  # [batch, dim, 1]
        
        # Term 1: W_t(I - x_t x_t^T)
        # x_t x_t^T: [batch, dim, dim]
        outer_x = torch.bmm(x_col, x_col.transpose(1, 2))  # [batch, dim, dim]
        
        # I - x_t x_t^T
        I = torch.eye(self.dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        decay_term = I - outer_x  # [batch, dim, dim]
        
        # W_t @ (I - x_t x_t^T): [batch, head_dim, dim] @ [batch, dim, dim]
        term1 = torch.bmm(W, decay_term)
        
        # Term 2: eta * lss ⊗ x_t
        # lss: [batch, head_dim] -> [batch, head_dim, 1]
        # x: [batch, dim] -> [batch, 1, dim]
        # outer product: [batch, head_dim, dim]
        lss_col = lss.unsqueeze(-1)  # [batch, head_dim, 1]
        x_row = x.unsqueeze(1)  # [batch, 1, dim]
        term2 = eta * torch.bmm(lss_col, x_row)  # [batch, head_dim, dim]
        
        # Final update: W_{t+1} = term1 - term2
        W_next = term1 - term2
        
        return W_next
    
    def compute_lss(self, W, x, target):
        """
        Compute Local Surprise Signal (LSS): gradient of loss w.r.t. output.
        
        For regression: LSS = prediction - target = W @ x - target
        
        Args:
            W: Weight matrix [batch, head_dim, dim]
            x: Input [batch, dim]
            target: Target output [batch, head_dim]
            
        Returns:
            lss: Local Surprise Signal [batch, head_dim]
        """
        # Prediction: y = W @ x
        # x: [batch, dim] -> [batch, dim, 1]
        x_col = x.unsqueeze(-1)
        pred = torch.bmm(W, x_col).squeeze(-1)  # [batch, head_dim]
        
        # LSS = prediction - target (for L2 loss)
        lss = pred - target
        
        return lss
    
    def forward(self, x, targets=None, states=None):
        """
        Forward pass with dynamic weight updates.
        
        Args:
            x: Input sequence [batch, seq_len, dim]
            targets: Optional targets for LSS computation [batch, seq_len, head_dim]
                     If None, uses auto-associative learning (target = projection)
            states: Optional initial weight states (W_q, W_k, W_v)
                    If None, initializes from learned parameters
                    
        Returns:
            q, k, v: Projected outputs [batch, seq_len, head_dim]
            new_states: Updated weight states (W_q, W_k, W_v) for next call
        """
        if not self.use_dynamic:
            # Static mode: just apply linear projections
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return q, k, v
        
        # Dynamic mode: per-token weight updates
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Initialize or use provided weight states
        if states is None:
            W_q, W_k, W_v = self.reset_weights(batch_size, device)
        else:
            W_q, W_k, W_v = states
        
        # Storage for outputs
        q_list = []
        k_list = []
        v_list = []
        
        # Process sequence token by token
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, dim]
            
            # Compute projections using current weights W_t
            x_t_col = x_t.unsqueeze(-1)  # [batch, dim, 1]
            q_t = torch.bmm(W_q, x_t_col).squeeze(-1)  # [batch, head_dim]
            k_t = torch.bmm(W_k, x_t_col).squeeze(-1)  # [batch, head_dim]
            v_t = torch.bmm(W_v, x_t_col).squeeze(-1)  # [batch, head_dim]
            
            q_list.append(q_t)
            k_list.append(k_t)
            v_list.append(v_t)
            
            # Compute LSS for weight updates
            if targets is not None:
                # Supervised: use provided targets
                target_q = targets[:, t, :]
                target_k = targets[:, t, :]
                target_v = targets[:, t, :]
            else:
                # Auto-associative: predict the projection itself
                # This creates a self-organizing memory
                target_q = q_t.detach()
                target_k = k_t.detach()
                target_v = v_t.detach()
            
            # Compute Local Surprise Signals
            lss_q = self.compute_lss(W_q, x_t, target_q)
            lss_k = self.compute_lss(W_k, x_t, target_k)
            lss_v = self.compute_lss(W_v, x_t, target_v)
            
            # Update weights: W_{t+1} = f(W_t, x_t, lss_t, eta)
            W_q = self.update_weight(W_q, x_t, lss_q, self.learning_rate)
            W_k = self.update_weight(W_k, x_t, lss_k, self.learning_rate)
            W_v = self.update_weight(W_v, x_t, lss_v, self.learning_rate)
        
        # Stack outputs
        q = torch.stack(q_list, dim=1)  # [batch, seq_len, head_dim]
        k = torch.stack(k_list, dim=1)
        v = torch.stack(v_list, dim=1)
        
        # Return projections and updated states
        new_states = (W_q, W_k, W_v)
        return q, k, v, new_states
