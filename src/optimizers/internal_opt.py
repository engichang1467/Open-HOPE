import torch
import torch.nn as nn
import torch.nn.functional as F

class InternalOptimizer(nn.Module):
    """
    Implements the internal optimizer for the Self-Modifying Titans module.
    Based on Equation 28 from the paper:
    W_{t+1} = W_t(I - x_t x_t^T) - eta_{t+1} * grad_{y_t} L(W_t; x_t) outer x_t
    """
    def __init__(self, hidden_dim, learning_rate=1e-2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

    def forward(self, W, x, grad_L):
        """
        Args:
            W: Current weight matrix [batch_size, hidden_dim, hidden_dim]
            x: Input vector [batch_size, hidden_dim, 1]
            grad_L: Gradient of loss w.r.t output [batch_size, hidden_dim, 1]
        Returns:
            W_next: Updated weight matrix
        """
        # x_t x_t^T
        # x is [B, D, 1]
        # x_t_T is [B, 1, D]
        # outer_x is [B, D, D]
        outer_x = torch.bmm(x, x.transpose(1, 2))
        
        # Identity matrix
        I = torch.eye(self.hidden_dim, device=x.device).unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Term 1: W_t(I - x_t x_t^T)
        # We assume x is normalized or small enough, or this is a decay term.
        # In the paper Eq 28, it looks like a forgetting mechanism.
        term1 = torch.bmm(W, (I - outer_x))
        
        # Term 2: eta * grad_L outer x_t
        # grad_L is [B, D, 1]
        # x is [B, D, 1] -> x^T is [B, 1, D]
        # We need grad_L outer x_t.
        # Wait, Eq 28 says: - eta * grad_{y_t} L \otimes x_t
        # This usually means grad * x^T.
        term2 = self.learning_rate * torch.bmm(grad_L, x.transpose(1, 2))
        
        W_next = term1 - term2
        return W_next
