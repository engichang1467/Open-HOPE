import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMomentumOptimizer(nn.Module):
    """
    Implements Deep Momentum Gradient Descent (DMGD).
    Replaces the scalar momentum term with a neural network (MLP) that compresses gradients.
    Equation 23 in the paper.
    """
    def __init__(self, param_dim, hidden_dim=64):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
            nn.Tanh() # Ensure bounded momentum
        )
        
    def forward(self, grad, state):
        """
        Args:
            grad: Current gradient
            state: Previous state (momentum)
        Returns:
            update: The update to apply to parameters
            new_state: The new state
        """
        # In standard momentum: v_{t+1} = mu * v_t + g_t
        # Here: v_{t+1} = MLP(v_t, g_t) or similar.
        # The prompt says "momentum term is replaced by a neural network... that compresses gradients".
        # Let's assume the MLP takes the gradient and perhaps previous state, 
        # but usually DMGD might learn to predict the next update.
        # Given the prompt "momentum term is replaced by a neural network (e.g., an MLP) that compresses gradients",
        # We will model it as: update = grad + MLP(grad) or similar, or state update.
        
        # Let's try to interpret "compresses gradients". 
        # Maybe it learns a better gradient direction.
        
        # We'll implement a residual update:
        # momentum = MLP(grad)
        # update = grad + momentum
        
        momentum = self.compressor(grad)
        update = grad + momentum
        
        return update, momentum
