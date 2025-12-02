import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepMomentumOptimizer(nn.Module):
    """
    Implements Deep Momentum Gradient Descent (DMGD).
    Replaces the scalar momentum term with a neural network (MLP) that compresses gradients.
    Equation 23 in the paper:
    m_{i+1} = alpha * m_i - eta * grad_m(L2(m_i, u_i))
    W_{i+1} = W_i + m_{i+1}
    """
    def __init__(self, param_dim, hidden_dim=64, alpha=0.9, eta=1e-3):
        super().__init__()
        self.alpha = alpha
        self.eta = eta
        
        # The "Energy" function L2(m, u)
        # Takes concatenation of momentum (m) and gradient (u)
        # Outputs a scalar "energy"
        self.l2_energy_net = nn.Sequential(
            nn.Linear(param_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, grad, state):
        """
        Args:
            grad: Current gradient (u_i) [batch_size, param_dim] or [param_dim]
            state: Previous momentum (m_i) [batch_size, param_dim] or [param_dim]
                   If None, initializes to zeros.
        Returns:
            update: The update to apply to parameters (m_{i+1})
            new_state: The new state (m_{i+1})
        """
        if state is None:
            state = torch.zeros_like(grad)
            
        # We use torch.enable_grad() to ensure that we can compute the gradient of the energy function
        # even if the model is running in inference mode (torch.no_grad()).
        with torch.enable_grad():
            # Ensure state requires grad for autograd of the Energy function
            # We detach state because we treat the previous state as a constant starting point for this step's optimization
            # relative to the internal energy minimization.
            m_i = state.detach().requires_grad_(True)
            
            # We do NOT detach grad to allow gradients to flow back through it (e.g. for meta-learning)
            u_i = grad 
            
            # Compute Energy L2(m_i, u_i)
            # Concatenate along the last dimension
            inputs = torch.cat([m_i, u_i], dim=-1)
            energy = self.l2_energy_net(inputs)
            
            # Compute gradient of Energy w.r.t m_i
            # grad_energy = dL2/dm_i
            grad_energy = torch.autograd.grad(
                outputs=energy,
                inputs=m_i,
                grad_outputs=torch.ones_like(energy),
                create_graph=True, # Allow higher-order derivatives
                retain_graph=True
            )[0]
        
        # Update rule: m_{i+1} = alpha * m_i - eta * grad_energy
        m_next = self.alpha * m_i - self.eta * grad_energy
        
        # The update to W is m_next
        update = m_next
        
        return update, m_next.detach()
