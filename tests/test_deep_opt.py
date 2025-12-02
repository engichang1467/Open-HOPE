import sys
from pathlib import Path

import pytest
import torch


# # make "src" importable when running pytest from the repo root
# def _ensure_src_on_path():
#     this_file = Path(__file__).resolve()
#     for base in [this_file.parent, *this_file.parents]:
#         candidate = base / "src"
#         if (candidate / "optimizers" / "deep_opt.py").exists():
#             if str(candidate) not in sys.path:
#                 sys.path.insert(0, str(candidate))
#             return
#     # best-effort fallback: repo/tests -> repo/src
#     fallback = this_file.parent.parent / "src"
#     if fallback.exists() and str(fallback) not in sys.path:
#         sys.path.insert(0, str(fallback))


# _ensure_src_on_path()
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.optimizers.deep_opt import DeepMomentumOptimizer  # noqa: E402


@pytest.mark.parametrize(
    "param_dim,batch,hidden_dim",
    [
        (8, 1, 16),
        (32, 4, 64),
    ],
)
def test_forward_backward_shapes_and_grads(param_dim, batch, hidden_dim):
    torch.manual_seed(0)
    model = DeepMomentumOptimizer(param_dim=param_dim, hidden_dim=hidden_dim).train()

    grad_in = torch.randn(batch, param_dim, dtype=torch.float32, requires_grad=True)
    state_in = torch.zeros_like(grad_in)

    update, momentum = model(grad_in, state_in)

    # shapes preserved
    assert update.shape == grad_in.shape
    assert momentum.shape == grad_in.shape


    # outputs are finite
    assert torch.isfinite(update).all()
    assert torch.isfinite(momentum).all()

    # backprop through module + into the input gradient tensor
    loss = (update ** 2).mean()
    loss.backward()

    # module parameters received gradients
    assert any(p.grad is not None for p in model.parameters()), "no parameter received gradients"

    # input also received gradients through autograd
    assert grad_in.grad is not None, "input did not receive gradients"


def test_eval_mode_runs():
    torch.manual_seed(123)
    param_dim, batch = 24, 3
    model = DeepMomentumOptimizer(param_dim=param_dim).eval()

    grad_in = torch.randn(batch, param_dim)
    state_in = torch.zeros_like(grad_in)

    with torch.no_grad():
        update, momentum = model(grad_in, state_in)

    assert update.shape == grad_in.shape
    assert torch.isfinite(update).all()
    assert torch.isfinite(momentum).all()


def test_deterministic_given_fixed_seed():
    # same seed + identical construction -> identical outputs
    param_dim, batch = 16, 2
    grad = torch.randn(batch, param_dim)  # fixed input; doesn't affect module init when we reseed later
    state = torch.zeros_like(grad)

    torch.manual_seed(2025)
    m1 = DeepMomentumOptimizer(param_dim=param_dim)
    u1, s1 = m1(grad, state)

    torch.manual_seed(2025)
    m2 = DeepMomentumOptimizer(param_dim=param_dim)
    u2, s2 = m2(grad, state)

    assert torch.allclose(u1, u2, atol=0, rtol=0)
    assert torch.allclose(s1, s2, atol=0, rtol=0)


def test_wrong_shape_raises():
    model = DeepMomentumOptimizer(param_dim=10)
    bad_grad = torch.randn(2, 11)  # last dim must match param_dim
    state = torch.zeros_like(bad_grad)

    with pytest.raises((RuntimeError, ValueError)):
        _ = model(bad_grad, state)


def test_update_differs_from_raw_grad_in_general():
    # very unlikely for update == grad due to random MLP + tanh
    torch.manual_seed(7)
    param_dim, batch = 12, 2
    model = DeepMomentumOptimizer(param_dim=param_dim)

    grad = torch.randn(batch, param_dim)
    state = torch.zeros_like(grad)
    update, _ = model(grad, state)

    assert not torch.allclose(update, grad, atol=1e-6), "update should generally differ from raw grad"

def test_deep_momentum_optimizer_init():
    param_dim = 10
    optimizer = DeepMomentumOptimizer(param_dim=param_dim)
    assert optimizer.l2_energy_net is not None
    assert optimizer.alpha == 0.9
    assert optimizer.eta == 1e-3

def test_deep_momentum_optimizer_forward():
    param_dim = 10
    batch_size = 5
    optimizer = DeepMomentumOptimizer(param_dim=param_dim)
    
    # Create dummy inputs
    grad = torch.randn(batch_size, param_dim)
    state = torch.zeros(batch_size, param_dim)
    
    # Forward pass
    update, new_state = optimizer(grad, state)
    
    # Check shapes
    assert update.shape == (batch_size, param_dim)
    assert new_state.shape == (batch_size, param_dim)
    
    # Check that state is updated (unlikely to be exactly zero unless gradients are zero and network is zero)
    # With random init, it should be non-zero
    assert not torch.allclose(new_state, state)
    
    # Check that update equals new_state (as per implementation)
    assert torch.allclose(update, new_state)

def test_deep_momentum_optimizer_autograd():
    # Verify that the optimizer itself is differentiable (for meta-learning)
    param_dim = 10
    optimizer = DeepMomentumOptimizer(param_dim=param_dim)
    
    grad = torch.randn(1, param_dim)
    state = torch.randn(1, param_dim)
    
    # We need to check if gradients flow back to optimizer parameters
    # But wait, in the forward pass:
    # m_next = alpha * m_i - eta * grad_energy
    # grad_energy depends on l2_energy_net parameters.
    # So m_next should depend on l2_energy_net parameters.
    
    update, new_state = optimizer(grad, state)
    
    # Compute a loss on the update
    loss = update.sum()
    loss.backward()
    
    # Check if optimizer parameters have gradients
    has_grad = False
    for param in optimizer.l2_energy_net.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
            
    assert has_grad, "Optimizer parameters should receive gradients"