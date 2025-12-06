import sys
from pathlib import Path

import pytest
import torch

# # make "src" importable when running pytest from the repo root
# THIS_DIR = Path(__file__).parent
# REPO_ROOT = THIS_DIR.parent.resolve()
# SRC_PATH = REPO_ROOT / "src"
# if str(SRC_PATH) not in sys.path:
#     sys.path.insert(0, str(SRC_PATH))

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
from src.layers.projections import DynamicProjections  # noqa: E402


@pytest.mark.parametrize("dim,head_dim", [(64, 16), (128, 32)])
@pytest.mark.parametrize("batch,seq_len", [(2, 4), (4, 8)])
def test_static_mode_forward_shapes_and_backward(dim, head_dim, batch, seq_len):
    """Test static mode (use_dynamic=False) for backward compatibility."""
    torch.manual_seed(0)

    model = DynamicProjections(dim=dim, head_dim=head_dim, use_dynamic=False)
    model.train()

    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)

    q, k, v = model(x)

    assert q.shape == (batch, seq_len, head_dim)
    assert k.shape == (batch, seq_len, head_dim)
    assert v.shape == (batch, seq_len, head_dim)

    # simple scalar loss to test backward
    loss = q.sum() + k.sum() + v.sum()
    loss.backward()

    # ensure params received gradients
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "some parameters did not receive gradients"
    # basic sanity: no NaNs/Infs
    for tensor in (q, k, v):
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("dim,head_dim", [(64, 16), (128, 32)])
@pytest.mark.parametrize("batch,seq_len", [(2, 4), (4, 8)])
def test_dynamic_mode_forward_shapes_and_backward(dim, head_dim, batch, seq_len):
    """Test dynamic mode (use_dynamic=True) with Equation 28 updates."""
    torch.manual_seed(0)

    model = DynamicProjections(dim=dim, head_dim=head_dim, use_dynamic=True, learning_rate=0.01)
    model.train()

    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)

    q, k, v, states = model(x)

    assert q.shape == (batch, seq_len, head_dim)
    assert k.shape == (batch, seq_len, head_dim)
    assert v.shape == (batch, seq_len, head_dim)
    
    # Check states are returned
    assert states is not None
    W_q, W_k, W_v = states
    assert W_q.shape == (batch, head_dim, dim)
    assert W_k.shape == (batch, head_dim, dim)
    assert W_v.shape == (batch, head_dim, dim)

    # simple scalar loss to test backward
    loss = q.sum() + k.sum() + v.sum()
    loss.backward()

    # ensure params received gradients
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "some parameters did not receive gradients"
    # basic sanity: no NaNs/Infs
    for tensor in (q, k, v):
        assert torch.isfinite(tensor).all()


def test_eval_mode_runs_static():
    """Test evaluation mode with static projections."""
    torch.manual_seed(123)
    model = DynamicProjections(dim=32, head_dim=8, use_dynamic=False).eval()
    x = torch.randn(1, 3, 32)
    q, k, v = model(x)
    assert q.shape == (1, 3, 8)
    assert k.shape == (1, 3, 8)
    assert v.shape == (1, 3, 8)


def test_eval_mode_runs_dynamic():
    """Test evaluation mode with dynamic projections."""
    torch.manual_seed(123)
    model = DynamicProjections(dim=32, head_dim=8, use_dynamic=True).eval()
    x = torch.randn(1, 3, 32)
    q, k, v, states = model(x)
    assert q.shape == (1, 3, 8)
    assert k.shape == (1, 3, 8)
    assert v.shape == (1, 3, 8)
    assert states is not None


def test_deterministic_given_fixed_seed_static():
    """Test determinism with static projections."""
    # with the same seed and identical construction, weights should match → outputs should match
    dim, head_dim = 48, 12
    batch, seq_len = 2, 5

    torch.manual_seed(2024)
    m1 = DynamicProjections(dim, head_dim, use_dynamic=False)
    x = torch.randn(batch, seq_len, dim)

    q1, k1, v1 = m1(x)

    torch.manual_seed(2024)
    m2 = DynamicProjections(dim, head_dim, use_dynamic=False)
    q2, k2, v2 = m2(x)

    assert torch.allclose(q1, q2, atol=0, rtol=0)
    assert torch.allclose(k1, k2, atol=0, rtol=0)
    assert torch.allclose(v1, v2, atol=0, rtol=0)


def test_deterministic_given_fixed_seed_dynamic():
    """Test determinism with dynamic projections."""
    dim, head_dim = 48, 12
    batch, seq_len = 2, 5

    torch.manual_seed(2024)
    m1 = DynamicProjections(dim, head_dim, use_dynamic=True)
    x = torch.randn(batch, seq_len, dim)

    q1, k1, v1, states1 = m1(x)

    torch.manual_seed(2024)
    m2 = DynamicProjections(dim, head_dim, use_dynamic=True)
    q2, k2, v2, states2 = m2(x)

    assert torch.allclose(q1, q2, atol=1e-6, rtol=1e-6)
    assert torch.allclose(k1, k2, atol=1e-6, rtol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6, rtol=1e-6)


def test_weight_updates_occur():
    """Test that weights actually change during dynamic updates."""
    torch.manual_seed(42)
    
    dim, head_dim = 32, 16
    batch, seq_len = 2, 4
    
    model = DynamicProjections(dim, head_dim, use_dynamic=True, learning_rate=0.1)
    x = torch.randn(batch, seq_len, dim)
    
    # Get initial weights
    W_q_init, W_k_init, W_v_init = model.reset_weights(batch, x.device)
    
    # Run forward pass
    q, k, v, (W_q_final, W_k_final, W_v_final) = model(x)
    
    # Weights should have changed
    assert not torch.allclose(W_q_init, W_q_final, atol=1e-6), "W_q did not update"
    assert not torch.allclose(W_k_init, W_k_final, atol=1e-6), "W_k did not update"
    assert not torch.allclose(W_v_init, W_v_final, atol=1e-6), "W_v did not update"


def test_state_carryover():
    """Test that states can be carried over between forward passes."""
    torch.manual_seed(99)
    
    dim, head_dim = 24, 12
    batch = 2
    
    model = DynamicProjections(dim, head_dim, use_dynamic=True, learning_rate=0.05)
    
    # First chunk
    x1 = torch.randn(batch, 3, dim)
    q1, k1, v1, states1 = model(x1)
    
    # Second chunk with carried state
    x2 = torch.randn(batch, 3, dim)
    q2, k2, v2, states2 = model(x2, states=states1)
    
    # States should have evolved
    W_q1, W_k1, W_v1 = states1
    W_q2, W_k2, W_v2 = states2
    
    assert not torch.allclose(W_q1, W_q2, atol=1e-6)
    assert not torch.allclose(W_k1, W_k2, atol=1e-6)
    assert not torch.allclose(W_v1, W_v2, atol=1e-6)


def test_shared_weights_mode():
    """Test shared weights mode where Q, K, V use same weight matrix."""
    torch.manual_seed(777)
    
    dim, head_dim = 32, 16
    batch, seq_len = 2, 3
    
    model = DynamicProjections(dim, head_dim, use_dynamic=True, share_weights=True)
    x = torch.randn(batch, seq_len, dim)
    
    q, k, v, (W_q, W_k, W_v) = model(x)
    
    # Initially should be the same (they reference the same object or are equal)
    # After updates they will diverge because each gets different LSS
    # But we can at least verify the model runs
    assert q.shape == (batch, seq_len, head_dim)
    assert k.shape == (batch, seq_len, head_dim)
    assert v.shape == (batch, seq_len, head_dim)