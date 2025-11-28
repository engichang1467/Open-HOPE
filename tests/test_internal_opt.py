import sys
from pathlib import Path

import pytest
import torch


# # make "src" importable when running pytest from the repo root
# def _ensure_src_on_path():
#     this_file = Path(__file__).resolve()
#     for base in [this_file.parent, *this_file.parents]:
#         candidate = base / "src"
#         if (candidate / "optimizers" / "internal_opt.py").exists():
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
from src.optimizers.internal_opt import InternalOptimizer  # noqa: E402


@pytest.mark.parametrize(
    "batch,hidden_dim,lr",
    [
        (1, 4, 1e-2),
        (3, 8, 5e-3),
    ],
)
def test_forward_matches_equation_28(batch, hidden_dim, lr):
    torch.manual_seed(0)
    opt = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr).train()

    W = torch.randn(batch, hidden_dim, hidden_dim, dtype=torch.float32)
    x = torch.randn(batch, hidden_dim, 1, dtype=torch.float32)
    grad_L = torch.randn(batch, hidden_dim, 1, dtype=torch.float32)

    W_next = opt(W, x, grad_L)

    # expected per Eq. 28: W(I - x x^T) - lr * (grad_L x^T)
    outer_x = torch.bmm(x, x.transpose(1, 2))
    I = torch.eye(hidden_dim, device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1)
    term1 = torch.bmm(W, (I - outer_x))
    term2 = lr * torch.bmm(grad_L, x.transpose(1, 2))
    expected = term1 - term2

    assert W_next.shape == (batch, hidden_dim, hidden_dim)
    assert W_next.dtype == W.dtype
    assert torch.isfinite(W_next).all()
    assert torch.allclose(W_next, expected, atol=1e-6, rtol=1e-6)


def test_backward_propagates_to_inputs():
    torch.manual_seed(123)
    batch, hidden_dim, lr = 2, 6, 1e-2
    opt = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr).train()

    W = torch.randn(batch, hidden_dim, hidden_dim, requires_grad=True)
    x = torch.randn(batch, hidden_dim, 1, requires_grad=True)
    grad_L = torch.randn(batch, hidden_dim, 1, requires_grad=True)

    W_next = opt(W, x, grad_L)
    loss = (W_next ** 2).mean()
    loss.backward()

    assert W.grad is not None and torch.isfinite(W.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert grad_L.grad is not None and torch.isfinite(grad_L.grad).all()


@pytest.mark.parametrize("batch,hidden_dim", [(1, 5), (4, 7)])
def test_zero_x_returns_W(batch, hidden_dim):
    torch.manual_seed(7)
    lr = 1e-2
    opt = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr).eval()

    W = torch.randn(batch, hidden_dim, hidden_dim)
    x = torch.zeros(batch, hidden_dim, 1)  # zero -> I - x x^T == I and grad term == 0
    grad_L = torch.randn(batch, hidden_dim, 1)

    with torch.no_grad():
        W_next = opt(W, x, grad_L)

    assert torch.allclose(W_next, W, atol=0, rtol=0)


def test_learning_rate_scales_only_grad_term():
    torch.manual_seed(42)
    batch, hidden_dim = 2, 6
    lr1, lr2 = 1e-3, 4e-3

    # the optimizer's internal upcasting and the test's calculation.
    dtype = torch.float64

    W = torch.randn(batch, hidden_dim, hidden_dim, dtype=dtype)
    x = torch.randn(batch, hidden_dim, 1, dtype=dtype)
    grad_L = torch.randn(batch, hidden_dim, 1, dtype=dtype)

    opt1 = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr1)
    opt2 = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr2)

    with torch.no_grad():
        out1 = opt1(W, x, grad_L)
        out2 = opt2(W, x, grad_L)

    # out = term1 - lr * (grad_L x^T) -> difference is -(lr2 - lr1) * (grad_L x^T)
    G = torch.bmm(grad_L, x.transpose(1, 2))
    expected_diff = -(lr2 - lr1) * G
    assert torch.allclose(out2 - out1, expected_diff, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "W_shape,x_shape,grad_shape",
    [
        # batch mismatch
        ((2, 4, 4), (3, 4, 1), (2, 4, 1)),
        # hidden_dim mismatch in x
        ((2, 5, 5), (2, 6, 1), (2, 5, 1)),
        # hidden_dim mismatch in grad
        ((1, 3, 3), (1, 3, 1), (1, 2, 1)),
        # wrong rank for W
        ((3, 3), (1, 3, 1), (1, 3, 1)),
    ],
)
def test_wrong_shapes_raise(W_shape, x_shape, grad_shape):
    hidden_dim = W_shape[-1] if len(W_shape) == 3 else 3
    opt = InternalOptimizer(hidden_dim=hidden_dim)

    W = torch.randn(*W_shape)
    x = torch.randn(*x_shape)
    grad_L = torch.randn(*grad_shape)

    with pytest.raises((RuntimeError, ValueError)):
        _ = opt(W, x, grad_L)


def test_deterministic_for_same_inputs():
    torch.manual_seed(99)
    batch, hidden_dim, lr = 3, 5, 1e-2
    opt = InternalOptimizer(hidden_dim=hidden_dim, learning_rate=lr)

    W = torch.randn(batch, hidden_dim, hidden_dim)
    x = torch.randn(batch, hidden_dim, 1)
    grad_L = torch.randn(batch, hidden_dim, 1)

    with torch.no_grad():
        out1 = opt(W, x, grad_L)
        out2 = opt(W, x, grad_L)

    assert torch.allclose(out1, out2, atol=0, rtol=0)