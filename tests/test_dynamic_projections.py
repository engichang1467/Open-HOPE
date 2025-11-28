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
def test_forward_shapes_and_backward(dim, head_dim, batch, seq_len):
    torch.manual_seed(0)

    model = DynamicProjections(dim=dim, head_dim=head_dim)
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


def test_eval_mode_runs():
    torch.manual_seed(123)
    model = DynamicProjections(dim=32, head_dim=8).eval()
    x = torch.randn(1, 3, 32)
    q, k, v = model(x)
    assert q.shape == (1, 3, 8)
    assert k.shape == (1, 3, 8)
    assert v.shape == (1, 3, 8)


def test_deterministic_given_fixed_seed():
    # with the same seed and identical construction, weights should match → outputs should match
    dim, head_dim = 48, 12
    batch, seq_len = 2, 5

    torch.manual_seed(2024)
    m1 = DynamicProjections(dim, head_dim)
    x = torch.randn(batch, seq_len, dim)

    q1, k1, v1 = m1(x)

    torch.manual_seed(2024)
    m2 = DynamicProjections(dim, head_dim)
    q2, k2, v2 = m2(x)

    assert torch.allclose(q1, q2, atol=0, rtol=0)
    assert torch.allclose(k1, k2, atol=0, rtol=0)
    assert torch.allclose(v1, v2, atol=0, rtol=0)