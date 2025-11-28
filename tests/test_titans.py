import torch
import pytest
from pathlib import Path
import sys




# # make "src" importable when running pytest from the repo root
# def _ensure_src_on_path():
#     this_file = Path(__file__).resolve()
#     for base in [this_file.parent, *this_file.parents]:
#         candidate = base / "src"
#         if (candidate / "models" / "titans.py").exists():
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

from src.models.titan import SelfModifyingTitans

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_forward_shapes_and_no_nan():
    torch.manual_seed(123)
    device = _device()

    batch, seqlen, dim, head_dim = 2, 4, 16, 8
    model = SelfModifyingTitans(dim=dim, head_dim=head_dim).to(device)
    x = torch.randn(batch, seqlen, dim, device=device)

    outputs, state = model(x, state=None)

    assert outputs.shape == (batch, seqlen, dim)
    assert state.shape == (batch, head_dim, head_dim)

    assert outputs.dtype == x.dtype
    assert state.dtype == x.dtype

    assert torch.isfinite(outputs).all()
    assert torch.isfinite(state).all()


def test_state_carry_progresses():
    torch.manual_seed(321)
    device = _device()

    batch, dim, head_dim = 2, 32, 12
    s1, s2 = 3, 2
    model = SelfModifyingTitans(dim=dim, head_dim=head_dim).to(device)

    x1 = torch.randn(batch, s1, dim, device=device)
    x2 = torch.randn(batch, s2, dim, device=device)

    out1, st1 = model(x1, state=None)
    out2, st2 = model(x2, state=st1)

    # state should move away from the zero init after first chunk
    assert not torch.allclose(st1, torch.zeros_like(st1), atol=1e-7, rtol=1e-7)
    # and keep changing with new inputs
    assert not torch.allclose(st2, st1, atol=1e-7, rtol=1e-7)

    # quick extra shape/finite checks
    assert out1.shape == (batch, s1, dim)
    assert out2.shape == (batch, s2, dim)
    assert torch.isfinite(out1).all() and torch.isfinite(out2).all()


def test_backward_pass_has_grads():
    torch.manual_seed(999)
    device = _device()

    batch, seqlen, dim, head_dim = 2, 5, 24, 10
    model = SelfModifyingTitans(dim=dim, head_dim=head_dim).to(device)
    x = torch.randn(batch, seqlen, dim, device=device, requires_grad=True)

    outputs, _ = model(x)
    loss = outputs.pow(2).mean()
    loss.backward()

    # at least one parameter should receive a finite gradient
    grads = [p.grad for p in model.parameters()]
    assert any((g is not None) and torch.isfinite(g).all() for g in grads), "no finite gradients found on model params"


def test_deterministic_given_fixed_seed():
    device = _device()

    # fixed input
    torch.manual_seed(2024)
    batch, seqlen, dim, head_dim = 2, 3, 20, 7
    x = torch.randn(batch, seqlen, dim, device=device)

    # same seed -> identical weights -> identical outputs/states
    torch.manual_seed(1234)
    m1 = SelfModifyingTitans(dim=dim, head_dim=head_dim).to(device).eval()
    torch.manual_seed(1234)
    m2 = SelfModifyingTitans(dim=dim, head_dim=head_dim).to(device).eval()

    with torch.no_grad():
        y1, s1 = m1(x)
        y2, s2 = m2(x)

    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-6)