import sys
from pathlib import Path

import pytest
import torch


# # make "src" importable when running pytest from the repo root
# def _ensure_src_on_path():
#     this_file = Path(__file__).resolve()
#     for base in [this_file.parent, *this_file.parents]:
#         candidate = base / "src"
#         if (candidate / "memory" / "cms.py").exists():
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
from src.memory.cms import CMS  # noqa: E402


def _levels_config(freqs):
    return [{"frequency": f} for f in freqs]


@pytest.mark.parametrize(
    "dim,batch,seq_len,freqs",
    [
        (32, 2, 4, [1, 2, 4]),
        (64, 1, 3, [2, 3]),
    ],
)
def test_forward_backward_shapes_and_grads(dim, batch, seq_len, freqs):
    torch.manual_seed(0)
    model = CMS(dim=dim, levels_config=_levels_config(freqs)).train()

    x = torch.randn(batch, seq_len, dim, dtype=torch.float32)
    y = model(x)

    # shape preserved through CMS (residual inside)
    assert y.shape == (batch, seq_len, dim)

    # simple scalar loss for backward
    loss = y.sum()
    loss.backward()

    # ensure all params received gradients
    for p in model.parameters():
        assert p.grad is not None, "some parameters did not receive gradients"

    # no NaNs/Infs in outputs
    assert torch.isfinite(y).all()


def test_eval_mode_runs():
    torch.manual_seed(123)
    dim, batch, seq_len = 48, 2, 5
    model = CMS(dim=dim, levels_config=_levels_config([1, 2, 5])).eval()

    x = torch.randn(batch, seq_len, dim)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch, seq_len, dim)
    assert torch.isfinite(y).all()


def test_deterministic_given_fixed_seed():
    # same seed + identical construction -> identical outputs
    torch.manual_seed(2024)
    dim, batch, seq_len = 40, 3, 7
    freqs = [1, 3, 4]

    m1 = CMS(dim=dim, levels_config=_levels_config(freqs))
    x = torch.randn(batch, seq_len, dim)
    y1 = m1(x)

    torch.manual_seed(2024)
    m2 = CMS(dim=dim, levels_config=_levels_config(freqs))
    y2 = m2(x)

    assert torch.allclose(y1, y2, atol=0, rtol=0)


def test_get_parameters_by_frequency_selection():
    dim = 16
    freqs = [1, 2, 3, 5]
    model = CMS(dim=dim, levels_config=_levels_config(freqs))

    # steps to check: (step, expected layer indices included)
    checks = [
        (1, [0]),          # only freq 1
        (2, [0, 1]),       # 1 and 2
        (3, [0, 2]),       # 1 and 3
        (5, [0, 3]),       # 1 and 5
        (6, [0, 1, 2]),    # 1, 2, 3
        (10, [0, 1, 3]),   # 1, 2, 5
    ]

    for step, expected_layers in checks:
        params = model.get_parameters_by_frequency(step)
        got = {id(p) for p in params}

        expected = set()
        for li in expected_layers:
            for p in model.layers[li].parameters():
                expected.add(id(p))

        assert got == expected, f"step {step}: expected layers {expected_layers}, got different parameter set"