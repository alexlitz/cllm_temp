"""Smoke test for the runtime RoPE forward path on PureAttention.

Pins the Phase 8.O.1 ``positional_encoding`` flag end-to-end:

* The ``"alibi"`` default is byte-identical to the no-kwarg construction
  (the no-regress guarantee for every pre-toggle baseline).
* The ``"rope"`` branch builds a working PureAttention whose forward
  pass actually rotates Q/K with the precomputed cos/sin cache, so its
  output is numerically distinct from the ALiBi path under identical
  Q/K/V/O bakes.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureAttention
from neural_vm.config import reset_config


@pytest.fixture(autouse=True)
def _reset_vm_config(monkeypatch):
    for name in (
        "NEURAL_VM_POS_ENCODING",
        "NEURAL_VM_ROPE_BASE",
        "NEURAL_VM_ATTENTION_NORMALIZATION",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    yield
    reset_config()


def _bake_random_qkvo(attn: PureAttention, *, seed: int) -> None:
    """Fill Q/K/V/O with a small random tensor so the position-encoding
    branch can actually influence the attention scores.

    Zero-init weights would mask the toggle: any positional bias would
    multiply against a zero W_q/W_k matmul and the outputs would be
    indistinguishable.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ("W_q", "W_k", "W_v", "W_o"):
            param = getattr(attn, name)
            param.copy_(
                torch.randn(param.shape, generator=gen, dtype=param.dtype) * 0.1
            )


def test_pure_attention_alibi_default_byte_identical_to_no_kwarg():
    """The ``position_encoding`` default ("alibi") must be byte-identical
    to the no-kwarg PureAttention construction. This is the no-regress
    guarantee for every callsite that never opted into the toggle.
    """
    torch.manual_seed(0)
    attn_default = PureAttention(dim=32, num_heads=4)
    torch.manual_seed(0)
    attn_explicit = PureAttention(dim=32, num_heads=4, position_encoding="alibi")

    # Identical bakes — only the construction kwarg differs.
    _bake_random_qkvo(attn_default, seed=1)
    _bake_random_qkvo(attn_explicit, seed=1)

    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        y_default = attn_default(x)
        y_explicit = attn_explicit(x)

    assert torch.equal(y_default, y_explicit), (
        "PureAttention position_encoding default drifted from no-kwarg baseline."
    )
    # Defensive: ALiBi default must NOT allocate the RoPE cache.
    assert attn_default._rope_cos is None
    assert attn_explicit._rope_cos is None


def test_pure_attention_rope_forward_runs_and_differs_from_alibi():
    """``position_encoding="rope"`` must produce a forward output that is
    finite, well-shaped, AND numerically distinct from the ALiBi forward
    under identical Q/K/V/O bakes. The numerical difference pins the
    rotation to actual cos/sin matmul, not just buffer allocation.
    """
    torch.manual_seed(0)
    attn_alibi = PureAttention(dim=32, num_heads=4)
    attn_rope = PureAttention(
        dim=32, num_heads=4, position_encoding="rope", rope_base=1.0e6,
    )

    _bake_random_qkvo(attn_alibi, seed=2)
    # Copy the bakes onto the RoPE module so Q/K/V/O are identical and the
    # only axis of variation is the position-encoding branch.
    with torch.no_grad():
        for name in ("W_q", "W_k", "W_v", "W_o"):
            getattr(attn_rope, name).copy_(getattr(attn_alibi, name))

    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        y_alibi = attn_alibi(x)
        y_rope = attn_rope(x)

    assert y_rope.shape == y_alibi.shape == (1, 6, 32)
    assert torch.isfinite(y_rope).all()
    assert torch.isfinite(y_alibi).all()
    assert not torch.allclose(y_rope, y_alibi, atol=1e-6), (
        "RoPE and ALiBi PureAttention outputs are identical — the "
        "position_encoding toggle is not influencing the forward."
    )
    # Defensive: RoPE path must allocate the precomputed cos/sin cache.
    assert attn_rope._rope_cos is not None
    assert attn_rope._rope_sin is not None
    assert attn_rope._rope_cos.shape[-1] == attn_rope.head_dim


def test_pure_attention_rope_invalid_kind_rejected():
    """Unknown ``position_encoding`` values are rejected at construction
    time so callers can't silently fall back to ALiBi.
    """
    with pytest.raises(ValueError, match="position_encoding must be"):
        PureAttention(dim=32, num_heads=4, position_encoding="bogus")
