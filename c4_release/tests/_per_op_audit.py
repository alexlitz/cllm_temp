"""Per-op audit helpers shared by ``test_l<N>_per_op.py`` test modules.

* :func:`make_stub_block` constructs an FFN/attention/block stub for a
  single-op bake.
* :func:`fires_during_bake` bakes the op and reports which weight
  buffers acquired non-zero entries (regression guard against a flag
  stuck off).
* :func:`assert_no_drift` bakes the op twice and asserts the resulting
  weights are bit-identical (catches non-deterministic bakes).
* :func:`apply_attention` simulates a multi-head attention forward over
  a residual stream so a symbolic test can verify that a single
  attention op routes a hand-crafted source row to the expected
  ``OUTPUT_*`` slots.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

import torch

# Mirror the path-mangling other tests/_*.py helpers use so this module is
# importable both as ``tests._per_op_audit`` and directly from the worktree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StubFFN:
    """Mimics :class:`neural_vm.base_layers.PureFFN` without a real forward.

    Exposes ``W_up`` / ``b_up`` / ``W_gate`` / ``b_gate`` / ``W_down`` /
    ``b_down`` as bare tensors.  Bake helpers detect bare tensors via
    ``getattr(..., 'data', tensor)`` and mutate them with the same
    indexed-assignment syntax they use on real ``nn.Parameter`` objects.
    """

    def __init__(self, *, d_model: int = 512, hidden_dim: int = 4096):
        self.W_up = torch.zeros(hidden_dim, d_model)
        self.b_up = torch.zeros(hidden_dim)
        self.W_gate = torch.zeros(hidden_dim, d_model)
        self.b_gate = torch.zeros(hidden_dim)
        self.W_down = torch.zeros(d_model, hidden_dim)
        self.b_down = torch.zeros(d_model)


class StubAttn:
    """Minimal stand-in for :class:`AutoregressiveAttention`."""

    def __init__(self, *, d_model: int = 512, num_heads: int = 8):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.W_q = torch.zeros(d_model, d_model)
        self.W_k = torch.zeros(d_model, d_model)
        self.W_v = torch.zeros(d_model, d_model)
        self.W_o = torch.zeros(d_model, d_model)
        self.alibi_slopes = torch.zeros(num_heads)


class StubBlock:
    """``kind="block"`` ops receive a TransformerBlock-like wrapper."""

    def __init__(self, ffn: StubFFN, attn: Optional[StubAttn] = None):
        self.ffn = ffn
        if attn is not None:
            self.attn = attn


def make_stub_block(
    *,
    d_model: int = 512,
    ffn_hidden: int = 4096,
    num_heads: int = 8,
    with_attn: bool = True,
) -> StubBlock:
    """Construct a freshly-zeroed stub block for a single-op bake.

    Default ``ffn_hidden=4096`` matches the production block width and
    covers the largest single L14 op (addr_key_neural_decode at ~1728
    units).
    """

    ffn = StubFFN(d_model=d_model, hidden_dim=ffn_hidden)
    attn = StubAttn(d_model=d_model, num_heads=num_heads) if with_attn else None
    return StubBlock(ffn, attn)


def _module_for_op(op, block: StubBlock):
    """Return the target the op's bake_fn expects (block, ffn, or attn)."""

    kind = getattr(op, "kind", None)
    if kind == "ffn":
        return block.ffn
    if kind == "attn":
        return block.attn
    return block


_FFN_BUFFER_NAMES = ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down")
_ATTN_BUFFER_NAMES = ("W_q", "W_k", "W_v", "W_o", "alibi_slopes")


def _all_weight_buffers(block: StubBlock) -> Iterable[tuple[str, torch.Tensor]]:
    for name in _FFN_BUFFER_NAMES:
        yield f"ffn.{name}", getattr(block.ffn, name)
    attn = getattr(block, "attn", None)
    if attn is not None:
        for name in _ATTN_BUFFER_NAMES:
            t = getattr(attn, name, None)
            if isinstance(t, torch.Tensor):
                yield f"attn.{name}", t


def fires_during_bake(
    op,
    dim_positions,
    *,
    S: float = 100.0,
    block: Optional[StubBlock] = None,
) -> dict[str, int]:
    """Bake ``op`` against a fresh stub and report nonzero counts per buffer.

    Maps ``"attn.W_q"``/``"ffn.W_up"``/... to the number of non-zero
    entries the bake wrote.  A test asserts a particular buffer has a
    positive count (the op fired) or that the total non-zero count is
    positive (the op did *something*).
    """

    target_block = block if block is not None else make_stub_block()
    target = _module_for_op(op, target_block)
    op.bake_fn(target, dim_positions, S)
    return {
        name: int((buf != 0).sum().item())
        for name, buf in _all_weight_buffers(target_block)
    }


def assert_no_drift(op, dim_positions, *, S: float = 100.0) -> None:
    """Bake ``op`` twice and assert the resulting weights are bit-identical."""

    a = make_stub_block()
    b = make_stub_block()
    op.bake_fn(_module_for_op(op, a), dim_positions, S)
    op.bake_fn(_module_for_op(op, b), dim_positions, S)
    for (name_a, buf_a), (name_b, buf_b) in zip(
        _all_weight_buffers(a), _all_weight_buffers(b)
    ):
        assert name_a == name_b
        if torch.equal(buf_a, buf_b):
            continue
        diff = (buf_a - buf_b).abs()
        raise AssertionError(
            f"bake drift detected in {name_a} for op={op.name!r}: "
            f"{int((diff != 0).sum().item())} entries differ, "
            f"max |Δ|={float(diff.max())}"
        )


def apply_attention(
    attn: StubAttn,
    rows: torch.Tensor,
    *,
    alibi_slopes: Optional[torch.Tensor] = None,
    softmax1: bool = True,
) -> torch.Tensor:
    """Run a stripped-down multi-head attention forward over ``rows``.

    ``rows`` is ``[seq, d_model]``.  Output is ``[seq, d_model]`` ready
    to be indexed as ``out[query_pos, OUTPUT_LO + nibble]``.

    Mirrors :class:`AutoregressiveAttention.forward` numerics: scores
    are scaled by ``head_dim ** -0.5``, ALiBi bias is
    ``-slope * |i - j|`` (absolute distance — see
    ``vm_step.py:455-466``), and ``softmax1`` appends a zero-anchor
    sink that yields V=0 when no source qualifies.
    """

    seq, d_model = rows.shape
    H = attn.num_heads
    HD = d_model // H
    scale = getattr(attn, "scale", HD ** -0.5)

    q = (rows @ attn.W_q.t()).view(seq, H, HD)
    k = (rows @ attn.W_k.t()).view(seq, H, HD)
    v = (rows @ attn.W_v.t()).view(seq, H, HD)

    causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    pos = torch.arange(seq, dtype=torch.float32)
    dist_abs = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()

    out_per_head = torch.zeros(seq, H, HD)
    for h in range(H):
        scores = (q[:, h, :] @ k[:, h, :].t()) * scale
        if alibi_slopes is not None:
            scores = scores - float(alibi_slopes[h]) * dist_abs
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        if softmax1:
            ext = torch.cat([scores, torch.zeros(seq, 1)], dim=1)
            weights = torch.softmax(ext, dim=-1)[:, :seq]
        else:
            weights = torch.softmax(scores, dim=-1)
        out_per_head[:, h, :] = weights @ v[:, h, :]

    return out_per_head.reshape(seq, H * HD) @ attn.W_o.t()


__all__ = [
    "StubFFN",
    "StubAttn",
    "StubBlock",
    "make_stub_block",
    "fires_during_bake",
    "assert_no_drift",
    "apply_attention",
]
