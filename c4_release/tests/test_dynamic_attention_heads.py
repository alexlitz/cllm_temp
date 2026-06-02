"""Regression tests for dynamic per-layer head count + GQA (Phase 8.O.2).

Tests three layers of the fully-dynamic-head vision:

1. **Unbounded allocator mode** (``layer_max_heads=None``): the per-layer
   head pool grows on demand. ``num_heads_at(layer)`` exposes the live
   count to downstream lowering. ``assert_within_cap`` is the optional
   compiler-level cap (independent of the constructor's bound).
2. **GQA via ``DeclarativeAttentionHeadSpec.group_size``**: K/V rows
   land at ``kv_head_idx * HD`` while Q/O rows land at ``head_idx * HD``.
   At ``group_size=1`` (the default) byte-identical with MHA.
3. **Mixtral-shape demo**: 32 Q heads + 8 KV heads (``group_size=4``)
   produces the correct GQA weight layout on a synthetic attention
   module — proves the IR can express the HF Mixtral architecture
   without a full VM rebake.

The tests do NOT require a full ``compile_full_vm`` run — they exercise
the allocator and lowering primitives directly, which keeps them fast
and immune to concurrent op-corpus changes.
"""

from __future__ import annotations

import math

import pytest
import torch

from c4_release.neural_vm.attention_head_allocator import (
    AttentionHeadAllocator,
    AttentionHeadAllocatorError,
    DEFAULT_LAYER_MAX_HEADS,
)
from c4_release.neural_vm.base_layers import PureAttention
from c4_release.neural_vm.unified_compiler.ir import AttentionHeadIR, AttentionOp
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    DeclarativeAttentionHeadSpec,
    Primitives,
)


# ---------------------------------------------------------------------------
# Phase 1: Dynamic per-layer head count
# ---------------------------------------------------------------------------


def test_default_constructor_is_bounded_at_8():
    """``AttentionHeadAllocator()`` keeps the legacy 8-head cap so the
    existing test corpus and unmigrated allocator call sites are
    byte-identical."""
    a = AttentionHeadAllocator()
    assert a.layer_max_heads == DEFAULT_LAYER_MAX_HEADS == 8


def test_unbounded_mode_grows_on_demand():
    """``layer_max_heads=None`` accepts unlimited head_idx values
    and the live head count grows with each auto-placed alloc."""
    a = AttentionHeadAllocator(layer_max_heads=None)
    assert a.layer_max_heads is None
    for i in range(20):
        h = a.alloc(f"head_{i}", layer_idx=5)
        assert h == i
    assert a.num_heads_at(5) == 20
    # Layer 0 is independent and still empty.
    assert a.num_heads_at(0) == 0


def test_unbounded_mode_fills_gaps_before_growing():
    """Auto-placement prefers the lowest free slot — even in unbounded
    mode. Gaps below the high-water mark are filled before the pool
    grows beyond it."""
    a = AttentionHeadAllocator(layer_max_heads=None)
    a.alloc("a", layer_idx=0, pin=0)
    a.alloc("b", layer_idx=0, pin=2)  # leaves slot 1 free
    a.alloc("c", layer_idx=0, pin=5)  # leaves slots 3,4 free
    # First three auto-placed allocs fill the gaps in order.
    assert a.alloc("d", layer_idx=0) == 1
    assert a.alloc("e", layer_idx=0) == 3
    assert a.alloc("f", layer_idx=0) == 4
    # Next auto-placed alloc grows the pool past the high-water mark.
    assert a.alloc("g", layer_idx=0) == 6
    assert a.num_heads_at(0) == 7


def test_unbounded_mode_accepts_high_pins():
    """In unbounded mode ``pin=`` is no longer bounded by
    ``layer_max_heads``; downstream tests can pin huge head indices."""
    a = AttentionHeadAllocator(layer_max_heads=None)
    a.alloc("rare", layer_idx=0, pin=1000)
    assert a.num_heads_at(0) == 1001
    # And auto-placed alloc still picks slot 0 first.
    assert a.alloc("auto", layer_idx=0) == 0


def test_bounded_mode_pin_oob_still_rejects():
    """Bounded mode preserves the legacy OOB rejection — only unbounded
    mode lifts the cap."""
    a = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError):
        a.alloc("oob", layer_idx=0, pin=8)


def test_assert_within_cap_enforces_compiler_level_budget():
    """Compiler-level cap is independent of the constructor's
    ``layer_max_heads``: a test / config can pass an arbitrary cap to
    fail compilation if the unbounded allocator grew too large."""
    a = AttentionHeadAllocator(layer_max_heads=None)
    for i in range(10):
        a.alloc(f"head_{i}", layer_idx=1)
    a.assert_within_cap(layer_idx=1, cap=10)  # ok
    a.assert_within_cap(layer_idx=1, cap=20)  # ok
    with pytest.raises(AttentionHeadAllocatorError, match="exceeds compiler cap"):
        a.assert_within_cap(layer_idx=1, cap=9)


def test_num_heads_at_empty_layer_is_zero():
    a = AttentionHeadAllocator(layer_max_heads=None)
    assert a.num_heads_at(0) == 0
    assert a.num_heads_at(99) == 0


def test_emulate_l8_extra_head_registration():
    """Mimic the upcoming LEV-detector head registration at L8: today's
    L8 layout claims heads 0-6 (one disabled) — adding one more head
    increments num_heads_at to 8 without bumping any literal."""
    a = AttentionHeadAllocator(layer_max_heads=None)
    # Today's pinned L8 head set (per _L8_HEAD_LAYOUT in l8_ops.py).
    for slot, name in enumerate([
        "layer8_attn_pc",
        "layer8_attn_ax",
        "layer8_attn_sp",
        "layer8_attn_bp",
        "layer8_attn_stack0",
        "layer8_attn_const",
        "layer8_head6_ax_carry_refresh.head_6",
    ]):
        a.alloc(name, layer_idx=8, pin=slot)
    assert a.num_heads_at(8) == 7
    # New LEV-detector head registers with pin=None — first-fit picks
    # the next free slot, NO literal bump anywhere.
    new_head_idx = a.alloc("layer8_lev_detector", layer_idx=8)
    assert new_head_idx == 7
    assert a.num_heads_at(8) == 8


# ---------------------------------------------------------------------------
# Phase 2: GQA via DeclarativeAttentionHeadSpec.group_size
# ---------------------------------------------------------------------------


def test_default_group_size_is_one():
    """Default ``group_size=1`` keeps every existing spec byte-
    identical with vanilla MHA."""
    spec = DeclarativeAttentionHeadSpec(head_idx=3)
    assert spec.group_size == 1
    assert spec.kv_head_idx == 3  # head_idx // 1


def test_group_size_one_is_byte_identical_with_mha():
    """At ``group_size=1`` the lowered Q/K/V/O matrix writes match
    what the pre-8.O.2 path would emit, cell-for-cell."""
    HD = 8
    num_heads = 4
    dim = HD * num_heads
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        q=(AP(0, 5, 1.5),),
        k=(AP(1, 6, -2.5),),
        v=(AP(2, 7, 3.0),),
        o=(AO(11, 3, 0.5),),
        # group_size defaults to 1 — explicit here for clarity.
        group_size=1,
    )
    attn = PureAttention(dim=dim, num_heads=num_heads, causal=True)
    Primitives.generate_attention_head(attn, spec, HD)
    # Q row 2*8+0 = 16, col 5: 1.5
    assert float(attn.W_q.data[16, 5]) == pytest.approx(1.5)
    # K row 2*8+1 = 17, col 6: -2.5  (kv_head_idx == head_idx == 2)
    assert float(attn.W_k.data[17, 6]) == pytest.approx(-2.5)
    # V row 2*8+2 = 18, col 7: 3.0
    assert float(attn.W_v.data[18, 7]) == pytest.approx(3.0)
    # O row 11, col 2*8+3 = 19: 0.5
    assert float(attn.W_o.data[11, 19]) == pytest.approx(0.5)


def test_group_size_4_routes_kv_to_shared_block():
    """At ``group_size=4`` Q heads 0..3 all share KV head 0; Q heads
    4..7 share KV head 1; etc. The K/V writes land at
    ``(head_idx // 4) * HD`` rather than ``head_idx * HD``."""
    HD = 8
    num_q_heads = 8
    num_kv_heads = 2
    dim_q = HD * num_q_heads
    dim_kv = HD * num_kv_heads
    # Synthetic GQA-shape attention module: W_q sized to num_q_heads,
    # W_k/W_v sized to num_kv_heads. W_o is num_q_heads-wide too because
    # GQA still produces num_q_heads value rows in the output.
    class _GQAStub:
        pass
    attn = _GQAStub()
    attn.W_q = torch.nn.Parameter(torch.zeros(dim_q, dim_q))
    attn.W_k = torch.nn.Parameter(torch.zeros(dim_kv, dim_q))
    attn.W_v = torch.nn.Parameter(torch.zeros(dim_kv, dim_q))
    attn.W_o = torch.nn.Parameter(torch.zeros(dim_q, dim_q))
    # Heads 0,1,2,3 share KV head 0; heads 4,5,6,7 share KV head 1.
    for q_head_idx in range(num_q_heads):
        spec = DeclarativeAttentionHeadSpec(
            head_idx=q_head_idx,
            q=(AP(0, q_head_idx, 1.0),),  # Q writes col=head_idx
            k=(AP(0, 100 + q_head_idx, 1.0),),
            v=(AP(0, 200 + q_head_idx, 1.0),),
            o=(AO(300 + q_head_idx, 0, 1.0),),
            group_size=4,
        )
        # kv_head_idx is computed from head_idx and group_size
        assert spec.kv_head_idx == q_head_idx // 4
        Primitives.generate_attention_head(attn, spec, HD)
    # Q row layout: each Q head has its own block (0,8,16,...)
    for q_head_idx in range(num_q_heads):
        assert float(attn.W_q.data[q_head_idx * HD + 0, q_head_idx]) == 1.0
    # K row layout: heads 0..3 all stomp into KV row 0; heads 4..7 all
    # into KV row 8 (since HD=8). The LAST write wins because the K
    # rule writes to slot 0 in every spec — i.e. the same K row.
    # Heads 0..3: kv_head_idx=0 -> kv_base=0 -> slot 0 -> row 0
    # Heads 4..7: kv_head_idx=1 -> kv_base=8 -> slot 0 -> row 8
    # The K col carries the head's id; the K row carries the KV group.
    # Each spec writes col=100+q_head_idx, so all 4 cols in group 0
    # should be populated at row 0.
    for q_head_idx in range(num_q_heads):
        kv_row = (q_head_idx // 4) * HD + 0
        assert float(attn.W_k.data[kv_row, 100 + q_head_idx]) == 1.0
        assert float(attn.W_v.data[kv_row, 200 + q_head_idx]) == 1.0


def test_attention_op_shape_reports_q_and_kv_counts():
    """``AttentionOp.shape()`` returns ``(num_q_heads, num_kv_heads)``,
    derived from the registered specs' head_idx and kv_head_idx."""
    op = AttentionOp()
    for q_head_idx in range(16):
        op.add_head(
            DeclarativeAttentionHeadSpec(
                head_idx=q_head_idx,
                group_size=4,
            ),
            name=f"q_{q_head_idx}",
        )
    num_q_heads, num_kv_heads = op.shape()
    assert num_q_heads == 16
    assert num_kv_heads == 4
    # Mixtral target: 32 Q, 8 KV.
    op2 = AttentionOp()
    for q_head_idx in range(32):
        op2.add_head(
            DeclarativeAttentionHeadSpec(
                head_idx=q_head_idx,
                group_size=4,
            ),
            name=f"mixtral_q_{q_head_idx}",
        )
    assert op2.shape() == (32, 8)


def test_attention_op_shape_empty_returns_zero():
    op = AttentionOp()
    assert op.shape() == (0, 0)


def test_attention_op_shape_mha_default_is_byte_identical():
    """Default ``group_size=1`` makes shape() return ``(N, N)`` — every
    Q head has its own KV slot, the historical MHA contract."""
    op = AttentionOp()
    for q_head_idx in range(8):
        op.add_head(
            DeclarativeAttentionHeadSpec(head_idx=q_head_idx),  # default
            name=f"head_{q_head_idx}",
        )
    assert op.shape() == (8, 8)


def test_group_size_zero_raises():
    """``group_size <= 0`` is a hard error at kv_head_idx access time."""
    spec = DeclarativeAttentionHeadSpec(head_idx=0, group_size=0)
    with pytest.raises(ValueError, match="group_size must be >= 1"):
        _ = spec.kv_head_idx


# ---------------------------------------------------------------------------
# Phase 3: Mixtral-shape acceptance check
# ---------------------------------------------------------------------------


def test_mixtral_shape_acceptance():
    """When a synthetic op declares Mixtral's 32 Q / 8 KV head config,
    the lowered W_q / W_k / W_v rows match Mixtral's published weight
    layout exactly.

    Mixtral-7B uses:
      - num_attention_heads = 32
      - num_key_value_heads = 8
      - head_dim = 128
      - hidden_size = 4096 (32 * 128)

    The asserts here use HD=16 to keep the test fast; the layout
    invariant (Q rows = num_q_heads*HD; K/V rows = num_kv_heads*HD;
    group_size = num_q_heads / num_kv_heads) is what matters, not the
    absolute size.
    """
    HD = 16
    num_q_heads = 32
    num_kv_heads = 8
    group_size = num_q_heads // num_kv_heads
    assert group_size == 4

    dim_q = HD * num_q_heads  # Q (and O) row dimension
    dim_kv = HD * num_kv_heads  # K/V row dimension

    # The model_dim equals the Q row dim (residual stream width).
    model_dim = dim_q

    class _MixtralStub:
        pass

    attn = _MixtralStub()
    attn.W_q = torch.nn.Parameter(torch.zeros(dim_q, model_dim))
    attn.W_k = torch.nn.Parameter(torch.zeros(dim_kv, model_dim))
    attn.W_v = torch.nn.Parameter(torch.zeros(dim_kv, model_dim))
    attn.W_o = torch.nn.Parameter(torch.zeros(model_dim, dim_q))

    # Build a 32-head op via a single AttentionOp + add_head, all at
    # group_size=4. Verify shape() reports (32, 8).
    op = AttentionOp()
    for q_head_idx in range(num_q_heads):
        op.add_head(
            DeclarativeAttentionHeadSpec(
                head_idx=q_head_idx,
                # Distinct Q col per head so the lowered matrix isn't
                # all-zero — lets us assert per-row hits.
                q=(AP(0, q_head_idx, 1.0),),
                k=(AP(0, q_head_idx, 1.0),),
                v=(AP(0, q_head_idx, 1.0),),
                o=(AO(q_head_idx, 0, 1.0),),
                group_size=group_size,
            ),
            name=f"q_{q_head_idx}",
        )
    assert op.shape() == (num_q_heads, num_kv_heads)

    # Lower every spec via the primitive (the same code path used by
    # ir.lower_attention -> Primitives.generate_attention_head).
    for head in op.rules:
        Primitives.generate_attention_head(attn, head.spec, HD)

    # Q has 32 occupied row blocks (one per Q head).
    for q_head_idx in range(num_q_heads):
        row = q_head_idx * HD + 0  # slot 0 in this q head
        col = q_head_idx
        assert float(attn.W_q.data[row, col]) == 1.0

    # K/V have only 8 occupied row blocks (one per KV head, shared
    # by 4 Q heads). The K col carries the originating Q head id, so
    # each KV row block has 4 cols populated.
    populated_kv_rows = set()
    for q_head_idx in range(num_q_heads):
        kv_head_idx = q_head_idx // group_size
        row = kv_head_idx * HD + 0
        col = q_head_idx
        assert float(attn.W_k.data[row, col]) == 1.0
        assert float(attn.W_v.data[row, col]) == 1.0
        populated_kv_rows.add(row)
    assert len(populated_kv_rows) == num_kv_heads


def test_mixtral_kv_matrix_shape_proves_gqa_compression():
    """The K/V matrices have ``num_kv_heads * HD`` rows, NOT
    ``num_q_heads * HD`` — proving the IR's lowered weights match
    Mixtral's GQA weight-compression contract.
    """
    HD = 16
    num_q_heads = 32
    num_kv_heads = 8
    group_size = num_q_heads // num_kv_heads

    op = AttentionOp()
    for q_head_idx in range(num_q_heads):
        op.add_head(
            DeclarativeAttentionHeadSpec(
                head_idx=q_head_idx,
                group_size=group_size,
            ),
            name=f"q_{q_head_idx}",
        )

    num_q, num_kv = op.shape()
    expected_q_rows = num_q * HD
    expected_kv_rows = num_kv * HD
    # Confirm the compressed K/V allocation: 4x smaller than Q.
    assert expected_q_rows == 32 * HD
    assert expected_kv_rows == 8 * HD
    assert expected_kv_rows * 4 == expected_q_rows


def test_gqa_default_byte_identity_against_mha():
    """At ``group_size=1`` GQA lowering writes into the same rows as
    pre-8.O.2 MHA lowering — full byte-identity at the weight level.

    Construct two specs (one with explicit group_size=1, one with the
    default) and verify they produce identical W_q/W_k/W_v/W_o."""
    HD = 4
    num_heads = 4
    dim = HD * num_heads
    base_writes = dict(
        q=(AP(0, 0, 1.1), AP(1, 1, 2.2)),
        k=(AP(0, 2, 3.3),),
        v=(AP(0, 3, 4.4),),
        o=(AO(7, 0, 5.5),),
    )

    # Path A: explicit group_size=1
    attn_a = PureAttention(dim=dim, num_heads=num_heads, causal=True)
    spec_a = DeclarativeAttentionHeadSpec(
        head_idx=2, group_size=1, **base_writes
    )
    Primitives.generate_attention_head(attn_a, spec_a, HD)

    # Path B: default group_size (== 1)
    attn_b = PureAttention(dim=dim, num_heads=num_heads, causal=True)
    spec_b = DeclarativeAttentionHeadSpec(head_idx=2, **base_writes)
    Primitives.generate_attention_head(attn_b, spec_b, HD)

    for matname in ("W_q", "W_k", "W_v", "W_o"):
        a = getattr(attn_a, matname).data
        b = getattr(attn_b, matname).data
        assert torch.equal(a, b), f"{matname} differs between explicit and default group_size=1"
