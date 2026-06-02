"""Test V1/V2 vision: per-head dynamic ``head_dim``.

Covers:

1. ``DeclarativeAttentionHeadSpec.head_dim`` defaults to ``None`` and the
   spec stays byte-identical with pre-existing fixed-HD bakes when no
   head overrides the default.
2. ``DeclarativeAttentionHeadSpec.effective_head_dim`` returns the
   per-head override when present, else the layer default.
3. ``Primitives.generate_attention_head(head_base=...)`` lowers writes to
   the supplied row offset, validates slot bounds against
   ``effective_head_dim``, and remains identical to the legacy
   ``head_idx * HD`` formula when ``head_base is None``.
4. ``Primitives.generate_attention_heads`` falls through to the legacy
   path when no spec declares a non-default ``head_dim`` and cumulative-
   sums per-head bases otherwise.
5. ``AttentionHeadAllocator.alloc(head_dim=...)`` records the per-head
   dim, ``total_head_dim_at`` sums them, and ``head_base_at`` produces
   the cumulative-sum offset that the lowering pass needs.
6. The integration smoke test: registering an L8 head with ``head_dim=8``
   (half of today's default per-head width) — assert the lowered Q/K/V
   matrices land at the half-width row count for that head and that the
   layer's compiled ``num_heads * effective_head_dim`` accounts for the
   shorter head.

Per the task brief these are unit-level checks. We don't gate per-commit
byte-identity of ``compile_full_vm_dynamic`` at non-default settings.
"""

from __future__ import annotations

import pytest
import torch

from c4_release.neural_vm.attention_head_allocator import (
    AttentionHeadAllocator,
    AttentionHeadAllocatorError,
)
from c4_release.neural_vm.unified_compiler.primitives import (
    AO,
    AP,
    AttentionOutputWrite,
    AttentionProjectionWrite,
    DeclarativeAttentionHeadSpec,
    Primitives,
)


class _AttnStub:
    """Minimal stand-in for ``PureAttention`` for primitive-level tests.

    Only exposes ``W_q``/``W_k``/``W_v``/``W_o`` as ``.data`` tensors
    (the same surface that ``Primitives.generate_attention_head`` touches)
    so we can assert exact (row, col, weight) tuples after a bake without
    pulling the whole VM.
    """

    def __init__(self, dim_rows: int, dim_cols: int):
        # Use a tiny container that mimics the ``.data`` attribute the
        # primitive writes through.
        class _P:
            def __init__(self, t):
                self.data = t

        self.W_q = _P(torch.zeros(dim_rows, dim_cols))
        self.W_k = _P(torch.zeros(dim_rows, dim_cols))
        self.W_v = _P(torch.zeros(dim_rows, dim_cols))
        # W_o is (D, n_heads*HD); use a square stub for simplicity.
        self.W_o = _P(torch.zeros(dim_cols, dim_rows))
        self.alibi_slopes = None


# ----------------------------------------------------------------------
# 1. Spec default: head_dim is None.
# ----------------------------------------------------------------------
def test_spec_head_dim_defaults_to_none():
    spec = DeclarativeAttentionHeadSpec(head_idx=0)
    assert spec.head_dim is None


def test_spec_effective_head_dim_default():
    spec = DeclarativeAttentionHeadSpec(head_idx=0)
    assert spec.effective_head_dim(16) == 16


def test_spec_effective_head_dim_override():
    spec = DeclarativeAttentionHeadSpec(head_idx=0, head_dim=8)
    assert spec.effective_head_dim(16) == 8


def test_spec_effective_head_dim_rejects_non_positive():
    spec = DeclarativeAttentionHeadSpec(head_idx=0, head_dim=0)
    with pytest.raises(ValueError):
        spec.effective_head_dim(16)


# ----------------------------------------------------------------------
# 2. Primitive lowering: legacy base when head_base is None.
# ----------------------------------------------------------------------
def test_primitive_legacy_base_matches_head_idx_times_HD():
    attn = _AttnStub(dim_rows=64, dim_cols=32)
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        q=(AP(slot=0, dim=5, weight=0.5),),
        k=(AP(slot=1, dim=6, weight=0.25),),
        v=(AP(slot=2, dim=7, weight=0.125),),
        o=(AO(out_dim=3, slot=3, weight=0.0625),),
    )
    Primitives.generate_attention_head(attn, spec, HD=16)
    assert float(attn.W_q.data[2 * 16 + 0, 5]) == 0.5
    assert float(attn.W_k.data[2 * 16 + 1, 6]) == 0.25
    assert float(attn.W_v.data[2 * 16 + 2, 7]) == 0.125
    assert float(attn.W_o.data[3, 2 * 16 + 3]) == 0.0625


def test_primitive_head_base_overrides_legacy_offset():
    attn = _AttnStub(dim_rows=64, dim_cols=32)
    spec = DeclarativeAttentionHeadSpec(
        head_idx=2,
        head_dim=8,
        q=(AP(slot=0, dim=5, weight=1.0),),
    )
    Primitives.generate_attention_head(attn, spec, HD=16, head_base=24)
    # Row 24 (the supplied base) is written, NOT 2*16=32 (the legacy base).
    assert float(attn.W_q.data[24, 5]) == 1.0
    assert float(attn.W_q.data[32, 5]) == 0.0


def test_primitive_rejects_slot_outside_effective_head_dim():
    attn = _AttnStub(dim_rows=64, dim_cols=32)
    spec = DeclarativeAttentionHeadSpec(
        head_idx=0,
        head_dim=4,
        q=(AP(slot=4, dim=0, weight=1.0),),  # slot 4 >= head_dim 4
    )
    with pytest.raises(ValueError, match="slot=4"):
        Primitives.generate_attention_head(attn, spec, HD=16)


# ----------------------------------------------------------------------
# 3. Primitive multi-head: default-only path is byte-identical with
#    pre-change behaviour; mixed path uses cumulative-sum bases.
# ----------------------------------------------------------------------
def test_primitive_multi_head_default_only_uses_legacy_bases():
    attn = _AttnStub(dim_rows=64, dim_cols=32)
    specs = [
        DeclarativeAttentionHeadSpec(
            head_idx=h, q=(AP(slot=0, dim=h, weight=float(h + 1)),)
        )
        for h in range(4)
    ]
    Primitives.generate_attention_heads(attn, specs, HD=16)
    for h in range(4):
        assert float(attn.W_q.data[h * 16, h]) == float(h + 1)


def test_primitive_multi_head_mixed_uses_cumulative_sum_bases():
    attn = _AttnStub(dim_rows=64, dim_cols=32)
    specs = [
        # head 0: default HD=16 => rows 0..15
        DeclarativeAttentionHeadSpec(
            head_idx=0, q=(AP(slot=0, dim=0, weight=10.0),)
        ),
        # head 1: head_dim=8 => rows 16..23
        DeclarativeAttentionHeadSpec(
            head_idx=1, head_dim=8, q=(AP(slot=0, dim=1, weight=20.0),)
        ),
        # head 2: default HD=16 => rows 24..39
        DeclarativeAttentionHeadSpec(
            head_idx=2, q=(AP(slot=0, dim=2, weight=30.0),)
        ),
    ]
    Primitives.generate_attention_heads(attn, specs, HD=16)
    assert float(attn.W_q.data[0, 0]) == 10.0
    assert float(attn.W_q.data[16, 1]) == 20.0
    assert float(attn.W_q.data[24, 2]) == 30.0
    # Nothing else got touched.
    assert float(attn.W_q.data[32, 2]) == 0.0  # legacy formula would land here


# ----------------------------------------------------------------------
# 4. Allocator: tracks per-head head_dim and exposes sums/bases.
# ----------------------------------------------------------------------
def test_allocator_default_head_dim_is_none():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    head_idx = alloc.alloc("op_a", 0, pin=0)
    assert head_idx == 0
    rec = alloc.heads()[0]
    assert rec.head_dim is None


def test_allocator_records_explicit_head_dim():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    alloc.alloc("op_a", 0, pin=0, head_dim=8)
    rec = alloc.heads()[0]
    assert rec.head_dim == 8


def test_allocator_rejects_non_positive_head_dim():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    with pytest.raises(AttentionHeadAllocatorError):
        alloc.alloc("op_a", 0, pin=0, head_dim=0)
    with pytest.raises(AttentionHeadAllocatorError):
        alloc.alloc("op_b", 0, pin=1, head_dim=-1)


def test_allocator_total_head_dim_default_only():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    for h in range(4):
        alloc.alloc(f"op_{h}", layer_idx=0, pin=h)
    # All None -> total = num_heads * default
    assert alloc.total_head_dim_at(0, default_head_dim=16) == 4 * 16


def test_allocator_total_head_dim_mixed():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    alloc.alloc("op_0", 0, pin=0)               # default 16
    alloc.alloc("op_1", 0, pin=1, head_dim=8)   # custom 8
    alloc.alloc("op_2", 0, pin=2, head_dim=24)  # custom 24
    assert alloc.total_head_dim_at(0, default_head_dim=16) == 16 + 8 + 24


def test_allocator_head_base_at_default_only_matches_legacy_formula():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    for h in range(4):
        alloc.alloc(f"op_{h}", 0, pin=h)
    for h in range(4):
        assert alloc.head_base_at(0, h, default_head_dim=16) == h * 16


def test_allocator_head_base_at_mixed_uses_cumulative_sum():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    alloc.alloc("op_0", 0, pin=0)               # 16 wide, base 0
    alloc.alloc("op_1", 0, pin=1, head_dim=8)   # 8 wide, base 16
    alloc.alloc("op_2", 0, pin=2, head_dim=24)  # 24 wide, base 24
    assert alloc.head_base_at(0, 0, default_head_dim=16) == 0
    assert alloc.head_base_at(0, 1, default_head_dim=16) == 16
    assert alloc.head_base_at(0, 2, default_head_dim=16) == 24


def test_allocator_head_dim_at_returns_explicit_value():
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    alloc.alloc("op_a", 0, pin=0, head_dim=8)
    alloc.alloc("op_b", 0, pin=1)  # default
    assert alloc.head_dim_at(0, 0, default=16) == 8
    assert alloc.head_dim_at(0, 1, default=16) == 16
    assert alloc.head_dim_at(0, 5, default=16) is None  # unallocated


# ----------------------------------------------------------------------
# 5. L8 integration smoke: half-width head bakes to the expected rows.
#    Per the task brief, this is the in-process integration check —
#    not gated against compile_full_vm_dynamic byte-identity at non-default
#    settings.
# ----------------------------------------------------------------------
def test_l8_half_width_head_integration():
    """Register a half-width head at L8 head 7 and assert its writes
    land in an 8-row block, not the legacy 16-row block.

    The layer's default HD here is 16 (matching today's typical sizing).
    Head 7 is declared with ``head_dim=8`` — half-width. The allocator
    reports the per-layer total as ``7 * 16 + 8 = 120`` rather than
    ``8 * 16 = 128``, and the lowered Q/K/V matrices have weights only
    in the head's 8-row block.
    """
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    # Fill heads 0-6 with default HD.
    for h in range(7):
        alloc.alloc(f"l8_default_head_{h}", layer_idx=8, pin=h)
    # Head 7: half-width.
    alloc.alloc("l8_half_width_head", layer_idx=8, pin=7, head_dim=8)

    default_HD = 16
    total = alloc.total_head_dim_at(8, default_head_dim=default_HD)
    assert total == 7 * default_HD + 8
    assert total == 120
    # NOT 128 — that's what a fixed-HD layer would report.
    assert total != 8 * default_HD

    # head 7's cumulative base is 7 * 16 == 112; its 8-row block ends at 120.
    base = alloc.head_base_at(8, 7, default_head_dim=default_HD)
    assert base == 112
    assert alloc.head_dim_at(8, 7, default=default_HD) == 8

    # Lower the half-width head and assert the writes land in [112, 120).
    # Size the stub one row larger so the [base+8] sentinel-row check
    # below stays in-bounds without aliasing into another head's footprint.
    attn = _AttnStub(dim_rows=total + 1, dim_cols=64)
    spec = DeclarativeAttentionHeadSpec(
        head_idx=7,
        head_dim=8,
        q=tuple(AP(slot=s, dim=s, weight=float(s + 1)) for s in range(8)),
        k=tuple(AP(slot=s, dim=s + 8, weight=float(s + 1)) for s in range(8)),
        v=tuple(AP(slot=s, dim=s + 16, weight=float(s + 1)) for s in range(8)),
        o=tuple(AO(out_dim=s, slot=s, weight=float(s + 1)) for s in range(8)),
    )
    Primitives.generate_attention_head(
        attn, spec, default_HD, head_base=base
    )
    # Every Q slot in [112, 120) is non-zero with the expected weight.
    for s in range(8):
        assert float(attn.W_q.data[base + s, s]) == float(s + 1)
        assert float(attn.W_k.data[base + s, s + 8]) == float(s + 1)
        assert float(attn.W_v.data[base + s, s + 16]) == float(s + 1)
        assert float(attn.W_o.data[s, base + s]) == float(s + 1)
    # Nothing landed at row base + 8 (would be inside head 7 if it were
    # default-width 16) — the eff_hd guard kept the bake within bounds.
    for col in range(64):
        assert float(attn.W_q.data[base + 8, col]) == 0.0


# ----------------------------------------------------------------------
# 6. NOTE FOR INTEGRATION: at-default (no spec sets head_dim) the public
# bake site falls through to ``head_idx * HD`` so ``compile_full_vm_dynamic()``
# stays byte-identical against the pre-change baseline. Per the brief we
# don't gate that here; the unit-level checks above are the contract.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# 7. Asymmetric head_dim consumer test: head_dim=[16, 16, 8, 16] at L8.
#    Head 2 is half-width. Verifies the cumulative-sum layout the bake
#    must use to keep Q/K/V/O slot row widths per-head.
# ----------------------------------------------------------------------
def test_l8_asymmetric_head_dim_16_16_8_16():
    """Register heads at L8 with head_dim=[16, 16, 8, 16] (head 2 is
    half-width). Verify total = 56 (not 4*16=64), per-head cumulative
    bases line up, and Q/K/V/O writes land in each head's own row
    block at its declared width.

    Expected layout (default HD=16):
        head 0 (HD=16) -> rows  [0, 16)
        head 1 (HD=16) -> rows [16, 32)
        head 2 (HD= 8) -> rows [32, 40)
        head 3 (HD=16) -> rows [40, 56)
    Total = 56, NOT 4 * 16 = 64.
    """
    alloc = AttentionHeadAllocator(layer_max_heads=8)
    head_dims = [16, 16, 8, 16]
    for h, hd in enumerate(head_dims):
        # head_dim=None for default-width heads keeps byte-identity with
        # the legacy bake; only head 2 declares an override.
        kw = {"head_dim": hd} if hd != 16 else {}
        alloc.alloc(f"l8_asym_head_{h}", layer_idx=8, pin=h, **kw)

    default_HD = 16

    # ---- allocator-level totals + bases ----
    total = alloc.total_head_dim_at(8, default_head_dim=default_HD)
    assert total == 16 + 16 + 8 + 16
    assert total == 56
    assert total != 4 * default_HD  # naive fixed-HD formula would give 64

    expected_bases = [0, 16, 32, 40]
    for h, expected in zip(range(4), expected_bases):
        assert alloc.head_base_at(8, h, default_head_dim=default_HD) == expected

    # ---- consumer-side lowering: each head writes only inside its own
    #      row block, using its own head_dim as the slot width.
    attn = _AttnStub(dim_rows=total + 1, dim_cols=80)
    specs = []
    for h, hd in enumerate(head_dims):
        eff = hd
        spec_kw = {"head_dim": hd} if hd != 16 else {}
        specs.append(
            DeclarativeAttentionHeadSpec(
                head_idx=h,
                q=tuple(
                    AP(slot=s, dim=h * 4 + (s % 4), weight=float(10 * h + s + 1))
                    for s in range(eff)
                ),
                k=tuple(
                    AP(slot=s, dim=20 + h * 4 + (s % 4), weight=float(100 + 10 * h + s + 1))
                    for s in range(eff)
                ),
                v=tuple(
                    AP(slot=s, dim=40 + h * 4 + (s % 4), weight=float(200 + 10 * h + s + 1))
                    for s in range(eff)
                ),
                o=tuple(
                    AO(out_dim=60 + h * 4 + (s % 4), slot=s, weight=float(300 + 10 * h + s + 1))
                    for s in range(eff)
                ),
                **spec_kw,
            )
        )
    Primitives.generate_attention_heads(attn, specs, default_HD)

    # ---- head-by-head check: writes land at the cumulative base ----
    for h, hd in enumerate(head_dims):
        base = expected_bases[h]
        eff = hd
        for s in range(eff):
            assert float(attn.W_q.data[base + s, h * 4 + (s % 4)]) == float(
                10 * h + s + 1
            )
            assert float(
                attn.W_k.data[base + s, 20 + h * 4 + (s % 4)]
            ) == float(100 + 10 * h + s + 1)
            assert float(
                attn.W_v.data[base + s, 40 + h * 4 + (s % 4)]
            ) == float(200 + 10 * h + s + 1)
            assert float(
                attn.W_o.data[60 + h * 4 + (s % 4), base + s]
            ) == float(300 + 10 * h + s + 1)

    # ---- bounds-respect: head 2 must not bleed past its 8-row block.
    # Head 2's Q dims: 8, 9, 10, 11 (h=2: h*4=8; slot%4 in {0..3}).
    # Rows [40, 48) belong to head 3 — those columns must be zero there.
    for col in (8, 9, 10, 11):
        for r in range(40, 48):
            assert float(attn.W_q.data[r, col]) == 0.0


def test_lower_attention_consumer_respects_dynamic_head_dim():
    """Consumer-path test: ``CompilerIR.lower_attention`` must route
    through the cumulative-sum lowering path when any head declares a
    non-default ``head_dim``. Before the fix it called
    ``Primitives.generate_attention_head`` per head with no ``head_base``,
    so the half-width middle head landed at the legacy ``head_idx * HD``
    row, overlapping the next head's block.
    """
    from c4_release.neural_vm.unified_compiler.ir import (
        CompilerIR,
        LayerSpec,
    )

    head_dims = [16, 16, 8, 16]
    default_HD = 16
    specs = [
        DeclarativeAttentionHeadSpec(
            head_idx=h,
            head_dim=hd if hd != 16 else None,
            q=(AP(slot=0, dim=h, weight=float(h + 1)),),
            k=(AP(slot=0, dim=10 + h, weight=float(100 + h + 1)),),
            v=(AP(slot=0, dim=20 + h, weight=float(200 + h + 1)),),
            o=(AO(out_dim=30 + h, slot=0, weight=float(300 + h + 1)),),
        )
        for h, hd in enumerate(head_dims)
    ]
    layer = LayerSpec()
    for s in specs:
        layer.attention.add_head(s)
    ir = CompilerIR(layers=[layer])
    attn = _AttnStub(dim_rows=64, dim_cols=64)
    ir.lower_attention(attn, default_HD, layer_idx=0)

    # Expected cumulative bases for [16, 16, 8, 16].
    expected_bases = [0, 16, 32, 40]
    for h in range(4):
        base = expected_bases[h]
        assert float(attn.W_q.data[base, h]) == float(h + 1)
        assert float(attn.W_k.data[base, 10 + h]) == float(100 + h + 1)
        assert float(attn.W_v.data[base, 20 + h]) == float(200 + h + 1)
        assert float(attn.W_o.data[30 + h, base]) == float(300 + h + 1)

    # Pre-fix behaviour would have written head 3 at row 3*16=48 (legacy).
    # Confirm the new cumulative-sum base 40 was used instead.
    assert float(attn.W_q.data[48, 3]) == 0.0
