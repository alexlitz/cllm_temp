"""Softmax DSL variant — scaffold tests (2026-06-09).

Pins the compile-path plumbing for the parallel ``softmax_variant``
flag on ``compile_full_vm_dynamic``. See
``docs/SOFTMAX_DSL_VARIANT_DESIGN_2026_06_09.md`` for the full design.

Today only the kwarg surface is wired (validation + cache-key
plumbing); the bake-side sink-injection lowering is a follow-up wave.
The byte-identity contract between ``softmax_variant="softmax1"`` and
``softmax_variant="standard"`` on an unmapped LI is marked ``xfail``
until the lowering lands.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.unified_compiler.primitives import (  # noqa: E402
    AP,
    AO,
    DeclarativeAttentionHeadSpec,
    _inject_sink_k_row,
)


# ---------------------------------------------------------------------------
# _inject_sink_k_row helper
# ---------------------------------------------------------------------------


def _toy_spec() -> DeclarativeAttentionHeadSpec:
    """Tiny head spec covering the helper's pass-through invariants."""

    return DeclarativeAttentionHeadSpec(
        head_idx=3,
        q=(AP(slot=0, dim=10, weight=15.0),),
        k=(AP(slot=0, dim=20, weight=15.0),),
        v=(AP(slot=0, dim=30, weight=1.0),),
        o=(AO(out_dim=40, slot=0, weight=1.0),),
        alibi_slope=0.5,
        group_size=1,
        head_dim=None,
    )


def test_inject_sink_appends_one_k_slot():
    """Helper adds exactly one extra K-side write at ``sink_idx``."""

    spec = _toy_spec()
    sunk = _inject_sink_k_row(spec, sink_idx=7)

    # Original K row preserved.
    assert spec.k[0] in sunk.k
    # Exactly one new K row was added.
    assert len(sunk.k) == len(spec.k) + 1
    new_writes = [w for w in sunk.k if w not in spec.k]
    assert len(new_writes) == 1
    sink_write = new_writes[0]
    # The sink K write must score 0 against any Q (weight=0 ensures
    # K_sink row = 0 regardless of input).
    assert sink_write.slot == 7
    assert sink_write.weight == 0.0


def test_inject_sink_preserves_qvo_and_metadata():
    """All non-K fields pass through byte-identical."""

    spec = _toy_spec()
    sunk = _inject_sink_k_row(spec, sink_idx=7)

    assert sunk.head_idx == spec.head_idx
    assert sunk.q == spec.q
    assert sunk.v == spec.v
    assert sunk.o == spec.o
    assert sunk.alibi_slope == spec.alibi_slope
    assert sunk.group_size == spec.group_size
    assert sunk.head_dim == spec.head_dim


def test_inject_sink_is_idempotent():
    """Calling twice with the same sink_idx is a no-op."""

    spec = _toy_spec()
    once = _inject_sink_k_row(spec, sink_idx=7)
    twice = _inject_sink_k_row(once, sink_idx=7)
    assert twice.k == once.k


# ---------------------------------------------------------------------------
# compile_full_vm_dynamic softmax_variant kwarg plumbing
# ---------------------------------------------------------------------------


def test_softmax_variant_rejects_unknown_value():
    """Unknown ``softmax_variant`` strings raise ``ValueError``."""

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with pytest.raises(ValueError, match="softmax_variant must be"):
        compile_full_vm_dynamic(softmax_variant="bogus")


def test_softmax_variant_conflicts_with_explicit_attention_normalization():
    """Mixing ``softmax_variant`` and ``attention_normalization`` is rejected."""

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )

    with pytest.raises(TypeError, match="implies attention_normalization"):
        compile_full_vm_dynamic(
            softmax_variant="standard",
            attention_normalization="softmax1",
        )


# ---------------------------------------------------------------------------
# Byte-identity contract (xfail until lowering lands)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Sink-injection lowering not yet emitted by compile_full_vm_dynamic; "
        "see docs/SOFTMAX_DSL_VARIANT_DESIGN_2026_06_09.md §6 follow-up wave."
    ),
    strict=False,
)
def test_softmax_variant_zfod_byte_identity_on_unmapped_li():
    """One-step forward: softmax1 vs standard+sink agree on an unmapped LI.

    Contract (see design doc §5):
      * Compile two VMs identical except ``softmax_variant``.
      * Seed identical state with ``MEM_addr0`` pointing at an unmapped
        address (guaranteed LI miss).
      * Step both VMs one tick.
      * Assert ``OUTPUT_LO``/``OUTPUT_HI`` bytes match.

    Today this is ``xfail`` — the standard variant's bake still misses
    the sink K row, so the unmapped LI propagates a non-zero V into
    the residual under standard softmax. The lowering wave will flip
    this to a hard byte-identity gate.
    """

    pytest.skip("lowering wave pending; see design doc §6 wave A")
