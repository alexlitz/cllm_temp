"""Per-op tests for the B7-5 SP_GATHERED_THIS_STEP sentinel.

The sentinel lives on a single SwiGLU unit at the tail of L8's FFN. It
should:

  1. Register at L8, kind=block, phase=8.6, migrated=True with the
     expected reads/writes/produces contract.
  2. Allocate slot 98 in the legacy ``_SetDim`` layout (B6-K reclaim of
     dead L0 H5 sub-slot) and grow L8's PureFFN by exactly one hidden
     unit beyond ``layer8_multibyte_routing``'s 2055.
  3. Bake a SwiGLU unit at unit index 2055 with conditions=(MARK_SP,) and
     W_down writing to SP_GATHERED_THIS_STEP at amplitude ``2.0 / S``.
  4. Output 1.0 at MARK_SP positions and 0.0 elsewhere after one L8 FFN
     forward pass on a synthetic residual stream.
  5. Survive ``compile_full_vm_dynamic()`` with zero staleness warnings (the
     ``produces={"SP_GATHERED_THIS_STEP": "SP_marker"}`` annotation is the
     in-step producer for any future L10/L13 consumer's
     ``consumes_fresh`` claim).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.unified_compiler.ops.all_core_ops import all_core_ops  # noqa: E402
from neural_vm.unified_compiler.ops.l8_ops import (  # noqa: E402
    _L8_SP_GATHERED_SENTINEL_UNIT,
    _layer8_sp_gathered_sentinel_rule,
    make_layer8_sp_gathered_sentinel_op,
)
from neural_vm.vm_step import _SetDim  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Dim allocation
# ---------------------------------------------------------------------------


def test_setdim_allocates_sp_gathered_this_step_at_slot_98():
    """Sentinel dim lives at slot 98 in the legacy ``_SetDim`` layout."""
    assert hasattr(_SetDim, "SP_GATHERED_THIS_STEP")
    assert _SetDim.SP_GATHERED_THIS_STEP == 98


def test_sp_gathered_this_step_does_not_collide_with_live_dims():
    """Slot 98 falls inside the dead H5 range (95-101) per BD usage map."""
    # H5 occupies slots 95..101; H5+3 (slot 98) corresponds to L0 head 5's
    # MARK_BP marker output, which is written by ``layer0_threshold_attn``
    # but never read downstream. Aliasing SP_GATHERED_THIS_STEP onto that
    # slot is safe under the current consumer set.
    assert _SetDim.H5 == 95
    assert _SetDim.SP_GATHERED_THIS_STEP == _SetDim.H5 + 3
    # The aliased H5 sub-slot has no downstream reader: it is overwritten
    # by the L8 FFN sentinel before any consumer (L9+) inspects the dim.


# ---------------------------------------------------------------------------
# 2. Op registration
# ---------------------------------------------------------------------------


def test_layer8_sp_gathered_sentinel_op_is_registered():
    """The op is present in ``all_core_ops`` with the expected shape."""
    ops = all_core_ops()
    found = [op for op in ops if op.name == "layer8_sp_gathered_sentinel"]
    assert len(found) == 1, (
        f"Expected exactly 1 layer8_sp_gathered_sentinel op, "
        f"found {len(found)}"
    )
    op = found[0]
    assert op.layer_idx == 8
    assert op.kind == "block"
    assert op.migrated is True
    assert op.phase == pytest.approx(8.6)
    assert op.declarative_authority == "spec_generated"


def test_layer8_sp_gathered_sentinel_reads_writes_contract():
    """The op reads MARK_SP and writes SP_GATHERED_THIS_STEP only."""
    op = next(
        o for o in all_core_ops() if o.name == "layer8_sp_gathered_sentinel"
    )
    assert op.reads == {"MARK_SP"}
    assert op.writes == {"SP_GATHERED_THIS_STEP"}
    # Step 5 of IR_INCREMENTAL_IMPROVEMENTS.md: ``produces`` is a derived
    # @property — slot string is the constant ``<derived>``. Assert by
    # key set (the dim names are the load-bearing contract; the slot
    # string used to be the semantic ``"SP_marker"`` but is no longer
    # author-controlled).
    assert set(op.produces.keys()) == {"SP_GATHERED_THIS_STEP"}


def test_layer8_sp_gathered_sentinel_grows_l8_ffn_by_one_unit():
    """``ffn_units_used`` = ``multibyte_routing`` (2055) + 1."""
    op = next(
        o for o in all_core_ops() if o.name == "layer8_sp_gathered_sentinel"
    )
    assert op.ffn_units_used == _L8_SP_GATHERED_SENTINEL_UNIT + 1
    assert _L8_SP_GATHERED_SENTINEL_UNIT == 2055


def test_layer8_sp_gathered_sentinel_claims_single_w_down_cell():
    """The op claims exactly one ffn_W_down cell at the sentinel unit."""
    op = next(
        o for o in all_core_ops() if o.name == "layer8_sp_gathered_sentinel"
    )
    expected = {
        (8, "ffn_W_down", str(_L8_SP_GATHERED_SENTINEL_UNIT),
         "SP_GATHERED_THIS_STEP+0"),
    }
    assert op.claims == expected


def test_layer8_sp_gathered_sentinel_phase_runs_after_other_l8_ops():
    """Phase 8.6 is strictly greater than every other L8 op's phase."""
    ops = all_core_ops()
    sentinel = next(
        o for o in ops if o.name == "layer8_sp_gathered_sentinel"
    )
    sibling_l8_phases = [
        op.phase for op in ops
        if op.layer_idx == 8 and op.name != "layer8_sp_gathered_sentinel"
        and op.phase is not None
    ]
    assert sibling_l8_phases, "Expected sibling L8 ops"
    for p in sibling_l8_phases:
        assert sentinel.phase > p, (
            f"Sentinel phase {sentinel.phase} must run after sibling "
            f"phase {p}"
        )


# ---------------------------------------------------------------------------
# 3. Rule structure
# ---------------------------------------------------------------------------


def test_sentinel_rule_writes_w_down_at_2_over_s():
    """The SwiGLU rule's W_down magnitude is ``2.0 / S`` so output → 1.0."""
    S = 100.0
    rule = _layer8_sp_gathered_sentinel_rule(S)
    # Conditions: single MARK_SP at weight 1.0.
    assert len(rule.conditions) == 1
    cond = rule.conditions[0]
    assert cond.dim.name == "MARK_SP"
    assert cond.weight == pytest.approx(1.0)
    # Threshold: 0.5 (b_up = -S/2), so silu(S/2) wins only at MARK_SP=1.
    assert rule.threshold == pytest.approx(0.5)
    # Constant gate (bias = 1.0, no W_gate term).
    assert rule.gate is None
    assert rule.gate_bias == pytest.approx(1.0)
    # Single write to SP_GATHERED_THIS_STEP at 2.0/S.
    assert len(rule.writes) == 1
    w = rule.writes[0]
    assert w.dim.name == "SP_GATHERED_THIS_STEP"
    assert w.weight == pytest.approx(2.0 / S)


# ---------------------------------------------------------------------------
# 4. Symbolic forward (the FFN unit fires exactly at MARK_SP positions)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baked_full_model():
    """One compiled full-VM model + layout shared by symbolic-forward tests."""
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    model, layout = compile_full_vm_dynamic(disk_cache=False, S=100.0)
    return model, layout


def test_l8_ffn_emits_one_at_mark_sp_zero_elsewhere(baked_full_model):
    """L8 FFN forward writes SP_GATHERED_THIS_STEP = 1.0 only at MARK_SP."""
    model, layout = baked_full_model
    ssp = layout.dim_positions["SP_GATHERED_THIS_STEP"]
    mark_sp = layout.dim_positions["MARK_SP"]
    const_dim = layout.dim_positions["CONST"]

    l8_ffn = model.blocks[8].ffn

    # 35-token step: MARK_SP at position 2 (after PC+AX). All other token
    # positions have CONST=1.0 set (matches embedding for non-marker
    # tokens). MARK_SP is exclusive to the SP marker token.
    seq_len = 35
    x = torch.zeros(1, seq_len, layout.d_model)
    x[..., const_dim] = 1.0
    x[0, 2, mark_sp] = 1.0  # MARK_SP at SP marker position

    with torch.no_grad():
        y = l8_ffn(x)

    # SP_GATHERED_THIS_STEP at MARK_SP position should be ~1.0.
    assert y[0, 2, ssp].item() == pytest.approx(1.0, abs=1e-3), (
        f"Expected SP_GATHERED_THIS_STEP ≈ 1.0 at MARK_SP position, "
        f"got {y[0, 2, ssp].item()}"
    )
    # At every other position the sentinel must remain 0 (well below 0.5
    # so downstream consumers can use a +50 lifecycle gate per B6-K Sec 5).
    for pos in range(seq_len):
        if pos == 2:
            continue
        assert y[0, pos, ssp].item() == pytest.approx(0.0, abs=1e-3), (
            f"Expected SP_GATHERED_THIS_STEP ≈ 0.0 at non-MARK_SP "
            f"position {pos}, got {y[0, pos, ssp].item()}"
        )


def test_l8_ffn_sentinel_unit_writes_only_to_sp_gathered_dim(baked_full_model):
    """The sentinel hidden unit's W_down is non-zero ONLY at SP_GATHERED."""
    model, layout = baked_full_model
    ssp = layout.dim_positions["SP_GATHERED_THIS_STEP"]
    l8_ffn = model.blocks[8].ffn

    # The sentinel lives at unit index 2055 in L8's hidden dim.
    unit = _L8_SP_GATHERED_SENTINEL_UNIT
    assert l8_ffn.W_up.shape[0] > unit, (
        f"L8 FFN hidden_dim {l8_ffn.W_up.shape[0]} must cover sentinel "
        f"unit {unit}"
    )
    w_down_col = l8_ffn.W_down.data[:, unit]
    nonzero_rows = w_down_col.nonzero(as_tuple=False).flatten().tolist()
    assert nonzero_rows == [ssp], (
        f"Expected sentinel unit W_down nonzero only at "
        f"SP_GATHERED_THIS_STEP ({ssp}); got rows {nonzero_rows}"
    )


# ---------------------------------------------------------------------------
# 5. compile_full_vm_dynamic() succeeds with no STALENESS warnings
# ---------------------------------------------------------------------------


def test_compile_full_vm_emits_no_staleness_warnings_with_sentinel(caplog):
    """The full-VM compile path runs clean once the sentinel is wired."""
    import logging

    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

    caplog.set_level(logging.WARNING, logger="neural_vm")
    model, _layout = compile_full_vm_dynamic(disk_cache=False, S=100.0)
    # No "STALENESS" warning text in any captured record.
    for record in caplog.records:
        msg = record.getMessage()
        assert "STALENESS" not in msg, (
            f"Unexpected STALENESS warning: {msg}"
        )
    # Model object built successfully.
    assert model is not None
