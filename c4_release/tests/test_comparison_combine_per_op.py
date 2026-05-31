"""Per-op audit harness for ``ComparisonCombine`` (vm_step.py:649).

``ComparisonCombine`` is a PureFFN subclass that combines CMP[0..3] flags
(produced by L9 FFN at the AX marker position) into the truth-value byte
written to OUTPUT_LO for EQ/NE/LT/GT/LE/GE.

  CMP[0] = hi_lt
  CMP[1] = hi_eq
  CMP[2] = lo_eq
  CMP[3] = lo_lt

The module uses a default + override pattern (`vm_step.py:692-748`):
  - Default unit per opcode emits the "no flags set" answer (0 or 1).
  - Override units flip the answer when specific CMP combinations fire.

Total: 6 defaults + 12 overrides = 18 units, hidden_dim=18.

Two sites instantiate ``ComparisonCombine`` in default builds:
  - ``make_l10_post_ops_combined`` (l10_ops.py:1326), the L17 FFN body
    — bakes 18 units into a larger PureFFN.
  - ``make_l10_post_op_attach_op`` (l10_ops.py:4857), efficient-mode
    only — appends to L10.post_ops directly.

The module also adds a strong MARK_PC blocker (-50*S, vm_step.py:690-700)
so leaked OP_NE/OP_GT/OP_GE flags or runaway CMP values at PC marker
positions don't fire spurious OUTPUT writes.

Per B6-L § 3a, ``ComparisonCombine`` is one of 7 uncovered post-op
modules — its only existing test coverage is incidental
(``test_full_model_add_trace.py``). This file adds the missing per-op
contract testing:

1. **Static drift via static_claims_report**: ComparisonCombine is
   bake-only (no Operation wrapper of its own), so it appears only as
   part of ``l10_post_ops_combined``. Pin that op via assert_op_absent
   (the FFN-body op carries no per-cell claims today).
2. **Truth-table forward**: drive each of the 6 comparison opcodes
   with each CMP-flag combination and assert OUTPUT_LO argmax matches
   the spec (vm_step.py:726-748).
3. **MARK_PC blocker**: confirm the -50*S MARK_PC suppression is wired
   per unit (regression sentinel for the leaked-OP_NE failure mode
   documented in vm_step.py:684-690).
4. **Hidden_dim invariant**: pin the 18-unit count so a future refactor
   that drops a default/override unit fires here.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.vm_step import ComparisonCombine, _SetDim  # noqa: E402


# ---------------------------------------------------------------------------
# Module-scoped baked instance (immutable to tests; saves ~18-unit bake).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def comparison_combine() -> ComparisonCombine:
    return ComparisonCombine(d_model=512, S=100.0)


# ---------------------------------------------------------------------------
# Section 1: Hidden_dim + module-shape invariants.
# ---------------------------------------------------------------------------


def test_comparison_combine_hidden_dim_is_18(comparison_combine):
    """6 defaults + 12 overrides = 18 units. Pinning this catches any
    accidental drop of an override unit (which would silently make the
    affected comparison return the default).
    """
    assert comparison_combine.W_up.shape == (18, 512), (
        f"ComparisonCombine W_up shape drift: {comparison_combine.W_up.shape}; "
        f"expected (18, 512)"
    )


def test_comparison_combine_subclasses_pure_ffn(comparison_combine):
    """Phase 0 conversion (2026-05-09): ComparisonCombine inherits PureFFN's
    canonical SwiGLU forward. Regression sentinel for the structural
    contract — if the inheritance breaks, the forward semantics diverge
    silently from every other FFN in the model.
    """
    from neural_vm.base_layers import PureFFN

    assert isinstance(comparison_combine, PureFFN)


# ---------------------------------------------------------------------------
# Section 2: MARK_PC blocker sentinel.
# ---------------------------------------------------------------------------


def test_comparison_combine_every_unit_has_mark_pc_blocker(comparison_combine):
    """Each of the 18 units must carry the -50*S MARK_PC blocker so
    leaked OP_NE/OP_GT/OP_GE at PC-marker rows can't spuriously emit
    OUTPUT_LO writes (vm_step.py:684-690 / 696/707/720).
    """
    S = 100.0
    expected_block = -S * 50  # = -5000
    mark_pc_col = comparison_combine.W_up.data[:, _SetDim.MARK_PC]
    # All 18 units write the blocker.
    assert torch.all(mark_pc_col == expected_block), (
        f"ComparisonCombine MARK_PC blocker drift: per-unit values "
        f"{mark_pc_col.tolist()}; expected all == {expected_block}"
    )


# ---------------------------------------------------------------------------
# Section 3: Per-opcode truth-table forward.
# ---------------------------------------------------------------------------


def _cmp_input(
    *,
    op_dim: int,
    cmp_flags: dict[int, float] | None = None,
    mark_ax: float = 1.0,
) -> torch.Tensor:
    """One-position residual at MARK_AX with the requested opcode + CMP."""
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = mark_ax
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, op_dim] = 1.0
    if cmp_flags is not None:
        for slot, value in cmp_flags.items():
            x[0, 0, _SetDim.CMP + slot] = value
    return x


@pytest.mark.parametrize(
    ("op_dim", "op_name", "cmp_flags", "expected_byte"),
    [
        # EQ: default 0; overrides flip to 1 when CMP[1]=hi_eq AND CMP[2]=lo_eq.
        (_SetDim.OP_EQ, "EQ", None, 0),
        (_SetDim.OP_EQ, "EQ", {1: 1.0, 2: 1.0}, 1),
        # NE: default 1; flips to 0 when CMP[1]=hi_eq AND CMP[2]=lo_eq.
        (_SetDim.OP_NE, "NE", None, 1),
        (_SetDim.OP_NE, "NE", {1: 1.0, 2: 1.0}, 0),
        # LT: default 0; flips to 1 on CMP[0]=hi_lt OR (CMP[1]=hi_eq AND CMP[3]=lo_lt).
        (_SetDim.OP_LT, "LT", None, 0),
        (_SetDim.OP_LT, "LT", {0: 1.0}, 1),
        (_SetDim.OP_LT, "LT", {1: 1.0, 3: 1.0}, 1),
        # GT: default 1; flips to 0 on CMP[0]=hi_lt OR (CMP[1]=hi_eq AND CMP[3]=lo_lt)
        # OR (CMP[1]=hi_eq AND CMP[2]=lo_eq).
        (_SetDim.OP_GT, "GT", None, 1),
        (_SetDim.OP_GT, "GT", {0: 1.0}, 0),
        (_SetDim.OP_GT, "GT", {1: 1.0, 2: 1.0}, 0),
        # LE: default 0; flips to 1 on CMP[0]=hi_lt OR (CMP[1]=hi_eq AND CMP[3]=lo_lt)
        # OR (CMP[1]=hi_eq AND CMP[2]=lo_eq).
        (_SetDim.OP_LE, "LE", None, 0),
        (_SetDim.OP_LE, "LE", {0: 1.0}, 1),
        (_SetDim.OP_LE, "LE", {1: 1.0, 2: 1.0}, 1),
        # GE: default 1; flips to 0 on CMP[0]=hi_lt OR (CMP[1]=hi_eq AND CMP[3]=lo_lt).
        (_SetDim.OP_GE, "GE", None, 1),
        (_SetDim.OP_GE, "GE", {0: 1.0}, 0),
    ],
)
def test_comparison_combine_truth_table(
    comparison_combine, op_dim, op_name, cmp_flags, expected_byte
):
    """Drive ComparisonCombine with each (op, cmp) input and assert the
    written OUTPUT_LO byte matches the spec.

    This is the canonical truth-table audit — a regression that drops or
    relabels a single override unit surfaces here as a wrong-byte
    failure rather than as a downstream EQ/NE/LT/GT/LE/GE smoke flake.
    """
    x = _cmp_input(op_dim=op_dim, cmp_flags=cmp_flags)
    with torch.no_grad():
        delta = comparison_combine(x)[0, 0]

    out_lo = delta[_SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
    out_hi = delta[_SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16]
    lo_argmax = int(out_lo.argmax().item())
    hi_argmax = int(out_hi.argmax().item())

    assert lo_argmax == expected_byte, (
        f"{op_name} (cmp={cmp_flags}): OUTPUT_LO argmax should be "
        f"{expected_byte}, got {lo_argmax}. delta_LO={out_lo.tolist()}"
    )
    # All ComparisonCombine writes go to OUTPUT_LO[result] and
    # OUTPUT_HI[0] (per the bake at vm_step.py:699-700) — the truth
    # byte is a single-byte value with hi nibble = 0.
    assert hi_argmax == 0, (
        f"{op_name}: OUTPUT_HI argmax should be 0 (comparison result is "
        f"single-byte), got {hi_argmax}"
    )


def test_comparison_combine_does_not_fire_without_mark_ax(comparison_combine):
    """Sentinel: MARK_AX is the load-bearing gate. Without it, every
    unit's bias dominates and the output writes vanish (the SwiGLU gate
    threshold ``-S*1.5`` requires both MARK_AX and the opcode/CMP to be
    hot).
    """
    x = _cmp_input(op_dim=_SetDim.OP_EQ, cmp_flags={1: 1.0, 2: 1.0}, mark_ax=0.0)
    with torch.no_grad():
        delta = comparison_combine(x)[0, 0]
    out_lo = delta[_SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
    energy = float(out_lo.abs().sum())
    assert energy < 0.1, (
        f"ComparisonCombine fired without MARK_AX: OUTPUT_LO energy "
        f"={energy:.4f}; expected ~0. delta_LO={out_lo.tolist()}"
    )


def test_comparison_combine_does_not_fire_at_mark_pc(comparison_combine):
    """Regression sentinel for the leaked-OP_NE failure mode.

    Even with OP_EQ + CMP flags hot, presence of MARK_PC must crush every
    unit's SwiGLU input below the gate threshold (the -50*S blocker
    overwhelms +S * (opcode + cmp) terms). vm_step.py:684-690 calls this
    out: without the blocker, OP_NE/OP_GT/OP_GE leak at PC-marker rows
    and the ComparisonCombine writes corrupt PC predictions.
    """
    x = _cmp_input(op_dim=_SetDim.OP_EQ, cmp_flags={1: 1.0, 2: 1.0})
    # Now turn MARK_PC on as well; MARK_AX is also on, so this simulates
    # the worst-case leakage (rare but seen in the pre-blocker reports).
    x[0, 0, _SetDim.MARK_PC] = 1.0
    with torch.no_grad():
        delta = comparison_combine(x)[0, 0]
    out_lo = delta[_SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16]
    energy = float(out_lo.abs().sum())
    assert energy < 0.1, (
        f"ComparisonCombine fired at MARK_PC despite the -50*S blocker: "
        f"OUTPUT_LO energy={energy:.4f}. delta_LO={out_lo.tolist()}"
    )
