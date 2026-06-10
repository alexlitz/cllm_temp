"""Post-Wave-B STEP_END compute isolation tests.

Mirrors the L9 collapsed-IMM isolation pattern
(``tests/test_l9_collapsed_imm_input_isolated.py``) and the L15
memory-lookup isolation pattern
(``tests/test_l15_memory_lookup_isolated.py``) but targets the STEP_END
row instead of the MARK_AX row.

Background — STEP_END compute migration
---------------------------------------
See ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md``. The current
convention is that ALU / CMP / dispatch compute fires at the MARK_AX
row (row 5 of the 35-token VM-step window) where every byte is still
being relayed. Wave B will migrate each scheduler-bound rule from
``("MARK_AX", w)`` to ``("MARK_SE_ONLY", w)`` after Wave A lands a
``step_end_operand_relay`` head that broadcasts the operand slots from
MARK_AX to MARK_SE within the same step.

Each test here:

  1. Lowers the rule's current FFN via ``Primitives.lower_ffn_rules``.
  2. Synthesizes a residual at the **STEP_END row** with ``MARK_SE_ONLY=1``
     (NOT ``MARK_AX``) and all the operand slots Wave A is supposed to
     have relayed: ``OP_<NAME>``, ``CMP``, ``ALU_LO/HI``,
     ``AX_CARRY_LO/HI``.
  3. Forwards through the FFN.
  4. Asserts the target dim at STEP_END carries the expected value.

Today these tests **FAIL** because the rules still gate on MARK_AX and
the residual is at MARK_SE — so each rule's main MARK_AX condition is
zero and the threshold AND fails. They are marked
``@pytest.mark.xfail(strict=False, reason="pending Wave B migration")``.
After Wave B migrations land (per ``STEP_END_COMPUTE_ARCHITECTURE``
table §4 Wave B), the gates flip to ``MARK_SE_ONLY``, the rules fire
at STEP_END, and the tests become XPASS.

Test set (15 tests, one per Wave B rule + a few sub-variants)
-------------------------------------------------------------
Wave B row #1 — ``_layer10_alu_cmp_combine_rules``:
  test_l10_alu_cmp_combine_eq_override_at_step_end
  test_l10_alu_cmp_combine_ne_override_at_step_end
  test_l10_alu_cmp_combine_lt_override_at_step_end

Wave B row #2 — ``_l10_comparison_combine_rules``:
  test_l10_comparison_combine_eq_default_at_step_end
  test_l10_comparison_combine_ge_default_at_step_end

Wave B row #3 — ``_layer10_alu_bitwise_or_rules``:
  test_l10_alu_bitwise_or_at_step_end

Wave B row #4 — ``_layer10_alu_bitwise_xor_rules``:
  test_l10_alu_bitwise_xor_at_step_end

Wave B row #5 — ``_layer10_alu_bitwise_and_rules``:
  test_l10_alu_bitwise_and_at_step_end

Wave B row #6 — ``_layer10_alu_shl_shr_zero_rules``:
  test_l10_alu_shl_zero_at_step_end
  test_l10_alu_shr_zero_at_step_end

Wave B row #7 — ``_layer9_cmp_rules`` (hi_eq, lo_eq, lo_lt):
  test_l9_cmp_hi_eq_at_step_end
  test_l9_cmp_lo_eq_at_step_end
  test_l9_cmp_lo_lt_at_step_end

Wave B row #8 — ``_layer9_add_hi_nibble_rules``:
  test_l9_add_hi_nibble_at_step_end

Wave B row #9 — ``_layer9_sub_hi_nibble_rules``:
  test_l9_sub_hi_nibble_at_step_end

Whole file targets <2s runtime; module-scope fixture caches the nine
lowered FFNs.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Force CPU-only execution to avoid OOM with parallel agents.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.unified_compiler.ops.l9_ops import (  # noqa: E402
    _layer9_add_hi_nibble_rules,
    _layer9_cmp_rules,
    _layer9_sub_hi_nibble_rules,
)
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _l10_comparison_combine_rules,
    _layer10_alu_bitwise_and_rules,
    _layer10_alu_bitwise_or_rules,
    _layer10_alu_bitwise_xor_rules,
    _layer10_alu_cmp_combine_rules,
    _layer10_alu_shl_shr_zero_rules,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim as BD  # noqa: E402


S = 100.0
D_MODEL = 512


# ----------------------------------------------------------------------
# FFN construction helpers (mirrors ``test_l9_collapsed_imm_input_isolated``).
# ----------------------------------------------------------------------


def _build_ffn(rules) -> PureFFN:
    """Lower ``rules`` into a fresh ``PureFFN``."""
    n_units = max(len(rules), 1)
    ffn = PureFFN(dim=D_MODEL, hidden_dim=n_units)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(BD, names)
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == len(rules), (
        f"lower_ffn_rules wrote {end} units, expected {len(rules)}"
    )
    return ffn


@pytest.fixture(scope="module")
def ffns() -> dict:
    """Module-scoped FFN cache: nine FFNs are baked once per session."""
    return {
        "l9_add_hi": _build_ffn(_layer9_add_hi_nibble_rules(S)),
        "l9_sub_hi": _build_ffn(_layer9_sub_hi_nibble_rules(S)),
        "l9_cmp": _build_ffn(_layer9_cmp_rules(S)),
        "l10_or": _build_ffn(_layer10_alu_bitwise_or_rules(S)),
        "l10_and": _build_ffn(_layer10_alu_bitwise_and_rules(S)),
        "l10_xor": _build_ffn(_layer10_alu_bitwise_xor_rules(S)),
        "l10_shl_shr": _build_ffn(_layer10_alu_shl_shr_zero_rules(S)),
        "l10_alu_cmp_combine": _build_ffn(_layer10_alu_cmp_combine_rules(S)),
        "l10_comparison_combine": _build_ffn(_l10_comparison_combine_rules(S)),
    }


# ----------------------------------------------------------------------
# STEP_END residual synthesis.
# ----------------------------------------------------------------------


def _make_step_end_resid(
    *,
    opcode_dim=None,
    a: int = 5,
    b: int = 5,
    cmp_group: bool = False,
    cmp_flags: tuple[int, ...] = (),
    shift_count=None,
    set_alu: bool = True,
    set_carry: bool = True,
) -> torch.Tensor:
    """Synthesize the residual at the **STEP_END row** of a VM step.

    Encodes what Wave A's ``step_end_operand_relay`` head is supposed to
    deliver to the SE row: every operand slot that currently lives at
    MARK_AX is broadcast forward so the same compute can re-fire at SE.

    Args:
        opcode_dim: ``BD.OP_*`` dim to set (relayed from MARK_AX).
        a, b: byte values for operand A (AX byte 0 → ``ALU_LO/HI``) and
            operand B (TOS byte 0 → ``AX_CARRY_LO/HI``).
        cmp_group: set ``CMP_GROUP = 1`` (required gate for L9 CMP rules).
        cmp_flags: which ``CMP+i`` cells to set to 1.0 (the cascade
            inputs that the L10 cmp_combine overrides consume).
        shift_count: if set, encode the shift count in ``AX_CARRY_LO/HI``
            (overrides ``b`` for the SHL/SHR shortcut tests).
        set_alu: if False, omit the ALU_LO/HI operand-A encoding (useful
            for cmp_combine tests that only need CMP cells, not the raw
            operand).
        set_carry: if False, omit the AX_CARRY_LO/HI operand-B encoding.

    Note:
        ``MARK_AX`` is **deliberately zero**. This is what Wave A's
        relay produces: the SE row carries all operand slots without
        the MARK_AX identity. After Wave B migration the rules gate on
        ``MARK_SE_ONLY`` (which we set here), not ``MARK_AX``.
    """
    x = torch.zeros(1, 1, D_MODEL)
    # STEP_END row identity (post-Wave-A relayed; MARK_AX is deliberately 0).
    x[0, 0, BD.MARK_SE] = 1.0
    x[0, 0, BD.MARK_SE_ONLY] = 1.0
    x[0, 0, BD.HAS_SE] = 1.0
    x[0, 0, BD.CONST] = 1.0
    if opcode_dim is not None:
        x[0, 0, opcode_dim] = 1.0
    if cmp_group:
        x[0, 0, BD.CMP_GROUP] = 1.0
    for cmp_i in cmp_flags:
        x[0, 0, BD.CMP + cmp_i] = 1.0
    if set_alu:
        x[0, 0, BD.ALU_LO + (a & 0xF)] = 1.0
        x[0, 0, BD.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    if set_carry:
        if shift_count is not None:
            x[0, 0, BD.AX_CARRY_LO + (shift_count & 0xF)] = 1.0
            x[0, 0, BD.AX_CARRY_HI + ((shift_count >> 4) & 0xF)] = 1.0
        else:
            x[0, 0, BD.AX_CARRY_LO + (b & 0xF)] = 1.0
            x[0, 0, BD.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    return x


def _delta_band(y: torch.Tensor, x: torch.Tensor, base: int) -> torch.Tensor:
    """Return the 16-wide W_down contribution at ``base``."""
    return (y[0, 0] - x[0, 0])[base : base + 16]


# ======================================================================
# Wave B row #1 — _layer10_alu_cmp_combine_rules (3 tests)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_cmp_combine_eq_override_at_step_end(ffns):
    """L10 EQ override at STEP_END: CMP+1=1 + CMP+2=1 → OUTPUT_LO+1 wins.

    Wave B #1: ``_layer10_alu_cmp_combine_rules`` migrated to MARK_SE_ONLY.
    The EQ 3-way override (``MARK_AX + CMP+1 + CMP+2`` → today) should
    become (``MARK_SE_ONLY + CMP+1 + CMP+2``) and flip OUTPUT_LO from
    the default 0 to 1.
    """
    x = _make_step_end_resid(
        opcode_dim=BD.OP_EQ,
        cmp_flags=(1, 2),
        set_alu=False,
        set_carry=False,
    )
    with torch.no_grad():
        y = ffns["l10_alu_cmp_combine"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out1, out0 = float(lo[1]), float(lo[0])
    assert out1 > out0, (
        f"L10 EQ override @SE (hi_eq+lo_eq → 1): OUTPUT_LO+1 must beat "
        f"OUTPUT_LO+0.\n  OUTPUT_LO+1={out1:.4f}, OUTPUT_LO+0={out0:.4f}\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}\n"
        f"  Currently FAILS: rule still gates on MARK_AX (which is 0 at SE)."
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_cmp_combine_ne_override_at_step_end(ffns):
    """L10 NE override at STEP_END: CMP+1=1 + CMP+2=1 → OUTPUT_LO+0 wins.

    NE default is 1; the override flips to 0 when hi_eq AND lo_eq.
    """
    x = _make_step_end_resid(
        opcode_dim=BD.OP_NE,
        cmp_flags=(1, 2),
        set_alu=False,
        set_carry=False,
    )
    with torch.no_grad():
        y = ffns["l10_alu_cmp_combine"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0, out1 = float(lo[0]), float(lo[1])
    assert out0 > out1, (
        f"L10 NE override @SE (hi_eq+lo_eq → 0): OUTPUT_LO+0 must beat "
        f"OUTPUT_LO+1.\n  OUTPUT_LO+0={out0:.4f}, OUTPUT_LO+1={out1:.4f}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_cmp_combine_lt_override_at_step_end(ffns):
    """L10 LT 2-way override at STEP_END: CMP+0=1 (hi_lt) → OUTPUT_LO+1.

    LT default is 0; the 2-way override flips to 1 on hi_lt alone.
    """
    x = _make_step_end_resid(
        opcode_dim=BD.OP_LT,
        cmp_flags=(0,),
        set_alu=False,
        set_carry=False,
    )
    with torch.no_grad():
        y = ffns["l10_alu_cmp_combine"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out1, out0 = float(lo[1]), float(lo[0])
    assert out1 > out0, (
        f"L10 LT 2-way override @SE (hi_lt → 1): OUTPUT_LO+1 must beat "
        f"OUTPUT_LO+0.\n  OUTPUT_LO+1={out1:.4f}, OUTPUT_LO+0={out0:.4f}"
    )


# ======================================================================
# Wave B row #2 — _l10_comparison_combine_rules (2 tests)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_comparison_combine_eq_default_at_step_end(ffns):
    """L10 EQ default at STEP_END: no CMP flags → OUTPUT_LO+0 fires.

    Wave B #2: ``_l10_comparison_combine_rules`` migrated to
    MARK_SE_ONLY. The EQ default (writes OUTPUT_LO+0 baseline) should
    fire when no override condition is met.
    """
    x = _make_step_end_resid(
        opcode_dim=BD.OP_EQ,
        cmp_flags=(),
        set_alu=False,
        set_carry=False,
    )
    with torch.no_grad():
        y = ffns["l10_comparison_combine"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    assert out0 > 0.01, (
        f"L10 EQ default @SE: OUTPUT_LO+0 baseline write must fire, "
        f"got {out0:.4f}.\n  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_comparison_combine_ge_default_at_step_end(ffns):
    """L10 GE default at STEP_END: no CMP flags → OUTPUT_LO+1 fires.

    GE default is 1 (assume a >= b), overridden when CMP+0 / CMP+1+CMP+3.
    """
    x = _make_step_end_resid(
        opcode_dim=BD.OP_GE,
        cmp_flags=(),
        set_alu=False,
        set_carry=False,
    )
    with torch.no_grad():
        y = ffns["l10_comparison_combine"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out1 = float(lo[1])
    assert out1 > 0.01, (
        f"L10 GE default @SE: OUTPUT_LO+1 baseline write must fire, "
        f"got {out1:.4f}."
    )


# ======================================================================
# Wave B rows #3-#5 — bitwise OR/XOR/AND (3 tests)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_bitwise_or_at_step_end(ffns):
    """L10 OR at STEP_END: 5|3=7 → OUTPUT_LO+7 wins.

    Wave B #3. The 3-way AND ``MARK_AX + ALU_LO+5 + AX_CARRY_LO+3``
    becomes ``MARK_SE_ONLY + ALU_LO+5 + AX_CARRY_LO+3``.
    """
    x = _make_step_end_resid(opcode_dim=BD.OP_OR, a=5, b=3)
    with torch.no_grad():
        y = ffns["l10_or"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    argmax = int(lo.argmax().item())
    assert argmax == 7, (
        f"L10 OR @SE (5|3=7): expected OUTPUT_LO argmax=7, got {argmax}.\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_bitwise_xor_at_step_end(ffns):
    """L10 XOR at STEP_END: 5^3=6 → OUTPUT_LO+6 wins."""
    x = _make_step_end_resid(opcode_dim=BD.OP_XOR, a=5, b=3)
    with torch.no_grad():
        y = ffns["l10_xor"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    argmax = int(lo.argmax().item())
    assert argmax == 6, (
        f"L10 XOR @SE (5^3=6): expected OUTPUT_LO argmax=6, got {argmax}.\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_bitwise_and_at_step_end(ffns):
    """L10 AND at STEP_END: 5&3=1 → OUTPUT_LO+1 wins."""
    x = _make_step_end_resid(opcode_dim=BD.OP_AND, a=5, b=3)
    with torch.no_grad():
        y = ffns["l10_and"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    argmax = int(lo.argmax().item())
    assert argmax == 1, (
        f"L10 AND @SE (5&3=1): expected OUTPUT_LO argmax=1, got {argmax}.\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


# ======================================================================
# Wave B row #6 — SHL/SHR zero shortcut (2 tests)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_shl_zero_at_step_end(ffns):
    """L10 SHL shift>=16 at STEP_END: ZFOD → OUTPUT_LO+0 fires.

    Wave B #6. shift_count=16 (AX_CARRY_HI[1]=1) selects the
    high-nibble-non-zero shortcut path; OUTPUT_LO+0 = +2/S write.
    """
    x = _make_step_end_resid(opcode_dim=BD.OP_SHL, a=5, shift_count=16)
    with torch.no_grad():
        y = ffns["l10_shl_shr"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    assert out0 > 0.01, (
        f"L10 SHL @SE (shift>=16): expected OUTPUT_LO+0 ZFOD fire, "
        f"got {out0:.4f}.\n  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l10_alu_shr_zero_at_step_end(ffns):
    """L10 SHR shift>=16 at STEP_END: ZFOD → OUTPUT_LO+0 fires."""
    x = _make_step_end_resid(opcode_dim=BD.OP_SHR, a=5, shift_count=16)
    with torch.no_grad():
        y = ffns["l10_shl_shr"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    assert out0 > 0.01, (
        f"L10 SHR @SE (shift>=16): expected OUTPUT_LO+0 ZFOD fire, "
        f"got {out0:.4f}."
    )


# ======================================================================
# Wave B row #7 — _layer9_cmp_rules (3 tests)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l9_cmp_hi_eq_at_step_end(ffns):
    """L9 hi_eq at STEP_END: ALU_HI[0]=AX_CARRY_HI[0]=1 → CMP+1 fires.

    Wave B #7. For byte values 5 and 5, both hi nibbles are 0; the
    hi_eq_0 unit should fire and write CMP+1.
    """
    x = _make_step_end_resid(a=5, b=5, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert hi_eq > 0.5, (
        f"L9 hi_eq @SE (hi nibble 0 == 0): CMP+1 must fire, got {hi_eq:.4f}.\n"
        f"  CMP+0={hi_lt:.4f}, CMP+1={hi_eq:.4f}, CMP+2={lo_eq:.4f}, "
        f"CMP+3={lo_lt:.4f}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l9_cmp_lo_eq_at_step_end(ffns):
    """L9 lo_eq at STEP_END: ALU_LO[5]=AX_CARRY_LO[5]=1 → CMP+2 fires."""
    x = _make_step_end_resid(a=5, b=5, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert lo_eq > 0.5, (
        f"L9 lo_eq @SE (lo nibble 5 == 5): CMP+2 must fire, got {lo_eq:.4f}.\n"
        f"  CMP+0={hi_lt:.4f}, CMP+1={hi_eq:.4f}, CMP+2={lo_eq:.4f}, "
        f"CMP+3={lo_lt:.4f}"
    )


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l9_cmp_lo_lt_at_step_end(ffns):
    """L9 lo_lt at STEP_END: ALU_LO[3] < AX_CARRY_LO[7] → CMP+3 fires."""
    x = _make_step_end_resid(a=3, b=7, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert lo_lt > 0.5, (
        f"L9 lo_lt @SE (lo nibble 3 < 7): CMP+3 must fire, got {lo_lt:.4f}.\n"
        f"  CMP+0={hi_lt:.4f}, CMP+1={hi_eq:.4f}, CMP+2={lo_eq:.4f}, "
        f"CMP+3={lo_lt:.4f}"
    )


# ======================================================================
# Wave B row #8 — _layer9_add_hi_nibble_rules (1 test)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l9_add_hi_nibble_at_step_end(ffns):
    """L9 ADD hi nibble at STEP_END: 0x53 + 0x42, hi result = 5+4 = 9.

    Wave B #8. ``_layer9_add_hi_nibble_rules`` migrated to MARK_SE_ONLY.
    With a=0x53 (hi=5), b=0x42 (hi=4), and carry=0, the hi-nibble add
    unit ``l9_add_hi_c0_a5_b4`` should fire and write OUTPUT_HI+9.
    """
    x = _make_step_end_resid(opcode_dim=BD.OP_ADD, a=0x53, b=0x42)
    with torch.no_grad():
        y = ffns["l9_add_hi"](x)
    hi = _delta_band(y, x, BD.OUTPUT_HI)
    argmax = int(hi.argmax().item())
    assert argmax == 9, (
        f"L9 ADD hi @SE (0x53+0x42 → hi=9): expected OUTPUT_HI argmax=9, "
        f"got {argmax}.\n  top3 idx={hi.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in hi.topk(3).values.tolist()]}"
    )


# ======================================================================
# Wave B row #9 — _layer9_sub_hi_nibble_rules (1 test)
# ======================================================================


@pytest.mark.xfail(strict=False, reason="pending Wave B migration")
def test_l9_sub_hi_nibble_at_step_end(ffns):
    """L9 SUB hi nibble at STEP_END: 0x88 - 0x55, hi result = 8-5 = 3.

    Wave B #9. With a=0x88 (hi=8), b=0x55 (hi=5), no borrow, the
    hi-nibble sub unit should fire and write OUTPUT_HI+3.
    """
    x = _make_step_end_resid(opcode_dim=BD.OP_SUB, a=0x88, b=0x55)
    with torch.no_grad():
        y = ffns["l9_sub_hi"](x)
    hi = _delta_band(y, x, BD.OUTPUT_HI)
    argmax = int(hi.argmax().item())
    assert argmax == 3, (
        f"L9 SUB hi @SE (0x88-0x55 → hi=3): expected OUTPUT_HI argmax=3, "
        f"got {argmax}.\n  top3 idx={hi.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in hi.topk(3).values.tolist()]}"
    )
