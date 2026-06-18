"""Isolation tests for L9/L10 ALU/CMP under COLLAPSED IMM+OP step conditions.

Mirrors the L15 memory_lookup isolation test pattern
(``tests/test_l15_memory_lookup_isolated.py``) but targets the L9/L10
ALU/CMP layers when the IMM+OP scheduler stage collapses, leaving stale
residual contamination at the operand-A read position.

Background — collapsed-step diagnostic (agent a868bc38)
------------------------------------------------------
On a "right after IMM 5; PSH; IMM 5" collapsed step, the L9 input
position carries ``ALU_LO+0 ~= 1.04`` (stale, left over from a prior
sub-cycle) on top of the legitimate operand encoding
(``ALU_LO+5 = 1.0`` for AX byte 0 = 5, ``AX_CARRY_LO+5 = 1.0`` for
TOS byte 0 = 5). The stale ``+0`` channel dominates several L9/L10
per-nibble cross-product rules that fire on ``ALU_*+0`` operand keys,
spuriously firing units that shouldn't and corrupting the ALU/CMP
output. Full-model smoke runs catch the downstream symptom, but there
is no fast unit test that catches the L9 input contamination directly.

These tests build the per-opcode L9 attention/FFN rules (or the L10
equivalent for ops that don't live at L9) via the existing IR
(``Primitives.lower_ffn_rules``) and feed a synthetic residual at the
AX-marker position with stale ``+0`` contamination. The expected
behaviour for each opcode is encoded as an assertion against the L9
(or L10) output residual.

Test set (13 tests, one per opcode)
-----------------------------------
Group A — STALE codifies the bug (7 tests, FAIL today, PASS after fix):

  test_add_stale_breaks_hi_nibble ............ ADD via L9 (FAIL today)
  test_sub_stale_breaks_hi_nibble ............ SUB via L9 (FAIL today)
  test_lt_stale_fires_spurious_lo_lt ......... LT  via L9 cmp (FAIL today)
  test_gt_stale_fires_spurious_lo_lt ......... GT  via L9 cmp (FAIL today)
  test_or_stale_breaks_output ................ OR  via L10 (FAIL today)
  test_and_stale_breaks_output ............... AND via L10 (FAIL today)
  test_xor_stale_breaks_output ............... XOR via L10 (FAIL today)

Group B — CLEAN ratchets (6 tests, PASS today, lock correct behaviour):

  test_eq_clean_lo_eq_fires .................. EQ  via L9 cmp (PASS today)
  test_ne_clean_lo_eq_fires .................. NE  via L9 cmp (PASS today)
  test_le_clean_correct ...................... LE  via L9 cmp (PASS today)
  test_ge_clean_correct ...................... GE  via L9 cmp (PASS today)
  test_shl_clean_zero_shortcut ............... SHL via L10 (PASS today)
  test_shr_clean_zero_shortcut ............... SHR via L10 (PASS today)

The 7+6 split mirrors the agent a868bc38 diagnostic: ADD/SUB/LT/GT/
OR/AND/XOR are the opcodes where the stale ``+0`` channel directly
corrupts the per-nibble lookup, while EQ/NE/LE/GE/SHL/SHR either
self-correct via their dual-CMP cascade or aren't reached via the
collapsed-step path.

Each test runs <0.05s (the FFN modules are cached via a module-scope
fixture). Whole file runs in <2s.
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
    _add_hi_nibble_rules,
    _layer9_cmp_rules,
    _sub_hi_nibble_rules,
)
from neural_vm.unified_compiler.ops.l10_ops import (  # noqa: E402
    _layer10_alu_bitwise_and_rules,
    _layer10_alu_bitwise_or_rules,
    _layer10_alu_bitwise_xor_rules,
    _layer10_alu_shl_shr_zero_rules,
)
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.vm_step import _SetDim as BD  # noqa: E402


S = 100.0
D_MODEL = 512


# ----------------------------------------------------------------------
# FFN construction helpers.
# ----------------------------------------------------------------------


def _build_ffn(rules) -> PureFFN:
    """Lower ``rules`` into a fresh ``PureFFN``.

    Mirrors the pattern used by ``test_div_multibyte_isolated.py``:
    ``PureFFN(dim=512, hidden_dim=len(rules))`` then
    ``Primitives.lower_ffn_rules(..., start_unit=0, S=100.0)``.
    """
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
    """Module-scoped FFN cache so the 7 FFNs are baked once per session."""
    return {
        "l9_add": _build_ffn(_add_hi_nibble_rules(S)),
        "l9_sub": _build_ffn(_sub_hi_nibble_rules(S)),
        "l9_cmp": _build_ffn(_layer9_cmp_rules(S)),
        "l10_or": _build_ffn(_layer10_alu_bitwise_or_rules(S)),
        "l10_and": _build_ffn(_layer10_alu_bitwise_and_rules(S)),
        "l10_xor": _build_ffn(_layer10_alu_bitwise_xor_rules(S)),
        "l10_shl_shr": _build_ffn(_layer10_alu_shl_shr_zero_rules(S)),
    }


# ----------------------------------------------------------------------
# Residual synthesis.
# ----------------------------------------------------------------------


def _make_resid(
    opcode_dim,
    *,
    a: int = 5,
    b: int = 5,
    stale: bool = False,
    cmp_group: bool = False,
    shift_count=None,
) -> torch.Tensor:
    """Synthesize the residual at the AX-marker position of an ALU step.

    Encodes the "right after IMM 5; PSH; IMM 5" pre-collapse state:
    AX byte 0 = ``a`` (lo nibble at ``ALU_LO+a&0xF``, hi at
    ``ALU_HI+(a>>4)&0xF``) and TOS byte 0 = ``b`` (lo at
    ``AX_CARRY_LO+b&0xF``, hi at ``AX_CARRY_HI+(b>>4)&0xF``).

    Args:
        opcode_dim: ``BD.OP_*`` dim to set.
        a, b: byte values for operand A (AX byte 0) and B (TOS byte 0).
        stale: if True, add ``ALU_LO+0 = 1.04``, ``AX_CARRY_LO+0 = 1.04``
            (and the parallel HI channel) to mimic collapsed-step
            contamination at the operand-A read position.
        cmp_group: set ``CMP_GROUP = 1`` (required gate for the L9 CMP
            rules).
        shift_count: if set, encode the shift count in AX_CARRY_LO/HI
            (overrides ``b`` for the SHL/SHR shortcut tests).
    """
    x = torch.zeros(1, 1, D_MODEL)
    x[0, 0, BD.MARK_AX] = 1.0
    x[0, 0, BD.CONST] = 1.0
    if opcode_dim is not None:
        x[0, 0, opcode_dim] = 1.0
    if cmp_group:
        x[0, 0, BD.CMP_GROUP] = 1.0
    # Operand A (AX byte 0): low nibble and high nibble one-hot.
    x[0, 0, BD.ALU_LO + (a & 0xF)] = 1.0
    x[0, 0, BD.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    # Operand B (TOS byte 0) — or shift count for SHL/SHR shortcut.
    if shift_count is not None:
        x[0, 0, BD.AX_CARRY_LO + (shift_count & 0xF)] = 1.0
        x[0, 0, BD.AX_CARRY_HI + ((shift_count >> 4) & 0xF)] = 1.0
    else:
        x[0, 0, BD.AX_CARRY_LO + (b & 0xF)] = 1.0
        x[0, 0, BD.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    if stale:
        # Stale ALU_LO[0]/AX_CARRY_LO[0] residual (the diagnostic from
        # agent a868bc38). Also stale the HI channels for ops that read
        # the high nibble (ADD/SUB hi nibble rules).
        x[0, 0, BD.ALU_LO + 0] += 1.04
        x[0, 0, BD.AX_CARRY_LO + 0] += 1.04
        x[0, 0, BD.ALU_HI + 0] += 1.04
        x[0, 0, BD.AX_CARRY_HI + 0] += 1.04
    return x


def _delta_band(y: torch.Tensor, x: torch.Tensor, base: int) -> torch.Tensor:
    """Return the 16-wide delta (W_down contribution only) at ``base``."""
    return (y[0, 0] - x[0, 0])[base : base + 16]


# ======================================================================
# Group A — STALE tests (FAIL today, codify the collapsed-step bug).
# ======================================================================
#
# Each STALE test injects ``ALU_*+0 = 1.04`` and ``AX_CARRY_*+0 = 1.04``
# on top of the legitimate operand encoding. The stale ``+0`` channels
# leak into every per-nibble cross-product rule that reads ``+0`` as a
# legitimate operand value, firing spuriously and corrupting the L9/L10
# output. These tests will FAIL until the upstream "stale residual at
# collapsed IMM+OP step" issue is fixed (clear the ALU bus at the start
# of each step or sentinel the lookup).


def test_add_stale_breaks_hi_nibble(ffns):
    """L9 ADD with stale ALU_HI+0 / AX_CARRY_HI+0 = 1.04 breaks the lookup.

    Setup: a=0x55 (hi=5), b=0x33 (hi=3), correct hi nibble result = 8.

    Codifies the bug: with stale ``ALU_HI+0 = AX_CARRY_HI+0 = 1.04``,
    the spurious units ``l9_add_hi_c0_a0_b3`` (→3),
    ``l9_add_hi_c0_a0_b0`` (→0), and ``l9_add_hi_c0_a5_b0`` (→5)
    fire on top of the legitimate ``l9_add_hi_c0_a5_b3 → result=8``
    write. The spurious +0 cell wins argmax over the correct +8 cell.
    """
    x = _make_resid(BD.OP_ADD, a=0x55, b=0x33, stale=True)
    with torch.no_grad():
        y = ffns["l9_add"](x)
    hi = _delta_band(y, x, BD.OUTPUT_HI)
    argmax = int(hi.argmax().item())
    assert argmax == 8, (
        f"L9 ADD stale (0x55+0x33 hi=8): expected OUTPUT_HI argmax=8, "
        f"got {argmax}.\n"
        f"  top3 idx={hi.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in hi.topk(3).values.tolist()]}\n"
        f"  Bug: stale ALU_HI+0 / AX_CARRY_HI+0 = 1.04 fires the "
        f"l9_add_hi_c0_a0_b3 and l9_add_hi_c0_a5_b0 units, summing\n"
        f"  to overwhelm the correct l9_add_hi_c0_a5_b3 -> result=8 "
        f"write. Codified to FAIL until upstream clears the ALU bus."
    )


def test_sub_stale_breaks_hi_nibble(ffns):
    """L9 SUB with stale ALU_HI+0 / AX_CARRY_HI+0 = 1.04 breaks the lookup.

    Setup: a=0x88 (hi=8), b=0x55 (hi=5), correct hi nibble result = 3.

    Codifies the bug: with stale ``ALU_HI+0 = AX_CARRY_HI+0 = 1.04``,
    spurious sub units fire on the stale ``+0`` channels and corrupt
    OUTPUT_HI away from the correct nibble 3.
    """
    x = _make_resid(BD.OP_SUB, a=0x88, b=0x55, stale=True)
    with torch.no_grad():
        y = ffns["l9_sub"](x)
    hi = _delta_band(y, x, BD.OUTPUT_HI)
    argmax = int(hi.argmax().item())
    assert argmax == 3, (
        f"L9 SUB stale (0x88-0x55 hi=3): expected OUTPUT_HI argmax=3, "
        f"got {argmax}.\n"
        f"  top3 idx={hi.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in hi.topk(3).values.tolist()]}"
    )


def test_lt_stale_fires_spurious_lo_lt(ffns):
    """L9 LT with stale ALU_LO+0 / AX_CARRY_LO+0 = 1.04 fires spurious lo_lt.

    Setup: a=b=5, correct result: 5<5 is false, so CMP+3 (lo_lt)
    should be ~0 and the downstream LT result should be 0.

    Codifies the bug: ``l9_cmp_lo_lt_a0_b5`` reads
    ``ALU_LO+0 + AX_CARRY_LO+5`` — with stale ALU_LO+0=1.04 and
    legitimate AX_CARRY_LO+5=1, the score is 1+0+1.04+1.0=3.04 > 2.5
    and the rule fires, writing to CMP+3. The spurious firing makes
    the cascaded L10 cmp_combine LT override fire incorrectly,
    claiming 5<5 is true.
    """
    x = _make_resid(BD.OP_LT, a=5, b=5, stale=True, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    # lo_lt must remain ~0 for true a==b. Stale fires it spuriously.
    assert abs(lo_lt) < 0.1, (
        f"L9 LT stale (5<5 false): expected CMP+3 (lo_lt) ~= 0, "
        f"got {lo_lt:.4f}.\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+1 (hi_eq)={hi_eq:.4f} "
        f"CMP+2 (lo_eq)={lo_eq:.4f} CMP+3 (lo_lt)={lo_lt:.4f}\n"
        f"  Bug: stale ALU_LO+0=1.04 + AX_CARRY_LO+5=1.0 sums above "
        f"threshold for the l9_cmp_lo_lt_a0_b5 unit."
    )


def test_gt_stale_fires_spurious_lo_lt(ffns):
    """L9 GT with stale residual: lo_lt fires spuriously (same as LT).

    GT downstream uses CMP+3 (lo_lt) as part of its 3-way override
    ``CMP+1 AND CMP+3 → 0``. Spurious lo_lt firing flips the GT
    result from 0 (correct) to 1.
    """
    x = _make_resid(BD.OP_GT, a=5, b=5, stale=True, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    _, _, _, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert abs(lo_lt) < 0.1, (
        f"L9 GT stale: expected CMP+3 (lo_lt) ~= 0, got {lo_lt:.4f}.\n"
        f"  Same bug as test_lt_stale_fires_spurious_lo_lt — the "
        f"L9 cmp rules read the operand-A nibble from ALU_LO, so the "
        f"stale +0 channel leaks into every comparison opcode's "
        f"cascade."
    )


def test_or_stale_breaks_output(ffns):
    """L10 OR with stale: 5|5=5 → spurious OUTPUT_LO+0 fires near the correct 5.

    Codifies the bug: the 3-way AND rule
    ``l10_bitwise_or_lo_a0_b0`` (writes OUTPUT_LO+(0|0)=OUTPUT_LO+0)
    fires above threshold thanks to the stale +0 channels
    (40 + 30*1.04 + 30*1.04 = 102.4 > 80). The spurious value at
    OUTPUT_LO+0 must stay well below the correct OUTPUT_LO+5 level.
    """
    x = _make_resid(BD.OP_OR, a=5, b=5, stale=True)
    with torch.no_grad():
        y = ffns["l10_or"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out5 = float(lo[5])
    out0 = float(lo[0])
    # Bug: OUTPUT_LO+0 picks up ~44.8 of spurious mass (vs +5 at 124.8).
    # Contract: spurious cell should be < 25% of the correct cell.
    assert out0 < 0.25 * out5, (
        f"L10 OR stale (5|5=5): OUTPUT_LO+0 spurious mass "
        f"({out0:.4f}) must stay below 25% of OUTPUT_LO+5 "
        f"({out5:.4f}).\n"
        f"  full top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}\n"
        f"  Bug: stale ALU_LO+0=1.04 makes the bitwise OR cross-"
        f"product rules with a=0 fire above threshold."
    )


def test_and_stale_breaks_output(ffns):
    """L10 AND with stale: 5&5=5 → spurious OUTPUT_LO+0 overtakes correct +5.

    For AND, stale ``ALU_LO+0`` and ``AX_CARRY_LO+0`` cause the rule
    ``l10_bitwise_and_lo_a0_b0`` (writes OUTPUT_LO+(0&0)=OUTPUT_LO+0)
    to fire near-threshold; multiple ``a=0,b=*`` and ``a=*,b=0``
    rules also fire (all of which write to OUTPUT_LO+0 since AND with
    0 = 0), summing to overwhelm the single legitimate
    ``a=5,b=5 → OUTPUT_LO+5`` write.
    """
    x = _make_resid(BD.OP_AND, a=5, b=5, stale=True)
    with torch.no_grad():
        y = ffns["l10_and"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out5 = float(lo[5])
    out0 = float(lo[0])
    assert out0 < 0.25 * out5, (
        f"L10 AND stale (5&5=5): OUTPUT_LO+0 spurious mass "
        f"({out0:.4f}) must stay below 25% of OUTPUT_LO+5 "
        f"({out5:.4f}).\n"
        f"  full top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


def test_xor_stale_breaks_output(ffns):
    """L10 XOR with stale: 5^5=0 → spurious OUTPUT_LO+5 ties with correct +0.

    Codifies the bug: stale ALU_LO+0=1.04 + AX_CARRY_LO+5=1 fires
    ``l10_bitwise_xor_lo_a0_b5`` (writes OUTPUT_LO+(0^5)=OUTPUT_LO+5).
    Stale AX_CARRY_LO+0=1.04 + ALU_LO+5=1 fires
    ``l10_bitwise_xor_lo_a5_b0`` (writes OUTPUT_LO+(5^0)=OUTPUT_LO+5).
    Together these spurious writes match the legitimate
    ``l10_bitwise_xor_lo_a5_b5`` → OUTPUT_LO+(5^5)=OUTPUT_LO+0 — so
    OUTPUT_LO+0 (correct) and OUTPUT_LO+5 (spurious) end up at
    near-equal mass, breaking argmax tie-break determinism downstream.
    """
    x = _make_resid(BD.OP_XOR, a=5, b=5, stale=True)
    with torch.no_grad():
        y = ffns["l10_xor"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    out5 = float(lo[5])
    # The correct answer is +0; the spurious +5 must remain well below.
    assert out5 < 0.25 * max(out0, 1e-6), (
        f"L10 XOR stale (5^5=0): OUTPUT_LO+5 spurious mass "
        f"({out5:.4f}) must stay below 25% of OUTPUT_LO+0 "
        f"({out0:.4f}).\n"
        f"  full top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}\n"
        f"  Bug: stale ALU_LO+0 fires xor_lo_a0_b5 (→+5) and stale "
        f"AX_CARRY_LO+0 fires xor_lo_a5_b0 (→+5), summing to tie or "
        f"exceed the legitimate xor_lo_a5_b5 (→+0) write."
    )


# ======================================================================
# Group B — CLEAN ratchets (PASS today, lock correct behaviour).
# ======================================================================
#
# These tests use the CLEAN residual (no stale +0 contamination) to
# verify the per-opcode L9/L10 lookup contract. They PASS today and
# act as a ratchet: any future change that breaks the basic per-nibble
# lookup will trip these.


def test_eq_clean_lo_eq_fires(ffns):
    """L9 EQ with clean a=b=5: hi_eq AND lo_eq fire, hi_lt/lo_lt dormant.

    Codifies the EQ contract at L9 input: ``l9_cmp_hi_eq_0`` fires on
    the high nibble (a=b=0 for byte 0 of 5) writing CMP+1, and
    ``l9_cmp_lo_eq_5`` fires on the low nibble writing CMP+2. The
    downstream L10 cmp_combine EQ override 3-way (CMP+1 AND CMP+2 → 1)
    will then flip the default 0 to 1.
    """
    x = _make_resid(BD.OP_EQ, a=5, b=5, stale=False, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert hi_eq > 0.5 and lo_eq > 0.5, (
        f"EQ clean (5==5): hi_eq/lo_eq must fire.\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+1 (hi_eq)={hi_eq:.4f} "
        f"CMP+2 (lo_eq)={lo_eq:.4f} CMP+3 (lo_lt)={lo_lt:.4f}"
    )
    assert abs(hi_lt) < 0.1 and abs(lo_lt) < 0.1, (
        f"EQ clean (5==5): hi_lt/lo_lt must NOT fire.\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+3 (lo_lt)={lo_lt:.4f}"
    )


def test_ne_clean_lo_eq_fires(ffns):
    """L9 NE with clean a=b=5: same L9 CMP signal as EQ (gate-shared).

    The L9 ``_layer9_cmp_rules`` gates on CMP_GROUP, not on the
    specific OP_* dim — so NE produces the same hi_eq/lo_eq pattern
    as EQ for a==b. The downstream L10 cmp_combine NE rule uses the
    same CMP+1+CMP+2 signal to override its default 1 → 0.
    """
    x = _make_resid(BD.OP_NE, a=5, b=5, stale=False, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert hi_eq > 0.5 and lo_eq > 0.5, (
        f"NE clean (5!=5 false): hi_eq/lo_eq must fire.\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+1 (hi_eq)={hi_eq:.4f} "
        f"CMP+2 (lo_eq)={lo_eq:.4f} CMP+3 (lo_lt)={lo_lt:.4f}"
    )
    assert abs(hi_lt) < 0.1 and abs(lo_lt) < 0.1, (
        f"NE clean: hi_lt/lo_lt must be dormant when a==b.\n"
        f"  CMP+0={hi_lt:.4f} CMP+3={lo_lt:.4f}"
    )


def test_le_clean_correct(ffns):
    """L9 LE with clean a=5, b=7: hi_eq fires, lo_lt fires (5<7), lo_eq dormant.

    For a=5, b=7 (a<b): byte 0 hi nibble of both is 0 → hi_eq fires
    (CMP+1). Lo nibbles 5 < 7 → lo_lt fires (CMP+3). The downstream
    L10 cmp_combine LE 3-way override (CMP+1 AND CMP+3 → 1) flips
    LE default 0 → 1. The lookup signal must clearly differentiate.
    """
    x = _make_resid(BD.OP_LE, a=5, b=7, stale=False, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert hi_eq > 0.5, (
        f"LE clean (5<=7): hi_eq must fire (hi nibbles equal).\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+1 (hi_eq)={hi_eq:.4f} "
        f"CMP+2 (lo_eq)={lo_eq:.4f} CMP+3 (lo_lt)={lo_lt:.4f}"
    )
    assert lo_lt > 0.5, (
        f"LE clean (5<=7): lo_lt must fire (5<7 at lo nibble).\n"
        f"  CMP+3 (lo_lt)={lo_lt:.4f}"
    )
    assert abs(lo_eq) < 0.1, (
        f"LE clean (5<7): lo_eq must be dormant.\n"
        f"  CMP+2 (lo_eq)={lo_eq:.4f}"
    )


def test_ge_clean_correct(ffns):
    """L9 GE with clean a=7, b=5: hi_eq fires, hi_lt and lo_lt dormant.

    For a=7, b=5 (a>b): byte 0 hi nibble of both is 0 → hi_eq fires
    (CMP+1). Lo nibbles 7 > 5 → lo_lt does NOT fire (rules are
    a<b ordered). hi_lt also dormant.

    The downstream L10 cmp_combine GE rule keeps the default 1
    (no override fires), giving GE=1 correctly.
    """
    x = _make_resid(BD.OP_GE, a=7, b=5, stale=False, cmp_group=True)
    with torch.no_grad():
        y = ffns["l9_cmp"](x)
    cmp_delta = (y[0, 0] - x[0, 0])[BD.CMP : BD.CMP + 4]
    hi_lt, hi_eq, lo_eq, lo_lt = (float(v) for v in cmp_delta.tolist())
    assert hi_eq > 0.5, (
        f"GE clean (7>=5): hi_eq must fire (hi nibbles equal).\n"
        f"  CMP+0 (hi_lt)={hi_lt:.4f} CMP+1 (hi_eq)={hi_eq:.4f} "
        f"CMP+2 (lo_eq)={lo_eq:.4f} CMP+3 (lo_lt)={lo_lt:.4f}"
    )
    assert abs(hi_lt) < 0.1 and abs(lo_lt) < 0.1, (
        f"GE clean (7>5): hi_lt and lo_lt must be dormant (a>=b).\n"
        f"  CMP+0={hi_lt:.4f} CMP+3={lo_lt:.4f}"
    )


def test_shl_clean_zero_shortcut(ffns):
    """L10 SHL with shift_count=16: shift>=16 ZFOD → OUTPUT_LO+0 fires.

    Codifies ``_layer10_alu_shl_shr_zero_rules`` case A: shift count
    high nibble != 0 (so AX_CARRY_HI[0] = 0 fails the suppressor),
    threshold 59 only clears when MARK_AX is hot. Writes
    OUTPUT_LO+0 = +2/S.
    """
    # shift_count=16 -> AX_CARRY_LO+0=1, AX_CARRY_HI+1=1; AX_CARRY_HI[0]=0.
    x = _make_resid(BD.OP_SHL, a=5, b=0, stale=False, shift_count=16)
    with torch.no_grad():
        y = ffns["l10_shl_shr"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    assert out0 > 0.5, (
        f"L10 SHL clean (shift>=16): expected OUTPUT_LO+0 ~> 1.0 "
        f"(ZFOD write), got {out0:.4f}.\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )


def test_shr_clean_zero_shortcut(ffns):
    """L10 SHR with shift_count=16: shift>=16 ZFOD → OUTPUT_LO+0 fires."""
    x = _make_resid(BD.OP_SHR, a=5, b=0, stale=False, shift_count=16)
    with torch.no_grad():
        y = ffns["l10_shl_shr"](x)
    lo = _delta_band(y, x, BD.OUTPUT_LO)
    out0 = float(lo[0])
    assert out0 > 0.5, (
        f"L10 SHR clean (shift>=16): expected OUTPUT_LO+0 ~> 1.0 "
        f"(ZFOD write), got {out0:.4f}.\n"
        f"  top3 idx={lo.topk(3).indices.tolist()}, "
        f"vals={[round(v, 4) for v in lo.topk(3).values.tolist()]}"
    )
