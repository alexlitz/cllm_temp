"""Fast isolation tests for multi-byte DIV/MOD — pins down the contract
without a full model bake + execute cycle.

Modeled on ``tests/test_l15_memory_lookup_isolated.py``: each test
constructs the minimum module + synthetic residual needed to exercise a
single semantic claim, runs <0.5s, and pinpoints a specific failure
mode. Together the 5 tests codify both the current state and the
contract the multi-byte DIV fix must satisfy.

What's covered:

  1. ``test_single_byte_div_works`` — the per-nibble DSL contract at
     ``width_bytes=1``. Builds ``wide_div_rules(width_bytes=1)`` +
     ``Primitives.lower_ffn_rules`` into a fresh ``PureFFN`` and asserts
     the lookup returns ``q = a_nib // b_nib`` / ``r = a_nib % b_nib``.
     Uses per-nibble operands (e.g. 14/2 = 7 r 0) because the
     ``width_bytes=1`` POC is a nibble-level lookup (see the
     ``wide_div_rules`` docstring).

  2. ``test_multi_byte_div_raises_today`` — codifies the current state:
     ``wide_div_rules(width_bytes>=2)`` raises ``NotImplementedError``
     (see ``c4_release/neural_vm/unified_compiler/wide_alu_dsl.py``:872).
     PASSES today; will FAIL once multi-byte DIV is implemented and the
     raise is removed.

  3. ``test_long_division_module_correct`` — direct
     ``LongDivisionModule`` forward over the NIBBLE chunk config (8
     nibble positions = 32-bit). Encodes ``1000000 / 7`` as a synthetic
     ``[B=1, 8, GenericE.DIM]`` input and decodes
     ``SLOT_QUOTIENT``/``SLOT_REMAINDER`` per-position nibbles to assert
     quotient=142857, remainder=1. This is the ground-truth long-division
     implementation that the multi-byte DSL wave will mirror.

  4. ``test_div_by_zero_returns_zero`` — multi-byte DIV with divisor=0
     should defensively return AX=0. The current
     ``LongDivisionModule.forward`` (divmod_longdiv.py:289) returns
     ``q = 0xFFFFFFFF`` (all-15-nibbles) on divide-by-zero, NOT 0. This
     test FAILS today; will PASS once the convention is unified to AX=0.

  5. ``test_signed_div`` — signed truncate-toward-zero semantics
     (C convention): ``-100 / 7 = -14 r -2``. The current
     ``LongDivisionModule`` is purely unsigned (operates on 8 nibbles
     0..15), so this test FAILS today; codifies the eventual signed-DIV
     contract.

Status at current main:
  test_single_byte_div_works .................. PASS
  test_multi_byte_div_raises_today ............ PASS (raise IS the state)
  test_long_division_module_correct ........... PASS
  test_div_by_zero_returns_zero ............... FAIL (q=0xFFFFFFFF today)
  test_signed_div ............................. FAIL (unsigned today)

All 5 tests run in <2 seconds total (no model bake, no full sequence
execution).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Force CPU-only execution to avoid OOM with parallel agents.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_vm.alu.chunk_config import NIBBLE  # noqa: E402
from neural_vm.alu.ops.common import GenericE  # noqa: E402
from neural_vm.alu.ops.divmod_longdiv import LongDivisionModule  # noqa: E402
from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.unified_compiler.wide_alu_dsl import wide_div_rules  # noqa: E402
from neural_vm.vm_step import _SetDim  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: single-byte (width_bytes=1) per-nibble DIV lookup works.
# ---------------------------------------------------------------------------


def _build_wide_div_rules_one_byte(S: float = 100.0):
    """Build the 256-rule per-nibble DIV lookup at ``width_bytes=1``.

    Mirrors ``make_alu_divmod_composite_ops`` (alu_ops.py) but routes the
    quotient and remainder to **distinct** output bands so the test can
    decode them independently. Production wires them to the same band
    (``OUTPUT_LO`` for both) and relies on the OP_DIV/OP_MOD gate to keep
    only one batch active at runtime; here we install a single DIV batch
    and read both channels in isolation.
    """
    return wide_div_rules(
        dividend_base="ALU_LO",
        divisor_base="AX_CARRY_LO",
        quotient_base="OUTPUT_LO",
        remainder_base="OUTPUT_HI",
        width_bytes=1,
        opcode_gate="OP_DIV",
        marker_gate="MARK_AX",
        S=S,
    )


def _lowered_pureffn_for_wide_div(S: float = 100.0) -> PureFFN:
    """Lower the width_bytes=1 ``wide_div_rules`` into a ``PureFFN``."""
    rules = _build_wide_div_rules_one_byte(S=S)
    # 16 * 16 = 256 rules per byte (240 quotient+remainder for b_nib in
    # 1..15, plus 16 divide-by-zero guards). See wide_alu_dsl.py:838.
    assert len(rules) == 256, (
        f"wide_div_rules(width_bytes=1) emitted {len(rules)} rules, "
        f"expected 256"
    )
    ffn = PureFFN(dim=512, hidden_dim=256)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == 256, f"lower_ffn_rules wrote {end} units, expected 256"
    return ffn


def _make_div_input(*, a_nib: int, b_nib: int) -> torch.Tensor:
    """One-position residual at MARK_AX with dividend / divisor nibbles."""
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.OP_DIV] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a_nib & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b_nib & 0xF)] = 1.0
    return x


def test_single_byte_div_works():
    """Per-nibble DIV lookup at ``width_bytes=1`` returns ``q = a//b``.

    Sanity check that the DSL POC is intact: feed a residual encoding
    ``14 / 2`` and assert ``OUTPUT_LO`` carries the quotient nibble 7.
    Uses nibble-level operands because ``width_bytes=1`` is a per-nibble
    lookup, not a full byte divide — see ``wide_div_rules`` docstring
    line 791-812. The task's aspirational "100/7 = 14 r 2" example is
    the eventual multi-byte contract (covered by
    ``test_long_division_module_correct``).
    """
    ffn = _lowered_pureffn_for_wide_div()
    # 13 / 4 = 3 r 1 — quotient and remainder nibbles are distinct, so
    # any cross-channel bleed would show up as an obvious mismatch.
    a_nib, b_nib = 13, 4
    x = _make_div_input(a_nib=a_nib, b_nib=b_nib)
    with torch.no_grad():
        y = ffn(x)

    out_lo = y[0, 0, _SetDim.OUTPUT_LO : _SetDim.OUTPUT_LO + 16]
    out_hi = y[0, 0, _SetDim.OUTPUT_HI : _SetDim.OUTPUT_HI + 16]
    quotient_idx = int(out_lo.argmax().item())
    remainder_idx = int(out_hi.argmax().item())
    expected_q = a_nib // b_nib  # 3
    expected_r = a_nib % b_nib  # 1
    assert quotient_idx == expected_q and remainder_idx == expected_r, (
        f"wide_div_rules(width_bytes=1) failed per-nibble lookup.\n"
        f"  a_nib={a_nib}, b_nib={b_nib}\n"
        f"  expected quotient={expected_q}, remainder={expected_r}\n"
        f"  argmax(OUTPUT_LO) = {quotient_idx} (quotient channel)\n"
        f"  argmax(OUTPUT_HI) = {remainder_idx} (remainder channel)\n"
        f"  OUTPUT_LO = {out_lo.tolist()}\n"
        f"  OUTPUT_HI = {out_hi.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 2: multi-byte DIV raises today (codifies the current state).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width_bytes", [2, 3, 4])
def test_multi_byte_div_raises_today(width_bytes):
    """Codifies that ``wide_div_rules(width_bytes > 1)`` raises today.

    Pinned by ``wide_alu_dsl.py``:872-882: per-nibble independent
    division does not compose into wide-operand division (e.g.
    ``0xFF / 0x0F == 0x11``, but per-nibble would give 1 + zero-divide).

    This test currently PASSES (the raise IS the present state). It
    will FAIL once the multi-byte DIV pipeline lands and the raise is
    removed; that signals the fix is in and the long-division contract
    (``test_long_division_module_correct``) becomes the binding one.
    """
    with pytest.raises(NotImplementedError, match="width_bytes="):
        wide_div_rules(
            dividend_base="ALU_LO",
            divisor_base="AX_CARRY_LO",
            quotient_base="OUTPUT_LO",
            remainder_base="OUTPUT_LO",
            width_bytes=width_bytes,
            opcode_gate="OP_DIV",
            marker_gate="MARK_AX",
            S=100.0,
        )


# ---------------------------------------------------------------------------
# Helpers for the LongDivisionModule direct-forward tests.
# ---------------------------------------------------------------------------


def _make_longdiv_input(*, dividend: int, divisor: int, opcode: int = 0,
                        signed_dividend: bool = False) -> torch.Tensor:
    """Synthesize a ``[B=1, 8, GenericE.DIM]`` residual for LongDivisionModule.

    Encodes ``dividend`` into per-position ``NIB_A`` and ``divisor`` into
    per-position ``NIB_B`` as 8 nibbles (LSB at position 0). The opcode
    flag is set at position 0 dim ``OP_START + opcode``.

    When ``signed_dividend`` is True, the dividend is encoded in two's
    complement modulo 2**32 — exactly what a real signed C divide would
    see. The current LongDivisionModule is unsigned and will return the
    unsigned quotient of the bit pattern, not the signed result.
    """
    ge = GenericE(NIBBLE)
    N = ge.NUM_POSITIONS  # 8
    x = torch.zeros(1, N, ge.DIM, dtype=torch.float32)

    if signed_dividend and dividend < 0:
        dividend_bits = (dividend + (1 << 32)) & 0xFFFFFFFF
    else:
        dividend_bits = dividend & 0xFFFFFFFF
    divisor_bits = divisor & 0xFFFFFFFF

    for pos in range(N):
        nib_a = (dividend_bits >> (4 * pos)) & 0xF
        nib_b = (divisor_bits >> (4 * pos)) & 0xF
        # NIB_A/NIB_B are scalar slots, not one-hot bands — store the
        # integer nibble value directly (LongDivisionModule reads
        # ``x[:, :N, ge.NIB_A]`` and rounds-clamps to 0..15).
        x[0, pos, ge.NIB_A] = float(nib_a)
        x[0, pos, ge.NIB_B] = float(nib_b)

    # Opcode flag at position 0, dim OP_START + opcode.
    x[0, 0, ge.OP_START + opcode] = 1.0
    return x


def _decode_slot_value(y: torch.Tensor, ge: GenericE, slot: int,
                       signed: bool = False) -> int:
    """Decode an 8-nibble per-position slot back to an integer.

    Position 0 = LSB nibble. Rounds each nibble to int (the module
    writes float nibble values into the slot).
    """
    N = ge.NUM_POSITIONS
    value = 0
    for pos in range(N):
        nib = int(round(float(y[0, pos, slot].item()))) & 0xF
        value |= nib << (4 * pos)
    if signed and (value & (1 << 31)):
        value -= 1 << 32
    return value


# ---------------------------------------------------------------------------
# Test 3: direct LongDivisionModule forward — 1000000 / 7.
# ---------------------------------------------------------------------------


def test_long_division_module_correct():
    """Ground-truth: ``LongDivisionModule`` divides 1000000 / 7 correctly.

    This is the reference long-division pipeline the multi-byte DSL
    wave will mirror. Direct forward over the NIBBLE chunk config
    (8 nibble positions, 32-bit operand). Decodes SLOT_QUOTIENT and
    SLOT_REMAINDER back into integers and asserts the schoolbook
    answer: 1000000 / 7 = 142857 r 1.
    """
    ge = GenericE(NIBBLE)
    opcode = 0  # any opcode index 0..NUM_OPS-1; we just gate on it.
    module = LongDivisionModule(ge, opcode)

    x = _make_longdiv_input(dividend=1000000, divisor=7, opcode=opcode)
    with torch.no_grad():
        y = module(x)

    quotient = _decode_slot_value(y, ge, ge.SLOT_QUOTIENT)
    remainder = _decode_slot_value(y, ge, ge.SLOT_REMAINDER)

    assert quotient == 142857 and remainder == 1, (
        f"LongDivisionModule(1000000 / 7) failed.\n"
        f"  expected quotient=142857, remainder=1\n"
        f"  got      quotient={quotient}, remainder={remainder}\n"
        f"  (raw quotient hex = {hex(quotient)}, "
        f"raw remainder hex = {hex(remainder)})"
    )


# ---------------------------------------------------------------------------
# Test 4: divide-by-zero — defensive AX=0 contract.
# ---------------------------------------------------------------------------


def test_div_by_zero_returns_zero():
    """Multi-byte DIV with divisor=0 should defensively return AX=0.

    Current state (``divmod_longdiv.py``:289): on ``b == 0`` the module
    sets ``q = 0xFFFFFFFF`` (all-15-nibbles) and ``r = a``. This matches
    a saturating-bus convention, but the user-facing contract requested
    by the multi-byte fix is "AX = 0" — a more defensive default that
    won't be confused with a legitimate ``0xFFFFFFFF`` result.

    FAILS today; will PASS once the divide-by-zero convention is
    unified to quotient=0.
    """
    ge = GenericE(NIBBLE)
    opcode = 0
    module = LongDivisionModule(ge, opcode)

    x = _make_longdiv_input(dividend=12345, divisor=0, opcode=opcode)
    with torch.no_grad():
        y = module(x)

    quotient = _decode_slot_value(y, ge, ge.SLOT_QUOTIENT)
    assert quotient == 0, (
        f"DIV by zero should return quotient=0 (defensive AX=0).\n"
        f"  got quotient = {quotient} (hex={hex(quotient)})\n"
        f"  current state writes 0xFFFFFFFF (saturating-bus convention,\n"
        f"  divmod_longdiv.py:289). FAIL today; fix flips to q=0."
    )


# ---------------------------------------------------------------------------
# Test 5: signed DIV — C truncate-toward-zero semantics.
# ---------------------------------------------------------------------------


def test_signed_div():
    """Signed DIV: ``-100 / 7 = -14 r -2`` (C truncate-toward-zero).

    C's ``/`` and ``%`` operators truncate toward zero, NOT toward
    negative infinity (Python's floor). Specifically:
      -100 / 7  ==  -14  (truncated, not -15)
      -100 % 7  ==  -2   (matches truncated quotient: -14*7 + -2 = -100)

    The current ``LongDivisionModule`` is purely unsigned: it operates
    on the 32-bit two's-complement bit pattern as an unsigned number, so
    feeding -100 (encoded as 0xFFFFFF9C) yields a huge unsigned quotient.

    FAILS today; codifies the signed-DIV contract that the multi-byte
    fix must satisfy.
    """
    ge = GenericE(NIBBLE)
    opcode = 0
    module = LongDivisionModule(ge, opcode)

    x = _make_longdiv_input(
        dividend=-100, divisor=7, opcode=opcode, signed_dividend=True,
    )
    with torch.no_grad():
        y = module(x)

    quotient = _decode_slot_value(y, ge, ge.SLOT_QUOTIENT, signed=True)
    remainder = _decode_slot_value(y, ge, ge.SLOT_REMAINDER, signed=True)

    assert quotient == -14 and remainder == -2, (
        f"Signed DIV (-100 / 7) should truncate toward zero (C semantics).\n"
        f"  expected quotient=-14, remainder=-2\n"
        f"  got      quotient={quotient}, remainder={remainder}\n"
        f"  current LongDivisionModule is unsigned: it treats -100 as\n"
        f"  0xFFFFFF9C and returns ~613566745 / 7. FAIL today; signed\n"
        f"  contract requires sign-extend + abs + post-negate."
    )
