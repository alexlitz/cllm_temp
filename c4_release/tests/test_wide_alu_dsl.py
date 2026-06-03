"""POC byte-identity test for ``wide_alu_dsl.bitwise_rules`` (Task 81).

Validates the IR-DSL contract from ``docs/IR_DSL_DESIGN.md`` Section 4:

  ``rule_lowered_ffn.forward(x)  decodes to the same OUTPUT byte
   as  hand_composite.forward(x)``

on randomized AND/OR/XOR inputs at the MARK_AX position.

The rule-lowered PureFFN uses ``Primitives.lower_ffn_rules`` over the
512 rules emitted by ``bitwise_rules`` for one opcode at a time. The
hand composite is ``ALUAndOrXor`` (= ``PureNeuralALU('bitwise').forward``)
which is the existing imperative ``efficient_alu_neural.py`` reference.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural_vm.base_layers import PureFFN  # noqa: E402
from neural_vm.efficient_alu_neural import ALUAndOrXor  # noqa: E402
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.unified_compiler.wide_alu_dsl import (  # noqa: E402
    bitwise_rules,
    wide_add_rules,
)
from neural_vm.vm_step import _SetDim  # noqa: E402


# Opcode flag dim names used by the wide_alu_dsl bitwise helper. Matches
# the legacy ``OP_AND/OR/XOR`` dims that the ALUAndOrXor composite reads
# via BDToGEConverter (see ``efficient_alu_neural.py`` line 522-524).
_OP_DIM_NAMES = {
    "and": "OP_AND",
    "or": "OP_OR",
    "xor": "OP_XOR",
}

_PY_OP = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
}


def _build_bitwise_rules_for(op: str, S: float = 100.0):
    """Helper: bitwise_rules with the legacy ALU dim wiring (lookup mode)."""
    return bitwise_rules(
        op=op,
        operand_a_lo="ALU_LO",
        operand_a_hi="ALU_HI",
        operand_b_lo="AX_CARRY_LO",
        operand_b_hi="AX_CARRY_HI",
        result_lo="OUTPUT_LO",
        result_hi="OUTPUT_HI",
        opcode_gate=_OP_DIM_NAMES[op],
        marker_gate="MARK_AX",
        S=S,
    )


def _lowered_pureffn_for(op: str, S: float = 100.0) -> PureFFN:
    """Build a fresh ``PureFFN(dim=512, hidden_dim=512)`` lowered from
    the 512 rules emitted by ``bitwise_rules`` for ``op``.
    """
    rules = _build_bitwise_rules_for(op, S=S)
    assert len(rules) == 512, (
        f"bitwise_rules({op!r}) emitted {len(rules)} rules, expected 512"
    )
    ffn = PureFFN(dim=512, hidden_dim=512)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == 512, f"lower_ffn_rules wrote {end} units, expected 512"
    return ffn


def _make_input(*, a: int, b: int, op: str) -> torch.Tensor:
    """One-position residual at MARK_AX with operand A/B nibbles + opcode.

    Mirrors ``test_alu_wide_composites_per_op._make_alu_byte_input``:
    ALU_LO/HI carries operand A; AX_CARRY_LO/HI carries operand B.
    """
    op_dim = getattr(_SetDim, _OP_DIM_NAMES[op])
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a & 0xF)] = 1.0
    x[0, 0, _SetDim.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    return x


def _decode_output_byte(y: torch.Tensor) -> int:
    lo = int(y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item())
    hi = int(y[0, 0, _SetDim.OUTPUT_HI:_SetDim.OUTPUT_HI + 16].argmax().item())
    return lo | (hi << 4)


@pytest.fixture(scope="module")
def composite() -> ALUAndOrXor:
    return ALUAndOrXor(S=100.0, BD=_SetDim)


@pytest.fixture(scope="module")
def lowered_ffns() -> dict:
    return {op: _lowered_pureffn_for(op) for op in ("and", "or", "xor")}


def test_bitwise_rules_emit_512_units_per_op():
    """Sanity: 256 lo + 256 hi = 512 rules per opcode (matches the
    legacy ``_layer10_alu_bitwise_rules`` shape).
    """
    for op in ("and", "or", "xor"):
        rules = _build_bitwise_rules_for(op)
        assert len(rules) == 512, (
            f"{op!r} emitted {len(rules)} rules, expected 512"
        )


@pytest.mark.parametrize("op", ["and", "or", "xor"])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (0x00, 0xFF),  # all-zero vs all-ones
        (0xA5, 0x5A),  # alternating bits
        (0xCC, 0x33),  # nibble-aligned
        (0x12, 0x34),  # arbitrary
        (0xDE, 0xAD),  # arbitrary
        (0xBE, 0xEF),  # arbitrary
    ],
)
def test_bitwise_rules_byte_identity_vs_composite(
    composite, lowered_ffns, op, a, b
):
    """Lowered ``bitwise_rules`` decodes to the same OUTPUT byte as
    ``ALUAndOrXor.forward`` on the same input.

    Both paths must agree with Python's reference op (``a OP b``) — the
    composite already has its byte-identity gate in
    ``test_alu_wide_composites_per_op``, so matching the composite is
    sufficient for the DSL POC contract.
    """
    x = _make_input(a=a, b=b, op=op)
    with torch.no_grad():
        y_composite = composite(x)
        y_lowered = lowered_ffns[op](x)

    composite_byte = _decode_output_byte(y_composite)
    lowered_byte = _decode_output_byte(y_lowered)
    expected = _PY_OP[op](a, b) & 0xFF

    assert composite_byte == expected, (
        f"composite drift: {op} 0x{a:02X} 0x{b:02X} "
        f"-> expected 0x{expected:02X}, got 0x{composite_byte:02X}"
    )
    assert lowered_byte == expected, (
        f"lowered drift: bitwise_rules({op!r}) 0x{a:02X} 0x{b:02X} "
        f"-> expected 0x{expected:02X}, got 0x{lowered_byte:02X}"
    )
    assert lowered_byte == composite_byte, (
        f"DSL byte-identity: bitwise_rules vs ALUAndOrXor disagree on "
        f"{op} 0x{a:02X} 0x{b:02X} (lowered=0x{lowered_byte:02X}, "
        f"composite=0x{composite_byte:02X})"
    )


def test_bitwise_rules_byte_identity_randomized(composite, lowered_ffns):
    """Randomized sweep — 32 (a, b) pairs per op, decoded OUTPUT byte
    must match the composite on every one. Seeded for determinism.
    """
    gen = torch.Generator().manual_seed(0x1B17C155)  # "BITWISE" hex-ish
    n_trials = 32
    pairs = torch.randint(0, 256, (n_trials, 2), generator=gen).tolist()

    mismatches = []
    for op in ("and", "or", "xor"):
        for a, b in pairs:
            x = _make_input(a=a, b=b, op=op)
            with torch.no_grad():
                lowered_byte = _decode_output_byte(lowered_ffns[op](x))
                composite_byte = _decode_output_byte(composite(x))
            expected = _PY_OP[op](a, b) & 0xFF
            if lowered_byte != expected or composite_byte != expected:
                mismatches.append(
                    f"{op} 0x{a:02X} 0x{b:02X}: "
                    f"expected=0x{expected:02X} "
                    f"lowered=0x{lowered_byte:02X} "
                    f"composite=0x{composite_byte:02X}"
                )

    assert not mismatches, (
        f"Byte-identity failures ({len(mismatches)}/{3 * n_trials}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# Wave W3: wide_add_rules byte-identity test (one byte / one nibble lane).
# ---------------------------------------------------------------------------


def _build_wide_add_rules_one_byte(S: float = 100.0):
    """Construct the 256-rule per-byte ADD lookup for width_bytes=1.

    Uses the same band-wiring convention as the bitwise POC: operand A on
    ``ALU_LO``, operand B on ``AX_CARRY_LO``, result on ``OUTPUT_LO``,
    carry on ``CARRY``. Opcode gate is ``OP_ADD``; marker is ``MARK_AX``.
    """
    return wide_add_rules(
        operand_a_base="ALU_LO",
        operand_b_base="AX_CARRY_LO",
        result_base="OUTPUT_LO",
        carry_base="CARRY",
        width_bytes=1,
        opcode_gate="OP_ADD",
        marker_gate="MARK_AX",
        S=S,
    )


def _lowered_pureffn_for_wide_add(S: float = 100.0) -> PureFFN:
    """Lower the width_bytes=1 wide_add rules into a PureFFN."""
    rules = _build_wide_add_rules_one_byte(S=S)
    assert len(rules) == 256, (
        f"wide_add_rules(width_bytes=1) emitted {len(rules)} rules, "
        f"expected 256"
    )
    # Size the FFN to hold all 256 lookup units. PureFFN's hidden_dim is
    # the unit budget; dim must match the residual width (512) so the
    # band dim positions resolve into a valid slot.
    ffn = PureFFN(dim=512, hidden_dim=256)
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == 256, f"lower_ffn_rules wrote {end} units, expected 256"
    return ffn


def _make_add_input(*, a_nib: int, b_nib: int) -> torch.Tensor:
    """One-position residual at MARK_AX with operand A/B nibbles + OP_ADD.

    Layout mirrors ``_make_input`` for bitwise: ALU_LO carries operand A
    nibble, AX_CARRY_LO carries operand B nibble. CARRY band is left
    zero (no incoming carry for byte 0).
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.OP_ADD] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a_nib & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b_nib & 0xF)] = 1.0
    return x


def _decode_add_output(y: torch.Tensor) -> tuple[int, int]:
    """Return ``(sum_nib, carry_out)`` decoded from the lowered FFN's
    residual output for byte 0.

    ``sum_nib`` is the argmax over ``OUTPUT_LO[0:16]``; ``carry_out`` is
    1 iff ``CARRY+0`` exceeds a small threshold (lookup-mode amplitude is
    ``2.0 / S``, which for S=100 is 0.02, well above zero).
    """
    sum_nib = int(
        y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item()
    )
    carry_out = int(y[0, 0, _SetDim.CARRY].item() > 0.005)
    return sum_nib, carry_out


@pytest.fixture(scope="module")
def lowered_add_ffn() -> PureFFN:
    return _lowered_pureffn_for_wide_add()


def test_wide_add_rules_emit_expected_unit_count():
    """Sanity: width_bytes=1 emits exactly 256 sum rules (no carry-in
    relay since byte 0 has no carry predecessor).
    """
    rules = _build_wide_add_rules_one_byte()
    assert len(rules) == 256


def test_wide_add_rules_byte_identity_one_byte(lowered_add_ffn):
    """Lowered ``wide_add_rules(width_bytes=1)`` matches Python's
    nibble-add for every (a, b) in 0..15 × 0..15.

    The DSL's per-byte abstraction treats each "byte" as one 4-bit
    nibble, so this is the byte-0 identity contract for an 8-bit add
    (byte 0 = low nibble of an 8-bit value). Concretely:

      * sum_nib_out == (a + b) & 0xF
      * carry_out   == ((a + b) >> 4) & 1
    """
    mismatches = []
    for a in range(16):
        for b in range(16):
            x = _make_add_input(a_nib=a, b_nib=b)
            with torch.no_grad():
                y = lowered_add_ffn(x)
            sum_nib, carry_out = _decode_add_output(y)

            expected_sum = (a + b) & 0xF
            expected_carry = ((a + b) >> 4) & 1
            if sum_nib != expected_sum or carry_out != expected_carry:
                mismatches.append(
                    f"a=0x{a:X} b=0x{b:X}: "
                    f"expected sum=0x{expected_sum:X} carry={expected_carry} "
                    f"got sum=0x{sum_nib:X} carry={carry_out}"
                )

    assert not mismatches, (
        f"wide_add byte-identity failures ({len(mismatches)}/256):\n  "
        + "\n  ".join(mismatches[:10])
    )
