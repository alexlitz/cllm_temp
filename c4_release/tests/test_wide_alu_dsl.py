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
    wide_mul_rules,
    wide_shift_rules,
    wide_sub_rules,
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


# ---------------------------------------------------------------------------
# Wave W3: wide_add_rules multi-byte byte-identity (width_bytes=4 = 16-bit,
# width_bytes=8 = 32-bit).
# ---------------------------------------------------------------------------
#
# Multi-byte semantics: each "byte" in this DSL slice is a single 4-bit
# nibble lane (see ``wide_add_rules`` docstring). So width_bytes=4 maps
# to a 16-bit add, width_bytes=8 to a 32-bit add. The helper emits
# per-nibble lookup units that read operand_a/b at offset (b*16 + nib)
# and write the sum at result_base + (b*16 + sum_nib). For byte > 0 the
# rule also reads carry_base + (b - 1) (positively at threshold 120 for
# cin=1, or with -50 weight for cin=0) and writes carry_base + b on
# carry-out.
#
# A single SwiGLU forward pass cannot self-cascade: W_up reads the
# input residual only, so a carry written via W_down to position
# carry_base+(b-1) won't be visible to byte b's lookup in the same
# pass. The wider architecture handles propagation across separate
# layers / passes. To exercise the per-byte rule semantics in a single
# pass, this test pre-injects the *expected* carry-in flags into the
# input residual alongside the operand one-hots. The decode then
# checks that every byte's sum nibble argmax matches the Python
# reference, end-to-end giving the full multi-nibble result.

# Ad-hoc dim layout for the wide_add multi-byte tests.
# Bands are sized for width_bytes <= 8 (8 nibble lanes = 32 bits is the
# largest case we exercise; we lay out 8 nibbles worth = 128 slots per band).
_ADD_MAX_BYTES = 8
_ADD_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_ADD_GATE": 1,
    "OPERAND_A": 2,                     # +k for k in 0..(16*B-1)
    "OPERAND_B": 2 + 16 * _ADD_MAX_BYTES,
    "RESULT":    2 + 32 * _ADD_MAX_BYTES,
    "CARRY":     2 + 48 * _ADD_MAX_BYTES,
}
_ADD_FFN_DIM = 2 + 48 * _ADD_MAX_BYTES + _ADD_MAX_BYTES  # +carry band


def _build_wide_add_rules_multi(width_bytes: int, S: float = 100.0):
    """Construct multi-byte wide_add rules for the ad-hoc dim layout."""
    return wide_add_rules(
        operand_a_base="OPERAND_A",
        operand_b_base="OPERAND_B",
        result_base="RESULT",
        carry_base="CARRY",
        width_bytes=width_bytes,
        opcode_gate="OP_ADD_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )


def _lowered_pureffn_for_wide_add_multi(
    width_bytes: int, S: float = 100.0
) -> PureFFN:
    """Lower the multi-byte wide_add rules into a PureFFN."""
    rules = _build_wide_add_rules_multi(width_bytes, S=S)
    # Per docstring: 256 for byte 0 + (512 + 1) per subsequent byte.
    expected_count = 256 + (width_bytes - 1) * (512 + 1)
    assert len(rules) == expected_count, (
        f"wide_add_rules(width_bytes={width_bytes}) emitted {len(rules)} "
        f"rules, expected {expected_count}"
    )
    ffn = PureFFN(dim=_ADD_FFN_DIM, hidden_dim=expected_count)
    end = Primitives.lower_ffn_rules(
        ffn, rules, _ADD_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == expected_count, (
        f"lower_ffn_rules wrote {end} units, expected {expected_count}"
    )
    return ffn


def _per_nibble(value: int, width_bytes: int) -> list[int]:
    """Split a value into ``width_bytes`` little-endian 4-bit nibbles."""
    return [(value >> (4 * b)) & 0xF for b in range(width_bytes)]


def _make_wide_add_input(
    *, a: int, b: int, width_bytes: int
) -> torch.Tensor:
    """Build a one-position residual carrying a multi-nibble (a, b) ADD
    plus the pre-computed carry-in one-hots needed for each nibble > 0.

    Operand A nibbles populate ``OPERAND_A+(b*16 + nib)``; operand B
    similarly. Carry-in flags ``CARRY+b`` for ``b in 0..width-2`` are set
    if the partial sum at nibble ``b`` carries (precomputed in Python so
    the single-pass FFN can resolve each nibble's lookup without needing
    a multi-pass cascade).
    """
    x = torch.zeros(1, 1, _ADD_FFN_DIM)
    x[0, 0, _ADD_DIM_LAYOUT["MARK_GATE"]] = 1.0
    x[0, 0, _ADD_DIM_LAYOUT["OP_ADD_GATE"]] = 1.0
    a_nibs = _per_nibble(a, width_bytes)
    b_nibs = _per_nibble(b, width_bytes)
    for bi, (an, bn) in enumerate(zip(a_nibs, b_nibs)):
        x[0, 0, _ADD_DIM_LAYOUT["OPERAND_A"] + bi * 16 + an] = 1.0
        x[0, 0, _ADD_DIM_LAYOUT["OPERAND_B"] + bi * 16 + bn] = 1.0
    # Precompute carry chain for the input residual so each nibble's
    # lookup sees the expected carry-in via CARRY+(b-1).
    carry = 0
    for bi in range(width_bytes - 1):
        carry = 1 if (a_nibs[bi] + b_nibs[bi] + carry) >= 16 else 0
        if carry:
            x[0, 0, _ADD_DIM_LAYOUT["CARRY"] + bi] = 1.0
    return x


def _decode_wide_add_result(y: torch.Tensor, width_bytes: int) -> int:
    """Reassemble the multi-nibble result by argmax over each nibble's
    RESULT lane and packing nibbles little-endian.
    """
    base = _ADD_DIM_LAYOUT["RESULT"]
    value = 0
    for bi in range(width_bytes):
        lane = y[0, 0, base + bi * 16:base + bi * 16 + 16]
        value |= int(lane.argmax().item()) << (4 * bi)
    return value


def test_wide_add_rules_emit_expected_multi_byte_count():
    """Sanity: rule count formula 256 + (W - 1) * 513 for width_bytes 1..4."""
    for w in (1, 2, 4, 8):
        rules = _build_wide_add_rules_multi(w)
        assert len(rules) == 256 + (w - 1) * (512 + 1), (
            f"width_bytes={w}: emitted {len(rules)} rules, "
            f"expected {256 + (w - 1) * (512 + 1)}"
        )


def test_wide_add_rules_rejects_bad_width_bytes():
    """``width_bytes < 1`` is a ValueError."""
    with pytest.raises(ValueError, match="width_bytes"):
        wide_add_rules(
            operand_a_base="OPERAND_A",
            operand_b_base="OPERAND_B",
            result_base="RESULT",
            carry_base="CARRY",
            width_bytes=0,
            opcode_gate="OP_ADD_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )


@pytest.fixture(scope="module")
def lowered_add_ffn_16bit() -> PureFFN:
    # width_bytes=4 → 4 nibbles → 16 bits
    return _lowered_pureffn_for_wide_add_multi(4)


@pytest.fixture(scope="module")
def lowered_add_ffn_32bit() -> PureFFN:
    # width_bytes=8 → 8 nibbles → 32 bits
    return _lowered_pureffn_for_wide_add_multi(8)


def test_wide_add_rules_byte_identity_16bit(lowered_add_ffn_16bit):
    """Randomised sweep: width_bytes=4 (16-bit, 4 nibble lanes) decoded
    result equals ``(a + b) & 0xFFFF`` for every sampled pair.

    The carry-in flags are pre-injected per ``_make_wide_add_input`` so
    each nibble's lookup sees its expected carry-in. This validates the
    *per-nibble rule semantics* (carry suppression for cin=0, positive
    carry-in activation for cin=1) and end-to-end nibble layout — the
    inter-nibble cascade across layers is a property of the wider
    pipeline.
    """
    gen = torch.Generator().manual_seed(0x16B17ADD)  # "16-bit add"
    n_trials = 64
    pairs = torch.randint(0, 0x10000, (n_trials, 2), generator=gen).tolist()
    # Boundary cases that force a full-width carry chain.
    pairs += [
        (0x0000, 0x0000),
        (0xFFFF, 0x0001),  # carry through every nibble
        (0xFFFF, 0xFFFF),  # full chain + overflow drop
        (0x0FFF, 0x0001),  # mid-width carry boundary
        (0xAAAA, 0x5555),  # interleaved bits, no carry
    ]

    mismatches = []
    for a, b in pairs:
        x = _make_wide_add_input(a=a, b=b, width_bytes=4)
        with torch.no_grad():
            y = lowered_add_ffn_16bit(x)
        decoded = _decode_wide_add_result(y, width_bytes=4)
        expected = (a + b) & 0xFFFF
        if decoded != expected:
            mismatches.append(
                f"a=0x{a:04X} b=0x{b:04X}: "
                f"expected=0x{expected:04X} got=0x{decoded:04X}"
            )

    assert not mismatches, (
        f"wide_add width=4 (16-bit) byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


def test_wide_add_rules_byte_identity_32bit(lowered_add_ffn_32bit):
    """Randomised sweep: width_bytes=8 (32-bit, 8 nibble lanes) decoded
    result equals ``(a + b) & 0xFFFFFFFF`` for every sampled pair.
    """
    gen = torch.Generator().manual_seed(0x32B17ADD)  # "32-bit add"
    n_trials = 64
    pairs = torch.randint(
        0, 2**31, (n_trials, 2), generator=gen
    ).tolist()
    # Add some boundary cases that force long carry chains.
    pairs += [
        (0x00000000, 0x00000000),
        (0xFFFFFFFF, 0x00000001),  # carry through all 8 nibbles
        (0xFFFFFFFF, 0xFFFFFFFF),  # full chain plus overflow drop
        (0x0FFF0FFF, 0x00010001),  # mid-width carry boundary
        (0xAAAAAAAA, 0x55555555),  # interleaved bits, no carry
    ]

    mismatches = []
    for a, b in pairs:
        x = _make_wide_add_input(a=a, b=b, width_bytes=8)
        with torch.no_grad():
            y = lowered_add_ffn_32bit(x)
        decoded = _decode_wide_add_result(y, width_bytes=8)
        expected = (a + b) & 0xFFFFFFFF
        if decoded != expected:
            mismatches.append(
                f"a=0x{a:08X} b=0x{b:08X}: "
                f"expected=0x{expected:08X} got=0x{decoded:08X}"
            )

    assert not mismatches, (
        f"wide_add width=8 (32-bit) byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# Wave W2: wide_shift_rules byte-identity (8-bit, width_bytes=1).
# ---------------------------------------------------------------------------
#
# The shift helper uses a 256-wide-per-byte one-hot operand band — distinct
# from the residual layout consumed by ALUShiftComposite (which splits the
# operand into 16-wide LO + 16-wide HI nibble bands). To validate the rule
# semantics independently of any existing dim allocation, the test builds an
# ad-hoc ``dim_positions`` mapping and a fresh ``PureFFN`` sized to cover
# all positions the rules reference. The reference is Python's native shift,
# which is the contract documented in ``wide_shift_rules``.


# Ad-hoc dim layout for the 8-bit shift test. ``operand_base`` and
# ``result_base`` are 256-wide one-hot bands per byte position; with
# ``width_bytes=1`` each consumes 256 slots. ``SHIFT_AMT`` is 8-wide.
# ``MARK_GATE`` and the opcode gates each take one slot.
_SHIFT_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_SHL_GATE": 1,
    "OP_SHR_GATE": 2,
    "SHIFT_AMT": 3,            # SHIFT_AMT+k for k in 0..7 → slots 3..10
    "OPERAND": 11,             # OPERAND+N for N in 0..255 → slots 11..266
    "RESULT": 267,             # RESULT+R for R in 0..255 → slots 267..522
}
_SHIFT_FFN_DIM = 523


def _lowered_wide_shift_ffn(direction: str, S: float = 100.0) -> PureFFN:
    """Build a ``PureFFN`` lowered from ``wide_shift_rules`` for one direction.

    ``width_bytes=1`` (an 8-bit shift). The FFN dim is sized to exactly
    cover the ad-hoc dim layout above.
    """
    rules = wide_shift_rules(
        direction=direction,
        operand_base="OPERAND",
        shift_amount_dim="SHIFT_AMT",
        result_base="RESULT",
        width_bytes=1,
        opcode_gate="OP_SHL_GATE" if direction == "left" else "OP_SHR_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )
    expected_count = 1 * 8 * 256
    assert len(rules) == expected_count, (
        f"wide_shift_rules({direction!r}) emitted {len(rules)} rules, "
        f"expected {expected_count}"
    )
    ffn = PureFFN(dim=_SHIFT_FFN_DIM, hidden_dim=expected_count)
    end = Primitives.lower_ffn_rules(
        ffn, rules, _SHIFT_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == expected_count, (
        f"lower_ffn_rules wrote {end} units, expected {expected_count}"
    )
    return ffn


def _make_shift_input(*, n: int, k: int, direction: str) -> torch.Tensor:
    """Build a one-position residual with the marker, opcode, shift, and
    operand one-hots set per the ad-hoc layout.
    """
    x = torch.zeros(1, 1, _SHIFT_FFN_DIM)
    x[0, 0, _SHIFT_DIM_LAYOUT["MARK_GATE"]] = 1.0
    gate_dim = "OP_SHL_GATE" if direction == "left" else "OP_SHR_GATE"
    x[0, 0, _SHIFT_DIM_LAYOUT[gate_dim]] = 1.0
    x[0, 0, _SHIFT_DIM_LAYOUT["SHIFT_AMT"] + k] = 1.0
    x[0, 0, _SHIFT_DIM_LAYOUT["OPERAND"] + n] = 1.0
    return x


def _decode_shift_result_byte(y: torch.Tensor) -> int:
    """argmax over the 256-wide RESULT band → byte value."""
    base = _SHIFT_DIM_LAYOUT["RESULT"]
    return int(y[0, 0, base:base + 256].argmax().item())


def test_wide_shift_rules_emits_expected_count():
    """Sanity: width_bytes * 8 shift amounts * 256 byte values per direction."""
    for direction in ("left", "right"):
        rules = wide_shift_rules(
            direction=direction,
            operand_base="OPERAND",
            shift_amount_dim="SHIFT_AMT",
            result_base="RESULT",
            width_bytes=1,
            opcode_gate=(
                "OP_SHL_GATE" if direction == "left" else "OP_SHR_GATE"
            ),
            marker_gate="MARK_GATE",
            S=100.0,
        )
        assert len(rules) == 8 * 256, (
            f"{direction}: emitted {len(rules)} rules, expected 2048"
        )

    # width_bytes=4 → 4 * 8 * 256 = 8192 rules.
    rules_wide = wide_shift_rules(
        direction="left",
        operand_base="OPERAND",
        shift_amount_dim="SHIFT_AMT",
        result_base="RESULT",
        width_bytes=4,
        opcode_gate="OP_SHL_GATE",
        marker_gate="MARK_GATE",
        S=100.0,
    )
    assert len(rules_wide) == 4 * 8 * 256


def test_wide_shift_rules_rejects_bad_args():
    """Validation errors for unsupported direction / width_bytes."""
    with pytest.raises(ValueError, match="direction"):
        wide_shift_rules(
            direction="middle",  # type: ignore[arg-type]
            operand_base="OPERAND",
            shift_amount_dim="SHIFT_AMT",
            result_base="RESULT",
            width_bytes=1,
            opcode_gate="OP_SHL_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )
    with pytest.raises(ValueError, match="width_bytes"):
        wide_shift_rules(
            direction="left",
            operand_base="OPERAND",
            shift_amount_dim="SHIFT_AMT",
            result_base="RESULT",
            width_bytes=0,
            opcode_gate="OP_SHL_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )


@pytest.fixture(scope="module")
def lowered_shift_ffns() -> dict:
    return {
        "left": _lowered_wide_shift_ffn("left"),
        "right": _lowered_wide_shift_ffn("right"),
    }


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize(
    ("n", "k"),
    [
        (0x00, 0),  # zero operand
        (0xFF, 0),  # zero shift
        (0x01, 4),  # left: 0x10, right: 0x00
        (0x0F, 4),  # left: 0xF0, right: 0x00
        (0xF0, 4),  # left: 0x00, right: 0x0F
        (0x80, 3),  # left: 0xC0 (drop high), right: 0x10
        (0x42, 1),  # arbitrary
        (0xAA, 7),  # max shift
        (0x55, 7),
        (0xC3, 2),
    ],
)
def test_wide_shift_rules_byte_identity_8bit(
    lowered_shift_ffns, direction, n, k
):
    """The lowered ``wide_shift_rules`` FFN reproduces the Python reference
    shift bit-for-bit on the RESULT band (argmax decode).
    """
    x = _make_shift_input(n=n, k=k, direction=direction)
    with torch.no_grad():
        y = lowered_shift_ffns[direction](x)
    decoded = _decode_shift_result_byte(y)

    if direction == "left":
        expected = (n << k) & 0xFF
    else:
        expected = (n >> k) & 0xFF

    assert decoded == expected, (
        f"wide_shift_rules({direction!r}) byte-identity drift: "
        f"n=0x{n:02X} k={k} -> expected=0x{expected:02X}, "
        f"got=0x{decoded:02X}"
    )


def test_wide_shift_rules_byte_identity_randomized(lowered_shift_ffns):
    """Randomized sweep: 24 (n, k) pairs per direction. Argmax-decoded
    RESULT byte must match the Python reference on every one. Seeded for
    determinism.
    """
    gen = torch.Generator().manual_seed(0x5417C757)  # "SHIFTS" hex-ish
    n_trials = 24
    ns = torch.randint(0, 256, (n_trials,), generator=gen).tolist()
    ks = torch.randint(0, 8, (n_trials,), generator=gen).tolist()

    mismatches = []
    for direction in ("left", "right"):
        for n, k in zip(ns, ks):
            x = _make_shift_input(n=n, k=k, direction=direction)
            with torch.no_grad():
                decoded = _decode_shift_result_byte(
                    lowered_shift_ffns[direction](x)
                )
            if direction == "left":
                expected = (n << k) & 0xFF
            else:
                expected = (n >> k) & 0xFF
            if decoded != expected:
                mismatches.append(
                    f"{direction} n=0x{n:02X} k={k}: "
                    f"expected=0x{expected:02X} got=0x{decoded:02X}"
                )

    assert not mismatches, (
        f"Byte-identity failures ({len(mismatches)}/{2 * n_trials}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# Wave W3: wide_sub_rules byte-identity test (one byte / one nibble lane).
# ---------------------------------------------------------------------------


def _build_wide_sub_rules_one_byte(S: float = 100.0):
    """Construct the 256-rule per-byte SUB lookup for width_bytes=1."""
    return wide_sub_rules(
        operand_a_base="ALU_LO",
        operand_b_base="AX_CARRY_LO",
        result_base="OUTPUT_LO",
        borrow_base="CARRY",
        width_bytes=1,
        opcode_gate="OP_SUB",
        marker_gate="MARK_AX",
        S=S,
    )


# ---------------------------------------------------------------------------
# Wave W5: wide_mul_rules byte-identity (8-bit POC, width_bytes=1).
# ---------------------------------------------------------------------------

_MUL_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_MUL_GATE": 1,
    "OPERAND_A": 2,
    "OPERAND_B": 18,
    "RESULT": 34,
}
_MUL_FFN_DIM = 66


def _build_wide_mul_rules_8bit(S: float = 100.0):
    """Construct the 256-rule wide_mul lookup for width_bytes=1."""
    return wide_mul_rules(
        operand_a_base="OPERAND_A",
        operand_b_base="OPERAND_B",
        result_base="RESULT",
        width_bytes=1,
        opcode_gate="OP_MUL_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )


def _lowered_pureffn_for_wide_sub(S: float = 100.0) -> PureFFN:
    """Lower the width_bytes=1 wide_sub rules into a PureFFN."""
    rules = _build_wide_sub_rules_one_byte(S=S)
    assert len(rules) == 256, (
        f"wide_sub_rules(width_bytes=1) emitted {len(rules)} rules, "
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


def _lowered_pureffn_for_wide_mul(S: float = 100.0) -> PureFFN:
    """Lower the width_bytes=1 wide_mul rules into a PureFFN."""
    rules = _build_wide_mul_rules_8bit(S=S)
    assert len(rules) == 256, (
        f"wide_mul_rules(width_bytes=1) emitted {len(rules)} rules, "
        f"expected 256"
    )
    ffn = PureFFN(dim=_MUL_FFN_DIM, hidden_dim=256)
    end = Primitives.lower_ffn_rules(
        ffn, rules, _MUL_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == 256, f"lower_ffn_rules wrote {end} units, expected 256"
    return ffn


def _make_sub_input(*, a_nib: int, b_nib: int) -> torch.Tensor:
    """One-position residual at MARK_AX with operand A/B nibbles + OP_SUB.

    Layout mirrors ``_make_add_input``: ALU_LO carries operand A (minuend)
    nibble, AX_CARRY_LO carries operand B (subtrahend) nibble. CARRY
    (borrow_base) is left zero — no incoming borrow for byte 0.
    """
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, _SetDim.CONST] = 1.0
    x[0, 0, _SetDim.OP_SUB] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a_nib & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b_nib & 0xF)] = 1.0
    return x


def _decode_sub_output(y: torch.Tensor) -> tuple[int, int]:
    """Return ``(diff_nib, borrow_out)`` decoded from the lowered FFN's
    residual output for byte 0.

    ``diff_nib`` is the argmax over ``OUTPUT_LO[0:16]``; ``borrow_out``
    is 1 iff the borrow dim at byte 0 (``CARRY+0`` = ``CARRY``) exceeds a
    small threshold (lookup-mode amplitude is ``2.0 / S``, 0.02 for
    S=100, well above zero).
    """
    diff_nib = int(
        y[0, 0, _SetDim.OUTPUT_LO:_SetDim.OUTPUT_LO + 16].argmax().item()
    )
    borrow_out = int(y[0, 0, _SetDim.CARRY].item() > 0.005)
    return diff_nib, borrow_out


@pytest.fixture(scope="module")
def lowered_sub_ffn() -> PureFFN:
    return _lowered_pureffn_for_wide_sub()


def test_wide_sub_rules_emit_expected_unit_count():
    """Sanity: width_bytes=1 emits exactly 256 diff rules (no borrow-in
    relay since byte 0 has no borrow predecessor).
    """
    rules = _build_wide_sub_rules_one_byte()
    assert len(rules) == 256


def test_wide_sub_rules_byte_identity_one_byte(lowered_sub_ffn):
    """Lowered ``wide_sub_rules(width_bytes=1)`` matches Python's
    nibble-sub for every (a, b) in 0..15 × 0..15.

    The DSL's per-byte abstraction treats each "byte" as one 4-bit
    nibble, so this is the byte-0 identity contract for an 8-bit sub
    (byte 0 = low nibble of an 8-bit value). Concretely:

      * diff_nib_out == (a - b) & 0xF
      * borrow_out   == 1 iff a < b   (i.e. (a - b) < 0)
    """
    mismatches = []
    for a in range(16):
        for b in range(16):
            x = _make_sub_input(a_nib=a, b_nib=b)
            with torch.no_grad():
                y = lowered_sub_ffn(x)
            diff_nib, borrow_out = _decode_sub_output(y)

            expected_diff = (a - b) & 0xF
            expected_borrow = 1 if a < b else 0
            if diff_nib != expected_diff or borrow_out != expected_borrow:
                mismatches.append(
                    f"a=0x{a:X} b=0x{b:X}: "
                    f"expected diff=0x{expected_diff:X} "
                    f"borrow={expected_borrow} "
                    f"got diff=0x{diff_nib:X} borrow={borrow_out}"
                )

    assert not mismatches, (
        f"wide_sub byte-identity failures ({len(mismatches)}/256):\n  "
        + "\n  ".join(mismatches[:10])
    )


def _make_mul_input(*, a_nib: int, b_nib: int) -> torch.Tensor:
    """One-position residual with marker, opcode, operand A/B one-hots."""
    x = torch.zeros(1, 1, _MUL_FFN_DIM)
    x[0, 0, _MUL_DIM_LAYOUT["MARK_GATE"]] = 1.0
    x[0, 0, _MUL_DIM_LAYOUT["OP_MUL_GATE"]] = 1.0
    x[0, 0, _MUL_DIM_LAYOUT["OPERAND_A"] + (a_nib & 0xF)] = 1.0
    x[0, 0, _MUL_DIM_LAYOUT["OPERAND_B"] + (b_nib & 0xF)] = 1.0
    return x


def _decode_mul_output(y: torch.Tensor) -> int:
    base = _MUL_DIM_LAYOUT["RESULT"]
    lo = int(y[0, 0, base:base + 16].argmax().item())
    hi = int(y[0, 0, base + 16:base + 32].argmax().item())
    return lo | (hi << 4)


@pytest.fixture(scope="module")
def lowered_mul_ffn() -> PureFFN:
    return _lowered_pureffn_for_wide_mul()


def test_wide_mul_rules_emit_expected_unit_count():
    rules = _build_wide_mul_rules_8bit()
    assert len(rules) == 256


def test_wide_mul_rules_rejects_bad_args():
    with pytest.raises(ValueError, match="width_bytes"):
        wide_mul_rules(
            operand_a_base="OPERAND_A",
            operand_b_base="OPERAND_B",
            result_base="RESULT",
            width_bytes=0,
            opcode_gate="OP_MUL_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )
    with pytest.raises(NotImplementedError, match="multi-byte"):
        wide_mul_rules(
            operand_a_base="OPERAND_A",
            operand_b_base="OPERAND_B",
            result_base="RESULT",
            width_bytes=2,
            opcode_gate="OP_MUL_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )


def test_wide_mul_rules_byte_identity_8bit(lowered_mul_ffn):
    """Lowered wide_mul_rules matches Python's 4-bit mul."""
    mismatches = []
    for a in range(16):
        for b in range(16):
            x = _make_mul_input(a_nib=a, b_nib=b)
            with torch.no_grad():
                y = lowered_mul_ffn(x)
            decoded = _decode_mul_output(y)
            expected = (a * b) & 0xFF
            if decoded != expected:
                mismatches.append(
                    f"a=0x{a:X} b=0x{b:X}: "
                    f"expected=0x{expected:02X} got=0x{decoded:02X}"
                )

    assert not mismatches, (
        f"wide_mul byte-identity failures ({len(mismatches)}/256):\n  "
        + "\n  ".join(mismatches[:10])
    )
