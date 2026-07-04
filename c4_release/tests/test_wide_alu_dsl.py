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
from neural_vm.unified_compiler.primitives import Primitives  # noqa: E402
from neural_vm.unified_compiler.wide_alu_dsl import (  # noqa: E402
    bitwise_rules,
    wide_add_rules,
    wide_div_rules_ge_format,
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
def test_bitwise_rules_byte_identity_vs_python(lowered_ffns, op, a, b):
    """Lowered ``bitwise_rules`` decodes to the same OUTPUT byte as
    Python's reference op (``a OP b``) on the same input.

    Post-V8: the legacy ``ALUAndOrXor`` composite was deleted (it was
    the imperative composite the rules replaced); the contract now
    asserted is ``lowered_byte == (a OP b) & 0xFF`` directly. Lookup-mode
    install matches this byte-for-byte via the factory in
    ``ops/alu_ops.py:make_lookup_mode_l10_bitwise_rules_op`` (separately
    sanity-checked by
    ``test_lookup_mode_l10_postop_factory_byte_identity``).
    """
    x = _make_input(a=a, b=b, op=op)
    with torch.no_grad():
        y_lowered = lowered_ffns[op](x)

    lowered_byte = _decode_output_byte(y_lowered)
    expected = _PY_OP[op](a, b) & 0xFF

    assert lowered_byte == expected, (
        f"lowered drift: bitwise_rules({op!r}) 0x{a:02X} 0x{b:02X} "
        f"-> expected 0x{expected:02X}, got 0x{lowered_byte:02X}"
    )


def test_bitwise_rules_byte_identity_randomized(lowered_ffns):
    """Randomized sweep — 32 (a, b) pairs per op, decoded OUTPUT byte
    must match Python's reference on every one. Seeded for determinism.
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
            expected = _PY_OP[op](a, b) & 0xFF
            if lowered_byte != expected:
                mismatches.append(
                    f"{op} 0x{a:02X} 0x{b:02X}: "
                    f"expected=0x{expected:02X} "
                    f"lowered=0x{lowered_byte:02X}"
                )

    assert not mismatches, (
        f"Byte-identity failures ({len(mismatches)}/{3 * n_trials}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# V8 follow-up: lookup-mode factory byte-identity test.
#
# Validates the new ``make_lookup_mode_l10_bitwise_rules_op`` install path
# against the legacy ``_make_alu_postop_attach_op`` install (which inserts an
# ``ALUAndOrXor`` composite). The new factory's bake function emits a
# ``PureFFN`` baked from 1,536 ``bitwise_rules`` (3 opcodes x 512 rules) and
# inserts it into ``block.post_ops[0]`` — the same slot the old install used.
# Decoded OUTPUT_LO/HI byte must match the composite forward on every
# (a, b, op) tuple.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lookup_mode_rule_postop():
    """Bake the new lookup-mode L10 rule-derived post_op into a mock block.

    Returns the ``PureFFN`` instance installed at ``block.post_ops[0]``.
    """
    import torch.nn as nn

    from neural_vm.unified_compiler.ops.alu_ops import (
        make_lookup_mode_l10_bitwise_rules_op,
    )
    from neural_vm.unified_compiler.ops.shared import _setdim_to_positions

    class _MockBlock:
        def __init__(self):
            self.ffn = PureFFN(dim=512, hidden_dim=64)
            self.post_ops = nn.ModuleList()

    block = _MockBlock()
    op = make_lookup_mode_l10_bitwise_rules_op()
    op.bake_fn(block, _setdim_to_positions(_SetDim), 100.0)
    assert len(block.post_ops) == 1, (
        f"factory bake expected 1 post_op; got {len(block.post_ops)}"
    )
    rule_ffn = block.post_ops[0]
    assert type(rule_ffn) is PureFFN, (
        f"factory bake expected vanilla PureFFN; got {type(rule_ffn).__name__}"
    )
    assert rule_ffn.hidden_dim == 1536, (
        f"factory bake expected hidden_dim=1536; got {rule_ffn.hidden_dim}"
    )
    return rule_ffn


def test_lookup_mode_l10_postop_factory_byte_identity(
    lookup_mode_rule_postop
):
    """V8 follow-up: factory-installed PureFFN matches Python's bitwise op.

    Same sweep as ``test_bitwise_rules_byte_identity_randomized`` but
    exercises the production install path (``bake_fn`` mutating
    ``block.post_ops``) rather than building the PureFFN directly. Ensures
    the dim-positions proxy threading + ``Primitives.lower_ffn_rules`` call
    inside the factory bake produce a PureFFN whose decoded OUTPUT byte
    matches Python's reference op (= the post-V8 byte-identity contract,
    since ``ALUAndOrXor`` was deleted in this same wave).
    """
    gen = torch.Generator().manual_seed(0x1B17C155)
    n_trials = 32
    pairs = torch.randint(0, 256, (n_trials, 2), generator=gen).tolist()

    mismatches = []
    for op in ("and", "or", "xor"):
        for a, b in pairs:
            x = _make_input(a=a, b=b, op=op)
            with torch.no_grad():
                rule_byte = _decode_output_byte(lookup_mode_rule_postop(x))
            expected = _PY_OP[op](a, b) & 0xFF
            if rule_byte != expected:
                mismatches.append(
                    f"{op} 0x{a:02X} 0x{b:02X}: expected=0x{expected:02X} "
                    f"rule=0x{rule_byte:02X}"
                )

    assert not mismatches, (
        f"Lookup-mode factory drift ({len(mismatches)}/{3 * n_trials}):\n  "
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
    # width_bytes=2 is SUPPORTED (8-bit x 8-bit flat lookup, 16-bit result);
    # see ``test_wide_mul_rules_byte_identity_16bit``. Only width_bytes > 2 is
    # deferred (the flat cross-product would emit 16 ** (2 * w) rules).
    with pytest.raises(NotImplementedError, match="partial-product"):
        wide_mul_rules(
            operand_a_base="OPERAND_A",
            operand_b_base="OPERAND_B",
            result_base="RESULT",
            width_bytes=3,
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


# ---------------------------------------------------------------------------
# Wave W5 (width=2): wide_mul_rules 8-bit x 8-bit -> 16-bit byte-identity.
#
# This is the correctness gate for the ``mul_overflow`` (100 * 5 = 500)
# fix. The width_bytes=2 helper emits 65,536 rules (one per
# (a_lo, a_hi, b_lo, b_hi) quad), each writing the full 16-bit product
# across four nibble lanes (result_base + lane*16 + nib). The decoded
# result must reproduce ``(a * b) & 0xFFFF`` for every sampled operand
# pair, including the canonical 100 * 5 = 0x01F4 overflow case where the
# product needs the byte-1 (high) result nibbles.
#
# Ad-hoc dim layout: operand bands are 32-wide (2 nibble lanes each);
# the result band is 64-wide (4 nibble lanes). No model build — pure
# rule-lowered PureFFN, same as the 8-bit POC above.
# ---------------------------------------------------------------------------

_MUL16_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_MUL_GATE": 1,
    "OPERAND_A": 2,        # +k for k in 0..31 (2 nibble lanes)
    "OPERAND_B": 34,       # +k for k in 0..31
    "RESULT": 66,          # +k for k in 0..63 (4 nibble lanes)
}
_MUL16_FFN_DIM = 66 + 64


def _build_wide_mul_rules_16bit(S: float = 100.0):
    """Construct the 65,536-rule wide_mul lookup for width_bytes=2."""
    return wide_mul_rules(
        operand_a_base="OPERAND_A",
        operand_b_base="OPERAND_B",
        result_base="RESULT",
        width_bytes=2,
        opcode_gate="OP_MUL_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )


def _lowered_pureffn_for_wide_mul_16bit(S: float = 100.0) -> PureFFN:
    """Lower the width_bytes=2 wide_mul rules into a PureFFN."""
    rules = _build_wide_mul_rules_16bit(S=S)
    assert len(rules) == 65536, (
        f"wide_mul_rules(width_bytes=2) emitted {len(rules)} rules, "
        f"expected 65536"
    )
    ffn = PureFFN(dim=_MUL16_FFN_DIM, hidden_dim=len(rules))
    end = Primitives.lower_ffn_rules(
        ffn, rules, _MUL16_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == 65536, f"lower_ffn_rules wrote {end} units, expected 65536"
    return ffn


def _make_mul16_input(*, a: int, b: int) -> torch.Tensor:
    """One-position residual with marker, opcode, and both operand bytes
    split into low/high nibble one-hots (2 nibble lanes per operand)."""
    x = torch.zeros(1, 1, _MUL16_FFN_DIM)
    x[0, 0, _MUL16_DIM_LAYOUT["MARK_GATE"]] = 1.0
    x[0, 0, _MUL16_DIM_LAYOUT["OP_MUL_GATE"]] = 1.0
    x[0, 0, _MUL16_DIM_LAYOUT["OPERAND_A"] + (a & 0xF)] = 1.0
    x[0, 0, _MUL16_DIM_LAYOUT["OPERAND_A"] + 16 + ((a >> 4) & 0xF)] = 1.0
    x[0, 0, _MUL16_DIM_LAYOUT["OPERAND_B"] + (b & 0xF)] = 1.0
    x[0, 0, _MUL16_DIM_LAYOUT["OPERAND_B"] + 16 + ((b >> 4) & 0xF)] = 1.0
    return x


def _decode_mul16_output(y: torch.Tensor) -> int:
    """Reassemble the 16-bit product by argmax over each of the 4 nibble
    lanes, packed little-endian."""
    base = _MUL16_DIM_LAYOUT["RESULT"]
    value = 0
    for lane in range(4):
        nib = int(y[0, 0, base + lane * 16:base + lane * 16 + 16].argmax().item())
        value |= nib << (4 * lane)
    return value


def test_wide_mul_rules_emit_expected_unit_count_16bit():
    """Sanity: width_bytes=2 emits exactly 65,536 rules (256 x 256)."""
    rules = _build_wide_mul_rules_16bit()
    assert len(rules) == 65536


@pytest.fixture(scope="module")
def lowered_mul_ffn_16bit() -> PureFFN:
    return _lowered_pureffn_for_wide_mul_16bit()


def test_wide_mul_rules_byte_identity_16bit(lowered_mul_ffn_16bit):
    """Lowered ``wide_mul_rules(width_bytes=2)`` decoded result equals
    ``(a * b) & 0xFFFF`` for every sampled (a, b) pair.

    This is the byte-identity correctness gate for the ``mul_overflow``
    (100 * 5 = 500) fix: 500 = 0x01F4 requires the byte-1 high result
    nibbles (nib2 = 0x1). The boundary cases force the high-byte lanes
    and the full 16-bit product range.
    """
    gen = torch.Generator().manual_seed(0x16B17005)  # "16-bit mul"
    n_trials = 200
    pairs = torch.randint(0, 256, (n_trials, 2), generator=gen).tolist()
    # Boundary / load-bearing cases.
    pairs += [
        (100, 5),    # mul_overflow: 500 = 0x01F4 (byte 1 = 0x01)
        (6, 7),      # mul_basic: 42 = 0x002A (byte 1 = 0x00)
        (0, 0),
        (255, 255),  # 0xFE01 — max product, all 4 nibbles non-trivial
        (1, 255),
        (255, 1),
        (16, 16),    # 0x0100 — byte-1-only product
        (170, 85),   # 0x3872 — interleaved bits
        (128, 2),    # 0x0100 — high-byte carry boundary
        (15, 15),    # 0x00E1 — low-byte only
    ]

    mismatches = []
    for a, b in pairs:
        x = _make_mul16_input(a=a, b=b)
        with torch.no_grad():
            y = lowered_mul_ffn_16bit(x)
        decoded = _decode_mul16_output(y)
        expected = (a * b) & 0xFFFF
        if decoded != expected:
            mismatches.append(
                f"a=0x{a:02X} b=0x{b:02X}: "
                f"expected=0x{expected:04X} got=0x{decoded:04X}"
            )

    assert not mismatches, (
        f"wide_mul width=2 (8-bit x 8-bit -> 16-bit) byte-identity "
        f"failures ({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# Wave W3: wide_sub_rules multi-byte byte-identity (width_bytes=2 = 8-bit
# value at 2 nibble lanes, width_bytes=4 = 16-bit at 4 nibble lanes).
# ---------------------------------------------------------------------------
#
# Same single-pass cascade limitation as the wide_add multi-byte tests:
# the FFN cannot self-feed a borrow written this pass back into a higher
# nibble's lookup. We pre-inject the expected borrow-in flags into the
# input residual so each nibble's rule sees the borrow chain it needs,
# isolating the per-rule semantics from the cross-pass cascade (the
# wider pipeline handles propagation across layers).
#
# Layout reuses the same ad-hoc band layout as the add multi-byte tests:
# nibble lane ``bi`` lives at ``OPERAND_*+bi*16+nib`` and the borrow band
# at ``CARRY+bi`` (here repurposed as the borrow_base for SUB).

_SUB_MAX_BYTES = _ADD_MAX_BYTES
_SUB_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_SUB_GATE": 1,
    "OPERAND_A": 2,
    "OPERAND_B": 2 + 16 * _SUB_MAX_BYTES,
    "RESULT":    2 + 32 * _SUB_MAX_BYTES,
    "BORROW":    2 + 48 * _SUB_MAX_BYTES,
}
_SUB_FFN_DIM = 2 + 48 * _SUB_MAX_BYTES + _SUB_MAX_BYTES


def _build_wide_sub_rules_multi(width_bytes: int, S: float = 100.0):
    """Construct multi-byte wide_sub rules for the ad-hoc dim layout."""
    return wide_sub_rules(
        operand_a_base="OPERAND_A",
        operand_b_base="OPERAND_B",
        result_base="RESULT",
        borrow_base="BORROW",
        width_bytes=width_bytes,
        opcode_gate="OP_SUB_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )


def _lowered_pureffn_for_wide_sub_multi(
    width_bytes: int, S: float = 100.0
) -> PureFFN:
    """Lower the multi-byte wide_sub rules into a PureFFN."""
    rules = _build_wide_sub_rules_multi(width_bytes, S=S)
    # Per docstring: 256 for byte 0 + (512 + 1) per subsequent byte
    # (mirror of wide_add_rules).
    expected_count = 256 + (width_bytes - 1) * (512 + 1)
    assert len(rules) == expected_count, (
        f"wide_sub_rules(width_bytes={width_bytes}) emitted {len(rules)} "
        f"rules, expected {expected_count}"
    )
    ffn = PureFFN(dim=_SUB_FFN_DIM, hidden_dim=expected_count)
    end = Primitives.lower_ffn_rules(
        ffn, rules, _SUB_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == expected_count, (
        f"lower_ffn_rules wrote {end} units, expected {expected_count}"
    )
    return ffn


def _make_wide_sub_input(
    *, a: int, b: int, width_bytes: int
) -> torch.Tensor:
    """Build a one-position residual carrying a multi-nibble (a, b) SUB
    plus the pre-computed borrow-in one-hots needed for each nibble > 0.

    Operand A (minuend) nibbles populate ``OPERAND_A+(bi*16 + nib)``;
    operand B (subtrahend) similarly. Borrow-in flags ``BORROW+bi`` for
    ``bi in 0..width-2`` are set if the partial diff at nibble ``bi``
    borrows (precomputed in Python so the single-pass FFN can resolve
    each nibble's lookup without needing a multi-pass cascade).
    """
    x = torch.zeros(1, 1, _SUB_FFN_DIM)
    x[0, 0, _SUB_DIM_LAYOUT["MARK_GATE"]] = 1.0
    x[0, 0, _SUB_DIM_LAYOUT["OP_SUB_GATE"]] = 1.0
    a_nibs = _per_nibble(a, width_bytes)
    b_nibs = _per_nibble(b, width_bytes)
    for bi, (an, bn) in enumerate(zip(a_nibs, b_nibs)):
        x[0, 0, _SUB_DIM_LAYOUT["OPERAND_A"] + bi * 16 + an] = 1.0
        x[0, 0, _SUB_DIM_LAYOUT["OPERAND_B"] + bi * 16 + bn] = 1.0
    # Precompute borrow chain for the input residual so each nibble's
    # lookup sees the expected borrow-in via BORROW+(bi-1).
    borrow = 0
    for bi in range(width_bytes - 1):
        raw = a_nibs[bi] - b_nibs[bi] - borrow
        borrow = 1 if raw < 0 else 0
        if borrow:
            x[0, 0, _SUB_DIM_LAYOUT["BORROW"] + bi] = 1.0
    return x


def _decode_wide_sub_result(y: torch.Tensor, width_bytes: int) -> int:
    """Reassemble the multi-nibble diff by argmax over each nibble's
    RESULT lane and packing nibbles little-endian.
    """
    base = _SUB_DIM_LAYOUT["RESULT"]
    value = 0
    for bi in range(width_bytes):
        lane = y[0, 0, base + bi * 16:base + bi * 16 + 16]
        value |= int(lane.argmax().item()) << (4 * bi)
    return value


def test_wide_sub_rules_emit_expected_multi_byte_count():
    """Sanity: rule count formula 256 + (W - 1) * 513 for width_bytes 1..8."""
    for w in (1, 2, 4, 8):
        rules = _build_wide_sub_rules_multi(w)
        assert len(rules) == 256 + (w - 1) * (512 + 1), (
            f"width_bytes={w}: emitted {len(rules)} rules, "
            f"expected {256 + (w - 1) * (512 + 1)}"
        )


def test_wide_sub_rules_rejects_bad_width_bytes():
    """``width_bytes < 1`` is a ValueError."""
    with pytest.raises(ValueError, match="width_bytes"):
        wide_sub_rules(
            operand_a_base="OPERAND_A",
            operand_b_base="OPERAND_B",
            result_base="RESULT",
            borrow_base="BORROW",
            width_bytes=0,
            opcode_gate="OP_SUB_GATE",
            marker_gate="MARK_GATE",
            S=100.0,
        )


@pytest.fixture(scope="module")
def lowered_sub_ffn_8bit_multi() -> PureFFN:
    # width_bytes=2 -> 2 nibbles -> 8 bits
    return _lowered_pureffn_for_wide_sub_multi(2)


@pytest.fixture(scope="module")
def lowered_sub_ffn_16bit() -> PureFFN:
    # width_bytes=4 -> 4 nibbles -> 16 bits
    return _lowered_pureffn_for_wide_sub_multi(4)


def test_wide_sub_rules_byte_identity_8bit_multi(lowered_sub_ffn_8bit_multi):
    """width_bytes=2 (8-bit via 2 nibble lanes) decoded diff equals
    ``(a - b) & 0xFF`` for every sampled pair, including the canonical
    0x100-1 borrow boundary.

    The pre-injected borrow-in flag per ``_make_wide_sub_input`` lets each
    nibble's lookup see its expected borrow-in. This validates the
    per-nibble rule semantics (borrow suppression for bin=0, positive
    borrow-in activation for bin=1) and the multi-nibble end-to-end
    layout for a single byte's worth of value split across 2 nibble lanes.
    """
    gen = torch.Generator().manual_seed(0x05B17506)  # "8-bit sub"
    n_trials = 64
    pairs = torch.randint(0, 256, (n_trials, 2), generator=gen).tolist()
    # Boundary cases: the task brief requests the 0x100-1 borrow boundary,
    # plus zero, alternating, and full-width borrow chain coverage.
    pairs += [
        (0x00, 0x00),
        (0xFF, 0xFF),
        (0x00, 0x01),  # 0x100 - 1 boundary: forces borrow through nibble 0
        (0x10, 0x01),  # mid-byte borrow boundary
        (0x80, 0x7F),
        (0x55, 0xAA),  # interleaved bits, full borrow
        (0xAA, 0x55),  # interleaved bits, no borrow
    ]

    mismatches = []
    for a, b in pairs:
        x = _make_wide_sub_input(a=a, b=b, width_bytes=2)
        with torch.no_grad():
            y = lowered_sub_ffn_8bit_multi(x)
        decoded = _decode_wide_sub_result(y, width_bytes=2)
        expected = (a - b) & 0xFF
        if decoded != expected:
            mismatches.append(
                f"a=0x{a:02X} b=0x{b:02X}: "
                f"expected=0x{expected:02X} got=0x{decoded:02X}"
            )

    assert not mismatches, (
        f"wide_sub width=2 (8-bit) byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


def test_wide_sub_rules_byte_identity_16bit(lowered_sub_ffn_16bit):
    """width_bytes=4 (16-bit, 4 nibble lanes) decoded diff equals
    ``(a - b) & 0xFFFF`` for every sampled pair, including the canonical
    0x10000-1 borrow boundary.
    """
    gen = torch.Generator().manual_seed(0x16B17506)  # "16-bit sub"
    n_trials = 64
    pairs = torch.randint(0, 0x10000, (n_trials, 2), generator=gen).tolist()
    # Boundary cases: full-width borrow chain, mid-width borrow, no-borrow.
    pairs += [
        (0x0000, 0x0000),
        (0xFFFF, 0xFFFF),
        (0x0000, 0x0001),  # 0x10000 - 1: full-chain borrow
        (0x0100, 0x0001),  # mid-width borrow boundary
        (0x1000, 0x0001),  # borrow through 3 nibbles
        (0x5555, 0xAAAA),  # interleaved bits, full borrow
        (0xAAAA, 0x5555),  # interleaved bits, no borrow
    ]

    mismatches = []
    for a, b in pairs:
        x = _make_wide_sub_input(a=a, b=b, width_bytes=4)
        with torch.no_grad():
            y = lowered_sub_ffn_16bit(x)
        decoded = _decode_wide_sub_result(y, width_bytes=4)
        expected = (a - b) & 0xFFFF
        if decoded != expected:
            mismatches.append(
                f"a=0x{a:04X} b=0x{b:04X}: "
                f"expected=0x{expected:04X} got=0x{decoded:04X}"
            )

    assert not mismatches, (
        f"wide_sub width=4 (16-bit) byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# Wave W2: wide_shift_rules multi-byte byte-identity (width_bytes=2 = 16-bit,
# width_bytes=4 = 32-bit).
# ---------------------------------------------------------------------------
#
# Shift semantics from ``wide_shift_rules`` docstring: "The shift is
# per-byte and independent across bytes -- no inter-byte propagation,
# matching the simplified semantics this Wave is targeting." So multi-byte
# shift = N independent 8-bit shifts, one per byte lane. The test
# validates the cross-byte layout (each byte's operand at ``OPERAND_BASE
# + b*256 + N`` writes to ``RESULT_BASE + b*256 + result`` without
# bleeding into adjacent lanes) and randomised + boundary correctness.
#
# Reuses the same per-byte 256-wide one-hot bands as the 8-bit shift
# test; we just widen the OPERAND / RESULT bands to cover ``width_bytes *
# 256`` slots.

_MULTI_SHIFT_MAX_BYTES = 4
_MULTI_SHIFT_DIM_LAYOUT = {
    "MARK_GATE": 0,
    "OP_SHL_GATE": 1,
    "OP_SHR_GATE": 2,
    "SHIFT_AMT": 3,                              # +k for k in 0..7
    "OPERAND": 11,                               # +(b*256 + N)
    "RESULT": 11 + 256 * _MULTI_SHIFT_MAX_BYTES,  # +(b*256 + R)
}
_MULTI_SHIFT_FFN_DIM = 11 + 512 * _MULTI_SHIFT_MAX_BYTES


def _lowered_wide_shift_ffn_multi(
    direction: str, width_bytes: int, S: float = 100.0
) -> PureFFN:
    """Build a multi-byte PureFFN lowered from ``wide_shift_rules`` for one
    direction at the given width.
    """
    rules = wide_shift_rules(
        direction=direction,
        operand_base="OPERAND",
        shift_amount_dim="SHIFT_AMT",
        result_base="RESULT",
        width_bytes=width_bytes,
        opcode_gate="OP_SHL_GATE" if direction == "left" else "OP_SHR_GATE",
        marker_gate="MARK_GATE",
        S=S,
    )
    expected_count = width_bytes * 8 * 256
    assert len(rules) == expected_count, (
        f"wide_shift_rules({direction!r}, width_bytes={width_bytes}) "
        f"emitted {len(rules)} rules, expected {expected_count}"
    )
    ffn = PureFFN(dim=_MULTI_SHIFT_FFN_DIM, hidden_dim=expected_count)
    end = Primitives.lower_ffn_rules(
        ffn, rules, _MULTI_SHIFT_DIM_LAYOUT, start_unit=0, S=S,
    )
    assert end == expected_count, (
        f"lower_ffn_rules wrote {end} units, expected {expected_count}"
    )
    return ffn


def _make_multi_shift_input(
    *, ns: list[int], k: int, direction: str, width_bytes: int
) -> torch.Tensor:
    """One-position residual with marker, opcode, shift amount, and one
    operand byte per lane. ``ns[bi]`` is the byte value at lane ``bi``.
    """
    assert len(ns) == width_bytes, (
        f"need exactly {width_bytes} byte values, got {len(ns)}"
    )
    x = torch.zeros(1, 1, _MULTI_SHIFT_FFN_DIM)
    x[0, 0, _MULTI_SHIFT_DIM_LAYOUT["MARK_GATE"]] = 1.0
    gate_dim = "OP_SHL_GATE" if direction == "left" else "OP_SHR_GATE"
    x[0, 0, _MULTI_SHIFT_DIM_LAYOUT[gate_dim]] = 1.0
    x[0, 0, _MULTI_SHIFT_DIM_LAYOUT["SHIFT_AMT"] + k] = 1.0
    for bi, n in enumerate(ns):
        x[0, 0, _MULTI_SHIFT_DIM_LAYOUT["OPERAND"] + bi * 256 + (n & 0xFF)] = 1.0
    return x


def _decode_multi_shift_result(
    y: torch.Tensor, width_bytes: int
) -> list[int]:
    """Decode each byte lane independently via argmax over its 256-wide
    RESULT slice.
    """
    base = _MULTI_SHIFT_DIM_LAYOUT["RESULT"]
    return [
        int(y[0, 0, base + bi * 256:base + bi * 256 + 256].argmax().item())
        for bi in range(width_bytes)
    ]


@pytest.fixture(scope="module")
def lowered_shift_ffns_16bit() -> dict:
    return {
        "left": _lowered_wide_shift_ffn_multi("left", 2),
        "right": _lowered_wide_shift_ffn_multi("right", 2),
    }


@pytest.fixture(scope="module")
def lowered_shift_ffns_32bit() -> dict:
    return {
        "left": _lowered_wide_shift_ffn_multi("left", 4),
        "right": _lowered_wide_shift_ffn_multi("right", 4),
    }


@pytest.mark.parametrize("direction", ["left", "right"])
def test_wide_shift_rules_byte_identity_16bit(
    lowered_shift_ffns_16bit, direction
):
    """width_bytes=2 cross-byte shift: each of the 2 byte lanes shifts
    independently. Lane ``bi`` must equal ``(ns[bi] OP k) & 0xFF`` with no
    bleed into the other lane.

    Boundary coverage:
      * 0x80 << 1 (carry-out lost — drop high bit per per-byte semantics)
      * 0x01 << 7 (max shift)
      * 0xFF >> 7 (max right shift)
      * Mixed lanes (one lane non-trivial, other lane zero) to verify
        cross-byte isolation.
    """
    gen = torch.Generator().manual_seed(0x16B175 if direction == "left" else 0x16B175F2)
    pairs = []
    for _ in range(32):
        b0 = int(torch.randint(0, 256, (1,), generator=gen).item())
        b1 = int(torch.randint(0, 256, (1,), generator=gen).item())
        k = int(torch.randint(0, 8, (1,), generator=gen).item())
        pairs.append(([b0, b1], k))
    # Boundary cases.
    pairs += [
        ([0x00, 0x00], 0),
        ([0xFF, 0xFF], 0),
        ([0x80, 0x80], 1),  # 0x80<<1 = 0x00 (drop high) on each lane
        ([0x01, 0x00], 7),  # left: 0x80,0; right: 0,0
        ([0xFF, 0x00], 7),  # cross-byte isolation: lane 1 must stay 0
        ([0x00, 0xFF], 7),  # mirror — lane 0 must stay 0
        ([0xAA, 0x55], 4),  # interleaved
    ]

    mismatches = []
    for ns, k in pairs:
        x = _make_multi_shift_input(
            ns=ns, k=k, direction=direction, width_bytes=2,
        )
        with torch.no_grad():
            y = lowered_shift_ffns_16bit[direction](x)
        decoded = _decode_multi_shift_result(y, width_bytes=2)
        if direction == "left":
            expected = [(n << k) & 0xFF for n in ns]
        else:
            expected = [(n >> k) & 0xFF for n in ns]
        if decoded != expected:
            mismatches.append(
                f"{direction} ns={[f'0x{n:02X}' for n in ns]} k={k}: "
                f"expected={[f'0x{e:02X}' for e in expected]} "
                f"got={[f'0x{d:02X}' for d in decoded]}"
            )

    assert not mismatches, (
        f"wide_shift width=2 (16-bit) {direction} byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


@pytest.mark.parametrize("direction", ["left", "right"])
def test_wide_shift_rules_byte_identity_32bit(
    lowered_shift_ffns_32bit, direction
):
    """width_bytes=4 cross-byte shift: each of the 4 byte lanes shifts
    independently with no inter-lane bleed.

    The single-pass FFN here exercises the wide layout end-to-end at
    32-bit operand width: 4 lanes * 8 shift amounts * 256 byte values =
    8192 rules.
    """
    gen = torch.Generator().manual_seed(
        0x32B175 if direction == "left" else 0x32B175F2
    )
    pairs = []
    for _ in range(24):
        ns = [
            int(torch.randint(0, 256, (1,), generator=gen).item())
            for _ in range(4)
        ]
        k = int(torch.randint(0, 8, (1,), generator=gen).item())
        pairs.append((ns, k))
    # Boundary cases.
    pairs += [
        ([0x00, 0x00, 0x00, 0x00], 0),
        ([0xFF, 0xFF, 0xFF, 0xFF], 0),
        ([0x80, 0x80, 0x80, 0x80], 1),  # 0x80<<1 boundary on every lane
        ([0x01, 0x01, 0x01, 0x01], 7),
        ([0xFF, 0x00, 0xFF, 0x00], 7),  # cross-lane isolation
        ([0xAA, 0x55, 0xAA, 0x55], 4),
    ]

    mismatches = []
    for ns, k in pairs:
        x = _make_multi_shift_input(
            ns=ns, k=k, direction=direction, width_bytes=4,
        )
        with torch.no_grad():
            y = lowered_shift_ffns_32bit[direction](x)
        decoded = _decode_multi_shift_result(y, width_bytes=4)
        if direction == "left":
            expected = [(n << k) & 0xFF for n in ns]
        else:
            expected = [(n >> k) & 0xFF for n in ns]
        if decoded != expected:
            mismatches.append(
                f"{direction} ns={[f'0x{n:02X}' for n in ns]} k={k}: "
                f"expected={[f'0x{e:02X}' for e in expected]} "
                f"got={[f'0x{d:02X}' for d in decoded]}"
            )

    assert not mismatches, (
        f"wide_shift width=4 (32-bit) {direction} byte-identity failures "
        f"({len(mismatches)}/{len(pairs)}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# wide_div_rules_ge_format — byte-accurate single-byte DIV/MOD lookup
# ---------------------------------------------------------------------------
#
# These tests validate the *symbolic* byte-identity contract of the
# byte-accurate flat 8-bit cross-product DIV/MOD lookup. Each rule fires
# on a specific (a, b) operand pair via a 5-way AND on
# (MARK_AX, ALU_LO+a_lo, ALU_HI+a_hi, AX_CARRY_LO+b_lo, AX_CARRY_HI+b_hi),
# gated on the opcode flag, writing the byte-accurate quotient/remainder
# nibbles to OUTPUT_LO/OUTPUT_HI.
#
# Runtime install gap (see docs/DIV_GE_FORMAT_INSTALL_BLOCKER_2026_06_10.md):
# the residual at the L10 post_op install point is not clean one-hot, so
# the lookup as currently wired does not produce the correct OUTPUT byte
# inside the full VM. The tests below exercise the rules with clean
# one-hot inputs to confirm the symbolic semantics; the architectural
# fix is deferred.


def _build_wide_div_ge_rules(op: str, S: float = 100.0):
    return wide_div_rules_ge_format(
        dividend_lo_base="ALU_LO",
        dividend_hi_base="ALU_HI",
        divisor_lo_base="AX_CARRY_LO",
        divisor_hi_base="AX_CARRY_HI",
        result_lo_base="OUTPUT_LO",
        result_hi_base="OUTPUT_HI",
        width_bytes=1,
        opcode_gate="OP_DIV" if op == "div" else "OP_MOD",
        marker_gate="MARK_AX",
        S=S,
        op=op,
    )


def _lowered_wide_div_ge_ffn(op: str, S: float = 100.0) -> PureFFN:
    rules = _build_wide_div_ge_rules(op, S=S)
    assert len(rules) == 256 * 256
    ffn = PureFFN(dim=512, hidden_dim=len(rules))
    names = Primitives.ffn_rule_dim_names(rules)
    dim_positions = Primitives.dim_positions_from_bd(_SetDim, names)
    end = Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=0, S=S,
    )
    assert end == len(rules)
    return ffn


def _make_div_input(*, a: int, b: int, op: str) -> torch.Tensor:
    """One-position residual at MARK_AX with the dividend/divisor."""
    op_dim = _SetDim.OP_DIV if op == "div" else _SetDim.OP_MOD
    x = torch.zeros(1, 1, 512)
    x[0, 0, _SetDim.MARK_AX] = 1.0
    x[0, 0, op_dim] = 1.0
    x[0, 0, _SetDim.ALU_LO + (a & 0xF)] = 1.0
    x[0, 0, _SetDim.ALU_HI + ((a >> 4) & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_LO + (b & 0xF)] = 1.0
    x[0, 0, _SetDim.AX_CARRY_HI + ((b >> 4) & 0xF)] = 1.0
    return x


def test_wide_div_rules_ge_format_emit_expected_count():
    """256 * 256 = 65,536 rules per opcode batch."""
    for op in ("div", "mod"):
        rules = _build_wide_div_ge_rules(op)
        assert len(rules) == 65_536, (
            f"wide_div_rules_ge_format(op={op!r}) emitted "
            f"{len(rules)} rules, expected 65,536"
        )


def test_wide_div_rules_ge_format_rejects_bad_args():
    """``width_bytes < 1`` is ValueError; ``width_bytes > 1`` is
    NotImplementedError; ``op`` outside {'div', 'mod'} is ValueError.
    """
    with pytest.raises(ValueError, match="width_bytes"):
        wide_div_rules_ge_format(
            dividend_lo_base="A_LO", dividend_hi_base="A_HI",
            divisor_lo_base="B_LO", divisor_hi_base="B_HI",
            result_lo_base="R_LO", result_hi_base="R_HI",
            width_bytes=0,
            opcode_gate="OP", marker_gate="MARK", S=100.0,
        )
    with pytest.raises(NotImplementedError, match="multi-byte"):
        wide_div_rules_ge_format(
            dividend_lo_base="A_LO", dividend_hi_base="A_HI",
            divisor_lo_base="B_LO", divisor_hi_base="B_HI",
            result_lo_base="R_LO", result_hi_base="R_HI",
            width_bytes=2,
            opcode_gate="OP", marker_gate="MARK", S=100.0,
        )
    with pytest.raises(ValueError, match="op must be"):
        wide_div_rules_ge_format(
            dividend_lo_base="A_LO", dividend_hi_base="A_HI",
            divisor_lo_base="B_LO", divisor_hi_base="B_HI",
            result_lo_base="R_LO", result_hi_base="R_HI",
            width_bytes=1,
            opcode_gate="OP", marker_gate="MARK", S=100.0,
            op="something_else",
        )


@pytest.fixture(scope="module")
def lowered_wide_div_ge_ffns() -> dict:
    return {op: _lowered_wide_div_ge_ffn(op) for op in ("div", "mod")}


@pytest.mark.parametrize(
    ("a", "b", "op"),
    [
        # Cross-nibble dividends (the per-nibble bug repro: 84 = 0x54).
        (84, 2, "div"),    # 42 = 0x2A
        (84, 2, "mod"),    # 0
        (43, 10, "div"),   # 4
        (43, 10, "mod"),   # 3
        # Multi-nibble quotients.
        (200, 7, "div"),   # 28 = 0x1C
        (200, 7, "mod"),   # 4
        # Same-nibble dividend (per-nibble case still works).
        (15, 3, "div"),    # 5
        (15, 3, "mod"),    # 0
        # Boundary: max dividend.
        (255, 16, "div"),  # 15 = 0x0F
        (255, 16, "mod"),  # 15 = 0x0F
        # Divide-by-zero: convention q = r = 0.
        (84, 0, "div"),    # 0
        (84, 0, "mod"),    # 0
        # Identity.
        (1, 1, "div"),     # 1
        (1, 1, "mod"),     # 0
    ],
)
def test_wide_div_rules_ge_format_byte_identity(
    lowered_wide_div_ge_ffns, a, b, op
):
    """Lowered ``wide_div_rules_ge_format`` decodes to the same OUTPUT
    byte as Python's ``a // b`` / ``a % b`` on clean one-hot input.

    This is the *byte-accurate* contract that the per-nibble
    ``wide_div_rules`` cannot satisfy for cross-nibble dividends — see
    docs/DSL_W5_MULDIV_LIMIT.md for the per-nibble failure mode.
    """
    x = _make_div_input(a=a, b=b, op=op)
    with torch.no_grad():
        y = lowered_wide_div_ge_ffns[op](x)
    decoded = _decode_output_byte(y)
    if b == 0:
        expected = 0
    else:
        expected = (a // b) if op == "div" else (a % b)
    assert decoded == expected, (
        f"wide_div_rules_ge_format({op}) {a} {b}: "
        f"expected 0x{expected:02X}, got 0x{decoded:02X}"
    )


def test_wide_div_rules_ge_format_byte_identity_randomized(
    lowered_wide_div_ge_ffns
):
    """Randomized sweep over 32 (a, b) pairs per opcode."""
    gen = torch.Generator().manual_seed(0xD1ED1ED1)  # "DIE-DIE" hex-ish
    n_trials = 32
    pairs = torch.randint(1, 256, (n_trials, 2), generator=gen).tolist()

    mismatches = []
    for op in ("div", "mod"):
        for a, b in pairs:
            x = _make_div_input(a=a, b=b, op=op)
            with torch.no_grad():
                decoded = _decode_output_byte(lowered_wide_div_ge_ffns[op](x))
            expected = (a // b) if op == "div" else (a % b)
            if decoded != expected:
                mismatches.append(
                    f"{op} {a} {b}: expected=0x{expected:02X} "
                    f"got=0x{decoded:02X}"
                )
    assert not mismatches, (
        f"wide_div_rules_ge_format byte-identity failures "
        f"({len(mismatches)}/{2 * n_trials}):\n  "
        + "\n  ".join(mismatches[:10])
    )


# ---------------------------------------------------------------------------
# GAP-PRIMITIVE #2 (docs/DSL_W5_MULDIV_LIMIT.md Path 1): multi_pass_mul_rules.
#
# The flat wide_mul_rules(width_bytes=2) is a 65,536-rule cross-product lookup
# that does not generalize (width-3 = 16.8M rules). multi_pass_mul_rules
# derives the SAME 16-bit product from a COMPACT schoolbook spec staged across
# 7 FFN passes (~2.8k units) via the MultiPassOp IR, with a cross-pass column
# carry chain the single-forward lookup cannot express. This test lowers the
# passes into a stack of PureFFNs and proves the decoded product is byte-
# identical to (a * b) & 0xFFFF for EVERY (a, b) in 0..255 x 0..255 — i.e.
# verdict-identical to what the flat wide_mul_rules lookup computes.
# ---------------------------------------------------------------------------

from neural_vm.unified_compiler.wide_alu_dsl import (  # noqa: E402
    multi_pass_mul_rules,
)

_MP_MUL_POS = {}
_MP_MUL_DIM = 0
for _nm, _w in (
    ("MARK", 1), ("OP_MUL", 1), ("OPA", 32), ("OPB", 32),
    ("RES", 64), ("WS", 240),
):
    _MP_MUL_POS[_nm] = _MP_MUL_DIM
    _MP_MUL_DIM += _w


def _lowered_multi_pass_mul(S: float = 100.0):
    """Lower each pass of the multi-pass MUL into its own PureFFN.

    Returns ``(ffns, positions)`` — the ordered PureFFN stack (one per
    pass) and the ad-hoc dim layout.
    """
    mp = multi_pass_mul_rules(
        operand_a_base="OPA", operand_b_base="OPB", result_base="RES",
        workspace_base="WS", opcode_gate="OP_MUL", marker_gate="MARK",
        S=S, width_bytes=2,
    )
    flat = mp.as_flat_ir()
    ffns = []
    for i, p in enumerate(mp.passes):
        f = PureFFN(dim=_MP_MUL_DIM, hidden_dim=max(1, p.hidden_units))
        flat.lower_ffn(f, _MP_MUL_POS, layer_idx=i, S=S)
        ffns.append(f)
    return ffns


@pytest.fixture(scope="module")
def lowered_multi_pass_mul_ffns():
    return _lowered_multi_pass_mul()


def test_multi_pass_mul_pass_structure():
    """The schoolbook cascade is 7 passes and O(width^2*256) units —
    NOT the flat lookup's 65,536 cross-product rules."""
    mp = multi_pass_mul_rules(
        operand_a_base="OPA", operand_b_base="OPB", result_base="RES",
        workspace_base="WS", opcode_gate="OP_MUL", marker_gate="MARK",
        S=100.0, width_bytes=2,
    )
    assert mp.num_passes == 7
    # Total units must be far below the flat 65,536-rule lookup.
    assert mp.hidden_units < 4000, mp.hidden_units
    # width>2 pilot boundary.
    with pytest.raises(NotImplementedError):
        multi_pass_mul_rules(
            operand_a_base="OPA", operand_b_base="OPB", result_base="RES",
            workspace_base="WS", opcode_gate="OP_MUL", marker_gate="MARK",
            S=100.0, width_bytes=3,
        )


def _mp_mul_batch_inputs():
    N = 65536
    X = torch.zeros(N, 1, _MP_MUL_DIM)
    ab = []
    idx = 0
    for a in range(256):
        for b in range(256):
            X[idx, 0, _MP_MUL_POS["MARK"]] = 1.0
            X[idx, 0, _MP_MUL_POS["OP_MUL"]] = 1.0
            X[idx, 0, _MP_MUL_POS["OPA"] + (a & 0xF)] = 1.0
            X[idx, 0, _MP_MUL_POS["OPA"] + 16 + ((a >> 4) & 0xF)] = 1.0
            X[idx, 0, _MP_MUL_POS["OPB"] + (b & 0xF)] = 1.0
            X[idx, 0, _MP_MUL_POS["OPB"] + 16 + ((b >> 4) & 0xF)] = 1.0
            ab.append((a, b))
            idx += 1
    return X, ab


def test_multi_pass_mul_byte_identity_full(lowered_multi_pass_mul_ffns):
    """The multi-pass schoolbook cascade decodes (a * b) & 0xFFFF for
    EVERY (a, b) in 0..255 x 0..255 — byte-identical to the function the
    flat wide_mul_rules(width_bytes=2) lookup computes.

    This is the GAP-PRIMITIVE #2 pilot gate: it proves the multi_pass_rules
    IR (MultiPassOp) can express the cross-pass column-carry chain of
    schoolbook MUL, which a single-forward FFN lookup fundamentally cannot.
    """
    X, ab = _mp_mul_batch_inputs()
    with torch.no_grad():
        Y = X
        for f in lowered_multi_pass_mul_ffns:
            Y = f(Y)
    base = _MP_MUL_POS["RES"]
    vals = torch.zeros(len(ab), dtype=torch.long)
    for lane in range(4):
        nib = Y[:, 0, base + lane * 16:base + lane * 16 + 16].argmax(dim=-1)
        vals |= nib.long() << (4 * lane)
    expected = torch.tensor([(a * b) & 0xFFFF for a, b in ab], dtype=torch.long)
    mismatch = (vals != expected)
    n_bad = int(mismatch.sum())
    if n_bad:
        bad = mismatch.nonzero().flatten()[:10]
        detail = ", ".join(
            f"a={ab[int(i)][0]} b={ab[int(i)][1]} "
            f"got=0x{int(vals[i]):04X} exp=0x{int(expected[i]):04X}"
            for i in bad
        )
    else:
        detail = ""
    assert n_bad == 0, f"multi_pass_mul mismatches {n_bad}/65536: {detail}"


# ---------------------------------------------------------------------------
# GAP-PRIMITIVE #2 (DIV pilot): multi_pass_div_rules — binary long division.
#
# The flat wide_div_rules_ge_format(width_bytes=1) is a 65,536-rule
# cross-product lookup per opcode (byte-accurate but O(256^width), does not
# generalize). multi_pass_div_rules derives the SAME single-byte quotient /
# remainder from a COMPACT bit-serial shift-subtract spec staged across FFN
# passes via the MultiPassOp IR, with a cross-pass running-remainder carry the
# single-forward lookup cannot express. This test lowers the passes into a
# stack of PureFFNs and proves the decoded (q, r) is byte-identical to Python
# divmod (with the b==0 -> q=0,r=a convention) for EVERY (a, b) in
# 0..255 x 0..255.
# ---------------------------------------------------------------------------

from neural_vm.unified_compiler.wide_alu_dsl import (  # noqa: E402
    multi_pass_div_rules,
)

_MP_DIV_POS = {}
_MP_DIV_DIM = 0
for _nm, _w in (
    ("MARK", 1), ("OP_DIV", 1), ("A", 32), ("B", 32),
    ("QLO", 16), ("QHI", 16), ("RLO", 16), ("RHI", 16),
    ("WS", 108 * 16),
):
    _MP_DIV_POS[_nm] = _MP_DIV_DIM
    _MP_DIV_DIM += _w


def _build_mp_div(S: float = 100.0):
    return multi_pass_div_rules(
        dividend_a_base="A", divisor_b_base="B",
        quotient_lane_bases=("QLO", "QHI"),
        remainder_lane_bases=("RLO", "RHI"),
        workspace_base="WS", opcode_gate="OP_DIV", marker_gate="MARK",
        S=S, width_bytes=1,
    )


def _lowered_multi_pass_div(S: float = 100.0):
    mp = _build_mp_div(S)
    flat = mp.as_flat_ir()
    ffns = []
    for i, p in enumerate(mp.passes):
        f = PureFFN(dim=_MP_DIV_DIM, hidden_dim=max(1, p.hidden_units))
        flat.lower_ffn(f, _MP_DIV_POS, layer_idx=i, S=S)
        ffns.append(f)
    return ffns


@pytest.fixture(scope="module")
def lowered_multi_pass_div_ffns():
    return _lowered_multi_pass_div()


def test_multi_pass_div_pass_structure():
    """The long-division cascade is O(width*256) units — NOT the flat
    wide_div_rules_ge_format 65,536-rule cross-product per opcode. width>1
    is the pilot boundary."""
    mp = _build_mp_div()
    # 8 bits x 4 passes + seed + assemble + bzero = 43 passes.
    assert mp.num_passes == 43, mp.num_passes
    # Far below the flat 65,536-rule lookup.
    assert mp.hidden_units < 20000, mp.hidden_units
    with pytest.raises(NotImplementedError):
        multi_pass_div_rules(
            dividend_a_base="A", divisor_b_base="B",
            quotient_lane_bases=("QLO", "QHI"),
            remainder_lane_bases=("RLO", "RHI"),
            workspace_base="WS", opcode_gate="OP_DIV", marker_gate="MARK",
            S=100.0, width_bytes=2,
        )


def _mp_div_batch_inputs():
    N = 65536
    X = torch.zeros(N, 1, _MP_DIV_DIM)
    ab = []
    idx = 0
    for a in range(256):
        for b in range(256):
            X[idx, 0, _MP_DIV_POS["MARK"]] = 1.0
            X[idx, 0, _MP_DIV_POS["OP_DIV"]] = 1.0
            X[idx, 0, _MP_DIV_POS["A"] + (a & 0xF)] = 1.0
            X[idx, 0, _MP_DIV_POS["A"] + 16 + ((a >> 4) & 0xF)] = 1.0
            X[idx, 0, _MP_DIV_POS["B"] + (b & 0xF)] = 1.0
            X[idx, 0, _MP_DIV_POS["B"] + 16 + ((b >> 4) & 0xF)] = 1.0
            ab.append((a, b))
            idx += 1
    return X, ab


def test_multi_pass_div_byte_identity_full(lowered_multi_pass_div_ffns):
    """The long-division cascade decodes (a // b, a % b) for EVERY (a, b) in
    0..255 x 0..255 — with the b==0 -> (q=0, r=a) zero-divide convention.

    GAP-PRIMITIVE #2 DIV pilot gate: proves the multi_pass_rules IR
    (MultiPassOp) can express the cross-pass running-remainder chain of binary
    long division, which a single-forward FFN lookup fundamentally cannot.
    """
    X, ab = _mp_div_batch_inputs()
    with torch.no_grad():
        Y = X
        for f in lowered_multi_pass_div_ffns:
            Y = f(Y)

    def dec(base):
        return Y[:, 0, _MP_DIV_POS[base]:_MP_DIV_POS[base] + 16].argmax(dim=-1)

    q = (dec("QHI").long() << 4) | dec("QLO").long()
    r = (dec("RHI").long() << 4) | dec("RLO").long()
    exp_q = torch.tensor([0 if b == 0 else a // b for a, b in ab])
    exp_r = torch.tensor([a if b == 0 else a % b for a, b in ab])
    mismatch = (q != exp_q) | (r != exp_r)
    n_bad = int(mismatch.sum())
    if n_bad:
        bad = mismatch.nonzero().flatten()[:10]
        detail = ", ".join(
            f"a={ab[int(i)][0]} b={ab[int(i)][1]} "
            f"got=({int(q[i])},{int(r[i])}) exp=({int(exp_q[i])},{int(exp_r[i])})"
            for i in bad
        )
    else:
        detail = ""
    assert n_bad == 0, f"multi_pass_div mismatches {n_bad}/65536: {detail}"
