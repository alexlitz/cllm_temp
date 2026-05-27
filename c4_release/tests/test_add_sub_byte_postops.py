import pytest
import torch

from c4_release.neural_vm.vm_step import (
    AddSubBytePropagationPostOp,
    BinaryOpByteZeroingPostOp,
    CarryPropagationPostOp,
    _SetDim as BD,
)


def _decode_output_byte(x):
    lo = x[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO + 16]
    hi = x[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI + 16]
    return int(lo.argmax()) | (int(hi.argmax()) << 4)


def _assert_no_positive_non_targets(x, expected_byte, tol=1e-5):
    lo_expected = expected_byte & 0xF
    hi_expected = expected_byte >> 4
    lo = x[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO + 16]
    hi = x[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI + 16]

    lo_non_targets = torch.cat((lo[:lo_expected], lo[lo_expected + 1:]))
    hi_non_targets = torch.cat((hi[:hi_expected], hi[hi_expected + 1:]))

    assert lo[lo_expected] > 0.5
    assert hi[hi_expected] > 0.5
    assert float(lo_non_targets.max()) <= tol
    assert float(hi_non_targets.max()) <= tol


def _add_byte1_input(stack_value, ax_value, carry=0.0, alu_amp=6.0):
    """Build the AX byte-1 query row seen by ADD high-byte post-ops."""
    x = torch.zeros(1, 1, 512)
    stack_byte = (stack_value >> 8) & 0xFF
    ax_byte = (ax_value >> 8) & 0xFF

    x[0, 0, BD.CONST] = 1.0
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, BD.BYTE_INDEX_0] = 1.0
    x[0, 0, BD.TEMP + 8] = 1.0
    x[0, 0, BD.CARRY + 1] = carry

    x[0, 0, BD.OUTPUT_LO + (ax_byte & 0xF)] = 1.0
    x[0, 0, BD.OUTPUT_HI + (ax_byte >> 4)] = 1.0
    x[0, 0, BD.ALU_LO + (stack_byte & 0xF)] = alu_amp
    x[0, 0, BD.ALU_HI + (stack_byte >> 4)] = alu_amp
    return x


@pytest.mark.parametrize(
    ("stack_value", "ax_value"),
    [
        (654, 114),
        (25, 759),
        (281, 250),
        (692, 758),
    ],
)
def test_add_byte1_base_then_carry(stack_value, ax_value):
    x = _add_byte1_input(stack_value, ax_value, carry=2.0)
    base = AddSubBytePropagationPostOp()(x)

    base_byte = (((stack_value >> 8) & 0xFF) + ((ax_value >> 8) & 0xFF)) & 0xFF
    assert _decode_output_byte(base) == base_byte
    _assert_no_positive_non_targets(base, base_byte)

    carried = CarryPropagationPostOp(byte_idx=0, cascade=False)(base)
    expected_byte = (base_byte + 1) & 0xFF
    assert _decode_output_byte(carried) == expected_byte
    _assert_no_positive_non_targets(carried, expected_byte)


def test_add_byte_base_requires_matching_output_nibble_under_alu_amplification():
    x = _add_byte1_input(654, 114, alu_amp=6.0)
    out = AddSubBytePropagationPostOp()(x)

    assert _decode_output_byte(out) == 0x02
    _assert_no_positive_non_targets(out, 0x02)


def test_add_byte_base_tolerates_stale_zero_output_residual():
    x = _add_byte1_input(692, 758, alu_amp=6.0)
    x[0, 0, BD.OUTPUT_LO + 0] = 0.94

    out = AddSubBytePropagationPostOp()(x)

    assert _decode_output_byte(out) == 0x04
    _assert_no_positive_non_targets(out, 0x04)


def test_sub_byte_base_clears_stale_zero_low_nibble():
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.CONST] = 1.0
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, BD.BYTE_INDEX_0] = 1.0
    x[0, 0, BD.TEMP + 9] = 1.0
    x[0, 0, BD.OUTPUT_LO + 0] = 2.94
    x[0, 0, BD.OUTPUT_HI + 0] = 2.94
    x[0, 0, BD.ALU_LO + 6] = 6.0
    x[0, 0, BD.ALU_HI + 0] = 6.0

    out = AddSubBytePropagationPostOp()(x)

    assert _decode_output_byte(out) == 0x06
    assert float(out[0, 0, BD.OUTPUT_LO + 0]) < 0.5
    assert float(out[0, 0, BD.OUTPUT_LO + 6]) > 0.5


def test_sub_byte_base_keeps_zero_high_nibble_under_amplification():
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.CONST] = 1.0
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, BD.BYTE_INDEX_0] = 0.9701380133628845
    x[0, 0, BD.BYTE_INDEX_1] = 0.0132971
    x[0, 0, BD.TEMP + 9] = 1.3048
    x[0, 0, BD.OUTPUT_LO + 0] = 2.9402759
    x[0, 0, BD.OUTPUT_HI + 0] = 2.9402759
    x[0, 0, BD.ALU_LO + 5] = 6.0
    x[0, 0, BD.ALU_HI + 0] = 6.237

    out = AddSubBytePropagationPostOp()(x)

    assert _decode_output_byte(out) == 0x05
    assert out[0, 0, BD.OUTPUT_HI + 0] > out[0, 0, BD.OUTPUT_HI + 8]


def test_add_byte_base_tolerates_weak_bitwise_residue():
    x = _add_byte1_input(0x0300, 0x0100, alu_amp=6.0)
    x[0, 0, BD.TEMP + 3] = 0.305
    x[0, 0, BD.OUTPUT_LO + 0] = 0.94
    x[0, 0, BD.OUTPUT_LO + 1] = 2.0
    x[0, 0, BD.OUTPUT_HI + 0] = 2.94

    out = AddSubBytePropagationPostOp()(x)

    assert _decode_output_byte(out) == 0x04
    _assert_no_positive_non_targets(out, 0x04)


def test_add_byte_base_blocks_step_end_transition_rows():
    x = _add_byte1_input(654, 114, alu_amp=6.0)
    x[0, 0, BD.NEXT_SE] = 1.0

    out = AddSubBytePropagationPostOp()(x)

    assert torch.allclose(
        out[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO + 16],
        x[0, 0, BD.OUTPUT_LO:BD.OUTPUT_LO + 16],
    )
    assert torch.allclose(
        out[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI + 16],
        x[0, 0, BD.OUTPUT_HI:BD.OUTPUT_HI + 16],
    )


@pytest.mark.parametrize("op_dim", [BD.OP_ADD, BD.OP_SUB])
def test_binary_zeroing_skips_add_sub_byte_rows(op_dim):
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, op_dim] = 5.0
    x[0, 0, BD.TEMP + 3] = 1.0
    x[0, 0, BD.TEMP + (8 if op_dim == BD.OP_ADD else 9)] = 1.0
    x[0, 0, BD.OUTPUT_LO + 3] = 2.0
    x[0, 0, BD.OUTPUT_HI + 0] = 2.0

    out = BinaryOpByteZeroingPostOp()(x)

    assert out[0, 0, BD.OUTPUT_LO + 3] == 2.0
    assert out[0, 0, BD.OUTPUT_HI + 0] == 2.0
    assert abs(float(out[0, 0, BD.OUTPUT_LO + 0])) < 1e-6


def test_binary_zeroing_still_zeros_bitwise_byte_rows():
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, BD.TEMP + 3] = 1.0
    x[0, 0, BD.OUTPUT_LO + 3] = 2.0
    x[0, 0, BD.OUTPUT_HI + 3] = 2.0

    out = BinaryOpByteZeroingPostOp()(x)

    assert out[0, 0, BD.OUTPUT_LO + 0] > 0.0
    assert out[0, 0, BD.OUTPUT_HI + 0] > 0.0
    assert out[0, 0, BD.OUTPUT_LO + 3] < 2.0
    assert out[0, 0, BD.OUTPUT_HI + 3] < 2.0


def test_binary_zeroing_blocks_weak_bitwise_residue():
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.IS_BYTE] = 1.0
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, BD.TEMP + 3] = 0.3037
    x[0, 0, BD.OUTPUT_LO + 3] = 2.0
    x[0, 0, BD.OUTPUT_HI + 0] = 2.0

    out = BinaryOpByteZeroingPostOp()(x)

    assert out[0, 0, BD.OUTPUT_LO + 3] == 2.0
    assert out[0, 0, BD.OUTPUT_HI + 0] == 2.0
    assert abs(float(out[0, 0, BD.OUTPUT_LO + 0])) < 1e-6


def test_add_carry_requires_matching_output_nibbles_under_carry_drift():
    x = _add_byte1_input(654, 114, carry=2.2)
    base = AddSubBytePropagationPostOp()(x)
    out = CarryPropagationPostOp(byte_idx=0, cascade=False)(base)

    assert _decode_output_byte(out) == 0x03
    _assert_no_positive_non_targets(out, 0x03)


@pytest.mark.parametrize("marker_dim", [BD.MARK_AX, BD.MARK_PC])
def test_add_carry_blocks_marker_rows_with_double_carry_relay(marker_dim):
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, marker_dim] = 1.0
    x[0, 0, BD.CARRY + 1] = 2.0
    x[0, 0, BD.OUTPUT_LO + 0] = 2.0
    x[0, 0, BD.OUTPUT_HI + 0] = 2.0

    out = CarryPropagationPostOp(byte_idx=0, cascade=False)(x)

    assert abs(float(out[0, 0, BD.OUTPUT_LO + 0]) - 2.0) < 1e-5
    assert abs(float(out[0, 0, BD.OUTPUT_HI + 0]) - 2.0) < 1e-5
    assert abs(float(out[0, 0, BD.OUTPUT_LO + 1])) < 1e-5


@pytest.mark.parametrize("marker_dim", [BD.MARK_AX, BD.MARK_PC])
def test_sub_borrow_blocks_marker_rows_with_double_borrow_relay(marker_dim):
    x = torch.zeros(1, 1, 512)
    x[0, 0, BD.H1 + 1] = 1.0
    x[0, 0, marker_dim] = 1.0
    x[0, 0, BD.CARRY + 2] = 2.0
    x[0, 0, BD.OUTPUT_LO + 0] = 2.0
    x[0, 0, BD.OUTPUT_HI + 0] = 2.0

    out = CarryPropagationPostOp(byte_idx=0, cascade=False)(x)

    assert abs(float(out[0, 0, BD.OUTPUT_LO + 0]) - 2.0) < 1e-5
    assert abs(float(out[0, 0, BD.OUTPUT_HI + 0]) - 2.0) < 1e-5
    assert abs(float(out[0, 0, BD.OUTPUT_LO + 15])) < 1e-5
    assert abs(float(out[0, 0, BD.OUTPUT_HI + 15])) < 1e-5


def test_add_cascade_uses_relayed_add_flag_for_shared_carry():
    first = _add_byte1_input(0xFF00, 0, carry=2.0)
    first_base = AddSubBytePropagationPostOp()(first)
    first_out = CarryPropagationPostOp(byte_idx=0, cascade=False)(first_base)

    assert _decode_output_byte(first_out) == 0x00
    assert first_out[0, 0, BD.CARRY + 3] > 1.5

    second = torch.zeros(1, 1, 512)
    second[0, 0, BD.IS_BYTE] = 1.0
    second[0, 0, BD.H1 + 1] = 1.0
    second[0, 0, BD.BYTE_INDEX_1] = 1.0
    second[0, 0, BD.TEMP + 8] = 1.0
    second[0, 0, BD.CARRY + 3] = first_out[0, 0, BD.CARRY + 3]
    second[0, 0, BD.OUTPUT_LO + 4] = 1.0
    second[0, 0, BD.OUTPUT_HI + 0] = 1.0

    second_out = CarryPropagationPostOp(byte_idx=1, cascade=True)(second)

    assert _decode_output_byte(second_out) == 0x05
    _assert_no_positive_non_targets(second_out, 0x05)
