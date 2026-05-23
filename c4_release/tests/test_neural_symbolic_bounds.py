"""Bounds checks for symbolic intent vs lowered neural residual values.

These tests are intentionally small: they run isolated neural modules on a
single residual row and assert that the symbolic winner is not merely the argmax
but also has sane residual margins.
"""

import torch

from c4_release.neural_vm.vm_step import (
    AddSubBytePropagationPostOp,
    CarryPropagationPostOp,
    _SetDim as BD,
)


def _band_values(row: torch.Tensor, base: int):
    return row[base:base + 16].detach().cpu()


def _assert_symbolic_band_bounds(
    row: torch.Tensor,
    base: int,
    expected: int,
    *,
    expected_min: float = 1.5,
    inactive_max: float = 0.5,
    old_index: int | None = None,
    old_min: float = -2.5,
    old_max: float = 0.5,
) -> None:
    values = _band_values(row, base)
    expected_value = float(values[expected].item())
    assert expected_value >= expected_min, (
        f"expected band {expected} too small: {expected_value}; "
        f"values={values.tolist()}"
    )
    for idx, value_t in enumerate(values):
        if idx == expected:
            continue
        value = float(value_t.item())
        if idx == old_index:
            assert old_min <= value <= old_max, (
                f"old band {idx} outside [{old_min}, {old_max}]: {value}; "
                f"values={values.tolist()}"
            )
        else:
            assert value <= inactive_max, (
                f"inactive band {idx} too large: {value}; "
                f"values={values.tolist()}"
            )


def _base_addsub_state(*, stack_byte: int, ax_byte: int, add: bool) -> torch.Tensor:
    x = torch.zeros(1, 1, 512)
    x[..., BD.CONST] = 1.0
    x[..., BD.IS_BYTE] = 1.0
    x[..., BD.H1 + 1] = 1.0
    x[..., BD.BYTE_INDEX_0] = 1.0
    x[..., BD.TEMP + (8 if add else 9)] = 1.0

    # Production L7 operand-gather residuals are intentionally amplified. The
    # post-op must still require the OUTPUT nibble predicate instead of letting
    # the amplified ALU band alone satisfy every rule for that nibble.
    x[..., BD.ALU_LO + (stack_byte & 0xF)] = 6.0
    x[..., BD.ALU_HI + ((stack_byte >> 4) & 0xF)] = 6.0
    x[..., BD.OUTPUT_LO + (ax_byte & 0xF)] = 1.0
    x[..., BD.OUTPUT_HI + ((ax_byte >> 4) & 0xF)] = 1.0
    return x


def test_add_byte_base_propagation_respects_symbolic_residual_bounds():
    module = AddSubBytePropagationPostOp(d_model=512, S=100.0)
    x = _base_addsub_state(stack_byte=0x02, ax_byte=0x00, add=True)

    y = module(x)[0, 0]

    _assert_symbolic_band_bounds(
        y,
        BD.OUTPUT_LO,
        expected=0x2,
        old_index=0x0,
    )
    _assert_symbolic_band_bounds(
        y,
        BD.OUTPUT_HI,
        expected=0x0,
        old_index=0x0,
    )


def test_add_carry_propagation_is_stable_across_expected_carry_amplitude():
    module = CarryPropagationPostOp(
        d_model=512,
        S=100.0,
        byte_idx=0,
        cascade=False,
    )
    x = torch.zeros(1, 1, 512)
    x[..., BD.IS_BYTE] = 1.0
    x[..., BD.H1 + 1] = 1.0
    x[..., BD.BYTE_INDEX_0] = 1.0
    x[..., BD.CARRY + 1] = 2.2
    x[..., BD.OUTPUT_LO + 0x2] = 1.0
    x[..., BD.OUTPUT_HI + 0x0] = 1.0

    y = module(x)[0, 0]

    _assert_symbolic_band_bounds(
        y,
        BD.OUTPUT_LO,
        expected=0x3,
        old_index=0x2,
    )
    _assert_symbolic_band_bounds(
        y,
        BD.OUTPUT_HI,
        expected=0x0,
        old_index=0x0,
    )
