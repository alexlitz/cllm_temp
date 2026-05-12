"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer12_mul_combine_op() -> Operation:
    """L12 FFN: combine MUL partial products into final result.

    Pinned to ``layer_idx=12`` via ``kind="block"``. See
    ``make_layer11_mul_partial_op``.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer12_mul_combine
        _set_layer12_mul_combine(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer12_mul_combine",
        phase=12,
        reads={"MARK_AX", "MUL_ACCUM", "OP_MUL"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=12,
        migrated=True,
        # Staleness invariants: L12 MUL combine consumes the same operands
        # as L11 MUL partial (ALU_HI for a_hi, AX_CARRY_LO for b_lo) at the
        # AX marker. These must be the in-step fresh values.
        consumes_fresh={
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
        },
    )


