"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer11_mul_partial_op() -> Operation:
    """L11 FFN: MUL partial product accumulation.

    Pinned to ``layer_idx=11`` via ``kind="block"``: dep-graph layer
    assignment otherwise places this op at L19 (downstream of
    layer6_routing_ffn at L18); legacy_bake no longer calls
    ``_set_layer11_mul_partial`` so without pinning block 11 would be
    zero-init.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer11_mul_partial
        _set_layer11_mul_partial(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer11_mul_partial",
        phase=11,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"MUL_ACCUM"},
        kind="block",
        bake_fn=bake,
        layer_idx=11,
        migrated=True,
    )


