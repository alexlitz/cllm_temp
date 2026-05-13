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
        # Staleness invariants (Phase 3 / Agent G): the L11 MUL partial unit
        # consumes ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) at
        # the AX marker for OP_MUL. Both must be the *current* step's fresh
        # values to produce the correct partial product.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        },
        # Produces the fresh MUL partial product staging at the AX marker
        # (gated on MARK_AX + OP_MUL). The lookup-mode MUL pipeline
        # computes ``partial = (carry + a_lo * b_hi) % 16`` into MUL_ACCUM
        # (== TEMP[partial] via _SetDim aliasing) for the L12 hi-nibble
        # combine to read.
        produces={
            "MUL_ACCUM": "AX_byte0",
        },
    )


