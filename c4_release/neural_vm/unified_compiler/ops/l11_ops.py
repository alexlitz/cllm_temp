"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer11_mul_partial_op(alu_mode: str = "lookup") -> Operation:
    """L11 FFN: MUL partial product accumulation.

    Pinned to ``layer_idx=11`` via ``kind="block"``: dep-graph layer
    assignment otherwise places this op at L19 (downstream of
    layer6_routing_ffn at L18); legacy_bake no longer calls
    ``_set_layer11_mul_partial`` so without pinning block 11 would be
    zero-init.

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.
    """
    def bake(block, dim_positions, S):
        if alu_mode == "efficient":
            return None
        from ...vm_step import _set_layer11_mul_partial
        _set_layer11_mul_partial(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer11_mul_partial",
        phase=11,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"MUL_ACCUM"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
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
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
    )
