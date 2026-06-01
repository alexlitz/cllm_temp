"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer12_mul_combine_op(alu_mode: str = "lookup") -> Operation:
    """L12 FFN: combine MUL partial products into final result.

    Pinned to ``layer_idx=12`` via ``kind="block"``. See
    ``make_layer11_mul_partial_op``.

    Declarations-only note: this migrated owner is now exposed through the
    declarations-only dispatcher so strict builds do not fall back to legacy
    model bake.
    """
    def bake(block, dim_positions, S):
        if alu_mode == "efficient":
            return None
        from ...vm_step import _set_layer12_mul_combine
        _set_layer12_mul_combine(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer12_mul_combine",
        phase=12,
        reads={"MARK_AX", "TEMP", "OP_MUL"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=12,
        migrated=True,
        # Staleness invariants: L12 MUL combine consumes the same operands
        # as L11 MUL partial (ALU_HI for a_hi, AX_CARRY_LO for b_lo) at the
        # AX marker. These must be the in-step fresh values. Also consumes
        # the fresh TEMP[partial] just written by ``layer11_mul_partial`` at
        # the AX marker (phase 11 < 12).
        consumes_fresh={
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "TEMP": "AX_byte0",
        } if alu_mode == "lookup" else {},
        # Produces the fresh MUL hi-nibble result at the AX marker (gated
        # on MARK_AX + OP_MUL): ``result_hi = (partial + a_hi*b_lo) % 16``
        # is written via 4-way AND units into OUTPUT_HI. ``_set_layer12_mul_combine``
        # itself writes only OUTPUT_HI (not OUTPUT_LO); the lo-nibble was
        # already populated upstream in L10's MUL units, so the staleness
        # contract only covers the hi half emitted here.
        produces={
            "OUTPUT_HI_THIS_STEP": "AX_byte0",
        },
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
    )
