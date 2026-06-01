"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


# === L12 FFN unit layout (pinned offsets) ===========================
#
# The ``layer12_mul_combine`` op owns the entire L12 FFN. The actual
# weight writes happen inside ``setup_helpers._set_layer12_mul_combine``,
# which uses a local ``unit = 0`` counter that increments through a
# 16 x 16 x 16 = 4096 sub-stage walk (partial, a_hi, b_lo). Migration
# to :class:`FFNUnitAllocator` keeps that helper byte-identical -- we
# just declare the sub-stage's range at its existing pinned offset so
# the layout is auditable rather than implicit. Adding a new L12 op
# family later will go through ``allocator.alloc(name, n)`` without a
# pin, and the allocator will pick the first free gap above 4096 (or
# any earlier gap, though none exist in the current bake).
#
# The offsets below mirror the unit-counter walk in
# ``_set_layer12_mul_combine``. Changing the helper's unit count
# requires updating this table in lock-step.
_L12_MUL_COMBINE_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer12_mul_combine.partials", 0, 4096),  # 16 partial x 16 a_hi x 16 b_lo
)


def _allocate_layer12_mul_combine_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with L12's MUL combine range.

    The sub-stage is pinned at its existing offset so the underlying
    ``setup_helpers._set_layer12_mul_combine`` helper -- which writes via
    its own monotonic ``unit = 0`` counter -- lands on exactly the same
    hidden-unit indices it always has. This call is byte-identical
    bookkeeping: the allocator declares the range by name, the helper
    writes the weights. A future refactor can split the monolithic
    helper into per-range bake functions that consume
    ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L12 op claims a free range past unit 4096 by widening the
    pool, or claims a sub-range if the helper is split).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L12_MUL_COMBINE_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


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
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. The L12 MUL combine sub-stage is
        # pinned to its existing offset so the call below lands
        # byte-identically. Exposed on the block for downstream tools
        # (e.g. a future L12 op family claiming a free gap), mirroring
        # the L9 convention introduced in ca775eb.
        allocator = _allocate_layer12_mul_combine_units()
        partials_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer12_mul_combine.partials"
        )
        block.ffn._l12_unit_allocator = allocator

        n12 = _set_layer12_mul_combine(block.ffn, S, proxy)
        # Byte-identity guard: the helper's local cursor MUST end exactly
        # where the allocator's partials range ends. If the table drifts
        # from the helper's writes, this assertion fires before any
        # downstream consumer reads the FFN.
        expected_end = partials_range.start + partials_range.n_units
        assert n12 == expected_end, (
            f"L12 MUL combine unit cursor drift: helper returned {n12}, "
            f"allocator expected {expected_end}"
        )

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
