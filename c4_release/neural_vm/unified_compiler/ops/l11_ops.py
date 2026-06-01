"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


# === L11 FFN unit layout (pinned offsets) ===========================
#
# The ``layer11_mul_partial`` op owns the entire L11 FFN. The actual
# weight writes happen inside ``setup_helpers._set_layer11_mul_partial``
# (re-exported via ``vm_step``), which walks a local ``unit = 0`` counter
# through a schoolbook ``(a_lo, b_lo, b_hi)`` triple-loop and fills all
# 4096 hidden units (16^3). Migration to :class:`FFNUnitAllocator` keeps
# that helper byte-identical -- we declare each ``a_lo`` slab at its
# existing pinned offset so the layout is auditable rather than implicit.
# Adding a new L11 op family later will go through ``allocator.alloc(name,
# n)`` without a pin, but since the MUL partial helper already saturates
# the 4096-unit pool there are no free gaps; a future op family would have
# to widen ``layer_max_units=`` or evict a slab.
#
# The offsets below mirror the unit-counter walk in
# ``_set_layer11_mul_partial``. The outer loop is over ``a_lo in
# range(16)``; each iteration writes ``16 (b_lo) * 16 (b_hi) = 256`` units
# at offset ``a_lo * 256``. Changing the helper's loop structure requires
# updating this table in lock-step.
_L11_MUL_PARTIAL_UNIT_LAYOUT = tuple(
    # (sub-stage name, pinned start, n_units)
    (f"layer11_mul_partial.a_lo_{a_lo:02d}", a_lo * 256, 256)
    for a_lo in range(16)
)


def _allocate_layer11_mul_partial_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the L11 MUL partial.

    Every ``a_lo`` slab is pinned at its existing offset so the underlying
    ``_set_layer11_mul_partial`` helper -- which writes via its own
    monotonic ``unit = 0`` counter -- lands on exactly the same hidden-unit
    indices it always has. This call is byte-identical bookkeeping: the
    allocator declares ranges by name, the helper writes the weights. A
    future refactor can split the monolithic helper into per-slab bake
    functions that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L11 op that widens ``layer_max_units=`` and claims a fresh
    range past unit 4096).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L11_MUL_PARTIAL_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


# Total unit footprint the helper is expected to consume. Computed from
# the layout table so a change to either side is loudly inconsistent.
_L11_MUL_PARTIAL_TOTAL_UNITS = sum(
    n_units for _, _, n_units in _L11_MUL_PARTIAL_UNIT_LAYOUT
)


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
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L11 MUL partial slab (one per
        # ``a_lo``) is pinned to its existing offset so the call below
        # lands byte-identically. The block-level attribute mirrors the
        # ``_l14_unit_counter`` convention used by sibling layers, but
        # carries the allocator object so the layout is structured, not
        # just a monotonic int. Downstream tools (e.g. a future L11 op
        # family widening ``layer_max_units=``) can introspect or extend
        # it here.
        allocator = _allocate_layer11_mul_partial_units()
        block.ffn._l11_unit_allocator = allocator

        n11 = _set_layer11_mul_partial(block.ffn, S, proxy)
        # Byte-identity guard: the helper's local cursor MUST end exactly
        # at the allocator's total footprint. If the layout table drifts
        # from the helper's writes, this assertion fires before any
        # weight surgery propagates downstream.
        assert n11 == _L11_MUL_PARTIAL_TOTAL_UNITS, (
            f"L11 MUL partial unit cursor drift: helper returned {n11}, "
            f"allocator expected {_L11_MUL_PARTIAL_TOTAL_UNITS}"
        )

    return Operation(
        name="layer11_mul_partial",
        phase=11,
        reads={"MARK_AX", "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"TEMP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=11,
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G): the L11 MUL partial unit
        # consumes ALU_LO/HI (operand A) and AX_CARRY_LO/HI (operand B) at
        # the AX marker for OP_MUL. Both must be the *current* step's fresh
        # values to produce the correct partial product. The lookup bake
        # stages the partial in TEMP[0..15] (not MUL_ACCUM/FETCH_LO), and L12
        # consumes that fresh same-step TEMP value.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        } if alu_mode == "lookup" else {},
        produces={
            "TEMP": "AX_byte0",
        } if alu_mode == "lookup" else {},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
    )
