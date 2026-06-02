"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
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


# Declarative-migration boundary: a_lo slabs ``[0, this)`` are baked via
# ``CompilerIR`` / ``FFNRule`` lowering; slabs ``[this, 16)`` still go
# through the inlined imperative tail in
# ``_bake_layer11_mul_partial_imperative_tail``. This constant advances
# 4 -> 8 -> 12 -> 16 across the Wave 4D substage commits; the final
# commit removes the imperative tail entirely and attaches the full
# ``compiler_ir`` to the op.
_L11_MUL_PARTIAL_MIGRATED_END_A_LO = 16


# === Declarative FFNRule generators for L11 MUL partial ===============
#
# Each a_lo slab walks the (b_lo, b_hi) grid (16 * 16 = 256 units). For
# every (a_lo, b_lo, b_hi) triple the helper bakes a 4-way AND unit that
# fires when ``MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo] +
# AX_CARRY_HI[b_hi]`` is hot, gated by ``OP_MUL``, and writes
# ``10.0/S`` into ``TEMP[partial]`` where
# ``partial = ((a_lo * b_lo) // 16 + a_lo * b_hi) % 16``.
#
# The lowerer (``CompilerIR.lower_ffn``) multiplies condition weights and
# the threshold by ``S`` but leaves ``writes`` / ``gate_weight`` /
# ``gate_bias`` unscaled. So:
#   ffn.W_up[unit, MARK_AX]            = S        -> ("MARK_AX",            1.0)
#   ffn.W_up[unit, ALU_LO + a_lo]      = S        -> (f"ALU_LO+{a_lo}",     1.0)
#   ffn.W_up[unit, AX_CARRY_LO + b_lo] = S        -> (f"AX_CARRY_LO+{b_lo}", 1.0)
#   ffn.W_up[unit, AX_CARRY_HI + b_hi] = S        -> (f"AX_CARRY_HI+{b_hi}", 1.0)
#   ffn.b_up[unit]                     = -S * 3.5 -> threshold = 3.5
#   ffn.W_gate[unit, OP_MUL]           = 1.0      -> gated_write(gate="OP_MUL",
#                                                                 gate_weight=1.0,
#                                                                 gate_bias=0.0)
#   ffn.W_down[TEMP + partial, unit]   = 10.0 / S -> writes=((f"TEMP+{partial}",
#                                                              10.0 / S),)
#
# Substage granularity = per a_lo slab (256 rules) so the migration can
# proceed in 16 byte-identical, independently-bisectable commits while
# matching the existing pinned offsets in
# ``_L11_MUL_PARTIAL_UNIT_LAYOUT``.


def _layer11_mul_partial_rules_for_a_lo(
    a_lo: int, S: float
) -> tuple[FFNRule, ...]:
    """Return the 256 ``FFNRule``s for one ``a_lo`` slab.

    The rules are emitted in the same ``(b_lo, b_hi)`` order as
    ``_set_layer11_mul_partial`` so the lowering cursor lands on the
    historical hidden-unit indices (slab ``a_lo`` occupies units
    ``a_lo*256 .. (a_lo+1)*256``).
    """
    if not 0 <= a_lo < 16:
        raise ValueError(f"a_lo must be in [0, 16), got {a_lo}")
    rules: list[FFNRule] = []
    for b_lo in range(16):
        carry = (a_lo * b_lo) // 16
        for b_hi in range(16):
            partial = (carry + a_lo * b_hi) % 16
            rules.append(FFNRule.gated_write(
                name=f"l11_mul_partial_a{a_lo:02d}_b{b_lo:02d}_h{b_hi:02d}",
                conditions=(
                    ("MARK_AX", 1.0),
                    (f"ALU_LO+{a_lo}", 1.0),
                    (f"AX_CARRY_LO+{b_lo}", 1.0),
                    (f"AX_CARRY_HI+{b_hi}", 1.0),
                ),
                threshold=3.5,
                gate="OP_MUL",
                gate_weight=1.0,
                gate_bias=0.0,
                writes=((f"TEMP+{partial}", 10.0 / S),),
            ))
    return tuple(rules)


def _layer11_mul_partial_rules(S: float) -> tuple[FFNRule, ...]:
    """Return the full 4096-rule ``FFNRule`` sequence for the L11 MUL partial.

    Concatenates ``_layer11_mul_partial_rules_for_a_lo`` for ``a_lo`` in
    ``range(16)`` so the lowering cursor walks 0..4096 with no gap, matching
    the historical ``_set_layer11_mul_partial`` unit numbering.
    """
    rules: list[FFNRule] = []
    for a_lo in range(16):
        rules.extend(_layer11_mul_partial_rules_for_a_lo(a_lo, S))
    return tuple(rules)


def _layer11_mul_partial_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` for the L11 MUL partial.

    Exposed via the op's ``compiler_ir=`` so symbolic execution,
    ``compare_symbolic_to_lowered_ffn``, and the declarative verifier
    can read the rules without going through the bake.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer11_mul_partial_rules(S))
    return ir


def _lower_layer11_mul_partial_rules_range(
    ffn,
    S: float,
    BD,
    *,
    start_a_lo: int,
    end_a_lo: int,
) -> int:
    """Lower a contiguous ``a_lo`` slab range into ``ffn``.

    Returns the post-bake unit cursor. Used during the multi-commit
    migration so each substage commit can lower its slabs declaratively
    while the remaining slabs continue through the inlined imperative
    tail below.
    """
    rules: list[FFNRule] = []
    for a_lo in range(start_a_lo, end_a_lo):
        rules.extend(_layer11_mul_partial_rules_for_a_lo(a_lo, S))
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=start_a_lo * 256,
        S=S,
    )


def _bake_layer11_mul_partial_imperative_tail(
    ffn,
    S: float,
    BD,
    *,
    start_a_lo: int,
    start_unit: int,
) -> int:
    """Inlined ``_set_layer11_mul_partial`` body, restricted to ``a_lo``
    slabs ``[start_a_lo, 16)``.

    Mirror of ``setup_helpers._set_layer11_mul_partial`` so the multi-
    commit migration can advance the declarative boundary
    ``start_a_lo`` one substage at a time without modifying the legacy
    helper (which other tests still call as a single 4096-unit bake).
    Final commit of the migration removes the call to this tail entirely.
    """
    assert start_unit == start_a_lo * 256, (
        f"L11 imperative tail expects start_unit = start_a_lo * 256, "
        f"got start_unit={start_unit} for start_a_lo={start_a_lo}"
    )
    unit = start_unit
    for a_lo in range(start_a_lo, 16):
        for b_lo in range(16):
            carry = (a_lo * b_lo) // 16
            for b_hi in range(16):
                partial = (carry + a_lo * b_hi) % 16
                # 4-way AND: MARK_AX + ALU_LO[a_lo] + AX_CARRY_LO[b_lo] + AX_CARRY_HI[b_hi]
                ffn.W_up[unit, BD.MARK_AX] = S
                ffn.W_up[unit, BD.ALU_LO + a_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_LO + b_lo] = S
                ffn.W_up[unit, BD.AX_CARRY_HI + b_hi] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.W_gate[unit, BD.OP_MUL] = 1.0
                # 10.0/S so hot TEMP[partial] lands at ~5.0 (L12 threshold).
                ffn.W_down[BD.TEMP + partial, unit] = 10.0 / S
                unit += 1
    return unit


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
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L11 MUL partial slab (one per
        # ``a_lo``) is pinned to its existing offset so the calls below
        # land byte-identically. The block-level attribute mirrors the
        # ``_l14_unit_counter`` convention used by sibling layers, but
        # carries the allocator object so the layout is structured, not
        # just a monotonic int. Downstream tools (e.g. a future L11 op
        # family widening ``layer_max_units=``) can introspect or extend
        # it here.
        allocator = _allocate_layer11_mul_partial_units()
        block.ffn._l11_unit_allocator = allocator

        # Migrated portion (declarative FFNRule lowering): a_lo slabs
        # ``[0, _L11_MUL_PARTIAL_MIGRATED_END_A_LO)``. The remaining
        # slabs run through ``_bake_layer11_mul_partial_imperative_tail``
        # below. Both halves are byte-identical to the legacy
        # ``setup_helpers._set_layer11_mul_partial`` -- verified via
        # ``compare_symbolic_to_lowered_ffn`` per substage commit.
        next_unit = _lower_layer11_mul_partial_rules_range(
            block.ffn, S, proxy,
            start_a_lo=0,
            end_a_lo=_L11_MUL_PARTIAL_MIGRATED_END_A_LO,
        )
        # Imperative tail covers the unmigrated a_lo slabs.
        next_unit = _bake_layer11_mul_partial_imperative_tail(
            block.ffn, S, proxy,
            start_a_lo=_L11_MUL_PARTIAL_MIGRATED_END_A_LO,
            start_unit=next_unit,
        )
        # Byte-identity guard: combined declarative + imperative cursor
        # MUST end exactly at the allocator's total footprint. If the
        # layout table drifts from the rules / helper writes, this
        # assertion fires before any weight surgery propagates downstream.
        assert next_unit == _L11_MUL_PARTIAL_TOTAL_UNITS, (
            f"L11 MUL partial unit cursor drift: bake returned "
            f"{next_unit}, allocator expected "
            f"{_L11_MUL_PARTIAL_TOTAL_UNITS}"
        )

    return Operation(
        name="layer11_mul_partial",
        phase=11,
        # ``_set_layer11_mul_partial`` reads ALU_LO[a_lo], AX_CARRY_LO[b_lo],
        # AX_CARRY_HI[b_hi], MARK_AX, gates on OP_MUL, writes TEMP[partial].
        # It does NOT read ALU_HI -- that's L12's job (``a_hi`` lookup).
        # Declaring a phantom ALU_HI read here understates L11's true producer
        # role and inflates ALU_HI's apparent in-step consumer count, which
        # makes the staleness analyzer harder to interpret. Removed.
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"TEMP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=11,
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G): the L11 MUL partial unit
        # consumes ALU_LO (operand A low nibble) and AX_CARRY_LO/HI (operand B
        # nibbles) at the AX marker for OP_MUL. All must be the *current*
        # step's fresh values to produce the correct partial product. The
        # lookup bake stages the partial in TEMP[0..15] (not
        # MUL_ACCUM/FETCH_LO), and L12 consumes that fresh same-step TEMP
        # value plus its own fresh ALU_HI / AX_CARRY_LO lookup. ALU_HI is NOT
        # consumed here -- the L11 helper writes ``a_hi``-independent
        # ``partial = (carry + a_lo * b_hi) % 16`` entries.
        consumes_fresh={
            "ALU_LO": "AX_byte0",
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
