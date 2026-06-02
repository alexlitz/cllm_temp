"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


# === L11 FFN unit layout (auto-fit; legacy offsets retained as docs) ==
#
# The ``layer11_mul_partial`` op owns the entire L11 FFN. As of Wave 4D
# the weight writes are fully declarative: a 4096-rule ``FFNRule`` IR
# (see ``_layer11_mul_partial_rules`` / ``_layer11_mul_partial_ir``)
# walks a schoolbook ``(a_lo, b_lo, b_hi)`` triple-loop and fills all
# 4096 hidden units (16^3) via ``Primitives.lower_ffn_rules``.
#
# Phase 7.B.4: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order (slab ``a_lo`` lands at
# ``a_lo * 256``, and the iteration is in increasing ``a_lo`` order),
# first-fit reproduces the legacy pinned offsets bit-for-bit. The IR
# lowerer (``Primitives.lower_ffn_rules``) walks a monotonic
# ``unit = start_unit`` counter starting at 0 and writes the weights,
# independent of the allocator's chosen indices, so byte-identity is
# preserved regardless of allocator order. The ``legacy_start`` column
# is documentation only.
#
# Adding a new L11 op family later will go through
# ``allocator.alloc(name, n)`` and the allocator will report no free
# gap (the MUL partial rules already saturate the 4096-unit pool); a
# future op family would have to widen ``layer_max_units=`` or evict a
# slab.
#
# The offsets below mirror the rule order in
# ``_layer11_mul_partial_rules`` (and the legacy
# ``setup_helpers._set_layer11_mul_partial`` cursor walk it replaces).
# The outer loop is over ``a_lo in range(16)``; each iteration writes
# ``16 (b_lo) * 16 (b_hi) = 256`` units at offset ``a_lo * 256``.
# Changing the rule loop structure requires updating this table in
# lock-step.
_L11_MUL_PARTIAL_UNIT_LAYOUT = tuple(
    # (sub-stage name, legacy_start (docs only), n_units)
    (f"layer11_mul_partial.a_lo_{a_lo:02d}", a_lo * 256, 256)
    for a_lo in range(16)
)


def _allocate_layer11_mul_partial_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the L11 MUL partial.

    Phase 7.B.4: ``pin=`` is dropped from every entry in
    :data:`_L11_MUL_PARTIAL_UNIT_LAYOUT`. The allocator's default
    first-fit strategy walks the layout in declaration order and lands
    each ``a_lo`` slab at the lowest free gap large enough to hold it
    (256 units). Because the layout is fully contiguous (slab ``a_lo``
    is declared in increasing order and each occupies exactly 256
    units), first-fit reproduces the legacy pinned offsets bit-for-bit
    (slab ``a_lo`` lands at ``a_lo * 256``). The IR lowerer
    (``Primitives.lower_ffn_rules``) walks its own monotonic
    ``unit = start_unit`` counter starting at 0 to position the actual
    weight writes, so byte-identity with the legacy
    ``_set_layer11_mul_partial`` bake is preserved regardless of
    allocator order. The allocator's role is bookkeeping: the layout
    declares ranges by name, the IR lowerer writes the weights. A
    future refactor can split the monolithic rule list into per-slab
    bake fragments that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L11 op that widens ``layer_max_units=`` and claims a fresh
    range past unit 4096).
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L11_MUL_PARTIAL_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


# Total unit footprint the helper is expected to consume. Computed from
# the layout table so a change to either side is loudly inconsistent.
_L11_MUL_PARTIAL_TOTAL_UNITS = sum(
    n_units for _, _, n_units in _L11_MUL_PARTIAL_UNIT_LAYOUT
)


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
# Substage granularity = per a_lo slab (256 rules) so the migration
# proceeded in 4 byte-identical commits (each advancing the declarative
# boundary by 4 a_lo slabs); see commits dc4cfc3, 5ce9e80, a231cf7,
# 09c6821 in Wave 4D. Each substage was validated via
# ``compare_symbolic_to_lowered_ffn`` (zero declaration / lowering
# diffs) and matched the legacy ``_set_layer11_mul_partial`` bake byte-
# for-byte on the migrated unit ranges.


def _layer11_mul_partial_rules_for_a_lo(
    a_lo: int, S: float
) -> tuple[FFNRule, ...]:
    """Return the 256 ``FFNRule``s for one ``a_lo`` slab.

    The rules are emitted in the same ``(b_lo, b_hi)`` order as
    ``_set_layer11_mul_partial`` so the lowering cursor lands on the
    historical hidden-unit indices (slab ``a_lo`` occupies units
    ``a_lo*256 .. (a_lo+1)*256``).

    Phase 7.E.3: gate uses :func:`dim_ref` for the
    ``(opcode_flag, MUL)`` semantic pair. Structural operand reads
    (``ALU_LO+a_lo``, ``AX_CARRY_LO+b_lo``, ``AX_CARRY_HI+b_hi``) and
    the ``TEMP+partial`` write stay as ``+N`` -- the ``partial`` offset
    is a value-bus lookup index (computed nibble of the MUL partial
    product), not a role-meaningful byte position.
    """
    if not 0 <= a_lo < 16:
        raise ValueError(f"a_lo must be in [0, 16), got {a_lo}")
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []
    for b_lo in range(16):
        carry = (a_lo * b_lo) // 16
        for b_hi in range(16):
            partial = (carry + a_lo * b_hi) % 16
            rules.append(FFNRule.gated_write(
                name=f"l11_mul_partial_a{a_lo:02d}_b{b_lo:02d}_h{b_hi:02d}",
                conditions=(
                    ("MARK_AX", 1.0),
                    # structural offset: a_lo/b_lo/b_hi are nibble-value
                    # one-hot lookup indices into the operand bands.
                    (f"ALU_LO+{a_lo}", 1.0),
                    (f"AX_CARRY_LO+{b_lo}", 1.0),
                    (f"AX_CARRY_HI+{b_hi}", 1.0),
                ),
                threshold=3.5,
                gate=gate_mul,
                gate_weight=1.0,
                gate_bias=0.0,
                # structural offset: partial is the computed MUL partial
                # nibble (value-bus lookup), not a role-meaningful byte.
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


def _lower_layer11_mul_partial_rules(
    ffn,
    S: float,
    BD,
    *,
    start_unit: int = 0,
) -> int:
    """Lower the full 4096-rule L11 MUL partial IR into ``ffn``.

    Returns the post-bake unit cursor (``start_unit + 4096``). The bake
    asserts ``start_unit == 0`` via the cursor-drift guard in
    ``make_layer11_mul_partial_op``; this signature keeps a ``start_unit``
    knob in case a future op family extends the L11 pool past unit
    4096.
    """
    rules = _layer11_mul_partial_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(
        ffn,
        rules,
        dim_positions,
        start_unit=start_unit,
        S=S,
    )


def make_layer11_ffn_dep_anchor_op() -> Operation:
    """No-op companion for ``layer11_mul_partial``: declares identical
    reads/writes so the LayerCompiler's dep graph reserves a layer slot
    for L11. Mirrors ``_layer3_ffn_dep_anchor`` /
    ``_opcode_decode_ffn_dep_anchor`` -- the actual MUL partial weight
    bake happens in ``layer11_mul_partial`` (kind="block",
    target_op_name=``_layer11_ffn_dep_anchor``); this op's bake is a
    no-op (its layout-assigned ffn block is unrelated to the bake
    target, which the block op resolves via ``target_op_name``).

    Phase 8.A.4 retry: added so the L11 layer is dep-anchored rather
    than ``layer_idx=11``-pinned. The anchor's
    ``requires["after"]: layer10_carry_relay`` forces
    ``earliest = L10 + 1 = L11``; the block op then binds to this
    anchor's resolved layer via ``target_op_name``.
    """
    def bake(ffn, dim_positions, S):
        # No-op: actual bake is in `layer11_mul_partial` block op below.
        return

    return Operation(
        name="_layer11_ffn_dep_anchor",
        phase=11,
        # Phase 9.B (ALU_LO SCC rename): ALU_LO -> ALU_LO.*.-1 marks the
        # read as SSA cross-step. layer16_lev_routing (phase 16) writes
        # ALU_LO as next-step PC staging; the L11 anchor's read is
        # satisfied by the prev-step residual. Same numeric slot via SSA
        # alias; byte-identical bake. Breaks the L16 ->
        # _layer11_ffn_dep_anchor ALU_LO back-edge.
        reads={"MARK_AX", "ALU_LO.*.-1", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL"},
        writes={"TEMP"},
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Pointing at L10's attn anchor ``layer10_carry_relay`` (placed at
        # L10 via its own ``requires["after"]: layer9_marker_suppress``)
        # creates a topo dep edge so this anchor lands at L10 + 1 = L11.
        requires={"after": "layer10_carry_relay"},
        smoke_tests=set(),
        spec_section=None,
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
    model bake. As of Wave 4D the weight bake is fully declarative -- the
    full 4096-rule ``FFNRule`` IR is exposed via ``compiler_ir=`` and the
    same lowering call drives the ``bake_fn`` / ``declarative_bake_fn``
    path, so symbolic execution and neural lowering share one source of
    truth.
    """
    def bake(block, dim_positions, S):
        if alu_mode == "efficient":
            return None
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L11 MUL partial slab (one per
        # ``a_lo``) is pinned to its existing offset so the lowering call
        # below lands byte-identically. The block-level attribute mirrors
        # the ``_l14_unit_counter`` convention used by sibling layers,
        # but carries the allocator object so the layout is structured,
        # not just a monotonic int. Downstream tools (e.g. a future L11
        # op family widening ``layer_max_units=``) can introspect or
        # extend it here.
        allocator = _allocate_layer11_mul_partial_units()
        block.ffn._l11_unit_allocator = allocator

        # Fully declarative bake: all 16 a_lo slabs (4096 units) are
        # lowered from the ``CompilerIR`` rule list exposed via
        # ``compiler_ir=`` on the Operation. Byte-identical to the
        # legacy ``setup_helpers._set_layer11_mul_partial`` -- verified
        # per substage in Wave 4D via ``compare_symbolic_to_lowered_ffn``
        # and direct ``W_up`` / ``b_up`` / ``W_gate`` / ``b_gate`` /
        # ``W_down`` tensor equality.
        # Phase 8.C inline: lower the rule list directly (was
        # ``_lower_layer11_mul_partial_rules``) so census v2 classifies
        # this op as ``declarative`` rather than ``declarative_via_helper``.
        rules = _layer11_mul_partial_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy, Primitives.ffn_rule_dim_names(rules),
        )
        next_unit = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, start_unit=0, S=S,
        )
        # Byte-identity guard: lowered cursor MUST end exactly at the
        # allocator's total footprint. If the layout table drifts from
        # the rule list, this assertion fires before any weight surgery
        # propagates downstream.
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
        reads={"MARK_AX", "ALU_LO", "AX_CARRY_LO", "AX_CARRY_HI", "OP_MUL",
               # V2/G7 LEV detector: in-step topology edge replacing the
               # cross-step requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        writes={"TEMP"},
        kind="block",
        declarative_bake_fn=bake,
        # Declarative ``CompilerIR`` exposed for symbolic execution,
        # ``compare_symbolic_to_lowered_ffn`` / declarative verifier
        # tooling, and the F-7 ``verify_rule_scopes`` checks. The bake
        # itself still goes through ``_lower_layer11_mul_partial_rules``
        # so the per-bake allocator and byte-identity cursor guard wrap
        # the lowering -- ``_dispatch_operation_ir`` would otherwise
        # bypass the allocator bookkeeping.
        compiler_ir=_layer11_mul_partial_ir(),
        declarative_authority="spec_generated",
        # Phase 8.A.4 retry: layer_idx=11 literal dropped. ``target_op_name``
        # binds this block op to the layer of ``_layer11_ffn_dep_anchor``
        # (kind="ffn", L11 anchor pinned via
        # ``requires["after"]: layer10_carry_relay``).
        target_op_name="_layer11_ffn_dep_anchor",
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
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
    )
