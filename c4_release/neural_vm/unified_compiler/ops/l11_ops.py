"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os as _os_stack0

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule, step_function_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy
from .residual_band_registry import register_residual_band


# ---------------------------------------------------------------------------
# Op-local residual-band declarations: AX byte-1 + Root 2 STACK0 byte-0 carry.
# ---------------------------------------------------------------------------
# These are PRODUCTION-DEFAULT residual geometry (always present, NOT
# flag-gated): the carry/dump ops below bake unconditionally; only the LM-head
# EMISSION columns are flag-gated (``C4_AX_BYTE1_DUMP`` / ``C4_STACK0_B0_DUMP``,
# both default-ON), and those are model-level head bakes that don't change
# d_model. Every band passes ``never_share=True`` so the dim-liveness allocator
# keeps it in a PRIVATE slot — a merge onto a same-width donor whose lifetime
# "ended" would leave stale residue in the shared slot and corrupt the carried
# one-hot (observed: H1_DUMP_OUT slot 323 returned a stale 1.0). The registry
# threads these names into ``LayerCompiler._LIVENESS_NEVER_SHARE`` per-compile.
# Registration order here is load-bearing: it fixes the tail dim_positions
# (AX bands, then the four Root 2 PREV/DUMP bands, then the four bounded Root 2
# flag bands) — byte-identical to the legacy central dict ordering.
#
# (1) AX byte-1 register-dump cross-step carry. ``H1_PREV_STEP`` carries the
#     prev step's H1 one-hot from the L13 carry head to the L25 dump FFN
#     (``make_layer11_ax_byte1_dump_carry_op`` /
#     ``make_ax_byte1_dump_repopulate_op``); ``H1_DUMP_OUT`` carries the
#     re-supplied one-hot from the dump FFN to the LM head; ``AX_CARRY_OVERFLOW``
#     is the band-pass UPPER-cut kill flag written by
#     ``make_ax_byte1_carry_overflow_flag_op``. See
#     docs/AX_BYTE1_DUMP_CARRY_LANDED_2026_06_13.md.
register_residual_band(
    "H1_PREV_STEP", 7, owner="make_layer11_ax_byte1_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "H1_DUMP_OUT", 7, owner="make_ax_byte1_dump_repopulate_op",
    never_share=True,
)
register_residual_band(
    "AX_CARRY_OVERFLOW", 1, owner="make_ax_byte1_carry_overflow_flag_op",
    never_share=True,
)
# (2) Root 2 STACK0 byte-0 cross-step emission carry. The four PREV/DUMP bands
#     are the carry-head -> dump-FFN -> LM-head relay
#     (``make_stack0_byte0_dump_carry_op`` /
#     ``make_stack0_byte0_dump_repopulate_op``); the four bounded flag bands are
#     the band-pass gates (``make_stack0_byte0_{carried,sharp,prev_dom,not_cmp}
#     _flag_op``) that discriminate carried-comparison rows from arithmetic / JMP
#     over-fire rows. See memory note ``project_1096_first_reliable_baseline``.
register_residual_band(
    "STACK0_B0_H1_PREV", 7, owner="make_stack0_byte0_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_H3_PREV", 7, owner="make_stack0_byte0_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_DUMP_H1", 7, owner="make_stack0_byte0_dump_repopulate_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_DUMP_H3", 7, owner="make_stack0_byte0_dump_repopulate_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_CARRIED", 1, owner="make_stack0_byte0_carried_flag_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_SHARP", 1, owner="make_stack0_byte0_sharp_flag_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_PREV_DOM", 1, owner="make_stack0_byte0_prev_dom_flag_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_NOT_CMP", 1, owner="make_stack0_byte0_not_cmp_flag_op",
    never_share=True,
)


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

    Wave B Cluster 4 (2026-06-10): position marker migrated from
    ``MARK_AX`` to ``MARK_SE_ONLY``. The Wave A ``step_end_operand_relay``
    head (``make_layer11_step_end_operand_relay_op``, commit 10ca51a7)
    broadcasts ALU_LO, AX_CARRY_LO/HI, and OP_MUL from MARK_AX to
    MARK_SE_ONLY within the same step, so the same SwiGLU 4-way AND
    fires at STEP_END over identical operand state. The TEMP[partial]
    write is intermediate (not a byte-emit slot) and the lowering
    cursor / unit layout are unchanged, preserving byte-identity. See
    ``docs/WAVE_B_CLUSTER_4_PLAN_2026_06_10.md`` and
    ``docs/STEP_END_MIGRATION_TEMPLATE.md`` for the recipe.
    """
    if not 0 <= a_lo < 16:
        raise ValueError(f"a_lo must be in [0, 16), got {a_lo}")
    gate_mul = dim_ref("opcode_flag", "MUL")
    rules: list[FFNRule] = []
    for b_lo in range(16):
        carry = (a_lo * b_lo) // 16
        for b_hi in range(16):
            partial = (carry + a_lo * b_hi) % 16
            rules.append(multi_way_and_rule(
                name=f"l11_mul_partial_a{a_lo:02d}_b{b_lo:02d}_h{b_hi:02d}",
                conditions=(
                    # Wave B Cluster 4: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 1.0),
                    # structural offset: a_lo/b_lo/b_hi are nibble-value
                    # one-hot lookup indices into the operand bands.
                    (f"ALU_LO+{a_lo}", 1.0),
                    (f"AX_CARRY_LO+{b_lo}", 1.0),
                    (f"AX_CARRY_HI+{b_hi}", 1.0),
                ),
                threshold=3.5,
                gate=gate_mul,
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
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
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
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmokeBasic::test_mul_basic",
        },
        spec_section="BLOG_SPEC.md#multiplication-implementation",
        # Tier A opcode gating: ``_set_layer11_mul_partial`` writes every one
        # of its 4096 hidden units with ``W_gate[unit, BD.OP_MUL] = 1.0``, so
        # every unit's SiLU output is gated on OP_MUL. The L11 partial-sum
        # FFN fires ONLY on OP_MUL steps; non-MUL opcodes leave block 11
        # untouched.
        opcodes={"OP_MUL"},
    )


# === STEP_END operand relay (Wave A, 2026-06-10) =====================
#
# Two declarative attention heads at L11 that broadcast the in-step
# operand/dispatch state from MARK_AX (row offset 5) to MARK_SE (row
# offset 34) of the same step. Wave A of the STEP_END compute
# architecture (docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md):
# enables Wave B migration of L8/L9/L10 dispatch+ALU+CMP rules from
# MARK_AX gating to MARK_SE gating.
#
# Why L11? The relay sources include ALU_LO/HI and CMP, which are not
# written until L9. The L10 attention block is already at 12 heads with
# slots 0..10 owned; L11's attention block is fresh (no other op writes
# Q/K/V/O there today), so claiming heads 0/1 is collision-free.
#
# Why two heads? d_model=512, default num_heads=8, so HD=64 V/O slots per
# head. Total relay payload = 31 OP_<NAME> + 32 AX_CARRY + 32 ALU_LO/HI +
# 4 CMP + 4 STACK0_BYTE0..3 = ~103 dims. Two heads (128 slots) fit with
# headroom.
#
# Head 0 (operand relay A): OP_<NAME> (31) + AX_CARRY_LO (16) +
#                           AX_CARRY_HI (16) = 63 slots
# Head 1 (operand relay B): ALU_LO (16) + ALU_HI (16) + CMP (4) +
#                           STACK0_BYTE0..3 (4) = 40 slots
#
# Q anchors on MARK_SE_ONLY; K matches MARK_AX. Positive ALiBi slope
# (1.0) keeps the head step-local: the most-recent MARK_AX K (29 rows
# back from the current MARK_SE Q) wins by ~35 nats over the previous
# step's MARK_AX K (64 rows back), so the relay reads the current
# step's operand state, not a stale value. Mirrors the L1 IN_STEP_FRESH
# slope convention (see ``make_layer1_threshold_attn_op``).
_STEP_END_OPERAND_RELAY_OPCODES = (
    "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_OR", "OP_AND", "OP_XOR", "OP_SHL", "OP_SHR",
    "OP_IMM", "OP_PSH", "OP_JSR", "OP_ENT", "OP_LEV",
    "OP_LI", "OP_LC", "OP_SI", "OP_SC",
    "OP_LEA", "OP_JMP", "OP_BZ", "OP_BNZ", "OP_ADJ", "OP_EXIT",
)


_L11_STEP_END_OPERAND_RELAY_HEAD_LAYOUT = (
    ("layer11_step_end_operand_relay.head_0",),  # OP_<NAME> + AX_CARRY
    ("layer11_step_end_operand_relay.head_1",),  # ALU + CMP + STACK0_BYTE
)


def _allocate_layer11_step_end_operand_relay_heads() -> AttentionHeadAllocator:
    """Per-bake :class:`AttentionHeadAllocator` for the L11 relay heads.

    Fresh first-fit pool over L11 attention's 8-head budget. With no
    other L11 attention ops baking today the two relay heads land at
    indices 0 and 1; future L11 attention ops can claim free slots
    without touching this table.
    """
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (op_name,) in _L11_STEP_END_OPERAND_RELAY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=11)
    return allocator


def _layer11_step_end_operand_relay_head_specs(
    BD,
    S: float,
    head_a_idx: int,
    head_b_idx: int,
    *,
    include_op_name: bool = True,
    include_ax_carry: bool = True,
    include_alu: bool = True,
    include_cmp: bool = True,
    include_stack0_byte: bool = True,
    op_name_subset: tuple = (),
) -> tuple[DeclarativeAttentionHeadSpec, DeclarativeAttentionHeadSpec]:
    """Build the two Wave A relay head specs.

    Q at MARK_SE_ONLY (the STEP_END row). K at MARK_AX (the in-step
    operand-state row). V copies each named source dim with weight 1.0;
    O writes the same dim back to the Q (MARK_SE) row with weight 1.0.

    A positive ALiBi slope (1.0) plus L=S Q/K weights keeps the relay
    step-local: at distance 29 (within-step MARK_AX -> MARK_SE) the score
    is ``L^2 / sqrt(HD) - 29``; the prior step's MARK_AX sits at
    distance ~64 (next-step offset 35 + 29), losing the softmax by
    ~35 nats. Mirrors the L1 IN_STEP_FRESH / HAS_SE broadcast pattern.

    Per-payload toggles (``include_*``) let the op restrict which V/O
    bands the relay broadcasts. The default relays every payload (the
    Wave-A-full configuration); ``enable=True`` callers can pass
    narrower toggles to scope which downstream consumers see the
    relayed value at MARK_SE, e.g. omitting ``OP_<NAME>`` to keep
    BZ/BNZ / dispatch-gated rules MARK_AX-pure.
    """
    L = float(S)
    HD_DEFAULT = 64  # default head_dim at L11 (d_model=512, num_heads=8)

    # Per-head Q/K bands: select MARK_SE rows on Q side, MARK_AX rows
    # on K side. The CONST anchor (Q slot 0 = -L) cancels at non-MARK_SE
    # Q rows so the slot-0 score collapses to ~-L^2 / sqrt(HD).
    q_band = (
        AP(0, BD.MARK_SE_ONLY, L),
    )
    k_band = (
        AP(0, BD.MARK_AX, L),
    )

    # --- Head A: OP_<NAME> + AX_CARRY_LO/HI ---------------------------
    v_a: list = []
    o_a: list = []
    slot = 0
    if include_op_name:
        op_names_iter = (
            op_name_subset if op_name_subset else _STEP_END_OPERAND_RELAY_OPCODES
        )
        for op_name in op_names_iter:
            dim = getattr(BD, op_name)
            v_a.append(AP(slot, dim, 1.0))
            o_a.append(AO(dim, slot, 1.0))
            slot += 1
    if include_ax_carry:
        for k_idx in range(16):
            v_a.append(AP(slot, BD.AX_CARRY_LO + k_idx, 1.0))
            o_a.append(AO(BD.AX_CARRY_LO + k_idx, slot, 1.0))
            slot += 1
        for k_idx in range(16):
            v_a.append(AP(slot, BD.AX_CARRY_HI + k_idx, 1.0))
            o_a.append(AO(BD.AX_CARRY_HI + k_idx, slot, 1.0))
            slot += 1
    assert slot <= HD_DEFAULT, (
        f"step_end_operand_relay head A overflowed HD={HD_DEFAULT} "
        f"with {slot} slots"
    )

    spec_a = DeclarativeAttentionHeadSpec(
        head_idx=head_a_idx,
        q=q_band,
        k=k_band,
        v=tuple(v_a),
        o=tuple(o_a),
        alibi_slope=1.0,
    )

    # --- Head B: ALU_LO/HI + CMP + STACK0_BYTE0..3 --------------------
    v_b: list = []
    o_b: list = []
    slot = 0
    if include_alu:
        for k_idx in range(16):
            v_b.append(AP(slot, BD.ALU_LO + k_idx, 1.0))
            o_b.append(AO(BD.ALU_LO + k_idx, slot, 1.0))
            slot += 1
        for k_idx in range(16):
            v_b.append(AP(slot, BD.ALU_HI + k_idx, 1.0))
            o_b.append(AO(BD.ALU_HI + k_idx, slot, 1.0))
            slot += 1
    if include_cmp:
        for k_idx in range(4):  # CMP is 4 wide in the registry
            v_b.append(AP(slot, BD.CMP + k_idx, 1.0))
            o_b.append(AO(BD.CMP + k_idx, slot, 1.0))
            slot += 1
    if include_stack0_byte:
        for byte_h in (0, 1, 2, 3):
            dim = getattr(BD, f"STACK0_BYTE{byte_h}")
            v_b.append(AP(slot, dim, 1.0))
            o_b.append(AO(dim, slot, 1.0))
            slot += 1
    assert slot <= HD_DEFAULT, (
        f"step_end_operand_relay head B overflowed HD={HD_DEFAULT} "
        f"with {slot} slots"
    )

    spec_b = DeclarativeAttentionHeadSpec(
        head_idx=head_b_idx,
        q=q_band,
        k=k_band,
        v=tuple(v_b),
        o=tuple(o_b),
        alibi_slope=1.0,
    )

    return spec_a, spec_b


# Default payload toggles for ``make_layer11_step_end_operand_relay_op``
# / ``_layer11_step_end_operand_relay_ir``. Mutated by ``make_*`` to thread
# the scope-restriction args through ``compiler_ir_factory`` (which the IR
# dispatcher invokes without forwarding kwargs). The factory reads this
# dict at lower-time; the bake closure captures its own copy.
_LAYER11_RELAY_PAYLOAD_DEFAULTS: dict = {
    "include_op_name": True,
    "include_ax_carry": True,
    "include_alu": True,
    "include_cmp": True,
    "include_stack0_byte": True,
    "op_name_subset": (),
}


def _layer11_step_end_operand_relay_ir(dim_positions, HD) -> CompilerIR:
    """``compiler_ir_factory`` for the L11 step_end_operand_relay heads."""
    del HD  # head_dim is layer-default; per-head fits within HD=64
    proxy = _as_setdim_proxy(dim_positions)
    allocator = _allocate_layer11_step_end_operand_relay_heads()
    by_name = {rec.op_name: rec.head_idx for rec in allocator.heads()}
    head_a_idx = by_name["layer11_step_end_operand_relay.head_0"]
    head_b_idx = by_name["layer11_step_end_operand_relay.head_1"]
    spec_a, spec_b = _layer11_step_end_operand_relay_head_specs(
        proxy, 100.0, head_a_idx, head_b_idx,
        **_LAYER11_RELAY_PAYLOAD_DEFAULTS,
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(
        spec_a, name="layer11_step_end_operand_relay.head_0",
    )
    ir.layer(0).attention.append(
        spec_b, name="layer11_step_end_operand_relay.head_1",
    )
    return ir


def make_layer11_step_end_operand_relay_op(
    enable: bool = False,
    *,
    include_op_name: bool = True,
    include_ax_carry: bool = True,
    include_alu: bool = True,
    include_cmp: bool = True,
    include_stack0_byte: bool = True,
    op_name_subset: tuple = (),
) -> Operation:
    """Wave A — broadcast operand/dispatch state from MARK_AX to MARK_SE.

    Per ``docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md``: two
    declarative attention heads at L11 that relay OP_<NAME>,
    AX_CARRY_LO/HI, ALU_LO/HI, CMP, and STACK0_BYTE0..3 from the
    MARK_AX K row (in-step row offset 5) to the MARK_SE Q row (offset
    34) of the same step, enabling Wave B migration of L8/L9/L10
    dispatch + ALU + CMP rules from MARK_AX gating to MARK_SE gating.

    Placed at L11 because:
      * ALU_LO/HI and CMP are written at L9 — the relay needs L10+.
      * AX_CARRY_LO/HI is written at L3/L6/L8 — visible by L11.
      * L10 attention is at 12-head capacity with slots 0..10 owned;
        L11 attention is currently empty so heads 0/1 are free.

    A positive ALiBi slope (1.0) keeps the relay step-local: the
    most-recent MARK_AX K (29 rows back from the current MARK_SE Q)
    wins by ~35 nats over the previous step's MARK_AX K (64 rows
    back), so the relay reads the current step's operand state.

    ``enable`` defaults to ``False`` (Wave A PoC gate). When False the
    op is registered so the dep graph + claims surface stay stable, but
    the bake is a no-op. When True (with the default full-payload
    toggles) the relay's downstream side-effects (extra ALU_LO/HI /
    OP_<NAME> signal at MARK_SE rows, picked up by the LM head + later
    layers' cross-step KV lookups) regress ~9 smoke tests on baseline.
    The per-payload toggles below let callers narrow the relay to the
    largest payload that keeps smoke neutral.

    Per-payload toggles
    -------------------
    Each toggle controls one V/O band in the two relay heads. Combined
    smoke-bisection on ``tests/test_smoke.py`` (Wave A scoping, 2026-
    06-10) found the maximum baseline-neutral payload:

      * ``include_op_name=True`` with the 27-opcode
        ``op_name_subset`` listed at the caller (excludes OP_IMM and
        OP_SHR; including those breaks ``test_add_basic`` /
        ``test_shr`` because the MARK_SE-side flag fires the L8
        MARK_SE_ONLY-migrated rules a second time over a stale
        same-step alias).
      * ``include_ax_carry=False`` — broadcasting AX_CARRY_LO/HI at
        MARK_SE clobbers the SI/LI memory tail (4 memory smoke tests +
        ADD_basic).
      * ``include_alu=False`` — ALU_LO/HI relay is the loudest race
        victim (10+ tests including BZ/BNZ and memory).
      * ``include_cmp=False`` — relayed CMP at MARK_SE racetracks the
        L10 cmp_combine override (test_eq_true + cmp_and_branch + 4
        memory tests).
      * ``include_stack0_byte=False`` — STACK0_BYTE relay nets -4
        smoke (gains or_basic but regresses ADD + 4 memory).

    ``op_name_subset`` empty means "relay every opcode in
    ``_STEP_END_OPERAND_RELAY_OPCODES``"; a non-empty tuple restricts
    the relay to the named opcodes only.

    Probe verification (``tools/probe_step_end_completeness.py``) shows
    that when ``enable=True`` the relay populates MARK_SE with the
    relayed operand band: AX_CARRY_LO/HI ~0.75 × source, ALU_LO/HI
    ~1.0 × source, STACK0_BYTE0..3 ~1.0 × source. OP_<NAME> + CMP need
    a longer-window programme to capture both MARK_AX and the
    same-step MARK_SE in one trace (the default ``IMM 5; PSH; IMM 5;
    EQ; EXIT`` runs OOM on a busy GPU before both rows materialise).
    """
    # Thread payload toggles through the module-level dict so
    # ``_layer11_step_end_operand_relay_ir`` (invoked via
    # ``compiler_ir_factory`` without kwargs by the IR dispatcher) sees
    # the same scope this op declares.
    _LAYER11_RELAY_PAYLOAD_DEFAULTS.update({
        "include_op_name": include_op_name,
        "include_ax_carry": include_ax_carry,
        "include_alu": include_alu,
        "include_cmp": include_cmp,
        "include_stack0_byte": include_stack0_byte,
        "op_name_subset": op_name_subset,
    })

    def bake(block, dim_positions, S):
        if not enable:
            return
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        head_allocator = _allocate_layer11_step_end_operand_relay_heads()
        attn._l11_step_end_relay_head_allocator = head_allocator
        head_a_idx = head_allocator.heads()[0].head_idx
        head_b_idx = head_allocator.heads()[1].head_idx
        HD = attn.W_q.shape[0] // attn.num_heads
        spec_a, spec_b = _layer11_step_end_operand_relay_head_specs(
            proxy, S, head_a_idx, head_b_idx,
            include_op_name=include_op_name,
            include_ax_carry=include_ax_carry,
            include_alu=include_alu,
            include_cmp=include_cmp,
            include_stack0_byte=include_stack0_byte,
            op_name_subset=op_name_subset,
        )
        Primitives.generate_attention_head(attn, spec_a, HD)
        Primitives.generate_attention_head(attn, spec_b, HD)

    # Dim-ownership claims: two heads, scoped by the ``include_*``
    # toggles above. Head A V slots cover the enabled subset of
    # OP_<NAME> (0..30), AX_CARRY_LO (16), AX_CARRY_HI (16). Head B V
    # slots cover the enabled subset of ALU_LO (16), ALU_HI (16),
    # CMP (4), STACK0_BYTE0..3 (4). O slots mirror V slot indices;
    # out_dim names match source dims (relay = identity copy).
    _claims: set = set()
    head_a_idx = 0
    head_b_idx = 1
    slot = 0
    if include_op_name:
        op_names_iter = (
            op_name_subset if op_name_subset else _STEP_END_OPERAND_RELAY_OPCODES
        )
        for op_name in op_names_iter:
            _claims.add((11, "attn_W_v", f"{head_a_idx}_{slot}", f"{op_name}+0"))
            _claims.add((11, "attn_W_o", f"{head_a_idx}_{slot}", f"{op_name}+0"))
            slot += 1
    if include_ax_carry:
        for k_idx in range(16):
            _claims.add(
                (11, "attn_W_v", f"{head_a_idx}_{slot}", f"AX_CARRY_LO+{k_idx}"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_a_idx}_{slot}", f"AX_CARRY_LO+{k_idx}"),
            )
            slot += 1
        for k_idx in range(16):
            _claims.add(
                (11, "attn_W_v", f"{head_a_idx}_{slot}", f"AX_CARRY_HI+{k_idx}"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_a_idx}_{slot}", f"AX_CARRY_HI+{k_idx}"),
            )
            slot += 1
    slot = 0
    if include_alu:
        for k_idx in range(16):
            _claims.add(
                (11, "attn_W_v", f"{head_b_idx}_{slot}", f"ALU_LO+{k_idx}"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_b_idx}_{slot}", f"ALU_LO+{k_idx}"),
            )
            slot += 1
        for k_idx in range(16):
            _claims.add(
                (11, "attn_W_v", f"{head_b_idx}_{slot}", f"ALU_HI+{k_idx}"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_b_idx}_{slot}", f"ALU_HI+{k_idx}"),
            )
            slot += 1
    if include_cmp:
        for k_idx in range(4):
            _claims.add(
                (11, "attn_W_v", f"{head_b_idx}_{slot}", f"CMP+{k_idx}"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_b_idx}_{slot}", f"CMP+{k_idx}"),
            )
            slot += 1
    if include_stack0_byte:
        for byte_h in (0, 1, 2, 3):
            _claims.add(
                (11, "attn_W_v", f"{head_b_idx}_{slot}", f"STACK0_BYTE{byte_h}+0"),
            )
            _claims.add(
                (11, "attn_W_o", f"{head_b_idx}_{slot}", f"STACK0_BYTE{byte_h}+0"),
            )
            slot += 1

    # Build reads/writes from the enabled payload subset. MARK_AX and
    # MARK_SE_ONLY are always read (Q/K anchors); per-payload dims are
    # only listed when their toggle is on so the dep graph + claims
    # surface stay tight to the actual bake.
    _reads = {"MARK_AX", "MARK_SE_ONLY"}
    _writes: set = set()
    if include_op_name:
        _op_names = (
            op_name_subset if op_name_subset else _STEP_END_OPERAND_RELAY_OPCODES
        )
        _reads.update(_op_names)
        _writes.update(_op_names)
    if include_ax_carry:
        _reads.update({"AX_CARRY_LO", "AX_CARRY_HI"})
        _writes.update({"AX_CARRY_LO", "AX_CARRY_HI"})
    if include_alu:
        _reads.update({"ALU_LO", "ALU_HI"})
        _writes.update({"ALU_LO", "ALU_HI"})
    if include_cmp:
        _reads.add("CMP")
        _writes.add("CMP")
    if include_stack0_byte:
        _reads.update({"STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"})
        _writes.update({"STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"})

    return Operation(
        name="layer11_step_end_operand_relay",
        # Q reads MARK_SE_ONLY (Q-row anchor); K reads MARK_AX (K-row
        # anchor). V reads the broadcast payload at the MARK_AX K rows.
        reads=_reads,
        # O writes the same dim names at the MARK_SE Q rows.
        writes=_writes,
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer11_step_end_operand_relay_ir,
        # Bind to the L11 FFN dep anchor so the block op lands at L11
        # alongside ``layer11_mul_partial``.
        target_op_name="_layer11_ffn_dep_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        claims=_claims,
        smoke_tests={"all"},
        spec_section="STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md#wave-a",
    )


# ---------------------------------------------------------------------------
# AX byte-1 DUMP carry head (the H1 cross-step SSA split)
# ---------------------------------------------------------------------------
#
# Root: the per-step AX byte-1 register dump reads its value EXCLUSIVELY from
# the ``H1`` band (L0 attn head-1, dims 67..73) as a one-hot ``H1+(byte1+2)``.
# That one-hot is born at block 10 (L9) on the producing (fresh-AX) step and is
# NEVER regenerated on a carried (non-AX-writing, e.g. PSH) step, so the dump
# emits 0x00 for byte 1 on carried steps. See
# ``docs/AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md``.
#
# Four prior agents proved no SINGLE-head carry that *writes ``H1``* can be
# scheduled: ``H1`` is read fresh (same-step) by 54 ops across L1..L16, and the
# carried-vs-fresh gate the carry head needs (``AX_CARRY``) is produced by ops
# that themselves read ``H1`` -> a 2-cycle the dim-only scheduler cannot break.
# See ``docs/AX_BYTE1_DUMP_CARRY_H1_WRITE_CYCLE_2026_06_13.md``.
#
# THE FIX (this op) is the H1 cross-step SSA split (mirror of the
# ``OUTPUT_HI`` -> ``OUTPUT_HI.*.-1`` split, Phase 7.A / 9.B):
#   * The head READS the previous step's H1 one-hot via the SSA cross-step
#     spelling ``H1.*.-1`` -- a DISTINCT dep-graph identity from the 54
#     same-step ``H1`` readers, so there is NO same-step back-edge.
#   * The head WRITES the re-supplied one-hot into ``H1_DUMP`` -- a same-slot
#     alias of ``H1`` (slots 67..73, declared via shared.py ``_ALIAS_OF``).
#     The LM head reads the PHYSICAL slots so emission is byte-identical; the
#     distinct *name* keeps the write off the 54 H1-reader dep edges. No cycle.
#
# Geometry (all measured spec_k=0 at the L11 read point = after block 13, see
# ``tools/probe_h1dump_design.py 13`` / ``tools/probe_carry_rowsig.py 13``):
#   * Byte-1 predictor row signature: ``ADDR_B0_LO+5`` ~= 0.97 on EVERY byte-1
#     predictor row (fresh + carried, all add programs), ~0 on the neighbouring
#     byte-0 / byte-2 rows. Fires the Q (current carried row) AND matches the K
#     (the prev fresh row that holds the one-hot).
#   * Carried-vs-fresh gate: ``Sum(AX_CARRY_LO+HI)`` ~= -988 on a FRESH-AX step
#     vs ~= +2.7 on a CARRIED step. Used BOTH on the Q side (gate the head ON
#     only on carried rows) AND on the K side (prefer the fresh prev row, whose
#     AX_CARRY is very negative, over the current carried row whose AX_CARRY is
#     ~0 -- this is what excludes the self-match that plain positive ALiBi
#     would otherwise pick).
#   * ALiBi: positive slope (recency) as a tie-breaker only; the AX_CARRY K
#     differential is the dominant prev-vs-current discriminator.
#
# Brittleness (documented, accepted for the add/sub corpus): ``H1`` is 7 wide
# so the ``H1+(v+2)`` one-hot caps byte-1 value at v=4. The whole 1096 add/sub
# corpus has high byte <= 4, so this carry covers it; a value-general fix needs
# a real 8-bit dump band (separate task).
# Host = L13 (physical block 16): a native logical layer with a REAL,
# non-passthrough attention block. Heads 0..5 are always occupied (0..2 =
# mem_addr_gather, 3 = bitwise_byte1, 4 = sub_minuend, 5 = add_addend). Head
# 6 is taken by ``layer13_mul_result_hi_relay`` whenever ``C4_MUL_WIDTH2`` is
# on (the production default since the mul width=2 landing) — two ops baking
# the SAME (layer, head) silently clobber each other and scramble the whole
# L13 attention block (observed: smoke 14/51 when the carry head ALSO pinned
# slot 6). The carry head is therefore pinned to slot 7, which is free at
# BOTH bake and runtime regardless of the mul flag (the head-dim-preserving
# widen adds a 9th head 8, also free, but slot 7 keeps the bake within the
# original 8-head band). Rejected hosts:
#   * L11 / block 14 — attention is shared and expands 8->13 heads at runtime,
#     clobbering any bake into heads 0..7.
#   * L12 / block 15 — attention block is a pure passthrough (W_q == 0 at
#     runtime), so a bake there has no effect on the forward.
# L13's read point (after block 15) holds both the crystallised AX_CARRY gate
# (-988 fresh / +2.7 carried) and the prev-step H1 one-hot.
_AX_BYTE1_DUMP_CARRY_HEAD_IDX = 7
_AX_BYTE1_DUMP_CARRY_HEAD_LAYOUT = (
    ("layer13_ax_byte1_dump_carry.head_7", _AX_BYTE1_DUMP_CARRY_HEAD_IDX),
)


def _allocate_layer13_ax_byte1_dump_carry_heads() -> AttentionHeadAllocator:
    """Per-bake head allocator for the AX byte-1 DUMP carry head (L13 host)."""
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (op_name, head_idx) in _AX_BYTE1_DUMP_CARRY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=13, pin=head_idx)
    return allocator


def _layer13_ax_byte1_dump_carry_head_spec(
    dim_positions: dict,
    head_idx: int,
    *,
    L: float = 15.0,
    sink_w: float = 8.0,
    k_axc_w: float = 0.2,
    alibi_slope: float = 0.5,
) -> DeclarativeAttentionHeadSpec:
    """Build the AX byte-1 DUMP carry head spec (``H1_PREV_STEP`` band).

    RE-ARCHITECTED (dedicated-band version): the carry head copies the
    PREVIOUS step's ``H1`` one-hot into the fresh ``H1_PREV_STEP`` band
    UNCONDITIONALLY (no carried-vs-fresh Q-gate, no sink-driven OFF state).
    The carried-vs-fresh gate is moved DOWNSTREAM to the
    ``ax_byte1_dump_repopulate`` FFN, which re-supplies ``H1_PREV_STEP`` into
    ``H1`` only on carried (non-AX-writing) steps via the crystallised
    ``AX_CARRY`` separation. This DISSOLVES the over-fire wall that blocked
    the prior ``H1_DUMP`` same-slot-alias attempt (whose Q-gate over-fired on
    fresh / multi-step rows and regressed smoke when enabled).

    Because ``H1_PREV_STEP`` is read by NOBODY except the gated dump FFN,
    writing the prev one-hot there on a fresh predictor row is harmless — the
    dump FFN gates it out. So the head only needs to ATTEND the correct row
    (the prior step's byte-1 predictor row that HOLDS the one-hot) and copy
    it; no Q-gate is required.

    Slots (V/O occupy 10..16 so they never collide with the Q/K gate slots):

    Q (byte-1 predictor row, fresh OR carried):
      slot 0 = ``ADDR_B0_LO+5`` * L  (fire on byte-1 predictor rows)
      slot 2 = ``CONST`` * L         (drives the one-hot-presence K-pref)
      slot 3 = ``ADDR_B0_LO+5`` * L  (drives the sink for rows with no prev
               one-hot, e.g. the very first step)
    K:
      slot 0 = ``ADDR_B0_LO+5`` * L  (match byte-1 predictor rows)
             - Sum_k (AX_CARRY_LO/HI + k) * k_axc_w  (prefer the FRESH prev
               row: its AX_CARRY ~= -988 -> huge +K boost; the current
               carried row's AX_CARRY ~= 0 -> no boost -> the prev fresh row
               wins, reinforced by positive-ALiBi recency)
      slot 2 = Sum_k (AX_CARRY + k) * -k_axc_w (fresh-preference: the prev FRESH
               predictor's very-negative AX_CARRY -> large +K bias, picking it
               over the current carried predictor)
      slot 3 = ``MARK_AX`` * sink_w   (V=0 sink: on the first step there is no
               prev predictor; the head falls onto an H1==0 marker row and
               writes ~0 into ``H1_PREV_STEP`` -> no garbage seed)
    V (slots 10..16): copy the prev step's H1 one-hot via ``H1.*.-1``
      (cross-step SSA read of the prev step's value through the KV cache).
    O (slots 10..16): write into ``H1_PREV_STEP`` (the dedicated band; read
      ONLY by the gated dump FFN -> NO edge from the 54 same-step H1 readers,
      NO 2-cycle). The band's physical position comes from the BAKE's
      ``dim_positions`` map (it is a NEW band, absent from the legacy
      registry).
    """
    # IMPORTANT — position source. The H1 pipeline (L0 producer, the 54
    # readers, the LM head) is baked at the LEGACY ``_SetDim`` / dynamic
    # *registry* positions (H1 -> 67; the LM head's byte-1 emission reads
    # ``head.weight[0x02]-[0x00]`` from dims 69/71 == H1+2 / H1+4). So the V
    # cross-step read of ``H1`` and the K-side AX_CARRY / ADDR_B0_LO / MARK_AX
    # taps MUST resolve from ``build_default_registry_dynamic()`` — the same
    # map every probe uses — or they tap dead slots. (Verified:
    # registry H1=67/ADDR_B0_LO=12/AX_CARRY_LO=328 == the actual residual.)
    #
    # The O write TARGET (``H1_PREV_STEP``) is a NEW band that does NOT exist
    # in the registry; its physical position is allocated by the dynamic
    # compiler and surfaced ONLY via the bake's ``dim_positions`` map. Resolve
    # it from there.
    from ...dim_registry_dynamic import build_default_registry_dynamic
    _reg = build_default_registry_dynamic()

    def _P(name: str) -> int:
        return int(_reg.slots[name].start)

    h1 = _P("H1")            # registry H1 (cross-step V read of prev one-hot)
    addr_sig = _P("ADDR_B0_LO") + 5
    axc_lo = _P("AX_CARRY_LO")
    axc_hi = _P("AX_CARRY_HI")
    const = _P("CONST")
    mark_ax = _P("MARK_AX")
    # H1_PREV_STEP O target: from the bake layout (new band, not in registry).
    h1_prev = int(dim_positions["H1_PREV_STEP"])

    # Slot design (the AX_CARRY fresh-preference MUST be on its OWN slot, not
    # mixed into the signature slot). The earlier mixed design failed because
    # the AX_CARRY boost (~+197) is CONSTANT across all prev-step AX byte rows
    # (byte 0/1/2/3 all fresh, AX_CARRY ~ -988), so it lifted them all equally
    # and the small signature gap (~14) could not single out byte-1 -> the V
    # copied the AVERAGE of prev AX byte 0..3's H1 (garbage). Fix: a SHARP
    # signature on slot 0 (large weight, byte-1-specific) makes byte-1 dominate
    # among the prev rows; the AX_CARRY fresh-preference on slot 2 (CONST-gated,
    # additive) then biases toward the FRESH byte-1 row over the current carried
    # byte-1 row.
    addr_b1_hi = _P("ADDR_B1_HI") + 8
    SIG_W = 60.0          # SHARP signature: byte-1 row dominates among prev rows
    q = [
        AP(0, addr_sig, SIG_W),   # K slot 0: byte-1 signature (sharp)
        AP(1, addr_b1_hi, L),     # K slot 1: ADDR_B1_HI+8 AX-register match
        AP(2, const, L),          # K slot 2: AX_CARRY fresh-preference driver
        AP(3, addr_sig, L),       # K slot 3: MARK_AX V=0 sink
    ]
    k = [
        AP(0, addr_sig, SIG_W),
        AP(1, addr_b1_hi, L),
        AP(3, mark_ax, sink_w),
    ]
    # K slot 2 — AX_CARRY fresh-preference (its OWN slot, CONST-driven Q): the
    # prev FRESH byte-1 predictor has very-negative AX_CARRY (sum ~ -988); the
    # current carried byte-1 predictor ~ +2.7 -> ``-AX_CARRY`` gives the prev
    # fresh row a large +K bias. With the sharp slot-0 signature already
    # confining attention to byte-1 rows, this slot picks the FRESH one.
    for j in range(16):
        k.append(AP(2, axc_lo + j, -k_axc_w))
        k.append(AP(2, axc_hi + j, -k_axc_w))

    v = []
    o = []
    H1_W = 7
    V_BASE = 10
    for j in range(H1_W):
        v.append(AP(V_BASE + j, h1 + j, 1.0))       # V: prev H1 (cross-step)
        o.append(AO(h1_prev + j, V_BASE + j, 1.0))  # O: write H1_PREV_STEP

    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
        alibi_slope=alibi_slope,
    )


def make_layer11_ax_byte1_dump_carry_op(enable: bool = True) -> Operation:
    """L13 attn head 7: copy the prev step's AX byte-1 H1 one-hot forward.

    The cross-step carry HEAD half of the AX byte-1 register-dump fix. It
    copies the PREVIOUS VM step's ``H1`` one-hot into the dedicated
    ``H1_PREV_STEP`` band UNCONDITIONALLY (via the ``H1.*.-1`` SSA cross-step
    read). The carried-vs-fresh gating + the re-supply into ``H1`` for the LM
    head live in the partner ``make_ax_byte1_dump_repopulate_op`` FFN. See the
    module-level comment above for the root and the geometry. (Function name
    keeps the ``layer11`` prefix for registration stability; the head is hosted
    on L13.)

    ``enable`` defaults to ``True`` (the production fix). The Operation's
    ``reads``/``writes`` participate in the dep graph: the ``H1.*.-1`` read +
    the distinct-band ``H1_PREV_STEP`` write break the H1-write 2-cycle (the
    compile would raise ``Dependency cycle detected`` otherwise).
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        proxy = _as_setdim_proxy(dim_positions)  # noqa: F841 (parity w/ peers)
        attn = block.attn
        allocator = _allocate_layer13_ax_byte1_dump_carry_heads()
        attn._l13_ax_byte1_dump_carry_head_allocator = allocator
        head_idx = allocator.heads()[-1].head_idx
        HD = attn.W_q.shape[0] // attn.num_heads
        spec = _layer13_ax_byte1_dump_carry_head_spec(
            dim_positions, head_idx,
        )
        Primitives.generate_attention_head(attn, spec, HD)

    def _ir(dim_positions, HD) -> CompilerIR:
        del HD
        allocator = _allocate_layer13_ax_byte1_dump_carry_heads()
        head_idx = allocator.heads()[-1].head_idx
        spec = _layer13_ax_byte1_dump_carry_head_spec(dim_positions, head_idx)
        ir = CompilerIR()
        if enable:
            ir.layer(0).attention.append(
                spec, name="layer13_ax_byte1_dump_carry.head_7",
            )
        return ir

    return Operation(
        name="layer13_ax_byte1_dump_carry",
        # K matches the prev fresh byte-1 predictor row (ADDR_B0_LO+5 signature
        # + AX_CARRY differential + one-hot-presence H1 preference + positive
        # ALiBi recency). V reads the prev step's H1 one-hot cross-step
        # (``H1.*.-1`` -> no same-step back-edge). O writes the dedicated
        # ``H1_PREV_STEP`` band (read ONLY by the gated dump FFN -> no edge from
        # the 54 same-step H1 readers). THIS is the cycle break. H1 (same-step)
        # is read on the K side as a one-hot-presence discriminator; it makes
        # the head one of the H1 readers but does NOT cycle (it WRITES
        # H1_PREV_STEP, read by nobody upstream). ``H1.*.-1`` is the V-side
        # cross-step read of the prev step's one-hot to copy forward.
        reads={"ADDR_B0_LO", "AX_CARRY_LO", "AX_CARRY_HI", "H1", "H1.*.-1"},
        writes={"H1_PREV_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_ir,
        # Bind to the L13 mem-addr dep anchor so the head lands on the L13 attn
        # block (physical block 16, slot 7 free at bake AND runtime — slot 6 is
        # the mul width=2 result relay when C4_MUL_WIDTH2 is on). The L13 read
        # point holds both the crystallised AX_CARRY gate signal and the held
        # prev-step H1 one-hot.
        target_op_name="_layer13_mem_addr_anchor",
        migrated=True,
        declarative_authority="spec_generated",
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# AX byte-1 DUMP repopulate FFN (the carried-vs-fresh GATE + emission band)
# ---------------------------------------------------------------------------
#
# The FFN half of the AX byte-1 register-dump carry. The carry head above
# unconditionally copies the PREVIOUS step's ``H1`` one-hot into the dedicated
# ``H1_PREV_STEP`` band. This FFN re-supplies it into the emission band
# ``H1_DUMP_OUT`` ONLY on carried (non-AX-writing) steps, where the LM head
# then emits the carried high byte (via the mirrored ``H1_DUMP_OUT`` columns
# in ``ax_byte1_dump_head_bake``).
#
# Gate (measured spec_k=0 at the final block, tools/probe_h1prev_carry.py):
#   * row signature: ``ADDR_B0_LO+5`` ~= 0.97 on EVERY byte-1 predictor row,
#     ~0 elsewhere -> fires the unit on the byte-1 predictor row only.
#   * carried-vs-fresh: ``sum(AX_CARRY_LO+HI)`` ~= -988 on a FRESH-AX step and
#     ~= +2.65 on a CARRIED step. A small POSITIVE weight on the AX_CARRY band
#     keeps the carried row above threshold (0.97 + 2.65*w) and drives the
#     fresh row far below it (0.97 - 988*w) -> the unit is DARK on fresh steps
#     (``H1_DUMP_OUT`` stays 0 -> the byte-1 emission is byte-identical) and
#     copies ``H1_PREV_STEP`` on carried steps.
# The copy preserves the one-hot's argmax slot (magnitude need not be exact);
# the LM-head column ``head.weight[token_v, H1_DUMP_OUT+(v+2)] = 5.0`` then
# emits the matching high-byte token.
#
# Writing the DISTINCT ``H1_DUMP_OUT`` band (not ``H1``) is what avoids the
# 2-cycle: a late FFN writing ``H1`` while reading ``AX_CARRY`` would cycle
# (``layer6_routing_ffn`` reads ``H1`` AND writes ``AX_CARRY``). ``H1_DUMP_OUT``
# is read only by the LM head (a model-level forward edge), so no back-edge.
_AX_BYTE1_DUMP_REPOPULATE_HIDDEN_DIM = 7


def _ax_byte1_dump_repopulate_rules() -> tuple[FFNRule, ...]:
    """7 rules: ``H1_DUMP_OUT+j = H1_PREV_STEP+j`` on carried byte-1 rows.

    Each rule fires at the byte-1 predictor row (``ADDR_B0_LO+5`` signature)
    AND carried (``+AX_CARRY`` keeps it above threshold; fresh's ~-988 sinks
    it), then gate-copies the carried one-hot from ``H1_PREV_STEP`` into the
    emission band ``H1_DUMP_OUT``.
    """
    H1_W = 7
    # Per-slot positive AX_CARRY weight (carried-vs-fresh discriminator). The
    # AX_CARRY band-SUM crisply separates THREE step classes at the AX byte-1
    # row (measured spec_k=0): a genuine multi-byte CARRY step (PSH/ADJ) ~ +2.7;
    # a memory-LOAD / non-IMM AX-write step (LI/LC) ~ +0.8; a fresh IMM/ADD
    # AX-write ~ -988. Weighted 1.0 (per cell over the 32-cell band the sum is
    # the raw band value, since only one cell per nibble is active) so the gate
    # threshold can sit ABOVE the +0.8 load class and BELOW the +2.7 carry class
    # -> the dump fires ONLY on a genuine carry, NOT on a memory load (which
    # would re-emit the STALE prev byte-1 onto the freshly loaded value;
    # observed: si_li 42 -> 0x22A=554). fresh IMM/ADD (-988) is darkened with a
    # huge margin.
    AXC_W = 1.0
    # AX-register discriminator: ``ADDR_B1_HI+8`` is ~4.0 on the AX byte rows
    # and ~0.0 on the PC / SP / BP byte rows (measured spec_k=0, PROGRAM-STABLE
    # across 654/754/913/432/692: AX=4.02 / PC=0.01 every time — unlike
    # ``H0+AX_I``, whose marker-distance value drifted to 0 on some programs).
    # It is high on ALL AX byte rows (it decays mildly with byte offset:
    # 4.91/4.02/3.29/2.70 at AX+0/+1/+2/+3), so it is NOT byte-1-specific — the
    # byte-1 selectivity comes from ``ADDR_B0_LO+5`` (0.97 ONLY at AX+1, ~0 at
    # AX+0/+2/+3). Weighted 0.25 so ``ADDR_B1_HI+8`` contributes ~1.0 (matching
    # the 0.97 signature term) -> a BALANCED AND that needs BOTH the AX-register
    # signal AND the byte-1-position signal. This prevents (a) the carried
    # PC/SP/BP byte-1 dump (no AX-register signal) and (b) the AX byte-2/3 dump
    # (no byte-1 signature) from firing.
    AX_REG_W = 0.25
    # Structural blockers (EXACTLY 0 at the byte-1 predictor row): markers off,
    # so a large magnitude only sinks the unit if it strays onto a marker row.
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_STACK0", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    blockers = marker_blockers
    SIG_W = 2.0  # byte-1 signature weight (load-bearing: separates byte-1 from
    #              byte-2/3, which lack ADDR_B0_LO+5 but share ADDR_B1_HI+8/AXC)
    conditions_base = [
        ("ADDR_B1_HI+8", AX_REG_W),   # AX-register discriminator (~4.0 AX / ~0 else)
        ("ADDR_B0_LO+5", SIG_W),      # byte-1 predictor row signature
    ]
    for k in range(16):
        conditions_base.append((f"AX_CARRY_LO+{k}", AXC_W))
        conditions_base.append((f"AX_CARRY_HI+{k}", AXC_W))
    conditions = tuple(conditions_base) + blockers

    # ------------------------------------------------------------------
    # Two-sided band-pass on Σ AX_CARRY (the §524 range check applied to a
    # CONTINUOUS band): an UPPER-bound OVERFLOW KILL on top of the LOWER bound.
    # ------------------------------------------------------------------
    # A SINGLE linear AND can only LOWER-bound Σ AX_CARRY (fire on >= lo). It
    # cannot exclude step classes whose AX_CARRY sits ABOVE the carry band:
    # measured (spec_k=0, efficient ALU, last block, AX byte-1 predictor row):
    #   * genuine CARRY (PSH/ADJ)  Σ AX_CARRY ~ +2.65 (tight cluster 2.65-2.70)
    #   * memory LOAD (LI/LC)       ~ +0.85   -> dark (below lo)
    #   * fresh IMM/ADD             ~ -988    -> dark (far below)
    #   * SHL/SHR result            ~ +12.85  -> MUST be dark (was over-firing)
    #   * JMP                       ~ +47.86  -> MUST be dark (was over-firing)
    # The fire cluster (2.65-2.70) and the over-fire classes (>= 12.85) have a
    # WIDE clean gap, so the upper cut is robust. The discriminator that is
    # bounded in the fire band but blows up for the over-fire classes is the
    # ``AX_CARRY_HI+2`` cell (carry ~0.63-1.31, SHL +6.41, JMP +23.93 — see
    # tools/probe_ax_carry_cells.py).
    #
    # A simple difference-of-SiLU-steps CANNOT band-pass a continuous input
    # (the two unbounded ReLU tails never cancel; verified: the negative-write
    # ceiling drove H1_DUMP_OUT NEGATIVE on SHR, flipping the byte-1 argmax to a
    # wrong token -> 0x2A became 0x102A). The robust form is a TWO-STAGE kill: a
    # precursor FFN (``ax_byte1_carry_overflow_flag``) writes a bounded-by-use
    # ``AX_CARRY_OVERFLOW`` step indicator that is ZERO in the carry band and
    # large-positive for SHL/JMP; this rule then adds it as a STRONG NEGATIVE
    # condition. Out of band the AND sum is driven far below threshold ->
    # silu ~= 0 -> H1_DUMP_OUT == EXACTLY 0 (NOT negative), so the real SHL/JMP
    # byte-1 comes through the normal H1 path untouched. In band the flag is 0,
    # so the dump fires byte-identically to the lower-bound-only design.
    OVERFLOW_KILL_W = 1_000.0
    conditions = conditions + (("AX_CARRY_OVERFLOW", -OVERFLOW_KILL_W),)

    rules: list[FFNRule] = []
    for j in range(H1_W):
        rules.append(multi_way_and_rule(
            name=f"ax_byte1_dump_repopulate_slot_{j}",
            # threshold 4.5 (balanced AND of AX-register + byte-1-signature*2 +
            # the +2.7-vs-+0.8 carry-vs-load AX_CARRY split). The OVERFLOW kill
            # term excludes SHL(+12.85)/JMP(+47.86) without touching the carry.
            conditions=conditions,
            threshold=4.5,
            gate=f"H1_PREV_STEP+{j}",
            # write_scale 2.0/S: silu(S*(cond-thr)) ~= 60 on a carried-AX row
            # -> output ~= 60 * H1_PREV_STEP * 0.02 ~= 1.2 * H1_PREV_STEP
            # (one-hot argmax slot preserved; LM head emits the right token).
            writes=((f"H1_DUMP_OUT+{j}", 0.02),),
        ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# AX byte-1 dump band-pass — UPPER-CUT precursor (overflow kill flag)
# ---------------------------------------------------------------------------
# Writes ``AX_CARRY_OVERFLOW = step(AX_CARRY_HI+2 >= 3.0)`` so the dump FFN can
# band-PASS Σ AX_CARRY (not just lower-bound it). The discriminator
# ``AX_CARRY_HI+2`` is BOUNDED in the carry band (~0.63-1.31) and large for the
# over-fire classes (SHL +6.41, JMP +23.93) — see tools/probe_ax_carry_cells.py.
# The step output is unbounded-positive out of band, which is exactly what a
# KILL term wants: it pushes the dump AND deeply below threshold (-> silu ~= 0
# -> H1_DUMP_OUT == EXACTLY 0, NOT negative, so the real SHL/JMP byte-1 stays on
# the normal H1 path). In the carry band the step input is far below 3.0, so the
# flag is ~0 and the dump fires byte-identically to the lower-bound-only design.
#
# UNIT 2 (2026-06-13 prologue-framing-drift fix): a SECOND kill condition on the
# NON-AX-register rows. The dump's lower bound is ``+1.0 * Σ AX_CARRY`` which is
# UNBOUNDED; on the FIRST step (step 0) of var/func/nested programs the AX_CARRY
# band carries large step-0 garbage (Σ AX_CARRY ~ +25..+112 at the REG_PC byte
# 2/3 predictor rows, spec_k=0 — vs ~+2.65 on a genuine carry), which SWAMPS the
# +4.5 threshold and FIRES the dump on PC byte 2/3 rows. The carried
# ``H1_PREV_STEP`` there is negative garbage (~-26.6), so the dump writes ~-3403
# into ``H1_DUMP_OUT`` and the LM head (reading ``H1_DUMP_OUT+(v+2)`` at +5.0)
# drives the byte-0 token to ~-1e8 -> a garbage PC high byte wins -> the
# fixed-35 step desyncs -> the dominant 1096 "PC-wrong" full-trace fail across
# the var/func/nested/loop/rec/gcd clusters (581 programs). The CLEAN
# discriminator (probe_groundtruth spec_k=0): ``ADDR_B1_HI+8`` is the AX-register
# signal — ~4.02/3.29/2.70 on AX byte 1/2/3 rows (PROGRAM-STABLE, both legit and
# over-fire programs) and EXACTLY ~0.00 on PC/SP/BP/STACK0/MEM byte rows. The
# dump must NEVER fire on a non-AX row regardless of Σ AX_CARRY, so this unit
# writes the kill flag whenever ``ADDR_B1_HI+8 <= 2.0`` (i.e. the AX-register
# signal is ABSENT). On a real AX byte row (>= 2.70) it stays dark, so the legit
# carry (add/sub byte-1) is byte-identical; on every PC/SP/BP/STACK0/MEM row it
# fires -> dump killed -> H1_DUMP_OUT == EXACTLY 0 (the byte stays on the normal
# H1 path). This makes the AX-register signal a HARD prerequisite instead of a
# +0.25-weighted term the unbounded Σ AX_CARRY could overwhelm.
_AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM = 2
# Separation threshold on AX_CARRY_HI+2: carry rows <= ~1.31, over-fire rows
# >= 6.41 -> 3.0 sits cleanly in the gap.
_AX_CARRY_OVERFLOW_HI2_THRESHOLD = 3.0
# Separation threshold on ADDR_B1_HI+8 (AX-register signal): AX byte rows
# >= 2.70 (byte 1/2/3 = 4.02/3.29/2.70), non-AX rows ~0.00 -> 2.0 sits cleanly
# in the gap (margin >= 0.7 from the lowest legit AX row).
_AX_REGISTER_PRESENT_B1HI8_THRESHOLD = 2.0


def _ax_byte1_carry_overflow_flag_rules() -> tuple[FFNRule, ...]:
    """2 rules into ``AX_CARRY_OVERFLOW`` (OR of two dump-kill conditions).

    Unit 0: ``step(AX_CARRY_HI+2 >= 3.0)`` — the SHL/JMP upper-cut.
    Unit 1: ``step(ADDR_B1_HI+8 <= 2.0)`` — the NON-AX-register kill (PC/SP/BP/
    STACK0/MEM byte rows), so the unbounded Σ AX_CARRY lower bound can never
    fire the dump on a non-AX row (the step-0 prologue framing-drift root).
    """
    return (
        step_function_rule(
            name="ax_byte1_carry_overflow_flag",
            input_dim="AX_CARRY_HI+2",
            # threshold target-0.5 convention: fire when the raw value is at or
            # above 3.0 (carry rows ~<=1.31 stay dark; SHL ~6.41 / JMP ~23.93
            # fire). write_value 2.0 -> the dump reads it with weight -1000, so
            # even a small positive flag deeply darkens the dump AND.
            threshold=_AX_CARRY_OVERFLOW_HI2_THRESHOLD,
            write_dim="AX_CARRY_OVERFLOW",
            write_value=2.0,
        ),
        # Unit 1: fire when ADDR_B1_HI+8 <= 2.0 (AX-register signal ABSENT).
        # conditions = ((ADDR_B1_HI+8, -1.0),), threshold = -2.0 -> the SiLU
        # score is ``-ADDR_B1_HI+8`` and the rule fires when
        # ``-ADDR_B1_HI+8 >= -2.0`` i.e. ``ADDR_B1_HI+8 <= 2.0``. On a PC row
        # (signal 0) score = 0 >= -2.0 -> fires strongly (silu(S*2.0)); on the
        # lowest legit AX byte-3 row (signal 2.70) score = -2.70 < -2.0 ->
        # silu(S*-0.7) ~= 0 -> dark. write_value 2.0 (same as unit 0) -> the
        # dump's -1000 read deeply kills it.
        multi_way_and_rule(
            name="ax_byte1_not_ax_register_kill",
            conditions=(("ADDR_B1_HI+8", -1.0),),
            threshold=-_AX_REGISTER_PRESENT_B1HI8_THRESHOLD,
            writes=(("AX_CARRY_OVERFLOW", 2.0 / 100.0),),
        ),
    )


def make_ax_byte1_carry_overflow_flag_op() -> Operation:
    """Precursor FFN: writes the AX byte-1 dump band-pass UPPER-cut kill flag.

    Standalone ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``tail_bit32_result_correction`` but BEFORE ``ax_byte1_dump_repopulate`` (so
    the dump reads a freshly-written ``AX_CARRY_OVERFLOW``). MIXED dim_map: the
    input ``AX_CARRY_HI`` resolves from the legacy dynamic *registry* (the model
    residual carries it there, NOT at the declarative layout position — same
    split the dump documents); the output ``AX_CARRY_OVERFLOW`` is a NEW
    declarative band, resolved from ``dim_positions``.
    """
    rules = _ax_byte1_carry_overflow_flag_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM, (
            f"ax_byte1_carry_overflow_flag rule-count drift: produced "
            f"{len(rules)}, expected {_AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        _new_bands = {"AX_CARRY_OVERFLOW"}
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="ax_byte1_carry_overflow_flag",
        reads={"AX_CARRY_HI", "ADDR_B1_HI"},
        writes={"AX_CARRY_OVERFLOW"},
        kind="block",
        # Append on the same L25 block AFTER the tail correction but BEFORE the
        # dump (the dump's ``requires after this`` pins the order).
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


def make_ax_byte1_dump_repopulate_op() -> Operation:
    """Append the AX byte-1 dump-repopulate FFN after the L25 tail block.

    Copies ``H1_PREV_STEP`` -> ``H1_DUMP_OUT`` on carried byte-1 predictor
    rows (gated on the AX_CARRY fresh/carried separation), so the LM head
    re-emits the carried high byte. Standalone ``PureFFN`` post_op appended on
    the L25 tail block (after ``tail_bit32_result_correction``), so it reads
    the crystallised AX_CARRY + the held ``H1_PREV_STEP`` and nothing
    downstream overrides ``H1_DUMP_OUT`` before the LM head.
    """
    rules = _ax_byte1_dump_repopulate_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _AX_BYTE1_DUMP_REPOPULATE_HIDDEN_DIM, (
            f"ax_byte1_dump_repopulate rule-count drift: produced "
            f"{len(rules)}, expected {_AX_BYTE1_DUMP_REPOPULATE_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        # POSITION SOURCE (mixed) — same split the carry head documents. The H1
        # pipeline, ADDR_B0_LO, AX_CARRY and the MARK_* flags are baked by the
        # LEGACY imperative path at the dynamic *registry* / ``_SetDim``
        # positions (ADDR_B0_LO=12, AX_CARRY_LO=328, ...), which DIFFER from
        # the declarative ``dim_positions`` layout (ADDR_B0_LO=506,
        # AX_CARRY_LO=362). The model residual carries those legacy dims at the
        # REGISTRY positions, so the gate reads MUST resolve from
        # ``build_default_registry_dynamic()`` or they tap dead slots
        # (verified: layout ADDR_B0_LO+5 reads ~0, registry reads 0.97). The
        # NEW bands ``H1_PREV_STEP`` / ``H1_DUMP_OUT`` exist ONLY in the
        # declarative layout, so THOSE resolve from ``dim_positions``.
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        # AX_CARRY_OVERFLOW is also a NEW declarative band (written by the
        # precursor at dim_positions), so it resolves from the layout, not the
        # legacy registry.
        _new_bands = {"H1_PREV_STEP", "H1_DUMP_OUT", "AX_CARRY_OVERFLOW"}
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="ax_byte1_dump_repopulate",
        reads={
            "ADDR_B0_LO", "ADDR_B1_HI", "AX_CARRY_LO", "AX_CARRY_HI",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "H1_PREV_STEP", "AX_CARRY_OVERFLOW",
        },
        writes={"H1_DUMP_OUT"},
        kind="block",
        # Append AFTER the tail correction AND after the overflow-flag precursor
        # on the same L25 block, so this op is the last writer of H1_DUMP_OUT
        # before the LM head reads it AND it reads a freshly-written
        # AX_CARRY_OVERFLOW (the band-pass UPPER-cut kill flag).
        target_op_name="l10_post_ops_combined",
        requires={"after": (
            "tail_bit32_result_correction", "ax_byte1_carry_overflow_flag",
        )},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ===========================================================================
# STACK0 byte-0 cross-step emission carry (Root 2 — the if/bool/expr framing
# drift). MIRRORS the AX byte-1 dump carry above for the stack-top low byte.
# ===========================================================================
#
# ROOT (spec_k=0, confirmed by tools/probe_stack0_byte0_*.py): the STACK0
# byte-0 (stack-top low byte) emission one-hot lives in the LM-head ``H1``
# (high nibble) + ``H3`` (low nibble) bands (each byte token reads
# ``head.weight[byte, H1+k]=5.0`` / ``head.weight[byte, H3+k]=5.0``). It is
# decoded FRESH at the PSH/producing step (L6 block-6 byte decode, because the
# value was just stored to STACK0), so the byte-0 token emits correctly there.
# On the NEXT (carried) comparison step L6 does NOT re-decode it (the value is
# only persisting on the stack, not freshly computed), so the clean one-hot is
# ABSENT at block 6, then the L21 (block-31/32) operand-gather smears a
# value-proportional copy into H1/H3 (e.g. H1+3 ~ 3718 for operand 0x11) and
# the L25 tail corrector NEGATES the whole H1..H5 band to ~-289M. The LM head
# reads that -289M at +5.0 -> logit[byte-0] ~ -100M -> a marker ([PC]) wins,
# so the model emits a SPURIOUS extra register block (a 57-token step instead
# of 35). The runner's fixed-35 slicer then misreads the PC -> the if/bool/expr
# full-trace fails at step 3 with the math CORRECT but the PC off by a register
# block ("framing drift", ~134 of the 183 if/bool/expr fails). It is
# VALUE-dependent (both-nibbles-nonzero operands like 0x11/0x23 drift; one-
# nibble-zero like 0x10/0x20 stay clean), NOT opcode-dependent.
#
# FIX (exact mirror of the AX byte-1 carry): carry the clean PSH-step STACK0
# byte-0 H1/H3 one-hot forward in a DEDICATED band and re-emit it on the
# carried step via a SEPARATE LM-head dump path the L21/L25 correctors never
# touch:
#   * The carry head (below) copies the PREVIOUS step's STACK0-marker H1/H3
#     one-hot into ``STACK0_B0_H1_PREV`` / ``STACK0_B0_H3_PREV`` UNCONDITION-
#     ALLY (via ``H1.*.-1`` / ``H3.*.-1`` SSA cross-step reads -> no same-step
#     back-edge; the head WRITES distinct bands read by nobody upstream).
#   * The dump FFN (``stack0_byte0_dump_repopulate``) copies the PREV bands
#     into ``STACK0_B0_DUMP_H1`` / ``STACK0_B0_DUMP_H3`` ONLY on a carried
#     STACK0-marker row (gated on the same-step H1 byte-0 one-hot being ABSENT
#     -- the crisp fresh-vs-carried discriminator here).
#   * ``stack0_byte0_dump_head_bake`` mirrors the byte-token H1/H3 emission
#     columns onto the DUMP bands (gated by ``C4_STACK0_B0_DUMP``), so the LM
#     head additively re-emits the carried byte-0 token on carried steps and is
#     byte-identical on fresh steps (DUMP bands all-zero).
_STACK0_B0_DUMP_CARRY_HEAD_IDX = 5  # free pre-widen slot on L9 (heads 0..4 used)
_STACK0_B0_DUMP_CARRY_HEAD_LAYOUT = (
    ("stack0_byte0_dump_carry.head_5", _STACK0_B0_DUMP_CARRY_HEAD_IDX),
)


def _allocate_stack0_byte0_dump_carry_heads() -> AttentionHeadAllocator:
    """Per-bake head allocator for the STACK0 byte-0 dump carry head (L9 host)."""
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (op_name, head_idx) in _STACK0_B0_DUMP_CARRY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=9, pin=head_idx)
    return allocator


def _stack0_byte0_dump_carry_head_spec(
    dim_positions: dict,
    head_idx: int,
    *,
    L: float = 15.0,
    sink_w: float = 8.0,
    alibi_slope: float = 0.5,
) -> DeclarativeAttentionHeadSpec:
    """Carry head: copy the prev step's STACK0-marker H1/H3 one-hot forward.

    Mirrors ``_layer13_ax_byte1_dump_carry_head_spec``. The head attends the
    PREVIOUS step's STACK0-marker row (the row that held the clean byte-0 H1/H3
    one-hot) and copies that one-hot into the dedicated ``STACK0_B0_H1_PREV`` /
    ``STACK0_B0_H3_PREV`` bands via the ``H1.*.-1`` / ``H3.*.-1`` SSA cross-step
    reads. ``STACK0_B0_*_PREV`` is read by NOBODY except the gated dump FFN, so
    writing the prev one-hot there on a fresh STACK0 row is harmless (the dump
    FFN gates it out).

    Slots:
      Q (current STACK0 marker row):
        slot 0 = ``MARK_STACK0`` * SIG_W  (fire on STACK0-marker rows, sharp)
        slot 1 = ``CONST`` * L            (one-hot-presence K driver)
        slot 3 = ``MARK_STACK0`` * L      (drives the V=0 sink for the 1st step)
      K:
        slot 0 = ``MARK_STACK0`` * SIG_W  (match STACK0-marker rows; ALiBi
                 positive-recency picks the NEAREST prev STACK0 marker)
        slot 3 = ``MARK_AX`` * sink_w     (V=0 sink: on the first STACK0 step
                 there is no prev STACK0 marker -> fall on an AX marker,
                 H1/H3==0)
      V (slots 1..7 low / 10..16 high): copy the prev step's H1 (high nibble)
        and H3 (low nibble) one-hots via ``H1.*.-1`` / ``H3.*.-1``.
      O: write ``STACK0_B0_H1_PREV`` (from H1) and ``STACK0_B0_H3_PREV`` (from
        H3) — dedicated bands, position from the bake ``dim_positions``.
    """
    # Position source (same MIXED split the AX carry documents): the H1/H3
    # pipeline and the MARK_* gate dims are baked at the LEGACY dynamic
    # *registry* positions (the model residual carries them there); the NEW
    # ``STACK0_B0_*`` bands exist ONLY in the declarative layout. So the
    # cross-step V reads (H1/H3) and the K-side MARK taps resolve from the
    # registry, and the O targets from ``dim_positions``.
    from ...dim_registry_dynamic import build_default_registry_dynamic
    _reg = build_default_registry_dynamic()

    def _P(name: str) -> int:
        return int(_reg.slots[name].start)

    h1 = _P("H1")                 # registry H1 (cross-step V read of prev hi-nibble)
    h3 = _P("H3")                 # registry H3 (cross-step V read of prev lo-nibble)
    const = _P("CONST")
    mark_stack0 = _P("MARK_STACK0")
    mark_ax = _P("MARK_AX")
    h1_prev = int(dim_positions["STACK0_B0_H1_PREV"])
    h3_prev = int(dim_positions["STACK0_B0_H3_PREV"])

    SIG_W = 60.0  # SHARP signature: STACK0-marker rows dominate among prev rows
    # ONE-HOT-PRESENCE K preference (the decisive fix): among the STACK0-marker
    # rows the head can attend, the FRESH (PSH) prev row HOLDS the clean byte-0
    # H1/H3 one-hot (Σ|H1|+|H3| ~ 4 at this block, BEFORE the L25 corruption);
    # the CURRENT (carried) STACK0-marker row's one-hot is ABSENT (Σ ~ 0). A
    # K-side +PRES_W on every H1/H3 cell, driven by the CONST Q slot, biases the
    # head toward the row that HAS the one-hot -> it picks the fresh prev STACK0
    # marker over the current carried one. Without this the positive-ALiBi
    # recency picks the NEAREST (current, empty) marker and the carry copies 0.
    PRES_W = 6.0
    q = [
        AP(0, mark_stack0, SIG_W),   # K slot 0: STACK0-marker signature (sharp)
        AP(1, const, L),             # K slot 1: one-hot-presence driver
        AP(3, mark_stack0, L),       # K slot 3: MARK_AX V=0 sink driver
    ]
    k = [
        AP(0, mark_stack0, SIG_W),
        AP(3, mark_ax, sink_w),
    ]
    # K slot 1 (CONST-driven): prefer the row WITH the one-hot present.
    for j in range(7):
        k.append(AP(1, h1 + j, PRES_W))
        k.append(AP(1, h3 + j, PRES_W))

    v = []
    o = []
    W = 7
    LO_BASE = 1    # V slots for the low-nibble (H3) copy
    HI_BASE = 10   # V slots for the high-nibble (H1) copy
    for j in range(W):
        v.append(AP(LO_BASE + j, h3 + j, 1.0))       # V: prev H3 (cross-step)
        v.append(AP(HI_BASE + j, h1 + j, 1.0))       # V: prev H1 (cross-step)
        o.append(AO(h3_prev + j, LO_BASE + j, 1.0))  # O: write STACK0_B0_H3_PREV
        o.append(AO(h1_prev + j, HI_BASE + j, 1.0))  # O: write STACK0_B0_H1_PREV

    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
        alibi_slope=alibi_slope,
    )


def make_stack0_byte0_dump_carry_op(enable: bool = True) -> Operation:
    """L9 attn head 5: copy the prev step's STACK0 byte-0 H1/H3 one-hot forward.

    The cross-step carry HEAD half of the STACK0 byte-0 register-dump fix
    (Root 2). Copies the PREVIOUS VM step's STACK0-marker ``H1`` / ``H3``
    one-hot into the dedicated ``STACK0_B0_H1_PREV`` / ``STACK0_B0_H3_PREV``
    bands UNCONDITIONALLY (via the ``H1.*.-1`` / ``H3.*.-1`` SSA cross-step
    reads). The carried-vs-fresh gate + the re-supply into the dump bands for
    the LM head live in the partner ``make_stack0_byte0_dump_repopulate_op``
    FFN. Hosted on L9 (physical block 10), head 5 (free in the pre-widen
    8-head band: L9 declares heads 0..4).
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        attn = block.attn
        allocator = _allocate_stack0_byte0_dump_carry_heads()
        attn._l9_stack0_byte0_dump_carry_head_allocator = allocator
        head_idx = allocator.heads()[-1].head_idx
        HD = attn.W_q.shape[0] // attn.num_heads
        spec = _stack0_byte0_dump_carry_head_spec(dim_positions, head_idx)
        Primitives.generate_attention_head(attn, spec, HD)

    def _ir(dim_positions, HD) -> CompilerIR:
        del HD
        allocator = _allocate_stack0_byte0_dump_carry_heads()
        head_idx = allocator.heads()[-1].head_idx
        spec = _stack0_byte0_dump_carry_head_spec(dim_positions, head_idx)
        ir = CompilerIR()
        if enable:
            ir.layer(0).attention.append(
                spec, name="stack0_byte0_dump_carry.head_5",
            )
        return ir

    return Operation(
        name="stack0_byte0_dump_carry",
        # Q@MARK_STACK0 (current STACK0 marker) / K@MARK_STACK0 (prev STACK0
        # markers, ALiBi picks the nearest) + MARK_AX V=0 sink. V reads the
        # prev step's H1/H3 one-hot cross-step (``H1.*.-1`` / ``H3.*.-1`` -> no
        # same-step back-edge). O writes the dedicated ``STACK0_B0_*_PREV``
        # bands (read ONLY by the gated dump FFN -> no edge from same-step H1/H3
        # readers). H1/H3 (same-step) appear in reads as the cross-step base;
        # the head WRITES the PREV bands, read by nobody upstream -> no cycle.
        reads={"MARK_STACK0", "MARK_AX", "CONST", "H1", "H3",
               "H1.*.-1", "H3.*.-1"},
        writes={"STACK0_B0_H1_PREV", "STACK0_B0_H3_PREV"},
        kind="block",
        # Bind to the L9 anchor so the head lands on the L9 attn block (physical
        # block 10). L9's read point holds the prev-step STACK0 one-hot via the
        # KV cache; head 5 is free in the pre-widen band.
        target_op_name="layer9_marker_suppress",
        declarative_bake_fn=bake,
        compiler_ir_factory=_ir,
        migrated=True,
        declarative_authority="spec_generated",
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# STACK0 byte-0 DUMP repopulate FFN (the carried-vs-fresh GATE + emission band)
# ---------------------------------------------------------------------------
#
# The FFN half of the STACK0 byte-0 carry. The carry head above unconditionally
# copies the PREVIOUS step's STACK0-marker H1/H3 one-hot into the
# ``STACK0_B0_H1_PREV`` / ``STACK0_B0_H3_PREV`` bands. This FFN re-supplies them
# into the emission bands ``STACK0_B0_DUMP_H1`` / ``STACK0_B0_DUMP_H3`` ONLY on
# a carried STACK0-marker row, where the LM head then re-emits the carried
# byte-0 (via the mirrored DUMP columns in ``stack0_byte0_dump_head_bake``).
#
# Gate (measured spec_k=0, tools/probe_stack0_byte0_*.py): the crisp fresh-vs-
# carried discriminator is the SAME-STEP H1 byte-0 one-hot at the STACK0-marker
# row. On a FRESH (PSH) step the byte-0 one-hot is decoded (Σ|H1| ~ 5-10, an
# argmax slot present) so the dump must NOT fire (the normal H1/H3 path emits
# correctly). On a CARRIED step the clean one-hot is ABSENT at block 6
# (Σ|H1| ~ 0 going into this FFN's read point) so the dump SHOULD fire and
# re-supply the carried one-hot from the PREV band. We therefore gate on
# MARK_STACK0 (row signature) with a NEGATIVE same-step H1 condition (fresh
# steps have H1 present -> darken; carried steps have H1==0 -> fire). The
# marker blockers keep the unit off non-STACK0 rows.
_STACK0_B0_DUMP_REPOPULATE_HIDDEN_DIM = 14  # 7 (H1) + 7 (H3)

# Re-point write_scale for the DIRECT H1/H3 emission re-supply (Root 2 fix).
# The dump runs at the LAST post_op block (block 42), AFTER the block-38
# ``tail_bit32_result_correction`` corruptor has nuked the byte's OWN H1/H3
# emission cells to ~-1e7 (the cells the LM head reads at +5.0:
# ``H1+(lo+2)`` low nibble, ``H3+(hi+4)`` high nibble — confirmed via
# tools/probe_stack0_byte0_logit.py: byte 0x11 reads H1+3=-1.05e7, H3+5=-9.98e6).
# An additive band in SEPARATE dims (the original DUMP-band approach) cannot
# overcome a NEGATIVE write in the SAME cells; so on a CARRIED step we re-supply
# the carried byte-0 one-hot DIRECTLY into ``H1+j`` / ``H3+j`` (the residual is
# additive, so this ADDS to the -1e7 nuke). The PREV band carries the clean
# one-hot (~160 at the argmax slot); to net a large POSITIVE at the read slot
# over the -1e7 nuke we scale: output ~= silu(AND)~=60 * PREV~=160 * write_scale.
# Default 70.0 -> the read slot nets ~+7e9 over the -1e7 nuke. Env-overridable
# for probe sweeps via ``C4_STACK0_B0_REPOINT_WS``.
_STACK0_B0_REPOINT_WS = float(
    _os_stack0.environ.get("C4_STACK0_B0_REPOINT_WS", "70.0")
)


def _stack0_byte0_dump_repopulate_rules() -> tuple[FFNRule, ...]:
    """14 rules: re-supply the carried byte-0 one-hot DIRECTLY into the byte's
    own ``H1+j`` / ``H3+j`` LM-head emission cells on carried STACK0-marker rows.

    THE RE-POINT (Root 2 fix): the byte-0 emission one-hot lives in the LM-head
    ``H1`` (low nibble) + ``H3`` (high nibble) bands; the byte token reads
    ``H1+(lo+2)`` and ``H3+(hi+4)`` at +5.0. On a CARRIED comparison step the
    block-38 ``tail_bit32_result_correction`` corruptor nukes those SAME cells to
    ~-1e7, suppressing the byte token so a ``[PC]`` marker wins (the 57-token
    framing-drift step). The carry head copied the PREVIOUS (PSH) step's clean
    byte-0 ``H1``/``H3`` one-hot into ``STACK0_B0_{H1,H3}_PREV`` (identity slot
    map: ``STACK0_B0_H1_PREV+j -> H1+j``, ``STACK0_B0_H3_PREV+j -> H3+j``). This
    FFN runs at the LAST post_op block (AFTER the nuke) and ADDS a large positive
    multiple of that carried one-hot back into ``H1+j`` / ``H3+j`` ONLY on a
    carried STACK0-marker row -> the read slot nets positive -> the byte token
    wins. BYTE-IDENTICAL on fresh steps: the gate AND requires the BOUNDED
    ``STACK0_B0_CARRIED`` flag (0 on fresh PSH steps and every non-STACK0 row),
    so the silu activation is ~0 and ``H1``/``H3`` are untouched.

    GATE (the BOUNDED design — the raw-H1 read is fatal): the dump reads the
    pre-computed ``STACK0_B0_CARRIED`` flag (a bounded 0/1 step indicator
    written by the ``stack0_byte0_carried_flag`` precursor at an EARLY block
    where H1/H3 are still bounded). It MUST NOT read the same-step H1/H3 here:
    at this L25-tail read point the L25 corruptor has nuked H1/H3 to ~-289M, so
    a negative-weighted H1 condition drives the silu gate to ~+10^9 and writes
    billions into the band (observed: the model emitted all-zeros). The flag is
    1 on a carried STACK0-marker row, 0 on a fresh PSH STACK0-marker row and on
    every non-STACK0 row.
    """
    W = 7
    # Row signature + carried flag (both bounded). A balanced AND that needs
    # BOTH the STACK0-marker row AND the carried flag.
    SIG_W = 2.0       # MARK_STACK0 ~ 1.0 on the row -> +2.0
    CARRIED_W = 3.0   # STACK0_B0_CARRIED ~ 1.0 on a carried row -> +3.0
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    SHARP_W = 3.0     # STACK0_B0_SHARP ~ 1.0 on a clean-carry row -> +3.0

    # The DIRECT H1/H3 re-point is gated on ``C4_STACK0_B0_DUMP`` (DEFAULT-ON;
    # opt OUT with ``C4_STACK0_B0_DUMP=0`` for the byte-identical pre-carry build).
    # OFF the dump targets the inert ``STACK0_B0_DUMP_{H1,H3}`` bands (read by
    # NOBODY — the LM-head dump columns are themselves flag-gated), so it
    # contributes nothing to any token the prior model emitted. ON re-points the
    # write into the byte's own ``H1``/``H3`` emission cells (the re-point fix),
    # gated by the carried/PREV-dominant/NON-COMPARISON discriminators so it fires
    # ONLY on the genuine if/bool/expr framing-drift rows. Same flag the head bake
    # uses, so the whole emission path flips together.
    _repoint_on = _os_stack0.environ.get("C4_STACK0_B0_DUMP", "1") != "0"

    # Re-point path ANDs the BOUNDED ``STACK0_B0_SHARP`` flag (PREV is a clean
    # single-slot one-hot, not a smear) AND a STRONG NEGATIVE ``STACK0_B0_NOT_CMP``
    # BLOCKER -- the DEFAULT-ON discriminator. NOT_CMP fires (=1) on the
    # ARITHMETIC-result / JMP rows the dump must NOT touch (their STACK0 row
    # carries a per-step OP_ADD/SUB/.../OP_JMP) and is 0 on COMPARISON rows. A
    # ``-1000`` blocker (NOT an additive positive term) is REQUIRED here: the
    # gate's CARRIED + SHARP terms are each ~100 (not ~1), so only a strongly
    # negative ABSENT-on-comparison blocker can overcome them on an arith/jmp row.
    # This darkens EXACTLY the add_16bit + jmp_forward over-fire the flag-ON build
    # regressed while leaving the if/bool/expr framing-drift rows (NOT_CMP = 0)
    # firing -> the re-point ships DEFAULT-ON (smoke 51/0). The inert DUMP path
    # keeps the original 2-way AND (byte-identical).
    NOT_CMP_BLOCK_W = -1_000.0
    if _repoint_on:
        conditions = (
            ("MARK_STACK0", SIG_W),
            ("STACK0_B0_CARRIED", CARRIED_W),
            ("STACK0_B0_SHARP", SHARP_W),
            ("STACK0_B0_NOT_CMP", NOT_CMP_BLOCK_W),
        ) + marker_blockers
        # 3-way AND + NON-COMPARISON blocker: a carried clean-carry COMPARISON
        # row -> 2 + 3*100 + 3*100 + 0 = 602 > 7 (fires); an ARITHMETIC/JMP
        # result row -> NOT_CMP ~ 100 -> 602 - 1000*100 << 7 (dark — fixes the
        # over-fire); a carried SMEAR row -> SHARP = 0 -> 302 (still fires unless
        # NOT_CMP blocks, which the arith/jmp result IS); a fresh row ->
        # CARRIED = 0 -> 302 (carried/sharp ~100 so the additive AND is loose;
        # the carried/sharp/marker/NOT_CMP terms gate it, not a tight margin).
        dump_threshold = 7.0
    else:
        conditions = (
            ("MARK_STACK0", SIG_W),
            ("STACK0_B0_CARRIED", CARRIED_W),
        ) + marker_blockers
        dump_threshold = 4.0
    rules: list[FFNRule] = []
    # Low-nibble lives in H1 (PREV idx = lo+2), high-nibble in H3 (idx = hi+4):
    # the carry head copied the prev step's registry-H1/H3 one-hots into the
    # like-named PREV bands at the SAME slot index, so the re-point write target
    # is the identity-indexed registry cell.
    for (emit_band, dump_band, prev_band) in (
        ("H3", "STACK0_B0_DUMP_H3", "STACK0_B0_H3_PREV"),
        ("H1", "STACK0_B0_DUMP_H1", "STACK0_B0_H1_PREV"),
    ):
        # Flag ON -> write the byte's own H1/H3 emission cell (the re-point);
        # flag OFF -> write the inert DUMP band (byte-identical, read by nobody).
        target = emit_band if _repoint_on else dump_band
        ws = _STACK0_B0_REPOINT_WS if _repoint_on else 0.02
        for j in range(W):
            rules.append(multi_way_and_rule(
                name=f"stack0_b0_repoint_{target.lower()}_slot_{j}",
                # Flag OFF (thr 4.0): fires on a carried STACK0-marker row
                # (2 + 3 = 5 > 4), dark on a fresh STACK0 row (2 + 0 < 4) and on
                # every non-STACK0 row (a -1000 marker blocker dominates).
                # Flag ON (thr 7.0): ADDS the SHARP flag -> fires ONLY on a
                # clean-carry row (2+3+3 > 7); a carried smear (SHARP=0 -> 5) and
                # a fresh row (CARRIED=0 -> 2) are dark. All gate inputs bounded.
                conditions=conditions,
                threshold=dump_threshold,
                gate=f"{prev_band}+{j}",
                # write_scale: silu(S*(cond-thr)) ~= 60 on a carried row,
                # gate = PREV+j (~160 at the argmax slot, ~0 elsewhere) ->
                # output ~= 60 * 160 * write_scale at the correct slot. Tuned
                # large so the read slot nets POSITIVE over the -1e7 nuke (the
                # residual is additive). The PREV one-hot keeps the write off
                # the wrong slots; the LM head emits the matching byte token.
                writes=((f"{target}+{j}", ws),),
            ))
    return tuple(rules)


# ---------------------------------------------------------------------------
# STACK0 byte-0 carried-step flag precursor (the BOUNDED gate signal)
# ---------------------------------------------------------------------------
# Writes ``STACK0_B0_CARRIED = AND(MARK_STACK0, H3-absent)`` at an EARLY block
# (bound to the L7 anchor) where the same-step H3 byte-0 one-hot is still
# BOUNDED (fresh PSH: Σ|H3| ~ 3.3 present; carried: Σ|H3| ~ 0 absent). Reading
# H3 here -- BEFORE the L25 corruptor nukes it to ~-289M -- keeps the gate's
# silu input bounded. The flag is a residual dim that nothing else writes, so
# it persists to the L25 tail where the dump FFN reads it. Same role as the AX
# carry's ``ax_byte1_carry_overflow_flag`` precursor.
_STACK0_B0_CARRIED_FLAG_HIDDEN_DIM = 1


def _stack0_byte0_carried_flag_rules() -> tuple[FFNRule, ...]:
    """1 rule: ``STACK0_B0_CARRIED = step(MARK_STACK0*2 - Σ|H3| >= 1.0)``.

    Fires on a STACK0-marker row whose same-step H3 byte-0 one-hot is ABSENT
    (a carried step), dark on a fresh PSH STACK0 row (H3 present ~3.3 pulls the
    AND below threshold) and on every non-STACK0 row.
    """
    SIG_W = 2.0
    # H3 present (fresh): Σ ~ 3.3 -> 2 - 3.3 = -1.3 < 1 (dark). H3 absent
    # (carried): Σ ~ 0 -> 2 - 0 = 2 >= 1 (fires). H3 is BOUNDED at this early
    # read point so the negative weight is safe (no silu blowup).
    H3_KILL_W = 1.0
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    conditions = [("MARK_STACK0", SIG_W)]
    for k in range(7):
        conditions.append((f"H3+{k}", -H3_KILL_W))
    conditions = tuple(conditions) + marker_blockers
    return (
        multi_way_and_rule(
            name="stack0_byte0_carried_flag",
            conditions=conditions,
            threshold=1.0,
            # write 1.0 (bounded): the dump reads it with weight 3.0.
            writes=(("STACK0_B0_CARRIED", 1.0),),
        ),
    )


def make_stack0_byte0_carried_flag_op() -> Operation:
    """Precursor FFN: writes the BOUNDED STACK0 byte-0 carried-step gate flag.

    Standalone ``PureFFN`` post_op bound to the L7 anchor (an EARLY block where
    the same-step H3 byte-0 one-hot is still bounded -- BEFORE the L25 corruptor
    nukes it). Writes ``STACK0_B0_CARRIED`` (a residual dim nothing else writes)
    so the dump FFN can gate on a bounded 0/1 flag instead of the corrupted
    H1/H3. MIXED dim_map: the MARK_* / H3 read taps resolve from the legacy
    dynamic *registry*; the NEW ``STACK0_B0_CARRIED`` band from ``dim_positions``.
    """
    rules = _stack0_byte0_carried_flag_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _STACK0_B0_CARRIED_FLAG_HIDDEN_DIM, (
            f"stack0_byte0_carried_flag rule-count drift: produced "
            f"{len(rules)}, expected {_STACK0_B0_CARRIED_FLAG_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        _new_bands = {"STACK0_B0_CARRIED"}
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="stack0_byte0_carried_flag",
        reads={
            "MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
            "MARK_MEM", "MARK_SE", "H3",
        },
        writes={"STACK0_B0_CARRIED"},
        kind="block",
        # Bind to the L8 attn op (an early block where H3 is still bounded --
        # the L25 corruptor that nukes H3 to ~-289M runs much later). The flag
        # persists to the L25 tail (nothing else writes it).
        target_op_name="layer8_multibyte_fetch",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# STACK0 byte-0 PREV-sharpness flag precursor (the re-point no-regress gate)
# ---------------------------------------------------------------------------
# Writes ``STACK0_B0_SHARP = OR_j step(PREV+j - Σ_{k!=j} PREV+k >= thr)`` on a
# STACK0-marker row. The carry head copies the prev step's byte-0 H1/H3 one-hot
# into ``STACK0_B0_H1_PREV``: a CLEAN single-slot one-hot (~160 at one slot, ~0
# elsewhere) on the if/bool/expr framing-drift case, but a SMEAR (~85 across all
# slots) on a multi-byte arithmetic result (add_16bit etc.) where there was no
# single prev STACK0 byte to copy. The DIRECT H1/H3 re-point ANDs this flag so
# it fires ONLY on the genuinely-corrupted clean-carry rows and is a NO-OP on
# healthy multi-byte emissions (re-supplying a smeared one-hot would corrupt
# them). Mirrors the carried-flag precursor; reads the (already-populated) PREV
# band rather than H3, so it binds to a LATER anchor (after the L9 carry head).
_STACK0_B0_SHARP_FLAG_HIDDEN_DIM = 7  # one step rule per H1_PREV slot (OR)


def _stack0_byte0_sharp_flag_rules() -> tuple[FFNRule, ...]:
    """7 rules (OR over slots): ``STACK0_B0_SHARP`` fires iff ONE H1_PREV slot
    dominates the rest (a clean carried one-hot), dark on a smear.

    PREV reads are SCALED DOWN (alpha) to keep the silu input bounded: PREV ~160
    raw would overflow. With alpha=0.02 the per-slot margin is
    ``alpha*(PREV+j - Σ_others)`` ~ +3.1 (clean) / -6.8 (smear), so a small
    MARK_STACK0 anchor + threshold cleanly separates them and the silu stays
    finite. Exactly one slot can be the argmax, so the OR never double-writes.
    """
    W = 7
    SIG_W = 2.0
    ALPHA = 0.02   # PREV scale-down: clean margin ~ +3.1, smear ~ -6.8
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    rules: list[FFNRule] = []
    for j in range(W):
        conds = [("MARK_STACK0", SIG_W), (f"STACK0_B0_H1_PREV+{j}", ALPHA)]
        for k in range(W):
            if k != j:
                conds.append((f"STACK0_B0_H1_PREV+{k}", -ALPHA))
        conds = tuple(conds) + marker_blockers
        # threshold 4.0: clean slot -> 2 + 3.1 = 5.1 > 4 (fires); smear ->
        # 2 - 6.8 = -4.8 < 4 (dark); non-argmax clean slot -> 2 + alpha*(~0 -
        # 160) = 2 - 3.1 = -1.1 < 4 (dark). So exactly the argmax slot of a clean
        # one-hot fires. write 1.0 (bounded); the dump reads SHARP with weight 3.
        rules.append(multi_way_and_rule(
            name=f"stack0_byte0_sharp_flag_slot_{j}",
            conditions=conds,
            threshold=4.0,
            writes=(("STACK0_B0_SHARP", 1.0),),
        ))
    return tuple(rules)


def make_stack0_byte0_sharp_flag_op() -> Operation:
    """Precursor FFN: writes the BOUNDED STACK0 byte-0 PREV-sharpness gate flag.

    Standalone ``PureFFN`` post_op on the L25 tail block, ordered BEFORE the dump
    FFN. Reads the (block-10-populated, liveness-private) ``STACK0_B0_H1_PREV``
    band and writes ``STACK0_B0_SHARP`` so the re-point dump can AND a bounded
    0/1 sharpness signal. MIXED dim_map: MARK_* taps from the legacy registry;
    the NEW ``STACK0_B0_*`` bands from ``dim_positions``.
    """
    rules = _stack0_byte0_sharp_flag_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _STACK0_B0_SHARP_FLAG_HIDDEN_DIM, (
            f"stack0_byte0_sharp_flag rule-count drift: produced "
            f"{len(rules)}, expected {_STACK0_B0_SHARP_FLAG_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        _new_bands = {"STACK0_B0_H1_PREV", "STACK0_B0_SHARP"}
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="stack0_byte0_sharp_flag",
        reads={
            "MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
            "MARK_MEM", "MARK_SE", "STACK0_B0_H1_PREV",
        },
        writes={"STACK0_B0_SHARP"},
        kind="block",
        # On the L25 tail block (where the PREV band — set at the L9 carry head,
        # block 10 — has long since persisted). Ordered AFTER the carried-flag
        # precursor and BEFORE the dump FFN that reads SHARP.
        target_op_name="l10_post_ops_combined",
        requires={"after": ("stack0_byte0_carried_flag",)},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# STACK0 byte-0 RATIO-based PREV-dominant flag precursor (smear gate, mag-indep)
# ---------------------------------------------------------------------------
# Writes ``STACK0_B0_PREV_DOM = OR_j step(PREV+j - Σ_others > 1)`` -- fires when
# ONE PREV slot dominates (a clean carried one-hot of ANY magnitude), DARK on a
# smear. Unlike ``STACK0_B0_SHARP`` (an ABSOLUTE per-slot margin that misses a
# small-magnitude clean one-hot -- e.g. comparison result byte 0x01, PREV ~6,
# CLEAN but SHARP = 0), this uses a LOW threshold so it fires for a clean one-hot
# of ANY size and is dark only for a true smear (add_16bit: every slot ~72..110,
# no slot dominant). The NON-COMPARISON blocker's smear rule reads PREV_DOM so it
# darkens the add_16bit smear WITHOUT darkening small comparison results (which a
# raw SHARP=0 test wrongly blocked -> if_gt 17 -> 9).
_STACK0_B0_PREV_DOM_FLAG_HIDDEN_DIM = 7  # one ratio rule per H1_PREV slot (OR)


def _stack0_byte0_prev_dom_flag_rules() -> tuple[FFNRule, ...]:
    """7 rules (OR over slots): ``STACK0_B0_PREV_DOM`` fires iff one PREV slot
    dominates (margin ``PREV+j - Σ_others > 1``, a clean one-hot of ANY magnitude),
    dark on a smear. Same shape as the SHARP rule but a LOWER threshold so a
    small-magnitude clean one-hot still clears it (the magnitude-independent ratio
    test). Exactly one slot can dominate, so the OR never double-writes.
    """
    W = 7
    SIG_W = 2.0
    ALPHA = 0.05   # PREV scale-down: clean margin (PREV+j - Σ_others) * 0.05 > thr
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    rules: list[FFNRule] = []
    for j in range(W):
        # margin = ALPHA*(PREV+j - Σ_others): slot j +ALPHA, every OTHER slot -ALPHA.
        conds = [("MARK_STACK0", SIG_W), (f"STACK0_B0_H1_PREV+{j}", ALPHA)]
        for k in range(W):
            if k != j:
                conds.append((f"STACK0_B0_H1_PREV+{k}", -ALPHA))
        conds = tuple(conds) + marker_blockers
        # threshold 2.05: a clean dominant slot -> 2 + ALPHA*(margin>1) > 2.05
        # (byte 0x01 margin ~6 -> +0.3; big byte margin ~162 -> +8.1); a smear ->
        # margin <= 0 -> < 2.05 (dark); empty PREV -> margin 0 -> dark.
        rules.append(multi_way_and_rule(
            name=f"stack0_byte0_prev_dom_slot_{j}",
            conditions=conds,
            threshold=2.05,
            writes=(("STACK0_B0_PREV_DOM", 1.0),),
        ))
    return tuple(rules)


def make_stack0_byte0_prev_dom_flag_op() -> Operation:
    """Precursor FFN: writes the RATIO-based BOUNDED STACK0 byte-0 PREV-dominant
    flag. Standalone ``PureFFN`` post_op on the L25 tail block, ordered AFTER the
    carry head populates PREV and BEFORE the NON-COMPARISON blocker that reads it.
    MIXED dim_map: MARK_* taps from the legacy registry; the NEW ``STACK0_B0_*``
    bands from ``dim_positions``.
    """
    rules = _stack0_byte0_prev_dom_flag_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _STACK0_B0_PREV_DOM_FLAG_HIDDEN_DIM, (
            f"stack0_byte0_prev_dom_flag rule-count drift: produced "
            f"{len(rules)}, expected {_STACK0_B0_PREV_DOM_FLAG_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        _new_bands = {"STACK0_B0_H1_PREV", "STACK0_B0_PREV_DOM"}
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="stack0_byte0_prev_dom_flag",
        reads={
            "MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
            "MARK_MEM", "MARK_SE", "STACK0_B0_H1_PREV",
        },
        writes={"STACK0_B0_PREV_DOM"},
        kind="block",
        target_op_name="l10_post_ops_combined",
        requires={"after": ("stack0_byte0_carried_flag",)},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )


# ---------------------------------------------------------------------------
# STACK0 byte-0 NON-COMPARISON blocker precursor (the DEFAULT-ON discriminator)
# ---------------------------------------------------------------------------
# Writes ``STACK0_B0_NOT_CMP = step(NO comparison/branch opcode on this row)`` --
# a BLOCKER that fires on EVERY carried STACK0 row EXCEPT the genuine if/bool/expr
# framing-drift rows (which carry a comparison opcode at the compare step and a
# consuming-branch opcode at the fused branch step). This is the byte-vs-marker
# discriminator the prior sessions claimed "does not exist" -- it DOES, and the
# miss was the same dim-mismap the Root-3 (func) fix corrected: the opcode signal
# at the dump block input is read from the WIDENED 981-dim ``dim_positions``
# (OP_GT=204, OP_BZ=32, ...), NOT the static 872-dim registry (OP_GT=282, ...)
# which mismaps onto the widened residual and so always looked like constant
# garbage.
#
# Measured (spec_k=0, tools/probe_stack0_smoke_gate.py / probe_stack0_arithgate.py,
# at the dump block input on the carried STACK0-marker row):
#   * COMPARISON drift rows (if_gt/lt/eq/ne/ge/le, bool_and/or): the compare step
#     carries OP_GT/LT/EQ/... = 0.11 and the fused consuming branch carries
#     OP_BZ/BNZ = 0.08..0.50          -> NOT_CMP = 0 (the dump FIRES; the fix).
#   * ARITHMETIC result rows + their operand/PSH SETUP rows (add/sub/mul/.../the
#     mul intermediate that carries a clean operand one-hot): NO cmp/branch opcode
#                                       -> NOT_CMP = 1 (the dump is DARK).
#   * JMP rows (jmp_forward): NO cmp/branch opcode -> NOT_CMP = 1 (dark).
# The re-point dump reads NOT_CMP as a STRONG NEGATIVE (-1000) blocker. A POSITIVE
# comparison-requirement (rather than a NOT-arith blocker) is REQUIRED: the dump
# gate's CARRIED + SHARP terms are each ~100, so the gate fires on ANY carried
# clean-PREV row unless STRONGLY blocked; and a NOT-arith blocker only darkens the
# arith RESULT row, NOT the arith operand SETUP rows (e.g. the 2-byte mul
# intermediate that carries a clean operand one-hot and NO opcode -> the dump
# fired there and corrupted the high byte, 1239 -> 61655). Requiring a comparison
# opcode POSITIVELY darkens all of those in one rule. Mirrors the SHARP / CARRIED
# bounded precursors; reads the cmp/branch opcode bands directly from
# ``dim_positions``.
_STACK0_B0_NOT_CMP_FLAG_HIDDEN_DIM = 2  # OR: (arith/JMP opcode) + (PREV smear)
# Arithmetic + unconditional-JMP opcodes whose per-step presence marks the
# multi-byte arithmetic-result / JMP STACK0 rows the re-point dump must NOT touch.
# OP_PSH / OP_IMM are DELIBERATELY excluded: in the BATCHED smoke/1096 gate (the
# authoritative path) a faint OP_PSH leak rides on the if/bool/expr comparison
# drift rows, so blocking OP_PSH there darkens the genuine fix (if_gt -> 4). The
# add_16bit corrupting row (which carries no arith opcode) is caught instead by
# the SMEAR rule below. Read from the WIDENED layout (``dim_positions``); a clean
# per-step signal the prior session missed by reading the mismapped static
# registry positions.
_STACK0_B0_NOT_CMP_BLOCK_OPS = (
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
    "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR", "OP_JMP",
)


def _stack0_byte0_not_cmp_flag_rules() -> tuple[FFNRule, ...]:
    """2-rule OR: ``STACK0_B0_NOT_CMP`` fires (=1, BLOCK) on a carried STACK0 row
    that is NOT a clean comparison-drift row, dark (=0, ALLOW) only on the drift
    rows. The dump reads it as a -1000 blocker.

      rule 1 (arith/JMP opcode) — fires when an arith or JMP opcode is present
        (the multi-byte arith RESULT row, the JMP transfer row):
          comparison drift row (no arith/jmp): 1.0(MARK) + 50*0    = 1.0 < 1.5 -> 0
          arith result (OP_ADD ~0.11):         1.0      + 50*0.11  = 6.5 > 1.5 -> 1
          JMP row (OP_JMP ~0.05):              1.0      + 50*0.05  = 3.5 > 1.5 -> 1

      rule 2 (PREV smear) — blocks the add_16bit over-fire whose corrupting STACK0
        row carries NO arith opcode but a SMEARED carried PREV one-hot (substantial
        Σ|PREV| mass with NO single dominant slot). Reads the RATIO-based
        ``STACK0_B0_PREV_DOM`` (magnitude-independent, so a SMALL but CLEAN
        comparison-result one-hot like byte 0x01 is NOT blocked) plus a Σ|PREV|
        mass term (so an EMPTY PREV -- harmless, re-supplying 0 is a no-op -- does
        not fire):
          clean carry (any size): PREV_DOM=1 -> -big -> dark (no block).
          smear (add_16bit):      mass high, PREV_DOM=0 -> fires (block).
          empty PREV:             mass ~0 -> dark (no block).

    Either rule writing 1.0 -> NOT_CMP = 1 (OR). On every non-STACK0 row a -1000
    marker blocker dominates both rules. All gate inputs bounded.
    """
    SIG_W = 1.0
    OP_DRIVE_W = 50.0        # 0.047 (smallest OP_JMP) * 50 = 2.35 clears 1.5
    PREV_DOM_KILL_W = 100.0  # PREV_DOM flag * 100 -> a clean carry strongly blocked
    PREV_MASS_W = 0.02       # Σ|PREV| ~505 (smear) * 0.02 = 10; ~0 (empty/clean)
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    # Rule 1: arithmetic / JMP opcode present.
    op_conditions = [("MARK_STACK0", SIG_W)]
    for op in _STACK0_B0_NOT_CMP_BLOCK_OPS:
        op_conditions.append((op, OP_DRIVE_W))
    op_conditions = tuple(op_conditions) + marker_blockers
    # Rule 2: PREV is a true SMEAR (mass high, no dominant slot -> PREV_DOM absent).
    smear_conditions = (
        ("MARK_STACK0", SIG_W),
        ("STACK0_B0_PREV_DOM", -PREV_DOM_KILL_W),
    ) + tuple(
        (f"STACK0_B0_H1_PREV+{j}", PREV_MASS_W) for j in range(7)
    ) + marker_blockers
    return (
        multi_way_and_rule(
            name="stack0_byte0_not_cmp_opcode",
            conditions=op_conditions,
            threshold=1.5,
            writes=(("STACK0_B0_NOT_CMP", 1.0),),
        ),
        multi_way_and_rule(
            name="stack0_byte0_not_cmp_smear",
            conditions=smear_conditions,
            threshold=1.5,
            writes=(("STACK0_B0_NOT_CMP", 1.0),),
        ),
    )


def make_stack0_byte0_not_cmp_flag_op() -> Operation:
    """Precursor FFN: writes the BOUNDED STACK0 byte-0 NON-COMPARISON blocker flag.

    Standalone ``PureFFN`` post_op on the L25 tail block, ordered BEFORE the dump
    FFN. Reads the per-step COMPARISON + consuming-BRANCH opcode bands at the dump
    block input and writes ``STACK0_B0_NOT_CMP`` = 1 on every carried row WITHOUT
    such an opcode (a -1000 blocker the dump reads) -- the discriminator that lets
    the carry go DEFAULT-ON without regressing add_16bit / jmp_forward / the
    multi-byte mul cluster. MIXED dim_map: the MARK_* gate dims resolve from the
    legacy registry; the NEW ``STACK0_B0_NOT_CMP`` band AND the opcode bands
    resolve from ``dim_positions`` (the WIDENED layout, where the opcodes actually
    live -- the static registry mismaps them).
    """
    rules = _stack0_byte0_not_cmp_flag_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _STACK0_B0_NOT_CMP_FLAG_HIDDEN_DIM, (
            f"stack0_byte0_not_cmp_flag rule-count drift: produced "
            f"{len(rules)}, expected {_STACK0_B0_NOT_CMP_FLAG_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        # The NOT_CMP / PREV_DOM / H1_PREV bands AND the per-step opcode reads
        # resolve from the WIDENED ``dim_positions``; only the MARK_* gate dims
        # come from the registry.
        _layout_bands = (
            {"STACK0_B0_NOT_CMP", "STACK0_B0_PREV_DOM", "STACK0_B0_H1_PREV"}
            | set(_STACK0_B0_NOT_CMP_BLOCK_OPS)
        )
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _layout_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="stack0_byte0_not_cmp_flag",
        reads={
            "MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
            "MARK_MEM", "MARK_SE", "STACK0_B0_PREV_DOM", "STACK0_B0_H1_PREV",
        } | set(_STACK0_B0_NOT_CMP_BLOCK_OPS),
        writes={"STACK0_B0_NOT_CMP"},
        kind="block",
        # On the L25 tail block, ordered AFTER the PREV_DOM precursor (rule 2 reads
        # ``STACK0_B0_PREV_DOM``) and BEFORE the dump FFN that reads NOT_CMP. The
        # per-step opcode bands are stable + bounded at this block input
        # (probe_stack0_arithgate.py).
        target_op_name="l10_post_ops_combined",
        requires={"after": (
            "stack0_byte0_carried_flag", "stack0_byte0_prev_dom_flag",
        )},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )


def make_stack0_byte0_dump_repopulate_op() -> Operation:
    """Append the STACK0 byte-0 dump-repopulate FFN after the L25 tail block.

    Copies ``STACK0_B0_{H1,H3}_PREV`` -> ``STACK0_B0_DUMP_{H1,H3}`` on carried
    STACK0-marker rows (gated on the same-step byte-0 one-hot being absent), so
    the LM head re-emits the carried byte-0. Standalone ``PureFFN`` post_op on
    the L25 tail block (after ``tail_bit32_result_correction``), so it reads the
    held PREV bands AFTER the L25 corruptor has already clobbered H1/H3, and
    nothing downstream overrides the DUMP bands before the LM head.
    """
    rules = _stack0_byte0_dump_repopulate_rules()

    def bake(block, dim_positions, S):
        from ...base_layers import PureFFN

        d_model = None
        attn = getattr(block, "attn", None)
        if attn is not None:
            d_model = getattr(attn, "dim", None)
            if d_model is None and hasattr(attn, "W_q"):
                try:
                    d_model = attn.W_q.shape[0]
                except (AttributeError, IndexError):
                    d_model = None
        if d_model is None and hasattr(block, "ffn") and hasattr(block.ffn, "W_up"):
            try:
                d_model = block.ffn.W_up.shape[1]
            except (AttributeError, IndexError):
                d_model = None
        if d_model is None and isinstance(dim_positions, dict) and dim_positions:
            try:
                d_model = max(int(v) for v in dim_positions.values()) + 1
            except (TypeError, ValueError):
                d_model = None
        if d_model is None:
            d_model = 512
        assert len(rules) == _STACK0_B0_DUMP_REPOPULATE_HIDDEN_DIM, (
            f"stack0_byte0_dump_repopulate rule-count drift: produced "
            f"{len(rules)}, expected {_STACK0_B0_DUMP_REPOPULATE_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        # POSITION SOURCE (mixed) — same split the AX carry / carry head use.
        # The H1/H3 read taps + MARK_* gate dims resolve from the legacy dynamic
        # *registry*; the NEW ``STACK0_B0_*`` bands from ``dim_positions``.
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()
        # ``STACK0_B0_*`` bands resolve from ``dim_positions`` (declarative-only
        # layout); the re-point's ``H1``/``H3`` emission targets and the MARK_*
        # gate dims resolve from the legacy dynamic registry.
        _new_bands = {
            "STACK0_B0_H1_PREV", "STACK0_B0_H3_PREV",
            "STACK0_B0_DUMP_H1", "STACK0_B0_DUMP_H3",
            "STACK0_B0_CARRIED", "STACK0_B0_SHARP", "STACK0_B0_NOT_CMP",
        }
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            if _base in _new_bands:
                dim_map[_nm] = int(dim_positions[_base]) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
            else:
                dim_map[_nm] = int(_reg.slots[_base].start) + (
                    int(_nm.split("+", 1)[1]) if "+" in _nm else 0
                )
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="stack0_byte0_dump_repopulate",
        reads={
            "MARK_STACK0", "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP",
            "MARK_MEM", "MARK_SE", "STACK0_B0_CARRIED", "STACK0_B0_SHARP",
            "STACK0_B0_NOT_CMP", "STACK0_B0_H1_PREV", "STACK0_B0_H3_PREV",
        },
        # Flag ON: re-points into the byte's own H1/H3 emission cells (the fix).
        # Flag OFF: writes the inert STACK0_B0_DUMP_{H1,H3} bands (byte-identical).
        writes={"H1", "H3", "STACK0_B0_DUMP_H1", "STACK0_B0_DUMP_H3"},
        kind="block",
        # Append AFTER the tail correction on the L25 tail block, so this op is
        # the LAST writer of the H1/H3 emission cells before the LM head reads
        # them (it runs after the block-38 corruptor that nukes them). It reads
        # the pre-computed BOUNDED ``STACK0_B0_CARRIED`` + ``STACK0_B0_SHARP``
        # flags (from the precursors) -- NOT the same-step H1/H3 (nuked here).
        target_op_name="l10_post_ops_combined",
        requires={"after": (
            "tail_bit32_result_correction", "stack0_byte0_carried_flag",
            "stack0_byte0_sharp_flag", "stack0_byte0_not_cmp_flag",
        )},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md",
    )
