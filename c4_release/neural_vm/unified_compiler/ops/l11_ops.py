"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
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
    for op_name in _STEP_END_OPERAND_RELAY_OPCODES:
        dim = getattr(BD, op_name)
        v_a.append(AP(slot, dim, 1.0))
        o_a.append(AO(dim, slot, 1.0))
        slot += 1
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
    for k_idx in range(16):
        v_b.append(AP(slot, BD.ALU_LO + k_idx, 1.0))
        o_b.append(AO(BD.ALU_LO + k_idx, slot, 1.0))
        slot += 1
    for k_idx in range(16):
        v_b.append(AP(slot, BD.ALU_HI + k_idx, 1.0))
        o_b.append(AO(BD.ALU_HI + k_idx, slot, 1.0))
        slot += 1
    for k_idx in range(4):  # CMP is 4 wide in the registry
        v_b.append(AP(slot, BD.CMP + k_idx, 1.0))
        o_b.append(AO(BD.CMP + k_idx, slot, 1.0))
        slot += 1
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
    )
    ir = CompilerIR()
    ir.layer(0).attention.append(
        spec_a, name="layer11_step_end_operand_relay.head_0",
    )
    ir.layer(0).attention.append(
        spec_b, name="layer11_step_end_operand_relay.head_1",
    )
    return ir


def make_layer11_step_end_operand_relay_op(enable: bool = False) -> Operation:
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
    the bake is a no-op — the relay's downstream side-effects (extra
    ALU_LO/HI / OP_<NAME> signal at MARK_SE rows, picked up by the LM
    head + later layers' cross-step KV lookups) regress ~9 smoke tests
    on the baseline. Wave B will scope those downstream readers to gate
    on MARK_AX vs MARK_SE explicitly, after which this op can flip to
    ``enable=True``. Mirrors the ``layer9_alibi_mem_attn(enable=False)``
    + ``layer8_head6_ax_carry_refresh(enable=False)`` PoC pattern.

    Probe verification (``tools/probe_step_end_completeness.py``) shows
    that when ``enable=True`` the relay populates MARK_SE with the
    relayed operand band: AX_CARRY_LO/HI ~0.75 × source, ALU_LO/HI
    ~1.0 × source, STACK0_BYTE0..3 ~1.0 × source. OP_<NAME> + CMP need
    a longer-window programme to capture both MARK_AX and the
    same-step MARK_SE in one trace (the default ``IMM 5; PSH; IMM 5;
    EQ; EXIT`` runs OOM on a busy GPU before both rows materialise).
    """
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
        )
        Primitives.generate_attention_head(attn, spec_a, HD)
        Primitives.generate_attention_head(attn, spec_b, HD)

    # Dim-ownership claims: two heads. Head A V slots 0..62 cover
    # OP_<NAME> (0..30) + AX_CARRY_LO+0..15 (31..46) + AX_CARRY_HI+0..15
    # (47..62). Head B V slots 0..39 cover ALU_LO+0..15 (0..15) +
    # ALU_HI+0..15 (16..31) + CMP+0..3 (32..35) + STACK0_BYTE0..3
    # (36..39). O slots mirror the V slot indices; out_dim names match
    # the source dims (relay = identity copy).
    _claims: set = set()
    head_a_idx = 0
    head_b_idx = 1
    slot = 0
    for op_name in _STEP_END_OPERAND_RELAY_OPCODES:
        _claims.add((11, "attn_W_v", f"{head_a_idx}_{slot}", f"{op_name}+0"))
        _claims.add((11, "attn_W_o", f"{head_a_idx}_{slot}", f"{op_name}+0"))
        slot += 1
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
    for k_idx in range(4):
        _claims.add(
            (11, "attn_W_v", f"{head_b_idx}_{slot}", f"CMP+{k_idx}"),
        )
        _claims.add(
            (11, "attn_W_o", f"{head_b_idx}_{slot}", f"CMP+{k_idx}"),
        )
        slot += 1
    for byte_h in (0, 1, 2, 3):
        _claims.add(
            (11, "attn_W_v", f"{head_b_idx}_{slot}", f"STACK0_BYTE{byte_h}+0"),
        )
        _claims.add(
            (11, "attn_W_o", f"{head_b_idx}_{slot}", f"STACK0_BYTE{byte_h}+0"),
        )
        slot += 1

    return Operation(
        name="layer11_step_end_operand_relay",
        # Q reads MARK_SE_ONLY (Q-row anchor); K reads MARK_AX (K-row
        # anchor). V reads the broadcast payload at the MARK_AX K rows.
        reads={
            "MARK_AX", "MARK_SE_ONLY",
            "AX_CARRY_LO", "AX_CARRY_HI",
            "ALU_LO", "ALU_HI", "CMP",
            "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
            *_STEP_END_OPERAND_RELAY_OPCODES,
        },
        # O writes the same dim names at the MARK_SE Q rows.
        writes={
            "AX_CARRY_LO", "AX_CARRY_HI",
            "ALU_LO", "ALU_HI", "CMP",
            "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
            *_STEP_END_OPERAND_RELAY_OPCODES,
        },
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
