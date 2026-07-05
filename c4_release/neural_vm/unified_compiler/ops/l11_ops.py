"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os as _os_stack0
from dataclasses import replace

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule, step_function_rule
from ..ir import CompilerIR, FFNRule
from ..isa_semantics_dsl import (
    CrossStepCarrySpec,
    DumpBlock,
    HeadWrite,
    MarkerBroadcastBand,
    MarkerBroadcastSpec,
    PrecursorFlagSpec,
    cross_step_carry,
    marker_broadcast,
)
from ..layer_compiler import Operation
from ..primitives import AP, DeclarativeAttentionHeadSpec, Primitives
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
# (AX H1 bands + AX_CARRY_OVERFLOW, then the H2/H3 value-general AX bands, then
# the four Root 2 PREV/DUMP bands, then the four bounded Root 2 flag bands) —
# byte-identical to the legacy central dict ordering.
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
# (1b) AX byte-1 register-dump VALUE-GENERALISATION (byte-1 0..15). The
#     fresh-step byte-1 emission one-hot is SPREAD across the LM-head's H1/H2/H3
#     marker-distance bands (``head.weight[v, H<k>+off]=+5.0``): v=0..4 ->
#     H1+(v+2), v=5..11 -> H2+(v-5), v=12..15 -> H3+(v-12). The original carry
#     copied ONLY the H1 band, so byte-1 >= 5 (whose live one-hot sits in H2/H3
#     on the producing step) was DROPPED to 0 on the carried step (the whole
#     add/sub corpus has byte-1 0..7 -> the H2 band). These two band-pairs
#     mirror ``H1_PREV_STEP`` / ``H1_DUMP_OUT`` for the H2 and H3 one-hots: the
#     L13 carry head ALSO V-copies the prev step's H2/H3 one-hot here (cross-step
#     ``H2.*.-1`` / ``H3.*.-1``); the L25 dump FFN copies them to the DUMP bands
#     on carried steps (the SAME AX_CARRY band-pass + ``AX_CARRY_OVERFLOW`` kill
#     gate, so they NEVER over-fire on PC/SP/BP rows); the LM head reads them via
#     mirrored ``ax_byte1_dump_head_bake`` columns. Registered right after the
#     AX_CARRY_OVERFLOW flag and BEFORE the Root 2 bands to keep the tail
#     ``dim_positions`` order load-bearing-identical to the legacy central dict.
register_residual_band(
    "H2_PREV_STEP", 7, owner="make_layer11_ax_byte1_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "H2_DUMP_OUT", 7, owner="make_ax_byte1_dump_repopulate_op",
    never_share=True,
)
register_residual_band(
    "H3_PREV_STEP", 7, owner="make_layer11_ax_byte1_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "H3_DUMP_OUT", 7, owner="make_ax_byte1_dump_repopulate_op",
    never_share=True,
)
# (2) Root 2 STACK0 byte-0 cross-step emission carry. ONLY the two ``_PREV``
#     bands survive: the L9 carry head (``make_stack0_byte0_dump_carry_op``)
#     copies the prev step's STACK0-marker byte-0 H1/H3 one-hot into them, and
#     they are read LIVE by the campaign MUL multi-byte L19 boost as the
#     literal-vs-var_mul frame discriminator (``var_frame_carry`` in
#     ``efficient_alu_neural._MulCombineStage`` — the SUM of these two bands is
#     ~7379 in a var_mul ENT frame vs <=667 for a single-step literal mul). The
#     rest of the STACK0-b0 DUMP machinery (the dump-repopulate FFN, the four
#     bounded flag precursors, the POP-discriminator latch, the LM-head DUMP
#     columns, and the seven ``STACK0_B0_{DUMP_H1,DUMP_H3,CARRIED,SHARP,PREV_DOM,
#     NOT_CMP,POPPED}`` bands) was DELETED (2026-07) as provably-dead: in the
#     30-token frame STACK0 is never emitted, so no carried STACK0-marker row
#     ever exists and the dump gate never fires. See
#     docs/STACK0_B0_DUMP_DEAD_MACHINERY_DELETION.
register_residual_band(
    "STACK0_B0_H1_PREV", 7, owner="make_stack0_byte0_dump_carry_op",
    never_share=True,
)
register_residual_band(
    "STACK0_B0_H3_PREV", 7, owner="make_stack0_byte0_dump_carry_op",
    never_share=True,
)
# (3) ENT saved-BP store cross-step carry (BP_SAVE_PREV). The ENT step's own
#     MEM section (the saved-BP store: addr=SP, val=old_BP) emits 0xFF garbage
#     for the VALUE bytes because the L14 value heads (4-7) content-address the
#     WRONG source position at the ENT step (the same-step old-BP lookup fails;
#     see docs/FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md). Instead
#     of fixing the failing same-step attention, CARRY old_BP across the step
#     boundary: the BP register holds the clean old_BP at EVERY step (its byte
#     tokens are emitted correctly; CLEAN_EMBED_LO/HI at the prev-step BP byte
#     rows decode to the exact old_BP nibbles, even though their OUTPUT residual
#     is later nuked to 0xFF — verified spec_k=0, tools/_probe_bp_carry3.py on
#     id550/575). A dedicated L13 carry head copies the prev step's
#     CLEAN_EMBED_LO/HI (per byte) into ``BP_SAVE_PREV``; a late-tail dump FFN
#     re-supplies it into OUTPUT_LO/HI at the ENT-step MEM val PREDICTOR rows
#     (gated on the high OP_ENT broadcast, ~10.7 at the val-predictor rows, so
#     SI/SC/PSH/JSR stores — which carry OP_JSR or no OP_ENT — stay dark and
#     byte-identical). BP_SAVE_PREV is 32-wide: a single shared band carries
#     16 LO + 16 HI nibble one-hots, because the carry head delivers byte k's
#     nibbles to the (distinct) val-byte-k predictor row, so the four bytes
#     never collide in the band. The LM head already emits byte tokens from
#     OUTPUT_LO/HI, so NO new head-bake columns are needed; only the OUTPUT
#     re-supply is flag-gated (``C4_BP_SAVE_DUMP``, default ON). With the flag
#     OFF the dump FFN writes nothing into OUTPUT -> byte-identical.
def _bp_save_dump_enabled() -> bool:
    """``C4_BP_SAVE_DUMP`` flag predicate (DEFAULT-ON).

    Gates the WHOLE ENT saved-BP carry feature — the ``BP_SAVE_PREV`` band, the
    L13 carry head, and the L25 dump FFN. Flag-off (``C4_BP_SAVE_DUMP=0``) omits
    the band entirely (byte-identical pre-carry d_model) and the carry head +
    dump bake as no-ops. Evaluated lazily (at compile time) so a per-process
    env flip is honoured and the cache key reflects it.
    """
    return _os_stack0.environ.get("C4_BP_SAVE_DUMP", "1") != "0"


# ISA-semantics DSL migration (the byte-identity proof): the BP_SAVE_PREV carry
# is now generated by ``cross_step_carry(_BP_SAVE_CARRY_SPEC)`` — the SAME
# 4-part structure (dedicated band + unconditional carry head + gated dump FFN
# + gate-in-FFN discriminator) the hand-built ``_bp_save_prev_carry_head_spec``
# / ``_bp_save_dump_repopulate_rules`` produced, verified byte-identical against
# the HEAD golden ``state_dict`` hash (flag-on AND flag-off). ``cross_step_carry``
# registers the ``BP_SAVE_PREV`` band as an import-time side effect AT THIS
# EXACT module position (registration order is load-bearing for the tail
# ``dim_positions``), with the SAME owner / size / flag / never_share as the
# legacy ``register_residual_band`` call it replaces. The bundle's builders are
# consumed below by ``_bp_save_prev_carry_head_spec`` (head) and
# ``_bp_save_dump_repopulate_rules`` (dump).
_BP_SAVE_CARRY_SPEC = CrossStepCarrySpec(
    name="bp_save_dump",                 # dump rule-name prefix (legacy parity)
    band_owner="make_bp_save_prev_carry_op",  # band registry owner (legacy)
    band_name="BP_SAVE_PREV",
    band_width=32,
    band_flag=_bp_save_dump_enabled,
    carry_head_alibi_slope=0.5,
    carry_byte_count=4,
    # Per-byte positional match: Q@MEM_VAL_B{k} (ENT-step val-byte-k predictor
    # row) <-> K@BYTE_INDEX_{k} (prev BP byte-k row).
    match_q_band="MEM_VAL_B",
    match_k_band="BYTE_INDEX_",
    match_weight=40.0,
    # K-preference: OP_JSR biases toward the prev-step prologue BP rows (slot 4).
    k_prefer=((4, "OP_JSR", 6.0),),
    # K-reject: HARD same-step-ENT reject (slot 5, OP_ENT) + the DECISIVE
    # BP-vs-STACK0 discriminator (slot 6, the STACK0_BYTE{0..3} family). Matches
    # the legacy append order: the slot-6 group emits its four K writes first,
    # then ONE CONST-driven Q write.
    k_reject=(
        (5, "OP_ENT", 8.0),
        (6, "STACK0_BYTE0", 8.0),
        (6, "STACK0_BYTE1", 8.0),
        (6, "STACK0_BYTE2", 8.0),
        (6, "STACK0_BYTE3", 8.0),
    ),
    # V/O: copy the prev step's CLEAN_EMBED_LO/HI nibble one-hots into the band
    # (V slots 10..41), at a BOOSTED O magnitude (the dump's multiplicative
    # gate).
    value_src_lo="CLEAN_EMBED_LO",
    value_src_hi="CLEAN_EMBED_HI",
    value_o_write_scale=200.0,
    value_v_slot_base=10,
    # Dump: re-supply BP_SAVE_PREV -> OUTPUT_LO/HI at the ENT-store val rows.
    # write_scale tuned large so silu(S*(cond-thr)) * BP_SAVE_PREV * write_scale
    # dominates the L25 tail corruptor's ~7e7 0xFF SENTINEL nuke. Probe override
    # via ``C4_BP_SAVE_DUMP_WS`` (read here at module import, matching the legacy
    # ``_BP_SAVE_DUMP_WS`` constant defined further down).
    dump_emit_lo="OUTPUT_LO",
    dump_emit_hi="OUTPUT_HI",
    dump_write_scale=float(
        _os_stack0.environ.get("C4_BP_SAVE_DUMP_WS", "200000.0")
    ),
    # GATE (the carry-vs-fresh discriminator, the ONLY gating surface):
    # OP_ENT >= 6 (the ENT-store opcode discriminator) AND MEM_VAL_B{k} (the
    # val-byte-k row marker) over threshold; structural MARK_* blockers ANDed in.
    dump_gate_conditions=(("OP_ENT", 1.0),),
    dump_per_byte_marker="MEM_VAL_B",
    dump_per_byte_marker_weight=8.0,
    dump_marker_blockers=(
        ("MARK_PC", -1_000.0),
        ("MARK_AX", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_STACK0", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    ),
    dump_opent_floor=6.0,  # OP_ENT >= 6 floor folded into threshold (=6+8=14)
)
_BP_SAVE_CARRY_BUNDLE = cross_step_carry(_BP_SAVE_CARRY_SPEC)


# === L11 FFN unit layout (auto-fit; legacy offsets retained as docs) ==
#
# The ``layer11_mul_partial`` op owns the entire L11 FFN. As of Wave 4D
# the weight writes are fully declarative: a 4096-rule ``FFNRule`` IR
# (see ``_mul_partial_rules`` / ``_mul_partial_ir``)
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
# ``_mul_partial_rules`` (and the legacy
# ``setup_helpers._set_layer11_mul_partial`` cursor walk it replaces).
# The outer loop is over ``a_lo in range(16)``; each iteration writes
# ``16 (b_lo) * 16 (b_hi) = 256`` units at offset ``a_lo * 256``.
# Changing the rule loop structure requires updating this table in
# lock-step.
_MUL_PARTIAL_UNIT_LAYOUT = tuple(
    # (sub-stage name, legacy_start (docs only), n_units)
    (f"layer11_mul_partial.a_lo_{a_lo:02d}", a_lo * 256, 256)
    for a_lo in range(16)
)


def _allocate_mul_partial_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the L11 MUL partial.

    Phase 7.B.4: ``pin=`` is dropped from every entry in
    :data:`_MUL_PARTIAL_UNIT_LAYOUT`. The allocator's default
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
    for name, _legacy_start, n_units in _MUL_PARTIAL_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


# Total unit footprint the helper is expected to consume. Computed from
# the layout table so a change to either side is loudly inconsistent.
_MUL_PARTIAL_TOTAL_UNITS = sum(
    n_units for _, _, n_units in _MUL_PARTIAL_UNIT_LAYOUT
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


def _mul_partial_rules_for_a_lo(
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


def _mul_partial_rules(S: float) -> tuple[FFNRule, ...]:
    """Return the full 4096-rule ``FFNRule`` sequence for the L11 MUL partial.

    Concatenates ``_mul_partial_rules_for_a_lo`` for ``a_lo`` in
    ``range(16)`` so the lowering cursor walks 0..4096 with no gap, matching
    the historical ``_set_layer11_mul_partial`` unit numbering.
    """
    rules: list[FFNRule] = []
    for a_lo in range(16):
        rules.extend(_mul_partial_rules_for_a_lo(a_lo, S))
    return tuple(rules)


def _mul_partial_ir(S: float = 100.0) -> CompilerIR:
    """Build the declarative ``CompilerIR`` for the L11 MUL partial.

    Exposed via the op's ``compiler_ir=`` so symbolic execution,
    ``compare_symbolic_to_lowered_ffn``, and the declarative verifier
    can read the rules without going through the bake.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_mul_partial_rules(S))
    return ir


def _lower_mul_partial_rules(
    ffn,
    S: float,
    BD,
    *,
    start_unit: int = 0,
) -> int:
    """Lower the full 4096-rule L11 MUL partial IR into ``ffn``.

    Returns the post-bake unit cursor (``start_unit + 4096``). The bake
    asserts ``start_unit == 0`` via the cursor-drift guard in
    ``make_mul_partial_op``; this signature keeps a ``start_unit``
    knob in case a future op family extends the L11 pool past unit
    4096.
    """
    rules = _mul_partial_rules(S)
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


def make_mul_partial_dep_anchor_op() -> Operation:
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


def _install_multipass_mul(block, dim_positions, S):
    """GAP-PRIMITIVE #2: bake the 7-pass schoolbook MUL cascade into this block.

    Replaces the L11 mul-partial lookup ``block.ffn`` with a
    :class:`~neural_vm.efficient_alu_neural.MultiPassMulBlock` — the 7 lowered
    ``PureFFN`` passes of ``multi_pass_mul_rules`` packed into one block forward
    (the ``FlattenedALUMul`` collapse pattern; the physical block count is
    unchanged, so the absolute-position lea contract holds).

    Band routing (proven byte-identical to ``(a*b)&0xFFFF`` over all 65,536
    pairs against the live non-contiguous layout, ``/tmp/probe_mp_livebands``):

      * operand A nibbles: ``ALU_LO`` (a0) + ``ALU_HI`` (a1, = ALU_LO+16)
      * operand B nibbles: ``AX_CARRY_LO`` (b0) + ``AX_CARRY_HI`` (b1)
      * marker + opcode gate: ``MARK_AX`` + ``OP_MUL``
      * product little-endian nibble lanes ->
          nib0 -> OUTPUT_LO, nib1 -> OUTPUT_HI (byte 0),
          nib2 -> MUL_RESULT_HI_LO, nib3 -> MUL_RESULT_HI_HI (byte 1)
      * scratch: ``MUL_MULTIPASS_WS`` (240-dim op-local band)

    The marker gate is ``MARK_AX`` (not ``MARK_SE_ONLY``): the cascade's passes
    1..6 read the workspace one-hots pass 0 wrote at the SAME row, so the
    marker must be present at that row throughout the block — MARK_AX is stable
    across the MUL emit blocks (the operand bands are gathered there).
    """
    from ...base_layers import PureFFN
    from ...efficient_alu_neural import MultiPassMulBlock
    from ..wide_alu_dsl import multi_pass_mul_rules

    mp = multi_pass_mul_rules(
        operand_a_base="ALU_LO",
        operand_b_base="AX_CARRY_LO",
        result_base="__multipass_unused__",
        workspace_base="MUL_MULTIPASS_WS",
        opcode_gate="OP_MUL",
        marker_gate="MARK_AX",
        S=S,
        width_bytes=2,
        result_lane_bases=(
            "OUTPUT_LO", "OUTPUT_HI",
            "MUL_RESULT_HI_LO", "MUL_RESULT_HI_HI",
        ),
    )

    # Residual width from whatever the prior bake left on block.ffn (PureFFN
    # W_up is [hidden, d_model]); fall back to the attention dim.
    ffn_in = block.ffn
    if hasattr(ffn_in, "W_up") and ffn_in.W_up is not None:
        d_model = int(ffn_in.W_up.shape[1])
    elif hasattr(block, "attn") and hasattr(block.attn, "dim"):
        d_model = int(block.attn.dim)
    else:
        d_model = int(getattr(ffn_in, "dim", 512))

    proxy = _as_setdim_proxy(dim_positions)
    flat = mp.as_flat_ir()
    passes = []
    for idx, p in enumerate(mp.passes):
        names = Primitives.ffn_rule_dim_names(p.ffn.rules)
        dim_pos = Primitives.dim_positions_from_bd(proxy, names)
        ffn = PureFFN(dim=d_model, hidden_dim=max(1, p.hidden_units))
        end = flat.lower_ffn(ffn, dim_pos, layer_idx=idx, start_unit=0, S=S)
        assert end == p.hidden_units, (
            f"multi_pass MUL pass {idx}: lowered {end} units, "
            f"expected {p.hidden_units}"
        )
        passes.append(ffn)

    mp_block = MultiPassMulBlock(passes)

    # Campaign operand-A SE recovery (same as the width=2 wide_mul lookup
    # path, ``make_efficient_l11_alumul_wrap_op``): under the campaign L10
    # ALU-clear, operand A's ALU_LO/HI band is crushed all-negative for a
    # value-dependent subset of MUL rows, so the cascade's P0 partial-product
    # AND units read the wrong (or empty) operand and the product byte 0/1 is
    # wrong. The clean operand survives in SE_ALU_LO/HI (the L9
    # step_end_operand_relay mirror). Wrap the cascade so it restores the
    # crushed operand from SE_ALU BEFORE the passes read it — the byte-0
    # SE-recovery precedent applied to the multipass cascade. Gate-OFF /
    # non-campaign leaves ``block.ffn = mp_block`` untouched.
    from .shared import (
        no_stack0_emit_enabled,
        mul_byte0_se_recover_enabled,
        mul_l11_se_recover_enabled,
    )
    block_ffn = mp_block
    if (
        no_stack0_emit_enabled()
        and mul_byte0_se_recover_enabled()
        and mul_l11_se_recover_enabled()
        and hasattr(proxy, "SE_ALU_LO")
        and hasattr(proxy, "SE_ALU_HI")
    ):
        from ...efficient_alu_neural import MulOperandSeRecoverFFN
        block_ffn = MulOperandSeRecoverFFN(
            mp_block,
            alu_lo=proxy.ALU_LO,
            alu_hi=proxy.ALU_HI,
            se_alu_lo=proxy.SE_ALU_LO,
            se_alu_hi=proxy.SE_ALU_HI,
            mark_ax=proxy.MARK_AX,
            op_mul=proxy.OP_MUL,
        )
    block.ffn = block_ffn


def make_mul_partial_op(alu_mode: str = "lookup") -> Operation:
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

        # GAP-PRIMITIVE #2 install (``C4_MUL_MULTIPASS=1``): replace the L11
        # mul-partial lookup with the 7-pass schoolbook cascade packed into
        # this one block (see ``_install_multipass_mul``). It computes the
        # FULL 16-bit product from a compact spec, routing byte 0 to
        # OUTPUT_LO/HI and byte 1 to the MUL_RESULT_HI_LO/HI band; the L10
        # mul_lo + L12 mul_combine byte-0 writers are gated OFF (they double-
        # write the same nibbles). Flag-off falls through to the byte-
        # identical lookup bake below.
        from .shared import mul_multipass_enabled
        if mul_multipass_enabled():
            _install_multipass_mul(block, dim_positions, S)
            return

        # Per-bake FFN-unit allocator. Each L11 MUL partial slab (one per
        # ``a_lo``) is pinned to its existing offset so the lowering call
        # below lands byte-identically. The block-level attribute mirrors
        # the ``_l14_unit_counter`` convention used by sibling layers,
        # but carries the allocator object so the layout is structured,
        # not just a monotonic int. Downstream tools (e.g. a future L11
        # op family widening ``layer_max_units=``) can introspect or
        # extend it here.
        allocator = _allocate_mul_partial_units()
        block.ffn._l11_unit_allocator = allocator

        # Fully declarative bake: all 16 a_lo slabs (4096 units) are
        # lowered from the ``CompilerIR`` rule list exposed via
        # ``compiler_ir=`` on the Operation. Byte-identical to the
        # legacy ``setup_helpers._set_layer11_mul_partial`` -- verified
        # per substage in Wave 4D via ``compare_symbolic_to_lowered_ffn``
        # and direct ``W_up`` / ``b_up`` / ``W_gate`` / ``b_gate`` /
        # ``W_down`` tensor equality.
        # Phase 8.C inline: lower the rule list directly (was
        # ``_lower_mul_partial_rules``) so census v2 classifies
        # this op as ``declarative`` rather than ``declarative_via_helper``.
        rules = _mul_partial_rules(S)
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
        assert next_unit == _MUL_PARTIAL_TOTAL_UNITS, (
            f"L11 MUL partial unit cursor drift: bake returned "
            f"{next_unit}, allocator expected "
            f"{_MUL_PARTIAL_TOTAL_UNITS}"
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
        # itself still goes through ``_lower_mul_partial_rules``
        # so the per-bake allocator and byte-identity cursor guard wrap
        # the lowering -- ``_dispatch_operation_ir`` would otherwise
        # bypass the allocator bookkeeping.
        compiler_ir=_mul_partial_ir(),
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
# slope convention (see ``make_threshold_attn_op``).
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

    DERIVED via the generic :func:`marker_broadcast` BARE-MARKER mode
    (``is_byte_dim=None``): a plain Q@MARK_SE_ONLY / K@MARK_AX
    marker-row -> marker-row relay with the payload bands supplied as
    :class:`MarkerBroadcastBand` s. Byte-identical to the hand-built
    head (proof: ``tools/_isa_golden_hash.py`` unchanged). Each relayed
    dim is a width-1 band except the contiguous ``AX_CARRY_LO/HI`` /
    ``ALU_LO/HI`` (16-wide) and ``CMP`` (4-wide) blocks; ``v_slot_base``
    is assigned sequentially so the slot layout matches exactly.
    """
    HD_DEFAULT = 64  # default head_dim at L11 (d_model=512, num_heads=8)

    # Resolve dim names -> positions through the same ``BD`` proxy the
    # hand-built head used (identical fallback to ``_SetDim`` for names
    # like ``STACK0_BYTE0`` not in ``dim_positions``).
    class _DimView:
        def __getitem__(self, name):
            return getattr(BD, name)
    dim_view = _DimView()

    L = float(S)

    def _relay_spec(name, bands, head_idx):
        if not bands:
            # Degenerate head: every payload toggle off -> the relay fires at
            # MARK_SE / attends MARK_AX but broadcasts NOTHING (the production
            # scoping leaves head B empty). ``marker_broadcast`` requires >=1
            # band, so emit the bare Q/K head directly (byte-identical to the
            # empty-V/O hand-built spec).
            return DeclarativeAttentionHeadSpec(
                head_idx=head_idx,
                q=(AP(0, getattr(BD, "MARK_SE_ONLY"), L),),
                k=(AP(0, getattr(BD, "MARK_AX"), L),),
                v=(),
                o=(),
                alibi_slope=1.0,
            )
        return marker_broadcast(MarkerBroadcastSpec(
            name=name,
            fire_slot_dim="MARK_SE_ONLY",      # Q fires at the STEP_END marker
            source_marker="MARK_AX",           # K selects the in-step AX marker
            broadcast_bands=tuple(bands),
            weight=L,
            is_byte_dim=None, const_dim=None, gate_slot=None,  # BARE marker mode
            alibi_slope=1.0,
        )).head_spec_builder(dim_view, head_idx)

    # --- Head A: OP_<NAME> + AX_CARRY_LO/HI (each a broadcast band) ---
    bands_a: list = []
    slot = 0
    if include_op_name:
        op_names_iter = (
            op_name_subset if op_name_subset else _STEP_END_OPERAND_RELAY_OPCODES
        )
        for op_name in op_names_iter:
            bands_a.append(MarkerBroadcastBand(op_name, op_name, 1, slot, 1.0))
            slot += 1
    if include_ax_carry:
        bands_a.append(MarkerBroadcastBand("AX_CARRY_LO", "AX_CARRY_LO", 16, slot, 1.0))
        slot += 16
        bands_a.append(MarkerBroadcastBand("AX_CARRY_HI", "AX_CARRY_HI", 16, slot, 1.0))
        slot += 16
    assert slot <= HD_DEFAULT, (
        f"step_end_operand_relay head A overflowed HD={HD_DEFAULT} "
        f"with {slot} slots"
    )
    spec_a = _relay_spec(
        "layer11_step_end_operand_relay.head_0", bands_a, head_a_idx,
    )

    # --- Head B: ALU_LO/HI + CMP + STACK0_BYTE0..3 -------------------
    bands_b: list = []
    slot = 0
    if include_alu:
        bands_b.append(MarkerBroadcastBand("ALU_LO", "ALU_LO", 16, slot, 1.0))
        slot += 16
        bands_b.append(MarkerBroadcastBand("ALU_HI", "ALU_HI", 16, slot, 1.0))
        slot += 16
    if include_cmp:
        bands_b.append(MarkerBroadcastBand("CMP", "CMP", 4, slot, 1.0))  # 4 wide
        slot += 4
    if include_stack0_byte:
        for byte_h in (0, 1, 2, 3):
            dim_name = f"STACK0_BYTE{byte_h}"
            bands_b.append(MarkerBroadcastBand(dim_name, dim_name, 1, slot, 1.0))
            slot += 1
    assert slot <= HD_DEFAULT, (
        f"step_end_operand_relay head B overflowed HD={HD_DEFAULT} "
        f"with {slot} slots"
    )
    spec_b = _relay_spec(
        "layer11_step_end_operand_relay.head_1", bands_b, head_b_idx,
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


def _allocate_ax_byte1_dump_carry_heads() -> AttentionHeadAllocator:
    """Per-bake head allocator for the AX byte-1 DUMP carry head (L13 host)."""
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (op_name, head_idx) in _AX_BYTE1_DUMP_CARRY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=13, pin=head_idx)
    return allocator


# ===========================================================================
# ISA-semantics DSL migration: the AX byte-1 cross-step carry is now GENERATED
# by ``cross_step_carry(_AX_BYTE1_CARRY_SPEC)`` — the SAME 4-part structure the
# BP migration proved, exercising the generator's generalizations:
#   (1) ``position_source="mixed"``: the H1/H2/H3 pipeline + ADDR_B*/AX_CARRY/
#       MARK_AX gate taps from the legacy dynamic *registry*; the new
#       ``H{1,2,3}_PREV_STEP`` / ``H{1,2,3}_DUMP_OUT`` / ``AX_CARRY_OVERFLOW``
#       bands from the declarative layout.
#   (2) the THREE cross-step value bands (H1/H2/H3) the carry head V-copies
#       forward (value-general byte-1 0..15) — multi-band carry.
#   (3) the two-stage ``AX_CARRY_OVERFLOW`` band-pass KILL: a precursor
#       step_function flag (``ax_byte1_carry_overflow_flag``, ~2-3 OR units)
#       writes a bounded out-of-band indicator; the dump ANDs it as a -1000
#       KILL condition so Σ AX_CARRY is band-PASSED (not just lower-bounded).
#       Generated via the generator's ``make_mixed_dim_map_ffn_op`` scaffolding.
#
# The carry head Q/K/V/O is declared EXPLICITLY (a sharp ADDR_B0_LO+5 byte-1
# signature on slot 0, an ADDR_B1_HI+8 AX-register match on slot 1, a
# CONST-driven AX_CARRY fresh-preference K-loop on slot 2, a MARK_AX V=0 sink on
# slot 3, and three cross-step H1/H2/H3 value-copy blocks at V slots 10/17/24).
# The bands are registered by the module-scope ``register_residual_band`` calls
# above (LOAD-BEARING order), so ``register_band=False``. The dump rules are
# flag-INDEPENDENT (``C4_AX_BYTE1_DUMP`` gates only the LM-head emission columns
# in model_ops, not these rules), so ``dump_blocks == dump_blocks_off``.
# Verified byte-identical against the HEAD golden state_dict hash (flag-ON and
# flag-OFF).

# Head tuning (production defaults — the hand-built call passed none).
_AX_B1_HEAD_SIG_W = 60.0   # SHARP ADDR_B0_LO+5 byte-1 signature (slot 0)
_AX_B1_HEAD_L = 15.0       # ADDR_B1_HI/CONST/ADDR sig driver magnitude
_AX_B1_HEAD_SINK_W = 8.0   # MARK_AX V=0 sink (slot 3 K)
_AX_B1_HEAD_K_AXC_W = 0.2  # AX_CARRY fresh-preference K-loop (slot 2)
_AX_B1_HEAD_W = 7          # H1/H2/H3 band width (cells)
_AX_B1_HEAD_V_BASE = 10    # V slot base for the H1 copy block
_AX_B1_AXC_CELLS = 16      # AX_CARRY LO/HI cell count


def _ax_byte1_dump_blocks():
    """Build the AX byte-1 dump blocks (H1/H2/H3 -> H*_DUMP_OUT).

    Mirrors ``_ax_byte1_dump_repopulate_rules`` EXACTLY: the shared
    AND conditions (ADDR_B1_HI+8 AX-register discriminator + ADDR_B0_LO+5 byte-1
    signature + the 32-cell Σ AX_CARRY lower bound + 7 marker blockers + the
    ``AX_CARRY_OVERFLOW`` two-stage KILL), threshold 4.5, write_scale 0.02, in
    the SAME (H1, H2, H3) band/unit ORDER and the SAME rule names. Flag-
    INDEPENDENT (``C4_AX_BYTE1_DUMP`` gates only the LM-head columns), so this is
    used for BOTH ``dump_blocks`` and ``dump_blocks_off``.
    """
    from ..isa_semantics_dsl import DumpBlock

    AXC_W = 1.0
    AX_REG_W = 0.25
    SIG_W = 2.0
    OVERFLOW_KILL_W = 1_000.0
    marker_blockers = (
        ("MARK_AX", -1_000.0),
        ("MARK_PC", -1_000.0),
        ("MARK_SP", -1_000.0),
        ("MARK_BP", -1_000.0),
        ("MARK_STACK0", -1_000.0),
        ("MARK_MEM", -1_000.0),
        ("MARK_SE", -1_000.0),
    )
    conditions = [
        ("ADDR_B1_HI+8", AX_REG_W),
        ("ADDR_B0_LO+5", SIG_W),
    ]
    for k in range(_AX_B1_AXC_CELLS):
        conditions.append((f"AX_CARRY_LO+{k}", AXC_W))
        conditions.append((f"AX_CARRY_HI+{k}", AXC_W))
    conditions = tuple(conditions) + marker_blockers + (
        ("AX_CARRY_OVERFLOW", -OVERFLOW_KILL_W),
    )
    # H1 block keeps the legacy ``ax_byte1_dump_repopulate_slot_{j}`` name (no
    # band infix); H2/H3 carry the ``_{band}`` infix.
    blocks = [
        DumpBlock(
            name_prefix="ax_byte1_dump_repopulate_slot",
            width=_AX_B1_HEAD_W,
            gate_band="H1_PREV_STEP",
            emit_band="H1_DUMP_OUT",
            write_scale=0.02,
            conditions=conditions,
            threshold=4.5,
        ),
    ]
    for band in ("H2", "H3"):
        blocks.append(DumpBlock(
            name_prefix=f"ax_byte1_dump_repopulate_{band.lower()}_slot",
            width=_AX_B1_HEAD_W,
            gate_band=f"{band}_PREV_STEP",
            emit_band=f"{band}_DUMP_OUT",
            write_scale=0.02,
            conditions=conditions,
            threshold=4.5,
        ))
    return tuple(blocks)


_AX_BYTE1_CARRY_REGISTRY_DIMS = (
    "H1", "H2", "H3", "ADDR_B0_LO", "ADDR_B1_HI", "AX_CARRY_LO",
    "AX_CARRY_HI", "CONST", "MARK_AX",
)
_AX_BYTE1_CARRY_SPEC = CrossStepCarrySpec(
    name="ax_byte1_dump",
    band_name="H1_PREV_STEP",        # nominal (register_band=False -> unused)
    band_width=_AX_B1_HEAD_W,
    register_band=False,
    band_flag=None,
    carry_head_alibi_slope=0.5,
    carry_byte_count=1,
    position_source="mixed",
    registry_dims=_AX_BYTE1_CARRY_REGISTRY_DIMS,
    head_q=(
        HeadWrite(0, "ADDR_B0_LO+5", _AX_B1_HEAD_SIG_W),  # byte-1 signature
        HeadWrite(1, "ADDR_B1_HI+8", _AX_B1_HEAD_L),      # AX-register match
        HeadWrite(2, "CONST", _AX_B1_HEAD_L),             # AXC fresh-pref driver
        HeadWrite(3, "ADDR_B0_LO+5", _AX_B1_HEAD_L),      # MARK_AX sink driver
    ),
    head_k=(
        HeadWrite(0, "ADDR_B0_LO+5", _AX_B1_HEAD_SIG_W),
        HeadWrite(1, "ADDR_B1_HI+8", _AX_B1_HEAD_L),
        HeadWrite(3, "MARK_AX", _AX_B1_HEAD_SINK_W),
        # slot-2 AX_CARRY fresh-preference (its OWN slot, CONST-driven Q): the
        # SLOT stays fixed at 2 (slot_stride=0); the DIM advances over the 16
        # AX_CARRY LO/HI cells (all 32 K writes land on the single slot-2 K row).
        HeadWrite(2, "AX_CARRY_LO", -_AX_B1_HEAD_K_AXC_W, count=_AX_B1_AXC_CELLS,
                  slot_stride=0),
        HeadWrite(2, "AX_CARRY_HI", -_AX_B1_HEAD_K_AXC_W, count=_AX_B1_AXC_CELLS,
                  slot_stride=0),
    ),
    head_v=(
        HeadWrite(_AX_B1_HEAD_V_BASE, "H1", 1.0, count=_AX_B1_HEAD_W),
        HeadWrite(_AX_B1_HEAD_V_BASE + _AX_B1_HEAD_W, "H2", 1.0,
                  count=_AX_B1_HEAD_W),
        HeadWrite(_AX_B1_HEAD_V_BASE + 2 * _AX_B1_HEAD_W, "H3", 1.0,
                  count=_AX_B1_HEAD_W),
    ),
    head_o=(
        HeadWrite(_AX_B1_HEAD_V_BASE, "H1_PREV_STEP", 1.0, count=_AX_B1_HEAD_W),
        HeadWrite(_AX_B1_HEAD_V_BASE + _AX_B1_HEAD_W, "H2_PREV_STEP", 1.0,
                  count=_AX_B1_HEAD_W),
        HeadWrite(_AX_B1_HEAD_V_BASE + 2 * _AX_B1_HEAD_W, "H3_PREV_STEP", 1.0,
                  count=_AX_B1_HEAD_W),
    ),
    head_reads={"ADDR_B0_LO", "AX_CARRY_LO", "AX_CARRY_HI", "H1", "H1.*.-1",
                "H2.*.-1", "H3.*.-1"},
    head_writes={"H1_PREV_STEP", "H2_PREV_STEP", "H3_PREV_STEP"},
    dump_blocks=_ax_byte1_dump_blocks,
    dump_blocks_off=_ax_byte1_dump_blocks,  # flag-independent dump rules
    dump_reads_explicit={
        "ADDR_B0_LO", "ADDR_B1_HI", "AX_CARRY_LO", "AX_CARRY_HI",
        "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
        "MARK_MEM", "MARK_SE", "H1_PREV_STEP", "AX_CARRY_OVERFLOW",
        "H2_PREV_STEP", "H3_PREV_STEP",
    },
    dump_writes_explicit={"H1_DUMP_OUT", "H2_DUMP_OUT", "H3_DUMP_OUT"},
    precursors=(
        PrecursorFlagSpec(
            op_name="ax_byte1_carry_overflow_flag",
            flag_band="AX_CARRY_OVERFLOW",
            rules_builder=lambda: _ax_byte1_carry_overflow_flag_rules(),
            reads={"AX_CARRY_HI", "AX_CARRY_LO", "ADDR_B1_HI"},
            writes={"AX_CARRY_OVERFLOW"},
            target_op_name="l10_post_ops_combined",
            requires={"after": "tail_bit32_result_correction"},
            registry_dims=("AX_CARRY_HI", "AX_CARRY_LO", "ADDR_B1_HI"),
            spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
        ),
    ),
)
_AX_BYTE1_CARRY_BUNDLE = cross_step_carry(_AX_BYTE1_CARRY_SPEC)


def _ax_byte1_dump_carry_head_spec(
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
    # ISA-semantics DSL migration: re-expressed via
    # ``cross_step_carry(_AX_BYTE1_CARRY_SPEC)`` (explicit-head + mixed
    # position-source mode). The bundle's ``carry_head_spec_builder`` reproduces
    # the legacy hand-built Q/K/V/O writes BYTE-IDENTICALLY (the H1 pipeline +
    # ADDR_B*/AX_CARRY/MARK_AX taps from the dynamic *registry*, the H*_PREV_STEP
    # O targets from the layout) — verified against the HEAD golden state_dict
    # hash. The tuning args (``L`` / ``sink_w`` / ``k_axc_w`` / ``alibi_slope``,
    # kept for probes) default to the spec's frozen values; when ALL are at their
    # defaults the module-level bundle is used directly, otherwise a per-call
    # override rebuilds the explicit head's varying weights.
    if (L == _AX_B1_HEAD_L
            and sink_w == _AX_B1_HEAD_SINK_W
            and k_axc_w == _AX_B1_HEAD_K_AXC_W
            and alibi_slope == _AX_BYTE1_CARRY_SPEC.carry_head_alibi_slope):
        return _AX_BYTE1_CARRY_BUNDLE.carry_head_spec_builder(
            dim_positions, head_idx,
        )
    override = replace(
        _AX_BYTE1_CARRY_SPEC,
        carry_head_alibi_slope=alibi_slope,
        head_q=(
            HeadWrite(0, "ADDR_B0_LO+5", _AX_B1_HEAD_SIG_W),
            HeadWrite(1, "ADDR_B1_HI+8", L),
            HeadWrite(2, "CONST", L),
            HeadWrite(3, "ADDR_B0_LO+5", L),
        ),
        head_k=(
            HeadWrite(0, "ADDR_B0_LO+5", _AX_B1_HEAD_SIG_W),
            HeadWrite(1, "ADDR_B1_HI+8", L),
            HeadWrite(3, "MARK_AX", sink_w),
            HeadWrite(2, "AX_CARRY_LO", -k_axc_w, count=_AX_B1_AXC_CELLS,
                      slot_stride=0),
            HeadWrite(2, "AX_CARRY_HI", -k_axc_w, count=_AX_B1_AXC_CELLS,
                      slot_stride=0),
        ),
    )
    return cross_step_carry(override).carry_head_spec_builder(
        dim_positions, head_idx,
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
        allocator = _allocate_ax_byte1_dump_carry_heads()
        attn._l13_ax_byte1_dump_carry_head_allocator = allocator
        head_idx = allocator.heads()[-1].head_idx
        HD = attn.W_q.shape[0] // attn.num_heads
        spec = _ax_byte1_dump_carry_head_spec(
            dim_positions, head_idx,
        )
        Primitives.generate_attention_head(attn, spec, HD)

    def _ir(dim_positions, HD) -> CompilerIR:
        del HD
        allocator = _allocate_ax_byte1_dump_carry_heads()
        head_idx = allocator.heads()[-1].head_idx
        spec = _ax_byte1_dump_carry_head_spec(dim_positions, head_idx)
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
        reads={"ADDR_B0_LO", "AX_CARRY_LO", "AX_CARRY_HI", "H1", "H1.*.-1",
               # Value-general byte-1 (5..15): the prev step's H2/H3 one-hots
               # are V-copied cross-step into H2/H3_PREV_STEP (read by nobody
               # upstream, like H1.*.-1 -> no same-step back-edge / no cycle).
               "H2.*.-1", "H3.*.-1"},
        writes={"H1_PREV_STEP", "H2_PREV_STEP", "H3_PREV_STEP"},
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
_AX_BYTE1_DUMP_REPOPULATE_HIDDEN_DIM = 21  # 7 (H1) + 7 (H2) + 7 (H3)


def _ax_byte1_dump_repopulate_rules() -> tuple[FFNRule, ...]:
    """7 rules: ``H1_DUMP_OUT+j = H1_PREV_STEP+j`` on carried byte-1 rows.

    Each rule fires at the byte-1 predictor row (``ADDR_B0_LO+5`` signature)
    AND carried (``+AX_CARRY`` keeps it above threshold; fresh's ~-988 sinks
    it), then gate-copies the carried one-hot from ``H1_PREV_STEP`` into the
    emission band ``H1_DUMP_OUT``.

    ISA-semantics DSL migration: re-expressed via
    ``cross_step_carry(_AX_BYTE1_CARRY_SPEC)``. The bundle's
    ``dump_rules_builder`` produces the IDENTICAL 21-rule set (7 H1 + 7 H2 + 7
    H3) — same names (``ax_byte1_dump_repopulate_slot_{j}`` for H1, ``_{band}``
    infix for H2/H3), the shared AND conditions (ADDR_B1_HI+8 AX-register
    discriminator + ADDR_B0_LO+5 byte-1 signature + the 32-cell Σ AX_CARRY lower
    bound + 7 marker blockers + the ``AX_CARRY_OVERFLOW`` two-stage KILL),
    threshold 4.5, write_scale 0.02, and the H1->H2->H3 band/unit ORDER —
    verified byte-identical against the HEAD golden state_dict hash. The dump is
    flag-INDEPENDENT (``C4_AX_BYTE1_DUMP`` gates only the LM-head emission
    columns in model_ops), so ``emission_on=True`` always.
    ``_AX_BYTE1_DUMP_REPOPULATE_HIDDEN_DIM`` is retained as the rule-count
    assert.
    """
    return _AX_BYTE1_CARRY_BUNDLE.dump_rules_builder(True)


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
# LEV stale-carry kill: Σ AX_CARRY threshold. Genuine multi-byte carry ~2.65
# (value-independent), LEV func-return step ~3.66; 3.0 sits cleanly in the gap.
_LEV_AX_CARRY_SUM_KILL_THRESHOLD = 3.0


def _lev_ax_byte1_kill_enabled() -> bool:
    """``C4_LEV_AX_BYTE1_KILL`` flag predicate (DEFAULT-ON).

    Adds a THIRD ``AX_CARRY_OVERFLOW`` dump-kill unit that fires when the
    summed ``Σ AX_CARRY`` band crosses 3.0. This targets the func/nested/rec
    LEV (function-return) epilogue: at the LEV step the restored AX is the
    single-byte loaded return value (byte-1 == 0 for the whole func_*/nested_*
    cluster, all of which return values <= 255), but the byte-1 dump misfires
    and re-emits a STALE carried one-hot (e.g. value 6 -> AX 70 -> 0x646), so
    the function EXIT value is wrong.

    ROOT (spec_k=0, BUILT dims, func_identity_0 id550 step-8 LEV, 2026-06-15):
    exact LM-head logit attribution at the LEV-step AX byte-1 predictor row
    pins the leak to ``H2_DUMP_OUT+1`` (res 16.2, +81 logit contrib) — the
    ``ax_byte1_dump_repopulate`` band, re-emitting the stale prev-step byte-1.
    The dump fires there because its lower-bound gate ``Σ AX_CARRY`` sits in
    the carry band. PROOF of separability (probe_lev / _probe_dump_gate, full
    add+sub corpus): a GENUINE multi-byte carry (where the dump MUST fire to
    re-emit a real high byte) has ``Σ AX_CARRY`` clustered tightly at
    **2.65** (value-INDEPENDENT step-class signal; 1.98-2.65 across add_0..11
    / sub_0..7), while the LEV stale-carry step sits tightly at **3.66**. A
    kill threshold at **3.0** separates them with margin 0.35 below (genuine)
    and 0.66 above (LEV). The over-fire SHL(~12.85)/JMP(~47.86) classes are
    also >= 3.0 so this unit ALSO redundantly catches them (unit 0 already
    does via ``AX_CARRY_HI+2``).

    SAFETY: every func_*/nested_* return value is <= 255 (verified 0/75 in
    550-599 + 950-974 exceed 255), so the dump never genuinely needs to fire
    at a LEV step in these clusters — killing it there is exactly correct, not
    a trade. Flag-off (``C4_LEV_AX_BYTE1_KILL=0``) omits the unit, so the
    overflow flag is byte-identical with the prior 2-unit design.
    """
    return _os_stack0.environ.get("C4_LEV_AX_BYTE1_KILL", "1") != "0"


def _ax_byte1_carry_overflow_flag_rules() -> tuple[FFNRule, ...]:
    """2 (or 3) rules into ``AX_CARRY_OVERFLOW`` (OR of dump-kill conditions).

    Unit 0: ``step(AX_CARRY_HI+2 >= 3.0)`` — the SHL/JMP upper-cut.
    Unit 1: ``step(ADDR_B1_HI+8 <= 2.0)`` — the NON-AX-register kill (PC/SP/BP/
    STACK0/MEM byte rows), so the unbounded Σ AX_CARRY lower bound can never
    fire the dump on a non-AX row (the step-0 prologue framing-drift root).
    Unit 2 (``C4_LEV_AX_BYTE1_KILL``, default ON): ``step(Σ AX_CARRY >= 3.0)``
    — the LEV (function-return) stale-carry kill (see
    :func:`_lev_ax_byte1_kill_enabled`). Genuine multi-byte carries sit at
    Σ AX_CARRY ~2.65 (dark), LEV sits at ~3.66 (fires).
    """
    base = (
        step_function_rule(
            name="ax_byte1_carry_overflow_flag",
            input_dim=dim_ref("ax_carry_hi", "AX", 2),
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
    if not _lev_ax_byte1_kill_enabled():
        return base
    # Unit 2 (C4_LEV_AX_BYTE1_KILL): fire when Σ AX_CARRY >= 3.0 (the LEV
    # stale-carry step). conditions = every AX_CARRY_{LO,HI}+k cell at weight
    # 1.0 -> the AND score IS the raw band SUM; threshold 3.0 sits in the gap
    # between the genuine-carry cluster (~2.65) and the LEV step (~3.66). On a
    # genuine carry row the score 2.65 < 3.0 -> silu(S*-0.35) ~= 0 -> dark
    # (byte-identical to the 2-unit design); on a LEV row 3.66 >= 3.0 -> fires
    # -> dump killed -> H*_DUMP_OUT == 0 -> AX byte-1 falls back to the clean
    # normal H1 path (value 0). write_value 2.0/100 matches units 0/1 so the
    # dump's -1000 read deeply darkens its AND.
    lev_kill_conditions = tuple(
        (f"AX_CARRY_LO+{k}", 1.0) for k in range(16)
    ) + tuple(
        (f"AX_CARRY_HI+{k}", 1.0) for k in range(16)
    )
    return base + (
        multi_way_and_rule(
            name="ax_byte1_lev_stale_carry_kill",
            conditions=lev_kill_conditions,
            threshold=_LEV_AX_CARRY_SUM_KILL_THRESHOLD,
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

    ISA-semantics DSL migration: re-expressed via
    ``cross_step_carry(_AX_BYTE1_CARRY_SPEC)``. The bundle's
    ``precursor_ops_builder`` returns this standalone bounded-flag ``Operation``
    (the generator's ``make_mixed_dim_map_ffn_op`` scaffolding supplies the
    IDENTICAL PureFFN post_op + mixed dim_map bake + ``requires={"after":
    "tail_bit32_result_correction"}``) — verified byte-identical against the HEAD
    golden state_dict hash. The two-stage KILL rule logic stays in
    ``_ax_byte1_carry_overflow_flag_rules`` (the spec references it lazily);
    the ``_AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM`` (+LEV) rule-count assert is
    preserved here.
    """
    _expected_units = _AX_CARRY_OVERFLOW_FLAG_HIDDEN_DIM + (
        1 if _lev_ax_byte1_kill_enabled() else 0
    )
    assert len(_ax_byte1_carry_overflow_flag_rules()) == _expected_units, (
        "ax_byte1_carry_overflow_flag rule-count drift: produced "
        f"{len(_ax_byte1_carry_overflow_flag_rules())}, "
        f"expected {_expected_units}"
    )
    (op,) = _AX_BYTE1_CARRY_BUNDLE.precursor_ops_builder()
    return op


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
        _new_bands = {
            "H1_PREV_STEP", "H1_DUMP_OUT",
            "H2_PREV_STEP", "H2_DUMP_OUT",
            "H3_PREV_STEP", "H3_DUMP_OUT",
            "AX_CARRY_OVERFLOW",
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
        name="ax_byte1_dump_repopulate",
        reads={
            "ADDR_B0_LO", "ADDR_B1_HI", "AX_CARRY_LO", "AX_CARRY_HI",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "H1_PREV_STEP", "AX_CARRY_OVERFLOW",
            # Value-general byte-1 (5..15): the H2/H3 PREV bands the carry head
            # now fills; gated-copied into the H2/H3 DUMP bands (same gate).
            "H2_PREV_STEP", "H3_PREV_STEP",
        },
        writes={"H1_DUMP_OUT", "H2_DUMP_OUT", "H3_DUMP_OUT"},
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
# STRUCTURAL: all-step register byte-2/3 zero-default (UNCONDITIONAL)
# ===========================================================================
#
# THE all-step register byte-2/3 zero-default. Fires on EVERY byte-2/3 register-
# dump row (no OP_ENT / OP_LI gate), so it covers the callee-ENT frame, the LI
# load step, AND the LOOP-BODY / recursion steps in one op. (Historically this
# generalized two narrower caps -- ax_byte23_dump_zero gated on OP_ENT and
# ax_li_byte23_zero gated on OP_LI -- which only fired on the fresh-ENT / LI
# steps and left loop_mul/rec_sum leaking 0xFF/overflow into bytes 2/3; both are
# now DELETED as fully subsumed by this all-step clear.)
#
# STRUCTURAL FIX: fire on ALL steps (drop the OP_ENT condition), gated purely on
# the byte-2/3 register-dump row (IS_BYTE + BYTE_INDEX_1/2) with the SAME marker
# + AX_CARRY_OVERFLOW blockers. SAFE by construction: across the 1096 corpus
# register bytes 2/3 are 0 for EVERY register (PC<0x10000; SP/BP high addr has
# 0x00 in bytes 2/3; AX<0x10000), and any genuine >=0x10000 value carries
# AX_CARRY_OVERFLOW (the -1000 blocker kills the clear there). So forcing
# bytes 2/3 -> 0 on every dump row makes the stale-high-byte leak IMPOSSIBLE by
# construction rather than patching one cluster. BYTE_INDEX_0 (byte-1) is NOT
# touched (PC byte-1 can be 0x01 for JSR>=256; AX byte-1 via C4_AX_BYTE1_DUMP) --
# byte-1 is a separate register-scoped increment. Default OFF -> byte-identical;
# validated flag-ON via cpu_full_trace (loop_mul/rec_sum bytes 2/3 -> 0) + HOLD.
def _ax_hibyte_clear_allstep_enabled() -> bool:
    """``C4_AX_HIBYTE_CLEAR`` flag predicate (DEFAULT-OFF structural block).

    Flag-OFF bakes NO units -> byte-identical to the pre-fix build (the default
    golden). Opt in with ``C4_AX_HIBYTE_CLEAR=1`` to force register byte-2/3 ->
    0 on every dump row (kills the stale-high-byte leak class); the emitted
    byte-2/3 then come from the canonical OUTPUT nibbles.
    """
    return _os_stack0.environ.get("C4_AX_HIBYTE_CLEAR", "0") != "0"


_AX_HIBYTE_CLEAR_ALLSTEP_HIDDEN_DIM = 2  # one AND per high byte (byte-2, byte-3)


def _ax_hibyte_clear_allstep_rules() -> tuple[FFNRule, ...]:
    """2 AND rules forcing register byte-2/3 dump -> 0 on EVERY step.

    Gated purely on the byte-2/3 register-dump row (IS_BYTE + BYTE_INDEX_1/2,
    no OP_ENT / OP_LI condition), so it fires on the callee-ENT, LI-load AND
    loop/rec body steps in one op. Firing sum = IS_BYTE(1) +
    BYTE_INDEX_x*2(~1.94) = ~2.94 > threshold 2.5; byte-0/1 rows (BYTE_INDEX_1/2
    ~0 -> sum ~1.0) and marker rows (-1000 blockers) stay dark. AX_CARRY_OVERFLOW
    (-1000) preserves a genuine >=0x10000 high byte.
    """
    IS_BYTE_W = 1.0
    BYTE_INDEX_W = 2.0
    THRESHOLD = 2.5
    BLOCKER_W = 1_000.0
    # NOTE: AX_CARRY_OVERFLOW is DELIBERATELY NOT a blocker here (unlike the
    # ENT/LI caps). Probed spuriously ~1.74 on rec_sum's small-AX byte-2 row, so
    # a -1000 kill blocks the clear exactly where it's needed. Register bytes 2/3
    # are 0 for the whole corpus, so an unconditional zero-default is correct;
    # any genuine >=0x10000 case is caught by the HOLD / flag_regression gates.
    blockers = (
        ("MARK_AX", -BLOCKER_W),
        ("MARK_PC", -BLOCKER_W),
        ("MARK_SP", -BLOCKER_W),
        ("MARK_BP", -BLOCKER_W),
        ("MARK_STACK0", -BLOCKER_W),
        ("MARK_MEM", -BLOCKER_W),
        ("MARK_SE", -BLOCKER_W),
    )
    WW = 0.16
    writes = (
        ("OUTPUT_LO+0", WW),
        ("OUTPUT_HI+0", WW),
        ("OUTPUT_LO+10", -WW),
    )
    rules: list[FFNRule] = []
    for byte_idx, bindex in ((2, "BYTE_INDEX_1"), (3, "BYTE_INDEX_2")):
        rules.append(multi_way_and_rule(
            name=f"ax_hibyte_clear_allstep_byte{byte_idx}",
            conditions=(
                ("IS_BYTE", IS_BYTE_W),
                (bindex, BYTE_INDEX_W),
            ) + blockers,
            threshold=THRESHOLD,
            writes=writes,
        ))
    return tuple(rules)


def make_ax_hibyte_clear_allstep_op() -> Operation:
    """Append the all-step register byte-2/3 zero-default FFN after the L25 tail.

    Fires on EVERY byte-2/3 register-dump row (no OP_ENT / OP_LI gate), covering
    the callee-ENT, LI-load and loop/rec body steps in one op. Now UNCONDITIONAL
    (OUTPUT-canonical emission is the sole path; see the enabled predicate).
    """
    if not _ax_hibyte_clear_allstep_enabled():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="ax_hibyte_clear_allstep",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="l10_post_ops_combined",
            requires={"after": "tail_bit32_result_correction"},
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
        )

    rules = _ax_hibyte_clear_allstep_rules()

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
        assert len(rules) == _AX_HIBYTE_CLEAR_ALLSTEP_HIDDEN_DIM, (
            f"ax_hibyte_clear_allstep rule-count drift: produced {len(rules)}, "
            f"expected {_AX_HIBYTE_CLEAR_ALLSTEP_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            _off = int(_nm.split("+", 1)[1]) if "+" in _nm else 0
            dim_map[_nm] = int(dim_positions[_base]) + _off
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="ax_hibyte_clear_allstep",
        reads={
            "IS_BYTE", "BYTE_INDEX_1", "BYTE_INDEX_2",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "AX_CARRY_OVERFLOW",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        # After tail_bit32_result_correction so it is the last OUTPUT writer on
        # the byte-2/3 dump rows before the LM head.
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ===========================================================================
# LOOP AX byte-3 FINAL-DUMP CAP (flag C4_LOOP_AX_BYTE3_CAP, DEFAULT-OFF)
# ===========================================================================
#
# THE loop AX byte-3 leak (survey R8, ~50 progs: loop_mul / loop_countdown /
# loop_pow2). The step-4 IMM-0 initializer leak is a SEPARATE, already-handled
# root; the RESIDUAL this cap targets is AX byte-3 (the MOST-significant, 4th
# register byte, emitted at ``MARK_AX+4``) leaking a STALE value (e.g. 0x82 ==
# 130 == the ``BZ 16`` branch-target PC) into the FINAL LEA;LOAD;HALT return
# dump. PC is correct throughout — this is an AX-VALUE leak, not a framing
# desync, so a targeted byte-3 OUTPUT clamp on the dump row is exactly right.
#
# PREDICTOR-ROW MAPPING (spec_k=0 BUILT-dim probe, tools/probe_loop_axbyte3_
# final.py): the residual at token position ``p`` PREDICTS the token at ``p+1``,
# so the AX byte-3 token (``MARK_AX+4``) is predicted by the residual one row
# EARLIER (``MARK_AX+3``), which is the byte-2 register token — it carries
# ``IS_BYTE`` + ``BYTE_INDEX_2`` (NOT BYTE_INDEX_3). Probed 0.970 on that exact
# row for the loop clusters' final dump. So the cap fires on the
# ``IS_BYTE + BYTE_INDEX_2`` register-dump row and clamps its emitted byte -> 0.
#
# RELATION to ``ax_hibyte_clear_allstep`` (C4_AX_HIBYTE_CLEAR, also default-OFF):
# that op zeros BOTH the byte-2 (BYTE_INDEX_1) AND byte-3 (BYTE_INDEX_2) dump
# rows in one bundle. This op is the BYTE-3-ONLY kill-switch the survey-R8 loop
# lane asked for: same value-invariant signature + marker/overflow blockers, but
# scoped to the SINGLE byte-3 predictor row so the loop byte-3 leak can be capped
# independently (and audited / reverted) without touching the byte-2 emission.
# Appended AFTER ``ax_hibyte_clear_allstep`` (and thus after
# ``tail_bit32_result_correction``) so it is a LAST OUTPUT writer on the byte-3
# row before the LM head. SAFE by construction: register byte-3 is 0x00 for
# EVERY register across the 1096 corpus (PC/SP/BP/AX all < 0x1000000 with a 0x00
# top byte; a genuine >= 0x1000000 value would carry AX_CARRY_OVERFLOW, whose
# -1000 blocker kills the clamp there). DEFAULT-OFF -> registered ONLY when the
# flag is on, so the flag-OFF build bakes NO units and is byte-identical to the
# golden state_dict hash.
def _loop_ax_byte3_cap_enabled() -> bool:
    """``C4_LOOP_AX_BYTE3_CAP`` flag predicate (DEFAULT-OFF).

    Flag-OFF: the op is NOT registered (see all_core_ops) so ZERO units bake ->
    byte-identical to golden ``91f55411``. Opt in with
    ``C4_LOOP_AX_BYTE3_CAP=1`` to clamp AX byte-3 -> 0 on the final-dump
    (BYTE_INDEX_2) predictor row, killing the loop_mul/countdown/pow2 stale
    byte-3 (0x82) return-dump leak.
    """
    return _os_stack0.environ.get("C4_LOOP_AX_BYTE3_CAP", "0") != "0"


# The two high-byte register-dump PREDICTOR rows (the residual at position ``p``
# predicts token ``p+1``): the AX byte-2 token (``MARK_AX+3``) is predicted by
# the byte-1 register row (carries ``BYTE_INDEX_1``, probed 0.97), and the AX
# byte-3 token (``MARK_AX+4``) is predicted by the byte-2 register row (carries
# ``BYTE_INDEX_2``, probed 0.97). One AND rule per high byte.
_LOOP_AX_BYTE3_CAP_HIDDEN_DIM = 2  # byte-2 (BYTE_INDEX_1) + byte-3 (BYTE_INDEX_2)

# STRONG-clamp write magnitude. The leak drives the emitted high byte via the
# canonical OUTPUT nibbles: spec_k=0 attribution (probe_loop_step4_byte2.py,
# loop_pow2_2 step-4 byte-2) shows ``OUTPUT_LO+15`` and ``OUTPUT_HI+15`` both at
# +12.5 (== 0xF/0xF == 0xFF), OUT-VOTING the clean +0 nibbles (~4-5). The prior
# hibyte clamp (WW=0.16, ~+10 delta, +0 nibbles ONLY) could NOT beat the +12.5
# competitor and did NOT kill the +15 nibbles, so byte-2 stayed 0xFF. This cap
# both KILLS every non-zero nibble (esp. +15) and boosts the +0 nibbles hard, so
# the 0x00 byte wins the argmax unconditionally on a fire.
_LOOP_CAP_WW_KILL = 0.5   # ~+/-30 OUTPUT delta on a saturated fire (kills +k>0)
_LOOP_CAP_WW_ZERO = 0.5   # boost the 0x00 nibbles


def _loop_ax_byte3_cap_rules() -> tuple[FFNRule, ...]:
    """2 AND rules clamping the AX high-byte (byte-2/byte-3) dump emission -> 0.

    Fires on the byte-2 predictor row (``IS_BYTE`` + ``BYTE_INDEX_1``, predicts
    ``MARK_AX+3``) and the byte-3 predictor row (``IS_BYTE`` + ``BYTE_INDEX_2``,
    predicts ``MARK_AX+4``), with the SAME marker + ``AX_CARRY_OVERFLOW``
    blockers as ``ax_hibyte_clear_allstep``. Firing sum = IS_BYTE(1.0) +
    BYTE_INDEX_x*2(~1.94) = ~2.94 > threshold 2.5; other byte rows (BYTE_INDEX_x
    ~0 -> sum ~1.0) and marker rows (-1000 blockers) stay dark.

    STRONG clamp: writes a large NEGATIVE to every non-zero OUTPUT nibble
    (``OUTPUT_LO+k`` / ``OUTPUT_HI+k`` for k=1..15 — this KILLS the leaked
    ``OUTPUT_LO+15`` / ``OUTPUT_HI+15`` == 0xF the loop leak sets) and a large
    POSITIVE to ``OUTPUT_LO+0`` / ``OUTPUT_HI+0`` so the emitted high byte is
    0x00. ``AX_CARRY_OVERFLOW`` (-1000) preserves a genuine >= 0x1000000 byte.
    """
    IS_BYTE_W = 1.0
    BYTE_INDEX_W = 2.0
    THRESHOLD = 2.5
    BLOCKER_W = 1_000.0
    blockers = (
        ("MARK_AX", -BLOCKER_W),
        ("MARK_PC", -BLOCKER_W),
        ("MARK_SP", -BLOCKER_W),
        ("MARK_BP", -BLOCKER_W),
        ("MARK_STACK0", -BLOCKER_W),
        ("MARK_MEM", -BLOCKER_W),
        ("MARK_SE", -BLOCKER_W),
        ("AX_CARRY_OVERFLOW", -BLOCKER_W),
    )
    # Kill every non-zero nibble in BOTH OUTPUT bands, then boost the 0x00 slot.
    writes = tuple(
        (f"OUTPUT_LO+{k}", -_LOOP_CAP_WW_KILL) for k in range(1, 16)
    ) + tuple(
        (f"OUTPUT_HI+{k}", -_LOOP_CAP_WW_KILL) for k in range(1, 16)
    ) + (
        ("OUTPUT_LO+0", _LOOP_CAP_WW_ZERO),
        ("OUTPUT_HI+0", _LOOP_CAP_WW_ZERO),
    )
    rules: list[FFNRule] = []
    for byte_label, bindex in (("byte2", "BYTE_INDEX_1"), ("byte3", "BYTE_INDEX_2")):
        rules.append(multi_way_and_rule(
            name=f"loop_ax_byte3_cap_{byte_label}",
            conditions=(
                ("IS_BYTE", IS_BYTE_W),
                (bindex, BYTE_INDEX_W),
            ) + blockers,
            threshold=THRESHOLD,
            writes=writes,
        ))
    return tuple(rules)


def make_loop_ax_byte3_cap_op() -> Operation:
    """Append the loop AX high-byte (byte-2/byte-3) dump clamp FFN after L25 tail.

    Fires on the two high-byte register-dump predictor rows (``IS_BYTE`` +
    ``BYTE_INDEX_1`` = byte-2 predictor, and ``IS_BYTE`` + ``BYTE_INDEX_2`` =
    byte-3 predictor) and STRONGLY clamps their OUTPUT emission -> 0x00, killing
    the loop_mul/countdown/pow2 stale high-byte leak (the step-4 byte-2 0xFF /
    the byte-3 stale-PC 0x82 return-dump leak). Registered ONLY when
    ``C4_LOOP_AX_BYTE3_CAP`` is on (see all_core_ops), so flag-OFF is
    byte-identical to golden; the op assumes it is only constructed under the
    flag.
    """
    rules = _loop_ax_byte3_cap_rules()

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
        assert len(rules) == _LOOP_AX_BYTE3_CAP_HIDDEN_DIM, (
            f"loop_ax_byte3_cap rule-count drift: produced {len(rules)}, "
            f"expected {_LOOP_AX_BYTE3_CAP_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            _off = int(_nm.split("+", 1)[1]) if "+" in _nm else 0
            dim_map[_nm] = int(dim_positions[_base]) + _off
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="loop_ax_byte3_cap",
        reads={
            "IS_BYTE", "BYTE_INDEX_1", "BYTE_INDEX_2",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "AX_CARRY_OVERFLOW",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        # After ax_hibyte_clear_allstep (thus after tail_bit32_result_correction)
        # so it is the last OUTPUT writer on the byte-3 dump row before the LM
        # head.
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ===========================================================================
# AX byte-1 DUMP -> OUTPUT decode (UNCONDITIONAL, OUTPUT-canonical)
# ===========================================================================
#
# ARCHITECTURAL REFACTOR increment 1 of "consolidate AX byte emission onto
# OUTPUT, then delete the H-band patchwork". The LM head reads EVERY byte token
# ``b`` from the CANONICAL nibble one-hots ``head.weight[b, OUTPUT_LO+(b&0xF)]``
# / ``head.weight[b, OUTPUT_HI+(b>>4)]`` (see model_ops ``_head_bake_rules``).
# The H1/H2/H3 ``*_DUMP_OUT`` bands that ALSO drive the AX byte-1 emission are
# EXTRA LM-head columns stacked on top by the ``C4_AX_BYTE1_DUMP`` machinery
# (``make_ax_byte1_dump_head_bake_op`` bakes ``head.weight[v, H1_DUMP_OUT+(v+2)]
# = 5.0``); the carry FFN ``_ax_byte1_dump_repopulate_rules`` fills
# ``H1_DUMP_OUT`` on carried (non-AX-writing) steps.
#
# This op DECODES the carried byte-1 one-hot back out of ``H1_DUMP_OUT`` into
# the canonical ``OUTPUT_LO/HI`` nibbles so the SAME byte-1 emits from OUTPUT
# that currently emits from the H-band. ``H1_DUMP_OUT`` is 7 wide (v in 0..4, a
# one-hot at ``H1_DUMP_OUT+(v+2)`` — see model_ops ``_ax_byte1_dump_band_for_
# value``). For each carried value v the op fires an AND on the byte-1-row
# signature AND the lit ``H1_DUMP_OUT+(v+2)`` slot and writes ``OUTPUT_LO+v``
# (v<=4 so the low nibble IS v) strongly + ``OUTPUT_HI+0`` (byte-1 high nibble
# is 0 for v<=4), AND KILLS the leaked byte-0 default ``OUTPUT_LO+0`` so the
# emitted byte is v, not 0. The gate mirrors the repopulate op's byte-1-row
# signature (``ADDR_B1_HI+8`` AX-register scope + ``ADDR_B0_LO+5`` byte-1 row +
# ``IS_BYTE``) so it fires ONLY where the H1 dump legitimately carries a byte-1.
#
# On a FRESH-AX step ``H1_DUMP_OUT`` is all-zero (the dump only fills it on
# carried steps) -> every AND's ``H1_DUMP_OUT+(v+2)`` gate is 0 -> the units are
# DARK -> the fresh byte-1 emission (from the normal H1 path) is byte-identical.
# On PC/SP/BP/STACK0/MEM/marker rows the marker blockers + AX_CARRY_OVERFLOW
# kill (mirrored from the dump) hold the op dark. UNCONDITIONAL: this decode is
# the OUTPUT-canonical byte-1 emission path (the H*_DUMP_OUT LM-head columns are
# dropped in ``make_ax_byte1_dump_head_bake_op``).


# The carried byte-1 one-hot lives at ``H<k>_DUMP_OUT+off`` per the LM-head
# H-band map (model_ops ``_ax_byte1_dump_band_for_value``): v in 0..4 ->
# H1_DUMP_OUT+(v+2); v in 5..11 -> H2_DUMP_OUT+(v-5); v in 12..15 ->
# H3_DUMP_OUT+(v-12). Increment 3 decodes the FULL byte-1 value range 0..15
# (was 0..4/H1-only in increment 1) so the H2/H3_DUMP_OUT LM-head columns can be
# dropped and byte-1 emits from OUTPUT alone across its whole corpus range (high
# byte 0..7 -> the H2 band). One AND per carried value.
_B1_TO_OUTPUT_HIDDEN_DIM = 16  # v in 0..15 (H1: 0..4, H2: 5..11, H3: 12..15)


def _b1_to_output_band_for_value(v: int) -> tuple[str, int]:
    """``(H<k>_DUMP_OUT band, one-hot offset)`` for carried byte-1 value ``v``.

    Mirrors model_ops ``_ax_byte1_dump_band_for_value`` EXACTLY (the LM-head
    H-band positional map the repopulate FFN fills): v 0..4 -> H1_DUMP_OUT+(v+2),
    v 5..11 -> H2_DUMP_OUT+(v-5), v 12..15 -> H3_DUMP_OUT+(v-12). The decode
    reads whichever DUMP slot the dump lit for that value.
    """
    if v <= 4:
        return "H1_DUMP_OUT", v + 2
    if v <= 11:
        return "H2_DUMP_OUT", v - 5
    return "H3_DUMP_OUT", v - 12


def _b1_to_output_rules() -> tuple[FFNRule, ...]:
    """16 AND rules decoding ``H<k>_DUMP_OUT+off`` -> ``OUTPUT`` (v in 0..15).

    Each unit fires on the byte-1-predictor-row signature (``IS_BYTE +
    ADDR_B1_HI+8 AX-register + ADDR_B0_LO+5 byte-1 row``) AND the lit carried
    one-hot slot ``H<k>_DUMP_OUT+off`` for value ``v`` (weighted so it is a HARD
    requirement: on a carried row exactly ONE slot is lit; on a fresh/non-AX row
    ALL slots are 0 so the AND is dark and the byte-1 emission stays
    byte-identical to the normal H-band path). On a fire it writes the canonical
    byte-1 nibbles ``OUTPUT_LO+(v & 0xF)`` + ``OUTPUT_HI+(v >> 4)`` strongly and
    KILLS the leaked byte-0 default ``OUTPUT_LO+0`` so the byte emits v, not 0.
    (byte-1 value v in 0..15 -> low nibble v, high nibble 0.) Increment 3
    extends increment 1's H1-only (v 0..4) decode to the whole H1/H2/H3 dump
    range so the H2/H3_DUMP_OUT LM-head columns can be dropped. Marker +
    AX_CARRY_OVERFLOW blockers mirror the dump gate.
    """
    # Condition weights. The byte-1-row signature (IS_BYTE ~1.0 + ADDR_B1_HI+8
    # ~4.02 * 1.0 + ADDR_B0_LO+5 ~0.97 * 2.0) contributes ~6.9 on the AX byte-1
    # row; the lit H1_DUMP_OUT slot is the DISCRIMINATOR (large weight so it is a
    # hard requirement — an unlit slot cannot clear threshold). Threshold sits
    # ABOVE the signature-only sum so a carried row with the WRONG slot lit (or a
    # fresh row with NO slot lit) stays dark.
    IS_BYTE_W = 1.0
    AX_REG_W = 1.0          # ADDR_B1_HI+8 AX-register scope
    SIG_W = 2.0            # ADDR_B0_LO+5 byte-1 row signature
    DUMP_SLOT_W = 20.0     # H<k>_DUMP_OUT+off HARD requirement (the value select)
    THRESHOLD = 12.0       # above signature-only (~6.9); needs the dump slot lit
    BLOCKER_W = 1_000.0
    OVERFLOW_KILL_W = 1_000.0
    blockers = (
        ("MARK_AX", -BLOCKER_W),
        ("MARK_PC", -BLOCKER_W),
        ("MARK_SP", -BLOCKER_W),
        ("MARK_BP", -BLOCKER_W),
        ("MARK_STACK0", -BLOCKER_W),
        ("MARK_MEM", -BLOCKER_W),
        ("MARK_SE", -BLOCKER_W),
        ("AX_CARRY_OVERFLOW", -OVERFLOW_KILL_W),
    )
    # Write magnitudes mirror ax_hibyte_clear_allstep (WW=0.16 -> ~+/-10 OUTPUT
    # delta on a saturated fire): push OUTPUT_LO+(v&0xF) / OUTPUT_HI+(v>>4) UP and
    # kill the leaked byte-0 default OUTPUT_LO+0 so the LM head emits v (byte-1
    # value v in 0..15 -> low nibble v, high nibble 0).
    WW = 0.16
    rules: list[FFNRule] = []
    for v in range(_B1_TO_OUTPUT_HIDDEN_DIM):
        band, slot = _b1_to_output_band_for_value(v)  # H<k>_DUMP_OUT+slot
        lo_nib = v & 0xF
        hi_nib = v >> 4
        writes = [
            (f"OUTPUT_LO+{lo_nib}", WW),
            (f"OUTPUT_HI+{hi_nib}", WW),
        ]
        # Kill the leaked byte-0 low default so the byte emits v, not 0. When the
        # target low nibble IS OUTPUT_LO+0 (v==0) skip the negative to avoid
        # self-cancel (we are writing OUTPUT_LO+0 UP for that value).
        if lo_nib != 0:
            writes.append(("OUTPUT_LO+0", -WW))
        rules.append(multi_way_and_rule(
            name=f"b1_to_output_v{v}",
            conditions=(
                ("IS_BYTE", IS_BYTE_W),
                ("ADDR_B1_HI+8", AX_REG_W),
                ("ADDR_B0_LO+5", SIG_W),
                (f"{band}+{slot}", DUMP_SLOT_W),
            ) + blockers,
            threshold=THRESHOLD,
            writes=tuple(writes),
        ))
    return tuple(rules)


def make_b1_to_output_op() -> Operation:
    """Append the AX byte-1 DUMP->OUTPUT decode FFN after the L25 tail block.

    Standalone ``PureFFN`` post_op on the L25 tail block, appended AFTER
    ``tail_bit32_result_correction``, the ENT/LI caps, ``ax_hibyte_clear_allstep``
    AND ``ax_byte1_dump_repopulate`` (which FILLS ``H1_DUMP_OUT``) so it reads
    the freshly-filled dump band and is among the LAST writers of OUTPUT on the
    AX byte-1 dump row before the LM head. UNCONDITIONAL: this is the
    OUTPUT-canonical byte-1 emission path. ALL gate + target dims resolve from
    the declarative ``dim_positions`` layout (no legacy-registry split).
    """
    rules = _b1_to_output_rules()

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
        assert len(rules) == _B1_TO_OUTPUT_HIDDEN_DIM, (
            f"b1_to_output rule-count drift: produced {len(rules)}, "
            f"expected {_B1_TO_OUTPUT_HIDDEN_DIM}"
        )
        ffn = PureFFN(d_model, len(rules))
        dim_map = {}
        for _nm in Primitives.ffn_rule_dim_names(rules):
            _base = _nm.split("+", 1)[0]
            _off = int(_nm.split("+", 1)[1]) if "+" in _nm else 0
            dim_map[_nm] = int(dim_positions[_base]) + _off
        Primitives.lower_ffn_rules(ffn, rules, dim_map, S=S)
        block.post_ops.append(ffn)

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(rules)

    return Operation(
        name="b1_to_output",
        reads={
            "IS_BYTE", "ADDR_B1_HI", "ADDR_B0_LO",
            "H1_DUMP_OUT", "H2_DUMP_OUT", "H3_DUMP_OUT",
            "MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "AX_CARRY_OVERFLOW",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        # After ax_hibyte_clear_allstep (which is after ax_byte1_dump_repopulate
        # -> H1/H2/H3_DUMP_OUT are filled) so this is among the last OUTPUT
        # writers on the AX byte-1 dump row before the LM head.
        target_op_name="l10_post_ops_combined",
        requires={"after": "ax_hibyte_clear_allstep"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="AX_HIGH_BYTE_DUMP_ROOT_IS_H1_ONEHOT_2026_06_13.md",
    )


# ===========================================================================
# STACK0 byte-0 cross-step carry HEAD (the sole survivor of the deleted Root 2
# register-dump machinery). MIRRORS the AX byte-1 dump carry head above.
# ===========================================================================
#
# The full Root 2 STACK0 byte-0 REGISTER-DUMP machinery — the dump-repopulate
# FFN, the four bounded flag precursors (carried / sharp / prev_dom / not_cmp),
# the POP-discriminator latch, the LM-head DUMP columns, and the seven
# ``STACK0_B0_{DUMP_H1,DUMP_H3,CARRIED,SHARP,PREV_DOM,NOT_CMP,POPPED}`` bands —
# was DELETED (2026-07) as PROVABLY-DEAD weight. It only ever fired on a CARRIED
# STACK0-marker row (a step that re-emits a stack-top byte the LM head decoded
# on a prior step); in the default 30-token frame STACK0 is NEVER emitted, so no
# such row exists and the dump gate never fires. Verdict-neutral geometry cut
# (verified field-identical). See docs/STACK0_B0_DUMP_DEAD_MACHINERY_DELETION.
#
# What survives, and WHY: the L9 carry HEAD (below) + the two ``_PREV`` bands.
# The head copies the PREVIOUS step's STACK0-marker ``H1`` / ``H3`` byte-0
# one-hot into ``STACK0_B0_H1_PREV`` / ``STACK0_B0_H3_PREV`` UNCONDITIONALLY
# (via the ``H1.*.-1`` / ``H3.*.-1`` SSA cross-step reads -> no same-step
# back-edge; it WRITES distinct bands read by nobody upstream). Those two bands
# are read LIVE by the campaign MUL multi-byte L19 boost
# (``efficient_alu_neural._MulCombineStage``): the SUM of the two bands is the
# literal-vs-var_mul frame discriminator (``var_frame_carry`` ~7379 in a
# var_mul ENT frame, <=667 for a single-step literal mul), so removing the head
# would zero the discriminator and mis-fire the boost on var_mul. Hence the head
# + the two PREV bands are KEPT even though the dump they once fed is gone.
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


# ===========================================================================
# ISA-semantics DSL: the STACK0 byte-0 cross-step CARRY HEAD is generated by
# ``cross_step_carry(_STACK0_B0_CARRY_SPEC)`` (explicit-head + mixed position
# source). The spec's dump / precursor fields are now inert (the dump FFN +
# flag precursors that once consumed them were deleted as dead machinery), so
# the bundle builds ONLY the carry head.
#
# The carry head's Q/K/V/O is declared EXPLICITLY (it is NOT the BP per-byte
# positional match): a sharp MARK_STACK0 signature on slot 0, a CONST-driven
# one-hot-presence K-pref on slot 1 (over the prev H1/H3 cells), a MARK_AX V=0
# sink on slot 3, and two cross-step value-copy blocks (prev H3 -> H3_PREV at V
# slots 1..7, prev H1 -> H1_PREV at V slots 10..16). ``position_source="mixed"``:
# the H1/H3 + MARK/CONST taps resolve from the legacy dynamic registry, the new
# ``STACK0_B0_*_PREV`` bands from the declarative layout. The two PREV bands are
# registered by the module-scope ``register_residual_band`` calls above
# (LOAD-BEARING order), so the spec sets ``register_band=False``.

# Head tuning (production defaults — the hand-built call passed none).
_STACK0_B0_HEAD_SIG_W = 60.0   # SHARP MARK_STACK0 signature (slot 0)
_STACK0_B0_HEAD_L = 15.0       # CONST/MARK_STACK0 driver magnitude
_STACK0_B0_HEAD_SINK_W = 8.0   # MARK_AX V=0 sink (slot 3 K)
_STACK0_B0_HEAD_PRES_W = 6.0   # one-hot-presence K preference (slot 1)
_STACK0_B0_HEAD_W = 7          # H1/H3 band width (cells)
_STACK0_B0_HEAD_LO_BASE = 1    # V slots for the low-nibble (H3) copy
_STACK0_B0_HEAD_HI_BASE = 10   # V slots for the high-nibble (H1) copy


# The STACK0 byte-0 cross-step carry spec. The two PREV bands are registered by
# the module-scope ``register_residual_band`` calls (load-bearing order), so
# ``register_band=False``. The carry head is explicit (NOT the BP per-byte
# match); the dump / precursor fields are inert (dead machinery deleted).
# ``position_source`` is ``"mixed"``: the H1/H3 + MARK/CONST taps from the
# dynamic registry, the new
# ``STACK0_B0_*`` bands from the layout.
_STACK0_B0_REGISTRY_DIMS = (
    "H1", "H3", "CONST", "MARK_STACK0", "MARK_AX",
)
_STACK0_B0_CARRY_SPEC = CrossStepCarrySpec(
    name="stack0_byte0_dump",
    band_name="STACK0_B0_H1_PREV",   # nominal (register_band=False -> unused)
    band_width=_STACK0_B0_HEAD_W,
    register_band=False,
    band_flag=None,
    carry_head_alibi_slope=0.5,
    carry_byte_count=1,              # explicit head: not per-byte (>=1 sentinel)
    position_source="mixed",
    registry_dims=_STACK0_B0_REGISTRY_DIMS,
    # Explicit carry head (resolved by ``position_source``):
    head_q=(
        HeadWrite(0, "MARK_STACK0", _STACK0_B0_HEAD_SIG_W),  # slot 0 signature
        HeadWrite(1, "CONST", _STACK0_B0_HEAD_L),            # slot 1 pres driver
        HeadWrite(3, "MARK_STACK0", _STACK0_B0_HEAD_L),      # slot 3 sink driver
    ),
    head_k=(
        HeadWrite(0, "MARK_STACK0", _STACK0_B0_HEAD_SIG_W),
        HeadWrite(3, "MARK_AX", _STACK0_B0_HEAD_SINK_W),
        # slot-1 one-hot-presence preference over the prev H1/H3 cells: the SLOT
        # stays fixed at 1 (slot_stride=0) while the DIM advances over the 7
        # H1/H3 cells (all 14 K writes land on the single slot-1 K row).
        HeadWrite(1, "H1", _STACK0_B0_HEAD_PRES_W, count=_STACK0_B0_HEAD_W,
                  slot_stride=0),
        HeadWrite(1, "H3", _STACK0_B0_HEAD_PRES_W, count=_STACK0_B0_HEAD_W,
                  slot_stride=0),
    ),
    head_v=(
        # V slots 1..7 <- prev H3 (lo nibble); 10..16 <- prev H1 (hi nibble).
        HeadWrite(_STACK0_B0_HEAD_LO_BASE, "H3", 1.0, count=_STACK0_B0_HEAD_W),
        HeadWrite(_STACK0_B0_HEAD_HI_BASE, "H1", 1.0, count=_STACK0_B0_HEAD_W),
    ),
    head_o=(
        # O writes STACK0_B0_H3_PREV from V slots 1..7; H1_PREV from 10..16.
        HeadWrite(_STACK0_B0_HEAD_LO_BASE, "STACK0_B0_H3_PREV", 1.0,
                  count=_STACK0_B0_HEAD_W),
        HeadWrite(_STACK0_B0_HEAD_HI_BASE, "STACK0_B0_H1_PREV", 1.0,
                  count=_STACK0_B0_HEAD_W),
    ),
    head_reads={"MARK_STACK0", "MARK_AX", "CONST", "H1", "H3",
                "H1.*.-1", "H3.*.-1"},
    head_writes={"STACK0_B0_H1_PREV", "STACK0_B0_H3_PREV"},
    # DUMP + precursor fields REMOVED (2026-07): the STACK0-b0 dump-repopulate
    # FFN, the four bounded flag precursors, and their target bands were deleted
    # as provably-dead in the 30-token frame (STACK0 is never emitted -> no
    # carried STACK0-marker row -> the dump gate never fires). ONLY the carry
    # head survives (it populates STACK0_B0_H1_PREV / STACK0_B0_H3_PREV, which
    # are read LIVE by the campaign MUL multi-byte L19 boost discriminator). No
    # dump op consumes the bundle's dump builder anymore, so ``dump_blocks`` is
    # an inert empty-tuple builder — its ONLY remaining role is to mark the spec
    # ``explicit_dump`` (so the even-width LO/HI check, which applies only to the
    # BP layout-mode dump, is skipped for this odd-width 7-cell explicit head).
    dump_blocks=lambda: (),
    precursors=(),
)
_STACK0_B0_CARRY_BUNDLE = cross_step_carry(_STACK0_B0_CARRY_SPEC)


def _stack0_byte0_dump_carry_head_spec(
    dim_positions: dict,
    head_idx: int,
    *,
    L: float = 15.0,
    sink_w: float = 8.0,
    alibi_slope: float = 0.5,
) -> DeclarativeAttentionHeadSpec:
    """Carry head: copy the prev step's STACK0-marker H1/H3 one-hot forward.

    Mirrors ``_ax_byte1_dump_carry_head_spec``. The head attends the
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
    # ISA-semantics DSL migration: re-expressed via
    # ``cross_step_carry(_STACK0_B0_CARRY_SPEC)`` (explicit-head + mixed
    # position-source mode). The bundle's ``carry_head_spec_builder`` reproduces
    # the legacy hand-built Q/K/V/O writes BYTE-IDENTICALLY (verified against the
    # HEAD golden state_dict hash). The tuning args (``L`` / ``sink_w`` /
    # ``alibi_slope``, kept for probes/tests) default to the spec's frozen
    # values; when ALL are at their defaults the module-level bundle is used
    # directly, otherwise a per-call override spec rebuilds the explicit head.
    if (L == _STACK0_B0_HEAD_L
            and sink_w == _STACK0_B0_HEAD_SINK_W
            and alibi_slope == _STACK0_B0_CARRY_SPEC.carry_head_alibi_slope):
        return _STACK0_B0_CARRY_BUNDLE.carry_head_spec_builder(
            dim_positions, head_idx,
        )
    override = replace(
        _STACK0_B0_CARRY_SPEC,
        carry_head_alibi_slope=alibi_slope,
        head_q=(
            HeadWrite(0, "MARK_STACK0", _STACK0_B0_HEAD_SIG_W),
            HeadWrite(1, "CONST", L),
            HeadWrite(3, "MARK_STACK0", L),
        ),
        head_k=(
            HeadWrite(0, "MARK_STACK0", _STACK0_B0_HEAD_SIG_W),
            HeadWrite(3, "MARK_AX", sink_w),
            HeadWrite(1, "H1", _STACK0_B0_HEAD_PRES_W, count=_STACK0_B0_HEAD_W,
                      slot_stride=0),
            HeadWrite(1, "H3", _STACK0_B0_HEAD_PRES_W, count=_STACK0_B0_HEAD_W,
                      slot_stride=0),
        ),
    )
    return cross_step_carry(override).carry_head_spec_builder(
        dim_positions, head_idx,
    )


def make_stack0_byte0_dump_carry_op(enable: bool = True) -> Operation:
    """L9 attn head 5: copy the prev step's STACK0 byte-0 H1/H3 one-hot forward.

    The cross-step carry HEAD. Copies the PREVIOUS VM step's STACK0-marker
    ``H1`` / ``H3`` one-hot into the dedicated ``STACK0_B0_H1_PREV`` /
    ``STACK0_B0_H3_PREV`` bands UNCONDITIONALLY (via the ``H1.*.-1`` /
    ``H3.*.-1`` SSA cross-step reads). Hosted on L9 (physical block 10), head 5
    (free in the pre-widen 8-head band: L9 declares heads 0..4).

    This head is the SOLE survivor of the former STACK0 byte-0 register-dump
    machinery (Root 2). The dump-repopulate FFN + flag precursors + POP latch +
    LM-head DUMP columns were DELETED (2026-07) as provably-dead in the 30-token
    frame (STACK0 is never emitted -> no carried STACK0-marker row -> the dump
    gate never fired). The carry head + the two ``_PREV`` bands are KEPT because
    they are read LIVE by the campaign MUL multi-byte L19 boost
    (``efficient_alu_neural._MulCombineStage``): the SUM of ``STACK0_B0_H1_PREV``
    + ``STACK0_B0_H3_PREV`` is the literal-vs-var_mul frame discriminator
    (~7379 in a var_mul ENT frame, <=667 for a single-step literal mul).
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


# ===========================================================================
# ENT saved-BP store cross-step carry (BP_SAVE_PREV — the func/nested/rec/var
# LI-from-frame 37-token desync). MIRRORS the AX byte-1 / STACK0 byte-0 carries
# above, but carries a 4-BYTE VALUE (old_BP) via CLEAN_EMBED -> OUTPUT instead
# of a single byte-1 one-hot via the H-bands.
# ===========================================================================
#
# ROOT (spec_k=0, tools/_probe_bp_carry*.py on id550/575): after a function ENT
# the callee's saved-BP store (the ENT step's MEM section value bytes) emits
# 0xFF garbage [.,0xff,.,0xff] instead of old_BP because the L14 value heads
# (4-7) content-address the WRONG source position at the ENT step (the same-step
# old-BP lookup via slot 44 fails). The garbage VALUE bytes (0xFF) then drive a
# +2-token 0xFF over-emit on the NEXT step (the post-ENT 37-token desync), which
# poisons the in-frame MEM section so a later frame-local ``LI`` returns 0 ->
# func/nested/rec/var diverge at the first ``LI``.
#
# FIX (the PROVEN cross-step carry-band pattern, NOT a same-step attention fix):
# the BP register holds the CLEAN old_BP at EVERY step (its byte TOKENS are
# emitted correctly, and the token EMBEDDING ``CLEAN_EMBED_LO/HI`` at the
# prev-step BP byte rows decodes to the exact old_BP nibbles even though the
# OUTPUT residual there is later nuked to 0xFF). So:
#   * The carry head (below, L13 block-16 head 8) attends from each ENT-step MEM
#     val-byte-k PREDICTOR row BACK to the prev step's BP byte-k row (matched by
#     ``MEM_VAL_B{k}`` Q <-> ``BYTE_INDEX_{k}`` K, with an ``OP_JSR`` K-preference
#     that selects the prev-step prologue BP rows + positive-ALiBi recency), and
#     V-copies ``CLEAN_EMBED_LO/HI`` there into the dedicated ``BP_SAVE_PREV``
#     band. Each byte k lands on a DISTINCT val-predictor row, so one 32-wide
#     band (16 LO + 16 HI) carries all four bytes without collision.
#   * The dump FFN (``bp_save_dump_repopulate``) re-supplies ``BP_SAVE_PREV`` ->
#     ``OUTPUT_LO/HI`` at the ENT-store val-predictor rows ONLY (gated on the
#     high ``OP_ENT`` broadcast ~10.7 + the ``MEM_VAL_B*`` markers, so SI/SC/PSH/
#     JSR stores — which carry ``OP_JSR`` or no ``OP_ENT`` — stay byte-identical),
#     overwriting the 0xFF garbage AFTER the tail corruptor runs. The LM head
#     already emits byte tokens from OUTPUT_LO/HI, so NO new head-bake columns
#     are needed; the OUTPUT re-supply is flag-gated (``C4_BP_SAVE_DUMP``,
#     default ON) -> flag-off is byte-identical (the dump writes nothing).
_BP_SAVE_PREV_CARRY_HEAD_IDX = 8  # heads 0..7 used on L13; 8/9 free pre-this-band
_BP_SAVE_PREV_CARRY_HEAD_LAYOUT = (
    ("bp_save_prev_carry.head_8", _BP_SAVE_PREV_CARRY_HEAD_IDX),
)


def _allocate_bp_save_prev_carry_heads() -> AttentionHeadAllocator:
    """Per-bake head allocator for the BP_SAVE_PREV carry head (L13 host).

    Head 8 is a WIDEN-padding slot on the physical L13 block (block 16): the
    AX byte-1 carry (head 7) + mul width=2 relay (head 6) consume the 8-head
    logical budget, and the residual-band auto-widen pads block 16 to 10
    physical heads (heads 8/9 all-zero). We pin head 8 (the first free padding
    slot), so ``layer_max_heads`` must be raised past the default 8.
    """
    allocator = AttentionHeadAllocator(
        strategy="dynamic_first_fit", layer_max_heads=10,
    )
    for (op_name, head_idx) in _BP_SAVE_PREV_CARRY_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=13, pin=head_idx)
    return allocator


def _bp_save_prev_carry_head_spec(
    dim_positions: dict,
    head_idx: int,
    *,
    idx_w: float = 40.0,
    jsr_w: float = 6.0,
    ent_block_w: float = 8.0,
    alibi_slope: float = 0.5,
) -> DeclarativeAttentionHeadSpec:
    """Carry head: copy the prev step's BP byte-k CLEAN_EMBED into BP_SAVE_PREV.

    Q row = the ENT-step MEM val-byte-k PREDICTOR row (``MEM_VAL_B{k}`` active).
    K row = the prev step's BP byte-k row (``BYTE_INDEX_{k}`` active +
    ``OP_JSR`` prologue broadcast). The per-byte ``MEM_VAL_B{k}`` <->
    ``BYTE_INDEX_{k}`` pairing (each on its OWN K-slot) makes val_k attend
    BP byte_k; the ``OP_JSR`` K-bias (its own slot) prefers the prev-step
    prologue BP rows over any other ``BYTE_INDEX``-carrying row, and the
    positive ALiBi slope prefers the NEAREST prev step (the immediate caller
    frame for recursion). V reads ``CLEAN_EMBED_LO/HI`` (the clean old_BP
    nibble one-hots) at the matched BP byte row; O writes ``BP_SAVE_PREV``
    (the dedicated band, read ONLY by the gated dump FFN -> no same-step
    back-edge, no cycle).
    """
    # ISA-semantics DSL migration: re-expressed via
    # ``cross_step_carry(_BP_SAVE_CARRY_SPEC)``. The bundle's
    # ``carry_head_spec_builder`` reproduces the legacy hand-built Q/K/V/O writes
    # BYTE-IDENTICALLY (verified against the HEAD golden state_dict hash). The
    # tuning args (``idx_w`` / ``jsr_w`` / ``ent_block_w`` / ``alibi_slope``,
    # kept for probes/tests) default to the spec's frozen values; when ALL are
    # at their defaults the module-level bundle is used directly, otherwise a
    # per-call override spec is built so probe sweeps still work.
    if (idx_w == _BP_SAVE_CARRY_SPEC.match_weight
            and jsr_w == _BP_SAVE_CARRY_SPEC.k_prefer[0][2]
            and ent_block_w == _BP_SAVE_CARRY_SPEC.k_reject[0][2]
            and alibi_slope == _BP_SAVE_CARRY_SPEC.carry_head_alibi_slope):
        return _BP_SAVE_CARRY_BUNDLE.carry_head_spec_builder(
            dim_positions, head_idx,
        )
    # Probe override: rebuild the spec's varying weights (band already
    # registered at import; ``cross_step_carry`` is idempotent on re-register).
    override = replace(
        _BP_SAVE_CARRY_SPEC,
        match_weight=idx_w,
        k_prefer=((4, "OP_JSR", jsr_w),),
        k_reject=(
            (5, "OP_ENT", ent_block_w),
            (6, "STACK0_BYTE0", ent_block_w),
            (6, "STACK0_BYTE1", ent_block_w),
            (6, "STACK0_BYTE2", ent_block_w),
            (6, "STACK0_BYTE3", ent_block_w),
        ),
        carry_head_alibi_slope=alibi_slope,
    )
    return cross_step_carry(override).carry_head_spec_builder(
        dim_positions, head_idx,
    )


def make_bp_save_prev_carry_op(enable: bool = True) -> Operation:
    """L13 attn head 8: carry the prev step's old_BP (CLEAN_EMBED) forward.

    The cross-step carry HEAD half of the ENT saved-BP store fix. Copies the
    PREVIOUS VM step's BP byte-k ``CLEAN_EMBED_LO/HI`` (the clean old_BP
    nibbles) into the dedicated ``BP_SAVE_PREV`` band, matched per-byte
    (``MEM_VAL_B{k}`` Q <-> ``BYTE_INDEX_{k}`` K + ``OP_JSR`` prev-prologue
    preference). The carried-vs-fresh gate + the re-supply into ``OUTPUT_LO/HI``
    live in the partner ``make_bp_save_dump_repopulate_op`` FFN. Hosted on L13
    (physical block 16), head 8 (free pre-this-band: L13 declares 0..7).
    """
    def bake(block, dim_positions, S):
        # Flag-gated (``C4_BP_SAVE_DUMP``, default ON): flag-off omits the
        # ``BP_SAVE_PREV`` band (and this head), so the head can only bake when
        # the band exists in ``dim_positions``. ``enable`` is the op-level
        # override (kept for tests/probes).
        if not enable or not _bp_save_dump_enabled():
            return
        attn = block.attn
        allocator = _allocate_bp_save_prev_carry_heads()
        attn._l13_bp_save_prev_carry_head_allocator = allocator
        head_idx = allocator.heads()[-1].head_idx
        HD = attn.W_q.shape[0] // attn.num_heads
        spec = _bp_save_prev_carry_head_spec(dim_positions, head_idx)
        Primitives.generate_attention_head(attn, spec, HD)

    def _ir(dim_positions, HD) -> CompilerIR:
        del HD
        ir = CompilerIR()
        if enable and _bp_save_dump_enabled():
            allocator = _allocate_bp_save_prev_carry_heads()
            head_idx = allocator.heads()[-1].head_idx
            spec = _bp_save_prev_carry_head_spec(dim_positions, head_idx)
            ir.layer(0).attention.append(
                spec, name="bp_save_prev_carry.head_8",
            )
        return ir

    # Flag-off: empty reads/writes so the op never references the OMITTED
    # ``BP_SAVE_PREV`` band (which would fail ``add_op`` dim validation); the
    # bake is a no-op in that case.
    _on = enable and _bp_save_dump_enabled()
    if _on:
        reads = {
            "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
            "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
            "STACK0_BYTE0", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
            "OP_JSR", "OP_ENT", "CONST", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
            "CLEAN_EMBED_LO.*.-1", "CLEAN_EMBED_HI.*.-1",
        }
        writes = {"BP_SAVE_PREV"}
    else:
        reads = set()
        writes = set()

    return Operation(
        name="bp_save_prev_carry",
        # Q@MEM_VAL_B{k} (ENT-step val-byte-k predictor) / K@BYTE_INDEX_{k}
        # (prev BP byte-k) + OP_JSR prev-prologue preference. V reads the prev
        # step's CLEAN_EMBED cross-step (``CLEAN_EMBED_LO/HI.*.-1`` -> no
        # same-step back-edge). O writes the dedicated ``BP_SAVE_PREV`` band
        # (read ONLY by the gated dump FFN -> no edge from same-step
        # CLEAN_EMBED/OUTPUT readers -> no cycle).
        reads=reads,
        writes=writes,
        audited_empty_produces=not _on,
        kind="block",
        # Bind to the L13 mem-addr anchor (physical block 16, head 8 free). The
        # L13 read point holds clean MEM_VAL_B / OP_ENT / OP_JSR signatures (the
        # 0xFF tail corruption is downstream at block 32+).
        target_op_name="_layer13_mem_addr_anchor",
        declarative_bake_fn=bake,
        compiler_ir_factory=_ir,
        migrated=True,
        declarative_authority="spec_generated",
        smoke_tests={"all"},
        spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
    )


# ---------------------------------------------------------------------------
# ENT saved-BP store dump FFN (the OP_ENT gate + OUTPUT re-supply)
# ---------------------------------------------------------------------------
#
# The FFN half of the ENT saved-BP carry. The carry head copies the prev step's
# old_BP nibbles into ``BP_SAVE_PREV`` (per byte, delivered to the val-byte-k
# predictor row). This FFN re-supplies them into ``OUTPUT_LO``/``OUTPUT_HI`` at
# the ENT-store val-predictor rows ONLY, where the LM head then emits the clean
# old_BP byte token. It runs on the L25 tail block AFTER the tail corruptor
# (which writes the 0xFF garbage), so it is the LAST writer of OUTPUT before the
# LM head.
#
# Gate (measured spec_k=0, tools/_probe_bp_carry4.py at block 32, the dump read
# point): the ENT-store val-predictor rows carry ``OP_ENT`` ~10.7 + a
# ``MEM_VAL_B{k}`` marker + ``MEM_STORE``; SI/SC/PSH stores carry ``OP_JSR`` or
# no ``OP_ENT`` (steps 2/3/6/7/8 in id550 have OP_ENT ~1.x or absent), and JSR
# pushes carry ``OP_JSR`` ~10.7. So a high-OP_ENT AND a MEM_VAL_B marker
# uniquely select the ENT saved-BP store value rows. The flag (``C4_BP_SAVE_DUMP``,
# default ON) gates whether OUTPUT is written: flag-off writes nothing (the
# carry head still fills the inert BP_SAVE_PREV band, read by nobody) ->
# byte-identical.
_BP_SAVE_DUMP_OP_ENT_THRESHOLD = 6.0  # OP_ENT ~10.7 on ENT store, ~1.x elsewhere
# OUTPUT re-supply write scale. The ENT-step val rows are overwritten with 0xFF
# SENTINEL-MAGNITUDE garbage (~5e3 to ~7e7) by the L25 tail corruptor BEFORE this
# dump runs; the residual is ADDITIVE, so the re-supply must net POSITIVE at the
# correct OUTPUT nibble slot to flip the argmax. write_scale tuned large so
# ``silu(S*(cond-thr)) * BP_SAVE_PREV * write_scale`` dominates the nuke. Probe
# override via ``C4_BP_SAVE_DUMP_WS``.
_BP_SAVE_DUMP_WS = float(
    _os_stack0.environ.get("C4_BP_SAVE_DUMP_WS", "200000.0")
)


def _bp_save_dump_repopulate_rules(emission_on: bool) -> tuple[FFNRule, ...]:
    """32 rules: ``OUTPUT_LO/HI[j] = BP_SAVE_PREV[j/16]`` on ENT-store val rows.

    For each val byte k (0..3), the predictor row carries ``MEM_VAL_B{k}``; the
    16 OUTPUT_LO + 16 OUTPUT_HI nibble cells are gate-copied from the carried
    ``BP_SAVE_PREV`` band (which holds byte k's CLEAN_EMBED nibbles, delivered to
    THIS row by the carry head). The gate AND is ``OP_ENT >= 6.0`` (the
    ENT-store discriminator) AND ``MEM_VAL_B{k}`` (the val-byte-k row) — so the
    rule fires ONLY on the ENT saved-BP store value rows, never on SI/SC/PSH/JSR
    stores. When ``emission_on`` is False the OUTPUT writes are dropped (the
    rules write the inert BP_SAVE_PREV band onto itself, a no-op) ->
    byte-identical to the pre-carry build.
    """
    # ISA-semantics DSL migration: re-expressed via
    # ``cross_step_carry(_BP_SAVE_CARRY_SPEC)``. The bundle's
    # ``dump_rules_builder`` produces the IDENTICAL per-cell ``multi_way_and_rule``
    # set — same names (``bp_save_dump_val{k}_{lo,hi}_{nib}``), conditions
    # (``OP_ENT``*1.0 + ``MEM_VAL_B{k}``*8.0 + the MARK_* blockers), threshold
    # (``dump_opent_floor`` 6.0 + marker 8.0 = 14.0), ``BP_SAVE_PREV+{j}`` gate,
    # and ``OUTPUT_{LO,HI}+{nib}`` writes at ``_BP_SAVE_DUMP_WS`` — verified
    # byte-identical against the HEAD golden state_dict hash. Returns () when
    # ``emission_on`` is False (flag-off: the band is omitted from the layout, so
    # the dump produces ZERO rules and the op is inert -> byte-identical
    # pre-carry build). The module-level constants
    # ``_BP_SAVE_DUMP_OP_ENT_THRESHOLD`` / ``_BP_SAVE_DUMP_WS`` are retained for
    # docs/probe parity; their values are folded into ``_BP_SAVE_CARRY_SPEC``.
    return _BP_SAVE_CARRY_BUNDLE.dump_rules_builder(emission_on)


# ---------------------------------------------------------------------------
# Root A: tighten the dump gate so the MEM_VAL_B{k} marker is MANDATORY
# ---------------------------------------------------------------------------
#
# THE BUG (CPU spec_k=0, var_mul id275 step-1 ENT frame, probe_root_a_*):
# At the ENT frame-establishing step BP = 0x0000fff0, so the BP register VALUE
# byte-1 row should decode 0xFF. It IS 0xFF through block 49 but block 50
# (the L25 tail, where this dump runs LAST) crushes it to 0x00. Runtime
# attribution (probe_root_a_attrib.py) pins the crush to THIS op's
# ``bp_save_dump_val{k}_{lo,hi}_0`` rules at +8.5e8/+1.2e9 — NOT to
# ``layer16_lev_routing`` (its ``l16_bp_frame_byte1_ff`` rule actually writes
# the CORRECT 0xFF, but at only -47.8, drowned 1e9-to-1).
#
# WHY IT MISFIRES: the dump gate is the ADDITIVE AND
# ``silu(S*(OP_ENT*1 + MEM_VAL_B{k}*8 - 14))``. The threshold 14 = floor 6 +
# marker 8 assumed OP_ENT ~10.7 on the legit MEM-store rows. But the ENT
# OP_ENT BROADCAST residue climbs to ~16-18 on the register VALUE byte rows
# (probe_root_a_step_scan.py: rows off 16-21 carry OP_ENT 16.0..17.85, with
# MEM_VAL_B{k} == 0 and NO MARK_* — the MARK blockers sit only on the MARKER
# rows, not the value-byte rows). So ``OP_ENT*1`` ALONE (16 > 14) clears the
# threshold with the marker absent, and the carried BP_SAVE_PREV band (the
# PREV step's old_BP = 0x00000000) writes 0x00 over the live BP=0xff byte.
#
# THE FIX (flag-gated, default-OFF -> byte-identical to current main): make
# the MEM_VAL_B{k} marker the DECISIVE term. The legit MEM-store rows carry
# OP_ENT in [6, 11] AND MEM_VAL_B{k} ~ 0.97; the spurious register rows carry
# OP_ENT in [12, 18] AND MEM_VAL_B{k} == 0; SI/SC/PSH non-ENT stores carry
# OP_ENT ~ 1.x AND a marker. With OP_ENT weight 1.0, marker weight 20.0,
# threshold 23.0:
#   * legit ENT-store (OP_ENT>=6, marker 0.97): 6 + 19.4 = 25.4  > 23 -> FIRE
#   * register broadcast (OP_ENT<=18, marker 0):       18 = 18   < 23 -> blocked
#   * SI/SC non-ENT store (OP_ENT~1.5, marker 0.97): 1.5 + 19.4 = 20.9 < 23 -> blocked
# i.e. the marker (19.4) now OUTWEIGHS the OP_ENT broadcast spread (18-6=12),
# so its ABSENCE alone drops the register rows below threshold regardless of
# how high the spurious OP_ENT broadcast spikes, while OP_ENT still keeps the
# SI/SC (low-OP_ENT) stores out. The MARK_* blockers + write_scale + the gate
# (BP_SAVE_PREV) are untouched; only the OP_ENT / MEM_VAL_B condition weights
# and the threshold are rewritten IN PLACE (rule count unchanged at 128).
def make_bp_save_dump_repopulate_op() -> Operation:
    """Append the ENT saved-BP dump FFN after the L25 tail block.

    Copies ``BP_SAVE_PREV`` -> ``OUTPUT_LO/HI`` at the ENT-store val-predictor
    rows (gated on the high OP_ENT broadcast + the MEM_VAL_B markers), so the LM
    head re-emits the clean old_BP byte tokens. Standalone ``PureFFN`` post_op on
    the L25 tail block, after ``tail_bit32_result_correction`` (the 0xFF-garbage
    corruptor), so it is the last writer of OUTPUT before the LM head.
    """
    emission_on = _bp_save_dump_enabled()
    rules = _bp_save_dump_repopulate_rules(emission_on)

    def bake(block, dim_positions, S):
        if not rules:
            return  # flag-off: inert (BP_SAVE_PREV band omitted)
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
        ffn = PureFFN(d_model, len(rules))
        # dim_map — ALL from the declarative LAYOUT (``dim_positions``). The gate
        # dims THIS dump taps (``OP_ENT``, ``MEM_VAL_B*``, ``OUTPUT_LO/HI``) are
        # carried by the model residual at the LAYOUT positions in this build
        # (verified spec_k=0: OP_ENT~10.7 + OUTPUT decode read via ``dim_positions``;
        # the registry positions for these DIFFER, e.g. OUTPUT_LO layout=69 vs
        # registry=174, and tap dead slots). The ``MARK_*`` blockers are at the
        # SAME index in both maps, and ``BP_SAVE_PREV`` exists only in the layout,
        # so resolving everything from ``dim_positions`` is correct. (This is the
        # OPPOSITE of the AX/STACK0 dumps, whose H1/ADDR_B0_LO/AX_CARRY pipeline
        # IS at the registry positions — those dims are baked by the legacy path.)
        from ...dim_registry_dynamic import build_default_registry_dynamic
        _reg = build_default_registry_dynamic()  # noqa: F841 (parity w/ peers)
        _new_bands = {
            "BP_SAVE_PREV", "OP_ENT", "OUTPUT_LO", "OUTPUT_HI",
            "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
            "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE",
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

    # Flag-off: empty reads/writes so the op never references the OMITTED
    # ``BP_SAVE_PREV`` band (which would fail ``add_op`` dim validation).
    if emission_on:
        reads = {
            "OP_ENT", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
            "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
            "MARK_MEM", "MARK_SE", "BP_SAVE_PREV",
        }
        writes = {"OUTPUT_LO", "OUTPUT_HI", "BP_SAVE_PREV"}
    else:
        reads = set()
        writes = set()

    return Operation(
        name="bp_save_dump_repopulate",
        reads=reads,
        writes=writes,
        audited_empty_produces=not emission_on,
        kind="block",
        # Append AFTER the tail correction on the L25 tail block, so this op is
        # the LAST writer of OUTPUT_LO/HI before the LM head reads them (it runs
        # after the block-32+ corruptor that writes the 0xFF garbage).
        target_op_name="l10_post_ops_combined",
        requires={"after": ("tail_bit32_result_correction",)},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="FUNC_LEV_IS_LI_FROM_FRAME_37TOKEN_DESYNC_2026_06_14.md",
    )
