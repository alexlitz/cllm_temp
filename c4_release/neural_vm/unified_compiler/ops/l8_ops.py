"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import dataclasses
import os as _os_l8

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule, step_function_rule
from ..ir import CompilerIR, ConditionTerm, DimRef, FFNRule, StepWindowConstraint
from ..layer_compiler import Operation
from ..positional_invariant import marker_bank_index
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy, no_stack0_emit_enabled


# === L8 attention head layout (auto-fit; legacy head_idx as docs) ====
#
# L8 hosts five attention-bake op families competing for an 8-head budget.
# Pre-migration each ``DeclarativeAttentionHeadSpec`` / hand-rolled
# weight-write block carried a literal ``head_idx=N``, making it fragile
# to add a new head (the author had to remember which slots were already
# taken). The layout is now structural: every load-bearing ``head_idx``
# is resolved via :data:`_L8_HEAD_LAYOUT_BY_NAME` inside the spec
# functions / inline blocks below, so the bakes stay byte-identical
# regardless of how the allocator orders its inventory. A future L8 attn
# op can claim a free slot via ``allocator.alloc(name, layer_idx=8)``
# without touching the existing weight writes.
#
# Heads 6 and 7 are intentionally listed twice — once as
# ``layer8_sp_gather_bake`` mirror heads (the production owners; both
# guarded ops below default to ``enable=False``) and once as the
# alternative owners (``layer8_head6_ax_carry_refresh`` /
# ``layer8_mem_to_alu``). The aliasing only matters when an alternative
# owner is flipped to ``enable=True``; in that configuration the
# sp_gather mirror writes are an acknowledged collision (see the head
# spec docstring in :func:`_layer8_sp_gather_head_specs`). Each bake
# instantiates its OWN allocator over only the heads it actually writes,
# so the collision is not raised by the allocator at registration
# time — but the table below documents the full picture so a future
# reconciliation has a single source of truth.
#
# Phase 7.B.3: ``pin=`` is dropped from ``_allocate_layer8_attn_heads``.
# Each per-bake allocator is constructed from a subset of op names; the
# subset is contiguous in declaration order so first-fit reproduces the
# legacy ``head_idx`` values bit-for-bit for the production subsets
# tested by ``tests/test_l8_per_op.py``. The ``legacy_head_idx`` column
# is documentation only -- weight writes still flow through
# :data:`_L8_HEAD_LAYOUT_BY_NAME`.
_L8_HEAD_LAYOUT = (
    # (op-name key,                                       legacy_head_idx (docs only))
    ("layer8_sp_gather_bake.head_0",                      0),  # SP gather j=0
    ("layer8_sp_gather_bake.head_1",                      1),  # SP gather j=1
    ("layer8_sp_gather_bake.head_2",                      2),  # SP gather j=2
    ("layer8_multibyte_fetch_bake.head_3",                3),  # multi-byte IMM fetch
    ("layer8_op_imm_relay.head_4",                        4),  # OP_IMM relay at AX bytes
    ("layer8_mem_to_alu.head_5",                          5),  # mem[SP] -> ALU at AX
    ("layer8_sp_gather_bake.head_6_mark_sp_mirror",       6),  # MARK_SP mirror of j=0
    ("layer8_sp_gather_bake.head_7_mark_sp_mirror",       7),  # MARK_SP mirror of j=1
    ("layer8_head6_ax_carry_refresh.head_6",              6),  # alt owner (enable=False)
    ("layer8_mem_to_alu.head_7",                          7),  # alt owner (enable=False)
)
_L8_HEAD_LAYOUT_BY_NAME = {name: head_idx for name, head_idx in _L8_HEAD_LAYOUT}


def _allocate_layer8_attn_heads(op_names: tuple[str, ...]) -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` for ``op_names``.

    Each L8 attn bake calls this with the subset of
    :data:`_L8_HEAD_LAYOUT` entries it owns and stashes the result on
    ``block.attn._l8_head_allocator`` for downstream inspection.

    Phase 7.B.3: ``pin=`` is dropped; the allocator first-fits each
    name in ``op_names`` order over the unclaimed slots in
    ``[0, layer_max_heads)``. The weight-write head indices are still
    looked up via :data:`_L8_HEAD_LAYOUT_BY_NAME` in the spec functions
    / inline blocks below, so byte-identity with the legacy bake is
    preserved regardless of allocator order. The allocator inventory
    is bookkeeping for layout audits (``block.attn._l8_head_allocator``).

    Subsetting (rather than including the full table on every call)
    sidesteps the head 6/7 dual ownership documented in
    :data:`_L8_HEAD_LAYOUT`: the sp_gather mirrors and the alt owners
    never share a single allocator instance, so duplicate-name errors
    fire only on intra-op duplicates (which is what we want).
    """
    allocator = AttentionHeadAllocator()
    for name in op_names:
        if name not in _L8_HEAD_LAYOUT_BY_NAME:
            raise KeyError(
                f"_allocate_layer8_attn_heads: unknown L8 attn op {name!r}"
            )
        allocator.alloc(name, layer_idx=8)
    return allocator


# === L8 FFN unit layout (auto-fit; legacy offsets retained as docs) ===
#
# The L8 FFN hosts four op families that historically picked their
# hidden-unit ranges by hand. ``layer8_alu`` runs the monolithic
# ``vm_step._set_layer8_alu`` helper which walks a local ``unit = 0``
# counter through 18 sub-stages (ADD/LEA/SUB/ADJ/ENT lo nibbles + carry
# detection + CMP_GROUP + ENT/ADJ defaults + LEV byte-0 relay + LEA
# first-step byte-2 output) totalling 2023 units. ``layer8_multibyte_routing``
# appends a further 32 units at offset 2023 (16 lo + 16 hi for the
# multi-byte IMM route). ``layer8_sp_gathered_sentinel`` lands a single
# SwiGLU unit at offset 2055 (the per-layer FFN width therefore is 2056).
# ``format_position_counter`` is an opt-in alias: when
# ``enable_conversational_io`` is set it overlays 16 units at offset 600,
# overwriting a slice of the ALU's SUB lo nibble cluster -- a deliberate
# legacy choice carried forward via ``allow_overlap=True``.
#
# Phase 7.B.3: ``pin=`` is dropped from every contiguous entry below.
# The layout is fully contiguous (every sub-stage starts exactly where
# the previous one ended) so :class:`FFNUnitAllocator` first-fit
# reproduces the legacy offsets bit-for-bit. The ALU sub-stage rows
# mirror the cursor walk inside ``vm_step._set_layer8_alu`` exactly; the
# carry/borrow widths are the number of ``(a, b)`` pairs that satisfy
# the helper's branch condition (e.g. ``a + b >= 16`` gives 120). Changing
# any helper's unit count requires updating this table in lock-step.
# The ``legacy_start`` column is documentation only -- the load-bearing
# pin is the cursor walk in the helpers themselves. ``_l8_ffn_range_start``
# resolves any downstream-needed start via the allocator's first-fit
# result (which equals the legacy offset for this contiguous layout).
#
# Retained pin: ``format_position_counter`` keeps its explicit
# ``pin=600, allow_overlap=True`` because it is a SEMANTIC alias (it
# intentionally overwrites a slice of ``layer8_alu.sub_lo``); first-fit
# would land it past the rest of the layout instead of on the trained
# overlay slot.
_L8_FFN_UNIT_LAYOUT = (
    # ---- layer8_alu sub-stages (cursor walk in _set_layer8_alu) ----
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer8_alu.add_lo",                0, 256),  # ADD lo nibble
    ("layer8_alu.lea_lo",              256, 256),  # LEA lo nibble (FETCH_LO)
    ("layer8_alu.sub_lo",              512, 256),  # SUB lo nibble
    ("layer8_alu.add_carry",           768, 120),  # ADD carry (a+b >= 16)
    ("layer8_alu.lea_carry",           888, 120),  # LEA carry (a+b >= 16)
    ("layer8_alu.adj_lo",             1008, 256),  # ADJ lo nibble (FETCH_LO)
    ("layer8_alu.adj_carry",          1264, 120),  # ADJ carry (a+b >= 16)
    ("layer8_alu.sub_borrow",         1384, 120),  # SUB borrow (a < b)
    ("layer8_alu.ent_lo",             1504, 256),  # ENT lo nibble (FETCH_LO)
    ("layer8_alu.ent_borrow",         1760, 220),  # ENT borrow (full_sum cases)
    ("layer8_alu.cmp_group",          1980,   1),  # CMP_GROUP flag
    ("layer8_alu.cmp_clear",          1981,   4),  # CMP[0..3] clear at AX
    ("layer8_alu.ent_adj_defaults",   1985,   2),  # ENT/ADJ first-step ALU defaults
    ("layer8_alu.lev_byte0_lo",       1987,  16),  # LEV BP byte0 lo -> ADDR_B0_LO
    ("layer8_alu.lev_byte0_hi",       2003,  16),  # LEV BP byte0 hi -> ADDR_B0_HI
    ("layer8_alu.lev_b1",             2019,   1),  # LEV ADDR_B1_LO zero
    ("layer8_alu.lev_b2",             2020,   1),  # LEV ADDR_B2_LO zero
    ("layer8_alu.lea_axb2",           2021,   2),  # LEA first-step AX byte 2
    # ---- standalone L8 FFN ops ----
    ("layer8_multibyte_routing",      2023,  32),  # 16 lo + 16 hi IMM route
    ("layer8_sp_gathered_sentinel",   2055,   1),  # MARK_SP SwiGLU sentinel
)

# format_position_counter is an opt-in alias that intentionally overwrites
# a slice of layer8_alu.sub_lo (units 600..615). Tracked as a separate
# entry because allocator pins must be declared with allow_overlap=True
# and aliases should not consume free units. Only registered when the
# conversational-io flag is set on bake. RETAINED PIN: this entry keeps
# ``pin=600, allow_overlap=True`` after Phase 7.B.3 because the overlay
# is semantically load-bearing (the position-counter rules overwrite a
# trained ADD-carry slice); first-fit would land elsewhere.
_L8_FFN_FORMAT_POS_COUNTER_PIN = 600
_L8_FFN_FORMAT_POS_COUNTER_UNITS = 16


def _allocate_layer8_ffn_units(
    *, include_format_position_counter: bool = False
) -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` for the L8 FFN.

    Phase 7.B.3: ``pin=`` is dropped from every contiguous entry in
    :data:`_L8_FFN_UNIT_LAYOUT`. First-fit over the contiguous layout
    reproduces the legacy offsets bit-for-bit, so downstream callers
    that read ``_l8_ffn_range_start(allocator, name)`` (e.g.
    ``layer8_multibyte_routing`` and ``layer8_sp_gathered_sentinel``)
    observe the same starts as before. The actual weight writes still
    flow through ``vm_step._set_layer8_alu`` /
    ``vm_step._set_layer8_multibyte_routing`` -- whose monotonic
    ``unit`` counters land on exactly the same hidden units they always
    have. The allocator is bookkeeping rather than persistent state:
    each bake instantiates a fresh one and stashes it on
    ``block.ffn._l8_unit_allocator`` so downstream tooling can inspect
    the layout.

    ``include_format_position_counter`` adds the convo-io alias range
    at unit 600 with ``allow_overlap=True``. The alias retains its
    explicit pin because the overlay is semantically load-bearing
    (see comment on :data:`_L8_FFN_FORMAT_POS_COUNTER_PIN`). Pass the
    flag only on bakes where the conversational-io flag is set;
    otherwise the legacy sub_lo cluster is left untouched.
    """

    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L8_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    if include_format_position_counter:
        allocator.alloc(
            "format_position_counter",
            _L8_FFN_FORMAT_POS_COUNTER_UNITS,
            pin=_L8_FFN_FORMAT_POS_COUNTER_PIN,
            allow_overlap=True,
        )
    return allocator


def _l8_ffn_range_start(allocator: FFNUnitAllocator, op_name: str) -> int:
    """Return the pinned start unit for ``op_name`` in ``allocator``."""

    for r in allocator.ranges():
        if r.op_name == op_name:
            return r.start
    raise KeyError(f"L8 FFN allocator missing range {op_name!r}")


def _band_projection_writes(slot_base: int, dim_base: int, weight: float = 1.0):
    return tuple(AP(slot_base + k, dim_base + k, weight) for k in range(16))


def _band_output_writes(dim_base: int, slot_base: int, weight: float = 1.0):
    return tuple(AO(dim_base + k, slot_base + k, weight) for k in range(16))


# attention_verifier honesty: the L8 SP-gather heads (0/1/2/6/7) write the
# ADDR_B*_LO/HI bands at a magnitude bound of 1.0 (|O| * |V| = 1*1 per
# output dim), which loses to L15 store_stack0_sp_byte0_addr (mag 5.0 via
# its -2/+3 cancel structure) and ties L7 memory heads / L5 fetch heads
# (mag 1.0, dim-aliased into ADDR_B*) under the V1 verifier's strict
# ``my_mag >= competing_max + margin`` rule.  These heads CAN'T be
# silenced by dropping declarative claims because the attention-side
# ``verify_attention_head`` emits ``attention_strength_violation``
# unconditionally (no honest "no-claim" gating today, unlike the FFN-side
# ``verify_rule_strength`` which honors a missing ``dominates_at``).
#
# The structural fix below is a SEMANTICALLY-NEUTRAL magnitude lift via
# a shared cancel-pair on two unused V slots (34, 35): each slot reads
# CONST (the always-1.0 dim) and the O-projection writes +ADDR_MAG_BOOST
# / -ADDR_MAG_BOOST for the same output dim. Because CONST is identically
# 1.0 at every key position, ``softmax · (W_v · residual)`` delivers
# +ADDR_MAG_BOOST and -ADDR_MAG_BOOST through the two slots; they cancel
# exactly at the q-position output regardless of attention sharpness or
# saturation (the softmax weights normalize to 1 across keys).
#
# The W_o bake uses ``=`` assignment per (out_dim, slot), so distinct
# slots never overwrite each other; the existing slot-1+k / slot-17+k
# bands continue to deliver the CLEAN_EMBED nibbles unchanged. Verifier
# magnitude bound becomes ``|O_old|*|V_old| + |+N|*|V_const| +
# |-N|*|V_const| = 1 + 2N`` per output dim, so N=3 gives bound 7,
# beating L15's 5.0 + margin 1.0 = 6.0 cleanly and dominating the L7
# memory-head / L5-fetch dim-aliasing competitors (mag 1.0, required 2.0).
#
# Slots 34/35 are free on heads 0/1/2/6/7 in the production build:
#   - head 3 (multibyte_fetch) uses slots 32..51 on its own head_idx=3
#     row range; head_idx is the outer index into W_q rows, so different
#     head_idx values never collide.
#   - head 6 (``make_layer8_head6_ax_carry_refresh_op``) and heads 5/7
#     (``make_layer8_mem_to_alu_op``) are ``enable=False`` in the
#     production bake, so their slot usage is irrelevant.
# HD=64 (NUM_HEADS=8, MODEL_DIM=512) leaves slots 34..63 free per head.
#
# Collateral noted in the design: lifting the L8 head magnitude to 7
# causes ``layer15_store_stack0_sp_byte0_addr`` head 12 (mag 5.0) to
# newly violate against L8 at ADDR_B0_LO/HI+0..15 (32 issues). The two
# heads NEVER fire at the same q-row at runtime — L8 gates on
# MARK_STACK0 / MARK_SP (its Q-side gate triplet), L15 gates on
# MEM_STORE + HAS_SE — but the V1 attention verifier does not model
# Q-side mutual-exclusion, so it conservatively reports them as
# competitors.  The verifier-strict symmetry forces an asymmetric
# tradeoff (one side must dominate); the L8 → ADDR_B0 fix is in scope,
# the L15 collateral is documented in the per-op verifier log.
_ADDR_MAG_BOOST_SLOT_PLUS = 34
_ADDR_MAG_BOOST_SLOT_MINUS = 35
_ADDR_MAG_BOOST_WEIGHT = 3.0


def _addr_mag_boost_v(BD):
    """V writes for the L8 SP-gather verifier-magnitude cancel pair.

    Both slots read CONST (=1.0 at every token) so their delivered values
    are identical at the q-position output. Paired with +N/-N O writes
    they net to zero — semantically neutral by construction.
    """
    return (
        AP(_ADDR_MAG_BOOST_SLOT_PLUS, BD.CONST, 1.0),
        AP(_ADDR_MAG_BOOST_SLOT_MINUS, BD.CONST, 1.0),
    )


def _addr_mag_boost_o(dim_base):
    """O writes for the verifier-magnitude cancel pair targeting one
    16-wide band (e.g. ``ADDR_B0_LO``). The +N and -N writes hit the same
    output dim through distinct slots (34, 35), so the
    ``W_o[dim, base + slot]`` matrix entries are independent and both
    bakes persist.
    """
    writes = []
    for k in range(16):
        writes.append(
            AO(dim_base + k, _ADDR_MAG_BOOST_SLOT_PLUS, _ADDR_MAG_BOOST_WEIGHT)
        )
        writes.append(
            AO(dim_base + k, _ADDR_MAG_BOOST_SLOT_MINUS, -_ADDR_MAG_BOOST_WEIGHT)
        )
    return tuple(writes)


# === layer8_alu FFNRule migration ===================================
#
# The L8 ALU helper (``vm_step._set_layer8_alu``) writes ~2023 hidden units
# arranged in 18 sub-stages (see :data:`_L8_FFN_UNIT_LAYOUT`). Each
# sub-stage below returns a tuple of :class:`FFNRule` that lowers to the
# same per-unit ``W_up`` / ``b_up`` / ``W_gate`` / ``b_gate`` / ``W_down``
# writes the imperative helper produces, byte-for-byte.
#
# Substage cursor layout (matches _L8_FFN_UNIT_LAYOUT, also re-stated here
# for cross-reference):
#
#   0..255      add_lo          (ADD lo nibble, gate=OP_ADD)
#   256..511    lea_lo          (LEA lo nibble, gate=OP_LEA, FETCH_LO)
#   512..767    sub_lo          (SUB lo nibble, gate=OP_SUB)
#   768..887    add_carry       (ADD carry, gate=OP_ADD, 120 pairs)
#   888..1007   lea_carry       (LEA carry, gate=OP_LEA, FETCH_LO, 120)
#   1008..1263  adj_lo          (ADJ lo nibble, gate=OP_ADJ, FETCH_LO)
#   1264..1383  adj_carry       (ADJ carry, gate=OP_ADJ, FETCH_LO, 120)
#   1384..1503  sub_borrow      (SUB borrow, gate=OP_SUB, 120 pairs a<b)
#   1504..1759  ent_lo          (ENT lo nibble, gate=OP_ENT, FETCH_LO)
#   1760..1979  ent_borrow      (ENT borrow, 220 pairs, gate=OP_ENT)
#   1980        cmp_group       (any comparison opcode flag; 1 unit)
#   1981..1984  cmp_clear       (clear CMP[0..3] at AX marker; 4 units)
#   1985..1986  ent_adj_defaults (ENT/ADJ first-step ALU defaults; 2)
#   1987..2002  lev_byte0_lo    (LEV BP byte0 lo → ADDR_B0_LO; 16)
#   2003..2018  lev_byte0_hi    (LEV BP byte0 hi → ADDR_B0_HI; 16)
#   2019        lev_b1          (LEV ADDR_B1_LO zero; 1)
#   2020        lev_b2          (LEV ADDR_B2_LO zero; 1)
#   2021..2022  lea_axb2        (LEA first-step AX byte 2; 2)
#
# Each rule's ``scope=`` mirrors the helper's intent (the dim-presence
# conjunction that the unit fires on). The ``dominates_at=`` field is
# left at the rule-level scope; per-output dominance disambiguation is
# only needed when a rule writes multiple output dims with different
# competition shapes -- in this helper every rule writes either a
# single OUTPUT_LO[k] / CARRY+0 / CMP_GROUP / ADDR_B*_LO[k] / ALU_*[0]
# cell, so the scope is unambiguous.


def _layer8_alu_block_non_ax_marker_conditions() -> tuple[tuple[str, float], ...]:
    """LEA/ADJ/ENT non-AX-marker blockers (-S * 1000 on each blocker dim).

    These mirror ``_block_non_ax_marker_sites`` inside ``_set_layer8_alu``:
    blockers prevent the unit from firing at non-AX marker sites where
    scale-40 LEA/ADJ/ENT operand fetch residuals could otherwise sneak
    the unit on. Returned as condition terms (weight=-1000) so the
    ``Primitives.lower_ffn_rules`` scales them by S, matching the
    ``-S * 1000`` imperative writes.
    """
    return (
        ("MARK_PC", -1000.0),
        ("MARK_SP", -1000.0),
        ("MARK_BP", -1000.0),
        ("MARK_STACK0", -1000.0),
        ("MARK_MEM", -1000.0),
        ("MARK_SE", -1000.0),
        ("IS_BYTE", -1000.0),
    )


def _layer8_alu_add_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """ADD lo nibble (256 units, offsets 0..255).

    For each (a, b) pair in 16x16, fire when MARK_AX active, ALU_LO[a]
    and AX_CARRY_LO[b] both one-hot, OP_ADD gating. MARK_PC blocker
    (-S * 4) suppresses the unit at PC marker where L6 head 0 / L7
    attention leakage could otherwise sneak it on. Writes
    OUTPUT_LO[(a+b) mod 16] at 2.0/S.

    Phase 8.D: the OP_ADD gate resolves through ``dim_ref("opcode_flag",
    "ADD")`` -- the gate dim names the opcode-flag family member.
    ALU_LO+a / AX_CARRY_LO+b operand reads stay structural (per-nibble
    one-hot lookups). OUTPUT_LO+result writes stay structural too --
    the per-nibble result is a value-bus lookup, not a role-meaningful
    byte position.
    """
    write_scale = 2.0 / S
    gate_add = dim_ref("opcode_flag", "ADD")
    rules = []
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_add_lo_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_add,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_SE_ONLY and OP_ADD",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_SE_ONLY and OP_ADD",
                },
            ))
    return tuple(rules)


def _layer8_alu_lea_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """LEA lo nibble (256 units, offsets 256..511).

    Like ADD lo but reads operand-B from FETCH_LO (one-hot) instead of
    AX_CARRY_LO, gates on OP_LEA, and uses the strong MARK_AX
    requirement (60) + the non-AX-marker blocker set (-1000 on each of
    MARK_PC/SP/BP/STACK0/MEM/SE/IS_BYTE). FETCH_LO contributes at
    weight 20.

    Phase 8.D: the OP_LEA gate resolves through ``dim_ref("opcode_flag",
    "LEA")``.
    """
    write_scale = 2.0 / S
    gate_lea = dim_ref("opcode_flag", "LEA")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    rules = []
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_lea_lo_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=80.5,
                gate=gate_lea,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_SE_ONLY and OP_LEA",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_SE_ONLY and OP_LEA",
                },
            ))
    return tuple(rules)


def _layer8_alu_sub_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """SUB lo nibble (256 units, offsets 512..767).

    C4 semantics: AX = stack_top - AX, so result = ALU_LO[a] - AX_CARRY_LO[b].
    Same structural shape as ADD lo (gate=OP_SUB).

    Phase 8.D: the OP_SUB gate resolves through ``dim_ref("opcode_flag",
    "SUB")``.
    """
    write_scale = 2.0 / S
    gate_sub = dim_ref("opcode_flag", "SUB")
    rules = []
    for a in range(16):
        for b in range(16):
            result = (a - b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_sub_lo_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_sub,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_SE_ONLY and OP_SUB",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_SE_ONLY and OP_SUB",
                },
            ))
    return tuple(rules)


def _layer8_alu_add_carry_rules(S: float) -> tuple[FFNRule, ...]:
    """ADD carry detection (120 units, offsets 768..887).

    Same conditions as add_lo (MARK_AX + ALU_LO[a] + AX_CARRY_LO[b],
    MARK_PC blocker, OP_ADD gate) but only emits a unit when
    ``a + b >= 16`` (carry-out from the lo nibble). Writes the
    ``(carry, alu)`` byte-0 cell normalized by 2.0/(S*5.0) so the
    gated output ~1 after scaling.

    Phase 7.E.2: gate + carry-output refs use the
    ``(category, role)`` form via :func:`dim_ref`. The byte-position
    semantics of the carry write (``offset=0`` = byte 0 of the
    inter-byte cascade) and the opcode-family semantics of the gate
    are now explicit in the rule definition. The ``ALU_LO+a`` /
    ``AX_CARRY_LO+b`` reads stay as ``+N`` because those offsets are
    structural per-nibble one-hot lookups, not byte-index roles.
    """
    carry_scale = 2.0 / (S * 5.0)
    carry_byte0 = dim_ref("carry", "alu", 0)
    gate_add = dim_ref("opcode_flag", "ADD")
    rules = []
    for a in range(16):
        for b in range(16):
            if a + b < 16:
                continue
            rules.append(multi_way_and_rule(
                name=f"l8_alu_add_carry_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_add,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_SE_ONLY and OP_ADD",
                dominates_at={carry_byte0: "MARK_SE_ONLY and OP_ADD"},
            ))
    return tuple(rules)


def _layer8_alu_lea_carry_rules(S: float) -> tuple[FFNRule, ...]:
    """LEA carry detection (120 units, offsets 888..1007).

    Same structural shape as lea_lo (gate=OP_LEA, FETCH_LO operand,
    non-AX blockers) but only emits when ``a + b >= 16``. Writes the
    ``(carry, alu)`` byte-0 cell at the same 2.0/(S*5.0) normalization
    as ADD carry.

    Phase 7.E.2: gate + carry-output refs use ``dim_ref`` for the
    semantic ``(opcode_flag, LEA)`` and ``(carry, alu, byte_index=0)``
    pairs. ALU_LO+a / FETCH_LO+b stay structural (per-nibble one-hot
    lookups).
    """
    carry_scale = 2.0 / (S * 5.0)
    carry_byte0 = dim_ref("carry", "alu", 0)
    gate_lea = dim_ref("opcode_flag", "LEA")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    rules = []
    for a in range(16):
        for b in range(16):
            if a + b < 16:
                continue
            rules.append(multi_way_and_rule(
                name=f"l8_alu_lea_carry_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=80.5,
                gate=gate_lea,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_SE_ONLY and OP_LEA",
                dominates_at={carry_byte0: "MARK_SE_ONLY and OP_LEA"},
            ))
    return tuple(rules)


def _adj_lo_ax_marker_blocker_on() -> bool:
    """DEFAULT-OFF guard (``C4_L8_ADJ_LO_AX_MARKER_BLOCKER=1``): add a MARK_AX
    NOT-blocker to the ``l8_alu_adj_lo_*_step_end`` ADJ low-nibble ALU rules so
    the SP-adjustment result cannot leak into the AX register dump.

    Root (spec_k=0, tools/_probe_l9_outlo8.py, func_identity_0 id550 with the
    C4_L15_LEV func chain ON): the func epilogue's ADJ step (``ADJ 8`` cleaning
    the call frame) computes ``SP = SP + 8`` in the L8 ADJ ALU. The result's low
    nibble (8) is staged into ``OUTPUT_LO[8]`` for relay to the SP register. But
    the ADJ-lo rule fires on the ``MARK_SE_ONLY`` residue WITHOUT a MARK_AX
    blocker (the shared ``_block_non_ax_marker_conditions`` set deliberately
    leaves MARK_AX un-blocked, because the sibling ADD/SUB ALU rules DO write
    their result at the AX row). On the ADJ AX byte-0 row the staged
    ``OUTPUT_LO[8]`` (the SP-adjustment ``8``) is the DOMINANT writer (+8.83,
    attributed to ``l8_alu_adj_lo_a0_b8_step_end``), overwhelming the correct
    AX-carry materialization of the returned value (``l6_adj_ax_to_output_lo_6``
    only reaches +5.13 at the weak post-LEV ADJ row) -> the AX byte-0 low nibble
    decodes as 8 instead of 6 -> ``identity(70)`` returns ``72`` (0x46 -> 0x48).
    Uniform low-nibble->8 corruption across func_identity because the ADJ amount
    is always one stack slot (8).

    Fix: ADJ writes SP, never AX, so a MARK_AX NOT-blocker on the ADJ-lo result
    write is semantically correct AND surgical -- it vetoes the leak onto the AX
    row while the legitimate SP-staging fires at MARK_SE_ONLY (MARK_AX=0) exactly
    as before. DEFAULT-OFF so HEAD stays byte-identical; ships with the
    C4_L15_LEV func chain. Same op-broadcast-corruptor family as the L16 LEV
    STACK0 preserve SE-blocker, now for the ADJ SP result on the AX row.
    """
    return _os_l8.environ.get("C4_L8_ADJ_LO_AX_MARKER_BLOCKER", "0") == "1"


def _layer8_alu_adj_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """ADJ lo nibble (256 units, offsets 1008..1263).

    ADJ computes SP = SP + signed_immediate. Reads ALU_LO (SP lo nibble
    from L7) and FETCH_LO (immediate). Gate=OP_ADJ; threshold tuned
    higher (85) than LEA (80.5) to reflect the helper's empirical
    margin.

    Phase 8.D: the OP_ADJ gate resolves through ``dim_ref("opcode_flag",
    "ADJ")``.
    """
    write_scale = 2.0 / S
    gate_adj_lo = dim_ref("opcode_flag", "ADJ")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    if _adj_lo_ax_marker_blocker_on():
        # ADJ writes SP, never AX: veto the SP-result leak onto the AX dump row.
        blockers = (*blockers, ("MARK_AX", -1000.0))
    rules = []
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_adj_lo_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=85.0,
                gate=gate_adj_lo,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_SE_ONLY and OP_ADJ",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_SE_ONLY and OP_ADJ",
                },
            ))
    return tuple(rules)


def _layer8_alu_adj_carry_rules(S: float) -> tuple[FFNRule, ...]:
    """ADJ carry detection (120 units, offsets 1264..1383).

    Same shape as lea_carry but gated on OP_ADJ with the ADJ-tuned
    threshold (85.0 vs LEA's 80.5).

    Phase 7.E.2: gate + carry-output refs use ``dim_ref`` for the
    ``(opcode_flag, ADJ)`` and ``(carry, alu, byte_index=0)`` pairs.
    ALU_LO+a / FETCH_LO+b stay structural.
    """

    carry_scale = 2.0 / (S * 5.0)
    carry_byte0 = dim_ref("carry", "alu", 0)
    gate_adj = dim_ref("opcode_flag", "ADJ")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    rules = []
    for a in range(16):
        for b in range(16):
            if a + b < 16:
                continue
            rules.append(multi_way_and_rule(
                name=f"l8_alu_adj_carry_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=85.0,
                gate=gate_adj,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_SE_ONLY and OP_ADJ",
                dominates_at={carry_byte0: "MARK_SE_ONLY and OP_ADJ"},
            ))
    return tuple(rules)


def _layer8_alu_sub_borrow_rules(S: float) -> tuple[FFNRule, ...]:
    """SUB borrow detection (120 units, offsets 1384..1503).

    Borrow occurs when ALU_LO[a] < AX_CARRY_LO[b] (stack_top < AX in
    this nibble). Mirrors SUB lo structure with the
    ``(opcode_flag, SUB)`` gate, writing the ``(carry, alu)`` byte-0
    cell.

    Phase 7.E.2: gate + carry-output refs use ``dim_ref`` (mirrors the
    add_carry / lea_carry pilots). ALU_LO+a / AX_CARRY_LO+b operand
    reads stay structural.
    """
    carry_scale = 2.0 / (S * 5.0)
    carry_byte0 = dim_ref("carry", "alu", 0)
    gate_sub = dim_ref("opcode_flag", "SUB")
    rules = []
    for a in range(16):
        for b in range(16):
            if a >= b:
                continue
            rules.append(multi_way_and_rule(
                name=f"l8_alu_sub_borrow_a{a}_b{b}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_sub,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_SE_ONLY and OP_SUB",
                dominates_at={carry_byte0: "MARK_SE_ONLY and OP_SUB"},
            ))
    return tuple(rules)


def _layer8_alu_ent_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT lo nibble subtraction (256 units, offsets 1504..1759).

    ENT computes SP = SP - (8 + signed_immediate). For lo nibble:
    result = (sp_lo - (8 + imm_lo)) mod 16. sp_lo from ALU_LO,
    imm_lo from FETCH_LO. Gate=OP_ENT, threshold=85.

    Phase 8.D: the OP_ENT gate resolves through ``dim_ref("opcode_flag",
    "ENT")``.
    """
    write_scale = 2.0 / S
    gate_ent_lo = dim_ref("opcode_flag", "ENT")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    rules = []
    for sp_lo in range(16):
        for imm_lo in range(16):
            effective_b = (8 + imm_lo) % 16
            result = (sp_lo - effective_b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_ent_lo_sp{sp_lo}_imm{imm_lo}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    # PER-STEP ACTUAL-IMM BLOCKER (func / simple_function fix).
                    # This ENT lo-nibble band (SP - (8+imm)) is gated on
                    # OP_ENT, which is durably carried across the call frame
                    # from the ENT step into the callee body. On the callee
                    # ``IMM`` step OP_ENT is still ~5 at the AX marker, so the
                    # band SPURIOUSLY computes the ENT SP-decrement constant
                    # over the IMM operand and writes it onto OUTPUT_LO,
                    # burying the genuine immediate -> ``ENT 0; IMM 42`` emits
                    # 40/8 (the func / 150-fail cluster). The L5 main-AX opcode
                    # decode produces a CLEAN per-step OP_IMM one-hot (= +5.0
                    # ONLY on real IMM steps, 0 on real ENT steps; probed
                    # spec_k=0) that is present at the AX marker where this
                    # band actually fires, so a strong OP_IMM NOT-blocker
                    # suppresses it on the callee IMM step while leaving a real
                    # ENT step (OP_IMM = 0) byte-identical. Same class as the
                    # HAS_SE blocker in ``l9_ops._layer9_ent_hi_nibble_rules``
                    # and the head-1 actual-IMM suppression in
                    # ``l7_ops._layer7_operand_gather_head_specs``.
                    ("OP_IMM", -1000.0),
                    # SUBSEQUENT-STEP ENT BLOCKER (absdiff / func / nested /
                    # rec step-1 ENT-AX leak fix, 2026-06-14). This band is
                    # the LO-nibble sibling of l9_ops._layer9_ent_hi_nibble_
                    # rules and leaks the SP-decrement result onto OUTPUT_LO
                    # at the AX register-marker row. AX must be PRESERVED
                    # across ENT (C calling convention), so on the JSR->ENT
                    # prologue step (HAS_SE=1: every corpus func/absdiff/
                    # nested/rec callee ENT) the band must NOT write the
                    # SP-decrement low nibble onto AX byte0. The leak was
                    # masked whenever (sp_lo-(8+imm_lo))%16 == 0 (e.g. main's
                    # ``ENT 8`` -> result 0), but ``ENT 0`` (every callee with
                    # no locals) computes result 8 -> AX byte0 = 0x08, the
                    # universal step-1 divergence for all function-call
                    # clusters. The FIRST-step ENT (HAS_SE=0, e.g. smoke
                    # ``test_lea_basic`` whose bytecode opens with ENT) is
                    # load-bearing (the LEA frame setup reads it) and stays
                    # byte-identical -- hard-block ONLY the HAS_SE=1 case, the
                    # exact mirror of the l9 ent_hi_nibble HAS_SE blocker.
                    ("HAS_SE", -1000.0),
                    (f"ALU_LO+{sp_lo}", 1.0),
                    (f"FETCH_LO+{imm_lo}", 20.0),
                ),
                threshold=85.0,
                gate=gate_ent_lo,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_SE_ONLY and OP_ENT",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_SE_ONLY and OP_ENT",
                },
            ))
    return tuple(rules)


def _layer8_alu_ent_borrow_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT borrow detection (220 units, offsets 1760..1979).

    Borrow when sp_lo < (8 + imm_lo) mod 16, or when (8 + imm_lo) >= 16
    (carry out of byte 0 into byte 1). The condition is asymmetric
    because the +8 constant offset can itself produce a byte-1 carry.

    Phase 7.E.2: gate + carry-output refs use ``dim_ref`` for the
    ``(opcode_flag, ENT)`` and ``(carry, alu, byte_index=0)`` pairs.
    ALU_LO+sp_lo / FETCH_LO+imm_lo stay structural.
    """
    carry_scale = 2.0 / (S * 5.0)
    carry_byte0 = dim_ref("carry", "alu", 0)
    gate_ent = dim_ref("opcode_flag", "ENT")
    blockers = _layer8_alu_block_non_ax_marker_conditions()
    rules = []
    for sp_lo in range(16):
        for imm_lo in range(16):
            full_sum = 8 + imm_lo
            if not (sp_lo < (full_sum % 16) or full_sum >= 16):
                continue
            rules.append(multi_way_and_rule(
                name=f"l8_alu_ent_borrow_sp{sp_lo}_imm{imm_lo}_step_end",
                conditions=(
                    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                    # the Wave A step_end_operand_relay (10ca51a7).
                    ("MARK_SE_ONLY", 60.0),
                    *blockers,
                    # PER-STEP ACTUAL-IMM BLOCKER (func / simple_function fix).
                    # Companion to the ent_lo blocker above: the ENT borrow
                    # band is also OP_ENT-gated and spuriously fires on the
                    # callee IMM step (durable OP_ENT carry). Suppress on the
                    # clean per-step OP_IMM so a real ENT (OP_IMM=0) is
                    # byte-identical. See ``_layer8_alu_ent_lo_rules``.
                    ("OP_IMM", -1000.0),
                    (f"ALU_LO+{sp_lo}", 1.0),
                    (f"FETCH_LO+{imm_lo}", 20.0),
                ),
                threshold=85.0,
                gate=gate_ent,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_SE_ONLY and OP_ENT",
                dominates_at={carry_byte0: "MARK_SE_ONLY and OP_ENT"},
            ))
    return tuple(rules)


def _layer8_alu_cmp_group_rules(S: float) -> tuple[FFNRule, ...]:
    """CMP_GROUP flag (1 unit, offset 1980).

    Fires ~1.0 when any comparison opcode (EQ/NE/LT/GT/LE/GE) is active
    at the AX marker. OP_* flags ~5 each, MARK_AX = 1; threshold=1.5
    keeps the unit silent at non-cmp opcodes. W_down normalized by
    2.0/(S*9) so silu(S*~4.5)*1 * 2/(S*9) ≈ 1.0.
    """
    write_scale = 2.0 / (S * 9.0)
    return (
        multi_way_and_rule(
            name="l8_alu_cmp_group_step_end",
            conditions=(
                ("OP_EQ", 1.0),
                ("OP_NE", 1.0),
                ("OP_LT", 1.0),
                ("OP_GT", 1.0),
                ("OP_LE", 1.0),
                ("OP_GE", 1.0),
                # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under
                # the Wave A step_end_operand_relay (10ca51a7).
                ("MARK_SE_ONLY", 1.0),
            ),
            threshold=1.5,
            writes=(("CMP_GROUP+0", write_scale),),
            scope="MARK_SE_ONLY and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
            dominates_at={
                "CMP_GROUP+0":
                    "MARK_SE_ONLY and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
            },
        ),
    )


def _layer8_alu_cmp_clear_rules(S: float) -> tuple[FFNRule, ...]:
    """CMP[0..3] clearing at AX marker (4 units, offsets 1981..1984).

    L6 attention relay heads write JMP/EXIT/PSH/POP flags to CMP[0..3]
    at every position, including the AX marker -- which would pollute
    the comparison flag dims at the AX marker if not cleared. Each
    unit negates CMP[k] at MARK_AX via the gated read:
    silu(S * CMP[k]) * silu(S/2) * (-2 / S^2) ≈ -CMP[k].

    Note: the legacy unit sets ``W_gate[unit, MARK_AX] = S`` (not 1.0)
    and ``b_gate = -S/2``. We encode this with ``gate_weight=S`` /
    ``gate_bias=-S/2`` because the lowerer applies them directly
    without the S scaling that conditions get.

    Phase 8.D: the MARK_AX gate resolves through ``dim_ref("marker",
    "AX")`` and each CMP+k write resolves through
    ``dim_ref("cmp_flag", "cascade", k)`` -- byte k of the inter-byte
    CMP cascade (hi_lt / hi_eq / lo_eq / lo_lt). The CMP+k condition
    read stays bare per the L8 pilot convention (the condition is the
    current CMP cell being negated, used as an up-branch operand).
    """
    write_scale = -2.0 / (S * S)
    # Wave B Cluster 2: MARK_AX -> MARK_SE_ONLY under the Wave A
    # step_end_operand_relay (10ca51a7). The gate dim now resolves to
    # MARK_SE_ONLY so the CMP clearing fires at STEP_END.
    gate_mark_se = dim_ref("marker", "SE_ONLY")
    rules = []
    for k in range(4):
        cmp_k = dim_ref("cmp_flag", "cascade", k)
        rules.append(multi_way_and_rule(
            name=f"l8_alu_cmp_clear_k{k}_step_end",
            conditions=((f"CMP+{k}", 1.0),),
            threshold=0.0,
            gate=gate_mark_se,
            gate_weight=S,
            gate_bias=-S * 0.5,
            writes=((cmp_k, write_scale),),
            scope="MARK_SE_ONLY",
            dominates_at={f"CMP+{k}": "MARK_SE_ONLY"},
        ))
    return tuple(rules)


def _layer8_alu_ent_adj_defaults_rules(S: float) -> tuple[FFNRule, ...]:
    """ENT/ADJ first-step ALU defaults (2 units, offsets 1985..1986).

    For the first step (NOT HAS_SE), L7 attention can't gather SP
    because the SP marker is AFTER the AX marker (causal attention).
    For ENT/ADJ with initial SP = 0, we need ALU_LO[0] > 0 and
    ALU_HI[0] > 0. These units fire when OP_ENT or OP_ADJ + MARK_AX +
    NOT HAS_SE. MARK_SP blocker prevents firing at the SP marker
    (where OP_ENT can be relayed). Output weight 50/S overrides L7's
    garbage write (~-32).

    Constant write (W_gate untouched, b_gate=1.0) because the original
    helper uses the SiLU path for gating (b_gate=1.0 only, no W_gate
    write).
    """
    output_weight = 50.0 / S
    rules = []
    for alu_base in ("ALU_LO", "ALU_HI"):
        rules.append(multi_way_and_rule(
            name=f"l8_alu_ent_adj_default_{alu_base.lower()}",
            conditions=(
                ("OP_ENT", 1.0 / 3.0),
                ("OP_ADJ", 1.0 / 3.0),
                ("MARK_AX", 2.0),
                ("MARK_SP", -10.0),
                ("HAS_SE", -10.0),
            ),
            threshold=6.0,
            writes=((f"{alu_base}+0", output_weight),),
            scope="MARK_AX and (OP_ENT or OP_ADJ) and not HAS_SE",
            dominates_at={
                f"{alu_base}+0":
                    "MARK_AX and (OP_ENT or OP_ADJ) and not HAS_SE",
            },
        ))
    return tuple(rules)


def _layer8_alu_lev_byte0_lo_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV BP byte 0 lo nibble relay (16 units, offsets 1987..2002).

    For L15 to read memory at BP and BP+8, BP's address value must be
    encoded in ADDR_B0/B1/B2 dims at the BP marker position. This
    substage relays OUTPUT_LO[k] -> ADDR_B0_LO[k] at MARK_BP when
    OP_LEV is active (gated on OUTPUT_LO[k]). MARK_PC blocker
    (-S * 10) excludes the PC marker where OP_LEV is amplified ~10.
    """
    write_scale = 2.0 / (S * 9.0)
    rules = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l8_alu_lev_byte0_lo_k{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_BP", 1.0),
                ("MARK_PC", -10.0),
            ),
            threshold=1.5,
            gate=f"OUTPUT_LO+{k}",
            writes=((f"ADDR_B0_LO+{k}", write_scale),),
            scope="OP_LEV and MARK_BP and not MARK_PC",
            dominates_at={
                f"ADDR_B0_LO+{k}": "OP_LEV and MARK_BP and not MARK_PC",
            },
        ))
    return tuple(rules)


def _layer8_alu_lev_byte0_hi_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV BP byte 0 hi nibble relay (16 units, offsets 2003..2018)."""

    write_scale = 2.0 / (S * 9.0)
    rules = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l8_alu_lev_byte0_hi_k{k}",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_BP", 1.0),
                ("MARK_PC", -10.0),
            ),
            threshold=1.5,
            gate=f"OUTPUT_HI+{k}",
            writes=((f"ADDR_B0_HI+{k}", write_scale),),
            scope="OP_LEV and MARK_BP and not MARK_PC",
            dominates_at={
                f"ADDR_B0_HI+{k}": "OP_LEV and MARK_BP and not MARK_PC",
            },
        ))
    return tuple(rules)


def _layer8_alu_lev_b1_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV ADDR_B1_LO[0] zero (1 unit, offset 2019).

    For small addresses (< 256), bytes 1-2 of the LEV-relayed BP
    address are zero. This unit writes ADDR_B1_LO[0] at the
    LEV + BP marker site (no MARK_PC exclusion in legacy -- the unit
    fires at PC marker too, which is harmless since the write target
    is ADDR_B1_LO, not the OUTPUT bank that drives PC).

    The legacy helper writes ``W_gate[unit, CONST] = 1.0`` (no
    explicit b_gate), so we use ``FFNRule.gated_write(gate="CONST",
    gate_weight=1.0, gate_bias=0.0)`` to reproduce the exact W_gate
    cell write. CONST is always 1.0 at runtime so the gate value is
    identical to the b_gate=1.0 form, but this keeps the matrix
    byte-identical for the verifier.
    """
    write_scale = 2.0 / (S * 9.0)
    return (
        multi_way_and_rule(
            name="l8_alu_lev_b1_step_end",
            conditions=(
                ("OP_LEV", 1.0),
                # PRODUCTION REALITY (2026-06-12, L8 ALU live-imperative cut):
                # the now-removed legacy ``_set_layer8_alu`` re-bake (phase 8.3)
                # used to write ``W_up[MARK_BP] = +S`` here via ``=``, which
                # OVERWROTE the Wave-B/broadcast ``MARK_BP=-1e6`` blocker laid
                # down by the phase-8.2 declarative pass. The two passes
                # superpose, so the LIVE model fires this unit at the BP marker
                # (MARK_BP=+S) AND keeps the leftover -1e6 hard NOT-blockers on
                # the OTHER markers (MARK_AX/PC/SP/MEM/STACK0/IS_BYTE) from the
                # phase-8.2 pass, plus MARK_SE_ONLY=+S. With the imperative
                # re-bake folded into this single declarative rule, MARK_BP is
                # the imperative ``=`` winner (+1.0), not the dead -1e6 blocker;
                # MARK_SE_ONLY stays +1.0 (Wave B). Byte-identity proof:
                # tools/verify_l8_alu_migration.py.
                ("MARK_SE_ONLY", 1.0),
                ("MARK_BP", 1.0),
                # Leftover phase-8.2 hard NOT-blockers (subtractive at the legit
                # BP/STEP_END row where IS_BYTE + the other register MARK_* are
                # all 0). The imperative re-bake never wrote these dims, so they
                # survive into the live weights at -1e6 (-> -1e8 after *S).
                ("IS_BYTE", -1e6),
                ("MARK_PC", -1e6),
                ("MARK_AX", -1e6),
                ("MARK_SP", -1e6),
                ("MARK_STACK0", -1e6),
                ("MARK_MEM", -1e6),
            ),
            threshold=1.5,
            gate="CONST",
            writes=(("ADDR_B1_LO+0", write_scale),),
            scope="OP_LEV and MARK_BP and MARK_SE_ONLY",
            dominates_at={"ADDR_B1_LO+0": "OP_LEV and MARK_BP and MARK_SE_ONLY"},
        ),
    )


def _layer8_alu_lev_b2_rules(S: float) -> tuple[FFNRule, ...]:
    """LEV ADDR_B2_LO[0] zero (1 unit, offset 2020).

    Same byte-identity rationale as ``_layer8_alu_lev_b1_rules``: the
    legacy helper writes ``W_gate[unit, CONST] = 1.0``, so the rule
    uses ``FFNRule.gated_write(gate="CONST", gate_weight=1.0,
    gate_bias=0.0)``.
    """
    write_scale = 2.0 / (S * 9.0)
    return (
        multi_way_and_rule(
            name="l8_alu_lev_b2_step_end",
            conditions=(
                ("OP_LEV", 1.0),
                # PRODUCTION REALITY (2026-06-12, L8 ALU live-imperative cut):
                # see l8_alu_lev_b1_step_end above. The now-removed imperative
                # ``_set_layer8_alu`` re-bake wrote ``W_up[MARK_BP] = +S`` here
                # via ``=``, overwriting the phase-8.2 ``MARK_BP=-1e6`` blocker;
                # MARK_BP folds to the ``=`` winner (+1.0). The leftover -1e6
                # NOT-blockers on the OTHER markers survive from the phase-8.2
                # pass. Byte-identity proof: tools/verify_l8_alu_migration.py.
                ("MARK_SE_ONLY", 1.0),
                ("MARK_BP", 1.0),
                ("IS_BYTE", -1e6),
                ("MARK_PC", -1e6),
                ("MARK_AX", -1e6),
                ("MARK_SP", -1e6),
                ("MARK_STACK0", -1e6),
                ("MARK_MEM", -1e6),
            ),
            threshold=1.5,
            gate="CONST",
            writes=(("ADDR_B2_LO+0", write_scale),),
            scope="OP_LEV and MARK_BP and MARK_SE_ONLY",
            dominates_at={"ADDR_B2_LO+0": "OP_LEV and MARK_BP and MARK_SE_ONLY"},
        ),
    )


def _layer8_alu_lea_axb2_rules(S: float) -> tuple[FFNRule, ...]:
    """LEA first-step AX byte 2 output (2 units, offsets 2021..2022).

    BP = 0x10000 so byte 2 = 0x01. Fires only on first step
    (NOT HAS_SE) at AX byte 1 position (BYTE_INDEX_1) when L7 head 5
    has relayed OP_LEA into CMP[7]. The two units write:
      * OUTPUT_LO+1 = +4/S  AND  OUTPUT_LO+0 = -4/S  (single unit, two writes)
      * OUTPUT_HI+0 = +2/S  (single unit, one write)

    Both use ``b_gate = 1.0`` (no explicit W_gate) so they're
    constant_write rules.
    """
    # Class-1 marker-relative anchor: ``H1+AX_I`` reads "nearest L1 threshold-bank
    # marker is AX". The ``+AX_I`` is the frame-invariant marker-TYPE bank slot
    # (``marker_bank_index`` resolves it from Token.STEP_TOKENS, byte-identical to
    # the literal 1 at STEP_TOKENS=35) so the audit recognises the ref — and the
    # mirrored ``scope`` / ``dominates_at`` claim strings — as declared-invariant
    # rather than an UNGUARDED bare offset.
    AX_I = marker_bank_index("AX")
    H1_AX = f"H1+{AX_I}"
    _scope = (
        f"CMP+7 and {H1_AX} and IS_BYTE and BYTE_INDEX_1 and not HAS_SE"
    )
    return (
        multi_way_and_rule(
            name="l8_alu_lea_axb2_lo",
            conditions=(
                ("CMP+7", 1.0),
                (H1_AX, 1.0),
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_1", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=3.5,
            writes=(
                ("OUTPUT_LO+1", 4.0 / S),
                ("OUTPUT_LO+0", -4.0 / S),
            ),
            scope=_scope,
            dominates_at={
                "OUTPUT_LO+1": _scope,
                "OUTPUT_LO+0": _scope,
            },
        ),
        multi_way_and_rule(
            name="l8_alu_lea_axb2_hi",
            conditions=(
                ("CMP+7", 1.0),
                (H1_AX, 1.0),
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_1", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=3.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
            scope=_scope,
            dominates_at={
                "OUTPUT_HI+0": _scope,
            },
        ),
    )


def _l8_alu_add_mark_ax_mirror(rules: tuple[FFNRule, ...]) -> tuple[FFNRule, ...]:
    """Fold the legacy ``_set_layer8_alu`` re-bake into the declarative rules.

    Until the 2026-06-12 L8 ALU cut, ``layer8_alu`` baked these rules
    (Wave-B ``MARK_SE_ONLY`` variant) at phase 8.2, and then the
    ``layer8_multibyte_routing`` bake re-ran the imperative
    ``vm_step._set_layer8_alu`` helper (the un-migrated ``MARK_AX``
    variant) at phase 8.3 via ``=`` assignment. The two passes
    SUPERPOSE: ``_set_layer8_alu`` only writes ``W_up[MARK_AX]`` /
    ``W_gate[MARK_AX]`` (it never touches ``MARK_SE_ONLY``), so the LIVE
    weights carry BOTH markers on every ALU unit -- the Wave-B marker
    migration was masked and never actually took effect in production.

    This transform makes the declared rules MATCH that live reality so
    the imperative re-bake can be removed (closing the last live
    imperative weight write): for every rule that does NOT already carry
    an explicit ``MARK_AX`` term, mirror each ``MARK_SE_ONLY`` term
    (condition AND gate) into an equal-weight ``MARK_AX`` term. Rules
    that already name ``MARK_AX`` (the LEV bytes-1/2 ``-1e6`` hard
    blocker) are left untouched -- their ``MARK_AX`` blocker survived the
    imperative ``=`` pass (which only wrote ``MARK_BP`` there), so the
    LEV factories already encode their folded form directly.

    Byte-identity vs the legacy two-pass bake is proven by
    ``tools/verify_l8_alu_migration.py``.
    """
    out: list[FFNRule] = []
    for rule in rules:
        has_ax = any(t.dim.name == "MARK_AX" for t in rule.conditions)
        if has_ax:
            out.append(rule)
            continue
        new_conditions = list(rule.conditions)
        for term in rule.conditions:
            if term.dim.name == "MARK_SE_ONLY":
                new_conditions.append(ConditionTerm(
                    dim=DimRef(name="MARK_AX", offset=term.dim.offset),
                    weight=term.weight,
                ))
        new_gate_terms = list(rule.gate_terms)
        if rule.gate is not None and rule.gate.name == "MARK_SE_ONLY":
            new_gate_terms.append(ConditionTerm(
                dim=DimRef(name="MARK_AX", offset=rule.gate.offset),
                weight=rule.gate_weight,
            ))
        for term in rule.gate_terms:
            if term.dim.name == "MARK_SE_ONLY":
                new_gate_terms.append(ConditionTerm(
                    dim=DimRef(name="MARK_AX", offset=term.dim.offset),
                    weight=term.weight,
                ))
        out.append(dataclasses.replace(
            rule,
            conditions=tuple(new_conditions),
            gate_terms=tuple(new_gate_terms),
        ))
    return tuple(out)


def _layer8_alu_rules(S: float) -> tuple[FFNRule, ...]:
    """Full ordered ``FFNRule`` sequence for ``layer8_alu``.

    Concatenates the 18 sub-stage rule tuples in cursor order, then folds
    the (now-removed) legacy ``_set_layer8_alu`` re-bake in via
    ``_l8_alu_add_mark_ax_mirror`` so the composite list lowers
    byte-identically against the live two-pass production weights. Total:
    2023 rules covering offsets 0..2022.
    """
    rules = (
        _layer8_alu_add_lo_rules(S)
        + _layer8_alu_lea_lo_rules(S)
        + _layer8_alu_sub_lo_rules(S)
        + _layer8_alu_add_carry_rules(S)
        + _layer8_alu_lea_carry_rules(S)
        + _layer8_alu_adj_lo_rules(S)
        + _layer8_alu_adj_carry_rules(S)
        + _layer8_alu_sub_borrow_rules(S)
        + _layer8_alu_ent_lo_rules(S)
        + _layer8_alu_ent_borrow_rules(S)
        + _layer8_alu_cmp_group_rules(S)
        + _layer8_alu_cmp_clear_rules(S)
        + _layer8_alu_ent_adj_defaults_rules(S)
        + _layer8_alu_lev_byte0_lo_rules(S)
        + _layer8_alu_lev_byte0_hi_rules(S)
        + _layer8_alu_lev_b1_rules(S)
        + _layer8_alu_lev_b2_rules(S)
        + _layer8_alu_lea_axb2_rules(S)
    )
    return _l8_alu_add_mark_ax_mirror(rules)


def _layer8_alu_ir(S: float = 100.0) -> CompilerIR:
    """Build the L8 ALU CompilerIR (single FFN op, 2023 rules).

    Exposes ``layer8_alu``'s full declarative spec as
    ``Operation.compiler_ir`` so the verifier, scope checker, and
    dominance auditor can read the per-rule semantics. The bake path
    drives the imperative pin via ``lower_layer8_alu_ir`` so the
    weights land at units 0..2022, exactly where the legacy
    ``_set_layer8_alu`` helper used to write.
    """

    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer8_alu_rules(S))
    return ir


def lower_layer8_alu_ir(
    ffn,
    S: float,
    BD,
    *,
    start_unit: int = 0,
) -> int:
    """Lower L8 ALU rules into ``ffn`` and return the next unit cursor.

    The companion to ``lower_layer8_multibyte_routing_ir``: the bake
    path calls this from ``make_layer8_alu_op``'s ``bake_fn`` so the
    weights land at the same offsets the imperative helper used to
    write. Symbolic / verifier tooling reads the same rule list via
    ``_layer8_alu_ir`` (attached as ``compiler_ir`` on the Operation).
    """

    rules = _layer8_alu_rules(S)
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


def make_layer8_alu_op() -> Operation:
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + LEA + CMP_GROUP.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.2, migrated=True. The inline
    call ``_set_layer8_alu(ffn8, S, BD)`` in ``set_vm_weights`` (both
    ``alu_mode == 'lookup'`` and ``alu_mode == 'efficient'`` branches)
    has been removed; this op now owns the bake. Phase=8.2 places it
    after format_pointer_extraction (7.5) and the L8 multibyte_fetch
    bake (8.1), and before format_position_counter (8.5) — matching the
    legacy in-set_vm_weights ordering.

    MIGRATED 2026 Phase 6 Wave 4F: the imperative ``_set_layer8_alu``
    bake is replaced with a declarative ``FFNRule`` list lowered via
    ``Primitives.lower_ffn_rules`` (see ``lower_layer8_alu_ir`` and the
    18 ``_layer8_alu_<substage>_rules`` factories). ``compiler_ir``
    exposes the rule spec to verifier / scope / dominance tooling.

    MIGRATED 2026-06-12 (L8 ALU live-imperative cut): the
    ``layer8_multibyte_routing`` bake used to re-invoke the imperative
    ``vm_step._set_layer8_alu`` helper over units 0..2022 (the LAST live
    imperative weight write in the production model). That ``MARK_AX``
    re-bake superposed onto these declarative ``MARK_SE_ONLY`` rules via
    ``=`` assignment, so the live weights carry BOTH markers. The
    superposition is now folded directly into ``_layer8_alu_rules`` (see
    ``_l8_alu_add_mark_ax_mirror`` + the LEV bytes-1/2 ``MARK_BP`` edit)
    and the imperative re-bake is removed; the ``vm_step`` helper stays
    defined only for the byte-identity verifier + legacy unit tests.
    Byte-identity proof: ``tools/verify_l8_alu_migration.py``.
    """
    def bake(block, dim_positions, S):
        # Per-bake FFN-unit allocator. ``layer8_alu`` claims the whole
        # 0..2023 cluster via its sub-stage rows; the declarative
        # lowerer's start_unit=0 cursor walks that range byte-identically.
        # The allocator is the structured manifest of those offsets so a
        # future op claiming a free L8 gap goes through
        # ``allocator.alloc(...)`` instead of hand-picking another
        # offset. Stash on ``block.ffn`` (mirrors the
        # ``_l9_unit_allocator`` convention) so downstream tooling can
        # inspect the layout.
        allocator = _allocate_layer8_ffn_units()
        block.ffn._l8_unit_allocator = allocator

        proxy = _as_setdim_proxy(dim_positions)
        n8 = lower_layer8_alu_ir(block.ffn, S, proxy, start_unit=0)
        # Byte-identity guard: the declarative lowerer's cursor must
        # end exactly where the next L8 FFN op
        # (``layer8_multibyte_routing``) is pinned. If the rule list
        # drifts from the table the assertion fires before any weight
        # surgery happens.
        expected_end = _l8_ffn_range_start(allocator, "layer8_multibyte_routing")
        assert n8 == expected_end, (
            f"L8 ALU unit cursor drift: rule lowering returned {n8}, "
            f"allocator expected {expected_end}"
        )

    return Operation(
        name="layer8_alu",
        # Phase 9.B (ALU_LO SCC rename): ALU_LO -> ALU_LO.*.-1 marks the
        # read as SSA cross-step. L10 stack0_byte_relay{,_bake} (phase 10/10.4)
        # and L16 lev_routing (phase 16) stage ALU_LO for the NEXT step's
        # L8 consumption; same-step fresh ALU_LO at AX_byte0 is still
        # observed via consumes_fresh declared below. Same numeric slot
        # via SSA alias; byte-identical bake. Breaks the 3 L10/L16 -> L8
        # ALU_LO back-edges inside SCC #1.
        reads={"MARK_AX", "MARK_PC", "ALU_LO.*.-1", "AX_CARRY_LO", "FETCH_LO",
               "OP_ADD", "OP_SUB", "OP_LEA",
               "OP_IMM",  # ent_lo/ent_borrow per-step actual-IMM blocker
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               # V2/G7 LEV detector: in-step topology edge from
               # lev_detector_head (phase=8.06) replaces the cross-step
               # requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        # UNDECLARED_DIM_AUDIT_2026_06_09: added 8 undeclared writes
        # (ADDR_B0_HI, ADDR_B0_LO, ADDR_B1_LO, ADDR_B2_LO, ALU_HI,
        # ALU_LO, CMP, OUTPUT_HI) so the dep graph sees this op's
        # full writer set. Declaration-only; byte-identical.
        writes={"OUTPUT_LO", "CARRY", "CMP_GROUP",
                "ADDR_B0_HI", "ADDR_B0_LO", "ADDR_B1_LO", "ADDR_B2_LO",
                "ALU_HI", "ALU_LO", "CMP", "OUTPUT_HI"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=_layer8_alu_ir(),
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # The L8 lookup ALU consumes the *current step's* AX value via
        # AX_CARRY_LO at the AX marker (operand 1 for ADD/SUB/LEA at the
        # AX byte 0 position). Without an in-step producer, AX_CARRY_LO
        # carries the prev-prev step's value (the stale-AX_CARRY bug
        # observed in the IMM 10 / PSH / IMM 32 / ADD sequence, fixed by
        # the L8 head 6 AX_CARRY refresh in commit 3d1b700). The
        # ``layer8_head6_ax_carry_refresh`` op (phase=8.05) is the
        # canonical in-step producer.
        # Phase 9.D: ALU_LO cycle-graph constraint satisfied by the
        # PC_VIA_LEV_DETECTOR_LO read above (lev_detector_head phase=8.06
        # is in-step producer). Previous: requires={"after":
        # "layer16_lev_routing"}. See CONTROL_FLOW_DETECTOR_HEADS.md §2.4.
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        compaction_safe=True,
    )


def make_format_position_counter_op(enable_conversational_io: bool = False) -> Operation:
    """L8 FFN: increment IO_FORMAT_POS after each output byte emission.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_position_counter(ffn8, S, BD)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=8`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was a side-effect of convo-io initially being co-located with
    the lookup ALU bake — the helper itself writes IO_FORMAT_POS units
    (starting at unit 600) and has no dependency on lookup-mode-specific
    weights, so this op fires regardless of alu_mode whenever the flag
    is set. Phase=8.5 so this runs AFTER ``layer8_alu`` (phase=8) since
    the position-counter units (600-615) intentionally overwrite a slice
    of the ADD-carry block, matching legacy behavior.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        # Per-bake allocator with the convo-io alias registered. The
        # alias overlaps ``layer8_alu.sub_lo`` at unit 600 by design
        # (the position-counter units intentionally overwrite that
        # slice). ``allow_overlap=True`` makes the layout auditable
        # without changing any baked weight.
        allocator = _allocate_layer8_ffn_units(
            include_format_position_counter=True
        )
        block.ffn._l8_unit_allocator = allocator
        start_unit = _l8_ffn_range_start(allocator, "format_position_counter")
        _lower_format_position_counter_ir(
            block.ffn,
            S,
            _as_setdim_proxy(dim_positions),
            start_unit=start_unit,
        )

    return Operation(
        name="format_position_counter",
        # Phase 11.A r3: dropped phase=8.5 — target_op_name +
        # requires['after']: layer8_alu already pin placement at
        # layer10_byte_passthrough and intra-target order.
        # Phase 9.B (IO_IN_OUTPUT_MODE SCC rename): IO_IN_OUTPUT_MODE ->
        # IO_IN_OUTPUT_MODE.*.-1 marks the read as SSA cross-step. The
        # sole writer ``null_terminator_detection`` runs at phase=10.6
        # (AFTER this op at phase=8.5), so the read sees the prev-step
        # residual. Same numeric slot via SSA alias.
        reads={"LAST_WAS_BYTE", "IO_IN_OUTPUT_MODE.*.-1", "IO_FORMAT_POS"},
        writes={"IO_FORMAT_POS"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_format_position_counter_ir(),
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        requires={"after": "layer8_alu"},
        migrated=True,
        # Wave 2 (docs/PRODUCES_CONSUMES_MIGRATION.md). Derived via
        # ``tools/derive_produces_consumes.py``. 16 rules write
        # IO_FORMAT_POS at byte k -> byte (k+1)%16 (a rotating counter)
        # gated by LAST_WAS_BYTE + IO_IN_OUTPUT_MODE.*.-1; the gate is
        # IO_FORMAT_POS itself (self-loop: read this position k, write
        # the rotated value). Slot tag "IO_FORMAT_POS" names the
        # dedicated counter register.
        # consumes_fresh notes:
        #   - LAST_WAS_BYTE: same-step writer (null_terminator_detection
        #     family at phase 10.6 — but that's LATER than this op at
        #     phase 8.5; the analyzer's same-step-writer warning is
        #     architecturally correct and tracked by the step-0 safety
        #     pass).
        #   - IO_FORMAT_POS: self-feedback (gate); no earlier-phase
        #     writer in the same step. Declaring it here flags the
        #     self-loop for the multistep verifier (the value carries
        #     forward across steps; the in-step read sees prev-step
        #     output).
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_format_position_counter_ir(S: float = 100.0) -> CompilerIR:
    """Declarative IR mirror of ``_lower_format_position_counter_ir``.

    Exposes the same 16-rule SwiGLU counter as ``Operation.compiler_ir`` so
    the verifier, scope checker, and dominance auditor can read the spec.
    The bake path still drives the imperative pin (unit 600) via
    ``_lower_format_position_counter_ir``; this IR is the semantic source of
    truth for symbolic tooling and is byte-identity-validated by
    ``compare_symbolic_to_lowered_ffn`` (start_unit=0 in the validator, no
    consumer-facing weights are baked through this path).
    """
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_format_position_counter_rules(S))
    return ir


def _format_position_counter_rules(S: float) -> tuple[FFNRule, ...]:
    rules = []
    write_scale = 2.0 / S
    conditions = (
        ("LAST_WAS_BYTE", 1.0),
        ("IO_IN_OUTPUT_MODE", 1.0),
    )
    for k in range(16):
        next_k = (k + 1) % 16
        rules.append(multi_way_and_rule(
            name=f"format_pos_inc_{k}",
            conditions=conditions,
            threshold=1.5,
            gate=f"IO_FORMAT_POS+{k}",
            writes=(
                (f"IO_FORMAT_POS+{k}", -write_scale),
                (f"IO_FORMAT_POS+{next_k}", write_scale),
            ),
        ))
    return tuple(rules)


def _lower_format_position_counter_ir(
    ffn, S: float, BD, *, start_unit: int = _L8_FFN_FORMAT_POS_COUNTER_PIN
) -> int:
    rules = _format_position_counter_rules(S)
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


def make_layer8_multibyte_fetch_op() -> Operation:
    """No-op dep anchor for ``layer8_multibyte_fetch_bake``.

    Kept as a ``kind="attn"`` placeholder with explicit declarative
    authority metadata: its declared reads/writes preserve the LayerCompiler
    dep-graph topology that places downstream ops at the right
    model.blocks indices. The actual weight bake now happens in
    ``make_layer8_multibyte_fetch_bake_op`` (kind="block", layer_idx=8,
    phase=8.1, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_multibyte_fetch_bake` block op.
        return

    return Operation(
        name="layer8_multibyte_fetch",
        # Declaration audit (2026-06-05): added HAS_SE, MARK_AX, CONST to
        # mirror the Q/K read set of _layer8_multibyte_fetch_head_spec
        # (the head spec lowered by the paired ``layer8_multibyte_fetch_bake``
        # block op uses HAS_SE at slot TOP, MARK_AX as a K-side exclusion at
        # slot 35, and CONST as the slot 33/34 K seed + slot 35 Q sink).
        # Anchor reads gate dim-lifetime analysis; the bake op's block kind
        # filters it out of the same analysis.
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "HAS_SE", "MARK_AX", "CONST",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 8.A targeted (SCC audit step 7): the Q-side reads
        # ADDR_KEY+32..47 at AX byte positions; those slots are written
        # same-step by ``layer4_pc_relay`` head 1. Anchoring the dep
        # explicitly lets R-OH-2 suppress the L14 ADDR_KEY-writer back
        # edges into this op.
        requires={"after": "layer4_pc_relay"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
    )


def make_layer8_multibyte_fetch_bake_op() -> Operation:
    """Bake ``_set_layer8_multibyte_fetch`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_multibyte_fetch(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.1 so this runs after ``layer8_sp_gather_bake`` (phase=8.0),
    preserving the legacy in-set_vm_weights ordering. The dep anchor
    ``layer8_multibyte_fetch`` (kind="attn") preserves the LayerCompiler
    topology so downstream ops remain placed at their legacy blocks.

    Post-bake adds a K-side MARK_AX exclusion (dim 35) so the head does
    not score the AX marker as a viable K candidate. Required because the
    L4 SP-to-ADDR_KEY op (when enabled) stages scale-10 ADDR_KEY content
    at the AX marker, which would otherwise outscore the scale-1 ADDR_KEY
    content at MEM val byte positions and cause multibyte_fetch to attend
    to the AX marker instead of the correct code byte. Safe to apply
    unconditionally — when L4 SP-staging is off the AX marker carries no
    ADDR_KEY content and the gate is a no-op.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator. Head 3 is pinned at its
        # existing slot so ``generate_attention_head`` writes the same
        # ``head_idx * HD + slot`` rows it always has. Stashed on the
        # attention module so a future L8 attn op claiming a free slot
        # can inspect the layout. See :data:`_L8_HEAD_LAYOUT`.
        attn._l8_head_allocator = _allocate_layer8_attn_heads(
            ("layer8_multibyte_fetch_bake.head_3",)
        )
        Primitives.generate_attention_head(
            attn, _layer8_multibyte_fetch_head_spec(proxy), HD
        )

    # Dim-ownership claims: L8 attn head 3 multibyte fetch.
    #   W_v[3*HD + 32 + k, CLEAN_EMBED_LO + k]  for k=0..15 (slot 32..47)
    #   W_v[3*HD + 48 + k, CLEAN_EMBED_HI + k]  for k=0..15 (slot 48..63)
    _claims = set()
    for k in range(16):
        _claims.add((8, "attn_W_v", f"3_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((8, "attn_W_v", f"3_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer8_multibyte_fetch_bake",
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1", "HAS_SE",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST", "MARK_AX"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer8_multibyte_fetch_ir,
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        claims=_claims,
        # Phase 8.A targeted (SCC audit step 7): mirror the
        # ``layer8_multibyte_fetch`` dep anchor's ``requires["after"]`` so
        # the same R-OH-2 suppression applies to the real baked op. See
        # the anchor's comment for the same-step ADDR_KEY rationale.
        requires={"after": "layer4_pc_relay"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
    )


def _layer8_multibyte_fetch_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer8_multibyte_fetch_head_spec(BD))
    return ir


def _layer8_multibyte_fetch_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative replacement for ``vm_step._set_layer8_multibyte_fetch``."""

    L = 20.0
    # Class-1 marker-relative anchor: ``H1+AX_I`` reads "nearest marker is the
    # AX register" via the L1 threshold-head distance bank. The ``+AX_I`` is the
    # marker-TYPE bank slot (frame-invariant), not a token distance —
    # ``marker_bank_index`` resolves it from Token.STEP_TOKENS (byte-identical to
    # the literal 1 at STEP_TOKENS=35) so the audit recognises the ref as
    # declared-invariant rather than an UNGUARDED bare offset.
    AX_I = marker_bank_index("AX")
    TOP = 36
    # Removal-1 (2026-06-06): Q-side MARK_AX gate + K-side HAS_SE blocker
    # on slot 52. Per docs/REMOVAL_1_REAL_SURFACE_2026_06_06.md Option A,
    # this is intended to disambiguate the IMM byte at PC+1 from the
    # opcode byte at PC+0 for high-bit IMM values where the ADDR_KEY
    # nibble pattern aliased. Empirically (smoke 2026-06-06): this gate
    # alone does not recover the override-removed regression because
    # HAS_SE is set on every bytecode row once a STEP_END exists, not
    # row-specific to opcode bytes; the K-side blocker therefore
    # penalises the IMM byte at PC+1 just as much as the opcode byte at
    # PC+0. The gate stays as a no-op structural improvement on top of
    # the existing ADDR_KEY+ FETCH discriminator; the override at
    # batched_pure_neural.py b5cf7099 remains the load-bearing path
    # pending Option B (an L9/L10 position-addressed corrective rule
    # that writes AX_CARRY_LO/HI from CLEAN_EMBED at PC+1 directly).
    MARK_GATE = 52
    return DeclarativeAttentionHeadSpec(
        head_idx=_L8_HEAD_LAYOUT_BY_NAME["layer8_multibyte_fetch_bake.head_3"],
        q=(
            tuple(AP(k, BD.FETCH_LO + k, L) for k in range(16))
            + tuple(AP(16 + k, BD.FETCH_HI + k, L) for k in range(16))
            + tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, L) for k in range(16))
            + (
                AP(32, BD.IS_BYTE, L),
                AP(33, BD.IS_BYTE, 500.0),
                AP(33, BD.CONST, -500.0),
                AP(34, BD.H1 + AX_I, 500.0),
                AP(34, BD.CONST, -500.0),
                # AX register K exclusion added after the original helper.
                # The byte-position queries can carry staged ADDR_KEY, so
                # without blocking AX byte K candidates they self-attend to
                # the old AX byte value and pollute AX_CARRY before routing.
                AP(35, BD.H1 + AX_I, 100.0),
                AP(35, BD.IS_BYTE, 100.0),
                AP(35, BD.CONST, -150.0),
                AP(TOP, BD.CONST, L),
                AP(TOP, BD.HAS_SE, -L),
                # Q-side MARK_AX gate (slot 52): only AX-marker query
                # positions activate the K-side HAS_SE penalty below.
                AP(MARK_GATE, BD.MARK_AX, L),
            )
        ),
        k=(
            tuple(AP(k, BD.ADDR_KEY + k, L) for k in range(16))
            + tuple(AP(16 + k, BD.ADDR_KEY + 16 + k, L) for k in range(16))
            + tuple(AP(TOP + k, BD.ADDR_KEY + 32 + k, L) for k in range(16))
            + (
                AP(33, BD.CONST, 5.0),
                AP(34, BD.CONST, 5.0),
                AP(35, BD.MARK_AX, -50.0),
                AP(35, BD.H1 + AX_I, -50.0),
                # K-side HAS_SE blocker (slot 52): paired with Q-side
                # MARK_AX gate; subtracts -2L^2 from any key position
                # carrying HAS_SE when the query is at the AX marker.
                AP(MARK_GATE, BD.HAS_SE, -L * 2.0),
            )
        ),
        v=(
            _band_projection_writes(32, BD.CLEAN_EMBED_LO, 3.0)
            + _band_projection_writes(48, BD.CLEAN_EMBED_HI, 3.0)
        ),
        o=(
            _band_output_writes(BD.AX_CARRY_LO, 32)
            + _band_output_writes(BD.AX_CARRY_HI, 48)
        ),
    )


def make_layer8_multibyte_routing_op() -> Operation:
    """L8 FFN extension: route FETCH → OUTPUT at AX byte positions for IMM.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.3, migrated=True. The inline
    call ``_set_layer8_multibyte_routing(ffn8, S, BD)`` in
    ``set_vm_weights`` (both alu_mode branches) has been removed; this op
    now owns the bake. Phase=8.3 places it after ``layer8_alu`` (8.2)
    so the shared unit counter starts after the ALU units.

    MIGRATED 2026-06-12 (L8 ALU live-imperative cut): the bake no longer
    re-invokes the imperative ``vm_step._set_layer8_alu`` helper. That
    re-call was the LAST live imperative weight write in the production
    bake -- it superposed the legacy ``MARK_AX`` ALU weights onto the
    phase-8.2 declarative ``MARK_SE_ONLY`` weights via ``=`` assignment.
    The superposition is now folded into ``layer8_alu``'s declarative
    rules (see ``_l8_alu_add_mark_ax_mirror``), so phase 8.2 already
    produces the both-markers production state. This bake only needs the
    allocator-pinned cursor (offset 2023) -- the source of truth for the
    multibyte-routing start -- and appends its own 32 routing units.
    Byte-identity proof: ``tools/verify_l8_alu_migration.py``.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake allocator. ``layer8_alu`` (phase 8.2) already produced
        # units 0..2022 (now including the folded ``MARK_AX`` ALU weights);
        # the allocator-pinned offset 2023 is the cursor source of truth.
        allocator = _allocate_layer8_ffn_units()
        block.ffn._l8_unit_allocator = allocator
        expected_start = _l8_ffn_range_start(
            allocator, "layer8_multibyte_routing"
        )
        lower_layer8_multibyte_routing_ir(
            block.ffn,
            S,
            proxy,
            start_unit=expected_start,
        )

    return Operation(
        name="layer8_multibyte_routing",
        # Phase 1 (memory cluster fix plan): shares L8 FFN unit range with
        # ``layer8_sp_gathered_sentinel`` at a disjoint sub-range (this op
        # owns units 0..2054; the sentinel owns one trailing unit at 2055).
        # See docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn_units",),
        reads={"IS_BYTE", "H1", "OP_IMM", "MARK_AX",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_layer8_multibyte_routing_ir(),
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # This op produces the fresh AX-byte-0 OUTPUT for IMM (routes
        # AX_CARRY -> OUTPUT at AX byte positions). The L6 routing FFN
        # produces AX_byte0 OUTPUT for the other AX-emitting opcodes; this
        # L8 FFN extension covers the IMM multi-byte path.
        # ``_set_layer8_multibyte_routing`` re-invokes ``_set_layer8_alu``
        # internally to recover the ALU-final unit cursor (~2023) and then
        # appends 32 multibyte-IMM routing units, reaching unit 2054 — so
        # the L8 FFN needs 2055 hidden units total. Sibling ``layer8_alu``
        # (phase=8.2) writes the 0-2022 cluster; this op holds the
        # per-layer max and dominates the dynamic-FFN allocation for L8.
        ffn_units_used=2055,
        # Tier A opcode gating: every one of the 32 multibyte-IMM routing
        # units writes ``W_up[unit, BD.OP_IMM] = S`` (see
        # ``_set_layer8_multibyte_routing`` at vm_step.py:5343). The units
        # fire only at AX byte positions when OP_IMM is the active opcode
        # (relayed by L8 head 4 to byte positions); non-IMM opcodes leave
        # the multibyte routing FFN silent.
        opcodes={"OP_IMM"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#mixture-of-experts-routing",
    )


def make_layer8_multibyte_routing_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer8_multibyte_routing_rules(S))
    return ir


def _layer8_multibyte_routing_rules(S: float) -> tuple[FFNRule, ...]:
    """Declarative L8 multibyte IMM routing extension after the ALU units."""

    rules = []
    # Class-1 marker-relative anchor: ``H1+AX_I`` keys on the AX-register
    # threshold-bank slot (frame-invariant; ``marker_bank_index`` resolves it
    # from Token.STEP_TOKENS, byte-identical to the literal 1 at STEP_TOKENS=35).
    AX_I = marker_bank_index("AX")
    conditions = (
        ("IS_BYTE", 1.0),
        (f"H1+{AX_I}", 1.0),
        ("OP_IMM", 1.0),
        ("MARK_AX", -4.0),
    )
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l8_multibyte_route_lo_{k}",
            conditions=conditions,
            threshold=6.5,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 8.0 / S),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"l8_multibyte_route_hi_{k}",
            conditions=conditions,
            threshold=6.5,
            gate=f"AX_CARRY_HI+{k}",
            writes=((f"OUTPUT_HI_THIS_STEP+{k}", 8.0 / S),),
        ))
    return tuple(rules)


def lower_layer8_multibyte_routing_ir(
    ffn,
    S: float,
    BD,
    *,
    start_unit: int,
) -> int:
    """Lower L8 multibyte IMM routing rules and return the next unit."""

    rules = _layer8_multibyte_routing_rules(S)
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


def make_layer8_sp_gather_op() -> Operation:
    """No-op dep anchor for ``layer8_sp_gather_bake``.

    Kept as a ``kind="attn"`` placeholder with explicit declarative
    authority metadata: its declared reads/writes preserve the LayerCompiler
    dep-graph topology. The actual weight bake now happens in
    ``make_layer8_sp_gather_bake_op`` (kind="block", layer_idx=8,
    phase=8.0, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_sp_gather_bake` block op.
        return

    return Operation(
        name="layer8_sp_gather",
        # Declaration audit (2026-06-05): added the ADDR_B0/B1/B2_LO/HI
        # writes the paired ``layer8_sp_gather_bake`` block op actually
        # produces via _layer8_sp_gather_head_specs (O-side writes the
        # three address-byte bands). Also expanded the reads to include
        # the marker / hop / byte-index / CLEAN_EMBED / CONST dims that
        # the head specs gate on. ALU_LO/HI writes are kept defensively
        # to preserve any downstream scheduling dep that referenced this
        # anchor as an ALU_LO/HI writer; the bake itself does NOT write
        # ALU_LO/HI -- that is owned by sibling ops (layer7_operand_gather,
        # layer10_stack0_byte_relay_bake, etc.).
        # The bake op's CMP.*.-1 cross-step read is NOT mirrored here
        # because at kind="attn" placement L8 it has same-step CMP writers
        # at L6 and trips the cross-step baseline allowlist gate; the bake
        # op's block kind carries the cross-step CMP semantics instead.
        reads={"MARK_AX", "MARK_STACK0", "MARK_SP", "MARK_BP",
               "H1", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST",
               "OP_ADJ", "OP_ENT", "OP_LEA", "EMBED_LO", "EMBED_HI"},
        writes={"ALU_LO", "ALU_HI",
                "ADDR_B0_LO", "ADDR_B0_HI",
                "ADDR_B1_LO", "ADDR_B1_HI",
                "ADDR_B2_LO", "ADDR_B2_HI"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer8_ffn_dep_anchor_op() -> Operation:
    """No-op FFN dep anchor for L8 with ``ffn_units_used=0``.

    Per ``docs/DEAD_UNIT_AUDIT_2026_06_05.md``: L8 is attention-routing-
    heavy and has NO FFN bake at the primary ``block[L8].ffn``. The
    actual ``layer8_multibyte_routing`` FFN work lives at L9 (via
    ``target_op_name="layer10_byte_passthrough"``) and the primary L8
    block FFN is 100% dead (4096 / 4096 = 100.0%) at the historical
    ``DEFAULT_LAYER_MAX_UNITS=4096`` budget.

    Declaring this kind="ffn" anchor with ``ffn_units_used=0`` lets the
    dynamic-FFN allocator pre-size ``block[L8].ffn.hidden_dim=0``
    instead of allocating 4096 dead rows. The anchor pins to the same
    layer as ``layer8_sp_gather`` via ``requires={"same_layer_as":
    "layer8_sp_gather"}`` so it lands at L8 in the dep-graph layout.

    Savings: 4096 * (2*d_model + 1) = ~6.55M params pre-rightsize.
    """
    def bake(ffn, dim_positions, S):
        return None

    return Operation(
        name="_layer8_ffn_dep_anchor",
        reads=set(),
        writes=set(),
        kind="ffn",
        migrated=True,
        declarative_authority="topology_anchor",
        compiler_ir=CompilerIR(),
        # Place at the same layer as ``layer8_sp_gather`` (the L8 attn
        # anchor) so this FFN anchor lands at L8.
        requires={"same_layer_as": "layer8_sp_gather"},
        smoke_tests=set(),
        spec_section=None,
        # Dead-unit budget: zero out the L8 primary FFN footprint.
        ffn_units_used=0,
    )


def make_layer8_sp_gather_bake_op() -> Operation:
    """Bake ``_set_layer8_sp_gather`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_sp_gather(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.0 so this runs before ``layer8_multibyte_fetch_bake``
    (phase=8.1), preserving the legacy in-set_vm_weights ordering. The
    dep anchor ``layer8_sp_gather`` (kind="attn") preserves the
    LayerCompiler topology so downstream ops remain placed at their
    legacy blocks.
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator. Heads 0-2 (j=0..2 main
        # gather rows) and the head 6/7 MARK_SP mirrors are pinned at
        # their existing slots so ``generate_attention_heads`` writes
        # the same ``head_idx * HD + slot`` rows it always has. Stashed
        # on the attention module so downstream tooling can inspect the
        # layout. See :data:`_L8_HEAD_LAYOUT`.
        attn._l8_head_allocator = _allocate_layer8_attn_heads((
            "layer8_sp_gather_bake.head_0",
            "layer8_sp_gather_bake.head_1",
            "layer8_sp_gather_bake.head_2",
            "layer8_sp_gather_bake.head_6_mark_sp_mirror",
            "layer8_sp_gather_bake.head_7_mark_sp_mirror",
        ))
        Primitives.generate_attention_heads(
            attn, _layer8_sp_gather_head_specs(proxy), HD
        )

    # Dim-ownership claims: L8 attn heads 0-2 SP gather (SP bytes → ADDR_B*).
    # Each head writes V slots 1..32 reading CLEAN_EMBED_LO/HI.
    # D3: heads 6/7 are dedicated MARK_SP mirrors of j=0/j=1 — they write
    # the same V band (CLEAN_EMBED_LO/HI → slots 1..32) and output to
    # ADDR_B0_*/ADDR_B1_* respectively. Claims declared here so the
    # registry sees the head 6/7 V ownership.
    _claims = set()
    for h in (0, 1, 2, 6, 7):
        for k in range(16):
            _claims.add((8, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((8, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer8_sp_gather_bake",
        # Phase 8.A targeted: cross-step CMP+3 read (STACK0-suppression
        # gate on the SP-gather Q rows) declared as CMP_PREV_STEP. L9
        # ALU writes CMP in the same step but AFTER L8; the gate
        # semantics use the previous step's CMP (the BZ/BNZ branch
        # decision that was just committed). Renaming the read to the
        # PREV_STEP alias retires the data-flow edge L9_ALU -> L8 on
        # CMP from the dep graph; the explicit cross-step boundary is
        # captured by the ``requires["after"] = layer9_alu`` declaration
        # below (mirrors the L3 OUTPUT_LO/HI PREV_STEP pattern from
        # Phase 7.A.3.b). Same numeric position as CMP so baked weight
        # cells are byte-identical. See .agent-logs/scc_audit_phase8.md.
        reads={"MARK_STACK0", "MARK_SP", "MARK_BP", "H1", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CMP.*.-1", "CONST"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI",
                "ADDR_B1_LO", "ADDR_B1_HI",
                "ADDR_B2_LO", "ADDR_B2_HI"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer8_sp_gather_ir,
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        claims=_claims,
        # Phase 8.A targeted: explicit cross-step boundary. The
        # CMP_PREV_STEP read above is satisfied by L9 ALU's previous-
        # step write via the KV residual; ``requires["after"]`` records
        # that producer relationship declaratively so the strict-mode
        # admission gate sees a layer-pinning dep (without it the op
        # would fall into ``phase_required_but_undeclared`` because its
        # remaining in-step reads only reach depth=3, while phase=8 /
        # layer_idx=8 pin current_layer=8). Mirrors
        # ``layer3_carry_forward_attn``'s ``requires["after"] =
        # layer16_lev_routing`` pattern for the L3 OUTPUT_LO/HI
        # PREV_STEP rename (Phase 7.A.3.b). See
        # .agent-logs/scc_audit_phase8.md §5 wave 2.
        requires={"after": "layer9_alu"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer8_sp_gather_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer8_sp_gather_head_specs(BD))
    return ir


def _layer8_sp_gather_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``vm_step._set_layer8_sp_gather``."""

    L = 15.0
    # Class-1 marker-relative anchors: ``H<k>+<marker_I>`` indexes the
    # marker-TYPE slot in the fixed-width 7-slot threshold-head bank (PC=0 AX=1
    # SP=2 BP=3 MEM=4 SE=5), a structural position keyed on marker TYPE whose
    # order is identical in both the 35- and 30-token frames (STACK0 is a
    # transition target, never a bank slot) — frame-INVARIANT by construction.
    # ``marker_bank_index`` is the single source of truth (byte-identical at
    # STEP_TOKENS=35, unchanged at 30) so the audit recognises these as
    # declared-invariant rather than bare UNGUARDED offsets.
    AX_I = marker_bank_index("AX")
    SP_I = marker_bank_index("SP")
    BP_I = marker_bank_index("BP")
    MEM_I = marker_bank_index("MEM")

    specs: list[DeclarativeAttentionHeadSpec] = []
    _sp_gather_main_names = (
        "layer8_sp_gather_bake.head_0",
        "layer8_sp_gather_bake.head_1",
        "layer8_sp_gather_bake.head_2",
    )
    for j in range(3):
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        specs.append(
            DeclarativeAttentionHeadSpec(
                head_idx=_L8_HEAD_LAYOUT_BY_NAME[_sp_gather_main_names[j]],
                q=(
                    AP(0, BD.MARK_STACK0, L),
                    AP(0, BD.H4 + BP_I, L),
                    AP(0, BD.H1 + AX_I, -L),
                    AP(0, BD.H1 + SP_I, -L),
                    AP(0, BD.H3 + MEM_I, -L),
                    AP(0, BD.MARK_BP, -L),
                    AP(33, BD.MARK_STACK0, L),
                    AP(33, BD.CONST, -L / 2),
                ),
                k=(
                    AP(0, byte_idx_dim, L),
                    AP(0, BD.H1 + SP_I, L),
                    # Phase 9.C: read from CMP at the same numeric position
                    # (396+3=399). The cross-step semantics are captured by
                    # the SSA ``reads={... "CMP.*.-1" ...}`` + ``requires=
                    # {"after": "layer9_alu"}`` block in
                    # ``make_layer8_sp_gather_bake_op``; the PREV_STEP alias
                    # was retired now that SSA spellings own the dep-graph
                    # contract.
                    AP(0, BD.CMP + 3, -L),
                    AP(33, BD.CONST, L),
                ),
                v=(
                    _band_projection_writes(1, BD.CLEAN_EMBED_LO)
                    + _band_projection_writes(17, BD.CLEAN_EMBED_HI)
                    + _addr_mag_boost_v(BD)
                ),
                o=(
                    _band_output_writes(addr_lo_out, 1)
                    + _band_output_writes(addr_hi_out, 17)
                    + _addr_mag_boost_o(addr_lo_out)
                    + _addr_mag_boost_o(addr_hi_out)
                ),
            )
        )

    # D3: dedicated MARK_SP mirror heads. The original B7-3 fix added
    # ``MARK_SP`` firing to heads 0-2 so ADDR_B0/B1 would carry a fresh
    # in-step SP-derived address at MARK_SP rows (feeding IN_STEP_FRESH /
    # SP_BYTE0_IS_F8 / ADDR_B0_VALID consumers). That bled into L10/L16
    # consumers authored against the MARK_STACK0-only Q semantics,
    # doubling SP_byte0 fatals across the 1096 suite. D3 relocates the
    # MARK_SP firing onto dedicated heads 6 and 7 (mirrors of j=0/j=1
    # respectively) so the heads-0-2 Q semantics revert to MARK_STACK0
    # only. Heads 6 and 7 are otherwise reserved for the
    # ``make_layer8_head6_ax_carry_refresh_op`` and ``make_layer8_mem_to_alu_op``
    # bakes, both ``enable=False`` by default so the physical head slots
    # are free in the production build.
    _sp_gather_mirror_names = {
        0: "layer8_sp_gather_bake.head_6_mark_sp_mirror",
        1: "layer8_sp_gather_bake.head_7_mark_sp_mirror",
    }
    for mirror_j in (0, 1):
        mirror_name = _sp_gather_mirror_names[mirror_j]
        head_idx = _L8_HEAD_LAYOUT_BY_NAME[mirror_name]
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][mirror_j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][mirror_j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][mirror_j]
        specs.append(
            DeclarativeAttentionHeadSpec(
                head_idx=head_idx,
                q=(
                    AP(0, BD.MARK_SP, 2 * L),
                    AP(0, BD.H4 + BP_I, L),
                    AP(0, BD.H1 + AX_I, -L),
                    AP(0, BD.H1 + SP_I, -L),
                    AP(0, BD.H3 + MEM_I, -L),
                    AP(0, BD.MARK_BP, -L),
                    AP(33, BD.MARK_SP, L),
                    AP(33, BD.CONST, -L / 2),
                ),
                k=(
                    AP(0, byte_idx_dim, L),
                    AP(0, BD.H1 + SP_I, L),
                    # Phase 9.C: read from CMP at the same numeric position
                    # (396+3=399). Mirror of head 0-2 above; SSA
                    # ``"CMP.*.-1"`` + ``requires["after"]`` carries the
                    # cross-step semantics that the PREV_STEP alias used to
                    # express.
                    AP(0, BD.CMP + 3, -L),
                    AP(33, BD.CONST, L),
                ),
                v=(
                    _band_projection_writes(1, BD.CLEAN_EMBED_LO)
                    + _band_projection_writes(17, BD.CLEAN_EMBED_HI)
                    + _addr_mag_boost_v(BD)
                ),
                o=(
                    _band_output_writes(addr_lo_out, 1)
                    + _band_output_writes(addr_hi_out, 17)
                    + _addr_mag_boost_o(addr_lo_out)
                    + _addr_mag_boost_o(addr_hi_out)
                ),
            )
        )

    return tuple(specs)


def make_layer8_head6_ax_carry_refresh_op(enable: bool = False) -> Operation:
    """L8 attn head 6: refresh AX_CARRY_LO/HI from prev step's AX marker OUTPUT.

    Mirrors the head-6 bake added to ``UnifiedVMCompiler._compile_l8_attention``
    in commit ``3d1b700`` (2026-05-12, fix-phase2-ax-carry-refresh). The bake
    reads ``OUTPUT_LO/HI`` from the previous step's AX marker (excluding the
    *current* AX marker via anti-OP_* gates) and writes the result to the
    current step's ``AX_CARRY_LO/HI``.

    Status: ``enable=False`` by default — the production ``full_vm_compiler``
    path does not yet route this bake (today's head-6 fix lives only in the
    separate ``UnifiedVMCompiler`` path, see ``unified_compiler/compiler.py``).
    The Operation is still registered (always) so its ``produces`` annotation
    participates in the staleness-invariant scan. Setting ``enable=True``
    flips the bake on so this op owns the head-6 weight programming in the
    ``full_vm_compiler`` path; until that wiring is validated end-to-end the
    default stays off to keep the production build byte-identical.

    The ``produces`` declaration is the canonical proof-of-concept for the
    staleness analyzer (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md):
    removing this op from ``all_core_ops`` would leave the L8 ALU op (which
    declares ``consumes_fresh={"AX_CARRY_LO": "AX_byte0", ...}``) without an
    in-step producer, surfacing today's stale-AX_CARRY bug at compile time.
    """
    def _bake(block, dim_positions, S):
        if not enable:
            return
        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator. Head 6 is pinned at its
        # existing slot so the hand-rolled weight writes below land
        # byte-identically. Stashed on the attention module so
        # downstream tooling can inspect the layout. See
        # :data:`_L8_HEAD_LAYOUT`.
        attn._l8_head_allocator = _allocate_layer8_attn_heads(
            ("layer8_head6_ax_carry_refresh.head_6",)
        )
        head = _L8_HEAD_LAYOUT_BY_NAME["layer8_head6_ax_carry_refresh.head_6"]
        base = head * HD
        AX_CARRY_L = 50.0  # head-local Q/K scale
        # Q[base+0]: fire only at current step's AX marker on subsequent
        # steps (HAS_SE = 1). The CONST baseline blocks first-step fires.
        attn.W_q.data[base, BD.MARK_AX] = AX_CARRY_L
        attn.W_q.data[base, BD.HAS_SE] = AX_CARRY_L
        attn.W_q.data[base, BD.CONST] = -AX_CARRY_L * 1.5
        # K[base+0]: match AX marker at any past position.
        attn.W_k.data[base, BD.MARK_AX] = AX_CARRY_L
        # V copies OUTPUT_LO/HI from the matched AX marker.
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # O writes to AX_CARRY_LO/HI at the query position (current AX marker).
        for k in range(16):
            attn.W_o.data[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate (dim 33): suppress at non-AX-marker queries.
        GATE = 33
        attn.W_q.data[base + GATE, BD.MARK_AX] = AX_CARRY_L
        attn.W_q.data[base + GATE, BD.CONST] = -AX_CARRY_L / 2
        attn.W_k.data[base + GATE, BD.CONST] = AX_CARRY_L
        # Anti-op gates: exclude the current AX marker from K via anti-OP_*.
        anti_ops = [
            BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_JMP, BD.OP_JSR, BD.OP_LEV,
            BD.OP_BZ, BD.OP_BNZ, BD.OP_PSH, BD.OP_ADJ, BD.OP_ENT,
            BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
            BD.OP_AND, BD.OP_OR, BD.OP_XOR,
            BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
            BD.OP_SHL, BD.OP_SHR,
            BD.OP_LI, BD.OP_LC, BD.OP_LEA,
        ]
        ANTI_OP_SLOT_START = 34
        for j, op_dim in enumerate(anti_ops):
            slot = ANTI_OP_SLOT_START + j
            if slot >= HD:
                break
            attn.W_q.data[base + slot, op_dim] = -AX_CARRY_L
            attn.W_k.data[base + slot, op_dim] = AX_CARRY_L

    return Operation(
        name="layer8_head6_ax_carry_refresh",
        # Phase 8.05 places it between layer8_sp_gather_bake (8.0) and
        # layer8_multibyte_fetch_bake (8.1) so the AX_CARRY refresh
        # completes before any downstream L8 op that consumes_fresh
        # AX_CARRY_LO/HI fires. The exact phase number is not load-bearing
        # for the staleness analyzer (it only checks producer.phase <=
        # consumer.phase); 8.05 keeps the L8 attn bakes contiguous.
        # Phase 7.A.3.b: OUTPUT_LO read is cross-step (the V slots pull
        # the prev step's AX marker residual via attention back-edge).
        # Phase 8.A G7: OUTPUT_HI_THIS_STEP read renamed to
        # OUTPUT_HI_PREV_STEP. The V slots pull the prev step's AX marker
        # residual via attention back-edge -- not a same-step data dep on
        # any L8+/L14+/L16 OUTPUT_HI_THIS_STEP writer. The alias shares
        # numeric position 190 with OUTPUT_HI so bakes stay byte-identical.
        # This rename breaks 13 cross-step back-edges into this op. The
        # previous requires["after"]=layer16_lev_routing cycle-break is
        # no longer needed (the dim algebra now breaks the back-edge
        # natively) and has been removed -- keeping it would make the B9
        # EXCEPTION mis-fire (writes∩reads now empty), forcing L8 past L16.
        #
        # Phase 9 SSA prototype demo: the *_PREV_STEP aliases are
        # re-spelled in SSA form (``BASE.*.STEP_OFFSET``). The ``*``
        # writer wildcard reflects the multi-producer reality (any of
        # L8/L14/L16's OUTPUT_LO/HI writers may have written this AX
        # marker's residual in the previous step); ``-1`` is the
        # prev-VM-step offset. The LayerCompiler auto-declares each SSA
        # form as an alias of its base dim, so
        # ``dim_positions["OUTPUT_LO.*.-1"] == dim_positions["OUTPUT_LO"]``
        # -- byte-identical to the OUTPUT_LO_PREV_STEP form. The bake
        # (``enable=False`` by default) is unchanged. See ssa_dim.py and
        # docs/PHASE_9_SSA_PROTOTYPE.md.
        reads={"MARK_AX", "HAS_SE", "OUTPUT_LO.*.-1",
               "OUTPUT_HI.*.-1", "CONST",
               "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR", "OP_LEV",
               "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_AND", "OP_OR", "OP_XOR",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_SHL", "OP_SHR",
               "OP_LI", "OP_LC", "OP_LEA"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=_bake,
        declarative_bake_fn=_bake if not enable else None,
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        # Always migrated=True so the bake runs when enable=True; when
        # enable=False the bake body is a no-op so production behavior is
        # unchanged. The Operation itself stays in the registry either way
        # so the staleness analyzer can see its ``produces`` annotation.
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        # Phase 11.A IR exposure: bake is `if not <flag>: return` at default
        # config, so an empty IR is byte-identical for default flag values.
        # Populating IR with the matching rules is Phase 11.A follow-up.
        compiler_ir=CompilerIR(),
    )


def make_layer8_op_imm_relay_op() -> Operation:
    """L8 head 4: Relay OP_IMM from AX marker to AX byte positions.

    Migrated 2026-05-11 from the inline block at vm_step.py:2029-2043 that
    programmed attn8 head 4's Q/K/V/O slots for OP_IMM relay (with a GATE4
    sub-head). At AX byte positions (IS_BYTE + H1[AX_I]), this head attends
    to AX marker (MARK_AX) and copies OP_IMM to byte positions.

    Phase=8.4 places it AFTER ``layer8_multibyte_routing`` (8.3) but BEFORE
    the L8 alu_postop_attach (phase 8.5) so attention bakes complete before
    the FFN wrap.
    """
    def _bake(block, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        attn8 = block.attn
        HD = attn8.W_q.shape[0] // attn8.num_heads
        # Per-bake attention-head allocator. Head 4 is pinned at its
        # existing slot so ``generate_attention_head`` writes the same
        # ``head_idx * HD + slot`` rows it always has. Stashed on the
        # attention module so downstream tooling can inspect the layout.
        # See :data:`_L8_HEAD_LAYOUT`.
        attn8._l8_head_allocator = _allocate_layer8_attn_heads(
            ("layer8_op_imm_relay.head_4",)
        )
        Primitives.generate_attention_head(
            attn8, _layer8_op_imm_relay_head_spec(BD), HD
        )

    # Dim-ownership claims: L8 attn head 4 OP_IMM relay.
    #   W_q[4*HD, IS_BYTE], W_q[4*HD, H1+AX_I], W_q[4*HD, CONST]
    #   W_k[4*HD, MARK_AX], W_k[4*HD, IS_BYTE], W_k[4*HD, CONST]
    #   W_v[4*HD, OP_IMM]
    #   W_o[OP_IMM, 4*HD]
    #   GATE4 = 1 sub-head: W_q[4*HD+1, IS_BYTE], W_q[4*HD+1, CONST]
    #                       W_k[4*HD+1, CONST]
    _claims = {
        (8, "attn_W_v", "4_0", "OP_IMM+0"),
        (8, "attn_W_o", "4_0", "OP_IMM+0"),
    }

    return Operation(
        name="layer8_op_imm_relay",
        reads={"IS_BYTE", "H1", "MARK_AX", "OP_IMM", "CONST"},
        writes={"OP_IMM"},
        kind="block",
        declarative_bake_fn=_bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer8_op_imm_relay_ir,
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        requires={"after": "layer8_alu"},
        # Phase 11.A r3: dropped phase=8.4 — target_op_name +
        # requires['after']: layer8_alu already pin placement and order.
        migrated=True,
        claims=_claims,
        # Staleness invariants: head 4 fires at AX byte positions
        # (IS_BYTE + H1[AX_I] Q gate), attends to the AX marker
        # (K[MARK_AX]=L), and copies OP_IMM forward so the L8
        # multibyte_routing FFN can gate on it at byte positions 1/2/3.
        # Declared at AX_byte1 (the first multi-byte slot) -- the same
        # write fires at AX_byte2 and AX_byte3 too, but the verifier only
        # needs one canonical position to confirm the producer fired.
        produces={
            "OP_IMM": "AX_byte1",
        },
        # Tier A opcode gating: V slot copies OP_IMM from the AX marker to
        # AX byte positions (``W_v[base, BD.OP_IMM]=1.0``,
        # ``W_o[BD.OP_IMM, base]=1.0``). The relay output is only non-zero
        # at the AX marker when OP_IMM is the active opcode, so head 4's
        # contribution to the residual is OP_IMM-gated by construction.
        opcodes={"OP_IMM"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer8_op_imm_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer8_op_imm_relay_head_spec(BD))
    return ir


def _layer8_op_imm_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative replacement for the L8 head-4 OP_IMM relay bake.

    ALiBi recency (slope 0.5) discriminates the slot-0 K signal across
    multiple prior MARK_AX rows. Without it, softmax averages OP_IMM
    across ALL prior MARK_AX positions; for multi-IMM programs (e.g.
    ``IMM 0x200; PSH; IMM 0x1234``) this dilutes the 3rd IMM's relay
    value below the L8 multibyte_routing threshold (6.5). Slope 0.5
    pulls the most-recent matching MARK_AX to the front and lets the
    IMM step's AX byte 0 land on the correct token. Matches the slope
    used by the L9 ALiBi relay heads (l9_ops.py:1375+).
    """

    # Class-1 marker-relative anchor (see ``_layer8_multibyte_fetch_head_spec``):
    # ``H1+AX_I`` keys on the AX-register threshold-bank slot, frame-invariant.
    AX_I = marker_bank_index("AX")
    L8_relay = 20.0
    return DeclarativeAttentionHeadSpec(
        head_idx=_L8_HEAD_LAYOUT_BY_NAME["layer8_op_imm_relay.head_4"],
        q=(
            AP(0, BD.IS_BYTE, L8_relay),
            AP(0, BD.H1 + AX_I, L8_relay),
            AP(0, BD.CONST, -L8_relay * 1.5),
            AP(1, BD.IS_BYTE, 500.0),
            AP(1, BD.CONST, -500.0),
        ),
        k=(
            AP(0, BD.MARK_AX, L8_relay),
            AP(0, BD.IS_BYTE, -L8_relay * 10),
            AP(0, BD.CONST, L8_relay * 0.5),
            AP(1, BD.CONST, 5.0),
        ),
        v=(AP(0, BD.OP_IMM, 1.0),),
        o=(AO(BD.OP_IMM, 0, 1.0),),
        alibi_slope=0.5,
        # b23f818c: the ALiBi slope keeps OP_IMM relay mass on the
        # CURRENT step's MARK_AX. CURRENT_STEP_ONLY makes this contract
        # explicit so the verifier flags any future regression that
        # drops the slope back to None (the pre-b23f818c IMM dilution
        # bug) at decl-time rather than waiting for a smoke failure.
        step_window=StepWindowConstraint.CURRENT_STEP_ONLY,
    )


def make_layer8_mem_to_alu_op(enable: bool = False) -> Operation:
    """L8 attention head 5: mem-attention reading mem[SP] → ALU_LO/HI at AX.

    Phase 1 of STACK0_VIA_MEM_ATTENTION_PLAN. Replaces L7 head 0's
    STACK0_BYTE0-keyed read with a direct ``mem[SP]`` lookup keyed by
    ADDR_KEY at the AX marker. The ADDR_KEY Q-side is staged by
    ``make_layer4_sp_to_addr_key_op`` (L4 attn heads 2-3) which writes
    the live SP value as nibble one-hots into the ADDR_KEY band at the
    AX marker. The K-side is the ADDR_KEY band at MEM val byte positions
    populated by ``_inject_mem_metadata`` (no embedding-side changes
    needed).

    Placement at L8 attn (not L9) is required because the L8 ALU FFN —
    in efficient mode, the ``AddSub5StageBlock`` post-op — reads ALU_LO
    and ALU_HI at the AX marker. The head must therefore write
    ALU_LO/HI BEFORE the L8 FFN/post-op runs. Writing at L9 attn would
    be too late.

    Design:
      - Q at AX marker, gated on POP-group / OP_LI_RELAY / OP_LC_RELAY
        so the head fires for binary ops + loads but not for IMM / NOP.
      - K at MEM val byte 0, gated on MEM_STORE + MEM_VAL_B0 so only
        store entries match. The most-recent matching store wins via
        ALiBi recency (slope tuned identically to the L9 ALiBi head:
        0.5 = 17.5 score margin per VM step).
      - Address matching uses the 12-bit (3-nibble) binary encoding
        identical to L15 head 0 and the L9 ALiBi proof-of-concept.
        Q-side reads the ADDR_KEY band (ADDR_B0/1/2_HI) staged by L4;
        K-side reads the same band populated by ``_inject_mem_metadata``.
      - V/O copy ``CLEAN_EMBED_LO/HI`` from the matching MEM val byte
        into ``ALU_LO/HI`` at the AX marker — exactly the source/dest
        L7 head 0 uses today.

    Score budget (per dim, after /sqrt(HD)=8):
      Dim 0 (Q gate):     -50 at non-fire / +50 at fire
      Dim 1 (K MEM_STORE): +312.5 at target+store, -312.5 at target+
                            non-store
      Dim 2 (K addr-anchor): -600 at store entries (ZFOD baseline)
      Dim 3 (byte select): +450 at MEM val byte 0
      Dims 4-27 (addr):    +300 at exact 12-bit match
      ALiBi (slope 0.5):   -17.5 per VM step distance

    Net at correct match (1 step back): +12.5 - 17.5 ≈ -5 → still attends
    via softmax1 (the only positive contributor beats the zero anchor).
    Net at wrong-addr store at same distance: -287.5 - 17.5 → suppressed.

    Disabled by default (``enable=False``). Flip together with
    ``make_layer4_sp_to_addr_key_op``.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return

        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator. Heads 5 and 7 are pinned
        # at their existing slots so the hand-rolled weight writes
        # below land byte-identically. Stashed on the attention module
        # so downstream tooling can inspect the layout. See
        # :data:`_L8_HEAD_LAYOUT`.
        attn._l8_head_allocator = _allocate_layer8_attn_heads((
            "layer8_mem_to_alu.head_5",
            "layer8_mem_to_alu.head_7",
        ))
        head = _L8_HEAD_LAYOUT_BY_NAME["layer8_mem_to_alu.head_5"]
        base = head * HD

        # Slope tuned to favor most-recent matching MEM_STORE.
        # 1 VM step = 35 tokens; slope 0.5 → 17.5 score margin per step.
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        L = 50.0

        # === Dim 0: bias — fire only at AX marker on binary-pop opcodes ===
        # Coordination with L8 multibyte_fetch (head 3): that head reads
        # ADDR_KEY across all K positions for multi-byte IMM fetch. With
        # L4 SP-to-ADDR_KEY staging the AX marker carries ADDR_KEY content
        # equal to SP, which would otherwise alias as a spurious K-match
        # candidate for multibyte_fetch. To keep the two heads disjoint
        # we restrict head 5 to fire only on the binary-pop opcode set
        # (the same set CMP[3] relays via L6 head 6) and add negative
        # blockers on OP_LI / OP_LC / OP_IMM / OP_LEA / OP_PSH / OP_JSR /
        # OP_ENT / OP_LEV / OP_JMP / OP_ADJ / OP_BZ / OP_BNZ / OP_EXIT so
        # head 5 stays silent on non-binary-pop steps. OP_* flags are
        # written at the AX marker by L5 FFN (opcode decode) so they are
        # available here as direct Q-side gates.
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.MARK_AX] = 2000.0     # require AX marker
        # Positive gates: any binary-pop opcode at the AX marker. Each
        # flag is ~1.0 when active, so a single +500 gate gives Q[0] = 500
        # when any one fires, comfortably above the -2000 baseline.
        for op_dim in (BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                       BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                       BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR,
                       BD.OP_SI, BD.OP_SC):
            attn.W_q[base, op_dim] = 500.0
        # Negative blockers — guarantee head 5 stays off when LI/LC/IMM/LEA/
        # PSH/etc. is active even if a binary-pop residual leaks. These
        # opcodes have their own ADDR_KEY/ALU paths and must not collide.
        for op_dim in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA,
                       BD.OP_PSH, BD.OP_JSR, BD.OP_ENT, BD.OP_LEV,
                       BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
                       BD.OP_EXIT):
            attn.W_q[base, op_dim] = -2000.0
        # Suppress at PC/SP/BP/STACK0/MEM markers — at these positions the
        # ADDR_KEY band carries other information (code addresses, mem
        # addresses) that would alias into this head's address match.
        attn.W_q[base, BD.MARK_PC] = -2000.0
        attn.W_q[base, BD.MARK_SP] = -2000.0
        attn.W_q[base, BD.MARK_BP] = -2000.0
        attn.W_q[base, BD.MARK_MEM] = -2000.0
        attn.W_q[base, BD.MARK_STACK0] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0
        # === Inc-3 (2026-06-19): kill the negative×negative cross-step
        # spurious attractor on the LI/PSH step (nested_quad, var, expr).
        #
        # GPU diagnosis (tools/probe_inc3_li_l8_perhead.py, campaign config,
        # spec_k=0): on a NON-firing step (e.g.
        # nested_quad step3 where the AX marker carries OP_PSH, head 5 is
        # gated OFF by the OP_PSH/OP_LI/OP_IMM Q-blockers), head 5's dim-0
        # K column carries a LATENT cross-op collision — a stale
        # ``MARK_AX=+30`` / ``OP_IMM=-30`` write from another L8 attn op
        # baked into the SAME physical column (head_idx 5 * HD). At a
        # PREVIOUS step's IMM AX-marker that K dim-0 evaluates to
        #   K[0] = MARK_AX*30 + CONST*10 + OP_IMM*(-30) = 30 + 10 - 150 = -110
        # and the suppressed query (off=AX[0], MARK_AX=0) gives
        #   Q[0] = CONST*(-2000) = -1985.
        # The product (-1985)*(-110) = +218,350 — a HUGE positive — pins
        # head 5's softmax onto that prior IMM AX-marker, and its V/O copies
        # the marker's CLEAN_EMBED garbage into ALU (0x0A -> 0xAA = 170),
        # corrupting the upstream-delivered LI byte-0. (35-tok golden does
        # not hit this: the IMM opcode flag does not sit on the prior AX
        # marker at the same cross-step distance, so the K dim-0 stays
        # non-negative and head 5's max score is ~449 instead of 218k.)
        #
        # FIX: explicitly clear head 5's dim-0 K column to its INTENDED
        # content (CONST=10 only). The colliding MARK_AX/OP_IMM writes are
        # not part of head 5's design (its row-select / store-gate lives on
        # dims 1-4); zeroing them makes K[0] >= 0 everywhere, so a
        # suppressed (negative) query can never produce a positive product.
        # head 5 bakes at phase 8.45, after the colliding 8.0-8.4 ops, so
        # this clear is order-safe; it is also a no-op flag-OFF (head 5 only
        # exists under enable=True/operand_from_memsp). Forceable kill-switch
        # C4_INC3_H5_DIM0_CLEAN=0 for an A/B differential.
        if _os_l8.environ.get("C4_INC3_H5_DIM0_CLEAN", "1") != "0":
            attn.W_k[base, BD.MARK_AX] = 0.0
            attn.W_k[base, BD.OP_IMM] = 0.0
        # === Dim 28: K-side AX marker exclusion (self-attention guard) ===
        # The L4 SP gather writes scale-10 ADDR_KEY content at the AX
        # marker. At head 5's firing Q position (AX marker), the address
        # encoding dims (4-27) match that staged content exactly, which
        # would otherwise drive softmax to self-attend to the AX marker
        # instead of the MEM val byte we want. The self-match contributes
        # up to 12 dims * (10 * 10)^2 = 120,000 raw score (= 15,000 after
        # /sqrt(HD)=8) at AX marker. Q[28]*K[28] = 100 * -2000 = -200,000
        # raw (= -25,000 after /sqrt(HD)=8) cleanly overwhelms it. This
        # dim only contributes when BOTH Q-side AND K-side are at AX
        # marker — i.e. only when head 5 would otherwise self-attend —
        # giving a clean negative penalty without leaking into non-firing
        # Q positions or non-AX K positions.
        AX_K_EXCLUDE = 28
        attn.W_q[base + AX_K_EXCLUDE, BD.MARK_AX] = 100.0
        attn.W_k[base + AX_K_EXCLUDE, BD.MARK_AX] = -2000.0

        # === REBUILD (Inc-1, 2026-06-18): recency-only mem[SP] byte-0 CAM ===
        #
        # GPU diagnosis (probe_operand_cam / probe_nostack0_dump) overturned
        # the original address-match design:
        #   (1) The MEM *value* byte-0 row carries MEM_STORE=0 (MEM_STORE fires
        #       on the addr rows, not the value rows), so the original dims
        #       1/2/3/29/30 — all keyed K-side on MEM_STORE — could never fire
        #       on the row that actually holds the operand. The value byte-0
        #       nibbles live on the MEM_VAL_B1-marked row (== L2H0[MEM]=1 AND
        #       H1[MEM]=0, the d=6-from-MEM "value byte 0" slot — the autoreg
        #       MEM layout offsets MEM_VAL_B1 onto value byte 0).
        #   (2) The address dims 4-27 produced a giant SELF-MATCH at the AX
        #       marker (L4 stages SP into ADDR_KEY there), so softmax pinned
        #       100% mass on the query row itself -> CLEAN_EMBED of the REG_AX
        #       marker token -> ALU=0. The AX_K_EXCLUDE guard (dim 28) was
        #       swamped because HD=109 (not the docstring's 8).
        #
        # The rebuilt head content-addresses the MEM value-byte-0 row directly
        # (no L4 ADDR_KEY staging — that op is the L19 OUTPUT-crush corruptor
        # and is disabled in this build, so there is no AX-marker self-match to
        # fight), placed at the L8 attn block (re-anchored to layer8_sp_gather)
        # so it writes ALU BEFORE the L8 ALU FFN consumes it. This advances the
        # equivalence-config gate from a STEP-1 fail (AX=0, the L4/L19 crush) to
        # a STEP-3 (ADD/SUB) fail, and is byte-identical flag-OFF.
        #
        # *** REMAINING Inc-1/Inc-2 BLOCKER (GPU-confirmed, not yet fixed) ***
        # Recency alone is INSUFFICIENT to deliver mem[SP]: the MEM value-byte-0
        # row carries NO store-commit marker (the ONLY dims that differ between a
        # real PSH store's value row and a non-store step's MEM value row are the
        # CLEAN_EMBED nibble values themselves — MEM_STORE / PSH_AT_SP / MARK_MEM
        # all sit on the section's MARK_MEM row, 5 positions earlier, never
        # broadcast to the value row). So on the canonical PSH;IMM;ADD pattern
        # (all 4 gate programs) ALiBi recency selects the more-recent IMM step's
        # PHANTOM value row (value 0) over the real PSH store's value row, and
        # the binary op reads operand-A = 0. A single value-row recency head
        # cannot discriminate. The fix is one of:
        #   (a) broadcast MEM_STORE/PSH_AT_SP to the section's value-byte rows
        #       (extend NeuralVMEmbedding._inject_mem_store) so K can gate on a
        #       per-value-row store bit, then recency picks the right store; or
        #   (b) restore address-matching but stage SP->ADDR_KEY WITHOUT crushing
        #       L19 OUTPUT (fix the L4/L19 interaction first).
        # Both are new multi-component builds (Inc-2 scope). Until one lands the
        # K-side below is a correct-placement scaffold, NOT a delivering CAM.
        #
        # Value-byte-0 row predicate (matches probe data, add_9 step1 off31):
        #   L2H0[MEM_I] = 1  AND  H1[MEM_I] = 0   (MEM_I = MEM marker index = 4)
        # Class-1 marker-relative: MEM_I is the MEM slot in the threshold-head
        # bank (frame-invariant — same in the 35- and 30-token frames), resolved
        # through ``marker_bank_index`` instead of the literal 4 so the audit
        # recognises ``L2H0 + MEM_I`` / ``H1 + MEM_I`` as declared-invariant.
        MEM_I = marker_bank_index("MEM")
        VR = 120.0  # value-row select strength (Q@AX * K@value-row)

        # Dim 1: value-byte-0 row select. Positive only when the K row is the
        # MEM value-byte-0 slot; negative const baseline keeps every other row
        # (incl. the AX-marker self-row, which has L2H0[MEM]=0) below zero so
        # softmax1's implicit zero logit suppresses non-value rows.
        attn.W_q[base + 1, BD.MARK_AX] = 1.0
        attn.W_k[base + 1, BD.L2H0 + MEM_I] = VR
        attn.W_k[base + 1, BD.H1 + MEM_I] = -VR
        attn.W_k[base + 1, BD.CONST] = -VR / 2

        # Dim 2: hard value-row confirm (signed, stronger) so address/marker
        # rows that happen to share one of the two predicate bits cannot beat
        # the true value-byte-0 row.
        attn.W_q[base + 2, BD.MARK_AX] = 1.0
        attn.W_k[base + 2, BD.L2H0 + MEM_I] = VR * 2
        attn.W_k[base + 2, BD.H1 + MEM_I] = -VR * 4
        attn.W_k[base + 2, BD.CONST] = -VR

        # Dim 3: AX-marker self-exclusion. Only contributes when BOTH the
        # query AND the key row are the AX marker (the self row); drives that
        # single candidate deeply negative so the head never self-attends even
        # though CLEAN_EMBED is present there.
        attn.W_q[base + 3, BD.MARK_AX] = 100.0
        attn.W_k[base + 3, BD.MARK_AX] = -VR * 20

        # === Dim 4: STORE-COMMIT GATE — the Inc-1 closing fix (2026-06-18) ===
        #
        # The blocker the prior rebuild left open: dims 1/2 select EVERY MEM
        # value-byte-0 row equally (all have MEM_VAL_B1=1 / L2H0[MEM]=1,
        # H1[MEM]=0), so on the PSH;IMM;ADD pattern ALiBi recency picked the
        # most-recent step's PHANTOM value row (mem byte=0) over the real PSH
        # store's value row (== mem[SP]); operand-A delivered as 0.
        #
        # GPU diagnosis (tools/probe_mem_store_rows / probe_head5_attn) found
        # the store-commit bit IS recoverable but is NOT on the value row at
        # head-5's read time: MEM_STORE sits ONLY on the section's MARK_MEM
        # marker row (set by the L6 head-6 relay, present from block 7) and is
        # not broadcast to the value rows until block 11 (AFTER this head's K
        # read). So ``make_layer7_mem_store_relay_op`` (L7, block 9) relays it
        # FORWARD marker-row → value-byte-0 row into the fresh per-value-row
        # band MEM_STORE_AT_VAL. add_9/sub_17 step3: pos 110 (real mem[SP])
        # gets MEM_STORE_AT_VAL=1.0; the phantom value rows (pos 74/146) get
        # 0.0.
        #
        # This dim adds a large positive boost to value-byte-0 rows that carry
        # the relayed store bit and a large negative penalty to value-byte-0
        # rows without it, so among the (otherwise equal, dims-1/2) value rows
        # only store rows survive softmax1's zero anchor; ALiBi recency then
        # picks the most-recent store (== current mem[SP]). The boost is sized
        # well above the worst-case ALiBi recency penalty (~-23.5 at the
        # 47-token store-to-query distance on the gate programs). Because
        # MEM_STORE_AT_VAL is a clean per-value-row 0/1 (no marker ±2 magnitude
        # to fight), a single signed K dim suffices (Q@MARK_AX=1):
        #   K = B·MEM_STORE_AT_VAL - C·CONST
        #   store value row (relay=1): B - C   = +400  (attend, beats recency)
        #   phantom value row (relay=0): -C     = -400  (suppress)
        # Non-value rows never reach here positive: dims 1/2 drive markers
        # (-420) and addr/register rows (-180) deeply negative, and they carry
        # MEM_STORE_AT_VAL=0 so dim 4 only deepens them (-400).
        STORE_B = 800.0  # MEM_STORE_AT_VAL boost
        STORE_C = 400.0  # CONST penalty (phantom value rows -> -400)
        attn.W_q[base + 4, BD.MARK_AX] = 1.0
        attn.W_k[base + 4, BD.MEM_STORE_AT_VAL] = STORE_B
        attn.W_k[base + 4, BD.CONST] = -STORE_C

        # === V/O: copy CLEAN_EMBED bytes → ALU_LO/HI at AX marker ===
        # This mirrors L7 head 0 (vm_step.py:_set_layer7_operand_gather)
        # which writes ALU_LO/HI from STACK0_BYTE0's CLEAN_EMBED. The L8
        # FFN (lookup ALU or AddSub5StageBlock in efficient mode) consumes
        # ALU_LO/HI as binary-op operand 2.
        SCALE_O = 6.0  # match L7 head 0 amplification (overcomes L4 ALU clear)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        # Matched default cancel: the upstream ALU clear leaves ALU_LO[0] and
        # ALU_HI[0] positive. Adding a fetched nonzero nibble without canceling
        # that default makes legacy L9 nibble gates see two active values
        # (0 and the real nibble), which can create false SUB byte borrows.
        # When the fetched nibble really is zero, the CLEAN_EMBED[0] write below
        # restores the zero slot by the same amount this cancel subtracts.
        attn.W_v[base + 0, BD.CONST] = 1.0
        for k in range(16):
            attn.W_o[BD.ALU_LO + k, base + 1 + k] = SCALE_O
            attn.W_o[BD.ALU_HI + k, base + 17 + k] = SCALE_O
        attn.W_o[BD.ALU_LO + 0, base + 0] = -SCALE_O
        attn.W_o[BD.ALU_HI + 0, base + 0] = -SCALE_O

        # Head 7: operand-A BYTE 1 staging for multi-byte ALU ops (16-bit
        # ADD/SUB, SHL/SHR/MUL wide). Delivers mem[SP] byte 1 to the AX marker
        # so the multi-byte adders/relays can compute byte 1 of the result.
        #
        # ONLY needed when the STACK0 emission is dropped (C4_NO_STACK0_EMIT):
        # in the 35-token build the wide-ALU byte-1 path reads stack byte 1
        # from the emitted STACK0 byte-1 token + STACK0_BYTE_VAL_1 (written by
        # layer10_psh_ax_broadcast at the STACK0 byte-1 row), so this
        # memory-sourced write would DOUBLE-WRITE / conflict and corrupt
        # SHL/SHR/MUL/16-bit. Gate head 7 on no_stack0_emit so it only supplies
        # byte 1 from memory when the emitted STACK0 token (and its
        # STACK0_BYTE_VAL_1 producer) are gone.
        if not no_stack0_emit_enabled():
            return
        head = _L8_HEAD_LAYOUT_BY_NAME["layer8_mem_to_alu.head_7"]
        base = head * HD
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        # === STACK0 campaign Part B (2026-06-18): rebuilt K to mirror the
        # WORKING head-5 byte-0 CAM ===
        #
        # The prior head-7 used the OLD address-match design (L4 ADDR_KEY
        # staging, disabled by the Inc-1 rebuild) + MEM_STORE row gating
        # (MEM_STORE is on the MARK_MEM marker row, NEVER on the value-byte
        # rows where this head reads K). So it delivered NOTHING under the flag
        # (AX_FULL_LO=-1 at the ADD step, GPU-confirmed). Rebuild it on the
        # same value-row + MEM_STORE_AT_VAL recency CAM head-5 uses for byte 0,
        # selecting the value-byte-1 row (MEM_VAL_B2) instead of byte-0.
        #
        # === Dim 0: bias — fire only at AX marker on binary-pop opcodes ===
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.MARK_AX] = 2000.0
        for op_dim in (BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                       BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                       BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR,
                       BD.OP_SI, BD.OP_SC):
            attn.W_q[base, op_dim] = 500.0
        for op_dim in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA,
                       BD.OP_PSH, BD.OP_JSR, BD.OP_ENT, BD.OP_LEV,
                       BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
                       BD.OP_EXIT):
            attn.W_q[base, op_dim] = -2000.0
        attn.W_q[base, BD.MARK_PC] = -2000.0
        attn.W_q[base, BD.MARK_SP] = -2000.0
        attn.W_q[base, BD.MARK_BP] = -2000.0
        attn.W_q[base, BD.MARK_MEM] = -2000.0
        attn.W_q[base, BD.MARK_STACK0] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0

        AX_K_EXCLUDE = 28
        attn.W_q[base + AX_K_EXCLUDE, BD.MARK_AX] = 100.0
        attn.W_k[base + AX_K_EXCLUDE, BD.MARK_AX] = -2000.0

        # Dim 27: MARK_MEM K-exclusion (dedicated, strong). The section's
        # MARK_MEM marker row carries the relayed MEM_STORE_AT_VAL (the bit
        # originates there) AND a large dim-0 CONST/Q-gate component, so it
        # out-scores the value-byte rows on the bias dim alone. A dedicated
        # exclusion dim (large Q at the firing AX query, large -K at any
        # MARK_MEM row) drives the marker row deeply negative without touching
        # the value-byte rows (MARK_MEM=0 there). Mirrors the AX_K_EXCLUDE
        # pattern. Slot 27 is free on head 7 (dims 4-27 were the retired
        # address-match band).
        MEM_K_EXCLUDE = 27
        attn.W_q[base + MEM_K_EXCLUDE, BD.MARK_AX] = 100.0
        attn.W_k[base + MEM_K_EXCLUDE, BD.MARK_MEM] = -2000.0

        # Dim 1/2: value-byte-1 row select. MEM_VAL_B2 marks value byte 1 in
        # the autoregressive MEM layout (MEM_VAL_B1 marks byte 0). Strong
        # positive on the byte-1 value row; negative CONST baseline buries
        # every other row below softmax1's zero anchor. HARD MARK_MEM / MARK_*
        # K-exclusion: the section's MARK_MEM marker row ALSO carries the
        # relayed MEM_STORE_AT_VAL (it is the bit's origin), so without a K-side
        # marker exclusion the dim-4 store gate would pull head-7 onto the
        # marker row (CLEAN_EMBED empty -> byte 1 = 0) instead of the value
        # row. Burying the marker (and the register markers) here keeps the
        # MEM_VAL_B2 value row the only positive candidate.
        VR = 120.0
        attn.W_q[base + 1, BD.MARK_AX] = 1.0
        attn.W_k[base + 1, BD.MEM_VAL_B2] = VR
        attn.W_k[base + 1, BD.CONST] = -VR / 2
        attn.W_k[base + 1, BD.MARK_MEM] = -VR * 20
        attn.W_k[base + 1, BD.MARK_PC] = -VR * 20
        attn.W_k[base + 1, BD.MARK_AX] = -VR * 20
        attn.W_k[base + 1, BD.MARK_SP] = -VR * 20
        attn.W_k[base + 1, BD.MARK_BP] = -VR * 20

        attn.W_q[base + 2, BD.MARK_AX] = 1.0
        attn.W_k[base + 2, BD.MEM_VAL_B2] = VR * 2
        attn.W_k[base + 2, BD.CONST] = -VR

        # Dim 3: AX-marker self-exclusion (mirror head-5 dim 3).
        attn.W_q[base + 3, BD.MARK_AX] = 100.0
        attn.W_k[base + 3, BD.MARK_AX] = -VR * 20

        # Dim 4: STORE-COMMIT GATE — same relayed per-value-row store bit
        # head-5 uses. make_layer7_mem_store_relay_op now broadcasts
        # MEM_STORE_AT_VAL onto ALL value-byte rows (B0..B3), so the byte-1
        # row carries it for a real store; recency then picks the most-recent
        # store == current mem[SP]. The marker row (where the bit originates) is
        # already buried by dim-1's MARK_MEM exclusion, so this gate only ranks
        # among the value-byte rows.
        STORE_B = 800.0
        STORE_C = 400.0
        attn.W_q[base + 4, BD.MARK_AX] = 1.0
        attn.W_k[base + 4, BD.MEM_STORE_AT_VAL] = STORE_B
        attn.W_k[base + 4, BD.CONST] = -STORE_C

        # === V/O: copy CLEAN_EMBED bytes → STACK0_BYTE_VAL_1 + AX_FULL ===
        # STACK0_BYTE_VAL_1 is the band the L13 add/sub minuend/addend relays +
        # the L10 high-byte adder + BDToGEConverter read for operand-A byte 1.
        # Also keep the AX_FULL write (the wide-ALU / GE-convert path consumes
        # AX_FULL_* as operand-A positions 2/3).
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        attn.W_v[base + 0, BD.CONST] = 1.0
        for k in range(16):
            attn.W_o[BD.STACK0_BYTE_VAL_1_LO + k, base + 1 + k] = SCALE_O
            attn.W_o[BD.STACK0_BYTE_VAL_1_HI + k, base + 17 + k] = SCALE_O
            attn.W_o[BD.AX_FULL_LO + k, base + 1 + k] = SCALE_O
            attn.W_o[BD.AX_FULL_HI + k, base + 17 + k] = SCALE_O
        attn.W_o[BD.STACK0_BYTE_VAL_1_LO + 0, base + 0] = -SCALE_O
        attn.W_o[BD.STACK0_BYTE_VAL_1_HI + 0, base + 0] = -SCALE_O
        attn.W_o[BD.AX_FULL_LO + 0, base + 0] = -SCALE_O
        attn.W_o[BD.AX_FULL_HI + 0, base + 0] = -SCALE_O

    # Dim-ownership claims: L8 attn head 5 mem-to-ALU.
    # Only claim load-bearing V/O slot/column pairs (not the dense Q/K gates
    # which are head-local and unlikely to collide with other ops).
    #   W_v[5*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15 (V slot 1..16)
    #   W_v[5*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15 (V slot 17..32)
    #   W_o[ALU_LO + k, 5*HD + 1 + k]            for k=0..15
    #   W_o[ALU_HI + k, 5*HD + 17 + k]           for k=0..15
    # Conditional on enable=True; declared unconditionally so the registry
    # catches latent collisions once the op is enabled.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((8, "attn_W_v", f"5_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((8, "attn_W_v", f"5_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
            _claims.add((8, "attn_W_v", f"7_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((8, "attn_W_v", f"7_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    # MEM_STORE_AT_VAL (read by head-5 dim 4, the store-commit gate) is a
    # flag-gated residual band — it is omitted from dim_positions on a
    # flag-OFF build. Declare it in the read set only when the op is enabled
    # so the dep-graph validator does not reject the undeclared dim flag-OFF.
    _store_at_val_read = {"MEM_STORE_AT_VAL"} if enable else set()

    # STACK0 campaign Part B: head 7 writes STACK0_BYTE_VAL_1 (operand-A byte 1
    # from mem[SP]) ONLY when the STACK0 emission is dropped (the byte-1 head
    # bakes under no_stack0_emit). Declare the write conditionally so the
    # dep-graph validator sees the producer only in that config. The band is a
    # production dim (always in dim_positions), so this is purely a graph hint.
    _byte1_write = (
        {"STACK0_BYTE_VAL_1_LO", "STACK0_BYTE_VAL_1_HI"}
        if (enable and no_stack0_emit_enabled())
        else set()
    )

    return Operation(
        name="layer8_mem_to_alu",
        # Phase 8.45 places this after layer8_op_imm_relay (8.4) and BEFORE
        # the L8 alu_postop_attach (8.5), keeping all L8 attn bakes in
        # phase order.
        # Phase 8.A: ADDR_B0_HI_PREV_STEP marks the read as cross-step
        # relative to L9 lev_addr_relay / L9 lev_bp_to_pc_relay / L15
        # store_stack0_sp_byte0_addr which fire after L8 in the same step.
        # L4 sp_to_addr_key, L8 sp_gather_bake, L13 mem_addr_gather are
        # same-step or earlier writers, but L8 mem_to_alu's read targets
        # the prev-step residual ADDR_B0_HI value (via the KV cache) — the
        # alias keeps the numeric position identical (slot 206) so weight
        # bakes stay byte-identical; only the dep graph view changes.
        # Phase 8.A follow-up: ADDR_B0_LO_PREV_STEP is the matching alias
        # for the LO-nibble band. L9 lev_addr_relay/lev_bp_to_pc_relay,
        # L13 mem_addr_gather, and L15 store_stack0_sp_byte0_addr all
        # write ADDR_B0_LO after L8 in the same step; the read at L8.45
        # targets the previous-step residual value (KV cache), so the
        # PREV_STEP alias retires the back-edges into L8 without changing
        # the baked weight position (slot is shared with ADDR_B0_LO).
        # Phase 8.A continuation: ADDR_B{1,2}_{LO,HI}_PREV_STEP retire the
        # same back-edge pattern for the B1/B2 nibble bands -- L13
        # mem_addr_gather and the L12 attn dep anchor write
        # ADDR_B{1,2}_{LO,HI} after L8 in the same step, but this op's
        # read targets the previous-step residual value via the KV cache.
        # Same numeric base; weight bakes byte-identical.
        reads={"MARK_AX", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_OR", "OP_XOR", "OP_AND", "OP_SHL", "OP_SHR",
               "OP_SI", "OP_SC", "OP_LI", "OP_LC", "OP_IMM", "OP_LEA",
               "OP_PSH", "OP_JSR", "OP_ENT", "OP_LEV", "OP_JMP", "OP_ADJ",
               "OP_BZ", "OP_BNZ", "OP_EXIT", "MEM_STORE",
               "MEM_VAL_B2", "L2H0",
               "H1", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1",
               "ADDR_B1_LO.*.-1", "ADDR_B1_HI.*.-1",
               "ADDR_B2_LO.*.-1", "ADDR_B2_HI.*.-1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"} | _store_at_val_read,
        writes={"ALU_LO", "ALU_HI", "AX_FULL_LO", "AX_FULL_HI"} | _byte1_write,
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # REBUILD (Inc-1, 2026-06-18): re-anchor from ``layer10_byte_passthrough``
        # (which placed this head at L9/L10 attn — AFTER the L8 ALU FFN already
        # consumed ALU_LO/HI, so the operand arrived one block too late and the
        # binary op read 0) to ``layer8_sp_gather`` — the SAME anchor the
        # (working flag-OFF) L7 operand_gather head-0 uses. This bakes head 5
        # into the L8 attention block, BEFORE the L8 ALU FFN/post-op runs, which
        # is exactly the placement this op's own docstring requires ("Writing at
        # L9 attn would be too late"). The prior anchor contradicted that.
        target_op_name="layer8_sp_gather",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
        # Phase 11.A IR exposure: bake is `if not <flag>: return` at default
        # config, so an empty IR is byte-identical for default flag values.
        # Populating IR with the matching rules is Phase 11.A follow-up.
        compiler_ir=CompilerIR(),
    )


# Hidden-unit slot for the SP_GATHERED_THIS_STEP sentinel. Placed after the
# L8 ALU + multibyte_routing unit cluster (which reaches 2054 — see the
# ``ffn_units_used=2055`` annotation on ``make_layer8_multibyte_routing_op``).
# Holding the constant here keeps the op factory and the per-op test in sync.
_L8_SP_GATHERED_SENTINEL_UNIT = 2055


def _layer8_sp_gathered_sentinel_rule(S: float) -> FFNRule:
    """SwiGLU rule: write 1.0 to SP_GATHERED_THIS_STEP at MARK_SP positions.

    Pattern (matches ``_bake_layer1_ffn``'s STACK0_BYTE0 unit):
      up    = S * MARK_SP
      b_up  = -S * 0.5
      gate  = 1.0 (constant)
      W_down = SP_GATHERED_THIS_STEP at (2.0 / S)

    At MARK_SP=1: up = S/2, silu(S/2) ≈ S/2 for large S; gate=1.0;
        hidden = S/2; delta = (2/S) * (S/2) = 1.0 → output = 1.0.
    At MARK_SP=0: up = -S/2, silu(-S/2) ≈ 0; output = 0.

    The sentinel is naturally reset at STEP_BOUNDARY because MARK_SP is a
    per-step embedding flag (only set on the current step's SP marker
    token, not carried into the next step's embedding).
    """

    # ``lower_ffn`` multiplies S into W_up + b_up but does NOT scale
    # W_down — the rule author owns the W_down magnitude. ``2.0 / S``
    # cancels the SwiGLU hidden amplitude (silu(S/2) ≈ S/2) so the
    # delta lands at 1.0 (matching ``_bake_layer1_ffn``'s
    # STACK0_BYTE0 unit and ``_format_position_counter_rules``).
    return step_function_rule(
        name="l8_sp_gathered_this_step_sentinel",
        input_dim="MARK_SP",
        threshold=0.5,
        write_dim="SP_GATHERED_THIS_STEP",
        write_value=2.0,
        S=S,
    )


def make_layer8_sp_gathered_sentinel_ir(S: float = 100.0) -> CompilerIR:
    """Declarative IR mirror of the SP_GATHERED_THIS_STEP sentinel SwiGLU unit.

    Wraps the single-rule ``_layer8_sp_gathered_sentinel_rule`` into a
    ``CompilerIR`` so the op exposes its semantic source of truth via
    ``Operation.compiler_ir``. The bake path still pins the unit at
    ``_L8_SP_GATHERED_SENTINEL_UNIT`` (2055); this IR is what the verifier,
    scope checker, and dominance auditor consume, and what
    ``compare_symbolic_to_lowered_ffn`` validates byte-for-byte.
    """
    ir = CompilerIR()
    ir.layer(0).ffn.append(_layer8_sp_gathered_sentinel_rule(S))
    return ir


def make_layer8_sp_gathered_sentinel_op() -> Operation:
    """L8 FFN: write SP_GATHERED_THIS_STEP=1.0 at MARK_SP positions.

    B7-5 lifecycle bit (see ``investigation/bd-dim-usage-map`` REPORT
    Section 5). The L8 SP-byte gather (``layer8_sp_gather_bake``, phase
    8.0, heads 0-2) populates ADDR_B0/B1/B2 with the current step's SP
    bytes. This op marks the MARK_SP row with a single bit indicating
    "L8 has just finished staging SP for THIS step" — letting L10
    ``tail_sp_marker_*`` rules use positive in-step evidence rather
    than HAS_SE -1e9 negative hammers (per B6-K's recommendation).

    Phase=8.6 places this AFTER:
      - layer8_sp_gather_bake (8.0)
      - layer8_multibyte_fetch_bake (8.1)
      - layer8_alu (8.2)
      - layer8_multibyte_routing (8.3)
      - layer8_op_imm_relay (8.4)
      - format_position_counter (8.5)
    so the sentinel write does not interfere with any in-layer L8 bake
    and is visible to all consumers from L9 onward.

    Implementation: single SwiGLU unit at index
    ``_L8_SP_GATHERED_SENTINEL_UNIT`` (= 2055 — the next slot after
    multibyte_routing's 0-2054 cluster). ``ffn_units_used=2056`` so the
    L8 PureFFN allocator pre-sizes hidden_dim correctly.
    """

    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake allocator. The sentinel claims a single SwiGLU unit
        # at the pinned offset 2055, immediately after the
        # ``layer8_multibyte_routing`` cluster.
        allocator = _allocate_layer8_ffn_units()
        block.ffn._l8_unit_allocator = allocator
        start_unit = _l8_ffn_range_start(
            allocator, "layer8_sp_gathered_sentinel"
        )
        # Byte-identity guard: the pinned start must match the legacy
        # constant. If the table ever drifts from
        # ``_L8_SP_GATHERED_SENTINEL_UNIT`` the assertion fires before
        # any weight surgery.
        assert start_unit == _L8_SP_GATHERED_SENTINEL_UNIT, (
            f"L8 SP-gathered sentinel pin drift: allocator returned "
            f"{start_unit}, legacy constant is "
            f"{_L8_SP_GATHERED_SENTINEL_UNIT}"
        )
        rule = _layer8_sp_gathered_sentinel_rule(S)
        dim_names = Primitives.ffn_rule_dim_names((rule,))
        dim_map = Primitives.dim_positions_from_bd(proxy, dim_names)
        Primitives.lower_ffn_rules(
            block.ffn,
            (rule,),
            dim_map,
            start_unit=start_unit,
            S=S,
        )

    _claims = {
        (8, "ffn_W_down", str(_L8_SP_GATHERED_SENTINEL_UNIT),
         "SP_GATHERED_THIS_STEP+0"),
    }

    return Operation(
        name="layer8_sp_gathered_sentinel",
        # Phase 1 (memory cluster fix plan): shares L8 FFN unit range with
        # ``layer8_multibyte_routing`` at a disjoint sub-range. See
        # docs/SLOT_REGISTRY_AUDIT_2026_06_05.md.
        slot_share=("ffn_units",),
        reads={"MARK_SP"},
        writes={"SP_GATHERED_THIS_STEP"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=make_layer8_sp_gathered_sentinel_ir(),
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
        migrated=True,
        claims=_claims,
        # +1 over multibyte_routing's 2055 so L8's PureFFN allocator
        # grows hidden_dim to cover the sentinel unit's index.
        ffn_units_used=_L8_SP_GATHERED_SENTINEL_UNIT + 1,
        # The op produces a fresh in-step sentinel at MARK_SP. Mark it
        # so the staleness scanner sees the producer when downstream
        # consumers (L10 tail rules) declare ``consumes_fresh``.
        # B12 backfill (wave 1c): the docstring lists six L8 ops this
        # sentinel must run after (phases 8.0..8.5). The last L8 writer
        # to MARK_SP-adjacent dims is layer8_multibyte_routing, so pin
        # strictly after it; same_layer_as layer8_alu keeps the op on
        # L8 even if the dynamic scheduler tries to float it later.
        # See docs/B12_BACKFILL_SPEC.md §11.
        requires={
            "after": "layer8_multibyte_routing",
            "same_layer_as": "layer8_alu",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
