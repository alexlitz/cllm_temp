"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


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
                name=f"l8_alu_add_lo_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_add,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_AX and OP_ADD",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_AX and OP_ADD",
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
                name=f"l8_alu_lea_lo_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=80.5,
                gate=gate_lea,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_AX and OP_LEA",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_AX and OP_LEA",
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
                name=f"l8_alu_sub_lo_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_sub,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_AX and OP_SUB",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_AX and OP_SUB",
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
                name=f"l8_alu_add_carry_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_add,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_AX and OP_ADD",
                dominates_at={carry_byte0: "MARK_AX and OP_ADD"},
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
                name=f"l8_alu_lea_carry_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=80.5,
                gate=gate_lea,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_AX and OP_LEA",
                dominates_at={carry_byte0: "MARK_AX and OP_LEA"},
            ))
    return tuple(rules)


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
    rules = []
    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            rules.append(multi_way_and_rule(
                name=f"l8_alu_adj_lo_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=85.0,
                gate=gate_adj_lo,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_AX and OP_ADJ",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_AX and OP_ADJ",
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
                name=f"l8_alu_adj_carry_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{a}", 1.0),
                    (f"FETCH_LO+{b}", 20.0),
                ),
                threshold=85.0,
                gate=gate_adj,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_AX and OP_ADJ",
                dominates_at={carry_byte0: "MARK_AX and OP_ADJ"},
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
                name=f"l8_alu_sub_borrow_a{a}_b{b}",
                conditions=(
                    ("MARK_AX", 1.0),
                    ("MARK_PC", -4.0),
                    (f"ALU_LO+{a}", 1.0),
                    (f"AX_CARRY_LO+{b}", 1.0),
                ),
                threshold=2.5,
                gate=gate_sub,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_AX and OP_SUB",
                dominates_at={carry_byte0: "MARK_AX and OP_SUB"},
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
                name=f"l8_alu_ent_lo_sp{sp_lo}_imm{imm_lo}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{sp_lo}", 1.0),
                    (f"FETCH_LO+{imm_lo}", 20.0),
                ),
                threshold=85.0,
                gate=gate_ent_lo,
                writes=((f"OUTPUT_LO+{result}", write_scale),),
                scope="MARK_AX and OP_ENT",
                dominates_at={
                    f"OUTPUT_LO+{result}": "MARK_AX and OP_ENT",
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
            rules.append(FFNRule.gated_write(
                name=f"l8_alu_ent_borrow_sp{sp_lo}_imm{imm_lo}",
                conditions=(
                    ("MARK_AX", 60.0),
                    *blockers,
                    (f"ALU_LO+{sp_lo}", 1.0),
                    (f"FETCH_LO+{imm_lo}", 20.0),
                ),
                threshold=85.0,
                gate=gate_ent,
                writes=((carry_byte0, carry_scale),),
                scope="MARK_AX and OP_ENT",
                dominates_at={carry_byte0: "MARK_AX and OP_ENT"},
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
        FFNRule.constant_write(
            name="l8_alu_cmp_group",
            conditions=(
                ("OP_EQ", 1.0),
                ("OP_NE", 1.0),
                ("OP_LT", 1.0),
                ("OP_GT", 1.0),
                ("OP_LE", 1.0),
                ("OP_GE", 1.0),
                ("MARK_AX", 1.0),
            ),
            threshold=1.5,
            writes=(("CMP_GROUP+0", write_scale),),
            scope="MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
            dominates_at={
                "CMP_GROUP+0":
                    "MARK_AX and (OP_EQ or OP_NE or OP_LT or OP_GT or OP_LE or OP_GE)",
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
    gate_mark_ax = dim_ref("marker", "AX")
    rules = []
    for k in range(4):
        cmp_k = dim_ref("cmp_flag", "cascade", k)
        rules.append(FFNRule.gated_write(
            name=f"l8_alu_cmp_clear_k{k}",
            conditions=((f"CMP+{k}", 1.0),),
            threshold=0.0,
            gate=gate_mark_ax,
            gate_weight=S,
            gate_bias=-S * 0.5,
            writes=((cmp_k, write_scale),),
            scope="MARK_AX",
            dominates_at={f"CMP+{k}": "MARK_AX"},
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
        rules.append(FFNRule.constant_write(
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
        rules.append(FFNRule.gated_write(
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
        rules.append(FFNRule.gated_write(
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
        FFNRule.gated_write(
            name="l8_alu_lev_b1",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_BP", 1.0),
            ),
            threshold=1.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("ADDR_B1_LO+0", write_scale),),
            scope="OP_LEV and MARK_BP",
            dominates_at={"ADDR_B1_LO+0": "OP_LEV and MARK_BP"},
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
        FFNRule.gated_write(
            name="l8_alu_lev_b2",
            conditions=(
                ("OP_LEV", 1.0),
                ("MARK_BP", 1.0),
            ),
            threshold=1.5,
            gate="CONST",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=(("ADDR_B2_LO+0", write_scale),),
            scope="OP_LEV and MARK_BP",
            dominates_at={"ADDR_B2_LO+0": "OP_LEV and MARK_BP"},
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
    return (
        FFNRule.constant_write(
            name="l8_alu_lea_axb2_lo",
            conditions=(
                ("CMP+7", 1.0),
                ("H1+1", 1.0),
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_1", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=3.5,
            writes=(
                ("OUTPUT_LO+1", 4.0 / S),
                ("OUTPUT_LO+0", -4.0 / S),
            ),
            scope="CMP+7 and H1+1 and IS_BYTE and BYTE_INDEX_1 and not HAS_SE",
            dominates_at={
                "OUTPUT_LO+1":
                    "CMP+7 and H1+1 and IS_BYTE and BYTE_INDEX_1 and not HAS_SE",
                "OUTPUT_LO+0":
                    "CMP+7 and H1+1 and IS_BYTE and BYTE_INDEX_1 and not HAS_SE",
            },
        ),
        FFNRule.constant_write(
            name="l8_alu_lea_axb2_hi",
            conditions=(
                ("CMP+7", 1.0),
                ("H1+1", 1.0),
                ("IS_BYTE", 1.0),
                ("BYTE_INDEX_1", 1.0),
                ("HAS_SE", -1.0),
            ),
            threshold=3.5,
            writes=(("OUTPUT_HI+0", 2.0 / S),),
            scope="CMP+7 and H1+1 and IS_BYTE and BYTE_INDEX_1 and not HAS_SE",
            dominates_at={
                "OUTPUT_HI+0":
                    "CMP+7 and H1+1 and IS_BYTE and BYTE_INDEX_1 and not HAS_SE",
            },
        ),
    )


def _layer8_alu_rules(S: float) -> tuple[FFNRule, ...]:
    """Full ordered ``FFNRule`` sequence for ``layer8_alu``.

    Concatenates the 18 sub-stage rule tuples in cursor order so the
    composite list lowers byte-identically against
    ``_set_layer8_alu``. Total: 2023 rules covering offsets 0..2022.
    """
    return (
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
    18 ``_layer8_alu_<substage>_rules`` factories). The ``vm_step``
    helper is still imported by ``layer8_multibyte_routing`` for cursor
    recovery (an idempotent overwrite of the same weights), so it
    stays in place; this op no longer calls it. ``compiler_ir`` exposes
    the rule spec to verifier / scope / dominance tooling.
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
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               # V2/G7 LEV detector: in-step topology edge from
               # lev_detector_head (phase=8.06) replaces the cross-step
               # requires["after"]=layer16_lev_routing below.
               "PC_VIA_LEV_DETECTOR_LO"},
        writes={"OUTPUT_LO", "CARRY", "CMP_GROUP"},
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
        rules.append(FFNRule.gated_write(
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
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
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
    AX_I = 1
    TOP = 36
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
    so the shared unit counter starts after the ALU units (the helper
    internally re-invokes ``_set_layer8_alu`` to compute ``unit_start``;
    that re-call is an idempotent overwrite of the same ALU weights).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_alu

        proxy = _as_setdim_proxy(dim_positions)
        # Per-bake allocator. ``layer8_alu`` (phase 8.2) already ran
        # and produced units 0..2022; the helper re-call below is the
        # legacy idempotent overwrite that recovers the cursor. With
        # the allocator in place the pinned offset 2023 is the source
        # of truth -- the helper return is now a byte-identity guard.
        allocator = _allocate_layer8_ffn_units()
        block.ffn._l8_unit_allocator = allocator
        unit_start = _set_layer8_alu(block.ffn, S, proxy)
        expected_start = _l8_ffn_range_start(
            allocator, "layer8_multibyte_routing"
        )
        assert unit_start == expected_start, (
            f"L8 multibyte_routing unit start drift: helper returned "
            f"{unit_start}, allocator expected {expected_start}"
        )
        lower_layer8_multibyte_routing_ir(
            block.ffn,
            S,
            proxy,
            start_unit=expected_start,
        )

    return Operation(
        name="layer8_multibyte_routing",
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
    conditions = (
        ("IS_BYTE", 1.0),
        ("H1+1", 1.0),
        ("OP_IMM", 1.0),
        ("MARK_AX", -4.0),
    )
    for k in range(16):
        rules.append(FFNRule.gated_write(
            name=f"l8_multibyte_route_lo_{k}",
            conditions=conditions,
            threshold=6.5,
            gate=f"AX_CARRY_LO+{k}",
            writes=((f"OUTPUT_LO+{k}", 8.0 / S),),
        ))
    for k in range(16):
        rules.append(FFNRule.gated_write(
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
        reads={"MARK_AX", "MARK_SP", "OP_ADJ", "OP_ENT", "OP_LEA",
               "EMBED_LO", "EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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
    AX_I = 1
    SP_I = 2
    BP_I = 3
    MEM_I = 4

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
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer8_op_imm_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer8_op_imm_relay_head_spec(BD))
    return ir


def _layer8_op_imm_relay_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative replacement for the L8 head-4 OP_IMM relay bake."""

    AX_I = 1
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

        # === Dim 1: store anchor ===
        attn.W_q[base + 1, BD.MARK_AX] = 50.0
        attn.W_k[base + 1, BD.MEM_STORE] = 100.0
        attn.W_k[base + 1, BD.CONST] = -50.0

        # === Dim 2: ZFOD baseline ===
        attn.W_q[base + 2, BD.CONST] = -96.0
        attn.W_k[base + 2, BD.MEM_STORE] = 50.0

        # === Dim 3: byte 0 selection (MEM val byte 0) ===
        BS = 60.0
        attn.W_q[base + 3, BD.MARK_AX] = BS
        # MEM val byte 0 is at d=5 from MEM marker: L2H0[MEM]=1, H1[MEM]=0
        MEM_I = 4
        attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
        attn.W_k[base + 3, BD.H1 + MEM_I] = -BS

        # === Dim 29: hard value-byte gate ===
        # Address bytes can carry an exact ADDR_B match and swamp the softer
        # byte-selection bonus above. Require the same d=5 MEM value-byte
        # predicate with a stronger signed gate so address bytes cannot win.
        VAL_GATE = 29
        VG = 200.0
        attn.W_q[base + VAL_GATE, BD.MARK_AX] = VG
        attn.W_k[base + VAL_GATE, BD.CONST] = -100.0
        attn.W_k[base + VAL_GATE, BD.L2H0 + MEM_I] = 300.0
        attn.W_k[base + VAL_GATE, BD.H1 + MEM_I] = -300.0

        # === Dim 30: hard store gate ===
        # Default MEM sections from non-store steps can share the same value
        # byte position and partially match a polluted ADDR_KEY query. Require
        # an actual historical store strongly enough that non-store MEM value
        # bytes cannot beat the stored stack value (observed on MUL 6*7).
        STORE_GATE = 30
        SG = 100.0
        attn.W_q[base + STORE_GATE, BD.MARK_AX] = SG
        attn.W_k[base + STORE_GATE, BD.MEM_STORE] = SG
        attn.W_k[base + STORE_GATE, BD.CONST] = -SG / 2

        # === Dims 4-27: 24-bit binary address encoding ===
        # Same encoding as L15 head 0 / L9 ALiBi head: iterate over both
        # _LO and _HI bases per address byte. Q and K read from the same
        # residual dims because the L4 SP gather writes into the same
        # ADDR_B*_HI bands that `_inject_mem_metadata` writes K-side into.
        # (Q-side ADDR_B*_LO bands carry zero contribution because the
        # SP-gather only writes HI bands — see make_layer4_sp_to_addr_key_op.)
        addr_dim = 4
        scale = 10.0
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

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

        # Head 7: stack byte 1 staging for wide ALU ops. SHL/MUL need the
        # full generic pipeline result byte 1 later; SHR by a full byte needs
        # stack byte 1 available at the marker before the shift pipeline runs.
        # Stage the historical MEM value byte 1 into AX_FULL_* at the AX
        # marker. GE conversion consumes AX_FULL_* as operand-A positions 2/3.
        head = _L8_HEAD_LAYOUT_BY_NAME["layer8_mem_to_alu.head_7"]
        base = head * HD
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.MARK_AX] = 2000.0
        for op_dim in (BD.OP_MUL, BD.OP_SHL, BD.OP_SHR):
            attn.W_q[base, op_dim] = 500.0
        for op_dim in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA,
                       BD.OP_PSH, BD.OP_JSR, BD.OP_ENT, BD.OP_LEV,
                       BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
                       BD.OP_EXIT):
            attn.W_q[base, op_dim] = -2000.0
        for marker_dim in (BD.MARK_PC, BD.MARK_SP, BD.MARK_BP,
                           BD.MARK_MEM, BD.MARK_STACK0):
            attn.W_q[base, marker_dim] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0

        AX_K_EXCLUDE = 28
        attn.W_q[base + AX_K_EXCLUDE, BD.MARK_AX] = 100.0
        attn.W_k[base + AX_K_EXCLUDE, BD.MARK_AX] = -2000.0

        attn.W_q[base + 1, BD.MARK_AX] = 50.0
        attn.W_k[base + 1, BD.MEM_STORE] = 100.0
        attn.W_k[base + 1, BD.CONST] = -50.0

        attn.W_q[base + 2, BD.CONST] = -96.0
        attn.W_k[base + 2, BD.MEM_STORE] = 50.0

        # Select MEM value byte 1. In the autoregressive MEM layout,
        # MEM_VAL_B1 marks value byte 0 and MEM_VAL_B2 marks value byte 1.
        BS = 60.0
        attn.W_q[base + 3, BD.MARK_AX] = BS
        attn.W_k[base + 3, BD.MEM_VAL_B2] = BS

        VAL_GATE = 29
        VG = 200.0
        attn.W_q[base + VAL_GATE, BD.MARK_AX] = VG
        attn.W_k[base + VAL_GATE, BD.CONST] = -100.0
        attn.W_k[base + VAL_GATE, BD.MEM_VAL_B2] = 300.0

        STORE_GATE = 30
        SG = 100.0
        attn.W_q[base + STORE_GATE, BD.MARK_AX] = SG
        attn.W_k[base + STORE_GATE, BD.MEM_STORE] = SG
        attn.W_k[base + STORE_GATE, BD.CONST] = -SG / 2

        addr_dim = 4
        scale = 10.0
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        attn.W_v[base + 0, BD.CONST] = 1.0
        for k in range(16):
            attn.W_o[BD.AX_FULL_LO + k, base + 1 + k] = SCALE_O
            attn.W_o[BD.AX_FULL_HI + k, base + 17 + k] = SCALE_O
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
               "OP_BZ", "OP_BNZ", "OP_EXIT", "MEM_STORE", "MEM_VAL_B2", "L2H0",
               "H1", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "ADDR_B0_LO.*.-1", "ADDR_B0_HI.*.-1",
               "ADDR_B1_LO.*.-1", "ADDR_B1_HI.*.-1",
               "ADDR_B2_LO.*.-1", "ADDR_B2_HI.*.-1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI", "AX_FULL_LO", "AX_FULL_HI"},
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Phase 8.G.6: drop ``layer_idx=8`` literal; bind to the L8 attn
        # anchor ``layer10_byte_passthrough`` so the block op resolves
        # to whichever layer the compiler places the anchor at.
        target_op_name="layer10_byte_passthrough",
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

    return FFNRule.gated_write(
        name="l8_sp_gathered_this_step_sentinel",
        conditions=(("MARK_SP", 1.0),),
        threshold=0.5,
        gate=None,
        gate_bias=1.0,
        # ``lower_ffn`` multiplies S into W_up + b_up but does NOT scale
        # W_down — the rule author owns the W_down magnitude. ``2.0 / S``
        # cancels the SwiGLU hidden amplitude (silu(S/2) ≈ S/2) so the
        # delta lands at 1.0 (matching ``_bake_layer1_ffn``'s
        # STACK0_BYTE0 unit and ``_format_position_counter_rules``).
        writes=(("SP_GATHERED_THIS_STEP", 2.0 / S),),
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
