"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L2 attention-head layout (pinned indices) ======================
#
# Two ops claim heads on the L2 attention block today:
#
#   * ``layer2_threshold_attn`` (phase=2)            -> head 0
#   * ``layer2_lookback_detection_head`` (phase=2.1) -> head 1
#
# The lookback head is conditionally enabled at runtime via
# ``enable_conversational_io``; we still claim its head index in the
# allocator so the layout is structurally stable across builds (the
# bake itself early-returns when the gate is off, so no weights move
# either way). Pre-migration each call site picked its ``head_idx``
# as a bare integer literal — ``heads=[0]`` in the threshold call and
# ``head_idx=1`` in the lookback :class:`DeclarativeAttentionHeadSpec`
# — which made adding a future L2 head fragile (the author had to
# remember which slots were already taken). With the allocator the
# handoff is structural: each bake instantiates its OWN allocator
# pre-loaded with the full L2 head layout (pinned to existing slots),
# stashes it on ``attn._l2_head_allocator`` for downstream inspection,
# and resolves its own head index by name. A future L2 attention op
# can claim a free head via ``allocator.alloc(name, layer_idx=2)`` --
# with no ``pin=`` -- without touching this table.
_L2_HEAD_LAYOUT = (
    # (op-name key,                          pinned head_idx)
    ("layer2_threshold_attn",                0),
    ("layer2_lookback_detection_head",       1),
)


def _allocate_layer2_attention_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L2 heads.

    Every entry in :data:`_L2_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying primitive calls -- which still write
    the same weights to the same heads -- land byte-identically. Both
    ``layer2_threshold_attn`` and ``layer2_lookback_detection_head``
    call this so each can look up its own head by name without baking
    in an integer literal at the call site.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L2_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=2, pin=head_idx)
    return allocator


def _l2_head_idx(op_name: str) -> int:
    """Return the pinned L2 ``head_idx`` for ``op_name``.

    Static lookup against :data:`_L2_HEAD_LAYOUT` for callers that
    cannot instantiate a per-bake allocator (e.g. ``compiler_ir_factory``
    helpers, which run outside the bake and receive only dim positions
    and ``HD``). The runtime bakes still go through
    :func:`_allocate_layer2_attention_heads` so the collision-checked
    allocator path is exercised on every weight write.
    """
    for name, head_idx in _L2_HEAD_LAYOUT:
        if name == op_name:
            return head_idx
    raise KeyError(f"_l2_head_idx: unknown L2 attention op {op_name!r}")


# === L2 FFN unit layout (pinned offsets) ============================
#
# Two ops share the L2 FFN today:
#
#   * ``layer2_mem_byte_flags`` (phase=2) owns units 0..7 -- 4 MEM_VAL_B*
#     flags + 4 BYTE_INDEX_*/STACK0_BYTE* flags.
#   * ``layer2_initial_pc_bake_cancel`` (phase=2.5) owns units 8..9 -- a
#     pair of EMBED_LO/EMBED_HI cancel taps for the initial-PC bake.
#
# Pre-migration the second op picked up its start unit from a stateful
# ``ffn._l2_unit_counter`` written by the first op. With the allocator
# the handoff is structural: each bake instantiates its OWN allocator
# pre-loaded with the full L2 layout (pinned to existing offsets), so
# the unit indices are reproducible without depending on cross-op
# attribute writes. Both bakes still stash the allocator on
# ``ffn._l2_unit_allocator`` so downstream tooling can inspect the
# layout, mirroring the L9 convention.
_L2_FFN_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer2_mem_byte_flags.flags",         0, 8),  # MEM_VAL + BYTE_INDEX
    ("layer2_initial_pc_bake_cancel.lo",    8, 1),  # EMBED_LO cancel
    ("layer2_initial_pc_bake_cancel.hi",    9, 1),  # EMBED_HI cancel
)


def _allocate_layer2_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L2 FFN sub-stages.

    Every sub-stage is pinned at its existing offset so the underlying
    bake helpers -- which still write the same weights to the same
    indices -- land byte-identically. Both ``layer2_mem_byte_flags`` and
    ``layer2_initial_pc_bake_cancel`` call this so each can look up its
    own start unit by name without depending on a cross-op
    ``_l2_unit_counter`` attribute. A future L2 FFN op family can claim
    a free range past unit 10 via ``allocator.alloc(name, n)`` without a
    pin.
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L2_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


# === L2 FFN rule families ===========================================
#
# ``layer2_mem_byte_flags`` (units 0..7) and ``layer2_initial_pc_bake_cancel``
# (units 8..9) are now expressed as ``FFNRule`` lists so the verifier and
# symbolic execution see the same declarations the lowering uses. The bake
# paths still drive ``Primitives.lower_ffn_rules`` with the same pinned
# ``start_unit`` offsets owned by ``_L2_FFN_UNIT_LAYOUT`` so the underlying
# weight cells land byte-identically.
#
# Phase 6 wave 3C migration.

_L2_MEM_FLAGS_TRANSITIONS = (
    # (src_dim_name_with_offset, blocker_dim_name_with_offset,
    #  out_dim_names_with_offset_tuple)
    ("H1+4",   "H0+4",   ("MEM_VAL_B0",)),
    ("L2H0+4", "H1+4",   ("MEM_VAL_B1",)),
    ("L1H4+4", "L2H0+4", ("MEM_VAL_B2",)),
    ("H2+4",   "L1H4+4", ("MEM_VAL_B3",)),
    ("L1H4+3", "H1+3",   ("BYTE_INDEX_0",)),
    ("H2+3",   "L1H4+3", ("BYTE_INDEX_1", "STACK0_BYTE1")),
    ("H3+3",   "H2+3",   ("BYTE_INDEX_2", "STACK0_BYTE2")),
    ("H4+3",   "H3+3",   ("BYTE_INDEX_3", "STACK0_BYTE3")),
)


def _layer2_mem_byte_flags_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule list mirroring ``_bake_layer2_mem_byte_flags``.

    Each unit fires when ``src_dim + IS_BYTE >= 1.5`` (i.e. the byte slot
    that immediately follows the marker, identified by the H-threshold
    pair) and the blocker ``H``-output is OFF (so only the boundary slot
    is selected, not every byte slot in the run). The gate computes
    ``1.0 - blocker``, so ``hidden = silu(S/2) * (1.0 - blocker) ~= S/2``
    at the boundary and ``0`` deeper into the run.
    """
    write_scale = 2.0 / S
    rules = []
    for idx, (src_dim, blocker_dim, out_dims) in enumerate(
        _L2_MEM_FLAGS_TRANSITIONS
    ):
        rules.append(FFNRule.gated_write(
            name=f"layer2_mem_byte_flags_{idx}_{out_dims[0].lower()}",
            conditions=(
                (src_dim, 1.0),
                ("IS_BYTE", 1.0),
            ),
            threshold=1.5,
            gate=None,
            gate_terms=((blocker_dim, -1.0),),
            gate_bias=1.0,
            writes=tuple((out_dim, write_scale) for out_dim in out_dims),
        ))
    return tuple(rules)


def _layer2_mem_byte_flags_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer2_mem_byte_flags_rules(S))
    return ir


def _bake_layer2_mem_byte_flags(ffn, S, BD, start_unit=0):
    """Thin compatibility shim around :func:`_layer2_mem_byte_flags_rules`.

    Lowers the FFNRule list through ``Primitives.lower_ffn_rules`` at
    ``start_unit``. Kept as the public entry point for parity tests in
    ``tests/test_declarative_ffn_bakes_l1_l5.py`` and any other consumer
    that drives the L2 mem-byte-flags bake directly.

    ``start_unit`` is the pinned hidden-unit base supplied by the
    :class:`FFNUnitAllocator`. Callers pass ``start_unit=0`` today (the
    pinned offset for ``layer2_mem_byte_flags.flags``); the parameter
    exists so the helper does not bake in an implicit ``0`` -- the
    layout is owned by ``_L2_FFN_UNIT_LAYOUT``.
    """
    rules = _layer2_mem_byte_flags_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD, Primitives.ffn_rule_dim_names(rules)
    )
    return Primitives.lower_ffn_rules(
        ffn, rules, dim_positions, start_unit=start_unit, S=S,
    )


def make_layer2_mem_byte_flags_op() -> Operation:
    """L2 FFN: MEM val byte position flags + extended BYTE_INDEX for STACK0.

    Uses units [0, 8) pinned via :class:`FFNUnitAllocator`. The allocator
    is stashed on ``ffn._l2_unit_allocator`` so downstream ops (e.g.
    ``make_layer2_initial_pc_bake_cancel_op``) can inspect or extend the
    layout. The legacy ``ffn._l2_unit_counter`` cross-op handoff is gone
    -- the cancel op now resolves its own start from
    ``_L2_FFN_UNIT_LAYOUT``, not from a mutated FFN attribute.

    Phase 6 wave 3C: the per-unit weight writes are now driven by an
    ``FFNRule`` list (``_layer2_mem_byte_flags_rules``) lowered through
    ``Primitives.lower_ffn_rules``. The IR is also exposed via
    ``compiler_ir`` so the declarative verifier and
    ``compare_symbolic_to_lowered_ffn`` see the same declarations.
    """
    def bake(ffn, dim_positions, S):
        # Per-bake FFN-unit allocator. Pin every existing L2 range so the
        # underlying weight writes land byte-identically; expose the
        # allocator for inspection on the FFN module.
        allocator = _allocate_layer2_ffn_units()
        flags_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer2_mem_byte_flags.flags"
        )
        ffn._l2_unit_allocator = allocator
        rules = _layer2_mem_byte_flags_rules(S)
        named_positions = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(
            ffn,
            rules,
            named_positions,
            start_unit=flags_range.start,
            S=S,
        )

    # Dim-ownership claims: ``_layer2_mem_byte_flags_rules`` writes units 0..7
    # (one per transition in ``_L2_MEM_FLAGS_TRANSITIONS``). Each unit writes
    # a unique W_down output dim:
    #   unit 0: MEM_VAL_B0
    #   unit 1: MEM_VAL_B1
    #   unit 2: MEM_VAL_B2
    #   unit 3: MEM_VAL_B3
    #   unit 4: BYTE_INDEX_0  (STACK0 byte 0)
    #   unit 5: BYTE_INDEX_1 + STACK0_BYTE1  (STACK0 byte 1)
    #   unit 6: BYTE_INDEX_2 + STACK0_BYTE2  (STACK0 byte 2)
    #   unit 7: BYTE_INDEX_3 + STACK0_BYTE3  (STACK0 byte 3)
    _claims = set()
    _outputs = [
        "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
        "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
    ]
    for u, out_dim in enumerate(_outputs):
        _claims.add((2, "ffn_W_down", str(u), f"{out_dim}+0"))
    for u, out_dim in (
        (5, "STACK0_BYTE1"),
        (6, "STACK0_BYTE2"),
        (7, "STACK0_BYTE3"),
    ):
        _claims.add((2, "ffn_W_down", str(u), f"{out_dim}+0"))

    return Operation(
        name="layer2_mem_byte_flags",
        phase=2,
        reads={"H0", "H1", "H4", "IS_BYTE", "BYTE_INDEX_0",
               "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"},
        writes={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
                "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
                "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3"},
        kind="ffn",
        layer_idx=2,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer2_mem_byte_flags_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        # ``_layer2_mem_byte_flags_rules`` produces 8 hidden units (4
        # MEM_VAL_B* + 4 BYTE_INDEX_*) pinned at offset 0 via
        # ``_L2_FFN_UNIT_LAYOUT``. The cancel op below resolves its own
        # start (unit 8) from the same shared layout.
        ffn_units_used=8,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer2_initial_pc_bake_cancel_rules(S: float) -> tuple[FFNRule, ...]:
    """FFNRule list mirroring the legacy two-unit initial-PC cancel.

    Each unit fires when ``HAS_SE >= 0.5`` (i.e. step >= 1, since HAS_SE
    is 0 at the first step and 1 afterwards) and is gated by MARK_PC so
    only PC-marker rows write. The W_down strength ``-2.0 / S`` cancels
    the +1.0 EMBED bake from ``make_initial_pc_bake_op``.

    The output offsets are derived from runtime ``PC_OFFSET`` so the
    cancel slots match the token-embedding bake exactly.
    """
    from ...constants import PC_OFFSET
    init_pc_lo = PC_OFFSET & 0xF
    init_pc_hi = (PC_OFFSET >> 4) & 0xF
    write_scale = -2.0 / S

    return (
        FFNRule.gated_write(
            name="layer2_initial_pc_bake_cancel_lo",
            conditions=(("HAS_SE", 1.0),),
            threshold=0.5,
            gate="MARK_PC",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"EMBED_LO+{init_pc_lo}", write_scale),),
        ),
        FFNRule.gated_write(
            name="layer2_initial_pc_bake_cancel_hi",
            conditions=(("HAS_SE", 1.0),),
            threshold=0.5,
            gate="MARK_PC",
            gate_weight=1.0,
            gate_bias=0.0,
            writes=((f"EMBED_HI+{init_pc_hi}", write_scale),),
        ),
    )


def _layer2_initial_pc_bake_cancel_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer2_initial_pc_bake_cancel_rules(S))
    return ir


def make_layer2_initial_pc_bake_cancel_op() -> Operation:
    """L2 FFN: cancel the REG_PC token-embedding initial-PC bake at MARK_PC AND HAS_SE.

    The token-embedding bake ``make_initial_pc_bake_op`` (phase=1001.5) adds
    ``EMBED_LO[PC_OFFSET & 0xF] += 1.0`` and ``EMBED_HI[(PC_OFFSET >> 4) & 0xF]
    += 1.0`` at EVERY REG_PC token. This is intentional for step 0 (the
    first PC marker needs PC=PC_OFFSET in EMBED so L4 attention can relay
    it to AX for L5's first-step opcode fetch), but at step 1+ it pollutes
    the residual stream — specifically EMBED_LO[init_pc_lo] persists at
    +1.0 even though the carry-forward attention has written the real
    prev-PC nibbles into EMBED_LO/HI.

    The legacy fix inlined two cancel units in ``_set_layer3_ffn`` (vm_step
    lines ~2598-2609) that subtract from EMBED_LO/HI at MARK_PC AND
    HAS_SE. Those cancels live in the SAME FFN block as the PC INCREMENT
    units, so the FFN's input residual still carries the +1.0 spurious
    contribution — and PC INCREMENT (which reads EMBED_LO[k] for every
    nibble k) leaks a phantom +1.0 contribution at OUTPUT_LO[
    (init_pc_lo + INSTR_WIDTH) % 16]. For PC_OFFSET=2, INSTR_WIDTH=8 the
    leak lands at OUTPUT_LO[10], aliasing with the FIRST-STEP DEFAULT's
    target (also OUTPUT_LO[10]) and corrupting step-1+ PC byte 0 by
    +INSTR_WIDTH, breaking ``test_two_imms``, ``test_jmp_from_step_2``,
    ``test_jmp_backward``, and the smoke ``test_add_basic`` multi-step.

    Cancelling the bake at L2 FFN (which runs strictly BEFORE L3 FFN)
    makes L3 see EMBED_LO[init_pc_lo]=0 at step 1+ MARK_PC, eliminating
    the leak entirely. Step 0's MARK_PC has HAS_SE=0 (no STEP_END in
    context yet), so the cancel does not fire and EMBED_LO[init_pc_lo]
    remains +1.0 for L4 first-step relay. The strength is -2.0/S × 50 =
    -1.0 (exact match for the +1.0 bake).

    Pinned to ``layer_idx=2``, ``kind="block"`` (so the bake can access
    the same FFN as ``make_layer2_mem_byte_flags_op``). Phase=2.5 so it
    runs after ``make_layer2_mem_byte_flags_op`` (phase=2). The unit
    indices (8 and 9) are now resolved through the shared
    :class:`FFNUnitAllocator` layout in ``_L2_FFN_UNIT_LAYOUT`` rather
    than via the historical ``ffn._l2_unit_counter`` handoff.

    Phase 6 wave 3C: the two cancel units are now expressed as
    ``FFNRule.gated_write`` declarations (see
    ``_layer2_initial_pc_bake_cancel_rules``). The bake lowers the rules
    through ``Primitives.lower_ffn_rules`` at the pinned start unit,
    and the ``compiler_ir`` attribute exposes the same declarations to
    ``compare_symbolic_to_lowered_ffn``.
    """
    def bake(block, dim_positions, S):
        ffn = block.ffn
        # Per-bake allocator with the full L2 FFN layout pinned. Looking
        # up the cancel-op ranges by name keeps the start unit identical
        # to the legacy ``_l2_unit_counter`` value (8 / 9) without
        # depending on any prior op having mutated the FFN module.
        allocator = _allocate_layer2_ffn_units()
        ffn._l2_unit_allocator = allocator
        lo_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer2_initial_pc_bake_cancel.lo"
        )
        hi_range = next(
            r for r in allocator.ranges()
            if r.op_name == "layer2_initial_pc_bake_cancel.hi"
        )
        assert hi_range.start == lo_range.start + 1, (
            "L2 initial_pc_bake_cancel: lo/hi must be contiguous to lower "
            "via a single Primitives.lower_ffn_rules call"
        )

        rules = _layer2_initial_pc_bake_cancel_rules(S)
        named_positions = Primitives.dim_positions_from_bd(
            _as_setdim_proxy(dim_positions),
            Primitives.ffn_rule_dim_names(rules),
        )
        Primitives.lower_ffn_rules(
            ffn,
            rules,
            named_positions,
            start_unit=lo_range.start,
            S=S,
        )

    # Dim-ownership claims: two FFN units pinned at indices 8 and 9 via
    # ``_L2_FFN_UNIT_LAYOUT`` (the same offsets the legacy
    # ``_l2_unit_counter`` handoff produced after _set_layer2_mem_byte_flags).
    # Each writes one EMBED nibble. PC_OFFSET is a runtime constant from
    # constants.py, so resolve init_pc_lo/hi at op-construction time for
    # the claim columns.
    from ...constants import PC_OFFSET as _PC_OFFSET
    _init_pc_lo = _PC_OFFSET & 0xF
    _init_pc_hi = (_PC_OFFSET >> 4) & 0xF
    _claims = set()
    # Unit 8: cancels EMBED_LO[init_pc_lo]. W_down[EMBED_LO+init_pc_lo, 8].
    _claims.add((2, "ffn_W_down", "8", f"EMBED_LO+{_init_pc_lo}"))
    # Unit 9: cancels EMBED_HI[init_pc_hi].
    _claims.add((2, "ffn_W_down", "9", f"EMBED_HI+{_init_pc_hi}"))

    return Operation(
        name="layer2_initial_pc_bake_cancel",
        phase=2.5,
        reads={"MARK_PC", "HAS_SE"},
        writes={"EMBED_LO", "EMBED_HI"},
        kind="block",
        layer_idx=2,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_layer2_initial_pc_bake_cancel_ir(),
        migrated=True,
        claims=_claims,
        declarative_authority="spec_generated",
        # Allocates 2 FFN units pinned at indices 8 and 9 via
        # ``_L2_FFN_UNIT_LAYOUT`` (same offsets the legacy
        # ``ffn._l2_unit_counter`` handoff produced). Result: writes
        # units 8 and 9 -> max index 10. The aggregator takes the
        # per-block max across all annotated ops, so reporting 10 here
        # covers both.
        ffn_units_used=10,
        # B12 backfill (wave 1c): basic ``after`` reference per
        # docs/B12_BACKFILL_SPEC.md §26 (manual-judgment bucket; the
        # register-aware ``produces`` model is deferred). This op cancels
        # ``phase_a_ffn``'s initial-PC token-embedding bake, so the strict
        # ``after`` edge is the load-bearing constraint. ``layer_pin=2``
        # via ``layer_idx`` already keeps it on L2 during the static path;
        # the dynamic scheduler honours the requires["after"] edge.
        requires={"after": "phase_a_ffn"},
        step_idx={0},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer2_threshold_attn_op() -> Operation:
    """L2 attention: threshold 5.5 head."""
    def bake(attn, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)
        ALIBI_S = 10.0
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator with the full L2 head layout
        # pinned. Looking up the threshold head by name keeps its index
        # identical to the legacy ``heads=[0]`` literal without baking
        # the integer into the call site.
        allocator = _allocate_layer2_attention_heads()
        attn._l2_head_allocator = allocator
        threshold_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer2_threshold_attn"
        )
        Primitives.generate_threshold_attention_heads(
            attn, [5.5], [proxy.L2H0], ALIBI_S, HD,
            heads=[threshold_head_idx], bd=proxy,
        )

    # Dim-ownership claims: 1 threshold head on L2 attn, head 0 writing L2H0.
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    for m, mark in enumerate(_MARKS):
        _claims.add((2, "attn_W_v", f"0_{1 + m}", f"{mark}+0"))
        _claims.add((2, "attn_W_o", f"0_{1 + m}", f"L2H0+{m}"))
    _claims.add((2, "attn_W_q", "0_0", "CONST+0"))
    _claims.add((2, "attn_W_k", "0_0", "IS_MARK+0"))

    return Operation(
        name="layer2_threshold_attn",
        phase=2,
        reads={"IS_MARK", "CONST"},
        writes={"L2H0"},
        kind="attn",
        layer_idx=2,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer2_threshold_ir,
        migrated=True,
        claims=_claims,
        # B12 backfill (wave 1b): pin strictly after layer1_threshold_attn.
        # Same shape as L1: reads IS_MARK/CONST from an upstream marker
        # bake (no in-DAG predecessor), so the structural L0->L1->L2
        # threshold-attn-stack ordering needs explicit after-edges to
        # surface in the dep graph. Chaining via layer1_threshold_attn
        # (rather than layer0_threshold_attn) gives this op dep_depth=2,
        # matching current_layer=2 and reaching the phase_pinned_by_deps
        # bucket. See docs/B12_BACKFILL_SPEC.md §24 recommended choice (a).
        requires={"after": "layer1_threshold_attn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer2_threshold_ir(dim_positions, HD) -> CompilerIR:
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    # Pull the head index from the shared L2 layout so the IR path and
    # the bake path stay in lockstep -- no integer literal here.
    ir.layer(0).attention.extend(Primitives.threshold_attention_head_specs(
        [5.5],
        [proxy.L2H0],
        10.0,
        HD,
        heads=[_l2_head_idx("layer2_threshold_attn")],
        bd=proxy,
    ))
    return ir


def make_layer2_lookback_detection_head_op(
    enable_conversational_io: bool = False,
) -> Operation:
    """L2 attention head 1: detect previous token type for conversational I/O.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``enable_conversational_io``):
        attn2.alibi_slopes[1] = 10.0
        _set_lookback_detection_head(attn2, S, BD, HD)

    Migrated as ``kind="block"`` pinned to ``layer_idx=2`` with
    ``migrated=True``. Phase=2.1 so this runs AFTER
    ``make_layer2_threshold_attn_op`` (phase=2), which fills all alibi
    slopes to 10.0 — the explicit ``[1] = 10.0`` is therefore a no-op
    today but preserved verbatim for parity with the legacy inline
    setup.

    The bake is unconditional in shape (always registered to keep the
    dep-graph stable), but the body is a no-op when
    ``enable_conversational_io`` is False so no weights are touched
    outside of conversational-I/O mode.
    """
    def bake(block, dim_positions, S):
        del S
        if not enable_conversational_io:
            return
        attn = block.attn
        # Per-bake attention-head allocator with the full L2 head layout
        # pinned. Resolving the lookback head by name reproduces the
        # legacy ``head_idx=1`` literal without depending on the
        # threshold op having already populated ``attn._l2_head_allocator``.
        allocator = _allocate_layer2_attention_heads()
        attn._l2_head_allocator = allocator
        lookback_head_idx = next(
            h.head_idx for h in allocator.heads()
            if h.op_name == "layer2_lookback_detection_head"
        )
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            # Steep slope to favor most recent token. Index from the
            # allocator so the slot stays in sync with the spec below.
            attn.alibi_slopes[lookback_head_idx] = 10.0
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer2_lookback_detection_head_spec(proxy),
            HD,
        )

    return Operation(
        name="layer2_lookback_detection_head",
        phase=2.1,
        # Reads: CONST (Q/K gate), MARK_THINKING_START/END + IS_BYTE (V copy).
        # Writes go to LAST_WAS_THINKING_START/END/BYTE which are not
        # declared in declare_setdim_compat_dims (conversational-I/O-only
        # dims); the bake resolves them via the _SetDim fallback in
        # _as_setdim_proxy, so no compiler-tracked write edge is needed.
        reads={"CONST", "MARK_THINKING_START", "MARK_THINKING_END", "IS_BYTE"},
        writes=set(),
        kind="block",
        layer_idx=2,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _layer2_lookback_detection_head_ir
            if enable_conversational_io else None
        ),
        declarative_authority="spec_generated",
        migrated=True,
        # B12 backfill (wave 1b): pin strictly after layer1_threshold_attn.
        # Flag-gated stub (enable_conversational_io); writes={} leaves the
        # dep DAG with no derived placement. The structural intent is
        # "L2-block resident, after the L1 threshold block has run." The
        # spec's §25 alternative (after layer2_threshold_attn) would push
        # dep_depth to 3 and trigger phase_inconsistent_with_deps at
        # current_layer=2; using layer1_threshold_attn keeps dep_depth at
        # the L2 floor (= current_layer) and reaches phase_pinned_by_deps.
        requires={"after": "layer1_threshold_attn"},
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer2_lookback_detection_head_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer2_lookback_detection_head_spec(proxy))
    return ir


def _layer2_lookback_detection_head_spec(BD) -> DeclarativeAttentionHeadSpec:
    L = 20.0
    # Pull the head index from the shared L2 layout rather than baking
    # in a ``head_idx=1`` literal here. Both the bake and IR paths
    # consult the same source of truth, so renumbering the layout in
    # one place stays consistent across every consumer.
    return DeclarativeAttentionHeadSpec(
        head_idx=_l2_head_idx("layer2_lookback_detection_head"),
        q=(AP(0, BD.CONST, L),),
        k=(AP(0, BD.CONST, L),),
        v=(
            AP(1, BD.MARK_THINKING_START, 1.0),
            AP(2, BD.MARK_THINKING_END, 1.0),
            AP(3, BD.IS_BYTE, 1.0),
        ),
        o=(
            AO(BD.LAST_WAS_THINKING_START, 1, 1.0),
            AO(BD.LAST_WAS_THINKING_END, 2, 1.0),
            AO(BD.LAST_WAS_BYTE, 3, 1.0),
        ),
    )
