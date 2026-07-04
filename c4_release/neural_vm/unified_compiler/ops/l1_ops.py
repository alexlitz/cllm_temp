"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule
from ..ir import CompilerIR, FFNRule, StepWindowConstraint
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy
from ..positional_invariant import invariant_threshold, marker_bank_index


# === L1 FFN unit layout (auto-fit offsets, Phase 7.B.1) =============
#
# The ``layer1_ffn`` op owns the entire L1 FFN. The weight writes happen
# via ``_threshold_ffn_rules`` (a list of :class:`FFNRule` declarations)
# which is lowered by ``Primitives.lower_ffn_rules`` through
# ``CompilerIR.lower_ffn``. Each rule consumes one hidden unit and they
# are appended monotonically starting at ``start_unit=0`` -- so the 5
# rules land on units 0..4 in the order declared. The actual FFN weight
# writes are bound to the ``lower_ffn_rules`` cursor (always 0..4 in
# declaration order), NOT to the allocator, so dropping the pins is
# purely bookkeeping.
#
# Phase 7.B.1: ``pin`` is dropped from every entry. The allocator is
# constructed in ``"dynamic_first_fit"`` mode and walks the layout in
# declaration order; first-fit on a 4096-wide pool lands the 5
# single-unit sub-stages back at indices 0..4 -- byte-identical to the
# legacy pins -- but the author no longer supplies the offsets. A
# future L1 op family can claim a free range past unit 5 via
# ``allocator.alloc(name, n)`` without a pin.
#
# The order below mirrors the rule order in ``_threshold_ffn_rules``
# (STACK0_BYTE0 followed by the four BYTE_INDEX_i thresholds).
# Changing the rule list requires updating this table in lock-step.
_THRESHOLD_FFN_UNIT_LAYOUT = (
    # (sub-stage name, n_units) -- ``pin=None`` everywhere; offsets are
    # picked by the FFNUnitAllocator in ``dynamic_first_fit`` mode.
    ("layer1_ffn.stack0_byte0",   1),  # STACK0_BYTE0 from L1H4 + IS_BYTE
    ("layer1_ffn.byte_index_0",   1),  # BYTE_INDEX_0 from L1H1 vs L1H0
    ("layer1_ffn.byte_index_1",   1),  # BYTE_INDEX_1 from L1H2 vs L1H1
    ("layer1_ffn.byte_index_2",   1),  # BYTE_INDEX_2 from H0   vs L1H2
    ("layer1_ffn.byte_index_3",   1),  # BYTE_INDEX_3 from H1   vs H0
)


def _allocate_threshold_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L1 FFN sub-stages.

    Phase 7.B.1 auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode with no ``pin=`` hints. Each sub-stage
    is allocated in declaration order; first-fit picks the lowest free
    unit each call, so the 5 single-unit sub-stages land at indices
    0..4 -- byte-identical to the legacy pinned offsets. The
    ``_threshold_ffn_rules`` lowering still drives weight writes at
    ``start_unit=0`` via ``Primitives.lower_ffn_rules`` so the
    allocator's pick is bookkeeping only.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L1 op claims a free range past unit 5).
    """
    allocator = FFNUnitAllocator(strategy="dynamic_first_fit")
    for name, n_units in _THRESHOLD_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def _threshold_ffn_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for the L1 FFN (STACK0_BYTE0 + BYTE_INDEX_0..3).

    Mirrors the legacy ``_set_layer1_ffn`` helper one-for-one:

    * Unit 0 — STACK0_BYTE0 flag at d=6 from BP: fires when
      ``L1H4[BP]`` (d<=6.5) and ``IS_BYTE`` both fire, blocked by
      ``H1[BP]`` (d>4.5). Implements
      ``silu(S*(L1H4_BP + IS_BYTE - 1.5)) * (1 - H1_BP)``.
    * Units 1..4 — marker-agnostic BYTE_INDEX_0..3 flags. Each fires
      when ``IS_BYTE`` and any-marker ``src_base[*]`` band activate,
      blocked by the lower any-marker ``blocker_base[*]`` band.

    The 7 (NUM_MARKERS) marker conditions per BYTE_INDEX_i are
    summed in the up branch with weight 1.0 each, and similarly in
    the gate branch with weight -1.0 each. Only one marker type is
    nearest at any position, so the sum is ~1 when active. The
    threshold/scale constants here (S, 1.5 threshold, 2.0/S write
    weight) are unchanged from the imperative helper.

    Phase 8.D: The four ``BYTE_INDEX_i`` writes use :func:`dim_ref`
    for the role-meaningful ``(byte_index, "i")`` family lookups --
    each rule's output dim is a slot whose semantic role is the
    byte index it asserts. The per-marker ``H*/L1H*+{i}`` reads stay
    structural (``i`` is a marker-bank slot index, a structural
    position into a fixed-width threshold-head bank, not a
    role-meaningful byte index).
    """
    # Marker-bank slot index, resolved through the positional-invariant
    # mechanism (Class-1 marker-relative anchor) instead of the literal
    # ``BP_I = 3``. ``marker_bank_index`` proves the slot is frame-INVARIANT
    # (the threshold-head bank is keyed on marker TYPE, whose order does not
    # change when the STACK0 *value* block is dropped) and is the single
    # source of truth the audit recognises as declared-invariant.
    BP_I = marker_bank_index("BP")
    NM = 7  # NUM_MARKERS — fixed-width threshold-head bank
    write_scale = 2.0 / S

    rules: list[FFNRule] = []

    # Unit 0: STACK0_BYTE0 = L1H4[BP] AND IS_BYTE AND NOT H1[BP].
    #
    # STACK0_BYTE0 is a d=6-from-BP POSITIONAL flag (L1H4[BP] => d<=6.5, IS_BYTE,
    # NOT H1[BP] => d>4.5). It fires on STACK0 byte 0 (offset 21 = d=6 from BP in
    # the 35-token layout). Under C4_NO_STACK0_EMIT the STACK0 register block is
    # dropped, so the MEM block shifts 5 tokens earlier and MEM addr byte 0 lands
    # at d=6 from BP — exactly where this flag fires. Left as-is, the positional
    # gate would MISFIRE onto MEM addr byte 0 and corrupt every consumer that
    # reads STACK0_BYTE0 (L3/L10/L11/L14/L16). With the operand-A read re-routed
    # to mem[SP] (C4_OPERAND_FROM_MEMSP) and the STACK0 emission gone, nothing
    # legitimately needs this flag, so neutralize it: keep the unit allocated
    # (5-unit byte-count guard) but raise the AND threshold unreachably high so
    # it never fires.
    #
    # Class-2 ABSOLUTE-SLOT anchor: this flag's target (STACK0 byte 0, d=6 from
    # BP) is the canonical anchor whose row VANISHES when STACK0 is dropped.
    # ``invariant_threshold`` AUTO-COMPUTES the live-vs-suppressed threshold
    # from ``Token.STEP_TOKENS`` — no per-op ``no_stack0_emit_enabled()``
    # branch. At STEP_TOKENS=35 it returns 1.5 (byte-identical to HEAD/golden);
    # at STEP_TOKENS=30 it auto-returns 1e9 (the make-unreachable suppress),
    # reproducing the prior hand-coded fix with ZERO hand-tuning. This is the
    # systematic mechanism replacing the cluster-by-cluster re-anchor; see
    # ``neural_vm/unified_compiler/positional_invariant.py`` +
    # ``docs/POSITIONAL_INVARIANT_MECHANISM_2026_06_20.md``.
    _stack0_byte0_threshold = invariant_threshold(
        live=1.5, suppressed=1.0e9, marker="BP", k=6,
    )
    rules.append(multi_way_and_rule(
        name="stack0_byte0_flag",
        conditions=(
            (f"L1H4+{BP_I}", 1.0),
            ("IS_BYTE", 1.0),
        ),
        threshold=_stack0_byte0_threshold,
        gate_terms=((f"H1+{BP_I}", -1.0),),
        gate_bias=1.0,
        writes=(("STACK0_BYTE0", write_scale),),
    ))

    # Units 1..4: BYTE_INDEX_0..3 = IS_BYTE AND any(src_base) AND NOT any(blocker_base).
    # Phase 8.D: pre-compute the four (byte_index, role) refs so the
    # rule writes name the semantic family lookup, not the bare slot
    # name. Each ``byte_index_<n>`` binding resolves to
    # ``"BYTE_INDEX_<n>+0"`` -- byte-identical to the legacy ``out_dim``
    # tuple-table strings via :meth:`DimRef.parse`.
    byte_index_0 = dim_ref("byte_index", "0")
    byte_index_1 = dim_ref("byte_index", "1")
    byte_index_2 = dim_ref("byte_index", "2")
    byte_index_3 = dim_ref("byte_index", "3")
    for src_base, blocker_base, out_dim in (
        ("L1H1", "L1H0", byte_index_0),
        ("L1H2", "L1H1", byte_index_1),
        ("H0",   "L1H2", byte_index_2),
        ("H1",   "H0",   byte_index_3),
    ):
        conditions = [("IS_BYTE", 1.0)]
        gate_terms = []
        for i in range(NM):
            conditions.append((f"{src_base}+{i}", 1.0))
            gate_terms.append((f"{blocker_base}+{i}", -1.0))
        # Recover the human-readable name (drops the "+0"). Semantic-naming
        # wave: the rule label is the function-descriptive ``byte_index_<n>_flag``
        # (was ``l1_byte_index_<n>``); these units compute the positional
        # BYTE_INDEX_<n> flags read by the L1 FFN consumers.
        out_name = out_dim.split("+", 1)[0].lower()
        rules.append(multi_way_and_rule(
            name=f"{out_name}_flag",
            conditions=tuple(conditions),
            threshold=1.5,
            gate_terms=tuple(gate_terms),
            gate_bias=1.0,
            writes=((out_dim, write_scale),),
        ))

    return tuple(rules)


def _threshold_ffn_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_threshold_ffn_rules(S))
    return ir


def _bake_layer1_ffn(ffn, S, BD) -> int:
    """Bake L1 FFN via :class:`FFNRule` lowering (byte-identical helper).

    Used both by the migrated :func:`make_threshold_ffn_op` bake path and as
    a standalone entry-point for callers wanting to drive the L1 weight
    writes without constructing a full ``Operation``.
    """
    rules = _threshold_ffn_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def make_threshold_ffn_op() -> Operation:
    """L1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags from threshold differences.

    Originally: `_set_layer1_ffn` at vm_step.py:2922. Migrated to
    :class:`FFNRule` declarations + ``compiler_ir`` in Phase 6 wave 3B.

    Reads L1H0/L1H1/L1H2/L1H4/H0/H1 threshold outputs and IS_BYTE.
    Writes STACK0_BYTE0, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3.
    """
    def bake(ffn, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator (Phase 7.B.1 auto-fit). The allocator
        # is built in ``dynamic_first_fit`` mode without pin hints; the
        # 5 single-unit sub-stages land at indices 0..4 by first-fit, the
        # same offsets the legacy pins produced. Stashed on the FFN module
        # so downstream tools (e.g. a future L1 op family claiming a free
        # gap) can inspect or extend the layout. Mirrors the L9 convention
        # from ca775eb.
        allocator = _allocate_threshold_ffn_units()
        ffn._l1_unit_allocator = allocator

        # Phase 8.C inline cut: lower the ``_threshold_ffn_rules`` IR
        # directly here so census v2 classifies this op as
        # ``declarative`` rather than ``declarative_via_helper`` (which
        # routed through the ``_bake_layer1_ffn`` trampoline).
        # Byte-identical to the prior ``_bake_layer1_ffn(ffn, S, proxy)``.
        rules = _threshold_ffn_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy,
            Primitives.ffn_rule_dim_names(rules),
        )
        n0 = Primitives.lower_ffn_rules(ffn, rules, rule_dim_positions, S=S)
        # Byte-identity guard: the FFNRule lowering MUST write exactly the
        # number of hidden units the allocator table declares. Mirrors the
        # L0 phase_a_ffn assertion in ``_bake_phase_a_ffn``.
        expected_total = sum(n for _, n in _THRESHOLD_FFN_UNIT_LAYOUT)
        assert n0 == expected_total, (
            f"L1 layer1_ffn unit cursor drift: rules wrote {n0} units, "
            f"allocator declared {expected_total}"
        )

    # Dim-ownership claims: L1 FFN writes 5 units at fixed positions:
    #   unit 0: STACK0_BYTE0
    #   unit 1: BYTE_INDEX_0
    #   unit 2: BYTE_INDEX_1
    #   unit 3: BYTE_INDEX_2
    #   unit 4: BYTE_INDEX_3
    _claims = set()
    _outputs = [
        "STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
        "BYTE_INDEX_2", "BYTE_INDEX_3",
    ]
    for u, out_dim in enumerate(_outputs):
        _claims.add((1, "ffn_W_down", str(u), f"{out_dim}+0"))

    return Operation(
        name="layer1_ffn",
        # Phase 7.A.1: L1H0/L1H1/L1H2/L1H4 are written by
        # ``layer1_threshold_attn`` at the SAME layer (L1 attn substage feeds
        # the L1 FFN substage inside the same transformer block). Express
        # this as a co-placement constraint via
        # ``requires["same_layer_as"]`` rather than a depth-bumping dataflow
        # edge, so analyze_scheduler.py and LayerCompiler._assign_layers
        # agree on the layer assignment. H0/H1 (written by
        # ``layer0_threshold_attn`` at L0) and IS_BYTE (markers from an
        # earlier setup) remain genuine cross-layer reads. The bake_fn
        # itself still consumes the L1H* dims from the residual; reads= is
        # declarative metadata only.
        reads={"H0", "H1", "IS_BYTE"},
        writes={"STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
                "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        # Phase 8.G.6: drop ``layer_idx=1`` literal. The ``requires
        # ["same_layer_as"]`` below co-places this op with the L1
        # threshold-attn (which lands at L1 via ``requires["after"]``
        # against layer0_threshold_attn), so the explicit layer pin is
        # redundant.
        declarative_bake_fn=bake,
        compiler_ir=_threshold_ffn_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        requires={"same_layer_as": "layer1_threshold_attn"},
        # ``_set_layer1_ffn`` writes 5 units (one per output: STACK0_BYTE0,
        # BYTE_INDEX_0..3). See setup_helpers.py:_set_layer1_ffn.
        ffn_units_used=5,
        postcondition={
            "STACK0_BYTE0": "0_or_1",
        },
        # Wave 4 (docs/PRODUCES_CONSUMES_MIGRATION.md): L1 token-feed
        # pipeline op. Writes STACK0_BYTE0 + BYTE_INDEX_0..3 — cross-
        # step durables (BYTE_INDEX_* on the _CROSS_STEP_DURABLE
        # allowlist; STACK0_BYTE0 is a per-position stable flag). No
        # in-step register-slot surface.
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


_THRESHOLD_HEAD_LAYOUT = (
    # (op_name,)  -- no pinned head_idx; allocator first-fits in
    # declaration order, landing at 0..6 byte-identically (Phase 7.B.2 attn).
    #
    # Heads 0..2: fine threshold heads producing L1H0/L1H1/L1H2 (thresholds
    # 0.5/1.5/2.5 against IS_MARK).
    # Head 3: HAS_SE global STEP_END detector (slope=0 disables ALiBi decay).
    # Head 4: threshold 6.5 producing L1H4 (STACK0 byte 0 identification).
    # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay producer (positive
    # ALiBi slope 0.5 so the softmax1 anchor wins as distance grows).
    # Head 6 (2026-06-10): STEP_END register-presence broadcast --
    # within-step relay of MARK_AX (and scaffolded slots for the other
    # register markers) from its own marker row into the matching
    # ``SE_REG_<NAME>_PRESENT`` slot at MARK_SE_ONLY. L0/L1 foundation
    # of the STEP_END compute migration; see
    # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md.
    ("layer1_threshold_attn.l1h0",),
    ("layer1_threshold_attn.l1h1",),
    ("layer1_threshold_attn.l1h2",),
    ("layer1_threshold_attn.has_se",),
    ("layer1_threshold_attn.l1h4",),
    ("layer1_threshold_attn.in_step_fresh",),
    ("layer1_threshold_attn.step_end_reg_present",),
)


def _allocate_threshold_attn_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` for the L1 heads.

    Phase 7.B.2 attn auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode without ``pin=`` hints. Each entry is
    allocated in declaration order; first-fit picks the lowest free
    head each call, so the 6 declarations land at indices 0..5 --
    byte-identical to the legacy pinned offsets. The alibi-slope
    overrides in the bake key off the allocator-resolved indices, so
    HAS_SE / IN_STEP_FRESH follow their semantic heads across any
    future allocator reshuffle.

    Returns the allocator so callers can inspect or extend it.
    """
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (op_name,) in _THRESHOLD_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=1)
    return allocator


def make_threshold_attn_op() -> Operation:
    """L1 attention: 3 fine threshold heads + STEP_END + L1H4 + IN_STEP_FRESH.

    Head 5 (B7-1) emits ``IN_STEP_FRESH``: a positive in-step lifecycle bit
    that decays from ~1.0 immediately after the most-recent ``MARK_SE_ONLY``
    (or ``MARK_CS`` at program start) toward 0.0 as more tokens accumulate
    within the current step, and resets to ~1.0 at the next STEP_END. The
    head uses the same Q/K/V/O shape as head 3 (HAS_SE) but with a positive
    ALiBi slope (``IN_STEP_FRESH_ALIBI_S = 0.5``) so the softmax1 anchor
    overtakes the score as the distance to the most-recent SE grows. See
    ``_SetDim.IN_STEP_FRESH`` docstring and
    ``investigation/l7-l9-structural-audit:REPORT.md`` Section 2.4 for the
    full design rationale and B5-J regression context.
    """
    def bake(attn, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake attention-head allocator. Every existing head_idx is
        # pinned at its current slot so the calls below land
        # byte-identically. Stashed on the attention module so downstream
        # tools (e.g. a future L1 op claiming a free gap past head 6) can
        # inspect or extend the layout. Mirrors the ``_l1_unit_allocator``
        # FFN convention from ``make_threshold_ffn_op``.
        head_allocator = _allocate_threshold_attn_heads()
        attn._l1_head_allocator = head_allocator
        h_l1h0 = head_allocator.heads()[0].head_idx
        h_l1h1 = head_allocator.heads()[1].head_idx
        h_l1h2 = head_allocator.heads()[2].head_idx
        h_has_se = head_allocator.heads()[3].head_idx
        h_l1h4 = head_allocator.heads()[4].head_idx
        h_in_step_fresh = head_allocator.heads()[5].head_idx
        h_step_end_reg = head_allocator.heads()[6].head_idx

        ALIBI_S = 10.0
        # B7-1: ALiBi slope chosen so IN_STEP_FRESH decays from ~1.0 right
        # after a STEP_END toward ~0 within the 35-token step window. The
        # head's base attention score for a single MARK_SE_ONLY key is
        # ``CONST_w * SE_w / sqrt(HD) = 10 * 10 / sqrt(64) = 12.5`` (matches
        # head 3 / HAS_SE). With slope 0.5 the score at distance d is
        # ``12.5 - 0.5 * d``; combined with softmax1's anchor=0 the output
        # crosses 0.5 near d=25 (about 70% through a 35-token step) and is
        # < 0.01 by d=35. Matches the L9 ALiBi memory-lookup slope for ABI
        # consistency (see test_alibi_mem_attn.py for the precedent).
        IN_STEP_FRESH_ALIBI_S = 0.5
        # Head 6 ALiBi slope: the broadcast head reads from MARK_AX
        # (single-K-hit per step), so a positive slope cleanly bounds
        # the broadcast to the current step. The MARK_AX-to-MARK_SE
        # within-step distance is 29 tokens (rows 5 -> 34); the
        # prior-step MARK_AX sits at distance 64 (rows 5 -> 34 + 35).
        # Q*K score base = 10 * 10 / sqrt(64) = 12.5. With slope 0.2
        # the within-step score is 12.5 - 5.8 = 6.7 and the prior-step
        # score is 12.5 - 12.8 = -0.3 -- a ratio of exp(7) ~= 1100 in
        # favour of within-step. softmax1 with anchor 0 then writes
        # ~exp(6.7)/(1 + exp(6.7) + exp(-0.3)) ~= 0.99 at MARK_SE,
        # dropping to ~5% the moment the same head fires beyond a step
        # boundary. Same shape and weight scale as head 3 / head 5.
        STEP_END_REG_ALIBI_S = 0.2
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[h_has_se] = 0.0  # global SE detection
            attn.alibi_slopes[h_in_step_fresh] = IN_STEP_FRESH_ALIBI_S  # B7-1: decay
            attn.alibi_slopes[h_step_end_reg] = STEP_END_REG_ALIBI_S  # SE relay
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_threshold_attention_heads(
            attn,
            [0.5, 1.5, 2.5],
            [proxy.L1H0, proxy.L1H1, proxy.L1H2],
            ALIBI_S,
            HD,
            heads=[h_l1h0, h_l1h1, h_l1h2],
            bd=proxy,
        )
        # Head 3: STEP_END existence detection (global). The spec is built
        # by the single-source ``_has_se_head_spec`` helper so the bake and
        # ``_threshold_attn_ir`` paths never drift (mirrors the shared
        # ``threshold_attention_head_specs`` + ``_step_end_reg_present_head_spec``
        # convention already used in this op).
        Primitives.generate_attention_head(
            attn,
            _has_se_head_spec(proxy, head_idx=h_has_se),
            HD,
        )
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        Primitives.generate_threshold_attention_heads(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[h_l1h4], bd=proxy,
        )
        # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE decay. Same Q/K/V/O
        # shape as head 3 but uses positive ALiBi slope so the attention
        # weight on the most-recent MARK_SE_ONLY (or MARK_CS at program
        # start) decays as the query position moves further from it. The
        # softmax1 ZFOD anchor produces output ~0 when no SE/CS lies
        # within ALiBi range. Built by the single-source
        # ``_in_step_fresh_head_spec`` helper so the bake and IR paths
        # share one definition.
        Primitives.generate_attention_head(
            attn,
            _in_step_fresh_head_spec(proxy, head_idx=h_in_step_fresh),
            HD,
        )
        # Head 6 (2026-06-10): STEP_END all-register-presence broadcast.
        # Q anchors on MARK_SE_ONLY (Q[0] = 10 only at the SE row); K
        # anchors as the OR of 5 register markers (MARK_AX / MARK_PC /
        # MARK_SP / MARK_BP / MARK_STACK0) -- each contributes K[0] =
        # 10. With ALiBi slope 0.2 the within-step markers (distances
        # 14..34 from SE) dominate softmax1 over prior-step markers
        # (distances 49..69) by ~exp(3)>=~20. The 5 within-step
        # markers split softmax1 mass unevenly by distance (closer
        # markers get more mass); per-marker V_GAINs invert the
        # mass split so each SE_REG_<NAME>_PRESENT lands at ~1.0
        # at the SE row (see ``_step_end_reg_present_head_spec``
        # docstring for the calibrated constants). This is the
        # L0/L1 foundation of the STEP_END compute migration -- the
        # parallel Wave-A L11 relay broadcasts the OP_<NAME> /
        # AX_CARRY / ALU / CMP / STACK0_BYTE slots that L0/L1 cannot
        # yet see (those are written by L3+/L5+/L8+/L9+ producers
        # downstream of L1). See
        # docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md.
        Primitives.generate_attention_head(
            attn,
            _step_end_reg_present_head_spec(proxy, head_idx=h_step_end_reg),
            HD,
        )

    # Dim-ownership claims: 7 heads on L1 attn.
    #   Heads 0,1,2,4: threshold heads writing to L1H0/L1H1/L1H2/L1H4
    #                  (each writes V slots 1..7 across MARKS).
    #   Head 3: STEP_END detector — Q[CONST], K[MARK_SE_ONLY],
    #           V[3*HD+1, MARK_SE_ONLY], O[HAS_SE, 3*HD+1].
    #   Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay — Q[CONST],
    #           K[MARK_SE_ONLY+MARK_CS], V[1, MARK_SE_ONLY+MARK_CS],
    #           O[IN_STEP_FRESH, 5*HD+1].
    #   Head 6 (2026-06-10): STEP_END all-register-presence broadcast —
    #           Q[MARK_SE_ONLY],
    #           K[MARK_AX+MARK_PC+MARK_SP+MARK_BP+MARK_STACK0],
    #           V[MARK_AX,MARK_PC,MARK_SP,MARK_BP,MARK_STACK0] (5 slots),
    #           O[SE_REG_{AX,PC,SP,BP,STACK0}_PRESENT, 6*HD+(1..5)].
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    # Threshold heads 0, 1, 2, 4 → L1H0, L1H1, L1H2, L1H4.
    for h, out_base in [(0, "L1H0"), (1, "L1H1"), (2, "L1H2"), (4, "L1H4")]:
        for m, mark in enumerate(_MARKS):
            _claims.add((1, "attn_W_v", f"{h}_{1 + m}", f"{mark}+0"))
            _claims.add((1, "attn_W_o", f"{h}_{1 + m}", f"{out_base}+{m}"))
        _claims.add((1, "attn_W_q", f"{h}_0", "CONST+0"))
        _claims.add((1, "attn_W_k", f"{h}_0", "IS_MARK+0"))
    # Head 3: STEP_END global detector.
    _claims.add((1, "attn_W_q", "3_0", "CONST+0"))
    _claims.add((1, "attn_W_k", "3_0", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_v", "3_1", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_o", "3_1", "HAS_SE+0"))
    # Head 5 (B7-1): IN_STEP_FRESH recency decay.
    _claims.add((1, "attn_W_q", "5_0", "CONST+0"))
    _claims.add((1, "attn_W_k", "5_0", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_k", "5_0", "MARK_CS+0"))
    _claims.add((1, "attn_W_v", "5_1", "MARK_SE_ONLY+0"))
    _claims.add((1, "attn_W_v", "5_1", "MARK_CS+0"))
    _claims.add((1, "attn_W_o", "5_1", "IN_STEP_FRESH+0"))
    # Head 6 (2026-06-10): STEP_END all-register-presence broadcast.
    # The W_o claims are deliberately omitted: SE_REG_<NAME>_PRESENT
    # slots share their compiled residual positions with other
    # compact-layout dims via the liveness-based slot-share pass, so
    # the verifier's ``_pos_to_column`` cannot uniquely resolve the
    # destination columns back to ``SE_REG_<NAME>_PRESENT+0``.
    # Declaring the writes would produce spurious DECLARATION DRIFT
    # entries (the verifier matches the column to whichever sharing
    # partner sorts first alphabetically). Q/K/V claims still record
    # the head's identity for the dim-ownership audit.
    _claims.add((1, "attn_W_q", "6_0", "MARK_SE_ONLY+0"))
    for _i, _mark in enumerate(
        ("MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0"),
        start=1,
    ):
        _claims.add((1, "attn_W_k", "6_0", f"{_mark}+0"))
        _claims.add((1, "attn_W_v", f"6_{_i}", f"{_mark}+0"))

    return Operation(
        name="layer1_threshold_attn",
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the MARK_* dims threshold heads 0, 1, 2, 4 read via ``bd.MARKS``
        # (V slot 1+m reads MARKS[m] for m=0..6 = MARK_PC, MARK_AX,
        # MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS). Previously only
        # MARK_SE_ONLY + MARK_CS were declared (head 3 / head 5 inputs).
        # Head 6 (2026-06-10): also reads MARK_SE_ONLY (Q-gate) and
        # MARK_AX/MARK_PC/MARK_SP/MARK_BP/MARK_STACK0 (K/V projection).
        # The original threshold-head MARKS list already covers all
        # 5 except MARK_STACK0, which head 6 adds.
        reads={"IS_MARK", "MARK_SE_ONLY", "MARK_CS", "CONST",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "MARK_MEM", "MARK_SE", "MARK_STACK0"},
        writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE", "IN_STEP_FRESH",
                "SE_REG_AX_PRESENT", "SE_REG_PC_PRESENT",
                "SE_REG_SP_PRESENT", "SE_REG_BP_PRESENT",
                "SE_REG_STACK0_PRESENT"},
        kind="attn",
        # Phase 8.G.6: drop ``layer_idx=1`` literal. ``requires["after"]
        # = layer0_threshold_attn`` (below) is the structural pin: the
        # dep edge forces this attn to land at L1 (one after L0).
        declarative_bake_fn=bake,
        compiler_ir_factory=_threshold_attn_ir,
        migrated=True,
        claims=_claims,
        # B12 backfill (wave 1b): pin strictly after L0's threshold-attn
        # block. Both L0 and L1 read IS_MARK from an earlier marker bake
        # (neither writes it), so the dep DAG cannot derive the structural
        # L0->L1 transformer-block ordering on its own. B12 manual-judgment
        # audit confirmed L0 writes {H0..H7} and reads {IS_MARK, CONST} —
        # there is no missing IS_MARK write to add on L0; the predecessor
        # gap is a true structural pin, not a declaration bug.
        # See docs/B12_BACKFILL_SPEC.md §27.
        requires={"after": "layer0_threshold_attn"},
        # Wave 4 (docs/PRODUCES_CONSUMES_MIGRATION.md): L1 token-feed
        # attention op (3 fine threshold heads + HAS_SE + L1H4 +
        # IN_STEP_FRESH). Derive yields empty (attention-only IR);
        # reads are all marker/CONST cross-step embed-time dims. No
        # in-step register-slot surface.
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _has_se_head_spec(
    proxy, *, head_idx: int,
) -> DeclarativeAttentionHeadSpec:
    """L1 head 3: global STEP_END existence detector (HAS_SE).

    Q anchors on ``CONST`` (fires at every row); K/V anchor on
    ``MARK_SE_ONLY``; O lifts the copied V slot into ``HAS_SE``.
    ``step_window=ANY_STEP`` because HAS_SE is a global existence flag
    that fires across every prior step's SE marker (the bake pins
    ``alibi_slopes[h_has_se] = 0.0`` so this head opts out of the L1-wide
    ALiBi decay).

    Single source of truth for both the bake path
    (:func:`make_threshold_attn_op`) and the ``compiler_ir_factory``
    (:func:`_threshold_attn_ir`) -- mirrors the
    ``threshold_attention_head_specs`` / ``_step_end_reg_present_head_spec``
    convention so the two paths cannot drift.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, proxy.CONST, 10.0),),
        k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
        v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
        o=(AO(proxy.HAS_SE, 1, 1.0),),
        step_window=StepWindowConstraint.ANY_STEP,
    )


def _in_step_fresh_head_spec(
    proxy, *, head_idx: int,
) -> DeclarativeAttentionHeadSpec:
    """L1 head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay.

    Same Q/K/V/O shape as :func:`_has_se_head_spec` but the K/V bank is
    the OR of ``MARK_SE_ONLY`` and ``MARK_CS`` (the program-start anchor),
    and the bake pins a POSITIVE ALiBi slope (``IN_STEP_FRESH_ALIBI_S =
    0.5``) so the attention weight on the most-recent SE/CS decays as the
    query row moves further from it. The softmax1 ZFOD anchor yields
    output ~0 when no SE/CS lies within ALiBi range.

    Single source of truth for both the bake path and the IR factory.
    """
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, proxy.CONST, 10.0),),
        k=(
            AP(0, proxy.MARK_SE_ONLY, 10.0),
            AP(0, proxy.MARK_CS, 10.0),
        ),
        v=(
            AP(1, proxy.MARK_SE_ONLY, 1.0),
            AP(1, proxy.MARK_CS, 1.0),
        ),
        o=(AO(proxy.IN_STEP_FRESH, 1, 1.0),),
    )


def _step_end_reg_present_head_spec(
    proxy, *, head_idx: int,
) -> DeclarativeAttentionHeadSpec:
    """L1 head 6: within-step all-register presence broadcast to MARK_SE.

    Multi-marker relay: Q anchors at ``MARK_SE_ONLY`` (Q[0] = 10
    only at the SE row); K anchors as the OR of 5 register markers
    (``MARK_AX``, ``MARK_PC``, ``MARK_SP``, ``MARK_BP``,
    ``MARK_STACK0``) -- each marker row contributes K[0] = 10. With
    5 simultaneous K hits, softmax1's mass distribution is shaped
    by ALiBi-decayed scores: the closer marker rows (STACK0 at
    distance 14 from SE) receive larger attention mass than the
    farther markers (PC at distance 34). Each register's V slot
    fires only at its own marker row (``V[i]`` = ``V_GAIN_<NAME>``
    at ``MARK_<NAME>``, 0 elsewhere); O lifts ``V[i]`` into the
    matching ``SE_REG_<NAME>_PRESENT`` slot at the SE row.

    Per-marker ``V_GAIN`` calibration: the ALiBi slope 0.2 splits
    softmax mass across the 5 within-step markers as approximately
    [PC=0.012, AX=0.032, SP=0.086, BP=0.234, STACK0=0.636] (within
    a step of 35 tokens where PC sits at distance 34 from SE,
    AX at 29, SP at 24, BP at 19, STACK0 at 14). Setting
    ``V_GAIN_<NAME>`` = 1/mass yields each ``SE_REG_<NAME>_PRESENT``
    at ~1.0 (presence-detection threshold > 0.5). The calibration
    is step-structure invariant (each step has the same marker
    offsets), not step-number invariant, so it holds across the
    program. Empirically derived from the probe trace 2026-06-10.

    With ALiBi slope 0.2 the within-step markers (distances 14..34
    from SE) dominate softmax1 over the prior-step markers
    (distances 49..69 from SE) by a factor of >=~exp(3) ~= 20; the
    prior-step contribution to each ``SE_REG_<NAME>_PRESENT`` slot
    is <5% of the within-step contribution.

    Wires all 5 ``SE_REG_<NAME>_PRESENT`` slots (AX/PC/SP/BP/STACK0)
    in a single head -- the L0/L1 foundation of the STEP_END compute
    migration: every register's presence is visible at MARK_SE for
    downstream L11+ consumers to gate on. The L11 Wave A relay
    broadcasts the post-producer compute slots (OP_<NAME>, AX_CARRY,
    ALU, CMP, STACK0_BYTE0..3) that L0/L1 cannot see yet. See
    docs/STEP_END_COMPUTE_ARCHITECTURE_2026_06_10.md §4 (Wave A).

    Downstream consumers MUST gate on ``MARK_SE_ONLY`` to filter
    out the V_GAIN-scaled leakage at non-SE marker rows. The
    leakage is a softmax-split artefact of the 5-way K-bank
    multi-marker design (at Q rows where Q[0]=0, softmax1 with
    anchor=0 gives ~1/(1+5) weight per K row, multiplied by the
    matching V_GAIN). For example,
    SE_REG_PC_PRESENT @ MARK_PC ~= V_GAIN_PC / 6 ~= 14 (vs. ~1.0
    at MARK_SE), so any reader of SE_REG_PC_PRESENT must constrain
    to MARK_SE_ONLY positions.

    ``SE_REG_MEM_PRESENT`` remains declared-but-unwritten: MARK_MEM
    fires multiple times per step (one per memory access byte), and
    its multi-row contribution would dilute the softmax mass for the
    other markers below the presence threshold. A dedicated head
    can be added if MEM-presence is ever needed by a downstream op.
    """

    L = 10.0
    # Per-marker V_GAIN calibrated to invert ALiBi-shaped softmax
    # mass at Q@MARK_SE. The constants below are derived from the
    # 2026-06-10 probe trace (slope=0.2, in-step distances
    # PC=34, AX=29, SP=24, BP=19, STACK0=14): mass[name] ~=
    # exp(12.5 - slope * dist) / sum(exp(...)) and V_GAIN[name] =
    # 1/mass[name]. Each SE_REG_<NAME>_PRESENT lands at ~1.0.
    V_GAIN_PC = 84.0
    V_GAIN_AX = 32.0
    V_GAIN_SP = 12.0
    V_GAIN_BP = 4.3
    V_GAIN_STACK0 = 1.6
    return DeclarativeAttentionHeadSpec(
        head_idx=head_idx,
        q=(AP(0, proxy.MARK_SE_ONLY, L),),
        k=(
            AP(0, proxy.MARK_AX, L),
            AP(0, proxy.MARK_PC, L),
            AP(0, proxy.MARK_SP, L),
            AP(0, proxy.MARK_BP, L),
            AP(0, proxy.MARK_STACK0, L),
        ),
        v=(
            AP(1, proxy.MARK_AX, V_GAIN_AX),
            AP(2, proxy.MARK_PC, V_GAIN_PC),
            AP(3, proxy.MARK_SP, V_GAIN_SP),
            AP(4, proxy.MARK_BP, V_GAIN_BP),
            AP(5, proxy.MARK_STACK0, V_GAIN_STACK0),
        ),
        o=(
            AO(proxy.SE_REG_AX_PRESENT, 1, 1.0),
            AO(proxy.SE_REG_PC_PRESENT, 2, 1.0),
            AO(proxy.SE_REG_SP_PRESENT, 3, 1.0),
            AO(proxy.SE_REG_BP_PRESENT, 4, 1.0),
            AO(proxy.SE_REG_STACK0_PRESENT, 5, 1.0),
        ),
    )


def _threshold_attn_ir(dim_positions, HD) -> CompilerIR:
    """``compiler_ir_factory`` for ``layer1_threshold_attn``.

    Resolves head indices via :func:`_allocate_threshold_attn_heads` so the
    IR and the bake share a single source of truth. After the Phase
    7.B.2-attn pin drop the allocator runs in ``dynamic_first_fit``
    mode; with no other claimants on the L1 attention pool the 7
    declaration-order entries always land at indices 0..6 -- so the
    IR's spec walk reproduces the legacy ``heads=[0,1,2]``+3+``[4]``+5
    schedule byte-identically and adds head 6 (the STEP_END register-
    presence broadcast) at slot 6.
    """

    proxy = _as_setdim_proxy(dim_positions)
    ALIBI_S = 10.0
    allocator = _allocate_threshold_attn_heads()
    # Resolve each L1 head index by op-name so the spec walk follows
    # the allocator across any future reshuffle.
    by_name = {rec.op_name: rec.head_idx for rec in allocator.heads()}
    h_l1h0 = by_name["layer1_threshold_attn.l1h0"]
    h_l1h1 = by_name["layer1_threshold_attn.l1h1"]
    h_l1h2 = by_name["layer1_threshold_attn.l1h2"]
    h_has_se = by_name["layer1_threshold_attn.has_se"]
    h_l1h4 = by_name["layer1_threshold_attn.l1h4"]
    h_in_step_fresh = by_name["layer1_threshold_attn.in_step_fresh"]
    h_step_end_reg = by_name["layer1_threshold_attn.step_end_reg_present"]
    ir = CompilerIR()
    specs = list(Primitives.threshold_attention_head_specs(
        [0.5, 1.5, 2.5],
        [proxy.L1H0, proxy.L1H1, proxy.L1H2],
        ALIBI_S,
        HD,
        heads=[h_l1h0, h_l1h1, h_l1h2],
        bd=proxy,
    ))
    specs.append(_has_se_head_spec(proxy, head_idx=h_has_se))
    specs.extend(Primitives.threshold_attention_head_specs(
        [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[h_l1h4], bd=proxy,
    ))
    # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay.
    specs.append(_in_step_fresh_head_spec(proxy, head_idx=h_in_step_fresh))
    # Head 6 (2026-06-10): STEP_END register-presence broadcast.
    specs.append(_step_end_reg_present_head_spec(
        proxy, head_idx=h_step_end_reg,
    ))
    ir.layer(0).attention.extend(specs)
    return ir
