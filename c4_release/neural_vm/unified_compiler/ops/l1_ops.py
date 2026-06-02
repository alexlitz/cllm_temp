"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L1 FFN unit layout (auto-fit offsets, Phase 7.B.1) =============
#
# The ``layer1_ffn`` op owns the entire L1 FFN. The weight writes happen
# via ``_layer1_ffn_rules`` (a list of :class:`FFNRule` declarations)
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
# The order below mirrors the rule order in ``_layer1_ffn_rules``
# (STACK0_BYTE0 followed by the four BYTE_INDEX_i thresholds).
# Changing the rule list requires updating this table in lock-step.
_L1_FFN_UNIT_LAYOUT = (
    # (sub-stage name, n_units) -- ``pin=None`` everywhere; offsets are
    # picked by the FFNUnitAllocator in ``dynamic_first_fit`` mode.
    ("layer1_ffn.stack0_byte0",   1),  # STACK0_BYTE0 from L1H4 + IS_BYTE
    ("layer1_ffn.byte_index_0",   1),  # BYTE_INDEX_0 from L1H1 vs L1H0
    ("layer1_ffn.byte_index_1",   1),  # BYTE_INDEX_1 from L1H2 vs L1H1
    ("layer1_ffn.byte_index_2",   1),  # BYTE_INDEX_2 from H0   vs L1H2
    ("layer1_ffn.byte_index_3",   1),  # BYTE_INDEX_3 from H1   vs H0
)


def _allocate_layer1_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L1 FFN sub-stages.

    Phase 7.B.1 auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode with no ``pin=`` hints. Each sub-stage
    is allocated in declaration order; first-fit picks the lowest free
    unit each call, so the 5 single-unit sub-stages land at indices
    0..4 -- byte-identical to the legacy pinned offsets. The
    ``_layer1_ffn_rules`` lowering still drives weight writes at
    ``start_unit=0`` via ``Primitives.lower_ffn_rules`` so the
    allocator's pick is bookkeeping only.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L1 op claims a free range past unit 5).
    """
    allocator = FFNUnitAllocator(strategy="dynamic_first_fit")
    for name, n_units in _L1_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def _layer1_ffn_rules(S: float) -> tuple[FFNRule, ...]:
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
    BP_I = 3
    NM = 7  # NUM_MARKERS — fixed-width threshold-head bank
    write_scale = 2.0 / S

    rules: list[FFNRule] = []

    # Unit 0: STACK0_BYTE0 = L1H4[BP] AND IS_BYTE AND NOT H1[BP].
    rules.append(FFNRule.gated_write(
        name="l1_stack0_byte0",
        conditions=(
            (f"L1H4+{BP_I}", 1.0),
            ("IS_BYTE", 1.0),
        ),
        threshold=1.5,
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
        # Recover the human-readable name (drops the "+0" so the rule
        # name keeps the legacy ``l1_byte_index_<n>`` form).
        out_name = out_dim.split("+", 1)[0].lower()
        rules.append(FFNRule.gated_write(
            name=f"l1_{out_name}",
            conditions=tuple(conditions),
            threshold=1.5,
            gate_terms=tuple(gate_terms),
            gate_bias=1.0,
            writes=((out_dim, write_scale),),
        ))

    return tuple(rules)


def _layer1_ffn_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_layer1_ffn_rules(S))
    return ir


def _bake_layer1_ffn(ffn, S, BD) -> int:
    """Bake L1 FFN via :class:`FFNRule` lowering (byte-identical helper).

    Used both by the migrated :func:`make_layer1_ffn_op` bake path and as
    a standalone entry-point for callers wanting to drive the L1 weight
    writes without constructing a full ``Operation``.
    """
    rules = _layer1_ffn_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def make_layer1_ffn_op() -> Operation:
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
        allocator = _allocate_layer1_ffn_units()
        ffn._l1_unit_allocator = allocator

        n0 = _bake_layer1_ffn(ffn, S, proxy)
        # Byte-identity guard: the FFNRule lowering MUST write exactly the
        # number of hidden units the allocator table declares. Mirrors the
        # L0 phase_a_ffn assertion in ``_bake_phase_a_ffn``.
        expected_total = sum(n for _, n in _L1_FFN_UNIT_LAYOUT)
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
        phase=1,
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
        layer_idx=1,
        declarative_bake_fn=bake,
        compiler_ir=_layer1_ffn_ir(),
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
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


_L1_HEAD_LAYOUT = (
    # (op_name,)  -- no pinned head_idx; allocator first-fits in
    # declaration order, landing at 0..5 byte-identically (Phase 7.B.2 attn).
    #
    # Heads 0..2: fine threshold heads producing L1H0/L1H1/L1H2 (thresholds
    # 0.5/1.5/2.5 against IS_MARK).
    # Head 3: HAS_SE global STEP_END detector (slope=0 disables ALiBi decay).
    # Head 4: threshold 6.5 producing L1H4 (STACK0 byte 0 identification).
    # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay producer (positive
    # ALiBi slope 0.5 so the softmax1 anchor wins as distance grows).
    ("layer1_threshold_attn.l1h0",),
    ("layer1_threshold_attn.l1h1",),
    ("layer1_threshold_attn.l1h2",),
    ("layer1_threshold_attn.has_se",),
    ("layer1_threshold_attn.l1h4",),
    ("layer1_threshold_attn.in_step_fresh",),
)


def _allocate_layer1_attn_heads() -> AttentionHeadAllocator:
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
    for (op_name,) in _L1_HEAD_LAYOUT:
        allocator.alloc(op_name, layer_idx=1)
    return allocator


def make_layer1_threshold_attn_op() -> Operation:
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
        # tools (e.g. a future L1 op claiming a free gap past head 5) can
        # inspect or extend the layout. Mirrors the ``_l1_unit_allocator``
        # FFN convention from ``make_layer1_ffn_op``.
        head_allocator = _allocate_layer1_attn_heads()
        attn._l1_head_allocator = head_allocator
        h_l1h0 = head_allocator.heads()[0].head_idx
        h_l1h1 = head_allocator.heads()[1].head_idx
        h_l1h2 = head_allocator.heads()[2].head_idx
        h_has_se = head_allocator.heads()[3].head_idx
        h_l1h4 = head_allocator.heads()[4].head_idx
        h_in_step_fresh = head_allocator.heads()[5].head_idx

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
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(ALIBI_S)
            attn.alibi_slopes[h_has_se] = 0.0  # global SE detection
            attn.alibi_slopes[h_in_step_fresh] = IN_STEP_FRESH_ALIBI_S  # B7-1: decay
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
        # Head 3: STEP_END existence detection (global)
        Primitives.generate_attention_head(
            attn,
            DeclarativeAttentionHeadSpec(
                head_idx=h_has_se,
                q=(AP(0, proxy.CONST, 10.0),),
                k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
                v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
                o=(AO(proxy.HAS_SE, 1, 1.0),),
            ),
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
        # within ALiBi range.
        Primitives.generate_attention_head(
            attn,
            DeclarativeAttentionHeadSpec(
                head_idx=h_in_step_fresh,
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
            ),
            HD,
        )

    # Dim-ownership claims: 6 heads on L1 attn.
    #   Heads 0,1,2,4: threshold heads writing to L1H0/L1H1/L1H2/L1H4
    #                  (each writes V slots 1..7 across MARKS).
    #   Head 3: STEP_END detector — Q[CONST], K[MARK_SE_ONLY],
    #           V[3*HD+1, MARK_SE_ONLY], O[HAS_SE, 3*HD+1].
    #   Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay — Q[CONST],
    #           K[MARK_SE_ONLY+MARK_CS], V[1, MARK_SE_ONLY+MARK_CS],
    #           O[IN_STEP_FRESH, 5*HD+1].
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

    return Operation(
        name="layer1_threshold_attn",
        phase=1,
        reads={"IS_MARK", "MARK_SE_ONLY", "MARK_CS", "CONST"},
        writes={"L1H0", "L1H1", "L1H2", "L1H4", "HAS_SE", "IN_STEP_FRESH"},
        kind="attn",
        layer_idx=1,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer1_threshold_ir,
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
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer1_threshold_ir(dim_positions, HD) -> CompilerIR:
    """``compiler_ir_factory`` for ``layer1_threshold_attn``.

    Resolves head indices via :func:`_allocate_layer1_attn_heads` so the
    IR and the bake share a single source of truth. After the Phase
    7.B.2-attn pin drop the allocator runs in ``dynamic_first_fit``
    mode; with no other claimants on the L1 attention pool the 6
    declaration-order entries always land at indices 0..5 -- so the
    IR's spec walk reproduces the legacy ``heads=[0,1,2]``+3+``[4]``+5
    schedule byte-identically.
    """

    proxy = _as_setdim_proxy(dim_positions)
    ALIBI_S = 10.0
    allocator = _allocate_layer1_attn_heads()
    # Resolve each L1 head index by op-name so the spec walk follows
    # the allocator across any future reshuffle.
    by_name = {rec.op_name: rec.head_idx for rec in allocator.heads()}
    h_l1h0 = by_name["layer1_threshold_attn.l1h0"]
    h_l1h1 = by_name["layer1_threshold_attn.l1h1"]
    h_l1h2 = by_name["layer1_threshold_attn.l1h2"]
    h_has_se = by_name["layer1_threshold_attn.has_se"]
    h_l1h4 = by_name["layer1_threshold_attn.l1h4"]
    h_in_step_fresh = by_name["layer1_threshold_attn.in_step_fresh"]
    ir = CompilerIR()
    specs = list(Primitives.threshold_attention_head_specs(
        [0.5, 1.5, 2.5],
        [proxy.L1H0, proxy.L1H1, proxy.L1H2],
        ALIBI_S,
        HD,
        heads=[h_l1h0, h_l1h1, h_l1h2],
        bd=proxy,
    ))
    specs.append(DeclarativeAttentionHeadSpec(
        head_idx=h_has_se,
        q=(AP(0, proxy.CONST, 10.0),),
        k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
        v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
        o=(AO(proxy.HAS_SE, 1, 1.0),),
    ))
    specs.extend(Primitives.threshold_attention_head_specs(
        [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[h_l1h4], bd=proxy,
    ))
    # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay.
    specs.append(DeclarativeAttentionHeadSpec(
        head_idx=h_in_step_fresh,
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
    ))
    ir.layer(0).attention.extend(specs)
    return ir
