"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


# === L1 FFN unit layout (pinned offsets) ============================
#
# The ``layer1_ffn`` op owns the entire L1 FFN. The weight writes happen
# inside ``_bake_layer1_ffn`` below, which uses a local ``unit = 0``
# counter that increments through 5 sub-stages (one unit per output
# dim). Migration to :class:`FFNUnitAllocator` keeps the helper
# byte-identical -- we just declare each sub-stage's range at its
# existing pinned offset so the layout is auditable rather than
# implicit. Adding a new L1 op family later will go through
# ``allocator.alloc(name, n)`` without a pin, and the allocator will
# pick the first free gap past unit 5.
#
# The offsets below mirror the unit-counter walk in ``_bake_layer1_ffn``
# (STACK0_BYTE0 followed by the four BYTE_INDEX_i thresholds).
# Changing the helper's unit count requires updating this table in
# lock-step.
_L1_FFN_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("layer1_ffn.stack0_byte0",   0, 1),  # STACK0_BYTE0 from L1H4 + IS_BYTE
    ("layer1_ffn.byte_index_0",   1, 1),  # BYTE_INDEX_0 from L1H1 vs L1H0
    ("layer1_ffn.byte_index_1",   2, 1),  # BYTE_INDEX_1 from L1H2 vs L1H1
    ("layer1_ffn.byte_index_2",   3, 1),  # BYTE_INDEX_2 from H0   vs L1H2
    ("layer1_ffn.byte_index_3",   4, 1),  # BYTE_INDEX_3 from H1   vs H0
)


def _allocate_layer1_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L1 FFN sub-stages.

    Every sub-stage is pinned at its existing offset so the underlying
    ``_bake_layer1_ffn`` helper -- which writes via its own monotonic
    ``unit = 0`` counter -- lands on exactly the same hidden-unit
    indices it always has. This call is byte-identical bookkeeping: the
    allocator declares ranges by name, the helper writes the weights. A
    future refactor can split the monolithic helper into per-range bake
    functions that consume ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L1 op claims a free range past unit 5).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _L1_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


def make_layer1_ffn_op() -> Operation:
    """L1 FFN: STACK0_BYTE0 flag + BYTE_INDEX flags from threshold differences.

    Originally: `_set_layer1_ffn` at vm_step.py:2922.

    Reads L1H0/L1H1/L1H2/L1H4/H0/H1 threshold outputs and IS_BYTE.
    Writes STACK0_BYTE0, BYTE_INDEX_0, BYTE_INDEX_1, BYTE_INDEX_2, BYTE_INDEX_3.
    """
    def bake(ffn, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L1 FFN sub-stage is pinned to
        # its existing offset so the call below lands byte-identically.
        # Stashed on the FFN module so downstream tools (e.g. a future
        # L1 op family claiming a free gap) can inspect or extend the
        # layout. Mirrors the L9 convention from ca775eb.
        allocator = _allocate_layer1_ffn_units()
        ffn._l1_unit_allocator = allocator

        _bake_layer1_ffn(ffn, S, proxy)

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
        reads={"L1H0", "L1H1", "L1H2", "L1H4", "H0", "H1", "IS_BYTE"},
        writes={"STACK0_BYTE0", "BYTE_INDEX_0", "BYTE_INDEX_1",
                "BYTE_INDEX_2", "BYTE_INDEX_3"},
        kind="ffn",
        layer_idx=1,
        bake_fn=bake,
        declarative_bake_fn=bake,
        migrated=True,
        claims=_claims,
        # ``_set_layer1_ffn`` writes 5 units (one per output: STACK0_BYTE0,
        # BYTE_INDEX_0..3). See setup_helpers.py:_set_layer1_ffn.
        ffn_units_used=5,
        postcondition={
            "STACK0_BYTE0": "0_or_1",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _bake_layer1_ffn(ffn, S, BD):
    """Declarative L1 FFN spec: STACK0_BYTE0 and byte-index flags."""

    BP_I = 3
    NM = BD.NUM_MARKERS
    unit = 0

    ffn.W_up.data[unit, BD.L1H4 + BP_I] = S
    ffn.W_up.data[unit, BD.IS_BYTE] = S
    ffn.b_up.data[unit] = -S * 1.5
    ffn.W_gate.data[unit, BD.H1 + BP_I] = -1.0
    ffn.b_gate.data[unit] = 1.0
    ffn.W_down.data[BD.STACK0_BYTE0, unit] = 2.0 / S
    unit += 1

    for src_base, blocker_base, out_dim in (
        (BD.L1H1, BD.L1H0, BD.BYTE_INDEX_0),
        (BD.L1H2, BD.L1H1, BD.BYTE_INDEX_1),
        (BD.H0, BD.L1H2, BD.BYTE_INDEX_2),
        (BD.H1, BD.H0, BD.BYTE_INDEX_3),
    ):
        ffn.W_up.data[unit, BD.IS_BYTE] = S
        for i in range(NM):
            ffn.W_up.data[unit, src_base + i] = S
            ffn.W_gate.data[unit, blocker_base + i] = -1.0
        ffn.b_up.data[unit] = -S * 1.5
        ffn.b_gate.data[unit] = 1.0
        ffn.W_down.data[out_dim, unit] = 2.0 / S
        unit += 1


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
            attn.alibi_slopes[3] = 0.0  # global SE detection
            attn.alibi_slopes[5] = IN_STEP_FRESH_ALIBI_S  # B7-1: decay
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_threshold_attention_heads(
            attn,
            [0.5, 1.5, 2.5],
            [proxy.L1H0, proxy.L1H1, proxy.L1H2],
            ALIBI_S,
            HD,
            heads=[0, 1, 2],
            bd=proxy,
        )
        # Head 3: STEP_END existence detection (global)
        Primitives.generate_attention_head(
            attn,
            DeclarativeAttentionHeadSpec(
                head_idx=3,
                q=(AP(0, proxy.CONST, 10.0),),
                k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
                v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
                o=(AO(proxy.HAS_SE, 1, 1.0),),
            ),
            HD,
        )
        # Head 4: threshold 6.5 for STACK0 byte 0 identification
        Primitives.generate_threshold_attention_heads(
            attn, [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4], bd=proxy,
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
                head_idx=5,
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
        bake_fn=bake,
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
    proxy = _as_setdim_proxy(dim_positions)
    ALIBI_S = 10.0
    ir = CompilerIR()
    specs = list(Primitives.threshold_attention_head_specs(
        [0.5, 1.5, 2.5],
        [proxy.L1H0, proxy.L1H1, proxy.L1H2],
        ALIBI_S,
        HD,
        heads=[0, 1, 2],
        bd=proxy,
    ))
    specs.append(DeclarativeAttentionHeadSpec(
        head_idx=3,
        q=(AP(0, proxy.CONST, 10.0),),
        k=(AP(0, proxy.MARK_SE_ONLY, 10.0),),
        v=(AP(1, proxy.MARK_SE_ONLY, 1.0),),
        o=(AO(proxy.HAS_SE, 1, 1.0),),
    ))
    specs.extend(Primitives.threshold_attention_head_specs(
        [6.5], [proxy.L1H4], ALIBI_S, HD, heads=[4], bd=proxy,
    ))
    # Head 5 (B7-1): IN_STEP_FRESH recency-to-SE/CS decay.
    specs.append(DeclarativeAttentionHeadSpec(
        head_idx=5,
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
