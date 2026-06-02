"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy

# L0 threshold-attention configuration shared between the bake_fn and the
# declarative ``compiler_ir_factory``. The two lists are positional siblings:
# index ``i`` in ``_L0_THRESHOLDS`` pairs with ``_L0_OUT_BASE_NAMES[i]`` and
# the head at ``_L0_HEAD_LAYOUT[i]``.
_L0_THRESHOLDS = (3.5, 4.5, 7.5, 8.5, 9.5, 14.5, 19.5, 24.5)
_L0_OUT_BASE_NAMES = ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7")
_L0_ALIBI_S = 10.0


# === L0 FFN unit layout (pinned offsets) ============================
#
# The ``phase_a_ffn`` op owns the entire L0 FFN. The actual weight writes
# happen inside ``vm_step._set_phase_a_ffn`` (and the equivalent
# ``_bake_phase_a_ffn`` declarative path used by the migrated bake), which
# iterates the 7-entry ``transitions`` list and writes one hidden unit per
# transition at indices 0..6 (W_up[i, up_dim] / W_down[out_dim, i]).
# Migration to :class:`FFNUnitAllocator` keeps the helper byte-identical --
# we just declare each transition's unit at its existing pinned offset so
# the layout is auditable rather than implicit. A future L0 op family would
# call ``allocator.alloc(name, n)`` without a pin and get a free gap above 7.
#
# The offsets below mirror the transition order in ``_set_phase_a_ffn``
# (SE->PC, PC->AX, AX->SP, SP->BP, BP->STACK0, STACK0->MEM, MEM->SE).
# Changing the helper's transition list requires updating this table in
# lock-step.
_PHASE_A_FFN_UNIT_LAYOUT = (
    # (sub-stage name, pinned start, n_units)
    ("phase_a_ffn.se_to_pc",      0, 1),  # SE -> NEXT_PC (constant write)
    ("phase_a_ffn.pc_to_ax",      1, 1),  # PC -> NEXT_AX (gated by H0+PC)
    ("phase_a_ffn.ax_to_sp",      2, 1),  # AX -> NEXT_SP (gated by H0+AX)
    ("phase_a_ffn.sp_to_bp",      3, 1),  # SP -> NEXT_BP (gated by H0+SP)
    ("phase_a_ffn.bp_to_stack0",  4, 1),  # BP -> NEXT_STACK0 (gated by H0+BP)
    ("phase_a_ffn.stack0_to_mem", 5, 1),  # STACK0 -> NEXT_MEM (gated by H3+BP)
    ("phase_a_ffn.mem_to_se",     6, 1),  # MEM -> NEXT_SE (gated by H2+MEM)
)


def _allocate_phase_a_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L0 phase-A units.

    Every transition is pinned at its existing offset (units 0..6) so the
    underlying ``_bake_phase_a_ffn`` helper -- which writes via its own
    sequential start_unit=0 cursor through ``Primitives.lower_ffn_rules``
    -- lands on exactly the same hidden-unit indices it always has. This
    call is byte-identical bookkeeping: the allocator declares ranges by
    name, the helper writes the weights. A future refactor can split the
    helper into per-transition bake functions that consume
    ``allocator.alloc(...)`` directly.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L0 op family claims a free range starting at unit 7).
    """
    allocator = FFNUnitAllocator()
    for name, start, n_units in _PHASE_A_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units, pin=start)
    return allocator


def _phase_a_ffn_rules(S: float) -> tuple[FFNRule, ...]:
    """CompilerIR rules for the L0 marker-transition detector."""

    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5
    write_scale = 2.0 / S
    transitions = (
        (f"H0+{SE_I}", None, "NEXT_PC"),
        (f"H1+{PC_I}", f"H0+{PC_I}", "NEXT_AX"),
        (f"H1+{AX_I}", f"H0+{AX_I}", "NEXT_SP"),
        (f"H1+{SP_I}", f"H0+{SP_I}", "NEXT_BP"),
        (f"H1+{BP_I}", f"H0+{BP_I}", "NEXT_STACK0"),
        (f"H4+{BP_I}", f"H3+{BP_I}", "NEXT_MEM"),
        (f"H3+{MEM_I}", f"H2+{MEM_I}", "NEXT_SE"),
    )
    rules = []
    for idx, (up_dim, gate_dim, out_dim) in enumerate(transitions):
        if gate_dim is None:
            rules.append(FFNRule.constant_write(
                name=f"phase_a_{idx}_{out_dim.lower()}",
                conditions=((up_dim, 1.0),),
                threshold=0.3,
                writes=((out_dim, write_scale),),
            ))
        else:
            rules.append(FFNRule.gated_write(
                name=f"phase_a_{idx}_{out_dim.lower()}",
                conditions=((up_dim, 1.0),),
                threshold=0.3,
                gate_terms=((gate_dim, -1.0),),
                gate_bias=1.0,
                writes=((out_dim, write_scale),),
            ))
    return tuple(rules)


def _phase_a_ffn_ir(S: float = 100.0) -> CompilerIR:
    ir = CompilerIR()
    ir.layer(0).ffn.rules.extend(_phase_a_ffn_rules(S))
    return ir


def _bake_phase_a_ffn(ffn, S, BD) -> int:
    rules = _phase_a_ffn_rules(S)
    dim_positions = Primitives.dim_positions_from_bd(
        BD,
        Primitives.ffn_rule_dim_names(rules),
    )
    return Primitives.lower_ffn_rules(ffn, rules, dim_positions, S=S)


def make_phase_a_ffn_op() -> Operation:
    """Step-structure FFN: detect marker transitions and emit NEXT_* flags.

    Originally: `_set_phase_a_ffn` at vm_step.py:2872. Lives at L0 in the
    hand-set layout.

    Reads H0/H1/H2/H3/H4 threshold-head outputs (per marker type).
    Writes NEXT_PC, NEXT_AX, NEXT_SP, NEXT_BP, NEXT_STACK0, NEXT_MEM, NEXT_SE.

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].ffn) the legacy path used. This sidesteps the
    LayerCompiler's dep-based assignment, which would otherwise place this FFN
    at L1 (advancing past L0 because it reads H0-H4 written by L0 attn).
    """
    def bake(block, dim_positions, S):
        proxy = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. Each L0 phase-A transition is pinned
        # to its existing offset so the call below lands byte-identically.
        # The total ``n_units`` across the allocator's ranges MUST equal the
        # helper's monotonic unit count, or the byte-identity guard fires.
        allocator = _allocate_phase_a_ffn_units()
        # Make the allocator available for inspection / extension by
        # downstream tools (e.g. a future L0 op family claiming a free
        # gap). Mirrors the ``_l9_unit_allocator`` convention used by L9.
        block.ffn._l0_unit_allocator = allocator

        n0 = _bake_phase_a_ffn(block.ffn, S, proxy)
        # Byte-identity guard: the helper's returned cursor MUST equal the
        # sum of all declared unit ranges. If the table drifts from the
        # helper's writes, this assertion fires before any weight surgery
        # propagates downstream.
        expected_total = sum(n for _, _, n in _PHASE_A_FFN_UNIT_LAYOUT)
        assert n0 == expected_total, (
            f"L0 phase_a_ffn unit cursor drift: helper wrote {n0} units, "
            f"allocator declared {expected_total}"
        )

    # Dim-ownership claims: ``_set_phase_a_ffn`` writes units 0..6 (one per
    # marker transition) into the L0 FFN. The W_up rows read H0..H4 marker
    # threshold outputs; W_down rows write NEXT_* dims. The 7 units are
    # stable across builds (always start at unit 0, one per transition).
    _claims = set()
    # Each transition row writes (unit, NEXT_* slot). Use NEXT_* dim names
    # as the column tag since W_down[NEXT_*, unit] = 2.0/S.
    # See _set_phase_a_ffn in vm_step.py for the transition list.
    _next_dim_names = [
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
        "NEXT_STACK0", "NEXT_MEM", "NEXT_SE",
    ]
    for u, next_dim in enumerate(_next_dim_names):
        _claims.add((0, "ffn_W_down", str(u), f"{next_dim}+0"))

    # The threshold heads write 7 dims each (one per marker type), so we
    # express reads as the head-base names; the FFN reads any element in the
    # H0..H4 ranges, which are size-7 dims.
    return Operation(
        name="phase_a_ffn",
        phase=0,
        reads={"H0", "H1", "H2", "H3", "H4"},
        writes={"NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir=_phase_a_ffn_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        # ``_set_phase_a_ffn`` writes one FFN hidden unit per transition in
        # the 7-entry ``transitions`` list (SE→PC, PC→AX, AX→SP, SP→BP,
        # BP→STACK0, STACK0→MEM, MEM→SE). Units 0..6.
        ffn_units_used=7,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        compaction_safe=True,
    )


# === L0 attention-head layout (pinned indices) ======================
#
# ``layer0_threshold_attn`` owns the entire L0 attention block: 8
# threshold heads (H0..H7) detecting whether the nearest marker is
# within a per-head distance cutoff (3.5/4.5/7.5/8.5/9.5/14.5/19.5/24.5
# tokens). Pre-migration the bake delegated to
# ``Primitives.generate_threshold_attention_heads`` with ``heads=None``,
# which defaulted to ``list(range(8))`` -- an implicit head-index
# claim that made adding a new L0 head fragile (the author had to
# remember which slots were already taken). With the allocator the
# handoff is structural: the bake instantiates its OWN allocator
# pre-loaded with the full L0 head layout (pinned to existing slots),
# stashes it on ``attn._l0_head_allocator`` for downstream inspection,
# and resolves each head's index by name. A future L0 attention op
# can claim a free head via ``allocator.alloc(name, layer_idx=0)`` --
# with no ``pin=`` -- without touching this table.
#
# Each entry's ``head_idx`` mirrors the implicit ``heads=range(8)``
# walk in ``Primitives.threshold_attention_head_specs``; changing the
# head/threshold pairing requires updating this table in lock-step.
_L0_HEAD_LAYOUT = (
    # (op-name key,                      pinned head_idx)
    ("layer0_threshold_attn.h0",         0),  # threshold 3.5  -> H0
    ("layer0_threshold_attn.h1",         1),  # threshold 4.5  -> H1
    ("layer0_threshold_attn.h2",         2),  # threshold 7.5  -> H2
    ("layer0_threshold_attn.h3",         3),  # threshold 8.5  -> H3
    ("layer0_threshold_attn.h4",         4),  # threshold 9.5  -> H4
    ("layer0_threshold_attn.h5",         5),  # threshold 14.5 -> H5
    ("layer0_threshold_attn.h6",         6),  # threshold 19.5 -> H6
    ("layer0_threshold_attn.h7",         7),  # threshold 24.5 -> H7
)


def _allocate_layer0_threshold_attn_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L0 heads.

    Every entry in :data:`_L0_HEAD_LAYOUT` is pinned at its existing
    ``head_idx`` so the underlying
    ``Primitives.generate_threshold_attention_heads`` call -- which
    writes ``W_q``/``W_k``/``W_v``/``W_o`` at ``head_idx * HD + slot``
    -- lands byte-identically. Replacing the implicit ``heads=None``
    default with an explicit allocator-resolved list keeps every head
    index auditable rather than buried in a ``range(8)`` fallback.
    A future L0 attention op can claim a free head past index 7 (when
    ``layer_max_heads`` widens beyond 8) via
    ``allocator.alloc(name, layer_idx=0)`` without a pin.
    """
    allocator = AttentionHeadAllocator()
    for name, head_idx in _L0_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=0, pin=head_idx)
    return allocator


def _layer0_threshold_head_specs(proxy, HD: int, heads):
    """Return the 8 declarative L0 threshold-attention head specs.

    ``heads`` is an explicit list of ``head_idx`` values (one per entry in
    :data:`_L0_THRESHOLDS`) so the spec is structurally pinned to the
    allocator layout rather than the implicit ``range(8)`` fallback. The
    resulting Q/K/V/O writes lower byte-identically to the legacy
    ``Primitives.generate_threshold_attention_heads`` walk because both
    paths funnel through ``Primitives.threshold_attention_head_specs``.
    """

    out_bases = [getattr(proxy, name) for name in _L0_OUT_BASE_NAMES]
    return Primitives.threshold_attention_head_specs(
        list(_L0_THRESHOLDS),
        out_bases,
        _L0_ALIBI_S,
        HD,
        heads=heads,
        bd=proxy,
    )


def _layer0_threshold_attn_ir(dim_positions, HD: int) -> CompilerIR:
    """``compiler_ir_factory`` for ``layer0_threshold_attn``.

    Builds a :class:`CompilerIR` whose ``layer(0).attention`` holds the 8
    declarative threshold-head specs (H0..H7). The pinned head indices
    come from :data:`_L0_HEAD_LAYOUT` so the IR stays in lockstep with the
    bake's allocator pinning, and ``compare_symbolic_to_lowered_attn``
    can audit byte-identity end-to-end against ``PureAttention``.
    """

    proxy = _as_setdim_proxy(dim_positions)
    heads = [head_idx for _name, head_idx in _L0_HEAD_LAYOUT]
    specs = _layer0_threshold_head_specs(proxy, HD, heads)
    ir = CompilerIR()
    ir.layer(0).attention.extend(specs)
    return ir


def make_layer0_threshold_attn_op() -> Operation:
    """L0 attention: 8 threshold heads detecting marker distance.

    Dispatched as a block op pinned to layer_idx=0 so the bake hits the same
    transformer block (block[0].attn) the legacy path used. Using kind="block"
    keeps the L0 op aligned with the hand-set block index regardless of
    LayerCompiler dep-based assignment.

    Phase 6 Wave 2A: migrated to ``DeclarativeAttentionHeadSpec`` form. The
    8 threshold heads are now expressed as data via
    :func:`_layer0_threshold_head_specs` and exposed through
    ``compiler_ir_factory=_layer0_threshold_attn_ir``. The bake_fn lowers
    the same specs (so byte-identity is mandated by spec equality, not by
    a separate call path). Head indices remain pinned via the allocator;
    the ALiBi slope override and allocator stash stay in the bake as
    residual-side bookkeeping until a future wave folds them into
    ``AttentionHeadIR.metadata``.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        proxy = _as_setdim_proxy(dim_positions)
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(_L0_ALIBI_S)
        HD = attn.W_q.shape[0] // attn.num_heads
        # Per-bake attention-head allocator with the full L0 head layout
        # pinned. Resolving each threshold head by name reproduces the
        # legacy implicit ``heads=range(8)`` walk without baking integer
        # literals into the call site. Stashed on the attention module so
        # downstream tools (e.g. a future L0 op claiming a free gap, or
        # the verifier auditing head ownership) can inspect or extend
        # the layout. Mirrors the ``_l1_head_allocator`` /
        # ``_l2_head_allocator`` conventions used by sibling layers.
        head_allocator = _allocate_layer0_threshold_attn_heads()
        attn._l0_head_allocator = head_allocator
        # Resolve the full 8-head ordered list from the allocator so the
        # call site is structural rather than positional.
        threshold_heads = [
            head_allocator.heads()[i].head_idx for i in range(8)
        ]
        # Lower via the same declarative spec used by ``compiler_ir_factory``.
        # This routes the bake through ``DeclarativeAttentionHeadSpec`` ->
        # ``Primitives.generate_attention_heads`` so spec equality implies
        # byte-equal Q/K/V/W_o (see compare_symbolic_to_lowered_attn).
        specs = _layer0_threshold_head_specs(proxy, HD, threshold_heads)
        Primitives.generate_attention_heads(attn, specs, HD)
        # H1's 4.5-token cutoff is semantically load-bearing: L1 marks
        # STACK0 byte 0 with L1H4[BP] AND NOT H1[BP]. Scaling this head makes
        # H1 fire at STACK0 byte 0 and blocks ADD/POP operand gather.
    # Dim-ownership claims: ``_set_threshold_attn`` writes per head h:
    #   W_q[h*HD, CONST]    : the head's bias (slot 0)
    #   W_k[h*HD, IS_MARK]  : K-side mark detector (slot 0)
    #   W_v[h*HD + 1 + m, MARKS[m]]  : V slot 1..7 (one per marker)
    #   W_o[out_base + m, h*HD + 1 + m] : O column 1..7
    # For 8 heads (h=0..7) writing to H0..H7 output bases.
    # MARKS = [MARK_PC, MARK_AX, MARK_SP, MARK_BP, MARK_MEM, MARK_SE, MARK_CS]
    _claims = set()
    _MARKS = ["MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
              "MARK_MEM", "MARK_SE", "MARK_CS"]
    _OUT_BASES = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    for h in range(8):
        # V slots 1..7: each reads MARKS[m] column.
        for m, mark in enumerate(_MARKS):
            _claims.add((0, "attn_W_v", f"{h}_{1 + m}", f"{mark}+0"))
        # Q slot 0: CONST column.
        _claims.add((0, "attn_W_q", f"{h}_0", "CONST+0"))
        # K slot 0: IS_MARK column.
        _claims.add((0, "attn_W_k", f"{h}_0", "IS_MARK+0"))
        # O writes 7 columns of W_o[out_base+m, h*HD + 1 + m]; in the
        # attn_W_o scope the identifier is column-of-W_o = "<head>_<slot>".
        for m, out_base in enumerate([_OUT_BASES[h]] * 7):
            _claims.add((0, "attn_W_o", f"{h}_{1 + m}", f"{out_base}+{m}"))

    return Operation(
        name="layer0_threshold_attn",
        phase=0,
        reads={"IS_MARK", "CONST"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="block",
        layer_idx=0,
        bake_fn=bake,
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer0_threshold_attn_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
