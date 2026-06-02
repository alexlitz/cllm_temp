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


# === L0 FFN unit layout (auto-fit offsets, Phase 7.B.1) =============
#
# The ``phase_a_ffn`` op owns the entire L0 FFN. The actual weight writes
# happen inside ``vm_step._set_phase_a_ffn`` (and the equivalent
# ``_bake_phase_a_ffn`` declarative path used by the migrated bake), which
# iterates the 7-entry ``transitions`` list and writes one hidden unit per
# transition starting at ``start_unit=0`` via
# ``Primitives.lower_ffn_rules`` (W_up[i, up_dim] / W_down[out_dim, i]).
# The bake's output unit positions are determined by the
# ``lower_ffn_rules`` cursor (always 0..6 in declaration order), NOT by
# the allocator -- so dropping the allocator pins is purely bookkeeping
# and produces byte-identical FFN weights.
#
# Phase 7.B.1: ``pin`` is dropped from every entry. The allocator is
# constructed in ``"dynamic_first_fit"`` mode and walks the layout in
# declaration order; first-fit on a 4096-wide pool lands sub-stages
# 0..6 (one unit each) back at indices 0..6 -- byte-identical to the
# legacy pins -- but the author no longer supplies the offsets. A
# future L0 op family can claim a free range past unit 7 via
# ``allocator.alloc(name, n)`` without a pin.
#
# The order below mirrors the transition order in ``_set_phase_a_ffn``
# (SE->PC, PC->AX, AX->SP, SP->BP, BP->STACK0, STACK0->MEM, MEM->SE).
# Changing the helper's transition list requires updating this table in
# lock-step.
_PHASE_A_FFN_UNIT_LAYOUT = (
    # (sub-stage name, n_units) -- ``pin=None`` everywhere; offsets are
    # picked by the FFNUnitAllocator in ``dynamic_first_fit`` mode.
    ("phase_a_ffn.se_to_pc",      1),  # SE -> NEXT_PC (constant write)
    ("phase_a_ffn.pc_to_ax",      1),  # PC -> NEXT_AX (gated by H0+PC)
    ("phase_a_ffn.ax_to_sp",      1),  # AX -> NEXT_SP (gated by H0+AX)
    ("phase_a_ffn.sp_to_bp",      1),  # SP -> NEXT_BP (gated by H0+SP)
    ("phase_a_ffn.bp_to_stack0",  1),  # BP -> NEXT_STACK0 (gated by H0+BP)
    ("phase_a_ffn.stack0_to_mem", 1),  # STACK0 -> NEXT_MEM (gated by H3+BP)
    ("phase_a_ffn.mem_to_se",     1),  # MEM -> NEXT_SE (gated by H2+MEM)
)


def _allocate_phase_a_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L0 phase-A units.

    Phase 7.B.1 auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode with no ``pin=`` hints. Each entry is
    allocated in declaration order; first-fit picks the lowest free
    unit each call, so the 7 single-unit transitions land at indices
    0..6 -- byte-identical to the legacy pinned offsets. The actual
    FFN weight writes are driven by ``Primitives.lower_ffn_rules`` at
    ``start_unit=0``, so the allocator's pick is bookkeeping only.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L0 op family claims a free range starting at unit 7).
    """
    allocator = FFNUnitAllocator(strategy="dynamic_first_fit")
    for name, n_units in _PHASE_A_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
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

        # Per-bake FFN-unit allocator (Phase 7.B.1 auto-fit). The allocator
        # is built in ``dynamic_first_fit`` mode without pin hints; the
        # 7 single-unit transitions land at indices 0..6 by first-fit, the
        # same offsets the legacy pins produced. The total ``n_units``
        # across the allocator's ranges MUST equal the helper's monotonic
        # unit count, or the byte-identity guard fires.
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
        expected_total = sum(n for _, n in _PHASE_A_FFN_UNIT_LAYOUT)
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
        # Phase 7.A.1: H0..H4 are written by ``layer0_threshold_attn`` at the
        # SAME layer (L0 attn substage feeds the L0 block-FFN substage within
        # the same transformer block). Express this as a co-placement
        # constraint via ``requires["same_layer_as"]`` rather than a depth-
        # bumping ``reads`` edge, so analyze_scheduler.py and
        # LayerCompiler._assign_layers agree on the layer assignment (both
        # treat ``same_layer_as`` as equal-depth). The bake itself still
        # consumes H0..H4 from the residual; ``reads`` is metadata only.
        reads=set(),
        writes={"NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP",
                "NEXT_STACK0", "NEXT_MEM", "NEXT_SE"},
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=0`` literal; bind to the L0 attn
        # dep anchor so the block op resolves to whichever layer the
        # compiler places the anchor at.
        target_op_name="_layer0_threshold_attn_dep_anchor",
        declarative_bake_fn=bake,
        compiler_ir=_phase_a_ffn_ir(),
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        requires={"same_layer_as": "layer0_threshold_attn"},
        # ``_set_phase_a_ffn`` writes one FFN hidden unit per transition in
        # the 7-entry ``transitions`` list (SE→PC, PC→AX, AX→SP, SP→BP,
        # BP→STACK0, STACK0→MEM, MEM→SE). Units 0..6.
        ffn_units_used=7,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
        compaction_safe=True,
    )


# === L0 attention-head layout (auto-fit offsets, Phase 7.B.2 attn) ===
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
# pre-loaded with the full L0 head layout, stashes it on
# ``attn._l0_head_allocator`` for downstream inspection, and resolves
# each head's index by name.
#
# Phase 7.B.2 attn: ``pin`` is dropped from every entry. The allocator
# is constructed in ``"dynamic_first_fit"`` mode and walks the layout
# in declaration order; first-fit on an 8-head pool lands the eight
# threshold heads at indices 0..7 -- byte-identical to the legacy
# pins -- but the author no longer supplies the offsets. A future L0
# attention op can claim a free head via
# ``allocator.alloc(name, layer_idx=0)`` without a pin.
#
# Output-base / alibi-slope mapping is now spec-carried: the
# ``alibi_slopes=`` list threaded through
# ``Primitives.generate_threshold_attention_heads`` is keyed by
# threshold POSITION (parallel to ``_L0_THRESHOLDS``), not head_idx,
# so first-fit drift in head_idx cannot scramble it. Likewise each
# spec captures its own ``out_base`` (a pre-resolved ``AO(out_base+m, ...)``
# write) so permuting head_idx never reroutes a threshold to a
# different ``H<n>`` output dim.
_L0_HEAD_LAYOUT = (
    # (op-name key,)  -- no pinned head_idx; allocator first-fits.
    # Declaration order is preserved here so the threshold-index
    # parallelism with ``_L0_THRESHOLDS`` / ``_L0_OUT_BASE_NAMES``
    # stays load-bearing for the spec output mapping.
    ("layer0_threshold_attn.h0",),    # threshold 3.5  -> H0
    ("layer0_threshold_attn.h1",),    # threshold 4.5  -> H1
    ("layer0_threshold_attn.h2",),    # threshold 7.5  -> H2
    ("layer0_threshold_attn.h3",),    # threshold 8.5  -> H3
    ("layer0_threshold_attn.h4",),    # threshold 9.5  -> H4
    ("layer0_threshold_attn.h5",),    # threshold 14.5 -> H5
    ("layer0_threshold_attn.h6",),    # threshold 19.5 -> H6
    ("layer0_threshold_attn.h7",),    # threshold 24.5 -> H7
)


def _allocate_layer0_threshold_attn_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L0 heads.

    Phase 7.B.2 attn auto-fit: the allocator is built in
    ``dynamic_first_fit`` mode without ``pin=`` hints. Each entry is
    allocated in declaration order; first-fit picks the lowest free
    head each call, so the 8 threshold-head entries land at indices
    0..7 -- byte-identical to the legacy pinned offsets.

    Returns the allocator so callers can inspect or extend it (e.g. a
    future L0 attention op claiming a free gap past head 7 when
    ``layer_max_heads`` widens).
    """
    allocator = AttentionHeadAllocator(strategy="dynamic_first_fit")
    for (name,) in _L0_HEAD_LAYOUT:
        allocator.alloc(name, layer_idx=0)
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
    declarative threshold-head specs (H0..H7). The head indices are
    resolved by walking :func:`_allocate_layer0_threshold_attn_heads`
    in declaration order so the IR and the bake share a single source
    of truth -- which lets ``compare_symbolic_to_lowered_attn`` audit
    byte-identity end-to-end against ``PureAttention``.
    """

    proxy = _as_setdim_proxy(dim_positions)
    allocator = _allocate_layer0_threshold_attn_heads()
    heads = [rec.head_idx for rec in allocator.heads()]
    specs = _layer0_threshold_head_specs(proxy, HD, heads)
    ir = CompilerIR()
    ir.layer(0).attention.extend(specs)
    return ir


def make_layer0_threshold_attn_dep_anchor_op() -> Operation:
    """No-op companion for ``layer0_threshold_attn``: declares mirrored
    reads/writes so the LayerCompiler's dep graph reserves an L0 slot
    for it. Mirrors ``_layer3_ffn_dep_anchor`` / ``_layer6_ffn_dep_anchor``:
    the actual weight bake happens in ``layer0_threshold_attn`` (kind=
    "block"); this op's bake is a no-op.

    Phase 8.G.6: lets L0 block ops (``phase_a_ffn``, ``layer0_threshold_attn``
    itself) declare ``target_op_name="_layer0_threshold_attn_dep_anchor"``
    and bind to whichever layer the compiler places the anchor at,
    instead of carrying a literal ``layer_idx=0``.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in ``layer0_threshold_attn`` block op below.
        return

    return Operation(
        name="_layer0_threshold_attn_dep_anchor",
        phase=0,
        # Mirror ``layer0_threshold_attn``'s reads/writes so the dep
        # graph places this anchor at L0. With no upstream writers for
        # IS_MARK / CONST (both set by token embedding pre-L0), the
        # earliest landable layer is 0.
        reads={"IS_MARK", "CONST"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        smoke_tests=set(),
        spec_section=None,
    )


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
        # Phase 8.G.6: drop ``layer_idx=0`` literal; bind to the L0 attn
        # dep anchor so the block op resolves to whichever layer the
        # compiler places the anchor at.
        target_op_name="_layer0_threshold_attn_dep_anchor",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer0_threshold_attn_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )
