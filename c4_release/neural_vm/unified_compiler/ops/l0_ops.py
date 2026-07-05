"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

import os as _os_l0

from ...attention_head_allocator import AttentionHeadAllocator
from ...dim_registry import dim_ref
from ...ffn_unit_allocator import FFNUnitAllocator
from ..building_blocks_dsl import multi_way_and_rule, step_function_rule
from ..ir import CompilerIR, FFNRule
from ..layer_compiler import Operation
from ..primitives import Primitives
from .shared import _as_setdim_proxy


def _no_stack0_emit() -> bool:
    """PROTOTYPE flag: drop the STACK0 register block from the emitted step.

    When ``C4_NO_STACK0_EMIT=1`` the L0 marker-transition chain skips
    NEXT_STACK0 (emits NEXT_MEM straight after BP's value bytes), so the model
    never emits ``Token.STACK0`` -> a 30-token step (matching
    ``Token.STEP_TOKENS == 30`` and the DraftVM oracle). Default-off =
    byte-identical 35-token build.
    """
    return _os_l0.environ.get("C4_NO_STACK0_EMIT", "1") != "0"

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
    """CompilerIR rules for the L0 marker-transition detector.

    Phase 8.D: the per-marker ``H<k>+{i}`` reads stay as ``+N`` -- the
    ``<i>`` offset indexes a fixed-width 7-slot threshold-head bank
    position, a structural slot index into the H<k> head's output
    (same convention as the L1 marker-bank reads). The transition's
    ``NEXT_*`` output names are slot-level singleton flags with no
    ``(category, role)`` binding in the dim registry, so the write
    targets also stay bare. No role-meaningful (category, role) refs
    surface in this rule generator.
    """

    # structural offsets: PC_I/AX_I/SP_I/BP_I/MEM_I/SE_I index a
    # fixed-width 7-slot threshold-head bank position, not a
    # role-meaningful byte index.
    PC_I, AX_I, SP_I, BP_I, MEM_I, SE_I = 0, 1, 2, 3, 4, 5
    write_scale = 2.0 / S
    if _no_stack0_emit():
        # PROTOTYPE (C4_NO_STACK0_EMIT=1): drop the STACK0 register block from
        # the emitted step. The marker chain skips STACK0: after BP's 4 value
        # bytes (BP byte-3 row = d=4 from BP, the H1+BP AND NOT H0+BP gate that
        # used to fire NEXT_STACK0) we emit NEXT_MEM directly, so the MEM marker
        # follows BP byte-3. The STACK0->MEM transition (d=9, H4+BP) is dropped.
        # The downstream MEM->SE transition is distance-from-MEM-marker so it is
        # unaffected by the removed STACK0 tokens. NEXT_STACK0 is never written,
        # so the LM head never emits Token.STACK0. (35-token: 7 transitions; the
        # 30-token build has 6 -- the byte-identity unit-count guard in the bake
        # is relaxed under the flag.)
        transitions = (
            (f"H0+{SE_I}", None, "NEXT_PC"),
            (f"H1+{PC_I}", f"H0+{PC_I}", "NEXT_AX"),
            (f"H1+{AX_I}", f"H0+{AX_I}", "NEXT_SP"),
            (f"H1+{SP_I}", f"H0+{SP_I}", "NEXT_BP"),
            (f"H1+{BP_I}", f"H0+{BP_I}", "NEXT_MEM"),
            (f"H3+{MEM_I}", f"H2+{MEM_I}", "NEXT_SE"),
        )
    else:
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
            rules.append(step_function_rule(
                name=f"phase_a_{idx}_{out_dim.lower()}",
                input_dim=up_dim,
                threshold=0.3,
                write_dim=out_dim,
                write_value=write_scale * S,
                S=S,
            ))
        else:
            rules.append(multi_way_and_rule(
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

        # Phase 8.C inline cut: lower the ``_phase_a_ffn_rules`` IR
        # directly here so census v2 classifies this op as
        # ``declarative`` rather than ``declarative_via_helper`` (which
        # routed through the ``_bake_phase_a_ffn`` trampoline).
        # Byte-identical to the prior helper call.
        rules = _phase_a_ffn_rules(S)
        rule_dim_positions = Primitives.dim_positions_from_bd(
            proxy,
            Primitives.ffn_rule_dim_names(rules),
        )
        n0 = Primitives.lower_ffn_rules(
            block.ffn, rules, rule_dim_positions, S=S,
        )
        # Byte-identity guard: the helper's returned cursor MUST equal the
        # sum of all declared unit ranges. If the table drifts from the
        # helper's writes, this assertion fires before any weight surgery
        # propagates downstream. The PROTOTYPE 30-token build drops the
        # BP->STACK0 + STACK0->MEM pair for a single BP->MEM transition (6
        # units, not 7), so relax the count under the flag.
        expected_total = sum(n for _, n in _PHASE_A_FFN_UNIT_LAYOUT)
        if _no_stack0_emit():
            expected_total -= 1  # one fewer transition (no STACK0 marker)
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
        # Wave 4 (docs/PRODUCES_CONSUMES_MIGRATION.md): L0 token-feed
        # pipeline op. Writes NEXT_* step-boundary flags consumed at
        # the next step's L0 phase rotation (cross-step). No same-step
        # per-register slot consumer reads NEXT_*; empty surface
        # affirms the cross-step boundary semantics.
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
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the MARK_* dims each threshold head reads via ``bd.MARKS`` (V
        # slot 1+m reads MARKS[m] for m=0..6 = MARK_PC, MARK_AX, MARK_SP,
        # MARK_BP, MARK_MEM, MARK_SE, MARK_CS). Previously only IS_MARK
        # + CONST were declared (Q/K gates).
        reads={"IS_MARK", "CONST",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "MARK_MEM", "MARK_SE", "MARK_CS"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="attn",
        migrated=True,
        declarative_authority="topology_anchor",
        # Phase 11.A IR exposure: empty IR exposes the topology-anchor's
        # noop weight semantics to the dim-multiplexer (Phase 10.E/F).
        compiler_ir=CompilerIR(),
        # Wave 4 (docs/PRODUCES_CONSUMES_MIGRATION.md): topology anchor
        # — empty IR, no bake, no in-step produce/consume_fresh surface.
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
        # Phase 11.A r3: dropped phase=0 — co-placement at
        # ``_layer0_threshold_attn_dep_anchor`` + explicit
        # ``requires['after']`` on the anchor already pin fire order.
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the MARK_* dims each threshold head reads via ``bd.MARKS`` (V
        # slot 1+m reads MARKS[m] for m=0..6 = MARK_PC, MARK_AX, MARK_SP,
        # MARK_BP, MARK_MEM, MARK_SE, MARK_CS). Previously only IS_MARK
        # + CONST were declared (Q/K gates).
        reads={"IS_MARK", "CONST",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
               "MARK_MEM", "MARK_SE", "MARK_CS"},
        writes={"H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"},
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=0`` literal; bind to the L0 attn
        # dep anchor so the block op resolves to whichever layer the
        # compiler places the anchor at.
        target_op_name="_layer0_threshold_attn_dep_anchor",
        requires={"after": "_layer0_threshold_attn_dep_anchor"},
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer0_threshold_attn_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        # Wave 4 (docs/PRODUCES_CONSUMES_MIGRATION.md): L0 token-feed
        # attention op. Writes H0..H7 threshold outputs via 8 heads;
        # derive yields empty (attention-only IR — no FFNRule.writes).
        # L0 attn is the first non-embed op; reads IS_MARK/CONST are
        # embed-time/structural. No same-step consumes_fresh surface.
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


# ===========================================================================
# No-STACK0 (30-token) PC value-byte OUTPUT clear (C4_NO_STACK0_EMIT) -- Inc 0
# ===========================================================================
#
# ROOT (spec_k=0 token dump, tools/probe_nostack0_{dump,pcband,blockattr}.py;
# add_1 id1 got_pc=0x01010111 vs 26): with the 30-token frame the REGISTER
# FRAME is byte-clean (markers at {0,5,10,15,20} every step -- NO drift), but
# the PC VALUE bytes are polluted by a self-reinforcing ``0x01``/nibble-1 leak.
# A positional ``0x01`` writer (block 33 == layer15_nibble_copy + its memory
# lookup attention, plus the block-41 L25 tail corrector) sprays
# ``OUTPUT_LO+1 = +large / OUTPUT_LO+0 = -large`` onto the SHIFTED 30-token
# MEM/BP region (the dropped STACK0 block left the MEM block 5 tokens early,
# so the distance-from-BP-marker gates and the PSH_AT_SP AND-headroom misfire).
# The ``0x01`` lands in OUTPUT at the BP/MEM-addr rows in step 0, the
# nibble-copy then re-reads the emitted ``0x01`` tokens' embeddings each step
# (self-reinforcement), and by step 2 the leak has propagated onto the PC
# rows -> every PC byte emits ``0x01`` (high bytes) / ``0x11`` (byte 0 low
# nibble clobbered from ``a`` to ``1``). The verdict
# (batched_pure_neural._decode_step_register) then reads got_pc=0x01010111.
# This is the dominant 30-token blocker (288/375 PC-wrong / PC-byte-replication
# = mechanism (a): positional gates corrupt the shifted frame; NOT (c) frame
# drift -- the markers never move).
#
# FIX: a standalone PureFFN post_op (after tail_bit32_result_correction, i.e.
# the LAST OUTPUT writer before the LM head on every row) that, at the PC
# value-byte rows ONLY (gated on the L0 ``H1+0`` PC-marker proximity head +
# IS_BYTE + the per-byte BYTE_INDEX one-hot), drives OUTPUT_LO/HI nibbles 1..15
# hugely negative so the PC HIGH bytes (1/2/3) default to ``0x00`` and the
# byte-0 row's leaked nibble-1 is sunk so the genuine low nibble wins. The PC
# high bytes are ``0x00`` for the entire 1096 corpus EXCEPT the rare linear-PC
# carry past 0x100, which the L3 ``pc_byte1`` rule emits via TEMP+16 -- so the
# byte-1-predicting row's clear is gated NOT(TEMP+16) to preserve that carry.
# Because this is the last writer at the PC rows it overrides ALL upstream
# leak sources at once (block 33 + block 41 + the attention copy), and because
# the emitted PC tokens become clean each step the cross-step self-reinforcement
# is broken at the root. Gated by ``C4_NO_STACK0_EMIT``: flag-OFF bakes NO
# units (byte-identical to golden b9d8861f).
#
# The PC value bytes are unread on the marker row itself (only the next REG_PC
# is emitted from STEP_END), so clobbering OUTPUT at the PC byte rows is
# semantically free for everything BUT the PC byte tokens we are correcting.
#
# CRITICAL magnitude note: the sink must be LARGE enough to bury the ~110-mag
# nibble-1 leak (so OUTPUT_LO+0 wins -> 0x00) but SMALL enough that the byte
# logit stays well ABOVE the register-marker logit at this row (the LM head
# emits a MARKER, not a byte, if every OUTPUT nibble sinks below the marker's
# baseline -- which would re-introduce the +1-token frame drift this campaign
# eliminates). ``-1e16`` (the §11 STEP_END value, where a marker IS due) drives
# the byte logit below the marker and DESYNCS the frame. ``-300`` buries the
# +110 leak with headroom while OUTPUT_LO+0 (~+1) keeps the 0x00 byte logit
# comfortably positive and above the marker.
_NO_STACK0_PC_CLEAR_WW = -300.0  # per-nibble sink: > leak (~110), < marker gap


def _no_stack0_pc_highbyte_clear_rules() -> tuple[FFNRule, ...]:
    """Clear OUTPUT high nibbles at the PC value-byte rows (30-token build).

    Four PC-block rows are corrected (each predicts the NEXT emitted PC token):

      * ``BYTE_INDEX_0`` row -> predicts PC byte 1 (0x00 for all add/sub PCs).
      * ``BYTE_INDEX_1`` row -> predicts PC byte 2 (always 0x00 in corpus).
      * ``BYTE_INDEX_2`` row -> predicts PC byte 3 (always 0x00 in corpus).
      * ``MARK_PC`` row (marker) -> predicts PC byte 0: ONLY the leaked
        nibble-1 is cancelled (the genuine value bits are kept).

    The three byte rows fire only when the L0 ``H1+0`` head (nearest marker is
    PC, within d<=4.5) is ON, so they isolate the PC register block from
    AX/SP/BP (which carry ``H1+1/2/3``). Each sinks OUTPUT_LO/HI nibbles 1..15
    and restores LO+0 / HI+0 so the byte logit collapses to ``0x00``.
    """
    PC_HEAD = "H1+0"  # L0 threshold head: nearest marker is PC, d<=4.5
    BLOCKER_W = 1_000.0
    WW = _NO_STACK0_PC_CLEAR_WW
    # Positive restore for the zero nibble (LO+0 / HI+0) -- the leak drives
    # OUTPUT_LO+0 NEGATIVE (~-90), so simply sinking the other nibbles is not
    # enough; the 0x00 byte logit (LO+0 + HI+0) must be POSITIVE and above the
    # marker. ``+300`` lifts the zero nibble decisively while staying well below
    # any value that would itself look like garbage.
    WPOS = 300.0
    # (byte-index row, extra NOT-conditions to protect a genuine value)
    #
    # PC byte 1 (BYTE_INDEX_0 row) is 0x00 for every linear-PC program in the
    # add/sub acceptance set (PC stays < 0x100: 10/18/26/34). The rare
    # linear-PC carry past 0x100 (corpus tops out at 0x14a) would want byte1 =
    # 0x01; the obvious ``NOT(TEMP+16)`` carry-tag guard is UNRELIABLE under the
    # 30-token frame (TEMP+16 reads ~0.99 even on the no-carry add_1 step-2 PC
    # row, so it spuriously blocks the clear and byte1 stays 0x01). For Inc 0
    # (add/sub PC-clean) we clear byte1 unconditionally; restoring the genuine
    # high-PC carry for the <1% of programs past 0x100 is deferred to a later
    # increment with a value-derived (not TEMP-tag) carry signal.
    # Phase 7.E sem-dim: each row's byte-index gate NAME resolves from the
    # ``byte_index`` family (role = the byte position "0"/"1"/"2"); the
    # human-readable ``byte_index_name`` legacy string is kept alongside so
    # the generated rule names stay unchanged. ``PC_HEAD`` / ``H1+1..3`` are
    # structural threshold-head bank slot indices (NOT role-meaningful byte
    # positions), and ``IS_BYTE`` is an unbound token-class flag, so all
    # three stay bare. The OUTPUT ``+k`` write offset is the raw nibble value.
    rows = (
        # (byte_index role, legacy NAME for rule naming, extra NOT-conditions)
        ("0", "BYTE_INDEX_0", ()),  # predicts byte1 (0x00 for all add/sub PCs)
        ("1", "BYTE_INDEX_1", ()),  # predicts byte2
        ("2", "BYTE_INDEX_2", ()),  # predicts byte3
    )
    output_lo0 = dim_ref("output_lo", "nibble", 0)
    output_hi0 = dim_ref("output_hi", "nibble", 0)
    rules: list[FFNRule] = []
    for byte_index_role, byte_index_name, extra in rows:
        base_conds = (
            (PC_HEAD, 1.0),
            ("IS_BYTE", 1.0),
            (dim_ref("byte_index", byte_index_role), 1.0),
            # Cross-marker blockers: PC head must be the nearest one. Other
            # register/section heads are -blockers (defensive -- H1 slots are
            # near one-hot per row, but PSH_AT_SP headroom can lift a runner-up).
            ("H1+1", -BLOCKER_W),
            ("H1+2", -BLOCKER_W),
            ("H1+3", -BLOCKER_W),
        ) + extra
        # Sink the high nibbles (the leaked nibble-1 + any stray high nibble)
        # AND restore the zero nibble (LO+0 / HI+0) positive so the 0x00 byte
        # wins decisively over both the leak and the register-marker logit.
        for k in range(1, 16):
            rules.append(multi_way_and_rule(
                name=f"no_stack0_pc_clear_{byte_index_name.lower()}_lo_{k}",
                conditions=base_conds,
                threshold=2.5,
                writes=((dim_ref("output_lo", "nibble", k), WW),),
            ))
        for k in range(1, 16):
            rules.append(multi_way_and_rule(
                name=f"no_stack0_pc_clear_{byte_index_name.lower()}_hi_{k}",
                conditions=base_conds,
                threshold=2.5,
                writes=((dim_ref("output_hi", "nibble", k), WW),),
            ))
        rules.append(multi_way_and_rule(
            name=f"no_stack0_pc_clear_{byte_index_name.lower()}_lo0_restore",
            conditions=base_conds,
            threshold=2.5,
            writes=((output_lo0, WPOS),),
        ))
        rules.append(multi_way_and_rule(
            name=f"no_stack0_pc_clear_{byte_index_name.lower()}_hi0_restore",
            conditions=base_conds,
            threshold=2.5,
            writes=((output_hi0, WPOS),),
        ))

    # PC BYTE 0 (the marker row, predicts the byte-0 token). Here the GENUINE
    # computed PC byte-0 low/high nibbles live in OUTPUT (LO+<lownib> / HI+
    # <hinib>); the block-33 leak adds a constant ``OUTPUT_LO+1 = +44 /
    # OUTPUT_LO+0 = -44`` on top, flipping the low nibble to ``1`` (0x1a -> 0x11).
    # We CANNOT blanket-clear this row (it carries the real value), so we only
    # CANCEL the leaked nibble-1: sink ``OUTPUT_LO+1`` (the 0x01-replication
    # nibble) hard, leaving every OTHER nibble (the genuine value bits, incl.
    # the real low nibble at LO+<n> and the high nibble at HI+<n>) untouched so
    # the genuine PC byte-0 wins. We do NOT restore LO+0 here (that would force
    # 0x00 and clobber the real value); the leak already drove LO+0 negative so
    # 0x00 is not a contender. Gated on ``MARK_PC`` (fires ONLY at the PC marker
    # row, IS_BYTE=0 there -- verified). For the add/sub acceptance set the
    # genuine PC low nibble is never 1 (PCs 0x0a/0x12/0x1a/0x22), so sinking
    # nibble-1 is exact; a PC genuinely ending in nibble-1 (e.g. a 0x11 branch
    # target) is out of the Inc-0 scope and deferred.
    #
    # Inc 3 (if_gt step-4 branch target, corruptor B): the BZ/BNZ branch step's
    # PC-marker row carries a GENUINE branch-target byte-0 whose LOW nibble is
    # NOT 1 (e.g. 0x3a -> low nibble 0xa). Worse, sinking ``OUTPUT_LO+1`` to a
    # large NEGATIVE here is SELF-DEFEATING: the L26 tail-amplify block
    # (``OP_BZ``-gated) MAGNIFIES the -15000 sink into a large POSITIVE
    # ``OUTPUT_LO+1 = +6685`` leak (a silu/amplify nonlinearity), which makes the
    # spurious byte ``0x?1`` (0x31) out-vote the real branch target (0x3a). So on
    # the BRANCH step (``OP_BZ``/``OP_BNZ`` active, ~+5 at the marker row, ~0 on
    # every add/sub PC row) the lo1_sink must NOT fire: leave ``OUTPUT_LO+1 = 0``
    # so the L26 amplify produces the genuine low nibble (0xa) and ``0x3a`` wins.
    # ``OP_BZ``/``OP_BNZ`` are 0.00 at every add/sub MARK_PC row (probe-verified),
    # so the -BLOCKER_W guards are inert for the Inc-0 add/sub acceptance set.
    rules.append(multi_way_and_rule(
        name="no_stack0_pc_clear_byte0_marker_lo1_sink",
        conditions=(
            # Phase 7.E sem-dim: PC marker gate + BZ/BNZ branch-step blockers
            # resolve from the ``marker`` / ``opcode_flag`` families; the
            # OUTPUT ``+1`` write offset is the raw nibble value.
            (dim_ref("marker", "PC"), 1.0),
            (dim_ref("opcode_flag", "BZ"), -BLOCKER_W),
            (dim_ref("opcode_flag", "BNZ"), -BLOCKER_W),
        ),
        threshold=0.5,
        writes=((dim_ref("output_lo", "nibble", 1), WW),),
    ))

    return tuple(rules)


# 3 byte rows x (LO 1..15 sink + HI 1..15 sink + LO+0 restore + HI+0 restore)
# + 1 PC marker-row rule (LO+1 sink only)
_NO_STACK0_PC_CLEAR_HIDDEN_DIM = 3 * (15 + 15 + 1 + 1) + 1


def make_no_stack0_pc_highbyte_clear_op() -> Operation:
    """Append the no-STACK0 PC value-byte OUTPUT-clear FFN after the L25 tail.

    Standalone ``PureFFN`` post_op on the L25 tail block (appended AFTER
    ``tail_bit32_result_correction``), so it is the LAST writer of OUTPUT on
    the PC value-byte rows before the LM
    head. Gated by ``C4_NO_STACK0_EMIT`` (default OFF); flag-OFF bakes NO units
    (byte-identical to golden). See the module-level block comment for the
    root cause / mechanism (Inc 0 of the STACK0-emission-removal campaign).
    """
    if not _no_stack0_emit():
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="no_stack0_pc_highbyte_clear",
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
            spec_section="STACK0_EMISSION_REMOVAL_CAMPAIGN_2026_06_17.md#inc0",
        )

    rules = _no_stack0_pc_highbyte_clear_rules()

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
        assert len(rules) == _NO_STACK0_PC_CLEAR_HIDDEN_DIM, (
            f"no_stack0_pc_highbyte_clear rule-count drift: produced "
            f"{len(rules)}, expected {_NO_STACK0_PC_CLEAR_HIDDEN_DIM}"
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
        name="no_stack0_pc_highbyte_clear",
        reads={
            "H1", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
            "MARK_PC",
            # Inc 3 corruptor-B gate: BZ/BNZ branch-step detect for the lo1_sink.
            "OP_BZ", "OP_BNZ",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        # Append AFTER tail_bit32_result_correction on the L25 tail block, so
        # this op writes OUTPUT on the PC byte rows before the clean_emitter.
        target_op_name="l10_post_ops_combined",
        requires={"after": "tail_bit32_result_correction"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="STACK0_EMISSION_REMOVAL_CAMPAIGN_2026_06_17.md#inc0",
    )


# ===========================================================================
# CLEAN EMITTER (C4_CLEAN_EMITTER) -- the generic all-marker-row OUTPUT sink
# ===========================================================================
#
# The R-FRAME marker-dominance invariant (docs/CLEAN_EMITTER_SCOPE_2026_07_04.md,
# docs/semantic_spec_EMIT_FRAMING.md §2.b / §G5): on ANY row where a NEXT_<REG>
# marker-schedule flag is lit, the LM head is DESIGNED to emit the register
# MARKER (head[marker, NEXT_*]=+20) and never a byte (head[byte, NEXT_*]=-80).
# That suppression is out-voted when the L20/L25 tail sprays OUTPUT_LO/HI up to
# ~1e15 there (head[byte, OUTPUT_*+k]=+5 -> byte logit ~5e15 >> +20 marker), so a
# STRAY value byte wins the marker row -> the step emits != STEP_TOKENS tokens ->
# the fixed-stride decode desyncs. This is the =/=STEP_TOKENS framing MEGAROOT
# (survey-R1 ~760 programs: var/expr/if/bool/nested/loops/func).
#
# The two ORIGINAL point-fixes (``no_stack0_se_output_clear`` gated on
# ``MARK_SE_ONLY`` = the STEP_END/PC row; ``no_stack0_mem_marker_output_clear``
# gated on ``NEXT_MEM`` = the MEM row) sank OUTPUT at ONE marker row each -- only
# 2 of the 6. The AX/SP/BP marker rows (``NEXT_AX``/``NEXT_SP``/``NEXT_BP``) had
# NO corrector, so a spray landing there still drifted, and the natural next fix
# was 3 MORE copies of the same 32-rule point-fix. Both point-fixes are now
# DELETED (this op's 6-way NEXT_* sink is a strict superset of all six rows).
#
# THIS op is the CLEAN generic emitter: ONE op that sinks OUTPUT_LO/HI at EVERY
# marker-predicting row, gated on the 6-way OR of the marker-schedule flags
# ``(NEXT_PC | NEXT_AX | NEXT_SP | NEXT_BP | NEXT_MEM | NEXT_SE)``. Those flags
# are MUTUALLY-EXCLUSIVE one-hots (exactly one lit per marker row, ~0 on every
# value-byte row and on the teacher-forced MEM addr/val rows), so summing them
# with an explicit ``threshold=0.5`` fires iff ANY one is ~1.0. On a fire the
# balanced silu saturates and the sink drives every OUTPUT nibble far below the
# marker baseline, so the marker wins its row -> the frame is exactly N tokens BY
# CONSTRUCTION and the drift class cannot occur for ANY register. Standalone
# ``PureFFN`` post_op appended AFTER ``no_stack0_pc_highbyte_clear`` on the L25
# tail block (the LAST OUTPUT writer before the LM head, the slot vacated by the
# two deleted point-fix correctors it subsumes).
#
# Gating: DEFAULT ON (``C4_CLEAN_EMITTER`` unset -> ON; opt out with =0). It
# additionally requires ``_no_stack0_emit()`` (the 30-token frame, DEFAULT-ON;
# the 35-token OFF build has STACK0-block decode slack that absorbs a +1 drift).
# PROVEN net-positive on the full 1096 (516 -> 525, +9, full_trace spec_k=0
# cap-600), so it is now the production default and SUBSUMES + REPLACES the two
# point-fixes (``no_stack0_se_output_clear`` + ``no_stack0_mem_marker_output_clear``),
# which are deleted (this op's 6-way NEXT_* sink is a strict superset). Because
# ``C4_NO_STACK0_EMIT`` also defaults ON, ``tools/_isa_golden_hash.py`` (bare env)
# now bakes this op -> the golden hash CHANGED from ``91f55411`` (intended
# verdict-change). ``C4_CLEAN_EMITTER=0`` opts out.
_CLEAN_EMITTER_HIDDEN_DIM = 32  # OUTPUT_LO[0..15] + OUTPUT_HI[0..15]
# Same magnitude as the two point-fixes' ``-1e20``: it must dominate the
# L20/L25 ALU/frame spray (~+8e14..+3e20 measured) so every byte logit at a
# marker row goes hugely negative and the register MARKER wins the argmax.
_CLEAN_EMITTER_WW = -1.0e20
# The 6 marker-schedule flags (one lit per marker-predicting row). Mutually
# exclusive one-hots, so their SUM is the 6-way OR gate.
_CLEAN_EMITTER_NEXT_FLAGS = (
    "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_MEM", "NEXT_SE",
)


def _clean_emitter_enabled() -> bool:
    """Kill-switch for the CLEAN_EMITTER generic marker-row OUTPUT sink.

    DEFAULT ON: ON iff ``C4_CLEAN_EMITTER`` is not explicitly disabled AND the
    30-token frame is active (``_no_stack0_emit()``, itself DEFAULT-ON). PROVEN
    net-positive on the full 1096 (516 -> 525, +9), so it is the production
    default and replaces the two deleted point-fixes. Because both this flag and
    ``C4_NO_STACK0_EMIT`` default ON, the golden byte-identity gate
    (``tools/_isa_golden_hash.py``, bare env) now BAKES this op -> the golden
    hash intentionally CHANGED from ``91f55411``. Opt out with
    ``C4_CLEAN_EMITTER=0``; ``tools/flag_regression_gate.py --flag
    C4_CLEAN_EMITTER`` A/Bs the op ON/OFF.
    """
    if not _no_stack0_emit():
        return False
    return _os_l0.environ.get("C4_CLEAN_EMITTER", "1") != "0"


def _clean_emitter_rules() -> tuple[FFNRule, ...]:
    """32 AND rules sinking OUTPUT_LO/HI hugely negative at EVERY marker row.

    Each unit fires when the SUM of the 6 marker-schedule flags clears
    ``threshold=0.5`` -- i.e. when ANY ``NEXT_*`` is lit (they are exclusive
    one-hots, so the sum is ~1.0 at a marker row and ~0 elsewhere) -- and writes
    a large-negative value into one OUTPUT nibble. Because value-byte rows and
    the teacher-forced MEM addr/val rows carry NO ``NEXT_*`` flag, the sink
    NEVER touches a real value byte; it only sinks the semantically-dead OUTPUT
    at a marker-predicting row so the register MARKER wins there (the R-FRAME
    marker-dominance invariant).
    """
    # 6-way OR gate: all flags at weight 1.0, threshold 0.5 -> fires iff the
    # summed one-hots >= 0.5 (any single flag ~1.0). NEXT_* are slot-level
    # singleton transition flags with no (category, role) binding, so they stay
    # bare NAMEs; the OUTPUT ``+k`` write offset is the raw nibble value.
    conditions = tuple((flag, 1.0) for flag in _CLEAN_EMITTER_NEXT_FLAGS)
    WW = _CLEAN_EMITTER_WW
    rules: list[FFNRule] = []
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"clean_emitter_marker_sink_lo_{k}",
            conditions=conditions,
            threshold=0.5,
            writes=((dim_ref("output_lo", "nibble", k), WW),),
        ))
    for k in range(16):
        rules.append(multi_way_and_rule(
            name=f"clean_emitter_marker_sink_hi_{k}",
            conditions=conditions,
            threshold=0.5,
            writes=((dim_ref("output_hi", "nibble", k), WW),),
        ))
    return tuple(rules)


def make_clean_emitter_op() -> Operation:
    """Append the CLEAN_EMITTER generic all-marker-row OUTPUT sink (increment 0).

    Standalone ``PureFFN`` post_op on the L25 tail block (appended AFTER the two
    existing point-fix correctors), so it is the LAST writer of OUTPUT on every
    marker-predicting row before the LM head. Gated by ``C4_CLEAN_EMITTER``
    (DEFAULT OFF) + ``_no_stack0_emit()``; OFF bakes NO units (byte-identical to
    golden ``91f55411``). See the module-level block comment + the scope doc
    ``docs/CLEAN_EMITTER_SCOPE_2026_07_04.md`` for the drift mechanism.
    """
    if not _clean_emitter_enabled():
        # OFF: register a no-op so the dep graph / op list is stable but no
        # weights change (byte-identical to the pre-flag build).
        def _noop_bake(block, dim_positions, S):
            del block, dim_positions, S

        return Operation(
            name="clean_emitter",
            reads=set(),
            writes=set(),
            audited_empty_produces=True,
            kind="block",
            target_op_name="l10_post_ops_combined",
            requires={"after": "no_stack0_pc_highbyte_clear"},
            declarative_bake_fn=_noop_bake,
            declarative_authority="spec_generated",
            migrated=True,
            smoke_tests={"all"},
            spec_section="CLEAN_EMITTER_SCOPE_2026_07_04.md#3",
        )

    rules = _clean_emitter_rules()

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
        assert len(rules) == _CLEAN_EMITTER_HIDDEN_DIM, (
            f"clean_emitter rule-count drift: produced {len(rules)}, "
            f"expected {_CLEAN_EMITTER_HIDDEN_DIM}"
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
        name="clean_emitter",
        reads={
            "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_MEM", "NEXT_SE",
        },
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        # Append AFTER no_stack0_pc_highbyte_clear on the L25 tail block, so
        # this op is the last writer of OUTPUT on every marker row before the
        # LM head (it subsumes + replaces the two deleted point-fix correctors).
        target_op_name="l10_post_ops_combined",
        requires={"after": "no_stack0_pc_highbyte_clear"},
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir=ir,
        migrated=True,
        smoke_tests={"all"},
        spec_section="CLEAN_EMITTER_SCOPE_2026_07_04.md#3",
    )
