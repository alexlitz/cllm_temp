"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR, StepWindowConstraint
from ..isa_semantics_dsl import (
    CamConfirmSlot,
    CamKeyMatch,
    CamLookupSpec,
    CamValueBand,
    cam_lookup,
)
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .residual_band_registry import register_residual_band
from .shared import (  # noqa: F401
    _as_setdim_proxy,
    _empty_compiler_ir_factory,
    operand_from_memsp_enabled,
)


# === STACK0 campaign Inc-1 store-commit relay band (2026-06-18) ===
#
# ``MEM_STORE_AT_VAL`` is the per-value-row store-commit bit that closes the
# byte-0 ``mem[SP]`` operand-A delivery (``C4_OPERAND_FROM_MEMSP``). The L6
# relay (head 6) writes ``MEM_STORE`` only at a MEM section's MARK_MEM marker
# row; it is NOT present on the section's value-byte rows until block 11
# (AFTER the L8 head-5 mem-to-ALU CAM reads its K). So at head-5's read time
# every MEM value-byte-0 row looks identical (MEM_VAL_B1=1, MEM_STORE=0) and
# ALiBi recency picks the most-recent step's PHANTOM value row (mem byte=0)
# over the real PSH store's value row (== mem[SP]). ``make_layer7_mem_store_
# relay_op`` broadcasts MEM_STORE FORWARD from the marker row to the same
# section's value-byte-0 row into THIS band (at L7, block 9, before L8 attn),
# so head-5's K can gate on a per-value-row store bit. Flag-gated so a
# flag-OFF build omits the band entirely (smaller d_model, byte-identical).
register_residual_band(
    "MEM_STORE_AT_VAL", 1, owner="make_layer7_mem_store_relay_op",
    flag=operand_from_memsp_enabled, never_share=True,
)


# === L7 attention head layout (auto-fit; legacy head_idx as docs) ====
#
# L7 is attention-heavy: every op in this file writes Q/K/V/O for one or
# more attention heads. This table is the single source of truth for the
# L7 head axis -- every ``head_idx=N`` literal in the spec functions
# below is resolved via :data:`_L7_HEAD_LAYOUT_BY_NAME`, so the bakes
# pull the same indices they always have and byte-identity is trivially
# preserved.
#
# Each row names a *primary owner* of its head_idx. Two L7 ops legitimately
# extend an already-owned head: ``layer7_sp_byte0_is_f8`` adds V/O slots
# 6+7 to head 6 (primary owner ``layer7_memory_heads``) and
# ``format_pointer_extraction`` (gated) reuses head 7's slot range.
# Extensions are documented in the comments below and resolve their
# head_idx via :data:`_L7_HEAD_LAYOUT_BY_NAME` rather than re-pinning the
# same slot, which the allocator forbids (heads cannot be aliased).
#
# Phase 7.B.3: ``pin=`` is dropped from the allocator. Because
# :data:`_L7_HEAD_LAYOUT` is contiguous (0..7) and declared in order,
# first-fit reproduces the legacy ``head_idx`` values bit-for-bit. The
# load-bearing copy is :data:`_L7_HEAD_LAYOUT_BY_NAME`, consumed by the
# head-spec factories that write Q/K/V/O weights at the resolved
# ``head_idx``; the ``legacy_head_idx`` column below is documentation only.
#
# Order mirrors the op-factory order in this file (operand_gather, then
# memory_heads heads 2-7) so the layout reads top-to-bottom alongside the
# specs that own each row.
_L7_HEAD_LAYOUT = (
    # (op_name, legacy_head_idx (docs only))
    ("layer7_operand_gather.head_0",     0),  # operand A gather (STACK0 byte 0 -> ALU)
    ("layer7_operand_gather.head_1",     1),  # operand A gather (BP/SP OUTPUT -> ALU for LEA/ADJ/ENT)
    ("layer7_memory_heads.head_2",       2),  # gather prev AX byte 0 -> ADDR_B0_LO/HI
    ("layer7_memory_heads.head_3",       3),  # gather prev AX byte 1 -> ADDR_B1_LO/HI
    ("layer7_memory_heads.head_4",       4),  # gather prev AX byte 2 -> ADDR_B2_LO/HI
    ("layer7_memory_heads.head_5",       5),  # LI/LC/LEA/bitwise/JSR/no-carry/ADD/SUB flag relay
    ("layer7_memory_heads.head_6",       6),  # PSH/CMP relay (extended by layer7_sp_byte0_is_f8)
    ("layer7_memory_heads.head_7",       7),  # MEM flag broadcast (reused by format_pointer_extraction when gated)
)
_L7_HEAD_LAYOUT_BY_NAME = {name: head_idx for name, head_idx in _L7_HEAD_LAYOUT}


def _allocate_layer7_heads() -> AttentionHeadAllocator:
    """Build a per-bake :class:`AttentionHeadAllocator` with all L7 heads.

    Phase 7.B.3: ``pin=`` is dropped from every entry. The allocator's
    first-fit picks the lowest free head index in declaration order;
    because :data:`_L7_HEAD_LAYOUT` is contiguous (0..7) and ordered,
    first-fit reproduces the legacy ``head_idx`` values bit-for-bit.
    The actual weight-write head indices are still looked up via
    :data:`_L7_HEAD_LAYOUT_BY_NAME` inside the head-spec factories
    below, so byte-identity with the legacy bake is preserved
    regardless of allocator order. Returns the allocator so callers
    can attach it to the ``attn`` module for inspection.
    """
    allocator = AttentionHeadAllocator(layer_max_heads=8)
    for name, _legacy_head_idx in _L7_HEAD_LAYOUT:
        allocator.alloc(name, 7)
    return allocator


# === L7 FFN unit layout (auto-fit; legacy offsets retained as docs) ===
#
# L7 is attention-only: every op in this file is ``kind="block"`` and
# writes attention weights (Q/K/V/O), not FFN hidden units. There is no
# ``_set_layer7_ffn`` helper in ``vm_step`` and no ``ffn_units_used``
# annotation on any L7 op. The allocator is bookkeeping-only: each op
# claims a 1-unit placeholder so the L7 FFN-unit layout is auditable in
# the same way as L1/L2/L9, and a future L7 op family that DOES need
# FFN units can claim a free range via ``allocator.alloc(name, n)``.
# Byte-identity is trivially preserved -- no FFN weights are touched by
# any L7 bake.
#
# Phase 7.B.3: every entry below is auto-placed by
# :class:`FFNUnitAllocator` first-fit. Because the layout is fully
# contiguous in declaration order, first-fit reproduces the legacy
# placeholder offsets (0..3) bit-for-bit; the ``legacy_start`` column
# is kept purely as documentation.
#
# Order mirrors the op-factory order in this file (operand_gather first,
# then memory_heads, format_pointer_extraction, sp_byte0_is_f8) so the
# layout reads top-to-bottom alongside the factories that own each slot.
_L7_FFN_UNIT_LAYOUT = (
    # (sub-stage name, legacy_start (docs only), n_units)
    ("layer7_operand_gather.placeholder",       0, 1),  # L7 head 0+1
    ("layer7_memory_heads.placeholder",         1, 1),  # L7 heads 2-7
    ("format_pointer_extraction.placeholder",   2, 1),  # L7 head 7 (gated)
    ("layer7_sp_byte0_is_f8.placeholder",       3, 1),  # L7 head 6 ext
)


def _allocate_layer7_ffn_units() -> FFNUnitAllocator:
    """Build a per-bake :class:`FFNUnitAllocator` with all L7 sub-stages.

    Phase 7.B.3: ``pin=`` is dropped from every entry. First-fit
    over the contiguous placeholder layout reproduces offsets 0..3
    bit-for-bit. No FFN weights are written by any L7 bake, so the
    allocator state is pure bookkeeping. Returns the allocator so
    callers can inspect or extend it.
    """
    allocator = FFNUnitAllocator()
    for name, _legacy_start, n_units in _L7_FFN_UNIT_LAYOUT:
        allocator.alloc(name, n_units)
    return allocator


def make_layer7_operand_gather_op() -> Operation:
    """L7 attention: operand A gather (prev STACK0 byte 0 → ALU at AX marker).

    Pinned to ``layer_idx=7`` via ``kind="block"``: legacy_bake no longer
    calls ``_set_layer7_operand_gather`` (it was migrated to the compiler),
    so without pinning the dep-graph would silently bake into a different
    block and leave block 7 zero-init.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. L7 is attention-only so every
        # range here is a 1-unit placeholder pinned to its existing
        # offset -- no FFN weights are written, byte-identity is
        # trivially preserved. Stashed on ``block.ffn`` for inspection /
        # extension by downstream tools, mirroring the L9 convention
        # from ca775eb.
        allocator = _allocate_layer7_ffn_units()
        block.ffn._l7_unit_allocator = allocator

        # Per-bake attention-head allocator. Pins every L7 head_idx at
        # its existing slot so the bake is byte-identical; stashed on
        # ``block.attn`` for inspection / extension by downstream tools.
        head_allocator = _allocate_layer7_heads()
        block.attn._l7_head_allocator = head_allocator

        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn, _layer7_operand_gather_head_specs(BD), HD
        )

    # Dim-ownership claims: L7 attn heads 0 + 1 operand gather.
    #   Head 0 V slot 1+k reads CLEAN_EMBED_LO+k (STACK0 byte 0 → ALU_LO at AX)
    #   Head 0 V slot 17+k reads CLEAN_EMBED_HI+k (STACK0 byte 0 → ALU_HI at AX)
    #   Head 1 V slot 1+k reads OUTPUT_LO+k (BP/SP OUTPUT → ALU_LO at AX for LEA/ADJ/ENT)
    #   Head 1 V slot 17+k reads OUTPUT_HI+k
    _claims = set()
    for k in range(16):
        _claims.add((7, "attn_W_v", f"0_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((7, "attn_W_v", f"0_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
        _claims.add((7, "attn_W_v", f"1_{1 + k}", f"OUTPUT_LO+{k}"))
        _claims.add((7, "attn_W_v", f"1_{17 + k}", f"OUTPUT_HI+{k}"))

    return Operation(
        name="layer7_operand_gather",
        # Phase 8.A targeted: head 1's V slots read BP/SP OUTPUT_LO via
        # attention back to the prev-step BP/SP marker row (LEA/ADJ/ENT
        # operand A relay). L7 fires before any same-step OUTPUT_LO
        # producer (L8-L17), so the available residual is step N-1's
        # value. Declare via the OUTPUT_LO_PREV_STEP alias (same numeric
        # slot 174, byte-identical bake) to retire the 31 OUTPUT_LO
        # back-edges into this op visible in the SCC audit. Mirrors the
        # L3 head 5 / L8 head 6 pattern from Phase 7.A.3.b.
        #
        # Phase 8.A G7 follow-up: OUTPUT_HI gets the same cross-step
        # rename. Head 1's V slot 17+k reads OUTPUT_HI at the attended
        # BP/SP marker row -- that token was last a current-step AX
        # marker in a previous step, so its residual OUTPUT_HI carries
        # step N-1's value. L7 (phase 7) fires before any same-step
        # OUTPUT_HI writer (L8+), so the read is genuinely cross-step.
        # Retarget to OUTPUT_HI_PREV_STEP (alias of OUTPUT_HI at numeric
        # position 190; byte-identical bake) to retire 7 back-edges into
        # this op (1 from layer9_alu, 1 from
        # layer10_psh_stack0_passthrough_bake, and 5 L14 OUTPUT_HI
        # writers). See .agent-logs/scheduler_phase_a_2026_06_02.md.
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — add
        # MARK_BP, MARK_SP (head 1 K reads ``BD.MARK_BP`` + ``BD.MARK_SP``
        # at slot 0; BP/SP marker self-attention gate for LEA/ADJ/ENT).
        reads={"MARK_AX", "STACK0_BYTE0", "OP_LEA", "OP_ADJ", "OP_ENT",
               "OP_IMM",  # head 1 Q: per-step actual-IMM suppression gate
               "CONST",
               "MARK_BP", "MARK_SP",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "OUTPUT_LO.*.-1", "OUTPUT_HI.*.-1"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer7_operand_gather_ir,
        declarative_authority="spec_generated",
        # Phase 8.G.6: drop ``layer_idx=7`` literal; bind to the L7 ffn
        # anchor ``layer8_sp_gather`` so the block op resolves to
        # whichever layer the compiler places the anchor at.
        target_op_name="layer8_sp_gather",
        migrated=True,
        claims=_claims,
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # L7 head 0 + head 1 produce the fresh in-step ALU_LO/HI at the AX
        # marker (operand A for binary ops + LEA destination address).
        # L8 ALU and L9 ALU consume these via their AX-marker reads.
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeComparison::test_eq_basic",
            "TestSmokeComparison::test_ne_basic",
        },
        spec_section="BLOG_SPEC.md#the-attention-layer",
        compaction_safe=True,
    )


def _layer7_operand_gather_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer7_operand_gather_head_specs(BD))
    return ir


# === L7 operand-gather head 0 as a CAM / frame-lookup head ==================
#
# Head 0 is the canonical content-addressable relay: Q@MARK_AX, K@STACK0_BYTE0,
# V=CLEAN_EMBED_{LO,HI}, O->ALU_{LO,HI} — attend to the prev STACK0 byte-0 token
# and relay its value into the ALU at the AX marker (operand A). It is the
# SETTLED, healthy head (memory ``project_l7_operand_gather_not_broken``:
# STACK0->ALU_LO relay prob=1.0). It is re-expressed here through the
# :func:`cam_lookup` generator (``isa_semantics_dsl``) — the attention-side
# DSL-parity increment future operand-relay / frame-lookup fixes route through
# instead of hand-building. Byte-identity is gated by the whole-model
# state_dict hash (CPU, disk_cache=False) staying unchanged.
#
# The CAM invariant is structural: the row select is ONE ``CamKeyMatch``
# (MARK_AX query / STACK0_BYTE0 key on slot 0), the operand opcode-blockers are
# the declared ``query_blockers`` overlay (OP_LEA/ADJ/ENT reject the LEA/ADJ/ENT
# frames head 1 owns), the CONST-anchored slot-33 sharpener is the
# ``CamConfirmSlot``, and the value flow is two ``CamValueBand`` relay blocks.
# The slope (0.5) is set by the op's ``attn.alibi_slopes.fill_(0.5)`` bake, NOT
# the spec, so ``alibi_slope`` stays ``None`` (byte-identical). ``value_active``
# carries the C4_OPERAND_FROM_MEMSP "keep the Q/K gates, drop the V/O relay"
# suppression so the head slot/layout is unchanged when the operand read is
# re-routed to mem[SP].
def _operand_gather_head0_cam_spec() -> CamLookupSpec:
    """Build the CAM spec for L7 operand-gather head 0 (names only).

    ``value_active`` reflects ``C4_OPERAND_FROM_MEMSP``: when the operand read
    is re-routed to mem[SP] (the L4 SP->ADDR_KEY + L8 mem-to-ALU CAM) head 0's
    STACK0->ALU relay must NOT also fire (it would double-write ALU from the
    emitted STACK0 token). Suppressing the V/O while keeping the Q/K gates
    leaves the head slot/layout unchanged so head 1 (LEA/ADJ/ENT frame relay)
    is untouched — STACK0_VIA_MEM_ATTENTION_PLAN.md Phase 1 step 3.
    """
    L = 15.0
    return CamLookupSpec(
        name="layer7_operand_gather_head0",
        key_match=CamKeyMatch(
            query_dim="MARK_AX", key_dim="STACK0_BYTE0", weight=L,
            query_slot=0,
        ),
        # Reject the LEA/ADJ/ENT frames head 1 owns (the operand-A gather only
        # fires for the binary-op AX markers).
        query_blockers=(("OP_LEA", -L), ("OP_ADJ", -L), ("OP_ENT", -L)),
        confirm=CamConfirmSlot(
            slot=33, marker_weight=L, const_q_weight=-L / 2,
            const_k_weight=L,
            blockers=(("OP_LEA", -L * 10), ("OP_ADJ", -L * 10),
                      ("OP_ENT", -L * 10)),
        ),
        value_bands=(
            CamValueBand("CLEAN_EMBED_LO", "ALU_LO", 16, 1, 6.0),
            CamValueBand("CLEAN_EMBED_HI", "ALU_HI", 16, 17, 6.0),
        ),
        const_dim="CONST",
        value_active=not operand_from_memsp_enabled(),
    )


def _operand_gather_head0_dim_map(BD) -> dict:
    """Resolve every dim name the head-0 CAM touches via the ``BD`` proxy.

    ``_layer7_operand_gather_head_specs`` receives a ``_SetDim``-like proxy
    whose ``.NAME`` attributes are the compiler-allocated positions; the CAM
    builder needs a name->int dict. Resolve the FULL superset (both flag
    states' value bands) so the dict is stable regardless of
    ``C4_OPERAND_FROM_MEMSP``, byte-identical to the hand-built ``BD.NAME``
    lookups.
    """
    names = {
        "MARK_AX", "STACK0_BYTE0", "CONST", "OP_LEA", "OP_ADJ", "OP_ENT",
        "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "ALU_LO", "ALU_HI",
    }
    return {n: int(getattr(BD, n)) for n in names}


def _layer7_operand_gather_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``vm_step._set_layer7_operand_gather``.

    Head 0 is generated by :func:`cam_lookup` (the CAM / frame-lookup DSL);
    head 1 (LEA/ADJ/ENT frame relay) stays hand-built. The CAM bundle is built
    per-call (NOT at module scope) so ``value_active`` re-reads
    ``C4_OPERAND_FROM_MEMSP`` at bake time, matching the original behaviour
    where the V/O suppression was re-evaluated on every bake.
    """

    L = 15.0
    AX_I = 1

    head0_idx = _L7_HEAD_LAYOUT_BY_NAME["layer7_operand_gather.head_0"]
    head0_cam = cam_lookup(_operand_gather_head0_cam_spec())
    head0 = head0_cam.head_spec_builder(
        _operand_gather_head0_dim_map(BD), head0_idx
    )

    return (
        head0,
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_operand_gather.head_1"],
            q=(
                AP(0, BD.MARK_AX, L * 10),
                AP(0, BD.OP_LEA, L),
                AP(0, BD.OP_ADJ, L),
                AP(0, BD.OP_ENT, L),
                AP(0, BD.CONST, -L * 5),
                # PER-STEP ACTUAL-IMM SUPPRESSION (func / simple_function fix).
                # Head 1 pulls the live frame BP/SP into ALU at the AX marker
                # for LEA/ADJ/ENT operand-A relay. Its dim-0 score is dominated
                # by MARK_AX * 10 (= 150), so it fires on EVERY AX marker
                # regardless of opcode -- including the callee `IMM` step that
                # follows an ENT (where it buries the magnitude-1 immediate
                # under the frame value and L8 emits the ENT frame constant 8;
                # this is the `ENT 0; IMM 42` -> 8 bug and the func / 150-fail
                # cluster). The per-step opcode decode at the L5 main-AX path
                # (``_opcode_decode_main_rules``) produces a CLEAN per-step
                # ``OP_IMM`` one-hot at the AX marker that survives intact to
                # this block's input (probed spec_k=0: OP_IMM = +5.0 ONLY on
                # real IMM steps, ~0 on ENT/ADJ/LEA/PSH/OR steps -- the
                # discriminator the prior "OP_ENT is a constant" analyses
                # missed by reading OP_ENT *after* block 8 re-broadcasts it).
                # A real ENT/LEA/ADJ step carries OP_IMM = 0, so this term is
                # inert there and head 1 still fires for its load-bearing
                # cases (ADJ + bitwise-pop). On a real IMM step OP_IMM = +5.0
                # drives the dim-0 score deeply negative, so head 1 does NOT
                # gather the frame and the genuine immediate reaches ALU
                # (mirroring the passing non-call IMM case).
                AP(0, BD.OP_IMM, -L * 30),
                AP(1, BD.CONST, -L * 2),
                AP(1, BD.MARK_AX, L * 3),
            ),
            k=(
                AP(0, BD.MARK_BP, L),
                AP(0, BD.MARK_SP, L),
                AP(1, BD.CONST, 1.0),
            ),
            v=(
                _band_projection_writes(1, BD.OUTPUT_LO)
                + _band_projection_writes(17, BD.OUTPUT_HI)
            ),
            o=(
                _band_output_writes(BD.ALU_LO, 1, 6.0)
                + _band_output_writes(BD.ALU_HI, 17, 6.0)
            ),
        ),
    )


def make_layer7_memory_heads_op() -> Operation:
    """L7 attention heads 2-7: memory + flag broadcast heads.

    Pinned to ``layer_idx=7``. See ``make_layer7_operand_gather_op``.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. See ``make_layer7_operand_gather_op``
        # for the rationale: L7 is attention-only so this is byte-identical
        # bookkeeping.
        allocator = _allocate_layer7_ffn_units()
        block.ffn._l7_unit_allocator = allocator

        # Per-bake attention-head allocator. See
        # ``make_layer7_operand_gather_op`` for the rationale: pins every
        # L7 head_idx at its existing slot so byte-identity is preserved.
        head_allocator = _allocate_layer7_heads()
        block.attn._l7_head_allocator = head_allocator

        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = 5.0  # head 1: MEM flag broadcast
            attn.alibi_slopes[5] = 5.0  # head 5: LI/LC flag relay
            attn.alibi_slopes[6] = 5.0  # head 6: PSH/store flag relay
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_heads(
            attn, _layer7_memory_head_specs(BD), HD
        )

    # Dim-ownership claims: L7 memory heads 2-4 (gather prev AX bytes →
    # ADDR_B*_LO/HI).
    #   For head h in {2, 3, 4}:
    #     W_v[h*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #     W_v[h*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   Head 5: scalar relay flags, each at distinct V slot/column.
    #   Head 6: scalar PSH/ENT/JSR relays.
    #   Head 7: scalar MEM marker flag broadcast.
    _claims = set()
    for h in range(2, 5):
        for k in range(16):
            _claims.add((7, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((7, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
    # Head 5 scalar relays (V slots 1..13 → distinct output dims).
    _claims.add((7, "attn_W_v", "5_1", "OP_LI+0"))
    _claims.add((7, "attn_W_v", "5_2", "OP_LC+0"))
    _claims.add((7, "attn_W_v", "5_3", "OP_LEA+0"))
    _claims.add((7, "attn_W_v", "5_8", "OP_JSR+0"))
    _claims.add((7, "attn_W_v", "5_10", "OP_SI+0"))
    _claims.add((7, "attn_W_v", "5_11", "OP_SC+0"))
    _claims.add((7, "attn_W_v", "5_12", "OP_ADD+0"))
    _claims.add((7, "attn_W_v", "5_13", "OP_SUB+0"))
    # Wave 1 Cluster B1 (2026-06-07): OP_ENT relay at head 5 V slot 14.
    _claims.add((7, "attn_W_v", "5_14", "OP_ENT+0"))
    # Head 7 MEM flag broadcast.
    _claims.add((7, "attn_W_v", "7_1", "MEM_STORE+0"))
    _claims.add((7, "attn_W_v", "7_2", "MEM_ADDR_SRC+0"))
    _claims.add((7, "attn_W_v", "7_3", "OP_JSR+0"))
    _claims.add((7, "attn_W_v", "7_4", "OP_ENT+0"))

    return Operation(
        name="layer7_memory_heads",
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L7 in the same
        # step). The same-step values written by L3 carry_forward / L5
        # opcode_decode are still picked up at the same numeric position.
        # L7 also self-writes TEMP for the NOCARRY_ALU_OP relay (head 5 V
        # slot 9) — that write is independent of the read. Breaks L11/L14 →
        # layer7_memory_heads back-edges on TEMP.
        # Phase 8.A targeted: AX_CARRY_HI_PREV_STEP marks the L7 read as
        # cross-step relative to L8 AX_CARRY_HI writers (multibyte_fetch
        # {,_bake}, head6_ax_carry_refresh). L7 fires before L8 in the
        # same step. Same-step L3 / L5 contributions still land at the
        # same numeric position.
        # Phase: docs/DIM_LIVENESS_FINDINGS_2026_06_05.md audit — declare
        # the previously-undeclared baked reads/writes for the L7 memory
        # heads (heads 2-7):
        #   - Q gates use BYTE_INDEX_0/1/2, H1/H3/H4 (marker-distance
        #     heads), IS_BYTE, MARK_SP, CONST. The structural offsets
        #     index size-7 marker-distance bands.
        #   - Head 7 V reads MEM_STORE/MEM_ADDR_SRC/OP_ENT (and writes
        #     them back via O); these are flag relays.
        #   - Head 6 V reads CMP+0..4 (CMP[3] relay) and writes CMP back.
        #   - Head 5 V reads OP_LEA (relayed to CMP[7]).
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "MARK_SP", "MARK_BP",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1", "H3", "H4",
               "IS_BYTE", "CONST",
               "MEM_STORE", "MEM_ADDR_SRC", "OP_ENT",
               "CMP",
               "OP_LI", "OP_LC", "OP_LEA", "OP_PSH", "OP_SI", "OP_SC",
               "OP_ADD", "OP_SUB",
               # Head 5 reads OP_AND/OP_OR/OP_XOR for the bitwise byte
               # propagation relays and OP_SHR for the byte-zero cleanup relay.
               "OP_AND", "OP_OR", "OP_XOR", "OP_SHR",
               "OP_JSR",  # head 5 V slot 8 (existing, declared for completeness)
               # Head 5 K-side blocker on OP_IMM (2026-06-05 OP_IMM blocker):
               # negative term pushes IMM-dispatch AX positions out of the
               # softmax mass so head 5 cannot relay OP_LEA on IMM steps.
               "OP_IMM",
               "PSH_AT_SP",
               "AX_CARRY_LO.*.-1", "AX_CARRY_HI.*.-1", "TEMP.*.-1"},
        writes={"OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP",
                "TEMP", "ADDR_KEY",
                # V7 Block 13 (2026-05-12): head 5 V slot 9 writes the
                # NOCARRY_ALU_OP relay to TEMP[7]. (TEMP is already in writes
                # but listed here for clarity.) Head 5 also writes the OP_JSR
                # relay back to OP_JSR at AX byte positions (added 2026-05-12).
                "OP_JSR", "OP_SI", "OP_SC",
                # Heads 2-4 O write ADDR_B0_LO/HI / ADDR_B1_LO/HI /
                # ADDR_B2_LO/HI bands (the "ADDR_KEY" entry above named
                # the conceptual band but the residual writes land in
                # these per-byte sub-bands).
                "ADDR_B0_LO", "ADDR_B0_HI",
                "ADDR_B1_LO", "ADDR_B1_HI",
                "ADDR_B2_LO", "ADDR_B2_HI",
                # Head 7 O writes MEM_STORE / MEM_ADDR_SRC / OP_ENT back
                # to MEM byte positions; head 5 / 6 O write CMP slots.
                "MEM_STORE", "MEM_ADDR_SRC", "OP_ENT", "CMP"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer7_memory_heads_ir,
        # Phase 8.G.6: drop ``layer_idx=7`` literal; bind to the L7 ffn
        # anchor ``layer8_sp_gather`` so the block op resolves to
        # whichever layer the compiler places the anchor at.
        target_op_name="layer8_sp_gather",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer7_memory_heads_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.extend(_layer7_memory_head_specs(BD))
    return ir


def make_layer7_mem_store_relay_op(enable: bool = False) -> Operation:
    """L7 attn head 8: relay the store-commit bit MARK_MEM-row → value rows.

    STACK0 campaign Inc-1 closing fix (2026-06-18). Flag-gated by
    ``C4_OPERAND_FROM_MEMSP`` (DEFAULT OFF = byte-identical; the band is also
    flag-gated so a flag-OFF build omits the dim).

    The L8 head-5 mem-to-ALU CAM (``make_layer8_mem_to_alu_op``) reads its K
    from the L8-attn block (block 11) INPUT, where every MEM value-byte-0 row
    is indistinguishable: (MEM_VAL_B1=1, MEM_STORE=0). The store-commit bit
    ``MEM_STORE`` is only present on the section's MARK_MEM marker row (set by
    the L6 head-6 relay, available from block 7 onward) — never on the value
    rows until block 11 (too late). So head-5's ALiBi recency picks the most
    recent step's PHANTOM value row (mem byte=0) over the real PSH store's
    value row (== mem[SP]); operand-A is delivered as 0.

    This head broadcasts MEM_STORE FORWARD from the marker row to the same
    MEM section's value-byte-0 row into the fresh ``MEM_STORE_AT_VAL`` band,
    at L7 (block 9 — BEFORE the L8 attn block), so head-5's K can gate on a
    PER-VALUE-ROW store bit:

      - Q fires at the MEM value-byte-0 row (gate ``MEM_VAL_B1``).
      - K matches the MARK_MEM marker row (the store-bit carrier). ALiBi
        recency (slope 0.5, set by the L7 op's ``alibi_slopes.fill_(0.5)``)
        selects the NEAREST preceding MARK_MEM — i.e. the value row's OWN
        section marker (+5 tokens earlier) — not an older section's marker.
      - V copies ``MEM_STORE`` from that marker row; O writes it to
        ``MEM_STORE_AT_VAL`` at the value-byte-0 row.

    GPU-confirmed (tools/probe_mem_store_rows.py): add_9/sub_17 step3, the
    real PSH store's value-byte-0 row (pos 110) attends its marker (pos 105,
    MEM_STORE=1.0) → MEM_STORE_AT_VAL=1.0; the step0/step3 phantom value rows
    (pos 74/146) attend their own markers (pos 69/141, MEM_STORE=0) → 0.0.

    Disabled by default (``enable=False``); bake is a guard-clause no-op.
    """
    HEAD_IDX = 8  # free L7 head (layout declares 0..7; num_heads=11)

    def bake(block, dim_positions, S):
        if not enable:
            return
        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        base = HEAD_IDX * HD
        # The L7 operand_gather op fills ALL L7 head slopes to 0.5, so this
        # head inherits recency favoring the nearest preceding MARK_MEM. Set
        # it explicitly too (op order is not guaranteed and this op may bake
        # before operand_gather, which fills the whole buffer).
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[HEAD_IDX] = 0.5

        L = 50.0
        # Q: fire ONLY at the MEM value-byte-0 row (MEM_VAL_B1=1). A strong
        # negative const baseline keeps non-value rows from firing the head
        # (their softmax over MARK_MEM keys would still copy a store bit, but
        # the O target MEM_STORE_AT_VAL is only read by head-5 at value rows,
        # so over-firing elsewhere is inert — restricting Q keeps it clean).
        attn.W_q[base, BD.MEM_VAL_B1] = L
        attn.W_q[base, BD.CONST] = -L / 2
        # K: match the MARK_MEM marker row (where MEM_STORE lives). CONST
        # baseline keeps non-marker rows below the marker; ALiBi recency then
        # picks the nearest preceding marker = this value row's own section.
        attn.W_k[base, BD.MARK_MEM] = L
        attn.W_k[base, BD.CONST] = -L / 2
        # V/O: copy MEM_STORE from the attended marker row into
        # MEM_STORE_AT_VAL at the firing value-byte-0 row.
        attn.W_v[base + 1, BD.MEM_STORE] = 1.0
        attn.W_o[BD.MEM_STORE_AT_VAL, base + 1] = 1.0

    _claims = set()
    if enable:
        _claims.add((7, "attn_W_v", f"{HEAD_IDX}_1", "MEM_STORE+0"))
        _claims.add((7, "attn_W_o", "MEM_STORE_AT_VAL+0", f"{HEAD_IDX}_1"))

    # reads/writes reference MEM_STORE_AT_VAL, which is a flag-gated residual
    # band (omitted from dim_positions when C4_OPERAND_FROM_MEMSP is off). The
    # bake is a no-op when disabled, so declare the band-referencing reads/
    # writes only when enabled — otherwise the dep-graph validator rejects the
    # undeclared dim on a flag-OFF build.
    _reads = {"MEM_VAL_B1", "MARK_MEM", "MEM_STORE", "CONST"} if enable else set()
    _writes = {"MEM_STORE_AT_VAL"} if enable else set()

    return Operation(
        name="layer7_mem_store_relay",
        reads=_reads,
        writes=_writes,
        kind="block",
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        # Bind to ``layer8_multibyte_fetch`` — the LOGICAL-L7 attn anchor
        # (block 9) — NOT ``layer8_sp_gather`` (logical L8, block 11, where
        # head-5 itself bakes and reads its K from the block INPUT). The
        # relay MUST land STRICTLY BEFORE block 11 so MEM_STORE_AT_VAL is
        # present on the value rows at head-5's read time. (The "L7" ops that
        # anchor to layer8_sp_gather are physically at logical L8 / block 11
        # despite the legacy naming; layer8_multibyte_fetch is the genuine
        # logical-L7 attn op, block 9, which runs before block 11.)
        target_op_name="layer8_multibyte_fetch",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
        compiler_ir=CompilerIR(),
    )


def _band_projection_writes(slot_base: int, dim_base: int, weight: float = 1.0):
    return tuple(AP(slot_base + k, dim_base + k, weight) for k in range(16))


def _band_output_writes(dim_base: int, slot_base: int, weight: float = 1.0):
    return tuple(AO(dim_base + k, slot_base + k, weight) for k in range(16))


def _layer7_memory_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``vm_step._set_layer7_memory_heads``.

    The specs below intentionally mirror the legacy matrix writes one-to-one.
    They are kept as data so this operation no longer needs to call the
    imperative helper, while byte-identity can still be tested against it.
    """

    L = 15.0
    MEM_I = 4
    AX_I = 1
    SP_I = 2
    BP_I = 3

    specs: list[DeclarativeAttentionHeadSpec] = [
        # Head 7: MEM flag broadcast (MEM marker -> MEM byte positions).
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_7"],
            q=(
                AP(0, BD.MARK_MEM, L),
                AP(0, BD.H3 + MEM_I, L),
                AP(0, BD.H1 + AX_I, -L),
                AP(0, BD.H1 + SP_I, -L),
                AP(0, BD.H1 + BP_I, -L),
                AP(0, BD.H4 + BP_I, -L),
            ),
            k=(AP(0, BD.MARK_MEM, L),),
            v=(
                AP(1, BD.MEM_STORE, 1.0),
                AP(2, BD.MEM_ADDR_SRC, 1.0),
                AP(3, BD.OP_JSR, 1.0),
                AP(4, BD.OP_ENT, 1.0),
            ),
            o=(
                AO(BD.MEM_STORE, 1, 1.0),
                AO(BD.MEM_ADDR_SRC, 2, 1.0),
                AO(BD.OP_JSR, 3, 1.0),
                AO(BD.OP_ENT, 4, 1.0),
            ),
            # ANY_STEP: MEM flag broadcast reads the MEM marker (a
            # memory-side cross-step persistence channel — MEM_STORE /
            # MEM_ADDR_SRC are programmed by prior SI/LI steps and read
            # by subsequent steps). Memory persistence across steps is
            # explicit, so cross-window K-reads here are correct, not
            # a leak.
            step_window=StepWindowConstraint.ANY_STEP,
        )
    ]

    # Heads 2-4: Gather previous AX bytes into address-byte staging dims.
    _head_2_4_layout_names = (
        "layer7_memory_heads.head_2",
        "layer7_memory_heads.head_3",
        "layer7_memory_heads.head_4",
    )
    for j in range(3):
        head = _L7_HEAD_LAYOUT_BY_NAME[_head_2_4_layout_names[j]]
        byte_idx_dim = [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2][j]
        addr_lo_out = [BD.ADDR_B0_LO, BD.ADDR_B1_LO, BD.ADDR_B2_LO][j]
        addr_hi_out = [BD.ADDR_B0_HI, BD.ADDR_B1_HI, BD.ADDR_B2_HI][j]
        specs.append(
            DeclarativeAttentionHeadSpec(
                head_idx=head,
                q=(
                    AP(0, BD.MARK_AX, L),
                    AP(0, BD.H1 + AX_I, L),
                    AP(0, BD.H3 + MEM_I, -L),
                    AP(0, BD.H4 + BP_I, -L),
                    AP(33, BD.CONST, -L / 2),
                    AP(33, BD.MARK_AX, L),
                ),
                k=(
                    AP(0, byte_idx_dim, L),
                    AP(0, BD.H1 + AX_I, L),
                    AP(33, BD.CONST, L),
                ),
                v=(
                    _band_projection_writes(1, BD.CLEAN_EMBED_LO)
                    + _band_projection_writes(17, BD.CLEAN_EMBED_HI)
                ),
                o=(
                    _band_output_writes(addr_lo_out, 1)
                    + _band_output_writes(addr_hi_out, 17)
                ),
            )
        )

    # Head 5: Relay OP_LI/OP_LC/LEA/bitwise/JSR/no-carry/add/sub flags from AX marker.
    # The K scale is doubled here to preserve the softmax-sharpness fix that
    # previously ran as a post-helper row multiply in ``bake``.
    #
    # OP_IMM blocker (2026-06-05, OPCODE_ONE_HOT_FINDINGS_2026_06_05.md): the
    # K-side previously gated only on ``MARK_AX``, which made head 5 attend to
    # *any* AX marker in scope -- including the IMM-dispatch AX marker on an
    # IMM step. Head 5's V projection has ``OP_LEA`` at slot 3 (-> CMP+7 in
    # O), so OP_LEA's residual at the source AX position broadcast into the
    # firing MARK_AX even when the current step was IMM. This was the
    # ``OP_LEA -> CMP+7`` leak path the L10 ``tail_lea_local_ax_marker_byte0_e8``
    # tail rule had to filter via a ``MEM_ADDR_SRC`` positive predicate.
    #
    # Source opcodes routed by head 5 (12 total): OP_LI, OP_LC, OP_LEA,
    # OP_AND, OP_OR, OP_XOR, OP_JSR, OP_SHR, OP_SI, OP_SC, OP_ADD, OP_SUB.
    # None of these can coexist with OP_IMM at the same AX position (L5
    # ``_opcode_decode_main_rules`` is strictly one-hot by construction), so
    # a negative term on ``OP_IMM`` only suppresses IMM-dispatch AX positions
    # and is byte-identical at every real source position for the 12 relays.
    specs.append(
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_5"],
            q=(AP(0, BD.MARK_AX, L), AP(0, BD.H1 + AX_I, L)),
            k=(
                AP(0, BD.MARK_AX, L * 2.0),
                AP(0, BD.OP_IMM, -L * 2.0),  # block IMM-dispatch AX positions
            ),
            v=(
                AP(1, BD.OP_LI, 0.2),
                AP(2, BD.OP_LC, 0.2),
                AP(3, BD.OP_LEA, 0.2),
                AP(4, BD.OP_AND, 0.2),
                AP(4, BD.OP_OR, 0.2),
                AP(4, BD.OP_XOR, 0.2),
                AP(5, BD.OP_AND, 0.2),
                AP(6, BD.OP_OR, 0.2),
                AP(7, BD.OP_XOR, 0.2),
                AP(8, BD.OP_JSR, 0.2),
                AP(9, BD.OP_SHR, 0.2),
                AP(10, BD.OP_SI, 0.2),
                AP(11, BD.OP_SC, 0.2),
                AP(12, BD.OP_ADD, 0.2),
                AP(13, BD.OP_SUB, 0.2),
                # Wave 1 Cluster B1 (2026-06-07): broadcast OP_ENT to AX
                # byte positions so ``layer14_ent_ax_bytes_zero`` can fire
                # at AX bytes 1-3 and restore the C4 8-bit-AX invariant
                # at ENT step 0. Mirrors the OP_JSR slot 8 pattern (V->O
                # routes the opcode flag onto itself with a +5.0 scale).
                AP(14, BD.OP_ENT, 0.2),
            ),
            o=(
                AO(BD.OP_LI_RELAY, 1, 1.0),
                AO(BD.OP_LC_RELAY, 2, 1.0),
                AO(BD.CMP + 7, 3, 1.0),
                AO(BD.TEMP + 3, 4, 1.0),
                AO(BD.TEMP + 4, 5, 1.0),
                AO(BD.TEMP + 5, 6, 1.0),
                AO(BD.TEMP + 6, 7, 1.0),
                AO(BD.OP_JSR, 8, 5.0),
                AO(BD.TEMP + 7, 9, 1.0),
                AO(BD.OP_SI, 10, 5.0),
                AO(BD.OP_SC, 11, 5.0),
                AO(BD.TEMP + 8, 12, 1.0),
                AO(BD.TEMP + 9, 13, 1.0),
                # Wave 1 Cluster B1 (2026-06-07): OP_ENT -> OP_ENT relay
                # broadcast at AX byte positions 1-3. Scale 5.0 mirrors
                # the OP_JSR slot-8 relay.
                AO(BD.OP_ENT, 14, 5.0),
            ),
        )
    )

    # Head 6: Relay PSH/ENT/JSR from STACK0 marker and PSH_AT_SP from SP.
    specs.append(
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_6"],
            q=(
                AP(0, BD.MARK_STACK0, L),
                AP(0, BD.H4 + BP_I, L),
                AP(0, BD.H1 + BP_I, -L),
                AP(0, BD.IS_BYTE, L),
                AP(0, BD.MARK_SP, L),
                AP(0, BD.H1 + SP_I, L),
                AP(0, BD.H1 + AX_I, -L),
                AP(0, BD.H3 + MEM_I, -L),
            ),
            k=(AP(0, BD.MARK_STACK0, L), AP(0, BD.MARK_SP, L)),
            v=(
                AP(1, BD.CMP + 0, 1.0),
                AP(2, BD.CMP + 2, 1.0),
                AP(3, BD.CMP + 4, 1.0),
                AP(4, BD.PSH_AT_SP, 1.0),
                AP(5, BD.CMP + 3, 1.0),
            ),
            o=(
                AO(BD.CMP + 0, 1, 1.0),
                AO(BD.CMP + 2, 2, 1.0),
                AO(BD.CMP + 4, 3, 1.0),
                AO(BD.PSH_AT_SP, 4, 1.0),
                AO(BD.CMP + 3, 5, 1.0),
            ),
        )
    )

    return tuple(specs)


def make_format_pointer_extraction_op(enable_conversational_io: bool = False) -> Operation:
    """L7 attention head 7: extract format string pointer from STACK0.

    Originally an inline call in ``set_vm_weights`` (gated by
    ``enable_conversational_io``):
        ``_set_format_pointer_extraction(attn7, S, BD, HD)``
        plus ``attn7.alibi_slopes[7] = 5.0``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=7`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False, mirroring the legacy
    flag gate. Phase=7.5 so this runs AFTER ``layer7_operand_gather``
    and ``layer7_memory_heads`` (both phase=7) — those bakes fill the
    same alibi_slopes vector via ``fill_(0.5)``, so the slope[7]=5.0
    override must apply after them.
    """
    def bake(block, dim_positions, S):
        del S

        # Per-bake FFN-unit allocator. See ``make_layer7_operand_gather_op``
        # for the rationale. Built unconditionally (even when the
        # conversational-IO gate is off) so the L7 layout stays consistent
        # across modes; this is byte-identical bookkeeping either way.
        allocator = _allocate_layer7_ffn_units()
        block.ffn._l7_unit_allocator = allocator

        # Per-bake attention-head allocator. Built unconditionally for
        # the same reason as the FFN-unit allocator above -- the L7 head
        # axis is declared once, regardless of the conversational-IO
        # gate. Byte-identical bookkeeping.
        head_allocator = _allocate_layer7_heads()
        block.attn._l7_head_allocator = head_allocator

        if not enable_conversational_io:
            return
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[7] = 5.0  # steep to attend back to prev step
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _format_pointer_extraction_spec(_as_setdim_proxy(dim_positions)),
            HD,
        )

    return Operation(
        name="format_pointer_extraction",
        # Phase 9.B (IO_IN_OUTPUT_MODE SCC rename): SSA cross-step form.
        # null_terminator_detection (phase 10.6) stages the value for the
        # NEXT step. Same numeric slot via alias; byte-identical bake.
        reads={"IO_IN_OUTPUT_MODE.*.-1", "MARK_STACK0", "EMBED_LO", "EMBED_HI"},
        writes={"FORMAT_PTR_LO", "FORMAT_PTR_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _format_pointer_extraction_ir
            if enable_conversational_io
            else _empty_compiler_ir_factory
        ),
        declarative_authority="spec_generated",
        # Phase 8.G.6: drop ``layer_idx=7`` literal; bind to the L7 ffn
        # anchor ``layer8_sp_gather`` so the block op resolves to
        # whichever layer the compiler places the anchor at.
        target_op_name="layer8_sp_gather",
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _format_pointer_extraction_ir(dim_positions, HD) -> CompilerIR:
    del HD
    proxy = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_format_pointer_extraction_spec(proxy))
    return ir


def _format_pointer_extraction_spec(BD) -> DeclarativeAttentionHeadSpec:
    L = 20.0
    v = []
    o = []
    for k in range(16):
        v.append(AP(1 + k, BD.EMBED_LO + k, 1.0))
        v.append(AP(17 + k, BD.EMBED_HI + k, 1.0))
        o.append(AO(BD.FORMAT_PTR_LO + k, 1 + k, 1.0))
        o.append(AO(BD.FORMAT_PTR_HI + k, 17 + k, 1.0))
    # head_idx sourced from _L7_HEAD_LAYOUT: format_pointer_extraction
    # reuses head 7's slot range when ``enable_conversational_io`` is
    # on. The primary owner of head 7 is ``layer7_memory_heads`` (MEM
    # flag broadcast); when the gate is on this op writes V/O slots
    # 1..32 that the primary owner does not touch.
    return DeclarativeAttentionHeadSpec(
        head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_7"],
        q=(AP(0, BD.IO_IN_OUTPUT_MODE, L),),
        k=(AP(0, BD.MARK_STACK0, L),),
        v=tuple(v),
        o=tuple(o),
    )


# ---------------------------------------------------------------------------
# B7-2: SP_BYTE0_IS_F8 producer
# ---------------------------------------------------------------------------

# V/O slots 6 and 7 are added to L7 head 6 (the PSH/CMP relay head). Slots
# 1..5 are already occupied by the relay writes in
# ``_layer7_memory_head_specs``; ``Primitives.generate_attention_head``
# writes only the slots referenced by the spec, so adding slots 6+7 here
# leaves the existing head 6 wiring intact. We use two slots (one per
# nibble bit) so the consumer threshold rejects the half-match cases
# where only one nibble lines up with 0xF8.
_SP_BYTE0_F8_V_SLOT_LO = 6
_SP_BYTE0_F8_V_SLOT_HI = 7


def make_layer7_sp_byte0_is_f8_op() -> Operation:
    """L7 head-6 extension: produce ``SP_BYTE0_IS_F8`` at MARK_SP rows.

    Designed per B6-K's BD dim usage map Section 5 and B6-G's L7-L9
    structural audit Section 2.1.

    Semantics:
      - Q fires at MARK_SP rows (same head 6 attention pattern that
        already relays PSH/CMP/JSR flags).
      - K fires at MARK_SP rows; head 6 ALiBi slope is 5.0 so self-
        attention dominates.
      - V slot 6 reads ``EMBED_LO+8`` from the attended MARK_SP row;
        slot 7 reads ``EMBED_HI+15``. L3 head 2 carry-forward writes
        SP byte 0's EMBED_LO/HI nibbles onto the MARK_SP row, so when
        the carry-forwarded SP byte 0 == 0xF8 both reads are 1.0.
      - O slots 6 and 7 both write to ``SP_BYTE0_IS_F8`` with weight
        0.5, summing to 1.0 only when both nibbles match. The L10
        consumer ``tail_sp_marker_byte0_f8_from_initial_stack_exact``
        gates on ``SP_BYTE0_IS_F8 +1e6`` (B6-K Section 5 design).

    Phase 7.6: runs after ``layer7_memory_heads`` (phase=7) and
    ``format_pointer_extraction`` (phase=7.5), so adding head-6 slots
    cannot be overwritten by either. Adding extra V/O slots on an
    already-baked head is safe because
    ``Primitives.generate_attention_head`` writes only the (head, slot)
    cells referenced by the spec — slots 1..5 from the prior bake stay
    intact.
    """

    def bake(block, dim_positions, S):
        del S
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)

        # Per-bake FFN-unit allocator. See ``make_layer7_operand_gather_op``
        # for the rationale: L7 is attention-only so this is byte-identical
        # bookkeeping.
        allocator = _allocate_layer7_ffn_units()
        block.ffn._l7_unit_allocator = allocator

        # Per-bake attention-head allocator. See
        # ``make_layer7_operand_gather_op`` for the rationale: pins every
        # L7 head_idx at its existing slot. Head 6 is owned by
        # ``layer7_memory_heads``; this op extends the same head with
        # V/O slots 6+7, sourced via ``_L7_HEAD_LAYOUT_BY_NAME`` so the
        # extension cannot drift from the primary owner.
        head_allocator = _allocate_layer7_heads()
        block.attn._l7_head_allocator = head_allocator

        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn, _layer7_sp_byte0_is_f8_spec(BD), HD,
        )

    _claims = set()
    _claims.add((7, "attn_W_v", f"6_{_SP_BYTE0_F8_V_SLOT_LO}", "EMBED_LO+8"))
    _claims.add((7, "attn_W_v", f"6_{_SP_BYTE0_F8_V_SLOT_HI}", "EMBED_HI+15"))
    _claims.add((7, "attn_W_o", f"6_{_SP_BYTE0_F8_V_SLOT_LO}", "SP_BYTE0_IS_F8+0"))
    _claims.add((7, "attn_W_o", f"6_{_SP_BYTE0_F8_V_SLOT_HI}", "SP_BYTE0_IS_F8+0"))

    return Operation(
        name="layer7_sp_byte0_is_f8",
        reads={"MARK_SP", "EMBED_LO", "EMBED_HI"},
        writes={"SP_BYTE0_IS_F8"},
        kind="block",
        # Phase 8.G.6: drop ``layer_idx=7`` literal; bind to the L7 ffn
        # anchor ``layer8_sp_gather`` so the block op resolves to
        # whichever layer the compiler places the anchor at.
        target_op_name="layer8_sp_gather",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer7_sp_byte0_is_f8_ir,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def _layer7_sp_byte0_is_f8_ir(dim_positions, HD) -> CompilerIR:
    del HD
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer7_sp_byte0_is_f8_spec(BD))
    return ir


def _layer7_sp_byte0_is_f8_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Declarative spec for the SP_BYTE0_IS_F8 extension on L7 head 6.

    The existing head 6 Q/K pattern (MARK_SP & MARK_STACK0 self-attention
    with ALiBi slope 5.0) is preserved; this spec adds only V slots
    6 and 7 (and matching O writes). Q/K are intentionally empty here so
    ``generate_attention_head`` does not overwrite the prior head-6 Q/K
    wiring with zero weights.
    """

    # head_idx sourced from _L7_HEAD_LAYOUT: this op extends head 6,
    # owned by ``layer7_memory_heads`` (PSH/CMP relay). Slots 1..5 are
    # written by the primary owner; we add slots 6+7 here.
    return DeclarativeAttentionHeadSpec(
        head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_6"],
        q=(),
        k=(),
        v=(
            AP(_SP_BYTE0_F8_V_SLOT_LO, BD.EMBED_LO + 8, 1.0),
            AP(_SP_BYTE0_F8_V_SLOT_HI, BD.EMBED_HI + 15, 1.0),
        ),
        o=(
            AO(BD.SP_BYTE0_IS_F8, _SP_BYTE0_F8_V_SLOT_LO, 0.5),
            AO(BD.SP_BYTE0_IS_F8, _SP_BYTE0_F8_V_SLOT_HI, 0.5),
        ),
    )
