"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ...attention_head_allocator import AttentionHeadAllocator
from ...ffn_unit_allocator import FFNUnitAllocator
from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


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
        phase=7,
        reads={"MARK_AX", "STACK0_BYTE0", "OP_LEA", "OP_ADJ", "OP_ENT",
               "CONST",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "OUTPUT_LO", "OUTPUT_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer7_operand_gather_ir,
        declarative_authority="spec_generated",
        layer_idx=7,
        migrated=True,
        claims=_claims,
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # L7 head 0 + head 1 produce the fresh in-step ALU_LO/HI at the AX
        # marker (operand A for binary ops + LEA destination address).
        # L8 ALU and L9 ALU consume these via their AX-marker reads.
        produces={
            "ALU_LO": "AX_byte0",
            "ALU_HI": "AX_byte0",
        },
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


def _layer7_operand_gather_head_specs(BD) -> tuple[DeclarativeAttentionHeadSpec, ...]:
    """Declarative replacement for ``vm_step._set_layer7_operand_gather``."""

    L = 15.0
    AX_I = 1

    return (
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_operand_gather.head_0"],
            q=(
                AP(0, BD.MARK_AX, L),
                AP(0, BD.OP_LEA, -L),
                AP(0, BD.OP_ADJ, -L),
                AP(0, BD.OP_ENT, -L),
                AP(33, BD.MARK_AX, L),
                AP(33, BD.CONST, -L / 2),
                AP(33, BD.OP_LEA, -L * 10),
                AP(33, BD.OP_ADJ, -L * 10),
                AP(33, BD.OP_ENT, -L * 10),
            ),
            k=(AP(0, BD.STACK0_BYTE0, L), AP(33, BD.CONST, L)),
            v=(
                _band_projection_writes(1, BD.CLEAN_EMBED_LO)
                + _band_projection_writes(17, BD.CLEAN_EMBED_HI)
            ),
            o=(
                _band_output_writes(BD.ALU_LO, 1, 6.0)
                + _band_output_writes(BD.ALU_HI, 17, 6.0)
            ),
        ),
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_operand_gather.head_1"],
            q=(
                AP(0, BD.MARK_AX, L * 10),
                AP(0, BD.OP_LEA, L),
                AP(0, BD.OP_ADJ, L),
                AP(0, BD.OP_ENT, L),
                AP(0, BD.CONST, -L * 5),
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
    # Head 7 MEM flag broadcast.
    _claims.add((7, "attn_W_v", "7_1", "MEM_STORE+0"))
    _claims.add((7, "attn_W_v", "7_2", "MEM_ADDR_SRC+0"))
    _claims.add((7, "attn_W_v", "7_3", "OP_JSR+0"))
    _claims.add((7, "attn_W_v", "7_4", "OP_ENT+0"))

    return Operation(
        name="layer7_memory_heads",
        phase=7,
        # Phase 8.A.6 v2: TEMP_PREV_STEP marks the TEMP read as cross-step
        # relative to L11/L14 TEMP writers (which fire after L7 in the same
        # step). The same-step values written by L3 carry_forward / L5
        # opcode_decode are still picked up at the same numeric position.
        # L7 also self-writes TEMP for the NOCARRY_ALU_OP relay (head 5 V
        # slot 9) — that write is independent of the read. Breaks L11/L14 →
        # layer7_memory_heads back-edges on TEMP.
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "OP_LI", "OP_LC", "OP_PSH", "OP_SI", "OP_SC",
               "OP_ADD", "OP_SUB",
               # Head 5 reads OP_AND/OP_OR/OP_XOR for the bitwise byte
               # propagation relays and OP_SHR for the byte-zero cleanup relay.
               "OP_AND", "OP_OR", "OP_XOR", "OP_SHR",
               "OP_JSR",  # head 5 V slot 8 (existing, declared for completeness)
               "AX_CARRY_LO", "AX_CARRY_HI", "TEMP_PREV_STEP"},
        writes={"OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP",
                "TEMP", "ADDR_KEY",
                # V7 Block 13 (2026-05-12): head 5 V slot 9 writes the
                # NOCARRY_ALU_OP relay to TEMP[7]. (TEMP is already in writes
                # but listed here for clarity.) Head 5 also writes the OP_JSR
                # relay back to OP_JSR at AX byte positions (added 2026-05-12).
        "OP_JSR", "OP_SI", "OP_SC"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=_layer7_memory_heads_ir,
        layer_idx=7,
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
    specs.append(
        DeclarativeAttentionHeadSpec(
            head_idx=_L7_HEAD_LAYOUT_BY_NAME["layer7_memory_heads.head_5"],
            q=(AP(0, BD.MARK_AX, L), AP(0, BD.H1 + AX_I, L)),
            k=(AP(0, BD.MARK_AX, L * 2.0),),
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
        phase=7.5,
        reads={"IO_IN_OUTPUT_MODE", "MARK_STACK0", "EMBED_LO", "EMBED_HI"},
        writes={"FORMAT_PTR_LO", "FORMAT_PTR_HI"},
        kind="block",
        declarative_bake_fn=bake,
        compiler_ir_factory=(
            _format_pointer_extraction_ir
            if enable_conversational_io else None
        ),
        declarative_authority="spec_generated",
        layer_idx=7,
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
        phase=7.6,
        reads={"MARK_SP", "EMBED_LO", "EMBED_HI"},
        writes={"SP_BYTE0_IS_F8"},
        kind="block",
        layer_idx=7,
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
