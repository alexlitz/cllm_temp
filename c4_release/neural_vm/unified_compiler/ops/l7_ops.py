"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


def make_layer7_operand_gather_op() -> Operation:
    """L7 attention: operand A gather (prev STACK0 byte 0 → ALU at AX marker).

    Pinned to ``layer_idx=7`` via ``kind="block"``: legacy_bake no longer
    calls ``_set_layer7_operand_gather`` (it was migrated to the compiler),
    so without pinning the dep-graph would silently bake into a different
    block and leave block 7 zero-init.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer7_operand_gather
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer7_operand_gather(attn, S, _as_setdim_proxy(dim_positions), HD)

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
        reads={"MARK_AX", "STACK0_BYTE0", "OP_LEA",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        bake_fn=bake,
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


def make_layer7_memory_heads_op() -> Operation:
    """L7 attention heads 2-7: memory + flag broadcast heads.

    Pinned to ``layer_idx=7``. See ``make_layer7_operand_gather_op``.
    """
    def bake(block, dim_positions, S):
        attn = block.attn
        BD = _as_setdim_proxy(dim_positions)
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
    # Head 5 scalar relays (V slots 1..8 → distinct output dims).
    _claims.add((7, "attn_W_v", "5_1", "OP_LI+0"))
    _claims.add((7, "attn_W_v", "5_2", "OP_LC+0"))
    _claims.add((7, "attn_W_v", "5_3", "OP_LEA+0"))
    _claims.add((7, "attn_W_v", "5_8", "OP_JSR+0"))
    # Head 7 MEM flag broadcast.
    _claims.add((7, "attn_W_v", "7_1", "MEM_STORE+0"))
    _claims.add((7, "attn_W_v", "7_2", "MEM_ADDR_SRC+0"))
    _claims.add((7, "attn_W_v", "7_3", "OP_JSR+0"))
    _claims.add((7, "attn_W_v", "7_4", "OP_ENT+0"))

    return Operation(
        name="layer7_memory_heads",
        phase=7,
        reads={"MARK_MEM", "MARK_AX", "MARK_STACK0",
               "OP_LI", "OP_LC", "OP_PSH", "OP_SI", "OP_SC",
               # V7 Block 13 (2026-05-12): head 5 now reads OP_AND/OP_OR/OP_XOR
               # /OP_SHR for the new V slot 9 (NOCARRY_ALU_OP relay → TEMP[7])
               # used by ``_set_layer14_alu_nocarry_ax_bytes_zero``.
               "OP_AND", "OP_OR", "OP_XOR", "OP_SHR",
               "OP_JSR",  # head 5 V slot 8 (existing, declared for completeness)
               "AX_CARRY_LO", "AX_CARRY_HI", "TEMP"},
        writes={"OP_LI_RELAY", "OP_LC_RELAY", "PSH_AT_SP",
                "TEMP", "ADDR_KEY",
                # V7 Block 13 (2026-05-12): head 5 V slot 9 writes the
                # NOCARRY_ALU_OP relay to TEMP[7]. (TEMP is already in writes
                # but listed here for clarity.) Head 5 also writes the OP_JSR
                # relay back to OP_JSR at AX byte positions (added 2026-05-12).
                "OP_JSR"},
        kind="block",
        bake_fn=bake,
        layer_idx=7,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
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
            head_idx=7,
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
    for j in range(3):
        head = 2 + j
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

    # Head 5: Relay OP_LI/OP_LC/LEA/bitwise/JSR/no-carry flags from AX marker.
    # The K scale is doubled here to preserve the softmax-sharpness fix that
    # previously ran as a post-helper row multiply in ``bake``.
    specs.append(
        DeclarativeAttentionHeadSpec(
            head_idx=5,
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
                AP(9, BD.OP_AND, 0.2),
                AP(9, BD.OP_OR, 0.2),
                AP(9, BD.OP_XOR, 0.2),
                AP(9, BD.OP_SHR, 0.2),
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
            ),
        )
    )

    # Head 6: Relay PSH/ENT/JSR from STACK0 marker and PSH_AT_SP from SP.
    specs.append(
        DeclarativeAttentionHeadSpec(
            head_idx=6,
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
        if not enable_conversational_io:
            return
        from ...vm_step import _set_format_pointer_extraction
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[7] = 5.0  # steep to attend back to prev step
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_format_pointer_extraction(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="format_pointer_extraction",
        phase=7.5,
        reads={"IO_IN_OUTPUT_MODE", "MARK_STACK0", "EMBED_LO", "EMBED_HI"},
        writes={"FORMAT_PTR_LO", "FORMAT_PTR_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=7,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )

