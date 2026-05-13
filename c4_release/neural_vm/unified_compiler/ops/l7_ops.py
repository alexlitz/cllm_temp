"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
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
        # Tier C annotations: L7 operand gather is the attention-only op
        # that stages ALU_LO/HI from STACK0 byte 0 / OUTPUT_LO/HI at the
        # AX marker for the downstream L8/L9 ALU. Exercised by every ALU
        # smoke test (the FFN side reads ALU_*; without this op operand A
        # is zero). ``compaction_safe=True`` is technically vacuous for an
        # attention-only op (no FFN units to partition) but keeps the
        # invariant uniform.
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
    """L7 attention heads 1-6: memory + flag broadcast heads.

    Pinned to ``layer_idx=7``. See ``make_layer7_operand_gather_op``.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer7_memory_heads
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = 5.0  # head 1: MEM flag broadcast
            attn.alibi_slopes[5] = 5.0  # head 5: LI/LC flag relay
            attn.alibi_slopes[6] = 5.0  # head 6: PSH/store flag relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer7_memory_heads(attn, S, _as_setdim_proxy(dim_positions), HD)
        # Softmax-sharpness fix (head 5 — LI/LC/LEA flag relay): the audit
        # (87442ad) flags head 5 with mass=0.9473, s_target=3.586, gap=3.586,
        # slope=5. The single-Q-cell K side (W_k[base, MARK_AX]=L with L=15)
        # gives only 3.6 nats of headroom against softmax1 anchor — close to
        # but not above ln(99) ~= 4.6 for >=99% mass. Audit recommends
        # "bump K-scale ~2.0x (raise s_target); close gap by ~2.0x via Q*K
        # bump". Doubling W_k[base, MARK_AX] doubles s_target and the gap
        # simultaneously (since the runner-up here is the softmax1 anchor),
        # lifting mass above 99%.
        attn.W_k[5 * HD] *= 2.0

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
    )


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
        # Staleness invariants: head 7 fires when IO_IN_OUTPUT_MODE just
        # activated, attends to the previous step's STACK0 marker, and
        # writes EMBED_LO/HI (format string pointer byte 0) into
        # FORMAT_PTR_LO/HI at the firing Q position. Skipped when
        # enable_conversational_io=False (no-op bake).
        produces={
            "FORMAT_PTR_LO": "STACK0",
            "FORMAT_PTR_HI": "STACK0",
        },
    )


