"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer8_alu_op() -> Operation:
    """L8 FFN: ADD/SUB lo nibble + carry/borrow + LEA + CMP_GROUP.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.2, migrated=True. The inline
    call ``_set_layer8_alu(ffn8, S, BD)`` in ``set_vm_weights`` (both
    ``alu_mode == 'lookup'`` and ``alu_mode == 'efficient'`` branches)
    has been removed; this op now owns the bake. Phase=8.2 places it
    after format_pointer_extraction (7.5) and the L8 multibyte_fetch
    bake (8.1), and before format_position_counter (8.5) — matching the
    legacy in-set_vm_weights ordering.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_alu
        _set_layer8_alu(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_alu",
        phase=8.2,
        reads={"MARK_AX", "MARK_PC", "ALU_LO", "AX_CARRY_LO", "FETCH_LO",
               "OP_ADD", "OP_SUB", "OP_LEA",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE"},
        writes={"OUTPUT_LO", "CARRY", "CMP_GROUP"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_format_position_counter_op(enable_conversational_io: bool = False) -> Operation:
    """L8 FFN: increment IO_FORMAT_POS after each output byte emission.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_position_counter(ffn8, S, BD)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=8`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was a side-effect of convo-io initially being co-located with
    the lookup ALU bake — the helper itself writes IO_FORMAT_POS units
    (starting at unit 600) and has no dependency on lookup-mode-specific
    weights, so this op fires regardless of alu_mode whenever the flag
    is set. Phase=8.5 so this runs AFTER ``layer8_alu`` (phase=8) since
    the position-counter units (600-615) intentionally overwrite a slice
    of the ADD-carry block, matching legacy behavior.
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        from ...vm_step import _set_format_position_counter
        _set_format_position_counter(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="format_position_counter",
        phase=8.5,
        reads={"LAST_WAS_BYTE", "IO_IN_OUTPUT_MODE", "IO_FORMAT_POS"},
        writes={"IO_FORMAT_POS"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_multibyte_fetch_op() -> Operation:
    """No-op dep anchor for ``layer8_multibyte_fetch_bake``.

    Kept as a ``kind="attn"`` placeholder (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology that places downstream ops at the right
    model.blocks indices. The actual weight bake now happens in
    ``make_layer8_multibyte_fetch_bake_op`` (kind="block", layer_idx=8,
    phase=8.1, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_multibyte_fetch_bake` block op.
        return

    return Operation(
        name="layer8_multibyte_fetch",
        phase=8,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_multibyte_fetch_bake_op() -> Operation:
    """Bake ``_set_layer8_multibyte_fetch`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_multibyte_fetch(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.1 so this runs after ``layer8_sp_gather_bake`` (phase=8.0),
    preserving the legacy in-set_vm_weights ordering. The dep anchor
    ``layer8_multibyte_fetch`` (kind="attn") preserves the LayerCompiler
    topology so downstream ops remain placed at their legacy blocks.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_multibyte_fetch
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_multibyte_fetch(attn, S, proxy, HD)

    return Operation(
        name="layer8_multibyte_fetch_bake",
        phase=8.1,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_multibyte_routing_op() -> Operation:
    """L8 FFN extension: route FETCH → OUTPUT at AX byte positions for IMM.

    MIGRATED 2026-05-10 (Wave 2 Unit 10): flipped from kind="ffn" to
    kind="block" with layer_idx=8, phase=8.3, migrated=True. The inline
    call ``_set_layer8_multibyte_routing(ffn8, S, BD)`` in
    ``set_vm_weights`` (both alu_mode branches) has been removed; this op
    now owns the bake. Phase=8.3 places it after ``layer8_alu`` (8.2)
    so the shared unit counter starts after the ALU units (the helper
    internally re-invokes ``_set_layer8_alu`` to compute ``unit_start``;
    that re-call is an idempotent overwrite of the same ALU weights).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_multibyte_routing
        _set_layer8_multibyte_routing(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer8_multibyte_routing",
        phase=8.3,
        reads={"IS_BYTE", "H1", "OP_IMM", "MARK_AX",
               "AX_CARRY_LO", "AX_CARRY_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_sp_gather_op() -> Operation:
    """No-op dep anchor for ``layer8_sp_gather_bake``.

    Kept as a ``kind="attn"`` placeholder (no ``migrated`` flag, no
    ``layer_idx``): its declared reads/writes preserve the LayerCompiler
    dep-graph topology. The actual weight bake now happens in
    ``make_layer8_sp_gather_bake_op`` (kind="block", layer_idx=8,
    phase=8.0, migrated=True); this op's bake_fn is a no-op.
    """
    def bake(attn, dim_positions, S):
        # No-op: actual bake is in `layer8_sp_gather_bake` block op.
        return

    return Operation(
        name="layer8_sp_gather",
        phase=8,
        reads={"MARK_AX", "MARK_SP", "OP_ADJ", "OP_ENT", "OP_LEA",
               "EMBED_LO", "EMBED_HI"},
        writes={"ALU_LO", "ALU_HI"},
        kind="attn",
        bake_fn=bake,
    )


def make_layer8_sp_gather_bake_op() -> Operation:
    """Bake ``_set_layer8_sp_gather`` into ``model.blocks[8].attn``.

    Originally an inline call in ``set_vm_weights``:
        ``_set_layer8_sp_gather(attn8, S, BD, HD)``

    Migrated as ``kind="block"`` with ``layer_idx=8`` and ``migrated=True``:
    the inline call has been removed, so the bake must happen here.
    Phase=8.0 so this runs before ``layer8_multibyte_fetch_bake``
    (phase=8.1), preserving the legacy in-set_vm_weights ordering. The
    dep anchor ``layer8_sp_gather`` (kind="attn") preserves the
    LayerCompiler topology so downstream ops remain placed at their
    legacy blocks.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_sp_gather
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_sp_gather(attn, S, proxy, HD)

    return Operation(
        name="layer8_sp_gather_bake",
        phase=8.0,
        reads={"MARK_STACK0", "MARK_BP", "H1", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI",
                "ADDR_B1_LO", "ADDR_B1_HI",
                "ADDR_B2_LO", "ADDR_B2_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


def make_layer8_op_imm_relay_op() -> Operation:
    """L8 head 4: Relay OP_IMM from AX marker to AX byte positions.

    Migrated 2026-05-11 from the inline block at vm_step.py:2029-2043 that
    programmed attn8 head 4's Q/K/V/O slots for OP_IMM relay (with a GATE4
    sub-head). At AX byte positions (IS_BYTE + H1[AX_I]), this head attends
    to AX marker (MARK_AX) and copies OP_IMM to byte positions.

    Phase=8.4 places it AFTER ``layer8_multibyte_routing`` (8.3) but BEFORE
    the L8 alu_postop_attach (phase 8.5) so attention bakes complete before
    the FFN wrap.
    """
    def _bake(block, dim_positions, S):
        BD = _as_setdim_proxy(dim_positions)
        attn8 = block.attn
        HD = attn8.W_q.shape[0] // attn8.num_heads
        base = 4 * HD
        AX_I = 1
        L8_relay = 20.0
        attn8.W_q[base, BD.IS_BYTE] = L8_relay
        attn8.W_q[base, BD.H1 + AX_I] = L8_relay
        attn8.W_q[base, BD.CONST] = -L8_relay * 1.5
        attn8.W_k[base, BD.MARK_AX] = L8_relay
        attn8.W_k[base, BD.IS_BYTE] = -L8_relay * 10
        attn8.W_k[base, BD.CONST] = L8_relay * 0.5
        attn8.W_v[base, BD.OP_IMM] = 1.0
        attn8.W_o[BD.OP_IMM, base] = 1.0
        GATE4 = 1
        attn8.W_q[base + GATE4, BD.IS_BYTE] = 500.0
        attn8.W_q[base + GATE4, BD.CONST] = -500.0
        attn8.W_k[base + GATE4, BD.CONST] = 5.0

    return Operation(
        name="layer8_op_imm_relay",
        reads={"IS_BYTE", "H1", "MARK_AX", "OP_IMM", "CONST"},
        writes={"OP_IMM"},
        kind="block",
        bake_fn=_bake,
        layer_idx=8,
        phase=8.4,
        migrated=True,
    )


def make_layer8_mem_to_alu_op(enable: bool = False) -> Operation:
    """L8 attention head 5: mem-attention reading mem[SP] → ALU_LO/HI at AX.

    Phase 1 of STACK0_VIA_MEM_ATTENTION_PLAN. Replaces L7 head 0's
    STACK0_BYTE0-keyed read with a direct ``mem[SP]`` lookup keyed by
    ADDR_KEY at the AX marker. The ADDR_KEY Q-side is staged by
    ``make_layer4_sp_to_addr_key_op`` (L4 attn heads 2-3) which writes
    the live SP value as nibble one-hots into the ADDR_KEY band at the
    AX marker. The K-side is the ADDR_KEY band at MEM val byte positions
    populated by ``_inject_mem_metadata`` (no embedding-side changes
    needed).

    Placement at L8 attn (not L9) is required because the L8 ALU FFN —
    in efficient mode, the ``AddSub5StageBlock`` post-op — reads ALU_LO
    and ALU_HI at the AX marker. The head must therefore write
    ALU_LO/HI BEFORE the L8 FFN/post-op runs. Writing at L9 attn would
    be too late.

    Design:
      - Q at AX marker, gated on POP-group / OP_LI_RELAY / OP_LC_RELAY
        so the head fires for binary ops + loads but not for IMM / NOP.
      - K at MEM val byte 0, gated on MEM_STORE + MEM_VAL_B0 so only
        store entries match. The most-recent matching store wins via
        ALiBi recency (slope tuned identically to the L9 ALiBi head:
        0.5 = 17.5 score margin per VM step).
      - Address matching uses the 12-bit (3-nibble) binary encoding
        identical to L15 head 0 and the L9 ALiBi proof-of-concept.
        Q-side reads the ADDR_KEY band (ADDR_B0/1/2_HI) staged by L4;
        K-side reads the same band populated by ``_inject_mem_metadata``.
      - V/O copy ``CLEAN_EMBED_LO/HI`` from the matching MEM val byte
        into ``ALU_LO/HI`` at the AX marker — exactly the source/dest
        L7 head 0 uses today.

    Score budget (per dim, after /sqrt(HD)=8):
      Dim 0 (Q gate):     -50 at non-fire / +50 at fire
      Dim 1 (K MEM_STORE): +312.5 at target+store, -312.5 at target+
                            non-store
      Dim 2 (K addr-anchor): -600 at store entries (ZFOD baseline)
      Dim 3 (byte select): +450 at MEM val byte 0
      Dims 4-27 (addr):    +300 at exact 12-bit match
      ALiBi (slope 0.5):   -17.5 per VM step distance

    Net at correct match (1 step back): +12.5 - 17.5 ≈ -5 → still attends
    via softmax1 (the only positive contributor beats the zero anchor).
    Net at wrong-addr store at same distance: -287.5 - 17.5 → suppressed.

    Disabled by default (``enable=False``). Flip together with
    ``make_layer4_sp_to_addr_key_op``.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return

        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        head = 5
        base = head * HD

        # Slope tuned to favor most-recent matching MEM_STORE.
        # 1 VM step = 35 tokens; slope 0.5 → 17.5 score margin per step.
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        L = 50.0

        # === Dim 0: bias — fire only at AX marker for POP/LI/LC ===
        # POP-group flag (CMP[3]) is relayed to AX marker by the L6 opcode
        # relay head (which fires at AX/SP/STACK0). OP_LI_RELAY/OP_LC_RELAY
        # are at AX byte positions (not the marker), so for binary ops we
        # rely primarily on CMP+3 + MARK_AX. The non-fire baseline scores
        # to -2000 so non-target positions cannot win even with worst-case
        # address aliasing.
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.MARK_AX] = 2000.0     # require AX marker
        # Either POP-group (binary op) OR OP_LI/LC active at AX marker.
        # CMP+3 may take a step to propagate; OP_LI/OP_LC are set directly
        # by L5 FFN at AX marker.
        attn.W_q[base, BD.CMP + 3] = 500.0      # POP group multiplier
        attn.W_q[base, BD.OP_LI] = 500.0
        attn.W_q[base, BD.OP_LC] = 500.0
        # Suppress at PC/SP/BP/STACK0/MEM markers — at these positions the
        # ADDR_KEY band carries other information (code addresses, mem
        # addresses) that would alias into this head's address match.
        attn.W_q[base, BD.MARK_PC] = -2000.0
        attn.W_q[base, BD.MARK_SP] = -2000.0
        attn.W_q[base, BD.MARK_BP] = -2000.0
        attn.W_q[base, BD.MARK_MEM] = -2000.0
        attn.W_q[base, BD.MARK_STACK0] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0

        # === Dim 1: store anchor ===
        attn.W_q[base + 1, BD.MARK_AX] = 50.0
        attn.W_k[base + 1, BD.MEM_STORE] = 100.0
        attn.W_k[base + 1, BD.CONST] = -50.0

        # === Dim 2: ZFOD baseline ===
        attn.W_q[base + 2, BD.CONST] = -96.0
        attn.W_k[base + 2, BD.MEM_STORE] = 50.0

        # === Dim 3: byte 0 selection (MEM val byte 0) ===
        BS = 60.0
        attn.W_q[base + 3, BD.MARK_AX] = BS
        # MEM val byte 0 is at d=5 from MEM marker: L2H0[MEM]=1, H1[MEM]=0
        MEM_I = 4
        attn.W_k[base + 3, BD.L2H0 + MEM_I] = BS
        attn.W_k[base + 3, BD.H1 + MEM_I] = -BS

        # === Dims 4-27: 24-bit binary address encoding ===
        # Same encoding as L15 head 0 / L9 ALiBi head: iterate over both
        # _LO and _HI bases per address byte. Q and K read from the same
        # residual dims because the L4 SP gather writes into the same
        # ADDR_B*_HI bands that `_inject_mem_metadata` writes K-side into.
        # (Q-side ADDR_B*_LO bands carry zero contribution because the
        # SP-gather only writes HI bands — see make_layer4_sp_to_addr_key_op.)
        addr_dim = 4
        scale = 10.0
        addr_bases = [
            (BD.ADDR_B0_LO, BD.ADDR_B0_HI),
            (BD.ADDR_B1_LO, BD.ADDR_B1_HI),
            (BD.ADDR_B2_LO, BD.ADDR_B2_HI),
        ]
        for ab_lo, ab_hi in addr_bases:
            for nibble_base in [ab_lo, ab_hi]:
                for bit in range(4):
                    for k in range(16):
                        bit_val = 2 * ((k >> bit) & 1) - 1
                        attn.W_q[base + addr_dim, nibble_base + k] = scale * bit_val
                        attn.W_k[base + addr_dim, nibble_base + k] = scale * bit_val
                    addr_dim += 1

        # === V/O: copy CLEAN_EMBED bytes → ALU_LO/HI at AX marker ===
        # This mirrors L7 head 0 (vm_step.py:_set_layer7_operand_gather)
        # which writes ALU_LO/HI from STACK0_BYTE0's CLEAN_EMBED. The L8
        # FFN (lookup ALU or AddSub5StageBlock in efficient mode) consumes
        # ALU_LO/HI as binary-op operand 2.
        SCALE_O = 6.0  # match L7 head 0 amplification (overcomes L4 ALU clear)
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
        for k in range(16):
            attn.W_o[BD.ALU_LO + k, base + 1 + k] = SCALE_O
            attn.W_o[BD.ALU_HI + k, base + 17 + k] = SCALE_O

    return Operation(
        name="layer8_mem_to_alu",
        # Phase 8.45 places this after layer8_op_imm_relay (8.4) and BEFORE
        # the L8 alu_postop_attach (8.5), keeping all L8 attn bakes in
        # phase order.
        phase=8.45,
        reads={"MARK_AX", "CMP", "OP_LI", "OP_LC", "MEM_STORE", "L2H0",
               "H1", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO",
               "ADDR_B1_HI", "ADDR_B2_LO", "ADDR_B2_HI",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
    )


