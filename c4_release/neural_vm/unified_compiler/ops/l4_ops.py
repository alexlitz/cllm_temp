"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer4_pc_relay_op() -> Operation:
    """L4 attention: relay PC marker EMBED → AX marker EMBED.

    Pinned to ``layer_idx=4`` via ``kind="block"`` because the legacy
    ``set_vm_weights`` pipeline targets block 4. Without pinning, the
    dep-graph layer assignment places this op at a later block (e.g. L5/L6),
    leaving block 4's attn zero-init and breaking the L5 fetch chain
    (the regression at commit b2d9f4c3).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer4_pc_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer4_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    # Dim-ownership claims: L4 attn heads 0 + 1 PC relay.
    #   Head 0: V slots 1..32 read EMBED_LO/HI → EMBED_LO/HI at AX marker.
    #   Head 1: V slots 1..32 read EMBED_LO/HI → TEMP[0..31] at AX byte pos.
    _claims = set()
    for k in range(16):
        _claims.add((4, "attn_W_v", f"0_{1 + k}", f"EMBED_LO+{k}"))
        _claims.add((4, "attn_W_v", f"0_{17 + k}", f"EMBED_HI+{k}"))
        _claims.add((4, "attn_W_v", f"1_{1 + k}", f"EMBED_LO+{k}"))
        _claims.add((4, "attn_W_v", f"1_{17 + k}", f"EMBED_HI+{k}"))

    return Operation(
        name="layer4_pc_relay",
        phase=4,
        reads={"MARK_PC", "MARK_AX", "EMBED_LO", "EMBED_HI", "CONST"},
        writes={"EMBED_LO", "EMBED_HI"},  # at AX marker
        kind="block",
        bake_fn=bake,
        layer_idx=4,
        migrated=True,
        claims=_claims,
    )


def make_layer4_ffn_op() -> Operation:
    """L4 FFN: compute PC+1/2/3/4 in FETCH dims for L5 fetch.

    Pinned to ``layer_idx=4`` via ``kind="block"``; see
    ``make_layer4_pc_relay_op``.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer4_ffn
        _set_layer4_ffn(block.ffn, S, _as_setdim_proxy(dim_positions))

    return Operation(
        name="layer4_ffn",
        phase=4,
        reads={"MARK_AX", "MARK_PC", "EMBED_LO", "EMBED_HI",
               "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
               "H1"},
        writes={"FETCH_LO", "FETCH_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=4,
        migrated=True,
        # ``_set_layer4_ffn`` writes the PC+1 (lo/hi/carry = 64 units) +
        # TEMP-clear (32) + multi-byte PC+2/+3/+4 (96+128+160 = 384) +
        # PC+1@PC marker (64) chains for a total of 544 units (0..543).
        # See bake body in vm_step.py:_set_layer4_ffn.
        ffn_units_used=544,
    )


def make_layer4_sp_to_addr_key_op(enable: bool = False) -> Operation:
    """L4 attention heads 2 + 3: gather SP value → ADDR_KEY at AX marker.

    Phase 1 of STACK0_VIA_MEM_ATTENTION_PLAN — Q-side staging for the
    mem-attention path that replaces L7 head 0's STACK0_BYTE0 read with a
    direct ``mem[SP]`` lookup. The mem-attention K-side (in
    ``_inject_mem_metadata``) already writes ``ADDR_KEY[lo, 16+hi, 32+top]``
    at MEM val byte positions; this op produces the matching encoding at
    the AX marker so a downstream attention head (at L8 attn — see
    ``make_layer8_mem_to_alu_op``) can match on address.

    Encoding (matches ``_inject_mem_metadata`` exactly):
      - Head 2 fires at AX marker, attends to SP byte 0 position. Copies
        CLEAN_EMBED_LO → ``ADDR_KEY[0..15]`` (lo nibble of SP byte 0) and
        CLEAN_EMBED_HI → ``ADDR_KEY[16..31]`` (hi nibble of SP byte 0).
      - Head 3 fires at AX marker, attends to SP byte 1 position. Copies
        CLEAN_EMBED_LO → ``ADDR_KEY[32..47]`` (lo nibble of SP byte 1,
        i.e. the "top" 4 bits of the 12-bit ADDR_KEY space).

    The two heads write to non-overlapping ADDR_KEY sub-bands. Since
    ADDR_B0_HI/ADDR_B1_HI/ADDR_B2_HI alias the ADDR_KEY band (dims
    206/222/238), the writes use those aliases for clarity.

    Gating: AX marker AND ``HAS_SE = 1`` (step 1+). Step 0 has no prior
    MEM section to read from, and the default STACK_INIT SP would
    otherwise be injected before any PSH has populated memory.

    Disabled by default (``enable=False``); the bake is a guard-clause
    no-op. Existing tests stay byte-identical until both this op and the
    L8 mem-to-ALU head are flipped on together.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return

        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        L = 50.0  # strong gate; overpowers any incidental L7 contamination
        SP_I = 2  # SP marker index in MARKS array
        # SCALE > 1 so the SP signal dominates any incidental writes by L7
        # heads 2-4 (which write prev-AX nibbles into the same ADDR_B*_HI
        # bands for LI/LC).
        SCALE_O = 10.0

        def _stage_sp_byte(head_idx, byte_idx_dim, write_lo_to, write_hi_to):
            """Stage one SP byte into the ADDR_KEY band at the AX marker.

            Each head fires at the AX marker (only on step 1+, gated by
            HAS_SE) and attends to a single SP byte position, copying the
            CLEAN_EMBED nibbles into the requested ADDR_B*_HI sub-bands.

            ``write_hi_to`` may be ``None`` for heads that only carry the
            lo nibble (e.g. SP byte 1's hi nibble extends past the 12-bit
            ADDR_KEY space and is intentionally dropped).
            """
            base = head_idx * HD
            # Q: AX marker AND HAS_SE = 1 (step 1+). Step 0 has no prior
            # MEM section to read from, and the default STACK_INIT SP
            # would corrupt downstream ADDR_KEY readers if injected before
            # any PSH has populated memory.
            attn.W_q[base, BD.MARK_AX] = L
            attn.W_q[base, BD.HAS_SE] = L
            attn.W_q[base, BD.CONST] = -L * 1.5
            # K: target SP byte position. BYTE_INDEX_n fires at byte n of
            # every register; H1[SP_I] localises to the SP byte area
            # (d=1..4 from the SP marker).
            attn.W_k[base, byte_idx_dim] = L
            attn.W_k[base, BD.H1 + SP_I] = L
            attn.W_k[base, BD.CONST] = -L
            # Anti-leakage gate dim
            attn.W_q[base + 33, BD.MARK_AX] = L
            attn.W_q[base + 33, BD.CONST] = -L / 2
            attn.W_k[base + 33, BD.CONST] = L
            # V: copy CLEAN_EMBED nibbles (LO always; HI only if requested)
            for k in range(16):
                attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0
                if write_hi_to is not None:
                    attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
            # O: write into the requested ADDR_KEY sub-band(s)
            for k in range(16):
                attn.W_o[write_lo_to + k, base + 1 + k] = SCALE_O
                if write_hi_to is not None:
                    attn.W_o[write_hi_to + k, base + 17 + k] = SCALE_O

        # Head 2: SP byte 0 → ADDR_KEY[0..15] (lo nibble) + [16..31] (hi nibble).
        _stage_sp_byte(2, BD.BYTE_INDEX_0, BD.ADDR_B0_HI, BD.ADDR_B1_HI)
        # Head 3: SP byte 1 → ADDR_KEY[32..47] (lo nibble only). The hi
        # nibble of SP byte 1 would extend ADDR_KEY past 48 dims; matches
        # the 12-bit "top" convention used by `_inject_mem_metadata`.
        _stage_sp_byte(3, BD.BYTE_INDEX_1, BD.ADDR_B2_HI, None)

    # Dim-ownership claims: L4 attn heads 2 + 3 SP-to-ADDR_KEY staging.
    # Each head writes V slots 1..32 + O writes into ADDR_B*_HI sub-bands.
    #   Head 2: SP byte 0 → ADDR_B0_HI (lo) + ADDR_B1_HI (hi).
    #     W_v[2*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #     W_v[2*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   Head 3: SP byte 1 → ADDR_B2_HI (lo only).
    #     W_v[3*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    # Only meaningful when enable=True; we declare claims unconditionally so
    # the registry can catch latent collisions once the op is enabled.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((4, "attn_W_v", f"2_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((4, "attn_W_v", f"2_{17 + k}", f"CLEAN_EMBED_HI+{k}"))
            _claims.add((4, "attn_W_v", f"3_{1 + k}", f"CLEAN_EMBED_LO+{k}"))

    return Operation(
        name="layer4_sp_to_addr_key",
        phase=4.5,  # after layer4_pc_relay (phase=4) so its writes don't clobber
        reads={"MARK_AX", "BYTE_INDEX_0", "BYTE_INDEX_1", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ADDR_B0_HI", "ADDR_B1_HI", "ADDR_B2_HI"},  # = ADDR_KEY band
        kind="block",
        bake_fn=bake,
        layer_idx=4,
        migrated=True,
        claims=_claims,
    )
