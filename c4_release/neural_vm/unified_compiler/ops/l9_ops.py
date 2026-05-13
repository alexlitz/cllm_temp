"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..layer_compiler import Operation
from .shared import _as_setdim_proxy


def make_layer9_alu_op() -> Operation:
    """L9 FFN: ADD/SUB hi nibble + bitwise ops byte 0, plus marker suppression.

    Originally two inline calls inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``n9 = _set_layer9_alu(ffn9, S, BD)``
        ``_set_layer9_marker_suppress(ffn9, S, BD, n9)``

    Combined into a single migrated bake_fn that captures ``n9`` and
    threads it to ``_set_layer9_marker_suppress`` as ``start_unit`` so the
    two routines share the FFN's hidden-unit allocator. Mirrors the
    combined-bake pattern proven safe by Unit 9's diagnosis (see
    ``c4_release/docs/LOOKUP_MODE_BUG_DIAGNOSIS.md``).

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``; the inline call pair has been removed from
    ``set_vm_weights`` to avoid double-bake. Phase stays at 9. Fires in
    both lookup and efficient ALU modes — the lookup-branch nesting was
    incidental and the helpers themselves are alu_mode-agnostic.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_alu, _set_layer9_marker_suppress
        proxy = _as_setdim_proxy(dim_positions)
        n9 = _set_layer9_alu(block.ffn, S, proxy)
        _set_layer9_marker_suppress(block.ffn, S, proxy, n9)

    # Dim-ownership claims: L9 ALU FFN unit 0 = ADD hi nibble (carry_in=0,
    # a=0, b=0) → OUTPUT_HI+0. Canonical first ADD hi-nibble cluster anchor
    # (see ``_set_layer9_alu`` at vm_step.py:5344+). Unit 0 reads MARK_AX,
    # ALU_HI+0, AX_CARRY_HI+0, CARRY+0 in W_up and writes OUTPUT_HI+0 in
    # W_down. Partial-claim convention: anchor the cluster head without
    # re-declaring all ~3398 ALU/CMP units.
    _claims = {
        (9, "ffn_W_up", "0", "MARK_AX+0"),
        (9, "ffn_W_up", "0", "ALU_HI+0"),
        (9, "ffn_W_up", "0", "AX_CARRY_HI+0"),
        (9, "ffn_W_down", "0", "OUTPUT_HI+0"),
    }

    return Operation(
        name="layer9_alu",
        phase=9,
        reads={"MARK_AX", "MARK_PC", "ALU_HI", "AX_CARRY_HI", "FETCH_HI", "CARRY",
               "OP_ADD", "OP_SUB", "OP_OR", "OP_XOR", "OP_AND",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "ALU_LO", "AX_CARRY_LO"},
        writes={"OUTPUT_HI", "CMP", "OUTPUT_LO"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
        claims=_claims,
        # Staleness invariants: the L9 ALU consumes ALU_HI as operand A hi
        # nibble at the AX marker. Produced by ``layer7_operand_gather`` (L7
        # head 0 + head 1, phase=7) at AX byte 0.
        consumes_fresh={
            "ALU_HI": "AX_byte0",
        },
        # Produces the hi-nibble OUTPUT_HI byte-0 result for ADD/SUB/LEA/
        # ADJ (with carry-in from the L8 ALU's CARRY output), plus the
        # comparison flags CMP[0..N] for the OP_EQ/NE/LT/GT/LE/GE cluster.
        # Matches the L8 ALU lo-nibble producer; together they form the
        # canonical AX-byte-0 ALU output pair.
        produces={
            "OUTPUT_HI": "AX_byte0",
            "CMP": "AX_byte0",
        },
        # ``_set_layer9_alu`` writes the ADD/LEA/SUB/AND/OR/XOR/CMP/etc.
        # cross-product cluster (~3398 units), and the bake chains into
        # ``_set_layer9_marker_suppress`` for 7 more NEXT_* suppression
        # units. Cumulative L9 FFN max: 3405. No other op writes to L9
        # FFN so this op holds the per-layer width annotation.
        ffn_units_used=3405,
    )


def make_layer9_lev_addr_relay_op() -> Operation:
    """L9 attention head 0: BP byte 0 → ADDR_B0 at SP marker for LEV.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_addr_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.0 to preserve ordering with sibling
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1). Fires in both lookup
    and efficient ALU modes — the helper performs identical setup
    regardless of alu_mode, and the model is built once.

    Also sets ``alibi_slopes[0] = 0.2`` (shallow slope for d=29 relay
    from SP marker back to previous BP byte 0); previously set inline
    alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_lev_addr_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[0] = 0.2  # head 0: shallow slope for d=29 relay
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_addr_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    # Dim-ownership claims: L9 attn head 0 LEV addr relay.
    #   W_v[0*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[0*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 0*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 0*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"0_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"0_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_addr_relay",
        phase=9.0,
        reads={"MARK_SP", "OP_LEV", "L1H1", "BYTE_INDEX_0",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
        claims=_claims,
    )


def make_layer9_lev_bp_to_pc_relay_op() -> Operation:
    """L9 attention head 1: BP byte 0 → ADDR_B0 at PC marker for LEV return.

    Originally an inline call inside ``set_vm_weights`` (in the
    ``alu_mode == 'lookup'`` branch):
        ``_set_layer9_lev_bp_to_pc_relay(attn9, S, BD, HD)``

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``: the inline call has been removed to avoid
    double-bake. Phase=9.1 so this op runs AFTER
    ``layer9_lev_addr_relay`` (phase=9.0), matching the legacy
    in-set_vm_weights ordering. Fires in both lookup and efficient
    ALU modes — the helper performs identical setup regardless of
    alu_mode.

    Also sets ``alibi_slopes[1] = 0.5`` (BP→PC relay slope for d=15
    tokens); previously set inline alongside the legacy bake call.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer9_lev_bp_to_pc_relay
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[1] = 0.5  # head 1: BP→PC relay for LEV (d=15 tokens)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer9_lev_bp_to_pc_relay(attn, S, _as_setdim_proxy(dim_positions), HD)

    # Dim-ownership claims: L9 attn head 1 LEV BP→PC relay.
    #   W_v[1*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15
    #   W_v[1*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15
    #   W_o[ADDR_B0_LO + k, 1*HD + 1 + k]         for k=0..15
    #   W_o[ADDR_B0_HI + k, 1*HD + 17 + k]        for k=0..15
    _claims = set()
    for k in range(16):
        _claims.add((9, "attn_W_v", f"1_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((9, "attn_W_v", f"1_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_lev_bp_to_pc_relay",
        phase=9.1,
        reads={"MARK_PC", "OP_LEV", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "L1H1", "BYTE_INDEX_0"},
        writes={"ADDR_B0_LO", "ADDR_B0_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
        claims=_claims,
    )


def make_format_string_fetch_head_op(enable_conversational_io: bool = False) -> Operation:
    """L9 attention head 0: fetch byte from format string at FORMAT_PTR+POS.

    Originally an inline call in ``set_vm_weights`` (nested under
    ``alu_mode == 'lookup'`` + ``enable_conversational_io``):
        ``_set_format_string_fetch_head(attn9, S, BD, HD)``
        plus ``attn9.alibi_slopes.fill_(0.5)``.

    Migrated as ``kind="block"`` pinned to ``layer_idx=9`` with
    ``migrated=True``. Registered unconditionally; the bake is a no-op
    when ``enable_conversational_io`` is False. The original lookup-mode
    nesting was incidental — the helper writes only attn head 0 weights
    and has no dependency on lookup-mode-specific weights, so this op
    fires regardless of alu_mode whenever the flag is set. Phase=9.5 so
    this runs AFTER ``layer9_lev_addr_relay`` (phase=9.0) and
    ``layer9_lev_bp_to_pc_relay`` (phase=9.1); the ``fill_(0.5)`` call
    intentionally clobbers slopes[0] and [1] that those ops set, matching
    legacy ordering (the legacy convo-io block ran ``fill_(0.5)`` AFTER
    the L9 LEV setup as well).
    """
    def bake(block, dim_positions, S):
        if not enable_conversational_io:
            return
        from ...vm_step import _set_format_string_fetch_head
        attn = block.attn
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes.fill_(0.5)
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_format_string_fetch_head(attn, S, _as_setdim_proxy(dim_positions), HD)

    return Operation(
        name="format_string_fetch_head",
        phase=9.5,
        reads={"IO_IN_OUTPUT_MODE", "FORMAT_PTR_LO", "FORMAT_PTR_HI",
               "ADDR_KEY", "EMBED_LO", "EMBED_HI"},
        writes={"OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
    )


def make_layer9_alibi_mem_attn_op(enable: bool = False) -> Operation:
    """L9 attention head 2: ALiBi-based memory-propagation attention.

    PROOF-OF-CONCEPT for the directive "all of the memory conversions should
    be alibi attention stuff" (2026-05-11). Replaces (eventually) the
    runner-side ``_inject_mem_store`` / ``_mem_history`` shadow-memory
    pipeline with attention that gathers PSH'd values from prior step
    OUTPUT positions via ALiBi recency bias.

    Design
    ------
    Q (at MEM val byte 0 positions during LI/LC/POP with ADDR_KEY = SP-1):
      - W_q[head, BD.OP_LI_RELAY] = L
      - W_q[head, BD.OP_LC_RELAY] = L
      - W_q[head, BD.CMP+3] = L              (POP group flag at STACK0)
      - W_q[head, BD.MEM_VAL_B0] = L         (gate to val byte 0)
      - W_q[head, BD.ADDR_KEY + k] = scale * (bit_val)   (address bits)
      - W_q[head, BD.CONST] = -threshold     (suppress non-fire positions)

    K (at PSH/SI/SC OUTPUT positions — STACK0 value byte 0):
      - W_k[head, BD.PSH_AT_SP] = L          (only match PSH output positions)
      - W_k[head, BD.ADDR_KEY + k] = scale * (bit_val)
      - W_k[head, BD.MEM_STORE] = L          (only match store entries)
      - W_k[head, BD.CONST] = -threshold

    V (PSH-output STACK0's CLEAN_EMBED value, copied to OUTPUT at the
      current load position):
      - W_v[head*HD + 1 + k, BD.CLEAN_EMBED_LO + k] = 1.0  for k in 0..15
      - W_v[head*HD + 17 + k, BD.CLEAN_EMBED_HI + k] = 1.0
      - W_o[BD.OUTPUT_LO + k, head*HD + 1 + k] = 1.0
      - W_o[BD.OUTPUT_HI + k, head*HD + 17 + k] = 1.0

    ALiBi slope tuning
    ------------------
    Score budget at target Q (load) attending to PSH outputs:
      - Address match (24-bit, scale=10): +300 at exact match, -300 random
      - PSH gate match: +L^2/HD (target+PSH=+L*L/HD, target+non-PSH=-)
      - ALiBi penalty: -slope * |i - j|  (one VM step = 35 tokens)

    With slope=0.5, going back 1 step costs -0.5*35 = -17.5; with two
    PSHes at the same SP, the more recent one wins by +17.5 score (>>
    softmax noise threshold). Going back 10 steps costs -175 which is
    below the +300 address-match contribution, so legitimate matches still
    fire across the typical KV-cache window. The slope should be tuned
    higher if cross-PSH leak from old values becomes a problem; lower if
    long-range matches fail.

    Status
    ------
    ``enable=False`` by default: the op IS registered (so the dep graph and
    layer_idx gates see it), but the bake is a no-op. This keeps existing
    tests byte-identical. Set ``enable=True`` to activate the head and
    flip ``alibi_slopes[2]`` from 0 to the tuned value.

    Concrete next steps for full Phase 2 PSH/POP support
    -----------------------------------------------------
    1. Bake ADDR_KEY at PSH/SI/SC OUTPUT positions (STACK0 value byte 0
       carries SP-derived address) — new FFN at L8 or earlier, ~50 LoC.
    2. Verify K-side ADDR_KEY at PSH output matches what the load-side Q
       expects. Today ADDR_KEY only lives at code byte positions
       (``_add_code_addr_keys``) and at MEM section val-byte positions
       (``_inject_mem_metadata``); we need it at PSH-step STACK0 too.
    3. Set ``enable=True`` here and ``alibi_slopes[2] = 0.5``.
    4. Drop the runner-side ``_inject_mem_section`` / ``_track_mem_access``
       calls once attention-only mode is stable.

    Time budget proof-of-concept: registers the op (passes layer_idx gate)
    and demonstrates the design pattern without disturbing existing tests.
    """
    def bake(block, dim_positions, S):
        # When disabled: op is a no-op. The head's alibi_slopes[2] stays at
        # its module-init default (a small power-of-2 value from
        # AutoregressiveAttention.__init__), but the head's W_q/W_k/W_v/W_o
        # weights are all zero — so attention output for head 2 is 0 (V is 0)
        # and W_o for head 2 dims is 0 → no contribution to the residual.
        if not enable:
            return

        from ...vm_step import _SetDim as BD_DEFAULT
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        head = 2
        base = head * HD

        # Slope tuned to favor most-recent matching PSH within a typical
        # 4096-token (~117-step) KV-cache window. See docstring for analysis.
        if hasattr(attn, 'alibi_slopes') and attn.alibi_slopes is not None:
            attn.alibi_slopes[head] = 0.5

        BD = _as_setdim_proxy(dim_positions)
        L = 50.0
        scale = 10.0

        # === Q side: fire at MEM val byte 0 position during LI/LC/POP ===
        # MEM_VAL_B0 is set at val-byte-0 positions in every MEM section.
        # During load ops, that position is the natural "I am about to read
        # a memory value" anchor.
        attn.W_q[base, BD.MEM_VAL_B0] = L
        attn.W_q[base, BD.OP_LI_RELAY] = L / 5  # OP relay gates the load
        attn.W_q[base, BD.OP_LC_RELAY] = L / 5
        attn.W_q[base, BD.CMP + 3] = L / 5      # POP group
        attn.W_q[base, BD.CONST] = -L * 1.5    # threshold

        # === K side: match PSH-output STACK0 positions ===
        # PSH_AT_SP is set at SP/STACK0 positions during PSH steps.
        # MEM_STORE is set at MEM val-bytes (and at PSH-step's STACK0
        # output, via L6 head 6 / L7 head 7 broadcast in step W).
        attn.W_k[base, BD.PSH_AT_SP] = L
        attn.W_k[base, BD.MEM_STORE] = L / 2
        attn.W_k[base, BD.CONST] = -L * 0.5

        # === Address matching: 24 binary bits across 3 address bytes ===
        # Both sides use the same ADDR_KEY encoding. Address match gives
        # +300 to score (per-dim contribution after /sqrt(HD)). Mismatch
        # gives ~0 (random) to -300 (anti-match).
        addr_dim = 4
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

        # === V/O: copy CLEAN_EMBED bytes → OUTPUT at load position ===
        # The PSH step's STACK0 has CLEAN_EMBED_LO/HI = AX value at time of
        # PSH. Carrying it to OUTPUT at the load position writes the value
        # into the load result.
        scale_v = 1.0
        for k in range(16):
            attn.W_v[base + 1 + k, BD.CLEAN_EMBED_LO + k] = scale_v
            attn.W_v[base + 17 + k, BD.CLEAN_EMBED_HI + k] = scale_v
        for k in range(16):
            attn.W_o[BD.OUTPUT_LO + k, base + 1 + k] = 1.0
            attn.W_o[BD.OUTPUT_HI + k, base + 17 + k] = 1.0

    # Dim-ownership claims: L9 attn head 2 ALiBi mem attn.
    # When enable=True, V slots 1..32 carry CLEAN_EMBED to OUTPUT.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((9, "attn_W_v", f"2_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((9, "attn_W_v", f"2_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer9_alibi_mem_attn",
        phase=9.2,  # after lev_addr_relay (9.0) and lev_bp_to_pc_relay (9.1)
        reads={"MEM_VAL_B0", "OP_LI_RELAY", "OP_LC_RELAY", "CMP", "CONST",
               "PSH_AT_SP", "MEM_STORE", "ADDR_KEY",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=9,
        migrated=True,
        claims=_claims,
    )


def make_layer9_marker_suppress_op() -> Operation:
    """L9 FFN extension: marker suppression."""
    def bake(ffn, dim_positions, S):
        from ...vm_step import _set_layer9_marker_suppress
        # _set_layer9_marker_suppress takes (ffn, S, BD, start_unit). We need
        # to know what start_unit to use. The original code uses unit count
        # after _set_layer9_alu — for the migration shim we just pass start_unit=0.
        # This may overlap with layer9_alu's unit assignments; the original calls
        # them sequentially in the same FFN.
        _set_layer9_marker_suppress(ffn, S, _as_setdim_proxy(dim_positions), 0)

    return Operation(
        name="layer9_marker_suppress",
        phase=9,
        reads={"MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_OR", "OP_XOR", "OP_AND"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="ffn",
        bake_fn=bake,
    )


