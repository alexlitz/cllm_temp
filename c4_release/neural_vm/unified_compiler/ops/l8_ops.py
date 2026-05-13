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
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # The L8 lookup ALU consumes the *current step's* AX value via
        # AX_CARRY_LO at the AX marker (operand 1 for ADD/SUB/LEA at the
        # AX byte 0 position). Without an in-step producer, AX_CARRY_LO
        # carries the prev-prev step's value (the stale-AX_CARRY bug
        # observed in the IMM 10 / PSH / IMM 32 / ADD sequence, fixed by
        # the L8 head 6 AX_CARRY refresh in commit 3d1b700). The
        # ``layer8_head6_ax_carry_refresh`` op (phase=8.05) is the
        # canonical in-step producer.
        consumes_fresh={
            "AX_CARRY_LO": "AX_byte0",
            # ALU_LO at AX marker is the operand-A input to ADD/SUB/LEA.
            # Produced by ``layer7_operand_gather`` (phase=7, L7 head 0 +
            # head 1) at the AX byte 0 position. Without an in-step
            # producer, ALU_LO would carry stale prev-step values, breaking
            # binary-op semantics for any operand A computation.
            "ALU_LO": "AX_byte0",
        },
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeBasic::test_sub_basic",
            "TestSmoke32Bit::test_add_16bit",
            "TestSmoke32Bit::test_sub_16bit",
        },
        spec_section="BLOG_SPEC.md#binary-ALU",
        compaction_safe=True,
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
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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
        declarative_authority="declarative",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
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

    Post-bake adds a K-side MARK_AX exclusion (dim 35) so the head does
    not score the AX marker as a viable K candidate. Required because the
    L4 SP-to-ADDR_KEY op (when enabled) stages scale-10 ADDR_KEY content
    at the AX marker, which would otherwise outscore the scale-1 ADDR_KEY
    content at MEM val byte positions and cause multibyte_fetch to attend
    to the AX marker instead of the correct code byte. Safe to apply
    unconditionally — when L4 SP-staging is off the AX marker carries no
    ADDR_KEY content and the gate is a no-op.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer8_multibyte_fetch
        proxy = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        _set_layer8_multibyte_fetch(attn, S, proxy, HD)

        # Dim 35: at firing Q positions (AX bytes: H1[AX_I]=1 + IS_BYTE=1
        # → Q[35]=50) and K=AX marker (K[35]=-50), produce a -2500 raw
        # score (-312.5 after /sqrt(HD)=8). Excludes AX marker from
        # head 3's K candidates so the L4 SP-staged ADDR_KEY at AX marker
        # does not outscore the correct code byte target.
        base = 3 * HD
        AX_MARKER_K_EXCLUDE = 35
        AX_I = 1
        attn.W_q[base + AX_MARKER_K_EXCLUDE, proxy.H1 + AX_I] = 100.0
        attn.W_q[base + AX_MARKER_K_EXCLUDE, proxy.IS_BYTE] = 100.0
        attn.W_q[base + AX_MARKER_K_EXCLUDE, proxy.CONST] = -150.0
        attn.W_k[base + AX_MARKER_K_EXCLUDE, proxy.MARK_AX] = -50.0

    # Dim-ownership claims: L8 attn head 3 multibyte fetch.
    #   W_v[3*HD + 32 + k, CLEAN_EMBED_LO + k]  for k=0..15 (slot 32..47)
    #   W_v[3*HD + 48 + k, CLEAN_EMBED_HI + k]  for k=0..15 (slot 48..63)
    _claims = set()
    for k in range(16):
        _claims.add((8, "attn_W_v", f"3_{32 + k}", f"CLEAN_EMBED_LO+{k}"))
        _claims.add((8, "attn_W_v", f"3_{48 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer8_multibyte_fetch_bake",
        phase=8.1,
        reads={"FETCH_LO", "FETCH_HI", "ADDR_KEY", "IS_BYTE", "H1",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST", "MARK_AX"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#how-bytecode-is-passed-to-the-network",
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
        # Staleness invariants (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md).
        # This op produces the fresh AX-byte-0 OUTPUT for IMM (routes
        # AX_CARRY -> OUTPUT at AX byte positions). The L6 routing FFN
        # produces AX_byte0 OUTPUT for the other AX-emitting opcodes; this
        # L8 FFN extension covers the IMM multi-byte path.
        produces={
            "OUTPUT_LO": "AX_byte0",
            "OUTPUT_HI": "AX_byte0",
        },
        # ``_set_layer8_multibyte_routing`` re-invokes ``_set_layer8_alu``
        # internally to recover the ALU-final unit cursor (~2023) and then
        # appends 32 multibyte-IMM routing units, reaching unit 2054 — so
        # the L8 FFN needs 2055 hidden units total. Sibling ``layer8_alu``
        # (phase=8.2) writes the 0-2022 cluster; this op holds the
        # per-layer max and dominates the dynamic-FFN allocation for L8.
        ffn_units_used=2055,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#mixture-of-experts-routing",
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
        declarative_authority="declarative",
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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

    # Dim-ownership claims: L8 attn heads 0-2 SP gather (SP bytes → ADDR_B*).
    # Each head writes V slots 1..32 reading CLEAN_EMBED_LO/HI.
    _claims = set()
    for h in range(3):
        for k in range(16):
            _claims.add((8, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((8, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

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
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer8_head6_ax_carry_refresh_op(enable: bool = False) -> Operation:
    """L8 attn head 6: refresh AX_CARRY_LO/HI from prev step's AX marker OUTPUT.

    Mirrors the head-6 bake added to ``UnifiedVMCompiler._compile_l8_attention``
    in commit ``3d1b700`` (2026-05-12, fix-phase2-ax-carry-refresh). The bake
    reads ``OUTPUT_LO/HI`` from the previous step's AX marker (excluding the
    *current* AX marker via anti-OP_* gates) and writes the result to the
    current step's ``AX_CARRY_LO/HI``.

    Status: ``enable=False`` by default — the production ``full_vm_compiler``
    path does not yet route this bake (today's head-6 fix lives only in the
    separate ``UnifiedVMCompiler`` path, see ``unified_compiler/compiler.py``).
    The Operation is still registered (always) so its ``produces`` annotation
    participates in the staleness-invariant scan. Setting ``enable=True``
    flips the bake on so this op owns the head-6 weight programming in the
    ``full_vm_compiler`` path; until that wiring is validated end-to-end the
    default stays off to keep the production build byte-identical.

    The ``produces`` declaration is the canonical proof-of-concept for the
    staleness analyzer (Phase 3 / Agent G of ARCH_LEAKAGE_FIX_PLAN.md):
    removing this op from ``all_core_ops`` would leave the L8 ALU op (which
    declares ``consumes_fresh={"AX_CARRY_LO": "AX_byte0", ...}``) without an
    in-step producer, surfacing today's stale-AX_CARRY bug at compile time.
    """
    def _bake(block, dim_positions, S):
        if not enable:
            return
        BD = _as_setdim_proxy(dim_positions)
        attn = block.attn
        HD = attn.W_q.shape[0] // attn.num_heads
        base = 6 * HD
        AX_CARRY_L = 50.0  # head-local Q/K scale
        # Q[base+0]: fire only at current step's AX marker on subsequent
        # steps (HAS_SE = 1). The CONST baseline blocks first-step fires.
        attn.W_q.data[base, BD.MARK_AX] = AX_CARRY_L
        attn.W_q.data[base, BD.HAS_SE] = AX_CARRY_L
        attn.W_q.data[base, BD.CONST] = -AX_CARRY_L * 1.5
        # K[base+0]: match AX marker at any past position.
        attn.W_k.data[base, BD.MARK_AX] = AX_CARRY_L
        # V copies OUTPUT_LO/HI from the matched AX marker.
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 1.0
        # O writes to AX_CARRY_LO/HI at the query position (current AX marker).
        for k in range(16):
            attn.W_o.data[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0
            attn.W_o.data[BD.AX_CARRY_HI + k, base + 17 + k] = 1.0
        # Anti-leakage gate (dim 33): suppress at non-AX-marker queries.
        GATE = 33
        attn.W_q.data[base + GATE, BD.MARK_AX] = AX_CARRY_L
        attn.W_q.data[base + GATE, BD.CONST] = -AX_CARRY_L / 2
        attn.W_k.data[base + GATE, BD.CONST] = AX_CARRY_L
        # Anti-op gates: exclude the current AX marker from K via anti-OP_*.
        anti_ops = [
            BD.OP_IMM, BD.OP_EXIT, BD.OP_NOP, BD.OP_JMP, BD.OP_JSR, BD.OP_LEV,
            BD.OP_BZ, BD.OP_BNZ, BD.OP_PSH, BD.OP_ADJ, BD.OP_ENT,
            BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
            BD.OP_AND, BD.OP_OR, BD.OP_XOR,
            BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
            BD.OP_SHL, BD.OP_SHR,
            BD.OP_LI, BD.OP_LC, BD.OP_LEA,
        ]
        ANTI_OP_SLOT_START = 34
        for j, op_dim in enumerate(anti_ops):
            slot = ANTI_OP_SLOT_START + j
            if slot >= HD:
                break
            attn.W_q.data[base + slot, op_dim] = -AX_CARRY_L
            attn.W_k.data[base + slot, op_dim] = AX_CARRY_L

    return Operation(
        name="layer8_head6_ax_carry_refresh",
        # Phase 8.05 places it between layer8_sp_gather_bake (8.0) and
        # layer8_multibyte_fetch_bake (8.1) so the AX_CARRY refresh
        # completes before any downstream L8 op that consumes_fresh
        # AX_CARRY_LO/HI fires. The exact phase number is not load-bearing
        # for the staleness analyzer (it only checks producer.phase <=
        # consumer.phase); 8.05 keeps the L8 attn bakes contiguous.
        phase=8.05,
        reads={"MARK_AX", "HAS_SE", "OUTPUT_LO", "OUTPUT_HI", "CONST",
               "OP_IMM", "OP_EXIT", "OP_NOP", "OP_JMP", "OP_JSR", "OP_LEV",
               "OP_BZ", "OP_BNZ", "OP_PSH", "OP_ADJ", "OP_ENT",
               "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_AND", "OP_OR", "OP_XOR",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_SHL", "OP_SHR",
               "OP_LI", "OP_LC", "OP_LEA"},
        writes={"AX_CARRY_LO", "AX_CARRY_HI"},
        kind="block",
        bake_fn=_bake,
        layer_idx=8,
        # Always migrated=True so the bake runs when enable=True; when
        # enable=False the bake body is a no-op so production behavior is
        # unchanged. The Operation itself stays in the registry either way
        # so the staleness analyzer can see its ``produces`` annotation.
        migrated=True,
        produces={
            "AX_CARRY_LO": "AX_byte0",
            "AX_CARRY_HI": "AX_byte0",
        },
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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

    # Dim-ownership claims: L8 attn head 4 OP_IMM relay.
    #   W_q[4*HD, IS_BYTE], W_q[4*HD, H1+AX_I], W_q[4*HD, CONST]
    #   W_k[4*HD, MARK_AX], W_k[4*HD, IS_BYTE], W_k[4*HD, CONST]
    #   W_v[4*HD, OP_IMM]
    #   W_o[OP_IMM, 4*HD]
    #   GATE4 = 1 sub-head: W_q[4*HD+1, IS_BYTE], W_q[4*HD+1, CONST]
    #                       W_k[4*HD+1, CONST]
    _claims = {
        (8, "attn_W_v", "4_0", "OP_IMM+0"),
        (8, "attn_W_o", "4_0", "OP_IMM+0"),
    }

    return Operation(
        name="layer8_op_imm_relay",
        reads={"IS_BYTE", "H1", "MARK_AX", "OP_IMM", "CONST"},
        writes={"OP_IMM"},
        kind="block",
        bake_fn=_bake,
        layer_idx=8,
        phase=8.4,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
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

        # === Dim 0: bias — fire only at AX marker on binary-pop opcodes ===
        # Coordination with L8 multibyte_fetch (head 3): that head reads
        # ADDR_KEY across all K positions for multi-byte IMM fetch. With
        # L4 SP-to-ADDR_KEY staging the AX marker carries ADDR_KEY content
        # equal to SP, which would otherwise alias as a spurious K-match
        # candidate for multibyte_fetch. To keep the two heads disjoint
        # we restrict head 5 to fire only on the binary-pop opcode set
        # (the same set CMP[3] relays via L6 head 6) and add negative
        # blockers on OP_LI / OP_LC / OP_IMM / OP_LEA / OP_PSH / OP_JSR /
        # OP_ENT / OP_LEV / OP_JMP / OP_ADJ / OP_BZ / OP_BNZ / OP_EXIT so
        # head 5 stays silent on non-binary-pop steps. OP_* flags are
        # written at the AX marker by L5 FFN (opcode decode) so they are
        # available here as direct Q-side gates.
        attn.W_q[base, BD.CONST] = -2000.0
        attn.W_q[base, BD.MARK_AX] = 2000.0     # require AX marker
        # Positive gates: any binary-pop opcode at the AX marker. Each
        # flag is ~1.0 when active, so a single +500 gate gives Q[0] = 500
        # when any one fires, comfortably above the -2000 baseline.
        for op_dim in (BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_DIV, BD.OP_MOD,
                       BD.OP_EQ, BD.OP_NE, BD.OP_LT, BD.OP_GT, BD.OP_LE, BD.OP_GE,
                       BD.OP_OR, BD.OP_XOR, BD.OP_AND, BD.OP_SHL, BD.OP_SHR,
                       BD.OP_SI, BD.OP_SC):
            attn.W_q[base, op_dim] = 500.0
        # Negative blockers — guarantee head 5 stays off when LI/LC/IMM/LEA/
        # PSH/etc. is active even if a binary-pop residual leaks. These
        # opcodes have their own ADDR_KEY/ALU paths and must not collide.
        for op_dim in (BD.OP_LI, BD.OP_LC, BD.OP_IMM, BD.OP_LEA,
                       BD.OP_PSH, BD.OP_JSR, BD.OP_ENT, BD.OP_LEV,
                       BD.OP_JMP, BD.OP_ADJ, BD.OP_BZ, BD.OP_BNZ,
                       BD.OP_EXIT):
            attn.W_q[base, op_dim] = -2000.0
        # Suppress at PC/SP/BP/STACK0/MEM markers — at these positions the
        # ADDR_KEY band carries other information (code addresses, mem
        # addresses) that would alias into this head's address match.
        attn.W_q[base, BD.MARK_PC] = -2000.0
        attn.W_q[base, BD.MARK_SP] = -2000.0
        attn.W_q[base, BD.MARK_BP] = -2000.0
        attn.W_q[base, BD.MARK_MEM] = -2000.0
        attn.W_q[base, BD.MARK_STACK0] = -2000.0
        attn.W_k[base, BD.CONST] = 10.0
        # === Dim 28: K-side AX marker exclusion (self-attention guard) ===
        # The L4 SP gather writes scale-10 ADDR_KEY content at the AX
        # marker. At head 5's firing Q position (AX marker), the address
        # encoding dims (4-27) match that staged content exactly, which
        # would otherwise drive softmax to self-attend to the AX marker
        # instead of the MEM val byte we want. The self-match contributes
        # up to 12 dims * (10 * 10)^2 = 120,000 raw score (= 15,000 after
        # /sqrt(HD)=8) at AX marker. Q[28]*K[28] = 100 * -2000 = -200,000
        # raw (= -25,000 after /sqrt(HD)=8) cleanly overwhelms it. This
        # dim only contributes when BOTH Q-side AND K-side are at AX
        # marker — i.e. only when head 5 would otherwise self-attend —
        # giving a clean negative penalty without leaking into non-firing
        # Q positions or non-AX K positions.
        AX_K_EXCLUDE = 28
        attn.W_q[base + AX_K_EXCLUDE, BD.MARK_AX] = 100.0
        attn.W_k[base + AX_K_EXCLUDE, BD.MARK_AX] = -2000.0

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

    # Dim-ownership claims: L8 attn head 5 mem-to-ALU.
    # Only claim load-bearing V/O slot/column pairs (not the dense Q/K gates
    # which are head-local and unlikely to collide with other ops).
    #   W_v[5*HD + 1 + k, CLEAN_EMBED_LO + k]    for k=0..15 (V slot 1..16)
    #   W_v[5*HD + 17 + k, CLEAN_EMBED_HI + k]   for k=0..15 (V slot 17..32)
    #   W_o[ALU_LO + k, 5*HD + 1 + k]            for k=0..15
    #   W_o[ALU_HI + k, 5*HD + 17 + k]           for k=0..15
    # Conditional on enable=True; declared unconditionally so the registry
    # catches latent collisions once the op is enabled.
    _claims = set()
    if enable:
        for k in range(16):
            _claims.add((8, "attn_W_v", f"5_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((8, "attn_W_v", f"5_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer8_mem_to_alu",
        # Phase 8.45 places this after layer8_op_imm_relay (8.4) and BEFORE
        # the L8 alu_postop_attach (8.5), keeping all L8 attn bakes in
        # phase order.
        phase=8.45,
        reads={"MARK_AX", "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
               "OP_EQ", "OP_NE", "OP_LT", "OP_GT", "OP_LE", "OP_GE",
               "OP_OR", "OP_XOR", "OP_AND", "OP_SHL", "OP_SHR",
               "OP_SI", "OP_SC", "OP_LI", "OP_LC", "OP_IMM", "OP_LEA",
               "OP_PSH", "OP_JSR", "OP_ENT", "OP_LEV", "OP_JMP", "OP_ADJ",
               "OP_BZ", "OP_BNZ", "OP_EXIT", "MEM_STORE", "L2H0",
               "H1", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO",
               "ADDR_B1_HI", "ADDR_B2_LO", "ADDR_B2_HI",
               "CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "CONST"},
        writes={"ALU_LO", "ALU_HI"},
        kind="block",
        bake_fn=bake,
        layer_idx=8,
        migrated=True,
        claims=_claims,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )

