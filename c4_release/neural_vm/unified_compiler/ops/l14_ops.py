"""Auto-extracted per-layer factories. See ../migrated_ops.py for history."""

from ..ir import CompilerIR
from ..layer_compiler import Operation
from ..primitives import AO, AP, DeclarativeAttentionHeadSpec, Primitives
from .shared import _as_setdim_proxy


def make_layer14_mem_generation_op() -> Operation:
    """L14 attention: generate MEM section tokens (addr + value) for SI/SC/PSH."""
    def bake(attn, dim_positions, S):
        from ...vm_step import _set_layer14_mem_generation
        HD = attn.W_q.shape[0] // attn.num_heads
        proxy = _as_setdim_proxy(dim_positions)
        _set_layer14_mem_generation(attn, S, proxy, HD)
        _clear_l14_mem_generation_overbroad_sp_suppression(attn, proxy, HD)
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[:8] = 5.0

    # Dim-ownership claims: L14 attn 8 heads MEM generation.
    # Each head writes V slots 1..32 reading CLEAN_EMBED + OUTPUT.  For V
    # slots, the CLEAN_EMBED column is the load-bearing "source" — the
    # +OUTPUT addition just lets V pick up either side (marker or byte
    # position). We claim only the CLEAN_EMBED rows (OUTPUT collisions on
    # the same V slot are recurring in this op family and acceptable).
    _claims = set()
    for h in range(8):
        for k in range(16):
            _claims.add((14, "attn_W_v", f"{h}_{1 + k}", f"CLEAN_EMBED_LO+{k}"))
            _claims.add((14, "attn_W_v", f"{h}_{17 + k}", f"CLEAN_EMBED_HI+{k}"))

    return Operation(
        name="layer14_mem_generation",
        phase=14,
        reads={"MARK_MEM", "MARK_SP", "MARK_STACK0", "OP_PSH", "OP_SI", "OP_SC",
               "OP_JSR", "OP_ENT", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "AX_CARRY_LO", "AX_CARRY_HI", "ADDR_B0_LO", "ADDR_B0_HI",
               "MEM_STORE", "MEM_ADDR_SRC", "STACK0_BYTE0", "L1H0", "L1H1", "L1H2",
               "H0", "H1", "L1H4", "H2", "H3", "H4",
               "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3", "IS_BYTE"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="attn",
        layer_idx=14,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        migrated=True,
        claims=_claims,
        alibi_slopes={head: 5.0 for head in range(8)},
        smoke_tests={
            "TestSmokeBasic::test_add_basic",
            "TestSmokeFunctionCall::test_simple_function",
            "TestSmokeMemory::test_sc_lc_roundtrip",
            "TestSmokeMemory::test_si_li_roundtrip",
        },
        spec_section="BLOG_SPEC.md#memory",
    )


def _layer14_alu_high_byte_relay_spec(BD) -> DeclarativeAttentionHeadSpec:
    """Relay staged wide-ALU byte 1 from the AX marker to AX byte 0."""

    AX_I = 1
    q = [
        AP(0, BD.IS_BYTE, 100.0),
        AP(0, BD.H1 + AX_I, 100.0),
        AP(0, BD.BYTE_INDEX_0, 100.0),
        AP(0, BD.BYTE_INDEX_1, -1000.0),
        AP(0, BD.BYTE_INDEX_2, -1000.0),
        AP(0, BD.BYTE_INDEX_3, -1000.0),
        AP(0, BD.CONST, -250.0),
        AP(1, BD.IS_BYTE, 100.0),
        AP(1, BD.H1 + AX_I, 100.0),
        AP(1, BD.BYTE_INDEX_0, 100.0),
        AP(1, BD.BYTE_INDEX_1, -1000.0),
        AP(1, BD.BYTE_INDEX_2, -1000.0),
        AP(1, BD.BYTE_INDEX_3, -1000.0),
        AP(1, BD.CONST, -250.0),
        AP(33, BD.IS_BYTE, 10000.0),
        AP(33, BD.H1 + AX_I, 10000.0),
        AP(33, BD.BYTE_INDEX_0, 10000.0),
        AP(33, BD.BYTE_INDEX_1, -100000.0),
        AP(33, BD.BYTE_INDEX_2, -100000.0),
        AP(33, BD.BYTE_INDEX_3, -100000.0),
        AP(33, BD.CONST, -25000.0),
        AP(34, BD.OP_LI_RELAY, -10000.0),
        AP(34, BD.OP_LC_RELAY, -10000.0),
        AP(35, BD.TEMP + 8, 10000.0),
        AP(35, BD.TEMP + 9, 10000.0),
    ]
    k = [
        AP(0, BD.MARK_AX, 100.0),
        AP(1, BD.OP_MUL, 100.0),
        AP(1, BD.OP_SHL, 100.0),
        AP(33, BD.CONST, 5.0),
        AP(34, BD.CONST, 5.0),
        AP(35, BD.CONST, -20.0),
    ]
    v = []
    o = []
    for idx in range(16):
        v.append(AP(1 + idx, BD.AX_FULL_LO + idx, 1.0))
        v.append(AP(17 + idx, BD.AX_FULL_HI + idx, 1.0))
        o.append(AO(BD.OUTPUT_LO + idx, 1 + idx, 20.0))
        o.append(AO(BD.OUTPUT_HI + idx, 17 + idx, 20.0))
    return DeclarativeAttentionHeadSpec(
        head_idx=8,
        q=tuple(q),
        k=tuple(k),
        v=tuple(v),
        o=tuple(o),
    )


def _layer14_alu_high_byte_relay_ir(dim_positions, HD) -> CompilerIR:
    BD = _as_setdim_proxy(dim_positions)
    ir = CompilerIR()
    ir.layer(0).attention.append(_layer14_alu_high_byte_relay_spec(BD))
    return ir


def make_layer14_alu_high_byte_relay_op() -> Operation:
    """L15 attention: relay staged MUL/SHL result byte 1 into AX bytes.

    The wide ALU composites stage result byte 1 in AX_FULL_LO/HI at the AX
    marker. The autoregressive token stream then needs the following AX byte
    token to emit that staged byte. This declarative head copies the staged
    byte from the current step's AX marker to the AX byte-0 query position.
    It lives in the resized L15 attention block because the base L14 attention
    has only heads 0-7 occupied by MEM generation.
    """

    def bake(target, dim_positions, S):
        del S
        attn = getattr(target, "attn", target)
        proxy = _as_setdim_proxy(dim_positions)
        HD = attn.W_q.shape[0] // attn.num_heads
        Primitives.generate_attention_head(
            attn,
            _layer14_alu_high_byte_relay_spec(proxy),
            HD,
        )
        if hasattr(attn, "alibi_slopes") and attn.alibi_slopes is not None:
            attn.alibi_slopes[8] = 1.0

    _claims = set()
    for k in range(16):
        _claims.add((15, "attn_W_v", f"8_{1 + k}", f"AX_FULL_LO+{k}"))
        _claims.add((15, "attn_W_v", f"8_{17 + k}", f"AX_FULL_HI+{k}"))

    return Operation(
        name="layer15_alu_high_byte_relay",
        phase=15.05,
        reads={"IS_BYTE", "H1", "BYTE_INDEX_0", "MARK_AX", "OP_MUL",
               "OP_SHL", "OP_LI_RELAY", "OP_LC_RELAY",
               "AX_FULL_LO", "AX_FULL_HI", "TEMP", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        layer_idx=15,
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        compiler_ir_factory=_layer14_alu_high_byte_relay_ir,
        migrated=True,
        claims=_claims,
        alibi_slopes={8: 1.0},
        smoke_tests={
            "TestSmoke32Bit::test_mul_overflow",
            "TestSmoke32Bit::test_shl_8bit",
        },
        spec_section="BLOG_SPEC.md#wide-alu-byte-relay",
    )


def _clear_l14_mem_generation_overbroad_sp_suppression(attn, BD, HD) -> None:
    """Keep L14 MEM generation live when H1[SP] also fires at MEM positions.

    The legacy helper suppresses queries whenever ``H1[SP]`` is high to avoid
    early SP-byte pollution. In the strict autoregressive trace, ``H1[SP]`` is
    also high at the MEM marker, so heads 0-3 select the softmax1 sink instead
    of the SP marker and emit zero address bytes. The position gate already
    suppresses non-MEM targets strongly; this removes only the overbroad
    H1[SP] query penalties.
    """

    pc_i = 0
    ax_i = 1
    sp_i = 2
    bp_i = 3
    mem_i = 4
    for head in range(8):
        base = head * HD
        attn.W_q.data[base, BD.H1 + sp_i] = 0.0
        # MEM_STORE is represented with strength 2 at MEM positions, so the
        # legacy +/-500 position gate can exactly cancel the +500 MEM_STORE
        # gate at non-target MEM offsets. Doubling the position gate leaves
        # target positions at zero net gate but makes non-target heads stay
        # below the softmax1 sink.
        attn.W_q.data[base + 33, :] *= 2.0
        attn.W_q.data[base + 33, BD.H1 + sp_i] = 0.0
        # L14's sources are the prior SP/AX/STACK0 sections. Once MEM bytes
        # start emitting, their BYTE_INDEX flags are more recent than the true
        # sources and can win via ALiBi. Use an otherwise-unused score slot to
        # exclude the current MEM section as an attention source for every head.
        attn.W_q.data[base + 35, BD.CONST] = 40.0
        attn.W_k.data[base + 35, BD.MARK_MEM] = -40.0
        attn.W_k.data[base + 35, BD.H3 + 4] = -40.0

        # The legacy position gate was calibrated before the stricter
        # source-bonus rows below. In neural-authoritative smoke, PC/AX/BP
        # byte queries can still overcome that gate and make MEM generation
        # write OUTPUT outside the MEM section. Add a separate non-MEM target
        # blocker that stays inactive at the real MEM addr/value targets.
        target_block_s = 2000.0
        for dim in (
            BD.MARK_PC,
            BD.MARK_AX,
            BD.MARK_BP,
            BD.MARK_STACK0,
            BD.H1 + pc_i,
            BD.H1 + ax_i,
            BD.H1 + bp_i,
            BD.H4 + bp_i,
        ):
            attn.W_q.data[base + 38, dim] = -target_block_s
        attn.W_k.data[base + 38, BD.CONST] = 5.0

    # Head 0 predicts address byte 0 from the SP marker for PSH. The legacy
    # source bonus keyed only on H1[SP], which is active on SP byte positions
    # but not on the SP marker where byte 0's fresh OUTPUT lives.
    attn.W_k.data[1, BD.MARK_SP] = 45.0
    # Heads 1-3 can read the already-emitted SP byte tokens directly. The
    # old shifted-OUTPUT path is fragile in strict neural mode; this bonus is
    # active only for PSH because Q[base+1] flips negative when MEM_ADDR_SRC=1.
    # Suppress the BP/STACK0 span on the same source dimension: BYTE_INDEX_*
    # is deliberately global, and without this guard the nearer STACK0 zero
    # bytes beat the SP address bytes under ALiBi in strict autoregressive
    # smoke.
    for head in range(4):
        attn.W_k.data[head * HD + 1, BD.H4 + bp_i] = -30.0
        attn.W_k.data[head * HD + 1, BD.MARK_STACK0] = -30.0
    for head, byte_dim in (
        (1, BD.BYTE_INDEX_1),
        (2, BD.BYTE_INDEX_2),
        (3, BD.BYTE_INDEX_3),
    ):
        # These heads now source the SP/STACK0 byte token directly, so reading
        # that token's OUTPUT would mix in the source position's prediction for
        # the following byte (usually the L3 zero default). Keep CLEAN_EMBED as
        # the only payload source while retaining V[0]'s default cancel.
        base = head * HD
        attn.W_v.data[base + 0, BD.CONST] = 1.0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.OUTPUT_LO + k] = 0.0
            attn.W_v.data[base + 17 + k, BD.OUTPUT_HI + k] = 0.0
        attn.W_k.data[head * HD + 1, byte_dim] = 45.0

    # SI/SC addresses must come from the current STACK0 byte tokens directly.
    # The legacy MEM_ADDR_SRC row selected the preceding byte (and head 0 could
    # still prefer the STACK0 marker), which is wrong once the current stack
    # bytes already exist in the autoregressive context. Keep the existing
    # MEM_ADDR_SRC query row, but retarget its keys to bytes 0..3.
    si_source_s = 60.0
    for head in range(4):
        base = head * HD
        for dim in (
            BD.STACK0_BYTE0,
            BD.H1 + bp_i,
            BD.L1H4 + bp_i,
            BD.H2 + bp_i,
            BD.H3 + bp_i,
            BD.H4 + bp_i,
            BD.MARK_STACK0,
            BD.H1 + ax_i,
            BD.H1 + sp_i,
        ):
            attn.W_k.data[base + 2, dim] = 0.0
        attn.W_k.data[base + 2, BD.MARK_STACK0] = -si_source_s
        attn.W_k.data[base + 2, BD.H1 + ax_i] = -si_source_s
        attn.W_k.data[base + 2, BD.H1 + sp_i] = -si_source_s
        if head == 0:
            attn.W_k.data[base + 2, BD.L1H4 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.H1 + bp_i] = -si_source_s
        elif head == 1:
            attn.W_k.data[base + 2, BD.H2 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.L1H4 + bp_i] = -si_source_s
        elif head == 2:
            attn.W_k.data[base + 2, BD.H3 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.H2 + bp_i] = -si_source_s
        else:
            attn.W_k.data[base + 2, BD.H4 + bp_i] = si_source_s
            attn.W_k.data[base + 2, BD.H3 + bp_i] = -si_source_s

    # Value heads source AX for PSH/SI/SC and STACK0 for JSR/ENT. With the
    # steeper L14 ALiBi slope, the old H1[AX] bonus is not strong enough to
    # beat the softmax1 sink from MEM value positions. This score slot makes
    # the source choice explicit without changing the payload path.
    source_s = 50.0
    for head in range(4, 8):
        base = head * HD
        # The value heads double the CLEAN_EMBED nibble payload to overcome
        # downstream defaults. Keep V[0] as a matched cancel so nonzero bytes
        # beat the L3 zero default while real zero bytes still emit 0.
        attn.W_v.data[base + 0, BD.CONST] = 2.0
        for k in range(16):
            attn.W_v.data[base + 1 + k, BD.CLEAN_EMBED_LO + k] = 2.0
            attn.W_v.data[base + 17 + k, BD.CLEAN_EMBED_HI + k] = 2.0
        attn.W_q.data[base + 36, BD.CONST] = source_s
        attn.W_q.data[base + 36, BD.OP_JSR] = -2.0 * source_s
        attn.W_q.data[base + 36, BD.OP_ENT] = -2.0 * source_s
        attn.W_k.data[base + 36, BD.H1 + ax_i] = source_s
        attn.W_k.data[base + 36, BD.H1 + sp_i] = -source_s
        attn.W_k.data[base + 36, BD.H4 + bp_i] = -source_s
        # STACK0 value bytes carry H4[BP], while the STACK0 marker carries
        # both H4[BP] and MARK_STACK0. For JSR/ENT the query is negative, so
        # H4[BP]'s negative K makes STACK0 bytes attractive. Give the marker a
        # matching positive term so it nets to zero instead of winning as a
        # payload-less source.
        attn.W_k.data[base + 36, BD.MARK_STACK0] = source_s
        attn.W_k.data[base + 36, BD.MARK_MEM] = -source_s
        attn.W_k.data[base + 36, BD.H3 + mem_i] = -source_s

        # PSH store values are sourced from AX. STACK0 is generated later in
        # the same step and is not authoritative for the MEM value bytes here.

        # SI/SC AX preservation now happens late in L16 before the MEM value
        # bytes are generated. Keep the source selector byte-indexed on the
        # current AX bytes; the previous "prefer older AX" penalty makes byte 0
        # lose to the nearer byte 3 zero under strict neural ALiBi.
        attn.W_q.data[base + 37, :] = 0.0
        attn.W_k.data[base + 37, :] = 0.0


def make_layer14_temp_clear_op() -> Operation:
    """L14 FFN: Clear TEMP[0] at PC marker when OP_LEV is active.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Chains with the other L14
    additive cleanup ops (``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``) via a shared FFN unit counter stored
    on ``block.ffn._l14_unit_counter``. First in the chain (phase=14.1).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_temp_clear
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_temp_clear(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_temp_clear",
        phase=14.1,
        reads={"OP_LEV", "MARK_PC", "CONST"},
        writes={"TEMP"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def make_layer14_clear_addr_key_pollution_op() -> Operation:
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Second in the chain (phase=14.2).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_addr_key_pollution
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_addr_key_pollution(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_addr_key_pollution",
        phase=14.2,
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "MARK_PC", "MARK_BP", "MARK_AX", "MARK_STACK0", "MARK_SP",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer14_clear_output_corruption_op() -> Operation:
    """L14 FFN: Boost OUTPUT[0] at STACK0 byte positions to fix attention bleed.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Third in the chain (phase=14.3).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_output_corruption
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_output_corruption(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_output_corruption",
        phase=14.3,
        reads={"H4", "H1", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "OP_JSR", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
               "MARK_STACK0", "IS_BYTE", "BYTE_INDEX_3", "PSH_AT_SP", "CMP",
               "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#registers",
    )


def make_layer14_clear_mem_marker_output_op() -> Operation:
    """L14 FFN: Clear OUTPUT at MEM marker for OP_JSR/OP_ENT.

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter on ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops
    (``layer14_temp_clear``, ``layer14_clear_addr_key_pollution``,
    ``layer14_clear_output_corruption``). Last in the chain (phase=14.4).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_clear_mem_marker_output
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_clear_mem_marker_output(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_clear_mem_marker_output",
        phase=14.4,
        reads={"OP_JSR", "OP_ENT", "MARK_MEM", "IS_BYTE",
               "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_STACK0",
               "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#memory",
    )


def make_layer14_jsr_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_JSR is active.

    FIX 2026-05-12 (fix-jsr-ax-bytes-1-3): Per C4's 8-bit-AX convention, AX
    bytes 1-3 must be 0. For OP_JSR, the L9 ALU JSR-preserve routing writes
    the previous AX byte 0 at MARK_AX (correctly emitting AX byte 0), but
    bytes 1-3 of AX are predicted at AX byte 0/1/2 positions where L14
    attention / L7 head 1 SP gather contaminate OUTPUT with SP/PC bytes.

    This op mirrors ``BinaryOpByteZeroingPostOp``: at AX byte positions
    (IS_BYTE + H1[AX]) when OP_JSR is active, write -3/S to every OUTPUT_LO
    /OUTPUT_HI nibble dim and +5/S to OUTPUT_LO[0]/OUTPUT_HI[0], so the
    byte-value-0 token wins argmax — producing AX bytes 1-3 = 0x00. OP_JSR
    is broadcast to AX byte positions by L7 head 5 (V slot 8, also new in
    this commit).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.6: runs AFTER ``layer14_addr_key_neural_decode`` (14.5).
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_jsr_ax_bytes_zero
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_jsr_ax_bytes_zero(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_jsr_ax_bytes_zero",
        phase=14.6,
        reads={"OP_JSR", "IS_BYTE", "H1", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"TestSmokeFunctionCall::test_simple_function"},
        spec_section="BLOG_SPEC.md#function-calls",
    )


def make_layer14_alu_nocarry_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when a non-carry ALU
    op is active (V7 Block 13, fix-v7-block13-ax-merge, 2026-05-12).

    Per ``c4_release/docs/V7_HEAP_OPS_NEURAL_PLAN.md`` §2 (AX merge): for
    non-carry ALU ops (AND/OR/XOR/SHR), AX bytes 1-3 must be 0 per C4's
    8-bit-AX-with-32-bit-register convention. The L10
    ``BinaryOpByteZeroingPostOp`` already attempts this clearing but can be
    overridden by downstream contamination at L11-L14 (MUL accumulators,
    L14 mem_addr_gather attention bleed, etc.). This L14 cleanup runs AFTER
    those sources and BEFORE L15 to restore AX bytes 1-3 = 0x00.

    Mirrors ``make_layer14_jsr_ax_bytes_zero_op`` and
    ``make_layer14_lc_ax_bytes_zero_op``: gates on ``TEMP[7]`` (NOCARRY_ALU_OP
    relay = OP_AND | OP_OR | OP_XOR | OP_SHR, supplied by L7 head 5 V slot
    9, also new in this commit) at AX byte positions 1-3, blocks at byte 0
    via ``BYTE_INDEX_0 = -S*4``, and zeros bytes 1-3 by writing -3/S to
    every OUTPUT_LO/HI nibble dim and +5/S to OUTPUT_LO[0] / OUTPUT_HI[0].

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.8: runs AFTER ``layer14_lc_ax_bytes_zero`` (14.7) — JSR / LC /
    nocarry-ALU gate on disjoint relays (OP_JSR vs OP_LC_RELAY vs TEMP[7])
    so the relative order within 14.6-14.8 only matters for unit allocation.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_alu_nocarry_ax_bytes_zero
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_alu_nocarry_ax_bytes_zero(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_alu_nocarry_ax_bytes_zero",
        phase=14.8,
        reads={"TEMP", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        # Last op in the L14 FFN chain (``_l14_unit_counter`` reaches 1311
        # after this op runs). The chain is: temp_clear (1 unit) →
        # clear_addr_key_pollution (48) → clear_output_corruption (3) →
        # clear_mem_marker_output (64) → addr_key_neural_decode (1184) →
        # jsr_ax_bytes_zero (4) → lc_ax_bytes_zero (4) → this op (4).
        # Annotating only the chain tail with the cumulative max is
        # sufficient — the compiler aggregates per-layer max across all
        # ops, so this single annotation suffices for L14 dynamic sizing.
        ffn_units_used=1312,
        smoke_tests={
            "TestSmoke32Bit::test_and_16bit",
            "TestSmoke32Bit::test_or_16bit",
            "TestSmoke32Bit::test_shr_8bit",
            "TestSmoke32Bit::test_xor_16bit",
            "TestSmokeBitwise::test_and_basic",
            "TestSmokeBitwise::test_or_basic",
            "TestSmokeBitwise::test_xor_basic",
            "TestSmokeShift::test_shr",
        },
        spec_section="BLOG_SPEC.md#shifts",
    )


def make_layer14_lc_ax_bytes_zero_op() -> Operation:
    """L14 FFN: Zero AX bytes 1-3 at AX byte positions when OP_LC is active.

    FIX V7 Phase 7a (LC byte clearing, per V7_HEAP_OPS_NEURAL_PLAN.md):
    Per C4's LC (load-char) semantics, the loaded value is a single byte
    placed in AX byte 0; AX bytes 1-3 must be 0. L15 attention head 0
    writes the resolved memory byte into ``OUTPUT_LO/HI`` at the AX byte 0
    position. Heads 1-3 fire for LI (writing AX bytes 1-3 from the loaded
    word), but for LC they must be suppressed so bytes 1-3 emit 0.

    Mirrors ``make_layer14_jsr_ax_bytes_zero_op``: at AX byte positions
    1-3 (IS_BYTE + H1[AX] + NOT BYTE_INDEX_0) when OP_LC_RELAY is active,
    write -3/S to every OUTPUT_LO/HI nibble dim and +5/S to OUTPUT_LO[0]
    / OUTPUT_HI[0] so the byte-value-0 token wins argmax — producing AX
    bytes 1-3 = 0x00. Byte 0 is protected by a strong BYTE_INDEX_0
    blocker so this op does NOT override the L15 head 0 load result.

    OP_LC_RELAY at AX byte positions is supplied by L7 head 5 (V slot 2,
    already wired in ``_set_layer7_memory_heads``).

    Pinned to ``layer_idx=14`` via ``kind="block"``. Shares the FFN unit
    counter ``block.ffn._l14_unit_counter`` with the other L14 cleanup ops.
    Phase 14.7: runs AFTER ``layer14_jsr_ax_bytes_zero`` (14.6) — the JSR
    and LC ops gate on disjoint relays (OP_JSR vs OP_LC_RELAY) so the
    relative order within phase 14.6-14.7 only matters for unit allocation.
    """
    def bake(block, dim_positions, S):
        from ...vm_step import _set_layer14_lc_ax_bytes_zero
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _set_layer14_lc_ax_bytes_zero(
            ffn, S, _as_setdim_proxy(dim_positions), start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_lc_ax_bytes_zero",
        phase=14.7,
        reads={"OP_LC_RELAY", "IS_BYTE", "H1", "BYTE_INDEX_3", "CONST"},
        writes={"OUTPUT_LO", "OUTPUT_HI"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"TestSmokeMemory::test_sc_lc_roundtrip"},
        spec_section="BLOG_SPEC.md#memory",
    )


def _bake_addr_key_neural_decode(ffn, dim_positions, S, start_unit=0):
    """Bake the BLOG_SPEC.md:830 ADDR_KEY nibble decode into ``ffn``.

    Computes the same per-val-byte ADDR_KEY one-hot writes that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, but via
    a baked FFN reading the addr byte values that L13's
    ``_set_layer13_mem_addr_gather`` has already gathered into
    ``ADDR_B0_LO``/``ADDR_B0_HI``/``ADDR_B1_LO`` at MEM val byte
    positions.

    Per val byte position (gated by MEM_VAL_B{0,1,2,3}):
      Let addr_b0 = (hi << 4 | lo) be the byte 0 of the MEM section's
      address.  Let addr_b1_lo be the low nibble of byte 1.  Then
      ``byte_addr = addr_b0 + byte_off`` for byte_off ∈ {0,1,2,3}.

      ADDR_KEY[byte_addr & 0xF]              = 1.0  (lo nibble)
      ADDR_KEY[16 + (byte_addr >> 4) & 0xF]  = 1.0  (hi nibble)
      ADDR_KEY[32 + (addr_b1 + carry) & 0xF] = 1.0  (top nibble)

      where carry = 1 iff (lo + byte_off) >= 16 and the hi nibble's
      addition (hi + carry_from_lo) overflowed past 15.  In the
      common case (byte_off ≤ 3, hi nibble < 15), the top nibble is
      just ``addr_b1_lo``; the high-byte carry case only occurs when
      hi==15 AND lo+byte_off >= 16, which is rare.

    Encoding strategy: enumerate the 16×16×4 combinations of
    (addr_b0_lo, addr_b0_hi, byte_off).  For each, compute byte_addr
    and emit two FFN units:
      - one writing the lo+hi nibbles of byte_addr into ADDR_KEY[0..31]
      - one writing the top nibble (addr_b1_lo+carry) into
        ADDR_KEY[32..47]

    Both units use a 3-way AND in the silu path:
      MEM_VAL_B{byte_off} + ADDR_B0_LO[lo] + ADDR_B0_HI[hi] >= 3
    with threshold ``-S*2.5`` so only all-3-match fires.

    The top-nibble unit additionally reads ADDR_B1_LO (a 4-way AND with
    threshold ``-S*3.5``) to source the high byte.  The carry case
    (rare) is handled by selecting addr_b1_lo+1 instead of addr_b1_lo
    when (lo + byte_off >= 16) AND (hi == 15).

    Returns ``next_unit`` so callers can chain.
    """
    BD = _as_setdim_proxy(dim_positions)
    unit = start_unit
    MEM_VAL_DIMS = [BD.MEM_VAL_B0, BD.MEM_VAL_B1, BD.MEM_VAL_B2, BD.MEM_VAL_B3]

    # Lo + Hi nibble units (write ADDR_KEY[0..31]): one unit per
    # (byte_off, hi, lo) combination, 4*16*16 = 1024 units.
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for hi in range(16):
            for lo in range(16):
                byte_addr = ((hi << 4) | lo) + byte_off
                new_lo = byte_addr & 0xF
                new_hi = (byte_addr >> 4) & 0xF
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + hi] = S
                ffn.b_up[unit] = -S * 2.5
                ffn.b_gate[unit] = 1.0
                ffn.W_down[BD.ADDR_KEY + new_lo, unit] = 2.0 / S
                ffn.W_down[BD.ADDR_KEY + 16 + new_hi, unit] = 2.0 / S
                unit += 1

    # --- Top nibble units (write ADDR_KEY[32..47]) ---
    # For each (lo, hi, byte_off, b1_lo): top = (b1_lo + carry) & 0xF
    # where carry = 1 iff (byte_addr_full = ((hi<<4|lo) + byte_off) >> 8) > 0.
    # That carry-into-byte-1 happens only when hi==15 AND lo+byte_off >= 16
    # for byte_off ≤ 3. For correctness we still iterate the full lookup.
    # 16 * 16 * 4 * 16 = 16384 units — too many. We exploit the fact that
    # the high carry condition only depends on (hi, lo, byte_off), so we
    # split into:
    #   (a) common case (no carry): always emit ADDR_B1_LO[b1_lo] →
    #       ADDR_KEY+32+b1_lo. This is independent of addr_b0 / byte_off
    #       and just needs MEM_VAL_B{0..3} to gate the val-byte position.
    #   (b) carry case (hi==15, lo+byte_off>=16): emit ADDR_KEY+32+((b1_lo+1)&0xF)
    #       AND subtract the (a) unit's contribution. This is the
    #       (hi==15, lo, byte_off, b1_lo) combo — bounded by 16*4*16 = 1024.

    # (a) Common case: 4 byte_off × 16 b1_lo = 64 units.
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for b1_lo in range(16):
            ffn.W_up[unit, gate_dim] = S
            ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
            ffn.b_up[unit] = -S * 1.5
            ffn.b_gate[unit] = 1.0
            ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = 2.0 / S
            unit += 1

    # (b) Carry-correction: when hi==15 AND (lo + byte_off) >= 16:
    #     - subtract from ADDR_KEY+32+b1_lo (cancel (a))
    #     - add to ADDR_KEY+32+((b1_lo+1)&0xF)
    # The trigger is a 4-way AND: MEM_VAL_B{byte_off} + ADDR_B0_HI[15] +
    # ADDR_B0_LO[lo] + ADDR_B1_LO[b1_lo], for lo such that lo+byte_off>=16.
    # That set of lo values is: byte_off=0 → none; byte_off=1 → {15};
    # byte_off=2 → {14,15}; byte_off=3 → {13,14,15}.
    # 4 byte_off × variable × 16 b1_lo = 0+1+2+3 = 6 × 16 = 96 units.
    carry_los = {
        0: [],
        1: [15],
        2: [14, 15],
        3: [13, 14, 15],
    }
    for byte_off in range(4):
        gate_dim = MEM_VAL_DIMS[byte_off]
        for lo in carry_los[byte_off]:
            for b1_lo in range(16):
                ffn.W_up[unit, gate_dim] = S
                ffn.W_up[unit, BD.ADDR_B0_HI + 15] = S
                ffn.W_up[unit, BD.ADDR_B0_LO + lo] = S
                ffn.W_up[unit, BD.ADDR_B1_LO + b1_lo] = S
                ffn.b_up[unit] = -S * 3.5
                ffn.b_gate[unit] = 1.0
                # Cancel the common-case write at b1_lo, add at (b1_lo+1)&0xF.
                ffn.W_down[BD.ADDR_KEY + 32 + b1_lo, unit] = -2.0 / S
                ffn.W_down[BD.ADDR_KEY + 32 + ((b1_lo + 1) & 0xF), unit] = 2.0 / S
                unit += 1

    return unit


def make_layer14_addr_key_neural_decode_op(enable: bool = False) -> Operation:
    """L14 FFN: BLOG_SPEC.md:830 neural ADDR_KEY decode at MEM val byte positions.

    Phase 0 of the V2 ADDR_KEY migration (see
    ``docs/V2_ADDR_KEY_NEURAL_DECODE_PLAN.md``).  Computes the same
    per-val-byte ADDR_KEY one-hot encoding that
    ``NeuralVMEmbedding._inject_mem_metadata`` produces today, baked
    into FFN weights instead of a Python loop.

    Per BLOG_SPEC.md:830:

        First the simple case key value retrieval, we want an exact
        match for the key so we take the binary key break it up into
        bytes or nibbles, perform an equality check for each byte or
        nibble, then in a subsequent layer we perform a logical AND
        over those results.

    The "binary key" here is the MEM section's 32-bit address.  L13's
    ``_set_layer13_mem_addr_gather`` already gathers the first 3 addr
    bytes to MEM val byte positions as 4-bit one-hot nibbles
    (``ADDR_B{0,1,2}_LO/HI``).  This op takes those nibbles plus the
    ``MEM_VAL_B{0..3}`` flags (= byte_off) and emits the
    ``ADDR_KEY[lo, 16+hi, 32+top]`` one-hot encoding of
    ``byte_addr = addr + byte_off`` (with carry handling).

    Gating: ``enable=False`` (default) → bake is a no-op.  Existing
    tests stay byte-identical.  When flipped to True (in a follow-up
    PR), the bake must produce identical ADDR_KEY values to
    ``_inject_mem_metadata`` on all (addr_b0_lo, addr_b0_hi,
    addr_b1_lo, byte_off) inputs.

    Pinned to ``layer_idx=14`` via ``kind="block"``.  Shares the FFN
    unit counter ``block.ffn._l14_unit_counter`` with the other L14
    cleanup ops.  Phase 14.5: runs AFTER
    ``layer14_clear_addr_key_pollution`` (phase 14.2) so the decode
    output is authoritative (not pre-cleared away).

    Total FFN units consumed when enabled: ~1184
    (1024 lo+hi + 64 common-top + 96 carry-top).  When disabled, 0.
    """
    def bake(block, dim_positions, S):
        if not enable:
            return
        ffn = block.ffn
        start_unit = getattr(ffn, "_l14_unit_counter", 0)
        next_unit = _bake_addr_key_neural_decode(
            ffn, dim_positions, S, start_unit=start_unit
        )
        ffn._l14_unit_counter = next_unit

    return Operation(
        name="layer14_addr_key_neural_decode",
        phase=14.5,
        reads={"MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
               "ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO",
               "CONST"},
        writes={"ADDR_KEY"},
        kind="block",
        bake_fn=bake,
        declarative_bake_fn=bake,
        declarative_authority="spec_generated",
        layer_idx=14,
        migrated=True,
        smoke_tests={"all"},
        spec_section="BLOG_SPEC.md#memory",
    )
