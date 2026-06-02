"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 5 helpers: fetch attention + tool-call/convo-IO opcode decoding.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


# DEPRECATED — parity oracle only, no production use (Phase 6 wave 6A).
def _set_layer5_fetch(attn, S, BD, HD):
    """Layer 5 attention: fetch opcode/immediate through memory keys.

    Head 0: fetch immediate byte at address PC+1 (TEMP[0..31]).
    Head 1: fetch opcode byte at address PC (EMBED_LO/HI).

    Both heads match query nibbles against ADDR_KEY. This removes the
    forward-time address injection path; code bytes are expected to be
    available as memory entries keyed by address.
    """
    L = 20.0
    ADDR_L = 20.0

    def set_addr_key_k(base, top_slot_base=35):
        for kk in range(16):
            attn.W_k[base + kk, BD.ADDR_KEY + kk] = ADDR_L
            attn.W_k[base + 16 + kk, BD.ADDR_KEY + 16 + kk] = ADDR_L
            attn.W_k[base + top_slot_base + kk, BD.ADDR_KEY + 32 + kk] = ADDR_L

    def set_dynamic_top_q(base, top_slot_base=35):
        for kk in range(16):
            attn.W_q[base + top_slot_base + kk, BD.ADDR_KEY + 32 + kk] = ADDR_L

    def set_first_step_top0_q(base, top_slot_base=35):
        attn.W_q[base + top_slot_base, BD.CONST] = ADDR_L
        attn.W_q[base + top_slot_base, BD.HAS_SE] = -ADDR_L

    # Head 0: fetch immediate byte (address = PC+1)
    # Only fires on non-first steps (HAS_SE > 0). For first step, head 3 fetches
    # immediate at PC marker and head 4 relays to AX marker.
    # TEMP contains PC+1 at AX marker (computed by L4 FFN from relayed PC).
    base = 0 * HD
    # Q: low two nibbles from TEMP (PC+1)
    for k in range(16):
        attn.W_q[base + k, BD.TEMP + k] = ADDR_L
        attn.W_q[base + 16 + k, BD.TEMP + 16 + k] = ADDR_L
    set_dynamic_top_q(base)
    # third address nibble fixed to zero (for <= 255 code addresses),
    # gated to AX marker to avoid global leakage.
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    set_addr_key_k(base)

    # Anti-leakage gate: suppress head activity at non-AX positions.
    # Must overwhelm worst-case address match (~+50 from 33 addr dims).
    # Non-AX: Q=-500, K=5 → -312.5.  AX: Q=0 → 0 contribution.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses head 3 (PC marker) + head 4 (relay to AX)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy byte value nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write immediate to FETCH_LO/HI
    for k in range(16):
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 1.0

    # Head 1: fetch opcode byte (address = PC)
    # EMBED_LO/HI at AX marker contain the PC value (relayed by L4 attention).
    base = 1 * HD
    # Q: low two nibbles from EMBED (PC)
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = ADDR_L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = ADDR_L
    set_dynamic_top_q(base)
    attn.W_q[base + 32, BD.MARK_AX] = L

    # K: address nibbles from memory key space
    set_addr_key_k(base)

    # Anti-leakage gate: suppress head activity at non-AX positions.
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: only fire on non-first steps (when HAS_SE > 0)
    # First step uses Head 4 (fetches opcode to AX marker at PC_OFFSET)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: writes to OPCODE_BYTE_LO/HI (staging for opcode decode)
    # Uses separate dims from ALU_LO/HI to avoid residual collision with L7 operand gather
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 2: fetch opcode for first-step (PC marker → address PC_OFFSET - 2)
    # FIX 2026-04-12: PC_OFFSET points to immediate byte, opcode is 2 bytes before
    # On the first step (NOT HAS_SE), PC = PC_OFFSET.
    # Uses ADDR_KEY matching (same mechanism as head 1, but fires at PC marker not AX).
    from .constants import PC_OFFSET
    base = 2 * HD
    # Q: fires at PC marker when NOT HAS_SE, queries for address PC_OFFSET
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = ADDR_L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = ADDR_L  # hi nibble
    attn.W_q[base + 35 + ((PC_OFFSET >> 8) & 0xF), BD.CONST] = ADDR_L  # top nibble
    attn.W_q[base + 32, BD.MARK_PC] = L  # third nibble gate

    # K: match ADDR_KEY nibbles (same as head 1)
    set_addr_key_k(base)

    # Anti-leakage gate (same as head 1)
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # FIX 2026-05-11: Add HAS_SE blocker via dedicated gate dim (like head 5
    # uses HAS_SE_GATE5). Without this, head 2 fires at step 1+ PC marker too:
    # the Q[base, HAS_SE]=-L blocker zeroes out only the MARK_PC component of
    # Q[base], but Q[base+32, MARK_PC]=L and the address-nibble Q components
    # remain non-zero and produce a positive score against the matching code
    # byte's ADDR_KEY. Result: head 2 leaks OPCODE_BYTE_LO[lo of PC_OFFSET=IMM]
    # and OPCODE_BYTE_HI[0] (both at PC_OFFSET=2 which is IMM) onto the PC
    # marker on every step. That doubles OPCODE_BYTE_HI[0] (head 5 already
    # writes 1 there for any opcode with hi=0) and breaks the 3-way AND
    # threshold (MARK_PC + LO + HI - 2.5) in _set_opcode_decode_ffn, causing
    # OP_BZ/BNZ/JMP to fire spuriously at MARK_PC for non-matching opcodes,
    # and to fire 3x stronger when the opcode actually matches.
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # negative when HAS_SE=1
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0

    # V: copy opcode byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to OPCODE_BYTE_LO/HI at PC marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 3: fetch immediate (PC marker → address from FETCH_LO/HI = PC+1)
    # FIX 2026-04-29: Changed from fixed address (PC_OFFSET+1) to dynamic address
    # read from FETCH_LO/HI at PC marker. L4 FFN now writes PC+1 to FETCH dims,
    # so head 3 fetches from the correct address for all steps, not just the first.
    # This fixes BZ/BNZ branch targets and JMP targets for non-first steps.
    base = 3 * HD
    # Q: read dynamic address from FETCH_LO/HI (PC+1 computed by L4 FFN)
    # FETCH is one-hot: only one lo and one hi position are active (~0.73).
    for k in range(16):
        attn.W_q[base + k, BD.FETCH_LO + k] = ADDR_L
        attn.W_q[base + 16 + k, BD.FETCH_HI + k] = ADDR_L
    set_dynamic_top_q(base)
    set_first_step_top0_q(base)
    attn.W_q[base + 32, BD.MARK_PC] = L  # gate for PC marker only

    # K: match ADDR_KEY nibbles
    set_addr_key_k(base)

    # Anti-leakage gate: suppress at non-PC positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0

    # HAS_SE gate: REMOVED for JMP support at non-first steps.
    # Previously this blocked immediate fetch on non-first steps, but JMP needs FETCH
    # (immediate = jump target) at any step, not just first step.
    # FIX 2026-04-16: Removed HAS_SE gate to enable JMP at any step position.
    # The L6 FFN JMP override uses OP_JMP + FETCH to set PC to jump target.
    # HAS_SE_GATE = 34  # Disabled
    # attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # Disabled
    # attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0  # Disabled

    # V: copy immediate byte nibbles
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write to FETCH_LO/HI at PC marker for first-step immediate
    # AMPLIFY FETCH: Use 40.0 instead of 1.0 to compensate for attenuation during
    # Layer 6 Head 5 relay (FETCH gets attenuated 37x: 1.0→0.027, while OP_IMM
    # only gets attenuated 17x: 6.0→0.351). With 40.0, FETCH should reach ~1.0 at AX.
    # BUG FIX 2026-04-09 (part 3): Removed AX_CARRY output from L5 head 3.
    # AX_CARRY should only be written for JMP operations, but L5 head 3 fires for ALL
    # first-step ops. L6 head 2 now reads FETCH and writes to AX_CARRY (gated on OP_JMP).
    for k in range(16):
        # attn.W_o[BD.AX_CARRY_LO + k, base + 32 + k] = 1.0  # REMOVED
        # attn.W_o[BD.AX_CARRY_HI + k, base + 48 + k] = 1.0  # REMOVED
        attn.W_o[BD.FETCH_LO + k, base + 32 + k] = 40.0  # Amplified for relay
        attn.W_o[BD.FETCH_HI + k, base + 48 + k] = 40.0  # Amplified for relay

    # Head 4: Fetch opcode to AX marker for first-step (duplicate of Head 2)
    # Head 2 fetches opcode to PC marker, but opcode decode FFN runs at AX marker.
    # This head fetches the same opcode at address PC_OFFSET but writes to AX marker.
    # (Cannot relay from Head 2 since attention reads from input, not other heads' outputs.)
    base = 4 * HD
    # Q: fires at AX marker when NOT HAS_SE (first step), queries for address PC_OFFSET
    # Q: address PC_OFFSET (e.g., 2: ADDR_KEY_LO[2]=1, ADDR_KEY_HI[0]=1)
    attn.W_q[base + (PC_OFFSET & 0xF), BD.CONST] = ADDR_L  # lo nibble
    attn.W_q[base + 16 + ((PC_OFFSET >> 4) & 0xF), BD.CONST] = ADDR_L  # hi nibble
    attn.W_q[base + 35 + ((PC_OFFSET >> 8) & 0xF), BD.CONST] = ADDR_L  # top nibble
    attn.W_q[base + 32, BD.MARK_AX] = L  # third nibble gate
    # K: match ADDR_KEY nibbles (code byte addresses)
    set_addr_key_k(base)
    # Anti-leakage gate: suppress at non-AX positions
    GATE = 33
    attn.W_q[base + GATE, BD.MARK_AX] = 500.0
    attn.W_q[base + GATE, BD.CONST] = -500.0
    attn.W_k[base + GATE, BD.CONST] = 5.0
    # HAS_SE gate: only fire on first step (when HAS_SE = 0)
    # On first step: Q[HAS_SE]=0, Q[CONST]=-500 → score = -312.5 (blocks from wrong positions)
    # On non-first: Q[HAS_SE]=500, Q[CONST]=-500 → score = 0 (neutral)
    # Combined with anti-leakage: non-first + AX = blocked
    HAS_SE_GATE = 34
    attn.W_q[base + HAS_SE_GATE, BD.HAS_SE] = -500.0  # negative: block when HAS_SE > 0
    attn.W_k[base + HAS_SE_GATE, BD.CONST] = 5.0
    # V: copy opcode byte nibbles from matched code position
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    # O: write OPCODE_BYTE_LO/HI at AX marker
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 5: Fetch opcode byte to OPCODE_BYTE at PC marker on non-first steps.
    # This enables opcode decode at PC marker (via L5 FFN) for BZ/BNZ/LEV/EXIT
    # using dynamic PC from EMBED_LO/HI.
    # HAS_SE gate: only fires on non-first steps (Head 2 handles first step).
    base = 5 * HD
    for k in range(16):
        attn.W_q[base + k, BD.EMBED_LO + k] = ADDR_L
        attn.W_q[base + 16 + k, BD.EMBED_HI + k] = ADDR_L
    set_dynamic_top_q(base)
    attn.W_q[base + 32, BD.MARK_PC] = L
    set_addr_key_k(base)
    GATE5 = 33
    attn.W_q[base + GATE5, BD.MARK_PC] = 500.0
    attn.W_q[base + GATE5, BD.CONST] = -500.0
    attn.W_k[base + GATE5, BD.CONST] = 5.0
    HAS_SE_GATE5 = 34
    attn.W_q[base + HAS_SE_GATE5, BD.HAS_SE] = 500.0
    attn.W_q[base + HAS_SE_GATE5, BD.CONST] = -500.0
    attn.W_k[base + HAS_SE_GATE5, BD.CONST] = 5.0
    for k in range(16):
        attn.W_v[base + 32 + k, BD.CLEAN_EMBED_LO + k] = 1.0
        attn.W_v[base + 48 + k, BD.CLEAN_EMBED_HI + k] = 1.0
    for k in range(16):
        attn.W_o[BD.OPCODE_BYTE_LO + k, base + 32 + k] = 1.0
        attn.W_o[BD.OPCODE_BYTE_HI + k, base + 48 + k] = 1.0

    # Head 6: DELETED 2026-05-11.
    #
    # Previously: "Direct OP_* flag relay from CODE to PC marker (non-first
    # steps)". Marked DEPRECATED on 2026-04-13 because OP_* flags were
    # removed from CODE-byte token embeddings. After that removal the head
    # had no functional value on its own — V[base + slot] reads OP_<NAME>
    # from the matched CODE byte, but those dims are all 0 on byte tokens.
    #
    # Worse, the head's V slots [base+1 .. base+16] COLLIDE with the
    # ``_set_function_call_weights`` ENT-relay writes on the SAME L5 attn:
    #   attn5.W_v[6*HD + 1 + k, EMBED_LO + k] = 1.0  (BP→TEMP ENT relay)
    # On byte token N, EMBED_LO[N & 0xF] = 1.0. That value gets aliased
    # onto OP_LEA / OP_EXIT / OP_JMP / OP_JSR / OP_ADD / OP_SUB / ...
    # via the deprecated head 6 W_v rows. The deprecated head's Q/K fire
    # at the PC marker on every non-first step, so the byte-value bits of
    # the current PC-byte get written to those OP_* dims at PC. With BZ
    # opcode = 4, EMBED_LO[4] = 1 leaks into OP_ADD (slot 5), which the
    # branch_override_patch then treats as a "non-BZ opcode active"
    # blocker and silences the BZ taken-path override unit.
    #
    # OP_BZ/BNZ/LEV/EXIT/JMP at PC come from the L5 FFN all-step decode
    # (vm_step.py:3271-3285), and the remaining OP_* flags propagate via
    # the L6 opcode_relay_head (AX → PC/SP/STACK0/BP/MEM). Head 6 here
    # was redundant, so we zero it out to eliminate the collision.
    #
    # NOTE: Q[base + k]/K[base + k] (slots 0..32) of head 6 are still
    # written by _set_function_call_weights for the BP→TEMP ENT relay;
    # those rows write *different* dim positions (MARK_STACK0, MARK_BP,
    # EMBED_LO/HI vs EMBED_LO/HI, ADDR_KEY), and the two attentions are
    # gated by mutually exclusive markers (STACK0 vs PC). So leaving the
    # ENT-relay Q/K/V/O in place is safe; only the deprecated PC-marker
    # OP_* relay needs to go.

    # Head 7: DELETED 2026-05-11 alongside Head 6 — same DEPRECATED rationale.
    # Head 7 was the first-step (NOT HAS_SE) counterpart; the first-step
    # PC-marker OP_* flags are now written directly by the L5 FFN
    # first-step decodes (vm_step.py:3054-3249). No active L5 attn writer
    # collides with Head 7's V slots today, but we drop it for symmetry and
    # to prevent the same class of latent collision in the future.



def _set_tool_call_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode all I/O opcodes → IO_IS_TOOL_CALL.

    Same pattern as _set_opcode_decode_ffn: 2-way AND on OPCODE_BYTE_LO/HI
    nibbles, gated by MARK_AX. All 6 opcodes write to the same
    IO_IS_TOOL_CALL dim (combined flag, ≈5.0 when any is active, ≈0 otherwise).

    Starts at unit 400 to avoid conflict with existing opcode decode units.

    OPEN=30 (lo=14, hi=1), READ=31 (lo=15, hi=1),
    CLOS=32 (lo=0, hi=2), PRTF=33 (lo=1, hi=2),
    GETCHAR=64 (lo=0, hi=4), PUTCHAR=65 (lo=1, hi=4).
    """
    unit = 400
    io_opcodes = [
        (14, 1),  # OPEN = 30 = 0x1E
        (15, 1),  # READ = 31 = 0x1F
        (0, 2),  # CLOS = 32 = 0x20
        (1, 2),  # PRTF = 33 = 0x21
        (0, 4),  # GETCHAR = 64 = 0x40
        (1, 4),  # PUTCHAR = 65 = 0x41
    ]
    for lo, hi in io_opcodes:
        ffn.W_up[unit, BD.OPCODE_BYTE_LO + lo] = S
        ffn.W_up[unit, BD.OPCODE_BYTE_HI + hi] = S
        ffn.b_up[unit] = -S * 1.5  # both must be ~1
        ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
        ffn.W_down[BD.IO_IS_TOOL_CALL, unit] = 10.0 / S  # ≈5.0 when active
        unit += 1




def _set_conversational_io_opcode_decode(ffn, S, BD):
    """L5 FFN addition: decode PRTF and READ opcodes for conversational I/O mode.

    Detects PRTF (33) and READ (31) opcodes at AX marker and writes to
    separate flags for autoregressive I/O generation:
    - PRTF → IO_IS_PRTF ≈ 5.0  (AND ACTIVE_OPCODE_PRTF for downstream gates)
    - READ → IO_IS_READ ≈ 5.0  (AND ACTIVE_OPCODE_READ)

    This is separate from tool_call detection to enable different routing:
    - Tool call mode: PRTF/READ → TOOL_CALL token (runner dispatches)
    - Conversational I/O mode: PRTF/READ → autoregressive sequence
      (THINKING_END → output bytes → THINKING_START)

    2026-05-12 fix: the previous bake assumed ``ACTIVE_OPCODE_PRTF`` was
    populated by the embedding (Python ``set_active_opcode`` peek). That
    Python peek was retired in commit 16efeeb — no layer writes
    ``ACTIVE_OPCODE_PRTF`` autoregressively. Result: ``IO_IS_PRTF`` stayed
    at zero on every PRTF step, the L6 relay never fired, the L6 FFN
    state-machine never emitted ``NEXT_THINKING_END`` at NEXT_SE, and the
    model emitted plain ``STEP_END`` instead of ``THINKING_END`` for
    printf.

    New: decode the PRTF/READ opcode bytes directly via
    OPCODE_BYTE_LO/HI nibbles (the same pattern used by
    ``_set_tool_call_opcode_decode`` for ``IO_IS_TOOL_CALL``). Each
    opcode contributes one FFN unit per output flag, gated on MARK_AX so
    the write only lands at the AX marker. We also write
    ``ACTIVE_OPCODE_PRTF`` / ``ACTIVE_OPCODE_READ`` from the same units so
    downstream bakes that read those dims (e.g. L4 attn head 4 / L7 FFN
    capture / L6 attn head 4) see the activation at the AX position.

    Starts at unit 410 to avoid conflict with tool_call units (400-405).

    PRTF = 33 = 0x21 (lo=1, hi=2)
    READ = 31 = 0x1F (lo=15, hi=1)
    """
    unit = 410

    # PRTF (lo=1, hi=2): writes IO_IS_PRTF AND ACTIVE_OPCODE_PRTF at AX marker
    # Threshold: -S*1.5 so the 2-way AND (OPCODE_BYTE_LO[1]=1 +
    # OPCODE_BYTE_HI[2]=1 + MARK_AX gate) fires only at the PRTF AX marker.
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + 1] = S  # PRTF lo nibble
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + 2] = S  # PRTF hi nibble
    ffn.b_up[unit] = -S * 1.5  # both nibbles must be ~1
    ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
    ffn.W_down[BD.IO_IS_PRTF, unit] = 10.0 / S  # ≈5.0 when active
    # Also write ACTIVE_OPCODE_PRTF (≈1.0) so downstream bakes (L4 attn
    # head 4 transport K-side, L7 FFN capture, L6 attn head 4 Q-gate) see
    # the flag at the AX marker.
    ffn.W_down[BD.ACTIVE_OPCODE_PRTF, unit] = 2.0 / S  # ≈1.0
    unit += 1

    # READ (lo=15, hi=1): writes IO_IS_READ AND ACTIVE_OPCODE_READ at AX marker
    ffn.W_up[unit, BD.OPCODE_BYTE_LO + 15] = S  # READ lo nibble
    ffn.W_up[unit, BD.OPCODE_BYTE_HI + 1] = S  # READ hi nibble
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.MARK_AX] = 1.0  # only at AX marker
    ffn.W_down[BD.IO_IS_READ, unit] = 10.0 / S  # ≈5.0
    ffn.W_down[BD.ACTIVE_OPCODE_READ, unit] = 2.0 / S  # ≈1.0
    unit += 1



