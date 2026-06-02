"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 6 helpers: BZ/BNZ relay, tool-call/convo-IO state.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_bz_bnz_relay(attn, S, BD, HD):
    """L6 attention: Relay AX zero detection to PC marker for BZ/BNZ.

    Head 4 (ALiBi slope=5.0): At step N's PC marker, attend to step (N-1)'s
    AX byte 0 (identified by L1H1[AX]=1 AND L1H0[AX]=0 — same pattern as L3
    AX carry-forward). The AX byte 0 token IS the AX value going INTO step N
    (= the AX state at the moment BZ/BNZ is evaluated), encoded as
    EMBED_LO[lo_nibble]=1, EMBED_HI[hi_nibble]=1.

    Writes to CMP[2..5] at PC marker (CMP[0]=IS_JMP, CMP[1]=IS_EXIT reserved):
      CMP[2] = OP_BZ, CMP[3] = OP_BNZ
      CMP[4] = EMBED_LO[0] at AX byte 0 (1.0 if lo nibble is 0, i.e. AX_LO==0)
      CMP[5] = EMBED_HI[0] at AX byte 0 (1.0 if hi nibble is 0, i.e. AX_HI==0)
    Also copies FETCH_LO/HI → TEMP (branch target) from the same K position.

    Why not MARK_AX as K? The AX *marker* itself doesn't carry the AX value —
    only the AX byte 0 token does. L3 head 1 was supposed to write AX_CARRY
    at the AX marker, but it has a one-step delay (writes step N-1's pre-step
    AX, not step N's input AX). Reading EMBED_LO/HI directly at AX byte 0
    bypasses that delay: token convention is that step N's AX bytes ARE step
    N's output AX (= step N+1's input AX), so at step N+1's PC marker we
    attend to step N's AX byte 0 to read the AX state for BZ/BNZ.

    L6 FFN uses these for conditional PC override:
      BZ:  4-way AND (MARK_PC + CMP[2] + CMP[4] + CMP[5]) → branch if AX==0
      BNZ: 2 groups covering AX!=0 cases
    """
    L = 50.0
    base = 4 * HD
    AX_I = 1  # marker index for AX (matches PC_I=0, AX_I=1, ... in MARKS)

    attn.W_q[base, BD.MARK_PC] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX
    # FIX 2026-04-16: Gate on OP_BZ or OP_BNZ to prevent firing for other opcodes.
    # Without this gate, the head fires for ALL opcodes and writes FETCH to TEMP,
    # which overlaps OUTPUT_BYTE (dims 480-511), corrupting PC byte generation.
    # OP_BZ and OP_BNZ reach 5.0 via the L5 PC-marker opcode decode (see
    # vm_step.py: all-step PC-marker decode block).
    # Score budget:
    #   At PC marker without OP_BZ/BNZ: 50 - 65 = -15 (blocked)
    #   At PC marker with OP_BZ=5: 50 - 65 + 50 = 35 (fires)
    #   At PC marker with OP_BNZ=5: 50 - 65 + 50 = 35 (fires)
    attn.W_q[base, BD.CONST] = -L * 1.3  # Baseline penalty (stronger than before)
    attn.W_q[base, BD.OP_BZ] = L / 5.0  # OP_BZ=5 → contributes L
    attn.W_q[base, BD.OP_BNZ] = L / 5.0  # OP_BNZ=5 → contributes L
    # PC byte 0 also needs AX-zero flags so it can predict byte1(target) for
    # taken branches whose target PC is above 255.
    branch_pc_byte0_relay = 5
    attn.W_q[base + branch_pc_byte0_relay, BD.IS_BYTE] = 60.0
    attn.W_q[base + branch_pc_byte0_relay, BD.H1 + 0] = 60.0
    attn.W_q[base + branch_pc_byte0_relay, BD.BYTE_INDEX_0] = 60.0
    attn.W_q[base + branch_pc_byte0_relay, BD.MARK_PC] = -60.0
    # K: fire at AX byte 0 (L1H1[AX]=1 AND L1H0[AX]=0), NOT at AX marker.
    # FIX 2026-05-11: Changed K from MARK_AX → (L1H1+AX_I - L1H0+AX_I) so the
    # relay reads from step (N-1)'s AX byte 0 token (which contains step N's
    # input AX as a one-hot byte embedding) instead of the AX marker token
    # (where AX_CARRY is stale due to L3 head 1's one-step delay).
    attn.W_k[base, BD.L1H1 + AX_I] = L
    attn.W_k[base, BD.L1H0 + AX_I] = -L
    # Add K-side constant so Q·K creates a negative score when Q is blocked.
    # Without this, K[0]=0 at non-AX-byte-0 positions makes Q·K=0 regardless of
    # Q, allowing leakage. With K[CONST]=L, blocked Q (-15 at non-BZ/BNZ) gives
    # score = -15*50/sqrt(HD), reliably suppressing leak via softmax1.
    attn.W_k[base, BD.CONST] = L
    attn.W_k[base + branch_pc_byte0_relay, BD.L1H1 + AX_I] = L
    attn.W_k[base + branch_pc_byte0_relay, BD.L1H0 + AX_I] = -L

    # V: copy OP_BZ, OP_BNZ flags (also present at the AX byte 0 row via
    # residual propagation from L5 decode — but only PC marker has them
    # strong; we accept slight degradation since CMP[2]/[3] also gate the
    # downstream FFN at MARK_PC, which strongly differentiates).
    attn.W_v[base + 1, BD.OP_BZ] = 1.0
    attn.W_v[base + 2, BD.OP_BNZ] = 1.0
    # V: AX_LO_IS_ZERO and AX_HI_IS_ZERO directly from the byte's one-hot
    # nibble embedding. For AX byte 0 token b:
    #   EMBED_LO[0]=1 iff (b & 0xF)==0  → AX_LO nibble is zero
    #   EMBED_HI[0]=1 iff (b >> 4)==0   → AX_HI nibble is zero
    # Both true → byte 0 == 0 (assuming higher AX bytes are also 0, which we
    # approximate by checking byte 0; matches existing BZ semantics).
    attn.W_v[base + 3, BD.EMBED_LO + 0] = 1.0
    attn.W_v[base + 4, BD.EMBED_HI + 0] = 1.0
    # V: copy FETCH_LO/HI (branch target). Note: FETCH is set at PC marker
    # by L5, not at AX byte 0. Since K no longer attends to AX marker, we
    # need a different source. FETCH_LO/HI at the AX byte 0 position would
    # be zero (no L5 fetch there). Workaround: bake separate V from the
    # query's own residual using a self-relay — but that's complex. For now
    # leave FETCH→TEMP as no-op (= 0); the BZ taken-path PC override is
    # gated by CMP[2]+CMP[4]+CMP[5] but the TEMP target must come from a
    # different relay path. Acceptable: not_taken case (the immediate goal)
    # doesn't need TEMP; taken-path is addressed separately via L6 head 0
    # JMP relay analogue.
    # (Leaving V[base+5..36] zero so TEMP write is zero.)

    # O: write to CMP[2..5] at PC marker for FFN to use
    # (CMP[0] reserved for IS_JMP from head 0, CMP[1] for IS_EXIT from head 1)
    # Normalize OP_BZ/BNZ: raw ≈5 × 0.2 → CMP ≈1.0 (same scale as zero flags)
    # FIX 2026-05-11: Reverted CMP[4]/[5] scale to 1.0. The 2.0 boost from
    # 2026-04-29 compensated for softmax1 weight-splitting when K=MARK_AX,
    # but with the new K=L1H1[AX]-L1H0[AX] the attention concentrates ~95%
    # weight on the target AX byte 0 (single dominant position), so EMBED_LO[0]
    # / EMBED_HI[0] read essentially full strength (~1.0). Keeping 2.0 would
    # make CMP[5]≈2 and break BZ-override threshold math (4-way AND threshold
    # 3.5 trips spuriously when only CMP[5] fires).
    attn.W_o[BD.CMP + 2, base + 1] = 0.2  # OP_BZ at PC (normalized)
    attn.W_o[BD.CMP + 3, base + 2] = 0.2  # OP_BNZ at PC (normalized)
    attn.W_o[BD.CMP + 4, base + 3] = 1.0  # AX_LO_IS_ZERO at PC
    attn.W_o[BD.CMP + 5, base + 4] = 1.0  # AX_HI_IS_ZERO at PC
    # No TEMP write (V at slots 5..36 left zero — see note above).




def _set_tool_call_relay_head(attn, S, BD, HD):
    """L6 attention head 5: relay IO_IS_TOOL_CALL from AX → SE position.

    Same pattern as head 1 (EXIT relay):
    - Q: NEXT_SE (query at SE position) + -L*MARK_AX (block at AX)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_TOOL_CALL
    - O: write to CMP[2] (IS_TOOL_CALL relay)

    Distance from SE to AX = 28 tokens. With L=50, ALiBi slope=5:
    score = 50²*0.7*0.125 - 5*28 = 79 (strong).
    """
    L = 50.0
    base = 5 * HD  # head 5

    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    # V: copy IO_IS_TOOL_CALL flag
    attn.W_v[base + 1, BD.IO_IS_TOOL_CALL] = 1.0
    # O: write to CMP[2]
    attn.W_o[BD.CMP + 2, base + 1] = 1.0




def _set_tool_call_detection(ffn, S, BD):
    """L6 FFN addition: CMP[2] AND NEXT_SE → NEXT_TOOL_CALL.

    Same pattern as HALT detection (CMP[1] AND NEXT_SE → NEXT_HALT).
    When both CMP[2] (IS_TOOL_CALL relay) and NEXT_SE are active:
    - Set NEXT_TOOL_CALL (emit TOOL_CALL token)
    - Clear NEXT_SE (suppress STEP_END)

    CMP[2] ≈ 5 for I/O ops, ≈ 0 otherwise.
    Threshold 3.0: I/O (5+0.68=5.68 > 3) fires, inactive (0+0.68 < 3) doesn't.

    Note: When PUTCHAR activates this, the AX→OUTPUT routing from
    _set_io_putchar_routing still works (it runs in parallel at L6 FFN).
    The output byte is produced by the model's weights as usual; the
    TOOL_CALL token just signals the runner to dispatch via tool_handler.

    Starts at unit 1300 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1300

    ffn.W_up[unit, BD.CMP + 2] = S
    ffn.W_up[unit, BD.NEXT_SE] = S
    ffn.b_up[unit] = -S * 3.0
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.NEXT_TOOL_CALL, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # clear NEXT_SE → no STEP_END




def _set_conversational_io_relay_heads(attn, S, BD, HD):
    """L6 attention heads 4-5: relay IO_IS_PRTF and IO_IS_READ from AX → SE.

    IMPORTANT: Changed from heads 6-7 to heads 4-5 to avoid conflict with
    _set_opcode_relay_head() which uses head 6 for PSH/ADJ/pop relay.

    Head 4: Relay IO_IS_PRTF
    - Q: NEXT_SE (query at SE position), gated by ACTIVE_OPCODE_PRTF
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_PRTF
    - O: write to CMP + 5 (changed from CMP[3] to avoid pop group conflict)

    FIX 2026-04-16: Gate head 4 with ACTIVE_OPCODE_PRTF to avoid conflict with
    BZ/BNZ relay which also uses head 4. Without this gate, the BZ/BNZ's
    W_q[base, CONST] = -65 penalty blocks PRTF relay (CONST=1.0 at all positions).
    Adding W_q[base, ACTIVE_OPCODE_PRTF] = +65 cancels the penalty when PRTF active.

    Head 5: Relay IO_IS_READ
    - Q: NEXT_SE (query at SE position)
    - K: MARK_AX (attend to AX marker)
    - V: copy IO_IS_READ
    - O: write to CMP + 6 (changed from TEMP[0] to use dedicated CMP slot)

    Uses steep ALiBi slope (5.0) for both heads to overcome distance penalty.
    """
    L = 50.0

    # Head 4: PRTF relay
    # FIX 2026-04-16: Use V[37] instead of V[1] to avoid conflict with BZ/BNZ relay.
    # BZ/BNZ relay uses V[1] for OP_BZ and writes to CMP+2 via O[CMP+2, V[1]]=0.2.
    # When PRTF is active, IO_IS_PRTF=5.0 would get multiplied by that 0.2, causing
    # spurious CMP+2=1.0 which triggers ENT logic at STACK0 marker.
    base = 4 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    # FIX 2026-04-16: Gate with ACTIVE_OPCODE_PRTF to overcome BZ/BNZ CONST penalty
    # BZ/BNZ relay sets W_q[base, CONST] = -65, which blocks at SE (CONST=1.0).
    # When PRTF active, ACTIVE_OPCODE_PRTF=1.0, contributing +L=50 to cancel part of penalty.
    # Combined with NEXT_SE contribution, this makes Q positive at SE during PRTF.
    attn.W_q[base, BD.ACTIVE_OPCODE_PRTF] = L * 1.5  # +75 to overcome -65 CONST penalty
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 37, BD.IO_IS_PRTF] = 1.0  # V[37] avoids BZ/BNZ V[1] conflict
    attn.W_o[BD.CMP + 5, base + 37] = 1.0  # Use CMP[5] instead of CMP[3]

    # Head 5: READ relay
    base = 5 * HD
    attn.W_q[base, BD.NEXT_SE] = L
    attn.W_q[base, BD.MARK_AX] = -L  # block at AX marker
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_v[base + 1, BD.IO_IS_READ] = 1.0
    attn.W_o[BD.CMP + 6, base + 1] = 1.0  # Use CMP[6] instead of TEMP[0]




def _set_conversational_io_state_machine(ffn, S, BD):
    """L6 FFN addition: Start conversational I/O sequence when PRTF/READ detected.

    State transitions:
    1. Normal execution (IO_STATE=0)
    2. PRTF/READ detected → set NEXT_THINKING_END, IO_STATE=1
    3. After THINKING_END → generate output, IO_STATE=2
    4. After output complete → set NEXT_THINKING_START, IO_STATE=3
    5. After THINKING_START → resume normal, IO_STATE=0

    For now, we implement step 2: detect PRTF → start sequence.
    Steps 3-5 will be added in L13 FFN (state tracking across generation steps).

    Condition: CMP[5] (PRTF flag) AND NEXT_SE
    - Set NEXT_THINKING_END (emit </thinking> token)
    - Clear NEXT_SE (suppress STEP_END)
    - Set IO_STATE = 1 (begin I/O sequence)

    IMPORTANT: Changed from CMP[3] to CMP[5] and TEMP[0] to CMP[6] to avoid
    conflicts with _set_opcode_relay_head() which uses CMP[3] for pop group.

    Starts at unit 1400 to avoid overlap with _set_layer6_routing_ffn (units 0-1033).
    """
    unit = 1400

    # PRTF triggers thinking end
    # FIX 2026-04-16: CMP[5] can be ~5.0 (scaled by attention), not just ~1.0.
    # With CMP[5]=5.0, old threshold (-3S) was exceeded without NEXT_SE.
    # Fix: Gate on NEXT_SE so output is suppressed when NEXT_SE is not active.
    #
    # IMPORTANT: PureFFN uses `silu(up) * gate` where gate is LINEAR (no sigmoid)!
    # If b_gate is negative, gate can be negative when NEXT_SE=0, causing:
    #   silu(up) * negative_gate * negative_W_down = POSITIVE output (wrong!)
    # Fix: Use b_gate=0 so gate=10*NEXT_SE, which is 0 when NEXT_SE=0.
    ffn.W_up[unit, BD.CMP + 5] = S  # PRTF flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[5] > 0.5 (CMP[5]~5.0 → 500-50=450)
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate = 10*NEXT_SE (0 when inactive, 7 when active)
    ffn.b_gate[unit] = 0.0  # No bias - gate is 0 when NEXT_SE=0
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S  # emit THINKING_END
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S  # suppress STEP_END
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S  # set IO_STATE = 1
    unit += 1

    # READ triggers thinking end (similar pattern)
    ffn.W_up[unit, BD.CMP + 6] = S  # READ flag from relay head
    ffn.b_up[unit] = -S * 0.5  # Fire when CMP[6] > 0.5
    ffn.W_gate[unit, BD.NEXT_SE] = 10.0  # Gate on NEXT_SE
    ffn.b_gate[unit] = 0.0  # No bias
    ffn.W_down[BD.NEXT_THINKING_END, unit] = 2.0 / S
    ffn.W_down[BD.NEXT_SE, unit] = -2.0 / S
    ffn.W_down[BD.IO_STATE, unit] = 2.0 / S
    unit += 1




def _set_convo_io_pc_sp_latch(ffn, S, BD):
    """L6 FFN addition (V18 Phase 1, bake 3b): PC/SP latch replay band.

    When the resumed step's REG_PC / REG_SP value bytes are about to be
    decoded (gated by LAST_WAS_THINKING_START — the position immediately
    after the model emitted THINKING_START), this band drives the
    OUTPUT_LO/HI nibbles for the new step's PC and SP values.

    Phase 1b (this PR) wires the source dims against dedicated
    ``POST_PRTF_PC_LO/HI`` and ``POST_PRTF_SP_LO/HI`` cache slots, written
    by the capture-side bake ``_set_convo_io_prtf_capture`` at the PRTF AX
    marker (V18_CONVO_IO_NEURAL_PLAN.md §3b). The two cache pairs alias
    AX_FULL_LO/HI and AX_CARRY_LO/HI respectively — both of which are dead
    at the PRTF AX marker (PRTF never PSHes AX and never runs the ALU), so
    the alias is collision-free.

    Unit layout (L6 FFN, starting at 1402, ending at 1465):
      1402..1417  PC lo nibble replay (16 units, one per nibble value)
      1418..1433  PC hi nibble replay
      1434..1449  SP lo nibble replay
      1450..1465  SP hi nibble replay

    All units gate on LAST_WAS_THINKING_START so they are dormant in the
    normal 35-token step cycle and only fire on the resume edge.
    """
    # (source_lo, source_hi) pairs for the PC and SP replay groups.
    # Sources are populated by ``_set_convo_io_prtf_capture`` at the PRTF
    # AX marker. The aliasing on AX_FULL / AX_CARRY (both dead at the PRTF
    # AX marker — PRTF doesn't PSH AX and doesn't run the ALU) keeps
    # d_model=512 stable.
    replay_groups = [
        (BD.POST_PRTF_PC_LO, BD.POST_PRTF_PC_HI),  # PC nibbles (alias AX_FULL)
        (BD.POST_PRTF_SP_LO, BD.POST_PRTF_SP_HI),  # SP nibbles (alias AX_CARRY)
    ]
    unit = 1402
    for src_lo, src_hi in replay_groups:
        for src_dim, out_dim in ((src_lo, BD.OUTPUT_LO), (src_hi, BD.OUTPUT_HI)):
            for k in range(16):
                ffn.W_up[unit, BD.LAST_WAS_THINKING_START] = S
                ffn.b_up[unit] = -S * 0.5
                ffn.W_gate[unit, src_dim + k] = 1.0
                ffn.W_down[out_dim + k, unit] = 2.0 / S
                unit += 1


