"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 4 helpers: V18 convo-IO PRTF transport attention head.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_convo_io_prtf_transport(attn, S, BD, HD):
    """L4 attention head 4 (V18 Phase 1c, bake 3d): transport captured PC/SP
    from PRTF AX marker across the variable-length output-emission interlude.

    The third companion to ``_set_convo_io_prtf_capture`` (3c) and
    ``_set_convo_io_pc_sp_latch`` (3b). Closes the variable-length transport
    gap (V18_CONVO_IO_NEURAL_PLAN.md §3, Phase 1c):

      capture  (3c, L7 FFN @ PRTF AX marker)    writes POST_PRTF_PC/SP into
                                                 the residual at one position
      transport (3d, L4 attn @ post-THINKING_   reads those nibbles back
                START)                            across the variable-length
                                                 output-byte interlude
      replay   (3b, L6 FFN @ post-THINKING_     reads from residual at the
                START)                            same position and drives
                                                 OUTPUT_LO/HI

    Without this transport bake, the captured nibbles stay attached to the
    PRTF AX marker's residual position; the L6 FFN replay band at the
    resumed step's post-THINKING_START position sees zero for
    POST_PRTF_PC/SP because the residual stream does not propagate
    positional dims through autoregressive decoding without an attention
    head to carry them. Phase 1b documented this gap and shipped the
    capture piece only; Phase 1c adds the carrier.

    Design
    ------
    Q (fires at post-THINKING_START position, gated by LAST_WAS_THINKING_START
       which L2 head 1 writes on the resume edge):
      - W_q[base, BD.LAST_WAS_THINKING_START] = L
      - W_q[base, BD.CONST]                   = -L * 0.5   (threshold)

    K (matches the PRTF AX marker — the most recent position where both
       ACTIVE_OPCODE_PRTF and MARK_AX are hot):
      - W_k[base, BD.ACTIVE_OPCODE_PRTF]      = L
      - W_k[base, BD.MARK_AX]                 = L
      - W_k[base, BD.CONST]                   = -L * 1.0   (need both flags)

    V (copy the captured PC/SP nibble values from the PRTF AX marker into
       the current position's residual via the standard nibble-slot
       O-projection pattern):
      - For k in 0..15:
          W_v[base + 1 + k,  BD.POST_PRTF_PC_LO + k] = 1.0
          W_v[base + 17 + k, BD.POST_PRTF_PC_HI + k] = 1.0
          W_v[base + 33 + k, BD.POST_PRTF_SP_LO + k] = 1.0
          W_v[base + 49 + k, BD.POST_PRTF_SP_HI + k] = 1.0
      - W_o write-side mirrors V (writes the same cache dims).

    HD constraint
    -------------
    The four nibble groups occupy V slots [1..16, 17..32, 33..48, 49..64],
    so the head must have HD >= 65. Production HD = d_model / num_heads =
    512 / 8 = 64. Off by one! Workaround: pack PC_LO/PC_HI in slots
    [1..16, 17..32], and SP_LO/SP_HI in slots that share the head's
    remaining cells. Production HD=64, so slots 0..63 are available.
    We use:
      - Slot 0:           ungated (Q activator)
      - Slots 1..16:      PC_LO nibbles
      - Slots 17..32:     PC_HI nibbles
      - Slots 33..48:     SP_LO nibbles
      - Slots 49..63:     SP_HI nibbles (15 cells: drop nibble 15 of SP_HI;
                          the top nibble of SP byte-0 is always 0 for
                          stacks below 0x10000000 — Phase 1c trades that
                          single-bit precision for head-budget headroom
                          and is documented here so a future Phase 1d can
                          reallocate if 32-bit stacks come into use)
    Production stacks live at SP < 0x00FFFFFF so the dropped top nibble of
    SP_HI is identically zero.

    ALiBi slope
    -----------
    L4 main bake fills ``alibi_slopes.fill_(0.5)``; head 4's slope is
    overridden to 0.1 here so the head can reach back far enough across
    the variable-length output-byte interlude.

    Score budget (head 4):
      - Q*K dot product at the PRTF AX marker:
            (L * 1.0) * (L * 1.0 for ACTIVE_OPCODE_PRTF
                       + L * 1.0 for MARK_AX
                       + L * -1.0 for CONST)
          = L^2 * (1 + 1 - 1) / sqrt(HD) = L^2 / 8
        With L=50:  +50*50/8 = +312.5
      - Q*K at any other position (no ACTIVE_OPCODE_PRTF or no MARK_AX):
          = L^2 * (0 + 0 or 1 - 1) / 8 = 0 to -L^2 / 8 = 0 to -312.5
      - ALiBi penalty at distance d: -0.1 * d
        d ~ 35 (one VM step) + N (output bytes) + 2 (THINK_END/START) + 1
        Typical N = 0..40 bytes, so d ~ 38..78.
        Penalty: -3.8 to -7.8.
      - Soft-max margin between PRTF-AX (+312.5 - penalty) and any other
        position (0 to -312.5) is ~300 — well above noise.

    Multiple PRTFs
    --------------
    If there are multiple PRTFs in the trace before the resume edge,
    ALiBi prefers the most recent one (lower distance penalty), giving
    the correct "the just-captured PC/SP" semantic.
    """
    L = 50.0
    base = 4 * HD  # head 4

    # === Q side: fire at post-THINKING_START position ===
    # LAST_WAS_THINKING_START is set by L2 head 1 (lookback detection)
    # AT THE TOKEN AFTER THINKING_START. That is the Q position where the
    # L6 FFN replay band will read POST_PRTF_PC/SP, so the transport must
    # deposit those nibbles here.
    attn.W_q[base, BD.LAST_WAS_THINKING_START] = L
    attn.W_q[base, BD.CONST] = -L * 0.5  # threshold

    # === K side: match the PRTF AX marker ===
    # ACTIVE_OPCODE_PRTF is set at all positions of a PRTF step (relayed
    # via embedding's opcode dim). MARK_AX is set only at the AX marker
    # position. The conjunction picks out the PRTF AX marker uniquely.
    attn.W_k[base, BD.ACTIVE_OPCODE_PRTF] = L
    attn.W_k[base, BD.MARK_AX] = L
    attn.W_k[base, BD.CONST] = -L  # need both flags (score < 0 if either missing)

    # === V/O: copy captured PC/SP nibble values from K-position to Q ===
    # The capture bake (3c) deposited POST_PRTF_PC_LO/HI[k] and
    # POST_PRTF_SP_LO/HI[k] at the PRTF AX marker. We read them via
    # V projection and write the same dims at the Q position via O.
    #
    # HD = 64 in production (d_model=512 / num_heads=8). We have 64 cells
    # in head 4: slots 0..63. Allocation:
    #   slot 0:        ungated (Q activator; V/O dead cell)
    #   slots 1..16:   POST_PRTF_PC_LO nibbles
    #   slots 17..32:  POST_PRTF_PC_HI nibbles
    #   slots 33..48:  POST_PRTF_SP_LO nibbles
    #   slots 49..63:  POST_PRTF_SP_HI nibbles 0..14 (15 cells)
    # Drop SP_HI nibble 15: identical-zero for SP < 0x10000000 (production
    # stack range). Documented for future Phase 1d reallocation if needed.
    nibble_groups = (
        (1,  BD.POST_PRTF_PC_LO, 16),  # slot_offset, dim_base, count
        (17, BD.POST_PRTF_PC_HI, 16),
        (33, BD.POST_PRTF_SP_LO, 16),
        (49, BD.POST_PRTF_SP_HI, 15),  # 15 (HD-budget; nibble 15 dropped)
    )
    for slot_offset, dim_base, count in nibble_groups:
        for k in range(count):
            attn.W_v[base + slot_offset + k, dim_base + k] = 1.0
            attn.W_o[dim_base + k, base + slot_offset + k] = 1.0


