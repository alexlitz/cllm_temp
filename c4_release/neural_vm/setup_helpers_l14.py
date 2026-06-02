"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 14 cleanup helpers: temp clear, addr-key/output corruption fixes.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


def _set_layer14_clear_addsub_temp_negative_residue(ffn, S, BD, start_unit=0):
    """L14 FFN: clamp negative ADD/SUB relay residue before late tail rules.

    ``TEMP[8]`` / ``TEMP[9]`` are positive ADD/SUB byte relays. On unrelated
    rows they can carry tiny negative attention dust (around 1e-22). The late
    result tail intentionally uses very large negative blockers on these lanes,
    so that dust can become a huge false-positive activation. This clamp is
    sign-selective: positive relays are left unchanged, while negative residue
    is lifted back to approximately zero.
    """
    unit = start_unit

    for temp_offset in (8, 9):
        ffn.W_up[unit, BD.TEMP + temp_offset] = -S
        ffn.W_gate[unit, BD.CONST] = 1.0
        # For small negative x, silu(-S*x) ~= -S*x/2, so 2/S cancels x.
        ffn.W_down[BD.TEMP + temp_offset, unit] = 2.0 / S
        unit += 1

    return unit



def _set_layer14_add_byte1_high_zero_cleanup(ffn, S, BD, start_unit=0):
    """L14 FFN: keep ADD byte-1 high nibble at zero before tail repair.

    The base layer-14 FFN can leave large positive OUTPUT_HI[1..15] residue on
    ADD AX byte-0 rows, where the next emitted token is AX byte 1. The final
    dependency tail treats that residue as a wide-MUL byte-preserve signature.
    ADD slice operands here still require byte-1 high nibble zero, so clear the
    nonzero high-nibble band while leaving the computed low nibble alone.
    """
    unit = start_unit
    AX_I = 1

    ffn.W_up[unit, BD.IS_BYTE] = S
    ffn.W_up[unit, BD.H1 + AX_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.TEMP + 8] = S
    ffn.W_up[unit, BD.TEMP + 9] = -S * 10
    # BP-relative LEA local addresses also use the ADD relay, but their byte 1
    # is 0xff. After the L14 boundary guard amplifies structural byte rows, a
    # normal-sized blocker is not enough to stop this cleanup from forcing the
    # high nibble to zero.
    ffn.W_up[unit, BD.AX_CARRY_HI + 15] = -S * 10_000_000
    ffn.W_up[unit, BD.BYTE_INDEX_1] = -S * 10
    ffn.W_up[unit, BD.BYTE_INDEX_2] = -S * 10
    ffn.W_up[unit, BD.BYTE_INDEX_3] = -S * 10
    ffn.W_up[unit, BD.MARK_AX] = -S * 100
    ffn.b_up[unit] = -S * 3.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 50.0 / S
    for nonzero in range(1, 16):
        ffn.W_down[BD.OUTPUT_HI + nonzero, unit] = -5000.0 / S
    unit += 1

    return unit



def _set_layer14_clear_addr_key_pollution(ffn, S, BD, start_unit=0):
    """L14 FFN: Clear ADDR_KEY pollution at non-MEM, non-marker positions.

    BUG FIX 2026-04-16: ADDR_KEY dims (206-253) are aliased with ADDR_B*_HI.
    L9 attention writes to ADDR_B*_HI for address gathering, which pollutes
    ADDR_KEY at non-MEM positions. This causes L15 to attend to wrong positions.

    Solution: Clear ADDR_KEY at positions that are:
    - NOT MEM value bytes (MEM_VAL_B* = 0)
    - NOT register markers where ADDR_B*_HI is needed for L15 queries
      (PC marker for LEV return_addr, BP marker for LEV saved_bp,
       AX marker for LI/LC, STACK0 marker for stack read)

    Pattern: Fire when NOT at MEM value position AND NOT at query markers.
    - W_up: Large negative weights for MEM_VAL_B* and MARK_* flags
    - b_up: Positive bias (fires when no flags present)
    - W_down: Write negative value to cancel ADDR_KEY pollution
    """
    unit = start_unit

    # Large value to suppress firing at MEM and marker positions
    suppress = S * 100  # When flag = 1.0, adds -100*S to activation

    # Clear all 48 ADDR_KEY dims at non-MEM, non-marker positions
    for k in range(48):  # ADDR_KEY is 48 dims (206-253)
        # Suppress at MEM value positions (any of B0/B1/B2/B3)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress

        # Suppress at register markers where ADDR_B*_HI is used for L15 queries
        ffn.W_up[unit, BD.MARK_PC] = -suppress  # LEV return_addr lookup
        ffn.W_up[unit, BD.MARK_BP] = -suppress  # LEV saved_bp lookup
        ffn.W_up[unit, BD.MARK_AX] = -suppress  # LI/LC address lookup
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress  # Stack read
        # FIX 2026-04-16: Also suppress at SP marker during LEV
        # SP marker needs ADDR_B0 for SP = BP + 16 computation
        ffn.W_up[unit, BD.MARK_SP] = -suppress

        # Positive bias to fire at non-MEM, non-marker positions
        ffn.b_up[unit] = S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write to cancel pollution and bring ADDR_KEY to 0
        # FIX 2026-04-16: Changed from -200/S (=-100 output) to -4/S (~=-1.4 output).
        # The original -100 clearing caused negative Q × negative K = positive score
        # at non-target positions in L15 LEV heads. Clearing to ~0 avoids this issue
        # while still preventing false address matches (0 × anything = 0).
        # The pollution to clear is small (typically ~1-2 from L9 ADDR_B*_HI writes),
        # so a small negative value is sufficient.
        ffn.W_down[BD.ADDR_KEY + k, unit] = -4.0 / S
        unit += 1

    return unit




def _set_layer14_clear_output_corruption(ffn, S, BD, start_unit=0):
    """L14 FFN: Fix OUTPUT at STACK0 byte positions (bytes 1-3 = 0).

    BUG FIX 2026-04-16: L14 attention V[0] cancelation and CLEAN_EMBED copying
    corrupts OUTPUT at non-MEM query positions (like STACK0 bytes). Even with
    strong Q suppression, softmax normalization ensures some attention weight
    distributes to source positions, causing OUTPUT to have wrong argmax.

    Solution: At STACK0 byte positions (d=5-9 from BP), boost OUTPUT_LO[0] and
    OUTPUT_HI[0] to ensure they win the argmax. This makes bytes 1-3 of STACK0
    (and similar) output 0, which is correct for return addresses < 256.

    Note: This approach assumes return_addr fits in 1 byte. For larger addresses,
    we'd need to compute bytes 1-3 properly (currently they'd be wrong).
    """
    unit = start_unit

    # Suppression value (prevents firing at MEM and register markers)
    suppress = S * 100
    PC_I = 0
    AX_I = 1
    SP_I = 2
    BP_I = 3  # Index for BP marker in threshold dims
    MEM_I = 4

    # Only boost OUTPUT_LO[0] and OUTPUT_HI[0]
    for k in [0, 16]:  # 0 = OUTPUT_LO[0], 16 = OUTPUT_HI[0]
        output_dim = BD.OUTPUT_LO if k == 0 else BD.OUTPUT_HI
        band_dim = output_dim

        # Fire at STACK0 byte area (d=5-9 from BP marker, excludes marker itself)
        # Use H4[BP] (d≤9.5) AND NOT H1[BP] (d>4.5) to select d ∈ (4.5, 9.5]
        ffn.W_up[unit, BD.H4 + BP_I] = S
        ffn.W_up[unit, BD.H1 + BP_I] = -S * 20

        # Suppress at MEM value byte positions (legitimate L14 targets)
        ffn.W_up[unit, BD.MEM_VAL_B0] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B1] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B2] = -suppress
        ffn.W_up[unit, BD.MEM_VAL_B3] = -suppress
        ffn.W_up[unit, BD.H1 + MEM_I] = -suppress
        ffn.W_up[unit, BD.H3 + MEM_I] = -suppress

        # Suppress at register markers where OUTPUT is needed
        ffn.W_up[unit, BD.MARK_PC] = -suppress
        ffn.W_up[unit, BD.MARK_AX] = -suppress
        ffn.W_up[unit, BD.MARK_SP] = -suppress
        ffn.W_up[unit, BD.MARK_BP] = -suppress
        ffn.W_up[unit, BD.MARK_MEM] = -suppress
        ffn.W_up[unit, BD.MARK_STACK0] = -suppress
        # Register byte rows can also sit inside the broad H4[BP] window.
        # This cleanup is only for STACK0-like rows; do not let it erase
        # actual PC/AX/SP byte outputs such as IMM AX byte 1.
        ffn.W_up[unit, BD.H1 + PC_I] = -suppress
        ffn.W_up[unit, BD.H1 + AX_I] = -suppress
        ffn.W_up[unit, BD.H1 + SP_I] = -suppress
        # PSH writes the pushed AX value through the dedicated STACK0 byte
        # path. The old zero-default cleanup is only safe for non-PSH STACK0
        # bytes; otherwise multi-byte addresses like 0x200 lose byte 1.
        ffn.W_up[unit, BD.PSH_AT_SP] = -suppress
        # Pop-group arithmetic/comparison ops also need the real STACK0 bytes
        # available downstream. L6 relays that group as CMP[3] onto STACK0
        # byte positions; do not force those bytes back to zero.
        ffn.W_up[unit, BD.CMP + 3] = -suppress
        # ADD/SUB byte rows can carry large signed OUTPUT residue after the
        # byte correction post-ops. The low-nibble zero-default repair is only
        # for STACK0 cleanup; do not let it zero AX arithmetic byte results.
        if k == 0:
            ffn.W_up[unit, BD.TEMP + 8] = -S * 1e20
            ffn.W_up[unit, BD.TEMP + 9] = -S * 1e20
        # Suppress at BYTE_INDEX_3 positions - byte 3's OUTPUT should predict
        # the NEXT marker (MEM), not force byte value 0.
        ffn.W_up[unit, BD.BYTE_INDEX_3] = -suppress

        # Preserve an already-computed nonzero nibble. This cleanup is a
        # zero-default repair; it should not turn a supported 0x02/0x03 stack
        # byte back into 0x00 after L10 has produced the arithmetic result.
        for nonzero in range(1, 16):
            ffn.W_up[unit, band_dim + nonzero] = -S * 2

        # Bias for activation
        ffn.b_up[unit] = -S * 0.5

        # Gate unconditionally
        ffn.W_gate[unit, BD.CONST] = 1.0

        # Write large POSITIVE value to OUTPUT[0] to make it the argmax winner
        # At d=6-9, activation ≈ 3.5S to 0.5S, so output ≈ S * 50/S = 50
        # This overcomes L14's corruption (~2-3) by a large margin.
        ffn.W_down[output_dim, unit] = 50.0 / S

        unit += 1

    # JSR's STACK0 marker emits the return address byte. L6 has already
    # routed the low/high nibbles from AX_CARRY, but L14 MEM-generation heads
    # cancel the L3 zero-default high nibble at the marker. Reassert HI[0] only
    # for the current JSR STACK0 marker; bytes are handled by the units above.
    ffn.W_up[unit, BD.OP_JSR] = S / 5
    ffn.W_up[unit, BD.MARK_STACK0] = S
    ffn.W_up[unit, BD.IS_BYTE] = -S * 10
    ffn.W_up[unit, BD.MARK_PC] = -suppress
    ffn.W_up[unit, BD.MARK_AX] = -suppress
    ffn.W_up[unit, BD.MARK_SP] = -suppress
    ffn.W_up[unit, BD.MARK_BP] = -suppress
    ffn.W_up[unit, BD.MARK_MEM] = -suppress
    ffn.b_up[unit] = -S * 1.5
    ffn.W_gate[unit, BD.CONST] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = 50.0 / S
    unit += 1

    return unit


