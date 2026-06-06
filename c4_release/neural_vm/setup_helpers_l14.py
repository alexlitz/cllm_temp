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



def _set_layer14_mem_addr_src_default_suppress(ffn, S, BD, start_unit=0):
    """L14 FFN: Cancel the L3 ``mem_byte_0_default`` baseline at SI/SC stores.

    BUG FIX 2026-06-05 (var-cluster follow-up): The L3 FFN ``MEM DEFAULT``
    rules (vm_step.py:3335-3369) unconditionally write +2.0/S to
    OUTPUT_LO[0]/OUTPUT_HI[0] at the MEM marker row and at the three MEM
    addr-byte query rows (H1[MEM]+BYTE_INDEX_{0,1,2}). Empirically each
    rule contributes ~+0.940 to the corresponding nibble dim. The intent
    of the L3 default is to predict ``0x00`` for non-store ops (where the
    MEM section is unused) and PSH/JSR/ENT stores (where the address is
    SP and bytes 1-3 happen to be 0 for small stacks).

    For SI/SC stores (``MEM_ADDR_SRC=1``) the addr comes from STACK0,
    which can have ANY byte value -- in particular the var-cluster tests
    (var_simple_0, if_var_0, var_three_0) write to addresses like
    0xFFFC where byte 1 is 0xff. The +0.940 baseline at OUTPUT_LO[0]/
    OUTPUT_HI[0] biases the argmax toward 0x00 and competes with L14's
    addr-emission writes, blocking the correct 0xff prediction.

    The brief identified the fix as a ``gated step-0 MEM_ADDR_SRC=0
    predicate on the L3 mem_byte_0_default rule''. ``MEM_ADDR_SRC`` is
    set at the MEM marker by L6 head 6 and broadcast to the MEM byte
    positions by L7 head 7, so it is not available at L3 itself. This
    L14 cleanup achieves the same algebra by SUBTRACTING the +0.940
    baseline at SI/SC store positions (``MEM_ADDR_SRC=1`` AND MARK_MEM /
    H1[MEM]+BYTE_INDEX_K). For non-SI/SC paths ``MEM_ADDR_SRC=0`` so the
    cancel does not fire and the L3 baseline survives intact.

    Activation calculation (S=100):
      * MARK_MEM marker row: up = S*MARK_MEM + S*MEM_ADDR_SRC - 1.5*S
        = S * (1 + 1 - 1.5) = 0.5*S → silu(0.5*S) ~= S/2 = 50
      * MEM byte rows: up = S*H1[MEM] + S*BYTE_INDEX_K + S*MEM_ADDR_SRC
        - 2.5*S = S * (1 + 1 + 1 - 2.5) = 0.5*S → silu(0.5*S) ~= S/2 = 50

    W_down = -2.0/S → output delta = (S/2) * (-2.0/S) = -1.0, which cancels
    the L3 +0.940 with a small safety margin. Mirrors the L3 rule shape
    (matching unit count: 2 marker + 6 byte = 8 units).
    """
    unit = start_unit
    MEM_I = 4  # MEM marker index in MARKS array

    # === Cancel L3 ``MEM DEFAULT'' (marker rule, vm_step.py:3335-3349) ===
    # Fires when MARK_MEM=1 AND MEM_ADDR_SRC=1.
    # LO nibble
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # cancel L3 +0.940
    unit += 1
    # HI nibble
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
    unit += 1

    # === Cancel L3 ``MEM addr bytes 1-3 default'' (byte rule, vm_step.py:3351-3369) ===
    # Fires when H1[MEM]=1 AND BYTE_INDEX_K=1 AND MEM_ADDR_SRC=1
    # (K = 0, 1, 2 → predicts MEM addr bytes 1, 2, 3 respectively).
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # LO nibble
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.MEM_ADDR_SRC] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S
        unit += 1
        # HI nibble
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.MEM_ADDR_SRC] = S
        ffn.b_up[unit] = -S * 2.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
        unit += 1

    return unit


def _set_layer14_jsr_mem_default_suppress(ffn, S, BD, start_unit=0):
    """L14 FFN: Cancel the L3 ``mem_byte_0_default`` baseline at JSR/PSH/ENT.

    BUG FIX 2026-06-06 (JSR-path sibling of f4f9103d): ``f4f9103d`` cancels
    the L3 ``MEM DEFAULT'' +0.940 baseline at SI/SC stores (gated on
    ``MEM_ADDR_SRC=1``). The var-cluster failures, however, are NOT at
    SI/SC — they are at **JSR step 0**, where the call instruction pushes
    the return-address bytes into memory at SP (which sits at 0xFFFC for
    the var-cluster fixtures, so addr byte 1 = 0xff). The same wrong-
    direction L3 baseline also corrupts the PSH and ENT address bytes
    when SP is in high memory.

    JSR, PSH, and ENT all set ``MEM_STORE=1`` (via the L6 opcode-relay
    head 6, broadcast to MEM byte positions by L7 head 7) but have
    ``MEM_ADDR_SRC=0`` (the address source is SP, not STACK0). So the
    gate ``MEM_STORE AND NOT MEM_ADDR_SRC`` selects exactly the JSR /
    PSH / ENT store path and is disjoint from the SI/SC gate already
    handled by ``_set_layer14_mem_addr_src_default_suppress``. Non-store
    opcodes have ``MEM_STORE=0`` so this cancel does not fire and the
    L3 baseline (which is correct for IMM etc.) survives intact.

    Activation calculation (S=100, with MEM_STORE-MEM_ADDR_SRC=1 for
    PSH/JSR/ENT):
      * MARK_MEM marker row: up = S*MARK_MEM + S*MEM_STORE
        - S*MEM_ADDR_SRC - 1.5*S = S * (1 + 1 - 0 - 1.5) = 0.5*S
        → silu(0.5*S) ~= S/2 = 50
      * MEM byte rows: up = S*H1[MEM] + S*BYTE_INDEX_K + S*MEM_STORE
        - S*MEM_ADDR_SRC - 2.5*S = S * (1 + 1 + 1 - 0 - 2.5) = 0.5*S
        → silu(0.5*S) ~= S/2 = 50
      * SI/SC (MEM_STORE=1, MEM_ADDR_SRC=1): gate = 0 → silu(<0) ≈ 0
      * Non-store ops (MEM_STORE=0): gate = -1.5*S → silu(<0) ≈ 0
    W_down = -2.0/S → output delta = (S/2) * (-2.0/S) = -1.0, cancels
    the L3 +0.940 with a small safety margin. Mirrors the L3 rule shape
    exactly (matching unit count: 2 marker + 6 byte = 8 units).
    """
    unit = start_unit
    MEM_I = 4  # MEM marker index in MARKS array

    # === Cancel L3 ``MEM DEFAULT'' (marker rule, vm_step.py:3335-3349) ===
    # Fires when MARK_MEM=1 AND MEM_STORE=1 AND MEM_ADDR_SRC=0 (PSH/JSR/ENT).
    # LO nibble
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.W_up[unit, BD.MEM_STORE] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S  # cancel L3 +0.940
    unit += 1
    # HI nibble
    ffn.W_up[unit, BD.MARK_MEM] = S
    ffn.W_up[unit, BD.MEM_STORE] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
    ffn.b_up[unit] = -S * 1.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
    unit += 1

    # === Cancel L3 ``MEM addr bytes 1-3 default'' (byte rule, vm_step.py:3351-3369) ===
    # Fires when H1[MEM]=1 AND BYTE_INDEX_K=1 AND MEM_STORE=1 AND
    # MEM_ADDR_SRC=0 (K = 0, 1, 2 → MEM addr bytes 1, 2, 3 respectively).
    for byte_idx_dim in [BD.BYTE_INDEX_0, BD.BYTE_INDEX_1, BD.BYTE_INDEX_2]:
        # LO nibble
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.MEM_STORE] = S
        ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
        ffn.b_up[unit] = -S * 2.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S
        unit += 1
        # HI nibble
        ffn.W_up[unit, BD.H1 + MEM_I] = S
        ffn.W_up[unit, byte_idx_dim] = S
        ffn.W_up[unit, BD.MEM_STORE] = S
        ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
        ffn.b_up[unit] = -S * 2.5
        ffn.b_gate[unit] = 1.0
        ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
        unit += 1

    return unit

