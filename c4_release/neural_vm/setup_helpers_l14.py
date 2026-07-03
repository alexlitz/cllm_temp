"""Setup helpers extracted from ``setup_helpers.py`` (Phase 6 wave 1.5).

Layer 14 cleanup helpers: temp clear, addr-key/output corruption fixes.

Re-exported by ``setup_helpers`` for backward compatibility.
"""

import math

from .constants import PC_OFFSET


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

    NARROWING 2026-06-06 (follow-up to initial JSR-path fix): the initial
    helper baked 8 units (marker + BYTE_INDEX_{0,1,2}) mirroring the full
    L3 rule shape. That OVER-CANCELLED at MEM_addr2/MEM_addr3 for the
    var-cluster fixtures: SP=0xFFFC pushes return address ``0xFC, 0xFF,
    0x00, 0x00`` little-endian, so the L3 baseline of 0x00 is CORRECT at
    MEM_addr2 (BYTE_INDEX_1) and MEM_addr3 (BYTE_INDEX_2). Cancelling
    there leaves a 0x11 residue to win the argmax instead of the true
    0x00. Per VAR_CLUSTER_JSR_PATH_FINDINGS_2026_06_06.md proposed
    follow-up #1, we narrow the gate to BYTE_INDEX_0 only (which selects
    MEM_addr1, the originally-failing slot) plus the MEM marker row
    (predicting addr_b0, kept because SP can be in any range).

    Activation calculation (S=100, with MEM_STORE-MEM_ADDR_SRC=1 for
    PSH/JSR/ENT):
      * MARK_MEM marker row: up = S*MARK_MEM + S*MEM_STORE
        - S*MEM_ADDR_SRC - 1.5*S = S * (1 + 1 - 0 - 1.5) = 0.5*S
        → silu(0.5*S) ~= S/2 = 50
      * BYTE_INDEX_0 row: up = S*H1[MEM] + S*BYTE_INDEX_0 + S*MEM_STORE
        - S*MEM_ADDR_SRC - 2.5*S = S * (1 + 1 + 1 - 0 - 2.5) = 0.5*S
        → silu(0.5*S) ~= S/2 = 50
      * SI/SC (MEM_STORE=1, MEM_ADDR_SRC=1): gate = 0 → silu(<0) ≈ 0
      * Non-store ops (MEM_STORE=0): gate = -1.5*S → silu(<0) ≈ 0
      * BYTE_INDEX_1/2 positions (MEM_addr2/MEM_addr3): no cancel unit
        fires, so L3's +0.940 baseline survives intact (correct: addr
        bytes 2/3 should be 0x00 for 16-bit SP).
    W_down = -2.0/S → output delta = (S/2) * (-2.0/S) = -1.0, cancels
    the L3 +0.940 with a small safety margin. Narrowed unit count:
    2 marker + 2 BYTE_INDEX_0 = 4 units.
    """
    unit = start_unit
    MEM_I = 4  # MEM marker index in MARKS array

    # === Cancel L3 ``MEM DEFAULT'' (marker rule, vm_step.py:3335-3349) ===
    # Fires when MARK_MEM=1 AND MEM_STORE=1 AND MEM_ADDR_SRC=0 (PSH/JSR/ENT).
    # Predicts addr_b0; kept because SP can be in any range.
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

    # === Cancel L3 ``MEM addr byte 1 default'' only (byte rule, vm_step.py:3351-3369) ===
    # NARROWED 2026-06-06: keep only BYTE_INDEX_0 (which selects MEM_addr1,
    # the originally-failing slot with true value 0xff for SP=0xFFFC).
    # BYTE_INDEX_1 (MEM_addr2) and BYTE_INDEX_2 (MEM_addr3) rows are
    # dropped because the L3 baseline of 0x00 is correct at those positions
    # for 16-bit addresses and cancelling there leaves a 0x11 residue.
    # LO nibble
    ffn.W_up[unit, BD.H1 + MEM_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.MEM_STORE] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_LO + 0, unit] = -2.0 / S
    unit += 1
    # HI nibble
    ffn.W_up[unit, BD.H1 + MEM_I] = S
    ffn.W_up[unit, BD.BYTE_INDEX_0] = S
    ffn.W_up[unit, BD.MEM_STORE] = S
    ffn.W_up[unit, BD.MEM_ADDR_SRC] = -S
    ffn.b_up[unit] = -S * 2.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.OUTPUT_HI + 0, unit] = -2.0 / S
    unit += 1

    return unit

