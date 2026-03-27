"""
Efficient ALU operations using pure FFN weight baking.

All computation is baked into FFN weights at initialization.
Forward pass is standard SwiGLU - no runtime integer ops.

Building blocks:
- Cancel pair (2 units): output = x * y
- Step pair (2 units): output = step(sum >= threshold)
- Clear pair (2 units): output = 0 (clears a slot)

Temporary dimensions (450-499) used for intermediate results between layers.
"""

import torch
import torch.nn as nn

# Temporary dimension allocation
TEMP_BASE = 450
# Decoded scalar values
A_LO = TEMP_BASE + 0
A_HI = TEMP_BASE + 1
B_LO = TEMP_BASE + 2
B_HI = TEMP_BASE + 3
# ADD/SUB intermediates
RAW_SUM_LO = TEMP_BASE + 4
RAW_SUM_HI = TEMP_BASE + 5
GEN_LO = TEMP_BASE + 6
GEN_HI = TEMP_BASE + 7
CARRY_TO_HI = TEMP_BASE + 8
# MUL intermediates
PROD_FULL = TEMP_BASE + 10  # Full 16-bit product (as scalar)
# SHIFT intermediates
SHIFT_AMT = TEMP_BASE + 15
SHIFTED_VAL = TEMP_BASE + 16


def bake_onehot_decode(ffn, unit, input_base, output_dim, opcode_dims, S):
    """
    Decode one-hot encoding to scalar value.

    Uses 16 units to detect each possible value 0-15.
    Each unit outputs its value k when input_base[k] is active.

    Args:
        ffn: FFN module to set weights on
        unit: Starting hidden unit index
        input_base: Base dimension of one-hot input (e.g., BD.ALU_LO)
        output_dim: Dimension to write scalar output
        opcode_dims: List of opcode dimensions that activate this decode
        S: SwiGLU scale

    Returns:
        Next available unit index
    """
    for k in range(16):
        # Detect input_base[k] == 1, output k to output_dim
        # When input = 1.0: up = silu(S*1 - S*0.5) = silu(0.5*S) ≈ 0.5*S
        # gate = k, hidden = 0.5*S*k, output = 0.5*S*k * (2/S) = k
        ffn.W_up.data[unit, input_base + k] = S
        ffn.b_up.data[unit] = -S * 0.5  # Threshold at 0.5

        # Gate on opcodes, weighted by value k
        for op_dim in opcode_dims:
            ffn.W_gate.data[unit, op_dim] = float(k)

        ffn.W_down.data[output_dim, unit] = 2.0 / S  # 2x to compensate for silu(0.5*S)
        unit += 1

    return unit


def bake_cancel_pair(ffn, unit, up_dim, gate_dim, out_dim, S, scale=1.0):
    """
    Cancel pair: output = up_value * gate_value * scale

    Uses silu(+S*up)*gate + silu(-S*up)*(-gate) ≈ up*gate when up > 0
    """
    ffn.W_up.data[unit, up_dim] = S
    ffn.W_gate.data[unit, gate_dim] = 1.0
    ffn.W_down.data[out_dim, unit] = scale / S

    ffn.W_up.data[unit + 1, up_dim] = -S
    ffn.W_gate.data[unit + 1, gate_dim] = -1.0
    ffn.W_down.data[out_dim, unit + 1] = scale / S

    return unit + 2


def bake_step_pair(ffn, unit, sum_dims_weights, gate_dim, out_dim, threshold, S, out_scale=1.0):
    """
    Step pair: output = step(weighted_sum >= threshold) * out_scale

    sum_dims_weights: list of (dim, weight) tuples
    """
    # Unit 1: fires when sum >= threshold - 1
    for dim, w in sum_dims_weights:
        ffn.W_up.data[unit, dim] = S * w
        ffn.W_up.data[unit + 1, dim] = S * w

    ffn.b_up.data[unit] = -S * (threshold - 1.0)
    ffn.b_up.data[unit + 1] = -S * threshold

    ffn.W_gate.data[unit, gate_dim] = 1.0
    ffn.W_gate.data[unit + 1, gate_dim] = 1.0

    ffn.W_down.data[out_dim, unit] = out_scale / S
    ffn.W_down.data[out_dim, unit + 1] = -out_scale / S

    return unit + 2


def bake_scalar_to_onehot(ffn, unit, scalar_dim, output_base, gate_dim, S, max_val=16):
    """
    Convert scalar value back to one-hot encoding.

    Uses step pairs to detect each value 0 to max_val-1.
    """
    for k in range(max_val):
        # Step pair: fires when scalar == k (i.e., >= k and < k+1)
        ffn.W_up.data[unit, scalar_dim] = S
        ffn.b_up.data[unit] = -S * (k - 0.5)
        ffn.W_gate.data[unit, gate_dim] = 1.0
        ffn.W_down.data[output_base + k, unit] = 1.0 / S

        ffn.W_up.data[unit + 1, scalar_dim] = S
        ffn.b_up.data[unit + 1] = -S * (k + 0.5)
        ffn.W_gate.data[unit + 1, gate_dim] = 1.0
        ffn.W_down.data[output_base + k, unit + 1] = -1.0 / S

        unit += 2

    return unit


# =============================================================================
# ADD/SUB Implementation
# =============================================================================

def set_efficient_add_sub_decode(ffn, S, BD):
    """
    Layer 1: Decode one-hot inputs to scalars.

    Shared by ADD and SUB operations.
    """
    unit = 0

    # Decode ALU_LO -> A_LO (for ADD, SUB, and other ops)
    unit = bake_onehot_decode(ffn, unit, BD.ALU_LO, A_LO,
                               [BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_AND, BD.OP_OR, BD.OP_XOR, BD.OP_SHL, BD.OP_SHR], S)

    # Decode ALU_HI -> A_HI
    unit = bake_onehot_decode(ffn, unit, BD.ALU_HI, A_HI,
                               [BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_AND, BD.OP_OR, BD.OP_XOR, BD.OP_SHL, BD.OP_SHR], S)

    # Decode AX_CARRY_LO -> B_LO
    unit = bake_onehot_decode(ffn, unit, BD.AX_CARRY_LO, B_LO,
                               [BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_AND, BD.OP_OR, BD.OP_XOR, BD.OP_SHL, BD.OP_SHR], S)

    # Decode AX_CARRY_HI -> B_HI
    unit = bake_onehot_decode(ffn, unit, BD.AX_CARRY_HI, B_HI,
                               [BD.OP_ADD, BD.OP_SUB, BD.OP_MUL, BD.OP_AND, BD.OP_OR, BD.OP_XOR, BD.OP_SHL, BD.OP_SHR], S)

    return unit


def set_efficient_add_compute(ffn, S, BD, start_unit=0):
    """
    Compute ADD: raw sums, carries, and final results.

    Assumes decoded scalars are in A_LO, A_HI, B_LO, B_HI.
    """
    unit = start_unit

    # === RAW_SUM_LO = A_LO + B_LO ===
    # Cancel pair to add two values
    ffn.W_up.data[unit, BD.OP_ADD] = S
    ffn.W_gate.data[unit, A_LO] = 1.0
    ffn.W_gate.data[unit, B_LO] = 1.0
    ffn.W_down.data[RAW_SUM_LO, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_ADD] = -S
    ffn.W_gate.data[unit, A_LO] = -1.0
    ffn.W_gate.data[unit, B_LO] = -1.0
    ffn.W_down.data[RAW_SUM_LO, unit] = 1.0 / S
    unit += 1

    # === RAW_SUM_HI = A_HI + B_HI ===
    ffn.W_up.data[unit, BD.OP_ADD] = S
    ffn.W_gate.data[unit, A_HI] = 1.0
    ffn.W_gate.data[unit, B_HI] = 1.0
    ffn.W_down.data[RAW_SUM_HI, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_ADD] = -S
    ffn.W_gate.data[unit, A_HI] = -1.0
    ffn.W_gate.data[unit, B_HI] = -1.0
    ffn.W_down.data[RAW_SUM_HI, unit] = 1.0 / S
    unit += 1

    # === GEN_LO = step(A_LO + B_LO >= 16) ===
    unit = bake_step_pair(ffn, unit, [(A_LO, 1.0), (B_LO, 1.0)], BD.OP_ADD, GEN_LO, 16.0, S)

    # === GEN_HI = step(A_HI + B_HI >= 16) ===
    unit = bake_step_pair(ffn, unit, [(A_HI, 1.0), (B_HI, 1.0)], BD.OP_ADD, GEN_HI, 16.0, S)

    # === CARRY_TO_HI = GEN_LO (for 8-bit, no propagate chain needed) ===
    unit = bake_cancel_pair(ffn, unit, BD.OP_ADD, GEN_LO, CARRY_TO_HI, S)

    return unit


def set_efficient_add_output(ffn, S, BD, start_unit=0):
    """
    Convert ADD results to one-hot outputs.

    result_lo = raw_sum_lo mod 16
    result_hi = (raw_sum_hi + carry_to_hi) mod 16
    carry_out = (raw_sum_hi + carry_to_hi >= 16)
    """
    unit = start_unit

    # === OUTPUT_LO: raw_sum_lo mod 16 ===
    # For values 0-30, output (value mod 16)
    for raw in range(31):
        result = raw % 16
        # Detect raw_sum_lo == raw using step pair
        ffn.W_up.data[unit, RAW_SUM_LO] = S
        ffn.b_up.data[unit] = -S * (raw - 0.5)
        ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 1.0 / S
        unit += 1

        ffn.W_up.data[unit, RAW_SUM_LO] = S
        ffn.b_up.data[unit] = -S * (raw + 0.5)
        ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + result, unit] = -1.0 / S
        unit += 1

    # === OUTPUT_HI: (raw_sum_hi + carry_to_hi) mod 16 ===
    # Need to handle sum + carry combinations
    for raw in range(31):
        for carry in range(2):
            total = raw + carry
            if total > 31:
                continue
            result = total % 16

            # Detect (raw_sum_hi, carry_to_hi) == (raw, carry)
            # Use combined threshold: raw_sum_hi + 32*carry_to_hi
            combined_val = raw + 32 * carry

            ffn.W_up.data[unit, RAW_SUM_HI] = S
            ffn.W_up.data[unit, CARRY_TO_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined_val - 0.5)
            ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 0.5 / S
            unit += 1

            ffn.W_up.data[unit, RAW_SUM_HI] = S
            ffn.W_up.data[unit, CARRY_TO_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined_val + 0.5)
            ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result, unit] = -0.5 / S
            unit += 1

    # === CARRY output: step(raw_sum_hi + carry_to_hi >= 16) ===
    for raw in range(31):
        for carry in range(2):
            total = raw + carry
            if total >= 16:
                combined_val = raw + 32 * carry

                ffn.W_up.data[unit, RAW_SUM_HI] = S
                ffn.W_up.data[unit, CARRY_TO_HI] = S * 32
                ffn.b_up.data[unit] = -S * (combined_val - 0.5)
                ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                ffn.W_down.data[BD.CARRY, unit] = 0.5 / S
                unit += 1

                ffn.W_up.data[unit, RAW_SUM_HI] = S
                ffn.W_up.data[unit, CARRY_TO_HI] = S * 32
                ffn.b_up.data[unit] = -S * (combined_val + 0.5)
                ffn.W_gate.data[unit, BD.OP_ADD] = 1.0
                ffn.W_down.data[BD.CARRY, unit] = -0.5 / S
                unit += 1

    return unit


def set_efficient_sub_compute(ffn, S, BD, start_unit=0):
    """
    Compute SUB: a - b with borrow.

    sub_lo = a_lo - b_lo + 16 (to keep positive)
    borrow = (a_lo < b_lo)
    sub_hi = a_hi - b_hi + borrow_correction
    """
    unit = start_unit

    # === RAW_DIFF_LO = A_LO - B_LO + 16 (always positive, range 0-31) ===
    ffn.W_up.data[unit, BD.OP_SUB] = S
    ffn.W_gate.data[unit, A_LO] = 1.0
    ffn.W_gate.data[unit, B_LO] = -1.0  # Subtract
    ffn.b_gate.data[unit] = 16.0  # Add 16 to keep positive
    ffn.W_down.data[RAW_SUM_LO, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_SUB] = -S
    ffn.W_gate.data[unit, A_LO] = -1.0
    ffn.W_gate.data[unit, B_LO] = 1.0
    ffn.b_gate.data[unit] = -16.0
    ffn.W_down.data[RAW_SUM_LO, unit] = 1.0 / S
    unit += 1

    # === RAW_DIFF_HI = A_HI - B_HI + 16 ===
    ffn.W_up.data[unit, BD.OP_SUB] = S
    ffn.W_gate.data[unit, A_HI] = 1.0
    ffn.W_gate.data[unit, B_HI] = -1.0
    ffn.b_gate.data[unit] = 16.0
    ffn.W_down.data[RAW_SUM_HI, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_SUB] = -S
    ffn.W_gate.data[unit, A_HI] = -1.0
    ffn.W_gate.data[unit, B_HI] = 1.0
    ffn.b_gate.data[unit] = -16.0
    ffn.W_down.data[RAW_SUM_HI, unit] = 1.0 / S
    unit += 1

    # === BORROW_LO = step(A_LO < B_LO) = step(B_LO > A_LO) ===
    # borrow when a_lo - b_lo + 16 < 16, i.e., raw_diff_lo < 16
    # Actually: borrow = step(A_LO < B_LO) = 1 - step(A_LO >= B_LO)
    # Simpler: borrow = step(raw_diff_lo < 16) = 1 - step(raw_diff_lo >= 16)

    # GEN_LO will hold: 1 if NO borrow (raw_diff >= 16), 0 if borrow
    unit = bake_step_pair(ffn, unit, [(A_LO, 1.0), (B_LO, -1.0)], BD.OP_SUB, GEN_LO, 0.0, S)

    # CARRY_TO_HI = 1 - GEN_LO (borrow propagation)
    # We'll handle this in output stage

    return unit


def set_efficient_sub_output(ffn, S, BD, start_unit=0):
    """
    Convert SUB results to one-hot outputs.
    """
    unit = start_unit

    # === OUTPUT_LO: raw_diff_lo mod 16 ===
    for raw in range(32):
        result = raw % 16

        ffn.W_up.data[unit, RAW_SUM_LO] = S
        ffn.b_up.data[unit] = -S * (raw - 0.5)
        ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 1.0 / S
        unit += 1

        ffn.W_up.data[unit, RAW_SUM_LO] = S
        ffn.b_up.data[unit] = -S * (raw + 0.5)
        ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
        ffn.W_down.data[BD.OUTPUT_LO + result, unit] = -1.0 / S
        unit += 1

    # === OUTPUT_HI: (raw_diff_hi - borrow) mod 16 ===
    # GEN_LO = 1 means no borrow, GEN_LO = 0 means borrow
    # result_hi = raw_diff_hi - (1 - GEN_LO) = raw_diff_hi - 1 + GEN_LO

    for raw in range(32):
        for no_borrow in range(2):
            # When GEN_LO = no_borrow, and raw_diff_hi = raw
            borrow = 1 - no_borrow
            adjusted = raw - borrow
            if adjusted < 0:
                adjusted += 16
            result = adjusted % 16

            combined_val = raw + 32 * no_borrow

            ffn.W_up.data[unit, RAW_SUM_HI] = S
            ffn.W_up.data[unit, GEN_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined_val - 0.5)
            ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result, unit] = 0.5 / S
            unit += 1

            ffn.W_up.data[unit, RAW_SUM_HI] = S
            ffn.W_up.data[unit, GEN_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined_val + 0.5)
            ffn.W_gate.data[unit, BD.OP_SUB] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result, unit] = -0.5 / S
            unit += 1

    # === BORROW output (stored in CARRY) ===
    # borrow = 1 - GEN_LO, but we detect when A < B overall
    # For simplicity: borrow = step(A_LO < B_LO) when hi nibbles equal
    # Full borrow logic is complex, use simplified version

    return unit


# =============================================================================
# MUL Implementation
# =============================================================================

def set_efficient_mul(ffn, S, BD, start_unit=0):
    """
    Efficient MUL: a * b mod 256

    Strategy: Compute full 8-bit product as scalar, then extract lo byte.
    Uses schoolbook: (a_hi*16 + a_lo) * (b_hi*16 + b_lo)
                   = a_hi*b_hi*256 + (a_hi*b_lo + a_lo*b_hi)*16 + a_lo*b_lo

    For mod 256: only need a_lo*b_lo + (a_hi*b_lo + a_lo*b_hi)*16 mod 256
    """
    unit = start_unit

    # We need the full 8-bit value first: a = a_hi*16 + a_lo, b = b_hi*16 + b_lo
    # Then compute a*b mod 256

    # For efficiency, compute partial products and combine
    # P0 = a_lo * b_lo (range 0-225)
    # P1 = a_lo * b_hi (range 0-225)
    # P2 = a_hi * b_lo (range 0-225)
    # result = (P0 + (P1 + P2) * 16) mod 256

    # This is still complex. For now, let's use a lookup-based approach
    # but organized by nibble products (16x16 = 256 units per product)

    # Actually, let's use the scalar multiplication approach:
    # 1. Compute A = a_hi * 16 + a_lo (range 0-255)
    # 2. Compute B = b_hi * 16 + b_lo (range 0-255)
    # 3. Compute A * B mod 256 using step detection

    # Step 1: Combine nibbles to 8-bit scalar
    A_FULL = TEMP_BASE + 20
    B_FULL = TEMP_BASE + 21

    # A_FULL = A_HI * 16 + A_LO
    ffn.W_up.data[unit, BD.OP_MUL] = S
    ffn.W_gate.data[unit, A_HI] = 16.0
    ffn.W_gate.data[unit, A_LO] = 1.0
    ffn.W_down.data[A_FULL, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_MUL] = -S
    ffn.W_gate.data[unit, A_HI] = -16.0
    ffn.W_gate.data[unit, A_LO] = -1.0
    ffn.W_down.data[A_FULL, unit] = 1.0 / S
    unit += 1

    # B_FULL = B_HI * 16 + B_LO
    ffn.W_up.data[unit, BD.OP_MUL] = S
    ffn.W_gate.data[unit, B_HI] = 16.0
    ffn.W_gate.data[unit, B_LO] = 1.0
    ffn.W_down.data[B_FULL, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_MUL] = -S
    ffn.W_gate.data[unit, B_HI] = -16.0
    ffn.W_gate.data[unit, B_LO] = -1.0
    ffn.W_down.data[B_FULL, unit] = 1.0 / S
    unit += 1

    # Now we need A * B mod 256
    # This requires multiplication which SwiGLU can approximate via cancel pairs
    # But we can't easily multiply two arbitrary scalars...

    # Alternative: Use partial product approach
    # P0 = a_lo * b_lo: need 16x16 lookup = 256 combinations
    # But we can factor: for each a_lo value, create step pairs

    # For proof of concept, let's do 16x16 partial product lookups
    # This is still more efficient than 256x256 full lookup

    P0 = TEMP_BASE + 22  # a_lo * b_lo
    P1 = TEMP_BASE + 23  # a_lo * b_hi
    P2 = TEMP_BASE + 24  # a_hi * b_lo

    # P0 = a_lo * b_lo using step detection on each combination
    for a in range(16):
        for b in range(16):
            product = a * b
            # Detect a_lo == a AND b_lo == b
            combined = a + b * 32  # Unique encoding

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P0, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P0, unit] = -1.0 / S
            unit += 1

    # P1 = a_lo * b_hi
    for a in range(16):
        for b in range(16):
            product = a * b
            combined = a + b * 32

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P1, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P1, unit] = -1.0 / S
            unit += 1

    # P2 = a_hi * b_lo
    for a in range(16):
        for b in range(16):
            product = a * b
            combined = a + b * 32

            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P2, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = float(product)
            ffn.W_down.data[P2, unit] = -1.0 / S
            unit += 1

    return unit


def set_efficient_mul_combine(ffn, S, BD, start_unit=0):
    """
    Combine MUL partial products and output.

    result = (P0 + (P1 + P2) * 16) mod 256
    """
    unit = start_unit

    P0 = TEMP_BASE + 22
    P1 = TEMP_BASE + 23
    P2 = TEMP_BASE + 24
    RESULT = TEMP_BASE + 25

    # RESULT = P0 + (P1 + P2) * 16
    # Range: P0 up to 225, (P1+P2)*16 up to 7200, total up to 7425
    # We only need mod 256

    # Combine: RESULT = P0 + P1*16 + P2*16
    ffn.W_up.data[unit, BD.OP_MUL] = S
    ffn.W_gate.data[unit, P0] = 1.0
    ffn.W_gate.data[unit, P1] = 16.0
    ffn.W_gate.data[unit, P2] = 16.0
    ffn.W_down.data[RESULT, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_MUL] = -S
    ffn.W_gate.data[unit, P0] = -1.0
    ffn.W_gate.data[unit, P1] = -16.0
    ffn.W_gate.data[unit, P2] = -16.0
    ffn.W_down.data[RESULT, unit] = 1.0 / S
    unit += 1

    # Now output RESULT mod 256 as one-hot nibbles
    # For each possible result 0-255
    for r in range(256):
        r_lo = r % 16
        r_hi = r // 16

        # Detect RESULT in range [r, r+1) or [r+256, r+257) etc.
        # Since max is ~7425, need to handle wraparound
        for mult in range(30):  # Handle values up to 7680
            val = r + mult * 256
            if val > 7500:
                break

            ffn.W_up.data[unit, RESULT] = S
            ffn.b_up.data[unit] = -S * (val - 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 0.1 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 0.1 / S
            unit += 1

            ffn.W_up.data[unit, RESULT] = S
            ffn.b_up.data[unit] = -S * (val + 0.5)
            ffn.W_gate.data[unit, BD.OP_MUL] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = -0.1 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = -0.1 / S
            unit += 1

    return unit


# =============================================================================
# BITWISE Implementation (AND, OR, XOR)
# =============================================================================

def set_efficient_bitwise(ffn, S, BD, start_unit=0):
    """
    Efficient bitwise ops using bit-level operations.

    For each bit position, extract bit from a and b, compute result.
    8 bits × 3 ops = 24 result bits to compute.
    """
    unit = start_unit

    # For bitwise ops, we work bit-by-bit
    # Bit k of a = floor(a / 2^k) mod 2
    #
    # For 8-bit values:
    # - Bits 0-3 come from a_lo, bits 4-7 come from a_hi
    # - Bit k of nibble = floor(nibble / 2^k) mod 2

    # Extract each bit and compute AND/OR/XOR
    # This requires detecting: (bit_a, bit_b) combinations

    # For nibble n and bit position b (0-3):
    # bit = step(n >= 2^b) XOR step(n >= 2^(b+1)) XOR step(n >= 3*2^b) XOR ...
    # Actually simpler: for each bit position, enumerate which nibble values have that bit set

    # Let's use direct enumeration: for each output nibble value
    # check if it matches the bitwise op result

    # AND output
    for a in range(16):
        for b in range(16):
            result_and = a & b
            result_or = a | b
            result_xor = a ^ b

            combined = a + b * 32

            # AND: lo nibble
            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_AND] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_and, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_AND] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_and, unit] = -1.0 / S
            unit += 1

            # OR: lo nibble
            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_OR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_or, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_OR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_or, unit] = -1.0 / S
            unit += 1

            # XOR: lo nibble
            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_XOR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_xor, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_LO] = S
            ffn.W_up.data[unit, B_LO] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_XOR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_xor, unit] = -1.0 / S
            unit += 1

    # Same for hi nibble
    for a in range(16):
        for b in range(16):
            result_and = a & b
            result_or = a | b
            result_xor = a ^ b

            combined = a + b * 32

            # AND: hi nibble
            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_AND] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_and, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_AND] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_and, unit] = -1.0 / S
            unit += 1

            # OR: hi nibble
            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_OR] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_or, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_OR] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_or, unit] = -1.0 / S
            unit += 1

            # XOR: hi nibble
            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_XOR] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_xor, unit] = 1.0 / S
            unit += 1

            ffn.W_up.data[unit, A_HI] = S
            ffn.W_up.data[unit, B_HI] = S * 32
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_XOR] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_xor, unit] = -1.0 / S
            unit += 1

    return unit


# =============================================================================
# SHIFT Implementation (SHL, SHR)
# =============================================================================

def set_efficient_shift(ffn, S, BD, start_unit=0):
    """
    Efficient SHIFT using lookup on (value, shift_amount).

    For 8-bit value and shift 0-7: 256 * 8 = 2048 combinations per op.
    But we can optimize by working nibble-wise.
    """
    unit = start_unit

    # Decode shift amount (0-7) from B_LO
    # Already decoded in shared decode stage

    # For shifts, we need the full 8-bit value
    A_FULL = TEMP_BASE + 20

    # Combine nibbles: A_FULL = A_HI * 16 + A_LO
    ffn.W_up.data[unit, BD.OP_SHL] = S
    ffn.W_up.data[unit, BD.OP_SHR] = S  # Both ops use same decode
    ffn.W_gate.data[unit, A_HI] = 16.0
    ffn.W_gate.data[unit, A_LO] = 1.0
    ffn.W_down.data[A_FULL, unit] = 1.0 / S
    unit += 1

    ffn.W_up.data[unit, BD.OP_SHL] = -S
    ffn.W_up.data[unit, BD.OP_SHR] = -S
    ffn.W_gate.data[unit, A_HI] = -16.0
    ffn.W_gate.data[unit, A_LO] = -1.0
    ffn.W_down.data[A_FULL, unit] = 1.0 / S
    unit += 1

    # SHL: for each (value, shift) -> (value << shift) mod 256
    for val in range(256):
        for shift in range(8):
            result = (val << shift) % 256
            r_lo = result % 16
            r_hi = result // 16

            combined = val + shift * 256

            ffn.W_up.data[unit, A_FULL] = S
            ffn.W_up.data[unit, B_LO] = S * 256
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_SHL] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 0.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 0.5 / S
            unit += 1

            ffn.W_up.data[unit, A_FULL] = S
            ffn.W_up.data[unit, B_LO] = S * 256
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_SHL] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = -0.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = -0.5 / S
            unit += 1

    # SHR: for each (value, shift) -> value >> shift
    for val in range(256):
        for shift in range(8):
            result = val >> shift
            r_lo = result % 16
            r_hi = result // 16

            combined = val + shift * 256

            ffn.W_up.data[unit, A_FULL] = S
            ffn.W_up.data[unit, B_LO] = S * 256
            ffn.b_up.data[unit] = -S * (combined - 0.5)
            ffn.W_gate.data[unit, BD.OP_SHR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 0.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 0.5 / S
            unit += 1

            ffn.W_up.data[unit, A_FULL] = S
            ffn.W_up.data[unit, B_LO] = S * 256
            ffn.b_up.data[unit] = -S * (combined + 0.5)
            ffn.W_gate.data[unit, BD.OP_SHR] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = -0.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = -0.5 / S
            unit += 1

    return unit


# =============================================================================
# Main integration function
# =============================================================================

def count_efficient_units(BD):
    """Count total hidden units needed for efficient ALU."""
    S = 100.0

    # Create dummy FFN to count
    class DummyFFN:
        def __init__(self):
            self.W_up = type('obj', (object,), {'data': {}})()
            self.b_up = type('obj', (object,), {'data': {}})()
            self.W_gate = type('obj', (object,), {'data': {}})()
            self.b_gate = type('obj', (object,), {'data': {}})()
            self.W_down = type('obj', (object,), {'data': {}})()

            # Track max unit used
            self.max_unit = 0

        def __setitem__(self, key, val):
            pass

    # This is a rough estimate
    decode_units = 64  # 16 × 4 nibbles
    add_units = 200    # compute + output
    sub_units = 200
    mul_units = 1600   # 3 × 16×16 partial products + combine
    bitwise_units = 1600  # 16×16 × 3 ops × 2 nibbles
    shift_units = 4096    # 256 × 8 × 2 ops

    total = decode_units + add_units + sub_units + mul_units + bitwise_units + shift_units

    return {
        'decode': decode_units,
        'add': add_units,
        'sub': sub_units,
        'mul': mul_units,
        'bitwise': bitwise_units,
        'shift': shift_units,
        'total': total
    }
