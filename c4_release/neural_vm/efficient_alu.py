"""
Efficient ALU operations using pure FFN weight baking.

All computation is baked into FFN weights at initialization.
Forward pass is standard SwiGLU - no runtime integer ops:
    hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
    output = x + W_down @ hidden

This is 100% neural - the same forward pass used in any transformer.
The "intelligence" is in the weight values, not in the architecture.

All operations use direct one-hot to one-hot mapping for reliability.
"""

import torch
import torch.nn as nn

# Temporary dimension for carry/borrow propagation between layers
TEMP_BASE = 450
GEN_LO = TEMP_BASE + 0  # Carry/borrow flag from lo nibble


# =============================================================================
# ADD Implementation (tested 65536/65536 = 100%)
# =============================================================================

def set_efficient_add_lo(ffn, S, BD, start_unit=0):
    """
    ADD stage 1: Compute OUTPUT_LO and carry flag (GEN_LO).

    Direct one-hot to one-hot mapping: 256 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            result = (a + b) % 16
            carry = 1 if (a + b) >= 16 else 0

            # 3-way AND: ALU_LO[a] AND AX_CARRY_LO[b] AND OP_ADD
            ffn.W_up.data[unit, BD.ALU_LO + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
            ffn.W_up.data[unit, BD.OP_ADD] = S
            ffn.b_up.data[unit] = -S * 2.5

            ffn.W_gate.data[unit, BD.ALU_LO + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if carry:
                ffn.W_down.data[GEN_LO, unit] = 1.0 / S

            unit += 1

    return unit


def set_efficient_add_hi(ffn, S, BD, start_unit=0):
    """
    ADD stage 2: Compute OUTPUT_HI using carry from GEN_LO.

    Must run AFTER set_efficient_add_lo. Uses 512 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            # Case 1: no carry from lo (GEN_LO = 0)
            result_no_carry = (a + b) % 16
            carry_out_no = 1 if (a + b) >= 16 else 0

            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_ADD] = S
            ffn.W_up.data[unit, GEN_LO] = -S  # Inhibit when carry present
            ffn.b_up.data[unit] = -S * 2.5

            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_HI + result_no_carry, unit] = 2.0 / S
            if carry_out_no:
                ffn.W_down.data[BD.CARRY, unit] = 2.0 / S
            unit += 1

            # Case 2: carry from lo (GEN_LO = 1)
            result_with_carry = (a + b + 1) % 16
            carry_out_yes = 1 if (a + b + 1) >= 16 else 0

            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_ADD] = S
            ffn.W_up.data[unit, GEN_LO] = S  # Require carry present
            ffn.b_up.data[unit] = -S * 3.5  # 4-way AND

            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_HI + result_with_carry, unit] = 2.0 / S
            if carry_out_yes:
                ffn.W_down.data[BD.CARRY, unit] = 2.0 / S
            unit += 1

    return unit


# =============================================================================
# SUB Implementation
# =============================================================================

def set_efficient_sub_lo(ffn, S, BD, start_unit=0):
    """
    SUB stage 1: Compute OUTPUT_LO and borrow flag (GEN_LO).

    result_lo = (a_lo - b_lo) mod 16
    borrow = 1 if a_lo < b_lo else 0

    Uses 256 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            result = (a - b) % 16
            borrow = 1 if a < b else 0

            # 3-way AND: ALU_LO[a] AND AX_CARRY_LO[b] AND OP_SUB
            ffn.W_up.data[unit, BD.ALU_LO + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
            ffn.W_up.data[unit, BD.OP_SUB] = S
            ffn.b_up.data[unit] = -S * 2.5

            ffn.W_gate.data[unit, BD.ALU_LO + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + result, unit] = 2.0 / S
            if borrow:
                ffn.W_down.data[GEN_LO, unit] = 1.0 / S  # GEN_LO = borrow flag

            unit += 1

    return unit


def set_efficient_sub_hi(ffn, S, BD, start_unit=0):
    """
    SUB stage 2: Compute OUTPUT_HI using borrow from GEN_LO.

    result_hi = (a_hi - b_hi - borrow) mod 16

    Uses 512 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            # Case 1: no borrow from lo (GEN_LO = 0)
            result_no_borrow = (a - b) % 16
            borrow_out_no = 1 if a < b else 0

            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_SUB] = S
            ffn.W_up.data[unit, GEN_LO] = -S  # Inhibit when borrow present
            ffn.b_up.data[unit] = -S * 2.5

            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_HI + result_no_borrow, unit] = 2.0 / S
            if borrow_out_no:
                ffn.W_down.data[BD.CARRY, unit] = 2.0 / S  # CARRY = final borrow
            unit += 1

            # Case 2: borrow from lo (GEN_LO = 1)
            result_with_borrow = (a - b - 1) % 16
            borrow_out_yes = 1 if a < b or (a == b) else 0  # a - b - 1 < 0 when a <= b

            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_SUB] = S
            ffn.W_up.data[unit, GEN_LO] = S  # Require borrow present
            ffn.b_up.data[unit] = -S * 3.5  # 4-way AND

            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0

            ffn.W_down.data[BD.OUTPUT_HI + result_with_borrow, unit] = 2.0 / S
            if borrow_out_yes:
                ffn.W_down.data[BD.CARRY, unit] = 2.0 / S
            unit += 1

    return unit


# =============================================================================
# BITWISE Implementation (AND, OR, XOR)
# =============================================================================

def set_efficient_bitwise_lo(ffn, S, BD, start_unit=0):
    """
    BITWISE stage 1: Compute OUTPUT_LO for AND, OR, XOR.

    Direct 16×16 lookup per operation. Uses 256×3 = 768 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            # AND
            result_and = a & b
            ffn.W_up.data[unit, BD.ALU_LO + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
            ffn.W_up.data[unit, BD.OP_AND] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_LO + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_and, unit] = 2.0 / S
            unit += 1

            # OR
            result_or = a | b
            ffn.W_up.data[unit, BD.ALU_LO + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
            ffn.W_up.data[unit, BD.OP_OR] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_LO + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_or, unit] = 2.0 / S
            unit += 1

            # XOR
            result_xor = a ^ b
            ffn.W_up.data[unit, BD.ALU_LO + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b] = S
            ffn.W_up.data[unit, BD.OP_XOR] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_LO + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_LO + result_xor, unit] = 2.0 / S
            unit += 1

    return unit


def set_efficient_bitwise_hi(ffn, S, BD, start_unit=0):
    """
    BITWISE stage 2: Compute OUTPUT_HI for AND, OR, XOR.

    Direct 16×16 lookup per operation. Uses 256×3 = 768 units.
    """
    unit = start_unit

    for a in range(16):
        for b in range(16):
            # AND
            result_and = a & b
            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_AND] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_and, unit] = 2.0 / S
            unit += 1

            # OR
            result_or = a | b
            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_OR] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_or, unit] = 2.0 / S
            unit += 1

            # XOR
            result_xor = a ^ b
            ffn.W_up.data[unit, BD.ALU_HI + a] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b] = S
            ffn.W_up.data[unit, BD.OP_XOR] = S
            ffn.b_up.data[unit] = -S * 2.5
            ffn.W_gate.data[unit, BD.ALU_HI + a] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b] = 1.0
            ffn.W_down.data[BD.OUTPUT_HI + result_xor, unit] = 2.0 / S
            unit += 1

    return unit


# =============================================================================
# MUL Implementation (nibble factoring)
# =============================================================================

def set_efficient_mul(ffn, S, BD, start_unit=0):
    """
    MUL: Compute (a * b) mod 256 using nibble factoring.

    a = a_hi*16 + a_lo
    b = b_hi*16 + b_lo
    result = (a_lo*b_lo + (a_lo*b_hi + a_hi*b_lo)*16) mod 256

    Note: a_hi*b_hi*256 always >= 256, so doesn't contribute to result mod 256.

    Uses direct 256×256 lookup split across nibble products.
    For each 8-bit combination, we compute the result.
    Uses 256×256 = 65536 units but can be factored to ~4096.

    This implementation uses full 8-bit lookup for correctness.
    """
    unit = start_unit

    # Full 8-bit × 8-bit lookup: for each (a, b) pair
    for a in range(256):
        a_lo, a_hi = a % 16, a // 16
        for b in range(256):
            b_lo, b_hi = b % 16, b // 16
            result = (a * b) % 256
            r_lo = result % 16
            r_hi = result // 16

            # 4-way AND: ALU_LO[a_lo] AND ALU_HI[a_hi] AND AX_CARRY_LO[b_lo] AND AX_CARRY_HI[b_hi] AND OP_MUL
            ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
            ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
            ffn.W_up.data[unit, BD.OP_MUL] = S
            ffn.b_up.data[unit] = -S * 4.5  # 5-way AND threshold

            ffn.W_gate.data[unit, BD.ALU_LO + a_lo] = 1.0
            ffn.W_gate.data[unit, BD.ALU_HI + a_hi] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b_lo] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b_hi] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0 / S

            unit += 1

    return unit


# =============================================================================
# SHIFT Implementation (SHL, SHR)
# =============================================================================

def set_efficient_shift(ffn, S, BD, start_unit=0):
    """
    SHIFT: Compute SHL and SHR using direct lookup.

    shift_amount = b_lo & 7 (0-7, higher bits ignored)
    SHL: result = (a << shift) mod 256
    SHR: result = a >> shift

    Uses 256×16×2 = 8192 units (handles all b_lo values 0-15).
    """
    unit = start_unit

    for a in range(256):
        a_lo, a_hi = a % 16, a // 16
        for b_lo in range(16):  # All possible b_lo values
            shift = b_lo & 7  # Actual shift amount is b_lo mod 8

            # SHL
            result_shl = (a << shift) % 256
            r_lo_shl = result_shl % 16
            r_hi_shl = result_shl // 16

            # 4-way AND: ALU_LO[a_lo] AND ALU_HI[a_hi] AND AX_CARRY_LO[b_lo] AND OP_SHL
            ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
            ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
            ffn.W_up.data[unit, BD.OP_SHL] = S
            ffn.b_up.data[unit] = -S * 3.5  # 4-way AND

            ffn.W_gate.data[unit, BD.ALU_LO + a_lo] = 1.0
            ffn.W_gate.data[unit, BD.ALU_HI + a_hi] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b_lo] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + r_lo_shl, unit] = 1.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi_shl, unit] = 1.5 / S
            unit += 1

            # SHR
            result_shr = a >> shift
            r_lo_shr = result_shr % 16
            r_hi_shr = result_shr // 16

            ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
            ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
            ffn.W_up.data[unit, BD.OP_SHR] = S
            ffn.b_up.data[unit] = -S * 3.5

            ffn.W_gate.data[unit, BD.ALU_LO + a_lo] = 1.0
            ffn.W_gate.data[unit, BD.ALU_HI + a_hi] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b_lo] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + r_lo_shr, unit] = 1.5 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi_shr, unit] = 1.5 / S
            unit += 1

    return unit


# =============================================================================
# DIV/MOD Implementation
# =============================================================================

def set_efficient_div(ffn, S, BD, start_unit=0):
    """
    DIV: Compute a // b using direct lookup.

    For b=0, result is undefined (we output 0).

    Uses 256×256 = 65536 units.
    """
    unit = start_unit

    for a in range(256):
        a_lo, a_hi = a % 16, a // 16
        for b in range(256):
            b_lo, b_hi = b % 16, b // 16

            if b == 0:
                result = 0  # Division by zero -> 0
            else:
                result = a // b

            r_lo = result % 16
            r_hi = result // 16

            # 5-way AND
            ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
            ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
            ffn.W_up.data[unit, BD.OP_DIV] = S
            ffn.b_up.data[unit] = -S * 4.5

            ffn.W_gate.data[unit, BD.ALU_LO + a_lo] = 1.0
            ffn.W_gate.data[unit, BD.ALU_HI + a_hi] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b_lo] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b_hi] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0 / S

            unit += 1

    return unit


def set_efficient_mod(ffn, S, BD, start_unit=0):
    """
    MOD: Compute a % b using direct lookup.

    For b=0, result is undefined (we output a).

    Uses 256×256 = 65536 units.
    """
    unit = start_unit

    for a in range(256):
        a_lo, a_hi = a % 16, a // 16
        for b in range(256):
            b_lo, b_hi = b % 16, b // 16

            if b == 0:
                result = a  # Mod by zero -> return a
            else:
                result = a % b

            r_lo = result % 16
            r_hi = result // 16

            # 5-way AND
            ffn.W_up.data[unit, BD.ALU_LO + a_lo] = S
            ffn.W_up.data[unit, BD.ALU_HI + a_hi] = S
            ffn.W_up.data[unit, BD.AX_CARRY_LO + b_lo] = S
            ffn.W_up.data[unit, BD.AX_CARRY_HI + b_hi] = S
            ffn.W_up.data[unit, BD.OP_MOD] = S
            ffn.b_up.data[unit] = -S * 4.5

            ffn.W_gate.data[unit, BD.ALU_LO + a_lo] = 1.0
            ffn.W_gate.data[unit, BD.ALU_HI + a_hi] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_LO + b_lo] = 1.0
            ffn.W_gate.data[unit, BD.AX_CARRY_HI + b_hi] = 1.0

            ffn.W_down.data[BD.OUTPUT_LO + r_lo, unit] = 1.0 / S
            ffn.W_down.data[BD.OUTPUT_HI + r_hi, unit] = 1.0 / S

            unit += 1

    return unit


# =============================================================================
# Legacy exports (for compatibility)
# =============================================================================

# Keep old names for backwards compatibility
set_efficient_add_output_lo = set_efficient_add_lo
set_efficient_add_output_hi = set_efficient_add_hi
