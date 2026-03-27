"""
Weight transformation utilities for integrating efficient ALU ops.

Converts GenericE format weights to BD format at initialization time.
Zero runtime overhead - just better weight patterns.
"""

import torch
import torch.nn as nn
from .alu.chunk_config import NIBBLE
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.common import GenericE


# Temporary dimension allocations in BD space
class TempDims:
    """Temporary dimensions for multi-layer op intermediate results."""
    BASE = 450  # Start of temp storage (BD uses 0-449)

    # SHIFT temps (SHL)
    SHL_S1_LO = 450  # s_1 for low nibble
    SHL_C1_LO = 451  # c_1 for low nibble
    SHL_S2_LO = 452
    SHL_C2_LO = 453
    SHL_S3_LO = 454
    SHL_C3_LO = 455

    SHL_S1_HI = 456  # s_1 for high nibble
    SHL_C1_HI = 457
    SHL_S2_HI = 458
    SHL_C2_HI = 459
    SHL_S3_HI = 460
    SHL_C3_HI = 461

    # SHIFT temps (SHR)
    SHR_S1_LO = 462
    SHR_C1_LO = 463
    SHR_S2_LO = 464
    SHR_C2_LO = 465
    SHR_S3_LO = 466
    SHR_C3_LO = 467

    SHR_S1_HI = 468
    SHR_C1_HI = 469
    SHR_S2_HI = 470
    SHR_C2_HI = 471
    SHR_S3_HI = 472
    SHR_C3_HI = 473


def _set_layer13_shift_stage1_efficient(ffn, S, BD):
    """
    L13 FFN: SHIFT stage 1 (precompute) using efficient implementation.

    Transforms ShlPrecomputeFFN and ShrPrecomputeFFN weights to BD format.

    Stage 1 computes sub-chunk shifts for r=1,2,3:
    - For SHL: s_r = (2^r * a) mod base, c_r = floor(2^r * a / base)
    - For SHR: s_r = floor(a / 2^r), c_r = (a mod 2^r) * 2^(k-r)

    GenericE version operates on x[pos, NIB_A] = scalar nibble value.
    BD version operates on x[ALU_LO[k]] = one-hot encoding.

    Writes intermediate results to temporary dimensions for stage 2.

    Returns:
        Number of units used
    """
    print("Building L13 SHIFT stage 1 (efficient)...")

    # Build efficient layers
    shl_layers = build_shl_layers(NIBBLE, opcode=23)
    shr_layers = build_shr_layers(NIBBLE, opcode=24)

    shl_stage1 = shl_layers[0]  # ShlPrecomputeFFN
    shr_stage1 = shr_layers[0]  # ShrPrecomputeFFN

    ge = GenericE(NIBBLE)
    unit = 0

    # ===== Transform SHL stage 1 weights =====
    # For each hidden unit in the GenericE version, we need to:
    # 1. Read GenericE weights
    # 2. Transform to BD equivalents
    # 3. Write to ffn weights

    # The GenericE version has 28 hidden units for SHL
    # It processes 2 positions (lo and hi nibble)
    # We need to replicate for both positions in BD format

    # For now, let's copy the GenericE weights directly for positions 0-1
    # and map GenericE slots to BD dimensions

    shl_hidden = shl_stage1.ffn.hidden_dim  # Should be 28

    for h in range(shl_hidden):
        # Read GenericE weights for this hidden unit
        w_up_ge = shl_stage1.ffn.W_up[h]      # [160] vector
        b_up_ge = shl_stage1.ffn.b_up[h]      # scalar
        w_gate_ge = shl_stage1.ffn.W_gate[h]  # [160] vector
        b_gate_ge = shl_stage1.ffn.b_gate[h]  # scalar
        w_down_ge = shl_stage1.ffn.W_down[:, h]  # [160] vector

        # Process position 0 (low nibble)
        # GenericE reads from NIB_A (slot 0)
        # BD reads from ALU_LO[0-15] (one-hot)

        # Transform: w_up_ge[NIB_A] * scalar_value
        # becomes: sum_k w_up_bd[ALU_LO + k] * (1 if k active else 0)
        # So: w_up_bd[ALU_LO + k] = w_up_ge[NIB_A] * k

        nib_a_weight = w_up_ge[ge.NIB_A].item()
        for k in range(16):
            ffn.W_up[unit, BD.ALU_LO + k] = S * nib_a_weight * k / S  # Scaled value

        # Copy bias
        ffn.b_up[unit] = b_up_ge

        # Gate reads from OP_START + opcode
        # This is the same in both formats
        for op_idx in range(min(72, w_gate_ge.shape[0] - ge.OP_START)):
            if ge.OP_START + op_idx < 160:
                op_weight = w_gate_ge[ge.OP_START + op_idx].item()
                if abs(op_weight) > 1e-6:
                    ffn.W_gate[unit, BD.OP_START + op_idx] = op_weight

        ffn.b_gate[unit] = b_gate_ge

        # Down writes to various slots
        # Map GenericE output slots to BD temp dimensions
        for slot_idx in range(160):
            down_weight = w_down_ge[slot_idx].item()
            if abs(down_weight) < 1e-6:
                continue

            # Map GenericE slots to BD dimensions
            # This is position-dependent and slot-dependent
            # For now, write to temp dimensions
            # Need to understand which slot corresponds to what

            # The precompute stage writes to s_slots[r] and c_slots[r]
            # We need to figure out the mapping
            pass

        unit += 1

    # Similar for SHR...
    shr_hidden = shr_stage1.ffn.hidden_dim
    unit += shr_hidden

    print(f"  Stage 1: {unit} units ({shl_hidden} SHL + {shr_hidden} SHR)")
    print(f"  Current L13: 4096 units")
    print(f"  Savings so far: {4096 - unit} units")

    return unit


def _set_layer14_shift_stage2_efficient(ffn, S, BD):
    """
    L14 FFN: SHIFT stage 2 (select) using efficient implementation.

    Reads precomputed values from L13 temp dimensions.
    Selects and routes based on shift amount.
    Writes final result to OUTPUT_LO/HI.

    Returns:
        Number of units used
    """
    print("Building L14 SHIFT stage 2 (efficient)...")

    shl_layers = build_shl_layers(NIBBLE, opcode=23)
    shr_layers = build_shr_layers(NIBBLE, opcode=24)

    shl_stage2 = shl_layers[1]  # ShiftSelectFFN
    shr_stage2 = shr_layers[1]  # ShiftSelectFFN

    unit = 0

    # Stage 2 is more complex - uses flattened FFN
    # For now, placeholder

    shl_hidden = shl_stage2.flat_ffn.hidden_dim  # Should be ~528
    shr_hidden = shr_stage2.flat_ffn.hidden_dim

    unit = shl_hidden + shr_hidden

    print(f"  Stage 2: {unit} units ({shl_hidden} SHL + {shr_hidden} SHR)")

    return unit


def test_transformation():
    """Test the weight transformation."""
    print("="*70)
    print("Testing Weight Transformation")
    print("="*70)

    # Build efficient layers
    from .alu.ops.shift import build_shl_layers
    shl = build_shl_layers(NIBBLE, 23)

    print(f"\nSHL has {len(shl)} layers:")
    for i, layer in enumerate(shl):
        params = sum((p != 0).sum().item() for p in layer.parameters())
        print(f"  Layer {i}: {params} params")

    # Check stage 1 structure
    stage1 = shl[0]
    print(f"\nStage 1 (Precompute):")
    print(f"  Hidden dim: {stage1.ffn.hidden_dim}")
    print(f"  Input dim: {stage1.ffn.W_up.shape[1]}")  # Should be 160
    print(f"  Output dim: {stage1.ffn.W_down.shape[0]}")  # Should be 160

    # Examine some weights
    print(f"\nSample weights:")
    print(f"  W_up[0, :5]: {stage1.ffn.W_up[0, :5]}")
    print(f"  W_gate[0, :5]: {stage1.ffn.W_gate[0, :5]}")
    print(f"  b_up[0]: {stage1.ffn.b_up[0]}")

    ge = GenericE(NIBBLE)
    print(f"\nGenericE slots:")
    print(f"  NIB_A: {ge.NIB_A}")
    print(f"  NIB_B: {ge.NIB_B}")
    print(f"  RESULT: {ge.RESULT}")
    print(f"  OP_START: {ge.OP_START}")


if __name__ == '__main__':
    test_transformation()
