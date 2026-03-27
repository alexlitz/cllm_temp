"""
Direct integration of proven efficient ALU ops from neural_vm/alu/ops/.

The ops are already implemented, tested (10K cases each), and working.
We just need to wire them into vm_step.py's transformer layers.

Key insight: Don't rewrite - just integrate!
"""

import torch
import torch.nn as nn
from .alu.chunk_config import NIBBLE
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.add import build_add_layers
from .alu.ops.sub import build_sub_layers
from .alu.ops.mul import build_mul_layers
from .alu.ops.bitwise import build_and_layers, build_or_layers, build_xor_layers


def set_shift_integrated_l13(ffn, S, BD):
    """
    L13 FFN: SHIFT stage 1 (precompute) - direct from alu/ops/shift.py.

    Uses the existing ShlPrecomputeFFN and ShrPrecomputeFFN implementations.
    These compute sub-chunk shifts for r=1,2,3.

    Returns:
        Number of units used
    """
    # Build the proven efficient layers
    shl_layers = build_shl_layers(NIBBLE, opcode=23)
    shr_layers = build_shr_layers(NIBBLE, opcode=24)

    # Extract stage 1 (precompute)
    shl_stage1 = shl_layers[0]  # ShlPrecomputeFFN
    shr_stage1 = shr_layers[0]  # ShrPrecomputeFFN

    # The layers are already built with correct weights!
    # We just need to copy them into the vm_step FFN

    # For now, use the layers directly in a wrapper
    # This requires modifying how vm_step.py calls the FFN

    unit_count = shl_stage1.ffn.hidden_dim + shr_stage1.ffn.hidden_dim
    print(f"L13 SHIFT stage 1: {unit_count} units (28 SHL + 28 SHR)")

    return unit_count, [shl_stage1, shr_stage1]


def set_shift_integrated_l14(ffn, S, BD):
    """
    L14 FFN: SHIFT stage 2 (select) - direct from alu/ops/shift.py.

    Uses the existing ShiftSelectFFN implementations.
    These select and route based on shift amount.

    Returns:
        Number of units used
    """
    shl_layers = build_shl_layers(NIBBLE, opcode=23)
    shr_layers = build_shr_layers(NIBBLE, opcode=24)

    # Extract stage 2 (select)
    shl_stage2 = shl_layers[1]  # ShiftSelectFFN
    shr_stage2 = shr_layers[1]  # ShiftSelectFFN

    unit_count = shl_stage2.flat_ffn.hidden_dim + shr_stage2.flat_ffn.hidden_dim
    print(f"L14 SHIFT stage 2: {unit_count} units (528 SHL + 528 SHR)")

    return unit_count, [shl_stage2, shr_stage2]


class EmbeddingAdapter(nn.Module):
    """
    Adapts between vm_step.py's BD format (512 dims) and GenericE format (1280 dims).

    BD format: One-hot encodings for ALU_LO[0-15], ALU_HI[0-15], etc.
    GenericE format: 8 positions × 160 dims, with scalar values in specific slots

    For SHIFT:
    - Extract ALU_LO/HI one-hot → convert to scalars → write to GenericE NIB_A slots
    - Extract shift amount from AX_CARRY_LO → write to GenericE NIB_B slots
    - Copy opcode flags
    """

    def __init__(self, config, BD):
        super().__init__()
        self.config = config
        self.BD = BD
        from .alu.ops.common import GenericE
        self.ge = GenericE(config)

        # Learnable projection is overkill - we can do deterministic mapping
        # But for now, let's expand the embedding space

    def bd_to_ge(self, x_bd):
        """Convert [B, 1, 512] BD format to [B, 8, 160] GenericE format."""
        B = x_bd.shape[0]
        N = self.config.num_positions  # 8 for NIBBLE
        D = self.ge.DIM  # 160

        x_ge = torch.zeros(B, N, D, dtype=x_bd.dtype, device=x_bd.device)

        # Map ALU_LO[0-15] → position 0, slot NIB_A (as scalar)
        for k in range(16):
            # When ALU_LO[k] is 1, add k to NIB_A slot
            x_ge[:, 0, self.ge.NIB_A] += x_bd[:, 0, self.BD.ALU_LO + k] * float(k)

        # Map ALU_HI[0-15] → position 1, slot NIB_A
        for k in range(16):
            x_ge[:, 1, self.ge.NIB_A] += x_bd[:, 0, self.BD.ALU_HI + k] * float(k)

        # Map shift amount from AX_CARRY_LO → position 0-1, slot NIB_B
        for k in range(16):
            val = x_bd[:, 0, self.BD.AX_CARRY_LO + k] * float(k)
            x_ge[:, 0, self.ge.NIB_B] += val % 16
            x_ge[:, 1, self.ge.NIB_B] += val // 16

        # Copy opcode flags to all positions
        for opcode in range(min(72, 512 - self.BD.OP_START)):
            op_val = x_bd[:, 0, self.BD.OP_START + opcode]
            for pos in range(N):
                if self.ge.OP_START + opcode < D:
                    x_ge[:, pos, self.ge.OP_START + opcode] = op_val

        return x_ge

    def ge_to_bd(self, x_ge, x_bd_orig):
        """Convert [B, 8, 160] GenericE back to [B, 1, 512] BD format (residual)."""
        B = x_ge.shape[0]
        x_bd = x_bd_orig.clone()

        # Extract RESULT from positions 0-1 and write to OUTPUT_LO/HI as one-hot
        result_lo = x_ge[:, 0, self.ge.RESULT].round().long().clamp(0, 15)
        result_hi = x_ge[:, 1, self.ge.RESULT].round().long().clamp(0, 15)

        for b in range(B):
            x_bd[b, 0, self.BD.OUTPUT_LO + result_lo[b]] = 2.0
            x_bd[b, 0, self.BD.OUTPUT_HI + result_hi[b]] = 2.0

        return x_bd


class MultiLayerFFN(nn.Module):
    """
    Runs multiple efficient op layers with embedding adaptation.

    Forward pass:
    1. Convert BD → GenericE
    2. Run efficient op layers
    3. Convert GenericE → BD
    """

    def __init__(self, layers, config, BD):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.adapter = EmbeddingAdapter(config, BD)

    def forward(self, x_bd):
        # Convert to GenericE
        x_ge = self.adapter.bd_to_ge(x_bd)

        # Run efficient op layers
        for layer in self.layers:
            x_ge = layer(x_ge)

        # Convert back to BD (with residual)
        x_bd_out = self.adapter.ge_to_bd(x_ge, x_bd)

        return x_bd_out


def create_integrated_shift_ffn():
    """
    Create a replacement FFN for L13-L14 that runs SHIFT efficiently.

    Returns combined FFN that does both stages.
    """
    shl_layers = build_shl_layers(NIBBLE, opcode=23)
    shr_layers = build_shr_layers(NIBBLE, opcode=24)

    # Combine all layers
    all_layers = list(shl_layers) + list(shr_layers)

    return MultiLayerFFN(all_layers)


if __name__ == '__main__':
    print("="*70)
    print("Path B Integration - Using Existing Efficient Ops")
    print("="*70)

    # Test building the integrated FFNs
    print("\nBuilding SHIFT integration...")
    shift_ffn = create_integrated_shift_ffn()
    print(f"Created integrated SHIFT FFN with {len(shift_ffn.layers)} layers")

    total_params = sum(
        sum((p != 0).sum().item() for p in layer.parameters())
        for layer in shift_ffn.layers
    )
    print(f"Total non-zero parameters: {total_params:,}")
    print(f"Current SHIFT: 36,864 params")
    print(f"Savings: {36864 - total_params:,} ({(36864-total_params)/36864*100:.1f}%)")

    print("\nNext: Integrate into vm_step.py")
