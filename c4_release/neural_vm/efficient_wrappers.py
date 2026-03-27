"""
Efficient wrappers for integrating alu/ops/ into vm_step.py.

Provides runtime conversion between BD format (512 dims flat) and GenericE format
(8 positions × 160 dims). Conversion is fast and parameter savings are huge.
"""

import torch
import torch.nn as nn
from .alu.chunk_config import NIBBLE
from .alu.ops.shift import build_shl_layers, build_shr_layers
from .alu.ops.common import GenericE


class EfficientShiftFFN(nn.Module):
    """
    Drop-in replacement for L13 SHIFT that uses efficient multi-layer implementation.

    Converts BD format → GenericE, runs efficient SHIFT, converts back.

    Input/Output: [B, seq_len, 512] in BD format (same as original)
    Internal: Uses GenericE format for computation

    Parameter count: ~5,624 (vs 36,864 current = 84.7% reduction)
    """

    def __init__(self, S, BD):
        super().__init__()
        self.S = S
        self.BD = BD
        self.config = NIBBLE
        self.ge = GenericE(NIBBLE)

        # Build efficient layers
        self.shl_layers = build_shl_layers(NIBBLE, opcode=23)
        self.shr_layers = build_shr_layers(NIBBLE, opcode=24)

        # Set to eval mode (no training)
        for layer in self.shl_layers:
            layer.eval()
        for layer in self.shr_layers:
            layer.eval()

    def forward(self, x_bd):
        """
        Forward pass with format conversion.

        Args:
            x_bd: [B, seq_len, 512] in BD format

        Returns:
            [B, seq_len, 512] in BD format (residual added)
        """
        B, seq_len, _ = x_bd.shape

        # Convert BD → GenericE for each position in sequence
        # Only process positions where MARK_AX is active
        x_bd_out = x_bd.clone()

        for b in range(B):
            for pos in range(seq_len):
                # Check if this is an AX marker position
                if x_bd[b, pos, self.BD.MARK_AX] < 0.5:
                    continue

                # Convert this position to GenericE format
                x_ge = self.bd_to_ge(x_bd[b, pos])  # [8, 160]

                # Run SHL if active
                if x_bd[b, pos, self.BD.OP_SHL] > 0.5:
                    x_ge = x_ge.unsqueeze(0)  # [1, 8, 160]
                    with torch.no_grad():
                        for layer in self.shl_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)  # [8, 160]

                # Run SHR if active
                elif x_bd[b, pos, self.BD.OP_SHR] > 0.5:
                    x_ge = x_ge.unsqueeze(0)  # [1, 8, 160]
                    with torch.no_grad():
                        for layer in self.shr_layers:
                            x_ge = layer(x_ge)
                    x_ge = x_ge.squeeze(0)  # [8, 160]
                else:
                    continue  # No shift operation

                # Convert result back to BD format
                self.ge_to_bd(x_ge, x_bd_out[b, pos])

        return x_bd_out

    def compact(self, block_size=1):
        """
        Compact method for compatibility with vm_step.py's model.compact().

        The efficient layers are already compact, so this is a no-op.
        """
        pass

    def sparsify(self):
        """
        Sparsify method for compatibility with vm_step.py's model.sparsify().

        The efficient layers are already using sparse representations where beneficial.
        """
        pass

    def compact_moe(self, opcode_range=None, relay_map=None):
        """
        MoE compaction for compatibility with vm_step.py.

        The efficient layers already separate SHL and SHR, so this is a no-op.
        """
        pass

    def bd_to_ge(self, x_bd_single):
        """
        Convert single position from BD to GenericE.

        Args:
            x_bd_single: [512] vector in BD format

        Returns:
            [8, 160] tensor in GenericE format
        """
        x_ge = torch.zeros(8, 160, dtype=x_bd_single.dtype, device=x_bd_single.device)

        # Extract operand A from one-hot ALU_LO/HI
        # Position 0 (low nibble)
        for k in range(16):
            if x_bd_single[self.BD.ALU_LO + k] > 0.5:
                x_ge[0, self.ge.NIB_A] = float(k)
                break

        # Position 1 (high nibble)
        for k in range(16):
            if x_bd_single[self.BD.ALU_HI + k] > 0.5:
                x_ge[1, self.ge.NIB_A] = float(k)
                break

        # Positions 2-7 are zero for 8-bit values

        # Extract shift amount from AX_CARRY_LO
        shift_amount = 0
        for k in range(16):
            if x_bd_single[self.BD.AX_CARRY_LO + k] > 0.5:
                shift_amount = k
                break

        # Encode shift amount in NIB_B across positions
        sa = shift_amount
        for i in range(8):
            x_ge[i, self.ge.NIB_B] = float(sa % 16)
            sa //= 16
            if sa == 0:
                break

        # Copy opcode flags to all positions
        for opcode_idx in [23, 24]:  # SHL, SHR
            op_val = x_bd_single[self.BD.OPCODE_BASE + opcode_idx]
            for pos in range(8):
                x_ge[pos, self.ge.OP_START + opcode_idx] = op_val

        return x_ge

    def ge_to_bd(self, x_ge, x_bd_single):
        """
        Convert GenericE result back to BD format (in-place update).

        Args:
            x_ge: [8, 160] tensor in GenericE format
            x_bd_single: [512] vector in BD format (modified in-place)
        """
        # Extract result from positions 0-1
        result_lo = int(x_ge[0, self.ge.RESULT].round().item())
        result_hi = int(x_ge[1, self.ge.RESULT].round().item())

        # Clamp to valid range
        result_lo = max(0, min(15, result_lo))
        result_hi = max(0, min(15, result_hi))

        # Write to OUTPUT_LO/HI as one-hot (with residual)
        x_bd_single[self.BD.OUTPUT_LO + result_lo] += 2.0
        x_bd_single[self.BD.OUTPUT_HI + result_hi] += 2.0


def replace_shift_in_model(model, S, BD):
    """
    Replace L13 SHIFT implementation with efficient version.

    This modifies model.blocks[13].ffn in-place.
    """
    print("Replacing L13 SHIFT with efficient implementation...")

    # Replace the FFN module
    old_ffn = model.blocks[13].ffn
    new_ffn = EfficientShiftFFN(S, BD)

    model.blocks[13].ffn = new_ffn

    # Count parameters
    old_params = sum((p != 0).sum().item() for p in old_ffn.parameters())
    new_params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                     for layer in new_ffn.shl_layers + new_ffn.shr_layers)

    print(f"  Old parameters: {old_params:,}")
    print(f"  New parameters: {new_params:,}")
    print(f"  Savings: {old_params - new_params:,} ({(old_params-new_params)/old_params*100:.1f}%)")

    return model


if __name__ == '__main__':
    print("="*70)
    print("Efficient SHIFT Wrapper")
    print("="*70)

    # Test parameter count
    from .alu.ops.shift import build_shl_layers, build_shr_layers
    shl = build_shl_layers(NIBBLE, 23)
    shr = build_shr_layers(NIBBLE, 24)

    total = sum(sum((p != 0).sum().item() for p in layer.parameters())
                for layer in shl + shr)

    print(f"\nEfficient SHIFT: {total:,} params")
    print(f"Current SHIFT: 36,864 params")
    print(f"Savings: {36864 - total:,} ({(36864-total)/36864*100:.1f}%)")

    print("\n✓ Wrapper ready for integration")
