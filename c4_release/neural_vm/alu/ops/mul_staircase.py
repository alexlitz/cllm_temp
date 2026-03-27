"""
Efficient BYTE MUL using staircase floor FFN (pure neural, no torch.round).

The staircase floor formula:
    floor(x/base) = Σ sigmoid(scale * (x/base - k + eps)) for k = 1..M

Each threshold k becomes one FFN hidden unit:
    W_up = scale/base
    b_up = scale * (-k + eps)
    W_down = 1 (sum all units)

This is O(max_quotient) params per position, but fully neural.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig, BYTE
from .common import GenericE, GenericFlattenedFFN, bake_clear_pair


# Staircase parameters
SCALE = 10000.0  # High scale for sharp steps
EPS = 0.002      # Small eps for 1/256 precision


class StaircaseCarryExtractFFN(nn.Module):
    """Extract carry using staircase floor FFN.

    For each position:
        carry = floor(RESULT / base) using staircase
        remainder = RESULT - carry * base

    Staircase floor:
        floor(x) = Σ sigmoid(scale * (x - k + eps)) for k = 1..M

    Hidden units per position:
        - M units for floor thresholds
        - 2 units for remainder computation

    Total hidden: N * (M + 2) + clear/add overhead
    """

    def __init__(self, ge: GenericE, opcode: int, pass_idx: int, max_value: int):
        super().__init__()
        N = ge.NUM_POSITIONS
        S = SCALE
        base = ge.BASE
        eps = EPS

        # Max quotient determines number of threshold units
        max_quotient = max(1, max_value // base)

        self.base = base
        self.max_quotient = max_quotient
        self.pass_idx = pass_idx
        self.ge = ge
        self.opcode = opcode
        self.N = N

        # Compute hidden dimension
        # - Clear CARRY_OUT: 2*N
        # - Add incoming carry (if not first): 2*(N-1)
        # - Staircase floor: max_quotient * N (one threshold per quotient value per position)
        # - Remainder: 2*N (subtract carry * base)

        staircase_units = max_quotient * N
        clear_units = 2 * N
        add_units = 2 * (N - 1) if pass_idx > 0 else 0
        remainder_units = 2 * N

        hidden_dim = clear_units + add_units + staircase_units + remainder_units

        # Create FFN
        input_dim = N * ge.DIM
        output_dim = N * ge.DIM

        self.W_up = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.b_gate = nn.Parameter(torch.ones(hidden_dim))  # Default gate = 1
        self.W_down = nn.Parameter(torch.zeros(output_dim, hidden_dim))
        self.b_down = nn.Parameter(torch.zeros(output_dim))

        # Helper to compute flat index
        def fi(pos, dim):
            return pos * ge.DIM + dim

        with torch.no_grad():
            h = 0
            op_idx = fi(0, ge.OP_START + opcode)

            # Clear CARRY_OUT
            for pos in range(N):
                carry_idx = fi(pos, ge.CARRY_OUT)
                # Clear pair: set to 0 when opcode active
                self.W_up[h, op_idx] = S
                self.b_up[h] = 0
                self.W_gate[h, op_idx] = 1.0
                self.W_down[carry_idx, h] = -1.0 / S  # Subtract current value
                # Also need to read current value... this is getting complex
                # Let's use simpler approach: just overwrite in staircase
                h += 1

                self.W_up[h, op_idx] = -S
                self.W_gate[h, op_idx] = -1.0
                self.W_down[carry_idx, h] = -1.0 / S
                h += 1

            # Add incoming carry from previous pass
            if pass_idx > 0:
                for pos in range(1, N):
                    carry_from = fi(pos - 1, ge.CARRY_OUT)
                    result_idx = fi(pos, ge.RESULT)

                    self.W_up[h, carry_from] = S
                    self.W_gate[h, op_idx] = 1.0
                    self.W_down[result_idx, h] = 1.0 / S
                    h += 1

                    self.W_up[h, carry_from] = -S
                    self.W_gate[h, op_idx] = -1.0
                    self.W_down[result_idx, h] = 1.0 / S
                    h += 1

            # Staircase floor for each position
            # floor(x/base) = Σ sigmoid(S * (x/base - k + eps))
            for pos in range(N):
                result_idx = fi(pos, ge.RESULT)
                carry_idx = fi(pos, ge.CARRY_OUT)

                for k in range(1, max_quotient + 1):
                    # Hidden unit for threshold k
                    # sigmoid(S * (x/base - k + eps))
                    # = sigmoid(S/base * x + S * (-k + eps))
                    self.W_up[h, result_idx] = S / base
                    self.b_up[h] = S * (-k + eps)
                    self.W_gate[h, op_idx] = 1.0  # Only when opcode active
                    self.b_gate[h] = 0.0  # Gate controlled by opcode
                    self.W_down[carry_idx, h] = 1.0  # Sum into carry
                    h += 1

            # Remainder: RESULT = RESULT - carry * base
            # Using SwiGLU: silu(S * carry) * base / S ≈ carry * base
            for pos in range(N):
                result_idx = fi(pos, ge.RESULT)
                carry_idx = fi(pos, ge.CARRY_OUT)

                # Subtract carry * base
                self.W_up[h, carry_idx] = S
                self.W_gate[h, op_idx] = float(base)
                self.b_gate[h] = 0.0
                self.W_down[result_idx, h] = -1.0 / S
                h += 1

                self.W_up[h, carry_idx] = -S
                self.W_gate[h, op_idx] = -float(base)
                self.b_gate[h] = 0.0
                self.W_down[result_idx, h] = -1.0 / S
                h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pure FFN forward - no torch.round()."""
        orig_shape = x.shape
        x_flat = x.view(-1, self.N * self.ge.DIM)

        # SwiGLU: silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
        up = F.linear(x_flat, self.W_up, self.b_up)
        gate = F.linear(x_flat, self.W_gate, self.b_gate)

        # For staircase units, use sigmoid instead of silu
        # But mixing sigmoid and silu in same layer is tricky...
        # Let's use sigmoid for all (staircase) and silu for multiplication parts
        # Actually, sigmoid(x) * gate is fine for staircase

        # Use sigmoid for threshold detection
        hidden = torch.sigmoid(up) * gate

        # Output
        output = F.linear(hidden, self.W_down, self.b_down)

        return output.view(orig_shape) + x  # Residual connection


def _compute_max_values(config: ChunkConfig):
    """Compute max values at each carry pass stage."""
    N = config.num_positions
    base = config.base
    chunk_max = config.chunk_max

    # After schoolbook: max sum at any position
    max_val = N * chunk_max * chunk_max

    max_values = [max_val]
    while True:
        max_carry = max_val // base
        if max_carry <= 1:
            break
        max_val = (base - 1) + max_carry
        max_values.append(max_val)

    return max_values


def build_staircase_mul_layers(config: ChunkConfig, opcode: int = 27) -> nn.ModuleList:
    """Build MUL pipeline using staircase floor for carry extraction."""
    from .mul_efficient import EfficientSchoolbookFFN
    from .mul import MulGenPropFFN, MulBinaryLookaheadFFN, MulFinalCorrectionFFN

    ge = GenericE(config)
    layers = []

    # Layer 1: Schoolbook multiplication (reuse efficient version)
    layers.append(EfficientSchoolbookFFN(ge, opcode))

    # Carry extraction passes using staircase floor
    max_values = _compute_max_values(config)
    for idx, max_val in enumerate(max_values):
        layers.append(StaircaseCarryExtractFFN(ge, opcode, pass_idx=idx, max_value=max_val))

    # Remaining layers (already efficient)
    layers.append(MulGenPropFFN(ge, opcode))
    layers.append(MulBinaryLookaheadFFN(ge, opcode))
    layers.append(MulFinalCorrectionFFN(ge, opcode))

    return nn.ModuleList(layers)


def count_staircase_params():
    """Count parameters for staircase MUL."""
    layers = build_staircase_mul_layers(BYTE, opcode=27)

    print("Staircase BYTE MUL (pure FFN floor):")
    total = 0
    for i, layer in enumerate(layers):
        params = sum(p.numel() for p in layer.parameters())
        nonzero = sum((p != 0).sum().item() for p in layer.parameters())
        print(f"  Layer {i}: {layer.__class__.__name__} - {nonzero:,} non-zero / {params:,} total")
        total += nonzero
    print(f"  Total non-zero: {total:,} params")

    return total


if __name__ == '__main__':
    print("=" * 70)
    print("Staircase BYTE MUL (Pure FFN)")
    print("=" * 70)
    count_staircase_params()

    print()
    print("Comparison:")
    print("-" * 50)
    from .mul import build_mul_layers
    original = build_mul_layers(BYTE, opcode=27)
    orig_params = sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in original)
    print(f"  Original (lookup tables): {orig_params:,} params")

    staircase_params = count_staircase_params()
    print(f"  Staircase (pure FFN): {staircase_params:,} params")
    print(f"  Reduction: {100*(1-staircase_params/orig_params):.1f}%")
