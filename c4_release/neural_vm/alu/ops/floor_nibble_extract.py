"""
Pure FFN floor via nibble extraction - no MAGIC trick, fp32 safe.

Key insight: Output to ONE-HOT slots, not scalar accumulation.
Each nibble value k gets its own output slot, with window detection
determining which slot gets high activation.

For nibble extraction from Q:
1. Scale Q by 1/base^j to get value in [0, base)
2. For each k in 0..base-1: detect if scaled value is in window [k, k+1)
3. Output window detection score to RESULT slot k
4. Argmax over RESULT slots gives the nibble value

Shifted thresholds (by eps=0.001) avoid exact integer boundary issues.

Total: 2 × base × base = 2 × 16 × 16 = 512 hidden units per layer
For 32-bit (8 layers): 512 × 8 = 4096 hidden units total

Works with fp32 - no IEEE 754 rounding tricks needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig
from .common import GenericE, GenericPureFFN, GenericFlattenedFFN


class HighNibbleOneHotFFN(nn.Module):
    """Extract the highest nibble using one-hot window detection.

    For highest position j (scale by 1/base^j):
    1. Compute scaled_val = Q / base^j (in range [0, base) for valid Q)
    2. For each nibble k: detect if scaled_val is in [k, k+1)
    3. Output detection score to RESULT[k] one-hot slot

    Uses shifted thresholds (eps=0.001) to handle exact integer boundaries.
    """

    def __init__(self, ge: GenericE, opcode: int, position: int,
                 input_slot: int, one_hot_base: int = None):
        super().__init__()
        self.ge = ge
        self.position = position
        N = ge.NUM_POSITIONS
        base = ge.BASE
        S = ge.SCALE

        if one_hot_base is None:
            one_hot_base = ge.RESULT

        # Window detection for each nibble value: 4 units per value
        hidden_dim = base * 4

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=torch.float32)
        fi = self.flat_ffn._flat_idx

        scale_factor = 1.0 / float(base ** position)
        eps = 0.001  # Small shift to avoid exact boundaries

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            op_idx = fi(0, ge.OP_START + opcode)
            input_idx = fi(0, input_slot)

            h = 0

            for k in range(base):
                # Output slot for nibble k at this position
                result_idx = fi(position, one_hot_base + k)

                threshold_lo = k - eps
                threshold_hi = k + 1 - eps

                # Step pair for step(scaled >= threshold_lo)
                W_up[h, input_idx] = S * scale_factor
                b_up[h] = -S * (threshold_lo - 0.5)
                W_gate[h, op_idx] = 1.0
                W_down[result_idx, h] = 1.0 / S
                h += 1

                W_up[h, input_idx] = S * scale_factor
                b_up[h] = -S * (threshold_lo + 0.5)
                W_gate[h, op_idx] = 1.0
                W_down[result_idx, h] = -1.0 / S
                h += 1

                # Step pair for step(scaled >= threshold_hi) - subtract from output
                W_up[h, input_idx] = S * scale_factor
                b_up[h] = -S * (threshold_hi - 0.5)
                W_gate[h, op_idx] = 1.0
                W_down[result_idx, h] = -1.0 / S
                h += 1

                W_up[h, input_idx] = S * scale_factor
                b_up[h] = -S * (threshold_hi + 0.5)
                W_gate[h, op_idx] = 1.0
                W_down[result_idx, h] = 1.0 / S
                h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


class LowNibbleWithSubtractionFFN(nn.Module):
    """Extract lower nibble by subtracting higher nibble contribution.

    For position j (where higher position j+1 result is available):
    1. Read high nibble h from one-hot at position j+1
    2. For each (h, l) pair: detect if Q is in [h*base + l, h*base + l + 1)
    3. Output detection score to RESULT[l] one-hot slot at position j

    Architecture: W_up does step detection on Q (silu gives step behavior),
    W_gate does 2-way AND gating (opcode AND hi_slot).
    """

    def __init__(self, ge: GenericE, opcode: int, position: int,
                 input_slot: int, higher_position: int,
                 one_hot_base: int = None, hi_one_hot_base: int = None):
        super().__init__()
        self.ge = ge
        self.position = position
        base = ge.BASE
        S = ge.SCALE

        if one_hot_base is None:
            one_hot_base = ge.RESULT
        if hi_one_hot_base is None:
            hi_one_hot_base = ge.RESULT

        # For each (high_val, low_val) pair: 4 units for window detection
        # Total: base * base * 4 = 16 * 16 * 4 = 1024 for base=16
        hidden_dim = base * base * 4

        self.flat_ffn = GenericFlattenedFFN(ge, hidden_dim=hidden_dim, dtype=torch.float32)
        fi = self.flat_ffn._flat_idx

        hi_contribution = float(base ** higher_position)
        eps = 0.001

        with torch.no_grad():
            W_up = self.flat_ffn.W_up
            b_up = self.flat_ffn.b_up
            W_gate = self.flat_ffn.W_gate
            W_down = self.flat_ffn.W_down

            op_idx = fi(0, ge.OP_START + opcode)
            input_idx = fi(0, input_slot)

            h = 0

            for hi_val in range(base):
                # Index into the high nibble one-hot result
                hi_slot_idx = fi(higher_position, hi_one_hot_base + hi_val)
                # Contribution of this high nibble value
                subtract = hi_val * hi_contribution

                for lo_val in range(base):
                    # Output slot for this low nibble value
                    result_idx = fi(position, one_hot_base + lo_val)

                    # Detect if Q is in [subtract + lo_val, subtract + lo_val + 1)
                    # Using absolute thresholds
                    threshold_lo = subtract + lo_val - eps
                    threshold_hi = subtract + lo_val + 1 - eps

                    # SwiGLU: hidden = silu(W_up @ x + b_up) * (W_gate @ x + b_gate)
                    # W_up: step detection on Q (silu gives step behavior)
                    # W_gate: 2-way AND (opcode + hi_slot - 1.0), positive when both active

                    # Step pair for step(Q >= threshold_lo)
                    W_up[h, input_idx] = S
                    b_up[h] = -S * (threshold_lo - 0.5)
                    W_gate[h, op_idx] = 1.0
                    W_gate[h, hi_slot_idx] = 1.0
                    self.flat_ffn.b_gate[h] = -1.5  # Fires when opcode + hi_slot >= 1.5
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

                    W_up[h, input_idx] = S
                    b_up[h] = -S * (threshold_lo + 0.5)
                    W_gate[h, op_idx] = 1.0
                    W_gate[h, hi_slot_idx] = 1.0
                    self.flat_ffn.b_gate[h] = -1.5
                    W_down[result_idx, h] = -1.0 / S
                    h += 1

                    # Step pair for step(Q >= threshold_hi) - subtract
                    W_up[h, input_idx] = S
                    b_up[h] = -S * (threshold_hi - 0.5)
                    W_gate[h, op_idx] = 1.0
                    W_gate[h, hi_slot_idx] = 1.0
                    self.flat_ffn.b_gate[h] = -1.5
                    W_down[result_idx, h] = -1.0 / S
                    h += 1

                    W_up[h, input_idx] = S
                    b_up[h] = -S * (threshold_hi + 0.5)
                    W_gate[h, op_idx] = 1.0
                    W_gate[h, hi_slot_idx] = 1.0
                    self.flat_ffn.b_gate[h] = -1.5
                    W_down[result_idx, h] = 1.0 / S
                    h += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flat_ffn(x)


def build_onehot_floor_layers(config: ChunkConfig, opcode: int,
                               input_slot: int) -> nn.ModuleList:
    """Build one-hot nibble floor extraction pipeline.

    Returns N layers that extract floor(input) as one-hot encoded nibbles.
    Each layer writes to RESULT[pos] one-hot slots for its position.

    Architecture:
    - Layer 0: Extract highest nibble using window detection on scaled value
    - Layer 1..N-1: Extract lower nibbles by subtracting higher nibble contribution

    Args:
        config: Chunk configuration
        opcode: Operation opcode to gate on
        input_slot: Which slot contains the scalar value to floor

    Returns:
        ModuleList of N layers
    """
    ge = GenericE(config)
    N = ge.NUM_POSITIONS

    layers = []

    # First layer: highest nibble (direct window detection)
    highest_pos = N - 1
    layers.append(HighNibbleOneHotFFN(
        ge, opcode, position=highest_pos, input_slot=input_slot
    ))

    # Remaining layers: extract by subtracting higher nibble contribution
    for pos in range(N - 2, -1, -1):  # From second-highest to lowest
        layers.append(LowNibbleWithSubtractionFFN(
            ge, opcode, position=pos, input_slot=input_slot,
            higher_position=pos + 1  # Read from one position higher
        ))

    return nn.ModuleList(layers)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    config = ChunkConfig(chunk_bits=4, total_bits=8)
    ge = GenericE(config)

    print(f"Testing one-hot nibble floor extraction")
    print(f"Config: {ge.NUM_POSITIONS} positions × base {ge.BASE}")

    # Use a non-colliding input slot (100 is outside RESULT range [5, 21))
    INPUT_SLOT = 100
    print(f"Using INPUT_SLOT={INPUT_SLOT} to avoid collision with RESULT={ge.RESULT}")

    layers = build_onehot_floor_layers(config, opcode=31, input_slot=INPUT_SLOT)
    print(f"Layers: {len(layers)}")

    total_params = sum((p != 0).sum().item() for layer in layers for p in layer.parameters())
    print(f"Total non-zero params: {int(total_params)}")

    # Test all 256 values
    correct = 0
    for Q in range(256):
        x = torch.zeros(1, ge.NUM_POSITIONS, ge.DIM)
        x[0, 0, INPUT_SLOT] = float(Q)
        x[0, 0, ge.OP_START + 31] = 1.0

        with torch.no_grad():
            for layer in layers:
                x = layer(x)

        # Extract one-hot results
        hi_onehot = x[0, 1, ge.RESULT:ge.RESULT + 16]
        lo_onehot = x[0, 0, ge.RESULT:ge.RESULT + 16]

        got_hi = hi_onehot.argmax().item()
        got_lo = lo_onehot.argmax().item()
        got_Q = got_hi * 16 + got_lo

        expected_hi = Q // 16
        expected_lo = Q % 16

        if got_Q == Q:
            correct += 1
        elif Q < 10:  # Show first few failures
            print(f"  Q={Q}: got hi={got_hi} lo={got_lo}, exp hi={expected_hi} lo={expected_lo}")

    print(f"\nResult: {correct}/256 correct")
