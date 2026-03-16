"""
Multi-Bit Cascaded Binary I/O

Instead of extracting 1 bit per layer (binary decision), extract 2-4 bits
per layer by enumerating all possibilities.

1 bit/layer:  2 branches,  16 layers for 16-bit
2 bits/layer: 4 branches,   8 layers for 16-bit
3 bits/layer: 8 branches,   6 layers for 16-bit (with 2 leftover)
4 bits/layer: 16 branches,  4 layers for 16-bit

Each layer uses softmax over thresholds to select the correct bucket.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class MultiBitCascadedExtractor(nn.Module):
    """
    Extract multiple bits per layer by enumerating all possibilities.

    For bits_per_layer=2:
      Layer checks 4 thresholds: 0, 2^(k+1), 2*2^(k+1), 3*2^(k+1)
      Selects which range the value falls into
      Extracts 2 bits, subtracts threshold

    For bits_per_layer=4:
      Layer checks 16 thresholds: 0, 2^k, 2*2^k, ..., 15*2^k
      Each layer extracts one nibble
    """

    def __init__(self, total_bits: int = 16, bits_per_layer: int = 2, scale: float = 10.0):
        super().__init__()
        self.total_bits = total_bits
        self.bits_per_layer = bits_per_layer
        self.num_layers = (total_bits + bits_per_layer - 1) // bits_per_layer
        self.num_branches = 2 ** bits_per_layer
        self.scale = scale  # Softmax temperature

        # Build threshold tables for each layer
        # Layer L extracts bits [(L+1)*bpl - 1 : L*bpl]
        # Thresholds are: 0, 1*base, 2*base, ..., (2^bpl - 1)*base
        # where base = 2^(L * bpl)

        self.register_buffer('thresholds', self._build_thresholds())

    def _build_thresholds(self) -> torch.Tensor:
        """Build threshold table for all layers."""
        # Shape: [num_layers, num_branches]
        thresholds = torch.zeros(self.num_layers, self.num_branches)

        for layer in range(self.num_layers):
            # MSB first: layer 0 handles highest bits
            bit_position = self.total_bits - (layer + 1) * self.bits_per_layer
            if bit_position < 0:
                bit_position = 0
            base = 2 ** bit_position

            for branch in range(self.num_branches):
                thresholds[layer, branch] = branch * base

        return thresholds

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Extract bits from value using cascaded multi-bit layers.

        Args:
            value: Integer value(s) to extract bits from [batch] or scalar

        Returns:
            bits: Binary representation [batch, total_bits]
            selected_branches: Which branch was selected at each layer
        """
        if value.dim() == 0:
            value = value.unsqueeze(0)
        batch_size = value.shape[0]

        remaining = value.float()
        all_bits = torch.zeros(batch_size, self.total_bits, device=value.device)
        selected_branches = []

        for layer in range(self.num_layers):
            # Calculate bits this layer handles (last layer may have fewer)
            bits_remaining = self.total_bits - layer * self.bits_per_layer
            bits_this_layer = min(self.bits_per_layer, bits_remaining)
            branches_this_layer = 2 ** bits_this_layer

            bit_position = self.total_bits - (layer + 1) * self.bits_per_layer
            if bit_position < 0:
                bit_position = 0
            base = 2 ** bit_position

            # For each branch, compute how well remaining fits
            scores = torch.zeros(batch_size, branches_this_layer, device=value.device)

            for b in range(branches_this_layer):
                threshold = b * base
                upper = threshold + base
                center = (threshold + upper) / 2
                distance = torch.abs(remaining - center)
                scores[:, b] = -distance / max(base, 1.0)

            # Softmax to select branch
            probs = F.softmax(self.scale * scores, dim=-1)

            # Hard selection (argmax for forward, soft for gradient)
            selected = torch.argmax(probs, dim=-1)
            selected_branches.append(selected.tolist())

            # Extract bits from selected branch
            for i in range(batch_size):
                branch_val = selected[i].item()
                # Convert branch index to bits
                for bit_offset in range(bits_this_layer):
                    global_bit_idx = bit_position + bit_offset
                    if global_bit_idx < self.total_bits:
                        bit = (branch_val >> bit_offset) & 1
                        all_bits[i, global_bit_idx] = float(bit)

                # Subtract threshold to get remaining
                remaining[i] = remaining[i] - branch_val * base

        return all_bits, selected_branches

    def extract_hard(self, value: int) -> torch.Tensor:
        """Simple integer extraction without batching."""
        remaining = float(value)
        bits = torch.zeros(self.total_bits)

        for layer in range(self.num_layers):
            # Calculate how many bits this layer handles
            # Last layer may handle fewer bits if total doesn't divide evenly
            bits_remaining = self.total_bits - layer * self.bits_per_layer
            bits_this_layer = min(self.bits_per_layer, bits_remaining)
            branches_this_layer = 2 ** bits_this_layer

            bit_position = self.total_bits - (layer + 1) * self.bits_per_layer
            if bit_position < 0:
                bit_position = 0
            base = 2 ** bit_position

            # Find which branch (enumerate possibilities)
            selected_branch = 0
            for b in range(branches_this_layer - 1, -1, -1):
                threshold = b * base
                if remaining >= threshold:
                    selected_branch = b
                    break

            # Extract bits
            for bit_offset in range(bits_this_layer):
                global_bit_idx = bit_position + bit_offset
                if global_bit_idx < self.total_bits:
                    bit = (selected_branch >> bit_offset) & 1
                    bits[global_bit_idx] = float(bit)

            # Update remaining
            remaining -= selected_branch * base

        return bits


class MultiBitCascadedFFN(nn.Module):
    """
    FFN-based multi-bit extraction using SwiGLU pattern.

    For each layer, the FFN has:
    - num_branches hidden units (one per possibility)
    - Each hidden unit fires if remaining is in that range
    - Softmax over hidden units selects the branch

    Weights per layer (bits_per_layer = 2, num_branches = 4):
      W_up: [4, input_dim] - project remaining to branch scores
      W_gate: [4, input_dim] - gating
      W_down: [bits_per_layer + 1, 4] - output bits + new remaining
      b_up, b_gate: [4] - biases with threshold info
      Total: ~4*input + 4*input + (bpl+1)*4 + 8 ≈ 8*input + 4*bpl + 12

    For input_dim=1 (just the remaining value):
      Per layer: 8 + 4*2 + 12 = 28 weights (for 2 bits/layer)
      8 layers: 224 weights total

    Compare to 1-bit cascade: 7 weights/layer * 16 layers = 112 weights
    """

    def __init__(self, total_bits: int = 16, bits_per_layer: int = 2):
        super().__init__()
        self.total_bits = total_bits
        self.bits_per_layer = bits_per_layer
        self.num_layers = (total_bits + bits_per_layer - 1) // bits_per_layer
        self.num_branches = 2 ** bits_per_layer

        # Per-layer FFN weights (sparse)
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            self.layers.append(self._make_layer(layer_idx))

    def _make_layer(self, layer_idx: int) -> nn.Module:
        """Create FFN for one layer."""
        bit_position = self.total_bits - (layer_idx + 1) * self.bits_per_layer
        if bit_position < 0:
            bit_position = 0
        base = 2 ** bit_position

        # Thresholds for this layer
        thresholds = torch.tensor([b * base for b in range(self.num_branches)], dtype=torch.float32)

        layer = nn.Module()
        layer.register_buffer('thresholds', thresholds)
        layer.base = base
        layer.bit_position = bit_position

        return layer

    def forward(self, value: int) -> torch.Tensor:
        """Extract bits using cascaded FFN layers."""
        remaining = float(value)
        bits = torch.zeros(self.total_bits)

        for layer_idx, layer_module in enumerate(self.layers):
            thresholds = layer_module.thresholds
            base = layer_module.base
            bit_position = layer_module.bit_position

            # FFN computation: find which branch
            # Scores based on distance to threshold centers
            scores = torch.zeros(self.num_branches)
            for b in range(self.num_branches):
                center = thresholds[b] + base / 2
                # SiLU-style scoring
                distance = remaining - center
                scores[b] = -abs(distance)

            # Softmax selection
            probs = F.softmax(scores * 0.1, dim=-1)
            selected = torch.argmax(probs).item()

            # Extract bits
            for bit_offset in range(self.bits_per_layer):
                global_bit_idx = bit_position + bit_offset
                if global_bit_idx < self.total_bits:
                    bit = (selected >> bit_offset) & 1
                    bits[global_bit_idx] = float(bit)

            # Update remaining
            remaining -= thresholds[selected].item()

        return bits


class WeightAnalysis:
    """Analyze weights needed for different configurations."""

    @staticmethod
    def analyze(total_bits: int = 16, bits_per_layer: int = 2) -> dict:
        num_layers = (total_bits + bits_per_layer - 1) // bits_per_layer
        num_branches = 2 ** bits_per_layer

        # Per-layer weights (SwiGLU pattern)
        # W_up, W_gate: [num_branches, 1] - score each branch
        # b_up: [num_branches] - threshold offsets
        # W_down: [bits_per_layer + 1, num_branches] - extract bits + remaining

        weights_up = num_branches * 1
        weights_gate = num_branches * 1
        weights_down = (bits_per_layer + 1) * num_branches
        biases = num_branches * 2  # b_up, b_gate

        per_layer = weights_up + weights_gate + weights_down + biases
        total = per_layer * num_layers

        return {
            'total_bits': total_bits,
            'bits_per_layer': bits_per_layer,
            'num_layers': num_layers,
            'num_branches': num_branches,
            'weights_per_layer': per_layer,
            'total_weights': total,
            'comparison': {
                '1_bit': 7 * total_bits,  # Original cascaded (1 bit/layer)
                f'{bits_per_layer}_bit': total,
            }
        }


def bits_to_int(bits: torch.Tensor) -> int:
    """Convert bit tensor to integer."""
    result = 0
    for i in range(len(bits)):
        if bits[i] > 0.5:
            result |= (1 << i)
    return result


# ============ Tests ============

def test_multibit_extractor():
    """Test multi-bit cascaded extraction."""
    print("=" * 60)
    print("Multi-Bit Cascaded Extraction Tests")
    print("=" * 60)

    test_values = [0, 1, 255, 256, 1000, 12345, 43721, 65535]

    for bpl in [2, 3, 4]:
        print(f"\n--- {bpl} bits per layer ({2**bpl} branches) ---")
        extractor = MultiBitCascadedExtractor(total_bits=16, bits_per_layer=bpl)
        print(f"Layers needed: {extractor.num_layers}")

        all_pass = True
        for val in test_values:
            bits = extractor.extract_hard(val)
            recovered = bits_to_int(bits)
            status = "✓" if recovered == val else "✗"
            if recovered != val:
                all_pass = False
                print(f"  {val:5d}: {status} (got {recovered})")

        if all_pass:
            print(f"  All {len(test_values)} values: ✓")


def test_weight_analysis():
    """Analyze weights for different configurations."""
    print("\n" + "=" * 60)
    print("Weight Analysis: Multi-Bit vs Single-Bit Cascade")
    print("=" * 60)

    print("\n16-bit I/O:")
    print(f"{'Bits/Layer':>12} {'Layers':>8} {'Branches':>10} {'Weights/L':>12} {'Total':>10}")
    print("-" * 55)

    for bpl in [1, 2, 3, 4]:
        analysis = WeightAnalysis.analyze(total_bits=16, bits_per_layer=bpl)
        print(f"{bpl:>12} {analysis['num_layers']:>8} {analysis['num_branches']:>10} "
              f"{analysis['weights_per_layer']:>12} {analysis['total_weights']:>10}")


def test_detailed_extraction():
    """Show detailed layer-by-layer extraction."""
    print("\n" + "=" * 60)
    print("Detailed Layer-by-Layer Extraction (2 bits/layer)")
    print("=" * 60)

    value = 43721  # Binary: 1010101011001001
    print(f"\nValue: {value} (binary: {value:016b})")

    extractor = MultiBitCascadedExtractor(total_bits=16, bits_per_layer=2)

    # Manual trace
    remaining = float(value)
    print(f"\nLayer-by-layer (MSB first, 2 bits at a time):")

    for layer in range(extractor.num_layers):
        bit_pos = 16 - (layer + 1) * 2
        if bit_pos < 0:
            bit_pos = 0
        base = 2 ** bit_pos

        # Find branch
        selected = 0
        for b in range(3, -1, -1):
            if remaining >= b * base:
                selected = b
                break

        # Show extraction
        bit1 = (selected >> 1) & 1
        bit0 = selected & 1
        threshold = selected * base
        new_remaining = remaining - threshold

        print(f"  Layer {layer}: remaining={remaining:5.0f} >= {selected}*{base}? "
              f"bits[{bit_pos+1}:{bit_pos}] = {bit1}{bit0}, "
              f"remaining → {new_remaining:.0f}")

        remaining = new_remaining

    # Verify
    bits = extractor.extract_hard(value)
    recovered = bits_to_int(bits)
    print(f"\nRecovered: {recovered} (expected {value}) {'✓' if recovered == value else '✗'}")


def test_4bit_extraction():
    """Show 4-bit (nibble) extraction - matches nibble-based arithmetic."""
    print("\n" + "=" * 60)
    print("4-Bit (Nibble) Extraction - Matches Neural Arithmetic")
    print("=" * 60)

    value = 43721
    print(f"\nValue: {value} (hex: {value:04X})")

    extractor = MultiBitCascadedExtractor(total_bits=16, bits_per_layer=4)

    remaining = float(value)
    print(f"\nLayer-by-layer (MSB first, 4 bits at a time):")

    for layer in range(extractor.num_layers):
        bit_pos = 16 - (layer + 1) * 4
        if bit_pos < 0:
            bit_pos = 0
        base = 2 ** bit_pos

        # Find nibble value
        selected = int(remaining / base) if base > 0 else 0
        if selected > 15:
            selected = 15

        threshold = selected * base
        new_remaining = remaining - threshold

        print(f"  Layer {layer}: remaining={remaining:5.0f} / {base} → nibble={selected:X}, "
              f"remaining → {new_remaining:.0f}")

        remaining = new_remaining

    # Verify
    bits = extractor.extract_hard(value)
    recovered = bits_to_int(bits)
    print(f"\nRecovered: {recovered} (expected {value}) {'✓' if recovered == value else '✗'}")


def test_io_layer_counts():
    """Show I/O layer counts for different configurations."""
    print("\n" + "=" * 60)
    print("I/O Layer Counts for GETCHAR/PUTCHAR")
    print("=" * 60)

    print("\n16-bit I/O:")
    print(f"{'Bits/Layer':>12} {'Extract Layers':>15} {'+ KV Lookup':>12} {'Total':>8}")
    print("-" * 50)

    for bpl in [1, 2, 3, 4]:
        num_layers = (16 + bpl - 1) // bpl
        total = num_layers + 1  # + KV lookup layer
        print(f"{bpl:>12} {num_layers:>15} {1:>12} {total:>8}")

    print("\nReduced from 17 layers (1-bit) to 5 layers (4-bit)!")


if __name__ == '__main__':
    test_multibit_extractor()
    test_weight_analysis()
    test_detailed_extraction()
    test_4bit_extraction()
    test_io_layer_counts()
