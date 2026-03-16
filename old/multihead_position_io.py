#!/usr/bin/env python3
"""
Multi-Head Binary Position I/O with ALiBi-style Slopes

Extends binary position matching with multiple attention heads,
each using different slopes for position encoding. This provides:
1. Redundancy for robustness
2. Different "zoom levels" for position matching
3. Better gradient flow during training

Each head uses slope m_h = 2^(-8/H * h) where h is head index and H is num_heads.
This follows ALiBi (Attention with Linear Biases) design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadBinaryPositionIO(nn.Module):
    """
    Multi-head binary position I/O with ALiBi-style slopes.

    Architecture:
    - H attention heads, each with its own slope
    - Slopes follow geometric sequence: m_h = 2^(-8/H * (h+1))
    - Each head does binary position matching independently
    - Results are combined via learned projection

    This gives multiple "views" of position at different scales,
    improving robustness and gradient flow.
    """

    def __init__(self, dim: int = 512, num_bits: int = 12, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_bits = num_bits
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # ALiBi-style slopes: m_h = 2^(-8/H * (h+1))
        # Gives slopes like: 0.5, 0.25, 0.125, ... for H=8
        slopes = torch.tensor([2**(-8.0/num_heads * (h+1)) for h in range(num_heads)])
        self.register_buffer('slopes', slopes)

        # Bit extractor (shared across heads)
        self.register_buffer('powers', torch.tensor([2**k for k in range(num_bits + 1)], dtype=torch.float32))

        # Per-head projections
        self.value_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim, bias=False) for _ in range(num_heads)
        ])

        # Output projection (combines all heads)
        self.output_proj = nn.Linear(dim, dim, bias=False)

        # Character extraction
        self.char_extract = nn.Linear(dim, 8, bias=False)

    def extract_bits(self, value: torch.Tensor) -> torch.Tensor:
        """Extract binary bits from value (shared implementation)."""
        v = value.clamp(0, 2**self.num_bits - 1)
        bits = torch.zeros(*value.shape, self.num_bits, device=value.device)

        for k in range(self.num_bits):
            power_k = self.powers[k]
            quotient = torch.floor(v / power_k)
            bit_raw = torch.fmod(quotient, 2.0)
            bits[..., k] = (bit_raw > 0.5).float()

        return bits

    def binary_match_attention(self, query_bits: torch.Tensor, key_bits: torch.Tensor,
                               values: torch.Tensor, mask: Optional[torch.Tensor],
                               slope: float, temperature: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binary matching attention with ALiBi slope.

        The slope modulates how "strict" the position matching is:
        - Higher slope = stricter matching (more peaked attention)
        - Lower slope = softer matching (more spread attention)
        """
        # query: [batch, query_len, num_bits]
        # key: [batch, key_len, num_bits]
        q = query_bits.unsqueeze(2)  # [batch, query_len, 1, num_bits]
        k = key_bits.unsqueeze(1)    # [batch, 1, key_len, num_bits]

        # Bit-wise match: 1 if equal, 0 if different
        bit_match = 1.0 - torch.abs(q - k)  # [batch, query_len, key_len, num_bits]

        # Product over bits (in log space)
        log_match = torch.log(bit_match + 1e-8)
        log_score = log_match.sum(dim=-1)  # [batch, query_len, key_len]

        # Apply slope scaling (ALiBi-style modulation)
        scores = log_score * slope / temperature

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # [batch, 1, key_len]
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Attend
        output = torch.matmul(weights, values)

        return output, weights

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                anchor: torch.Tensor, read_offset: torch.Tensor,
                input_length: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-head GETCHAR operation.

        Args:
            x: [batch, seq, dim] token embeddings
            positions: [batch, seq] position indices
            anchor: [batch] anchor position
            read_offset: [batch] current read offset
            input_length: [batch] optional input length for masking

        Returns:
            char_value: [batch, 8] character nibbles
            new_offset: [batch] incremented offset
            weights: [batch, num_heads, 1, seq] per-head attention weights
        """
        batch, seq, dim = x.shape

        # Create position keys (relative to anchor)
        anchor_expanded = anchor.unsqueeze(-1).expand_as(positions)
        relative_pos = (positions - anchor_expanded - 1).float()
        relative_pos = relative_pos.clamp(0, 2**self.num_bits - 1)

        # Extract binary keys
        keys = self.extract_bits(relative_pos)  # [batch, seq, num_bits]

        # Extract binary query from read offset
        query = self.extract_bits(read_offset)  # [batch, num_bits]
        query = query.unsqueeze(1)  # [batch, 1, num_bits]

        # Create mask
        mask = (positions > anchor_expanded).float()
        if input_length is not None:
            input_end = anchor.unsqueeze(-1) + input_length.unsqueeze(-1)
            mask = mask * (positions <= input_end).float()

        # Multi-head attention
        head_outputs = []
        all_weights = []

        for h in range(self.num_heads):
            # Project values for this head
            values_h = self.value_projs[h](x)  # [batch, seq, head_dim]

            # Binary matching with head-specific slope
            slope = self.slopes[h].item()
            output_h, weights_h = self.binary_match_attention(
                query, keys, values_h, mask, slope
            )

            head_outputs.append(output_h)  # [batch, 1, head_dim]
            all_weights.append(weights_h)  # [batch, 1, seq]

        # Concatenate heads
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [batch, 1, dim]

        # Stack weights for inspection
        weights = torch.stack(all_weights, dim=1)  # [batch, num_heads, 1, seq]

        # Output projection
        output = self.output_proj(multi_head_output)  # [batch, 1, dim]

        # Extract character
        char_value = self.char_extract(output.squeeze(1))  # [batch, 8]

        # Increment offset
        new_offset = read_offset + 1

        return char_value, new_offset, weights


def test_multihead_io():
    """Test multi-head binary position I/O."""
    print("=" * 70)
    print("Multi-Head Binary Position I/O Test")
    print("=" * 70)
    print()

    # Setup
    dim = 64
    num_bits = 12  # 4K positions
    num_heads = 4
    batch = 1

    io_system = MultiHeadBinaryPositionIO(dim=dim, num_bits=num_bits, num_heads=num_heads)
    io_system.eval()

    print(f"Configuration:")
    print(f"  Dimensions: {dim}")
    print(f"  Bits: {num_bits} ({2**num_bits} positions)")
    print(f"  Heads: {num_heads}")
    print(f"  Slopes: {[f'{s:.4f}' for s in io_system.slopes.tolist()]}")
    print()

    # Test sequence
    input_text = "Hello, World! This is a test of multi-head position I/O."
    anchor_pos = 10
    seq_len = anchor_pos + 1 + len(input_text)

    x = torch.randn(batch, seq_len, dim)

    # Encode input
    for i, c in enumerate(input_text):
        pos = anchor_pos + 1 + i
        char_val = ord(c)
        for nib in range(8):
            x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

    positions = torch.arange(seq_len).unsqueeze(0)
    anchor = torch.tensor([anchor_pos])
    input_length = torch.tensor([len(input_text)])

    print("GETCHAR Tests:")
    print("-" * 50)

    # Test several positions
    test_offsets = [0, 5, 10, 20, len(input_text) - 1]
    all_passed = True

    for offset in test_offsets:
        read_offset = torch.tensor([float(offset)])
        _, _, weights = io_system(x, positions, anchor, read_offset, input_length)

        expected_pos = anchor_pos + 1 + offset
        expected_char = input_text[offset]

        # Check each head
        head_results = []
        for h in range(num_heads):
            peak_pos = weights[0, h, 0].argmax().item()
            peak_weight = weights[0, h, 0, peak_pos].item()
            correct = peak_pos == expected_pos
            head_results.append((peak_pos, peak_weight, correct))

        # All heads should agree
        all_correct = all(r[2] for r in head_results)
        if not all_correct:
            all_passed = False

        print(f"  offset={offset:2d} ('{expected_char}'): ", end="")
        for h, (pos, w, ok) in enumerate(head_results):
            print(f"H{h}@{pos}({w:.2f}){'✓' if ok else '✗'} ", end="")
        print()

    print()
    print("-" * 50)
    print(f"Result: {'✓ ALL HEADS AGREE' if all_passed else '✗ SOME HEADS DISAGREE'}")
    print("=" * 70)

    return all_passed


def test_large_scale():
    """Test multi-head I/O at scale (4K positions)."""
    print("\n" + "=" * 70)
    print("Large Scale Test (4K positions)")
    print("=" * 70)
    print()

    dim = 64
    num_bits = 12
    num_heads = 8
    batch = 1

    io_system = MultiHeadBinaryPositionIO(dim=dim, num_bits=num_bits, num_heads=num_heads)
    io_system.eval()

    # 4K character input
    input_len = 4000
    input_text = "".join([chr(65 + (i % 26)) for i in range(input_len)])

    anchor_pos = 50
    seq_len = anchor_pos + 1 + input_len

    x = torch.randn(batch, seq_len, dim)

    for i, c in enumerate(input_text):
        pos = anchor_pos + 1 + i
        char_val = ord(c)
        for nib in range(8):
            x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

    positions = torch.arange(seq_len).unsqueeze(0)
    anchor = torch.tensor([anchor_pos])
    input_length = torch.tensor([input_len])

    test_offsets = [0, 100, 500, 1000, 2000, 3000, 3999]
    all_passed = True

    print(f"Testing offsets: {test_offsets}")
    print()

    for offset in test_offsets:
        read_offset = torch.tensor([float(offset)])
        _, _, weights = io_system(x, positions, anchor, read_offset, input_length)

        expected_pos = anchor_pos + 1 + offset

        # Check majority of heads agree
        head_peaks = [weights[0, h, 0].argmax().item() for h in range(num_heads)]
        head_weights = [weights[0, h, 0, expected_pos].item() for h in range(num_heads)]

        correct_heads = sum(1 for p in head_peaks if p == expected_pos)
        avg_weight = sum(head_weights) / num_heads

        passed = correct_heads == num_heads
        if not passed:
            all_passed = False

        print(f"  offset={offset:4d}: {correct_heads}/{num_heads} heads correct, "
              f"avg_weight={avg_weight:.4f} {'✓' if passed else '✗'}")

    print()
    print("-" * 50)
    print(f"Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    test_multihead_io()
    test_large_scale()
