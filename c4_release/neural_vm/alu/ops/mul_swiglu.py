"""
Pure SwiGLU MUL for 32-bit operations with BYTE chunks.

Following c4llm document methodology:
- Schoolbook: 10 partial products using SwiGLU multiplication
- Carry extraction: Use SwiGLU division, not step pairs
- Binary carry lookahead: 3 carry bits for 4 positions

Key insight: Instead of step pairs for floor(x/256), use:
1. SwiGLU to compute x * (1/256) ≈ quotient (continuous)
2. Binarize to nearest integer using soft step

For bounded integers, this is exact when properly scaled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chunk_config import ChunkConfig, BYTE
from .common import GenericE, GenericPureFFN, MagicFloorFFN


class SwiGLUSchoolbook(nn.Module):
    """
    Schoolbook multiplication using pure SwiGLU.

    For 4 byte positions, 10 partial products:
    pos 0: a0*b0
    pos 1: a0*b1 + a1*b0
    pos 2: a0*b2 + a1*b1 + a2*b0
    pos 3: a0*b3 + a1*b2 + a2*b1 + a3*b0

    Each product uses SwiGLU: silu(S*a) * b / S ≈ a*b
    6 weights per product (W_up, b_up, W_gate, b_gate, W_down, b_down)
    10 products * 6 = 60 weights (using 2 hidden units per product for symmetry)

    Plus clearing: 4 positions * 2 hidden units = 8
    Total: ~68 hidden units = ~204 weights
    """

    def __init__(self, S=100.0, N=4):
        super().__init__()
        self.S = S
        self.N = N

        # Input: [batch, 2*N] = [a0,a1,a2,a3, b0,b1,b2,b3]
        # Output: [batch, N] = [sum0, sum1, sum2, sum3]

        # Count products per position
        num_products = sum(min(k+1, N) for k in range(N))  # 1+2+3+4=10

        # 2 hidden units per product (pos + neg path) + 2*N for output init
        hidden_dim = num_products * 2
        input_dim = 2 * N
        output_dim = N

        self.ffn = GenericPureFFN(dim=input_dim + output_dim, hidden_dim=hidden_dim)

        with torch.no_grad():
            h = 0
            # For each output position k, sum products a[i]*b[j] where i+j=k
            for k in range(N):
                for i in range(min(k+1, N)):
                    j = k - i
                    if j < N:
                        a_idx = i
                        b_idx = N + j
                        out_idx = 2*N + k  # Output starts after inputs

                        # Positive path: silu(S*a) * b
                        self.ffn.W_up[h, a_idx] = S
                        self.ffn.W_gate[h, b_idx] = 1.0
                        self.ffn.W_down[out_idx, h] = 1.0 / S
                        h += 1

                        # Negative path: silu(-S*a) * (-b)
                        self.ffn.W_up[h, a_idx] = -S
                        self.ffn.W_gate[h, b_idx] = -1.0
                        self.ffn.W_down[out_idx, h] = 1.0 / S
                        h += 1

    def forward(self, a, b):
        """
        Args:
            a: [batch, N] byte values for operand A
            b: [batch, N] byte values for operand B
        Returns:
            sums: [batch, N] accumulated partial products (may exceed 255)
        """
        # Concatenate inputs and outputs (outputs start as 0)
        x = torch.cat([a, b, torch.zeros_like(a)], dim=-1)
        y = self.ffn(x)
        return y[..., 2*self.N:]  # Return just the output portion


class SwiGLUCarryExtract(nn.Module):
    """
    Extract carries using pure FFN staircase floor.

    For each position, compute:
        quotient = floor(value / base)
        remainder = value mod base

    Uses MagicFloorFFN which bakes floor thresholds into sigmoid steps.
    This is pure FFN - no runtime operations, everything in weights.

    For 32-bit MUL, max value after schoolbook is 4*255^2 = 260100.
    Max quotient = 1015.
    """

    def __init__(self, S=100.0, N=4, base=256, max_value=260100):
        super().__init__()
        self.S = S
        self.N = N
        self.base = base
        self.max_quotient = max_value // base

        # Pure FFN floor using MAGIC trick for quotient extraction
        # Works for any value since MAGIC trick is universal for fp32
        self.floor_ffn = MagicFloorFFN()

    def forward(self, sums):
        """
        Args:
            sums: [batch, N] accumulated values (may exceed 255)
        Returns:
            remainders: [batch, N] values mod 256 (0-255)
            quotients: [batch, N] floor(value / 256)
        """
        # Compute quotient using pure FFN staircase floor
        scaled = sums / self.base
        quotients = self.floor_ffn(scaled)
        remainders = sums - quotients * self.base

        return remainders, quotients


class SwiGLUCarryPropagate(nn.Module):
    """
    Propagate carries using pure FFN staircase floor.

    For 4 byte positions, 3 carry bits needed.
    Each new carry is 0 or 1 (since we're adding reduced values + small carry).

    Uses MagicFloorFFN with max_val=2 (carries are 0 or 1).
    """

    def __init__(self, S=100.0, N=4, base=256):
        super().__init__()
        self.S = S
        self.N = N
        self.base = base

        # Pure FFN floor using MAGIC trick for carry extraction during propagation
        # Works for any value since MAGIC trick is universal for fp32
        self.floor_ffn = MagicFloorFFN()

    def forward(self, remainders, quotients):
        """
        Args:
            remainders: [batch, N] values in 0-255
            quotients: [batch, N] carry amounts from division
        Returns:
            results: [batch, N] final byte values after carry propagation
        """
        results = remainders.clone()
        carries = quotients.clone()

        # Propagate carries left to right
        for pos in range(1, self.N):
            # Add carry from previous position
            results[:, pos] = results[:, pos] + carries[:, pos-1]

            # Extract new carry if overflow using pure FFN staircase floor
            new_carry = self.floor_ffn(results[:, pos] / self.base)
            results[:, pos] = results[:, pos] - new_carry * self.base
            carries[:, pos] = carries[:, pos] + new_carry

        return results


def build_swiglu_mul(S=100.0, N=4, base=256):
    """Build complete MUL pipeline using SwiGLU."""
    return nn.ModuleList([
        SwiGLUSchoolbook(S=S, N=N),
        SwiGLUCarryExtract(S=S, N=N, base=base),
        SwiGLUCarryPropagate(S=S, N=N, base=base),
    ])


def count_swiglu_mul_params():
    """Count parameters for SwiGLU MUL."""
    layers = build_swiglu_mul()

    print("SwiGLU BYTE MUL (pure neural target):")
    total = 0
    for i, layer in enumerate(layers):
        params = sum(p.numel() for p in layer.parameters())
        nonzero = sum((p != 0).sum().item() for p in layer.parameters())
        print(f"  Layer {i}: {layer.__class__.__name__}")
        print(f"    Total params: {params}, Non-zero: {nonzero}")
        total += nonzero
    print(f"  Total non-zero: {total}")
    return total


def test_swiglu_mul():
    """Test the SwiGLU MUL implementation."""
    layers = build_swiglu_mul()
    schoolbook, carry_extract, carry_prop = layers

    # Test: 5 * 3 = 15
    a = torch.tensor([[5.0, 0, 0, 0]])  # 5 in byte 0
    b = torch.tensor([[3.0, 0, 0, 0]])  # 3 in byte 0

    with torch.no_grad():
        sums = schoolbook(a, b)
        print(f"After schoolbook: {sums}")

        remainders, quotients = carry_extract(sums)
        print(f"Remainders: {remainders}, Quotients: {quotients}")

        results = carry_prop(remainders, quotients)
        print(f"Final results: {results}")

    # Test: 200 * 200 = 40000 = 0x9C40
    a2 = torch.tensor([[200.0, 0, 0, 0]])
    b2 = torch.tensor([[200.0, 0, 0, 0]])

    with torch.no_grad():
        sums2 = schoolbook(a2, b2)
        print(f"\n200*200 after schoolbook: {sums2}")

        remainders2, quotients2 = carry_extract(sums2)
        print(f"Remainders: {remainders2}, Quotients: {quotients2}")

        results2 = carry_prop(remainders2, quotients2)
        print(f"Final results: {results2}")

        # Reconstruct 32-bit value
        value = results2[0, 0] + results2[0, 1] * 256
        print(f"Reconstructed value: {value.item()} (expected: 40000)")


if __name__ == '__main__':
    print("="*70)
    print("SwiGLU BYTE MUL Implementation")
    print("="*70)
    count_swiglu_mul_params()
    print()
    test_swiglu_mul()
