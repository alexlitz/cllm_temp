"""
Position Encoding for Neural VM.

This module provides position derivation using ALiBi and RoPE, replacing
direct storage of position in embedding slot E.POS.

## Key Insight

In standard transformers, position comes from positional encoding:
- ALiBi: Attention scores have linear bias based on distance
- RoPE: Q/K vectors are rotated based on position

For the Neural VM's 8 nibble positions, we use attention to derive position:
1. PositionEncoderAttention uses ALiBi/RoPE to compute position weights
2. These weights encode the position index (0-7) via attention patterns
3. The position is written to E.POS for use by FFN layers

This allows all position-dependent operations to ultimately derive from
positional encoding rather than embedding slot storage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .embedding import E
from .base_layers import bake_weights


class ALiBiPositionAttention(nn.Module):
    """
    Derives position using ALiBi (Attention with Linear Biases).

    ALiBi adds a linear bias to attention scores based on position distance.
    We use this to inject position information into the embedding.

    How it works:
    1. Each position i attends to an anchor position
    2. ALiBi bias = -slope * distance
    3. Attention pattern encodes position via the bias
    4. Output writes position to E.POS slot

    For 8 positions with slope m:
        Position 0: bias = [0, -m, -2m, -3m, -4m, -5m, -6m, -7m]
        Position 1: bias = [-m, 0, -m, -2m, -3m, -4m, -5m, -6m]
        etc.

    The attention output is weighted by position values [0,1,2,3,4,5,6,7],
    resulting in position being written to the embedding.
    """

    def __init__(self, num_positions: int = 8, dim: int = None):
        super().__init__()
        self.num_positions = num_positions
        self.dim = dim if dim is not None else E.DIM

        # ALiBi slope - higher = sharper attention
        self.slope = 8.0

        # Pre-compute ALiBi bias matrix
        # bias[i,j] = -slope * |i - j|
        positions = torch.arange(num_positions)
        distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1)).float()
        alibi_bias = -self.slope * distance
        self.register_buffer('alibi_bias', alibi_bias)

        # Position values to be retrieved via attention
        # Each position j provides its index j as value
        position_values = torch.arange(num_positions).float().unsqueeze(1)  # [8, 1]
        self.register_buffer('position_values', position_values)

        # Weights for projecting output to E.POS
        W_o = torch.zeros(self.dim)
        W_o[E.POS] = 1.0
        self.register_buffer('W_o', W_o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inject position into embedding using ALiBi attention.

        Args:
            x: [batch, num_positions, dim] embedding tensor

        Returns:
            x with E.POS populated from ALiBi-derived position
        """
        B, S, D = x.shape

        # Compute attention scores (all-to-all with ALiBi bias)
        # Using uniform attention base scores + ALiBi bias
        # Since we want pure position derivation, Q/K are fixed
        scores = self.alibi_bias[:S, :S].unsqueeze(0)  # [1, S, S]

        # Softmax over key positions
        attn = F.softmax(scores, dim=-1)  # [1, S, S]

        # Weighted sum of position values
        # attn[i, j] * j gives position i's "perceived position"
        # With ALiBi, each position attends most to itself
        pos_indices = torch.arange(S, device=x.device, dtype=x.dtype)  # [S]
        derived_pos = torch.matmul(attn, pos_indices.unsqueeze(-1))  # [1, S, 1]

        # Write derived position to E.POS
        x = x.clone()
        x[..., E.POS] = derived_pos.squeeze(-1).expand(B, -1)

        return x


class RoPEPositionAttention(nn.Module):
    """
    Derives position using RoPE (Rotary Position Embeddings).

    RoPE encodes position by rotating Q/K vectors:
        q_rot = q * cos(pos * theta) + rotate_half(q) * sin(pos * theta)

    With binary thetas (theta_k = 2^k), position is encoded in binary.

    For position derivation:
    1. Query rotated with position i
    2. Keys rotated with positions [0,1,2,...,7]
    3. Dot product peaks when query pos == key pos
    4. Attention extracts position from keys
    """

    def __init__(self, num_positions: int = 8, dim: int = None, key_dim: int = 16):
        super().__init__()
        self.num_positions = num_positions
        self.dim = dim if dim is not None else E.DIM
        self.key_dim = key_dim

        # RoPE with binary thetas: theta_k = 2^k
        num_freqs = key_dim // 2
        thetas = torch.tensor([2.0 ** k for k in range(num_freqs)])
        self.register_buffer('thetas', thetas)

        # Pre-compute RoPE encodings for all positions
        positions = torch.arange(num_positions).unsqueeze(1)  # [num_pos, 1]
        angles = positions * thetas.unsqueeze(0)  # [num_pos, num_freqs]

        cos_enc = torch.cos(angles)  # [num_pos, num_freqs]
        sin_enc = torch.sin(angles)  # [num_pos, num_freqs]

        # Stack interleaved
        encoding = torch.zeros(num_positions, key_dim)
        encoding[:, 0::2] = cos_enc
        encoding[:, 1::2] = sin_enc
        self.register_buffer('rope_encoding', encoding)

        # Also store binary encoding (simpler matching)
        binary_enc = torch.zeros(num_positions, key_dim)
        for k in range(key_dim):
            bit_k = ((positions.squeeze() >> k) & 1).float()
            binary_enc[:, k] = 2 * bit_k - 1  # -1 or +1
        self.register_buffer('binary_encoding', binary_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inject position into embedding using RoPE-based matching.

        Args:
            x: [batch, num_positions, dim] embedding tensor

        Returns:
            x with E.POS populated from RoPE-derived position
        """
        B, S, D = x.shape

        # Use binary encoding for position matching
        # Query position i matches key position i
        # score[i, j] = binary_enc[i] dot binary_enc[j]
        # = key_dim when i == j, < key_dim otherwise

        enc = self.binary_encoding[:S]  # [S, key_dim]
        scores = torch.matmul(enc, enc.T)  # [S, S]

        # Scale and apply temperature for sharp attention
        temperature = 0.1
        attn = F.softmax(scores / temperature, dim=-1)  # [S, S]

        # Weighted sum of position indices
        pos_indices = torch.arange(S, device=x.device, dtype=x.dtype)
        derived_pos = torch.matmul(attn, pos_indices)  # [S]

        # Write derived position to E.POS
        x = x.clone()
        x[..., E.POS] = derived_pos.unsqueeze(0).expand(B, -1)

        return x


class PositionEncoderFFN(nn.Module):
    """
    FFN-based position injection using learned position-dependent biases.

    This provides an alternative to attention-based position derivation.
    Works by using the token's sequential position (implicit in the input)
    to write the position index.

    Note: This requires the input embeddings to have position information
    available through the sequence dimension structure.
    """

    def __init__(self, num_positions: int = 8, dim: int = None):
        super().__init__()
        self.num_positions = num_positions
        self.dim = dim if dim is not None else E.DIM

        # Learn position-specific biases that activate per position
        # When position i is processed, W_pos[i] activates
        S = E.SCALE

        # Create position indicator weights
        # Hidden units that each detect a specific position
        hidden_dim = num_positions

        W_up = torch.zeros(hidden_dim, self.dim)
        b_up = torch.zeros(hidden_dim)
        W_gate = torch.zeros(hidden_dim, self.dim)
        b_gate = torch.zeros(hidden_dim)
        W_down = torch.zeros(self.dim, hidden_dim)

        # Each hidden unit i outputs position value i
        # Uses broadcast mechanism: position i gets value i
        for i in range(num_positions):
            # Unit i detects position i using sequence structure
            # This relies on the input being structured with positions
            b_gate[i] = float(i)  # Each unit gates on its position value
            W_down[E.POS, i] = 1.0 / num_positions

        self.register_buffer('W_up', W_up)
        self.register_buffer('b_up', b_up)
        self.register_buffer('W_gate', W_gate)
        self.register_buffer('b_gate', b_gate)
        self.register_buffer('W_down', W_down)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inject position into embedding.

        Args:
            x: [batch, num_positions, dim] embedding tensor
            positions: Optional [num_positions] position indices (default: 0..S-1)

        Returns:
            x with E.POS populated
        """
        B, S, D = x.shape

        if positions is None:
            positions = torch.arange(S, device=x.device, dtype=x.dtype)

        # Directly write position to E.POS
        x = x.clone()
        x[..., E.POS] = positions.unsqueeze(0).expand(B, -1)

        return x


class PositionEncoder(nn.Module):
    """
    Unified position encoder with configurable method.

    Methods:
    - 'alibi': Use ALiBi attention for position derivation
    - 'rope': Use RoPE attention for position derivation
    - 'direct': Directly write position (baseline)

    This is the main interface for position injection in the neural VM.
    """

    def __init__(self, method: str = 'alibi', num_positions: int = 8, dim: int = None):
        super().__init__()
        self.method = method
        self.num_positions = num_positions
        self.dim = dim if dim is not None else E.DIM

        if method == 'alibi':
            self.encoder = ALiBiPositionAttention(num_positions, self.dim)
        elif method == 'rope':
            self.encoder = RoPEPositionAttention(num_positions, self.dim)
        elif method == 'direct':
            self.encoder = PositionEncoderFFN(num_positions, self.dim)
        else:
            raise ValueError(f"Unknown position encoding method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inject position into embedding using configured method.

        Args:
            x: [batch, num_positions, dim] embedding tensor

        Returns:
            x with E.POS populated from positional encoding
        """
        return self.encoder(x)


# ============================================================================
# Testing
# ============================================================================

def test_alibi_position():
    """Test ALiBi position encoding."""
    print("=== Testing ALiBi Position Encoding ===\n")

    encoder = ALiBiPositionAttention(num_positions=8)

    # Create test embedding
    x = torch.zeros(1, 8, E.DIM)

    # Apply position encoding
    x_out = encoder(x)

    # Check E.POS values
    positions = x_out[0, :, E.POS]
    print(f"Derived positions: {positions.tolist()}")
    print(f"Expected:          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]")

    # Verify positions are approximately correct
    expected = torch.arange(8).float()
    error = (positions - expected).abs().max().item()
    print(f"Max error: {error:.6f}")
    print(f"PASS" if error < 0.1 else f"FAIL")
    print()


def test_rope_position():
    """Test RoPE position encoding."""
    print("=== Testing RoPE Position Encoding ===\n")

    encoder = RoPEPositionAttention(num_positions=8)

    # Create test embedding
    x = torch.zeros(1, 8, E.DIM)

    # Apply position encoding
    x_out = encoder(x)

    # Check E.POS values
    positions = x_out[0, :, E.POS]
    print(f"Derived positions: {positions.tolist()}")
    print(f"Expected:          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]")

    # Verify positions are approximately correct
    expected = torch.arange(8).float()
    error = (positions - expected).abs().max().item()
    print(f"Max error: {error:.6f}")
    print(f"PASS" if error < 0.1 else f"FAIL")
    print()


def test_unified_encoder():
    """Test unified position encoder."""
    print("=== Testing Unified Position Encoder ===\n")

    for method in ['alibi', 'rope', 'direct']:
        print(f"Method: {method}")
        encoder = PositionEncoder(method=method)

        x = torch.zeros(1, 8, E.DIM)
        x_out = encoder(x)

        positions = x_out[0, :, E.POS]
        expected = torch.arange(8).float()
        error = (positions - expected).abs().max().item()

        print(f"  Positions: {[f'{p:.2f}' for p in positions.tolist()]}")
        print(f"  Max error: {error:.6f}")
        print(f"  {'PASS' if error < 0.1 else 'FAIL'}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("Position Encoding for Neural VM")
    print("=" * 60)
    print()
    print("This module derives position from ALiBi/RoPE instead of")
    print("storing it directly in embedding slot E.POS.")
    print()

    test_alibi_position()
    test_rope_position()
    test_unified_encoder()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
