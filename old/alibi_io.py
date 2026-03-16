#!/usr/bin/env python3
"""
ALiBi-Based I/O for Neural VM

Uses Attention with Linear Biases (ALiBi) for position-aware I/O:
- GETCHAR: Reads from token stream using position-based attention
- PUTCHAR: Writes to token stream (generates next token)

No special token tagging needed - position comes from ALiBi naturally.

The key insight:
  Standard ALiBi: score[j] = Q·K[j] - m * (i - j)     // peaks at current pos
  Reading ALiBi:  score[j] = Q·K[j] - m * |target - j| // peaks at target pos

Where target = anchor_pos + 1 + read_offset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class ALiBiIOConfig:
    """Configuration for ALiBi I/O attention."""
    dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    max_seq_len: int = 2048
    alibi_slope_base: float = 2.0  # Base for computing per-head slopes


class ALiBiIOState:
    """
    I/O state tracked in the embedding.

    Embedding slots:
      IO_READ_ANCHOR:   Position of <NEED_INPUT/> token (input starts after this)
      IO_READ_OFFSET:   Number of input chars read so far
      IO_WRITE_ANCHOR:  Position where output started
      IO_WRITE_OFFSET:  Number of output chars written
      IO_CHAR:          Current character (input or output)
      IO_MODE:          0=idle, 1=reading, 2=writing
    """
    # Embedding slot indices (relative to IO region start)
    READ_ANCHOR = 0
    READ_OFFSET = 1
    WRITE_ANCHOR = 2
    WRITE_OFFSET = 3
    CHAR_VAL = 4      # 8 slots for char value (nibbles)
    MODE = 12
    OUTPUT_READY = 13
    NEED_INPUT = 14
    PROGRAM_END = 15

    NUM_SLOTS = 16


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for each attention head.

    Following the ALiBi paper: slopes = 2^(-8/n * [1, 2, ..., n])
    for n heads.
    """
    # Standard ALiBi slope computation
    ratio = 2 ** (-8 / num_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
    return slopes


class ALiBiPositionalBias(nn.Module):
    """
    Computes ALiBi positional biases for attention.

    Standard: bias[i,j] = -slope * (i - j)  for causal (j <= i)
    Custom:   bias[i,j] = -slope * |target - j|  for reading from target
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Register slopes as buffer
        slopes = get_alibi_slopes(num_heads)
        self.register_buffer('slopes', slopes.view(num_heads, 1, 1))

        # Precompute position indices
        positions = torch.arange(max_seq_len)
        self.register_buffer('positions', positions)

    def forward_standard(self, seq_len: int) -> torch.Tensor:
        """
        Standard ALiBi bias: -slope * (i - j)

        Returns: [num_heads, seq_len, seq_len]
        """
        # Distance matrix: i - j
        positions = self.positions[:seq_len]
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq, seq]

        # Apply slopes: [heads, seq, seq]
        bias = -self.slopes * distance.unsqueeze(0).float()

        # Causal mask (can only attend to past)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=bias.device), diagonal=1)
        bias = bias - causal_mask * 1e9

        return bias

    def forward_targeted(self, seq_len: int, target_positions: torch.Tensor,
                         head_idx: int = 0) -> torch.Tensor:
        """
        Targeted ALiBi bias: -slope * |target - j|

        Args:
            seq_len: Sequence length
            target_positions: [batch] target position for each item
            head_idx: Which head's slope to use

        Returns: [batch, seq_len] bias for attending to each position
        """
        positions = self.positions[:seq_len].float()  # [seq]
        targets = target_positions.unsqueeze(-1).float()  # [batch, 1]

        # Distance from target: |target - j|
        distance = torch.abs(targets - positions)  # [batch, seq]

        # Apply slope
        slope = self.slopes[head_idx, 0, 0]
        bias = -slope * distance

        return bias


class ALiBiIOAttention(nn.Module):
    """
    Attention layer with ALiBi-based I/O support.

    Features:
    - Standard self-attention with ALiBi positional bias
    - Special "read head" that attends to target position for GETCHAR
    - Tracks read/write anchors and offsets in the embedding

    The read head computes:
        target = read_anchor + 1 + read_offset
        attention[j] ∝ e^(-slope * |target - j|)

    This peaks sharply at the target position, reading that character.
    """

    def __init__(self, config: ALiBiIOConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        # Projections
        self.q_proj = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.dim, bias=False)

        # ALiBi positional bias
        self.alibi = ALiBiPositionalBias(config.num_heads, config.max_seq_len)

        # IO state slots in embedding (at the end)
        self.io_start = config.dim - ALiBiIOState.NUM_SLOTS

    def _get_io_state(self, x: torch.Tensor) -> dict:
        """Extract I/O state from embedding."""
        io = x[..., self.io_start:]
        return {
            'read_anchor': io[..., ALiBiIOState.READ_ANCHOR],
            'read_offset': io[..., ALiBiIOState.READ_OFFSET],
            'write_anchor': io[..., ALiBiIOState.WRITE_ANCHOR],
            'write_offset': io[..., ALiBiIOState.WRITE_OFFSET],
            'char_val': io[..., ALiBiIOState.CHAR_VAL:ALiBiIOState.CHAR_VAL+8],
            'mode': io[..., ALiBiIOState.MODE],
            'output_ready': io[..., ALiBiIOState.OUTPUT_READY],
            'need_input': io[..., ALiBiIOState.NEED_INPUT],
            'program_end': io[..., ALiBiIOState.PROGRAM_END],
        }

    def _set_io_state(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        """Set I/O state in embedding."""
        out = x.clone()
        io_region = out[..., self.io_start:]

        if 'read_anchor' in state:
            io_region[..., ALiBiIOState.READ_ANCHOR] = state['read_anchor']
        if 'read_offset' in state:
            io_region[..., ALiBiIOState.READ_OFFSET] = state['read_offset']
        if 'write_anchor' in state:
            io_region[..., ALiBiIOState.WRITE_ANCHOR] = state['write_anchor']
        if 'write_offset' in state:
            io_region[..., ALiBiIOState.WRITE_OFFSET] = state['write_offset']
        if 'char_val' in state:
            io_region[..., ALiBiIOState.CHAR_VAL:ALiBiIOState.CHAR_VAL+8] = state['char_val']
        if 'mode' in state:
            io_region[..., ALiBiIOState.MODE] = state['mode']
        if 'output_ready' in state:
            io_region[..., ALiBiIOState.OUTPUT_READY] = state['output_ready']
        if 'need_input' in state:
            io_region[..., ALiBiIOState.NEED_INPUT] = state['need_input']
        if 'program_end' in state:
            io_region[..., ALiBiIOState.PROGRAM_END] = state['program_end']

        return out

    def forward(self, x: torch.Tensor,
                is_getchar: Optional[torch.Tensor] = None,
                is_putchar: Optional[torch.Tensor] = None,
                current_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with ALiBi attention and I/O handling.

        Args:
            x: Input embeddings [batch, seq, dim]
            is_getchar: [batch, seq] mask for GETCHAR operations
            is_putchar: [batch, seq] mask for PUTCHAR operations
            current_pos: [batch, seq] current position indices

        Returns:
            Output embeddings with I/O state updated
        """
        batch, seq_len, dim = x.shape

        # Standard attention projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores: [batch, heads, seq, seq]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add ALiBi bias
        alibi_bias = self.alibi.forward_standard(seq_len)  # [heads, seq, seq]
        scores = scores + alibi_bias.unsqueeze(0)

        # Handle GETCHAR with targeted attention
        if is_getchar is not None and is_getchar.any():
            io_state = self._get_io_state(x)

            # Compute target positions: anchor + 1 + offset
            target_pos = io_state['read_anchor'] + 1 + io_state['read_offset']

            # For positions where GETCHAR is active, modify attention
            # Use head 0 as the "read head"
            read_bias = self.alibi.forward_targeted(seq_len, target_pos.view(-1), head_idx=0)
            read_bias = read_bias.view(batch, seq_len, seq_len)

            # Apply targeted attention for GETCHAR positions
            is_getchar_expanded = is_getchar.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
            scores[:, 0:1, :, :] = torch.where(
                is_getchar_expanded > 0.5,
                read_bias.unsqueeze(1),  # Use targeted bias
                scores[:, 0:1, :, :]     # Keep standard bias
            )

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Attend
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        # Residual connection
        output = x + output

        # Update I/O state for GETCHAR
        if is_getchar is not None and is_getchar.any():
            io_state = self._get_io_state(output)

            # The attention has placed the read character in the output
            # Extract it from the read head's output (first head)
            read_char = attn_output[:, :, :self.head_dim]  # First head's output

            # Increment read offset
            new_offset = io_state['read_offset'] + is_getchar.float()

            output = self._set_io_state(output, {
                'read_offset': new_offset,
            })

        # Update I/O state for PUTCHAR
        if is_putchar is not None and is_putchar.any():
            io_state = self._get_io_state(output)

            # Increment write offset
            new_offset = io_state['write_offset'] + is_putchar.float()

            output = self._set_io_state(output, {
                'write_offset': new_offset,
                'output_ready': is_putchar.float(),
            })

        return output


class ALiBiIOLayer(nn.Module):
    """
    Complete ALiBi I/O layer combining attention and FFN.

    This handles:
    - GETCHAR: Read from input via targeted attention
    - PUTCHAR: Write to output (sets output_ready flag)
    - Position tracking via anchors and offsets

    Token Stream Example:

        Position:  0    1    2    3         4    5    6    7    8
        Tokens:    Hi   !    <NI> A    l    i    c    e    \n
                             ^anchor

        GETCHAR #1: target=3+1+0=4 → reads 'A'
        GETCHAR #2: target=3+1+1=5 → reads 'l'
        ...

    The model generates output tokens normally. PUTCHAR just signals
    that a character should be emitted. GETCHAR uses ALiBi-targeted
    attention to read from the input portion of the sequence.
    """

    def __init__(self, config: ALiBiIOConfig):
        super().__init__()
        self.attention = ALiBiIOAttention(config)

        # FFN for processing after attention
        self.ffn_up = nn.Linear(config.dim, config.dim * 4, bias=False)
        self.ffn_down = nn.Linear(config.dim * 4, config.dim, bias=False)

        self.io_start = config.dim - ALiBiIOState.NUM_SLOTS

    def forward(self, x: torch.Tensor,
                opcode: Optional[torch.Tensor] = None,
                current_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with I/O handling.

        Args:
            x: Input embeddings [batch, seq, dim]
            opcode: [batch, seq] opcode for each position (40=GETCHAR, 41=PUTCHAR)
            current_pos: [batch, seq] position indices
        """
        GETCHAR_OPCODE = 40
        PUTCHAR_OPCODE = 41

        is_getchar = None
        is_putchar = None

        if opcode is not None:
            is_getchar = (opcode == GETCHAR_OPCODE).float()
            is_putchar = (opcode == PUTCHAR_OPCODE).float()

        # Attention with I/O handling
        x = self.attention(x, is_getchar=is_getchar, is_putchar=is_putchar,
                          current_pos=current_pos)

        # FFN
        h = self.ffn_up(x)
        h = F.silu(h)
        h = self.ffn_down(h)
        x = x + h

        return x


def encode_char_to_nibbles(char: int) -> torch.Tensor:
    """Encode a character as 8 nibbles (4-bit values)."""
    nibbles = torch.zeros(8)
    for i in range(8):
        nibbles[i] = (char >> (i * 4)) & 0xF
    return nibbles


def decode_nibbles_to_char(nibbles: torch.Tensor) -> int:
    """Decode 8 nibbles to a character."""
    char = 0
    for i in range(8):
        char |= (int(nibbles[i].item()) & 0xF) << (i * 4)
    return char


def demo_alibi_io():
    """Demonstrate ALiBi-based I/O."""
    print("=" * 70)
    print("ALiBi I/O Demo")
    print("=" * 70)

    config = ALiBiIOConfig(dim=128, num_heads=4, head_dim=32)
    layer = ALiBiIOLayer(config)
    layer.eval()

    # Simulate a sequence: "Hi!<NEED_INPUT/>Alice"
    # Positions:           0  1  2  3  4  5  6  7  8
    seq_len = 9
    batch = 1

    # Create embeddings (simplified - just position encoding)
    x = torch.randn(batch, seq_len, config.dim)

    # Set up I/O state
    io_start = config.dim - ALiBiIOState.NUM_SLOTS

    # Anchor at position 2 (where <NEED_INPUT/> would be)
    x[:, :, io_start + ALiBiIOState.READ_ANCHOR] = 2.0
    x[:, :, io_start + ALiBiIOState.READ_OFFSET] = 0.0  # Start at offset 0

    # Encode "Alice" at positions 3-7
    input_text = "Alice"
    for i, c in enumerate(input_text):
        char_nibbles = encode_char_to_nibbles(ord(c))
        # Store in the embedding at that position
        x[:, 3 + i, io_start + ALiBiIOState.CHAR_VAL:io_start + ALiBiIOState.CHAR_VAL + 8] = char_nibbles

    print(f"\nSequence: 'Hi!<NI>Alice'")
    print(f"Input anchor at position 2")
    print(f"Input characters at positions 3-7")
    print()

    # Simulate GETCHAR operations
    for read_idx in range(5):
        # Set read offset
        x[:, :, io_start + ALiBiIOState.READ_OFFSET] = float(read_idx)

        # Mark position 8 as GETCHAR
        opcode = torch.zeros(batch, seq_len)
        opcode[:, 8] = 40  # GETCHAR at current position

        # Forward pass
        output = layer(x, opcode=opcode)

        # Read offset should have incremented
        new_offset = output[:, 8, io_start + ALiBiIOState.READ_OFFSET].item()

        # The target position
        target = 2 + 1 + read_idx  # anchor + 1 + offset = 3, 4, 5, 6, 7
        expected_char = input_text[read_idx] if read_idx < len(input_text) else '?'

        print(f"GETCHAR #{read_idx + 1}: offset={read_idx} → target_pos={target} → '{expected_char}'")

    print()
    print("=" * 70)
    print("ALiBi provides position math without token tagging!")
    print("=" * 70)


if __name__ == "__main__":
    demo_alibi_io()
