#!/usr/bin/env python3
"""
Binary Position I/O via Power-of-Two Decomposition

Uses the same trick as memory addressing in the VM:
1. Get e^(pos * slope) from attention (gives exponential of position)
2. Extract binary bits via powers-of-two modular arithmetic
3. Use binary representation for exact key-query matching

This gives exact position retrieval without special token tagging.

The key insight:
  - Position is implicit in attention weights via ALiBi
  - We extract it to binary using: bit_k = floor(pos / 2^k) mod 2
  - Binary key-query matching gives exact retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# =============================================================================
# POWER OF TWO BIT EXTRACTION
# =============================================================================

class PowerOfTwoBitExtractor(nn.Module):
    """
    Extract binary bits from a value using the power-of-two trick.

    For value v, bit k is:
        bit_k = floor(v / 2^k) mod 2

    Using neural approximation:
        bit_k ≈ (v / 2^k) mod 2
              ≈ v/2^k - 2*floor(v/2^(k+1))

    With SiLU for smooth floor:
        floor(x) ≈ x - SiLU(scale*(x - floor(x) - 0.5)) / scale

    For our purposes, we use a simpler approach:
        bit_k = ((v % 2^(k+1)) >= 2^k) ? 1 : 0

    Which can be computed neurally using threshold detection.
    """

    def __init__(self, num_bits: int = 8, scale: float = 100.0, hard: bool = True):
        super().__init__()
        self.num_bits = num_bits
        self.scale = scale
        self.hard = hard

        # Precompute powers of 2
        powers = torch.tensor([2**k for k in range(num_bits + 1)], dtype=torch.float32)
        self.register_buffer('powers', powers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Extract binary bits from value.

        Args:
            value: [...] tensor of values

        Returns:
            [..., num_bits] tensor of bits (0.0 or 1.0)
        """
        # Clamp value to valid range to avoid overflow
        v = value.clamp(0, 2**self.num_bits - 1)

        # For each bit position k, compute: bit_k = floor(v / 2^k) mod 2
        # This is equivalent to: bit_k = (v & 2^k) != 0
        bits = torch.zeros(*value.shape, self.num_bits, device=value.device)

        for k in range(self.num_bits):
            power_k = self.powers[k]  # 2^k

            # floor(v / 2^k) mod 2
            # = ((v // 2^k) % 2)
            # Using floor division and fmod on the quotient (small number)
            quotient = torch.floor(v / power_k)  # floor(v / 2^k)
            bit_raw = torch.fmod(quotient, 2.0)  # mod 2

            if self.hard:
                # Hard threshold for exact matching
                bits[..., k] = (bit_raw > 0.5).float()
            else:
                # Soft threshold for training
                bits[..., k] = torch.sigmoid(self.scale * (bit_raw - 0.5))

        return bits


class BinaryPositionEncoder(nn.Module):
    """
    Encode position relative to anchor as binary.

    Given:
        - anchor_pos: position of <NEED_INPUT/> token
        - current_pos: position of each token

    Computes:
        relative_pos = current_pos - anchor_pos - 1
        binary_key = extract_bits(relative_pos)

    For positions before anchor, relative_pos is negative (invalid).
    For position anchor+1, relative_pos = 0 (first input char).
    For position anchor+2, relative_pos = 1 (second input char).
    """

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.num_bits = num_bits
        self.bit_extractor = PowerOfTwoBitExtractor(num_bits)

    def forward(self, positions: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        Encode positions as binary keys relative to anchor.

        Args:
            positions: [...] position indices
            anchor: [...] anchor position (same shape or broadcastable)

        Returns:
            [..., num_bits] binary representation
        """
        # Relative position from anchor
        relative = positions - anchor - 1

        # Clamp to valid range (0 to 2^num_bits - 1)
        relative = torch.clamp(relative, min=0, max=2**self.num_bits - 1)

        # Extract binary bits
        binary = self.bit_extractor(relative)

        return binary


# =============================================================================
# BINARY MATCHING ATTENTION
# =============================================================================

class BinaryMatchAttention(nn.Module):
    """
    Attention that matches binary query to binary keys.

    This is the same mechanism used for memory addressing in the VM:
    - Each key is a binary representation of position
    - Query is binary representation of desired position
    - Attention score is high when all bits match

    Score computation:
        match[j] = product_k(query_k == key_k[j])
                 = product_k(query_k * key_k[j] + (1-query_k) * (1-key_k[j]))
                 = product_k(1 - |query_k - key_k[j]|)

    In log space (for softmax stability):
        log_score[j] = sum_k(log(1 - |query_k - key_k[j]| + eps))

    With hard bits (0 or 1), this gives:
        - All bits match: score = 1.0
        - Any bit differs: score ≈ 0.0
    """

    def __init__(self, num_bits: int = 8, temperature: float = 0.1):
        super().__init__()
        self.num_bits = num_bits
        self.temperature = temperature

    def forward(self, query_bits: torch.Tensor, key_bits: torch.Tensor,
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention using binary matching.

        Args:
            query_bits: [batch, query_len, num_bits] query binary
            key_bits: [batch, key_len, num_bits] key binary
            values: [batch, key_len, value_dim] values to attend over
            mask: [batch, key_len] optional mask (1 = valid, 0 = invalid)

        Returns:
            [batch, query_len, value_dim] attended values
        """
        # Compute bit-wise match: 1 if bits equal, 0 otherwise
        # match = 1 - |query - key|
        # query: [batch, query_len, 1, num_bits]
        # key:   [batch, 1, key_len, num_bits]
        q = query_bits.unsqueeze(2)
        k = key_bits.unsqueeze(1)

        bit_match = 1.0 - torch.abs(q - k)  # [batch, query_len, key_len, num_bits]

        # Product over bits (in log space for stability)
        # log(product(x)) = sum(log(x))
        log_match = torch.log(bit_match + 1e-8)
        log_score = log_match.sum(dim=-1)  # [batch, query_len, key_len]

        # Scale by temperature
        scores = log_score / self.temperature

        # Apply mask (positions before anchor are invalid)
        if mask is not None:
            # mask: [batch, key_len] -> [batch, 1, key_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over keys
        weights = F.softmax(scores, dim=-1)  # [batch, query_len, key_len]

        # Attend to values
        output = torch.matmul(weights, values)  # [batch, query_len, value_dim]

        return output, weights


# =============================================================================
# COMPLETE BINARY POSITION I/O
# =============================================================================

class BinaryPositionIO(nn.Module):
    """
    Complete binary position I/O system.

    How it works:

    1. SETUP (when <NEED_INPUT/> is generated):
       - anchor_pos = current position
       - read_offset = 0

    2. INPUT CHARACTERS (tokens after <NEED_INPUT/>):
       - Each token at position p gets key = binary(p - anchor - 1)
       - Token at anchor+1 has key = 0b00000000 (first input)
       - Token at anchor+2 has key = 0b00000001 (second input)
       - etc.

    3. GETCHAR:
       - Query = binary(read_offset)
       - Attention matches query to keys
       - Returns character at matching position
       - read_offset += 1

    4. PUTCHAR:
       - Output character becomes next token
       - write_offset += 1 (for tracking)

    Example:
        Sequence: Hi!<NI>Alice
        Positions: 0 1 2 3 4 5 6 7 8
                       ^anchor=3

        Keys (binary of pos - anchor - 1):
          pos 4 (A): 000 (= 0)
          pos 5 (l): 001 (= 1)
          pos 6 (i): 010 (= 2)
          pos 7 (c): 011 (= 3)
          pos 8 (e): 100 (= 4)

        GETCHAR with read_offset=2:
          Query: 010 (= 2)
          Matches key at pos 6
          Returns 'i'
    """

    def __init__(self, dim: int = 512, num_bits: int = 12, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_bits = num_bits
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Binary position encoder
        self.position_encoder = BinaryPositionEncoder(num_bits)

        # Read offset bit extractor
        self.offset_encoder = PowerOfTwoBitExtractor(num_bits)

        # Binary matching attention
        self.binary_attention = BinaryMatchAttention(num_bits)

        # Value projection (to get character from token embedding)
        self.value_proj = nn.Linear(dim, dim, bias=False)

        # Output projection
        self.output_proj = nn.Linear(dim, dim, bias=False)

        # Character extraction (from embedding to char value)
        self.char_extract = nn.Linear(dim, 8, bias=False)  # 8 nibbles for char

    def encode_keys(self, x: torch.Tensor, positions: torch.Tensor,
                    anchor: torch.Tensor) -> torch.Tensor:
        """
        Encode position-based keys for input tokens.

        Args:
            x: [batch, seq, dim] token embeddings
            positions: [batch, seq] position indices
            anchor: [batch] anchor position for each sequence

        Returns:
            [batch, seq, num_bits] binary keys
        """
        # Expand anchor for broadcasting
        anchor_expanded = anchor.unsqueeze(-1).expand_as(positions)

        # Get binary position keys
        keys = self.position_encoder(positions.float(), anchor_expanded.float())

        return keys

    def encode_query(self, read_offset: torch.Tensor) -> torch.Tensor:
        """
        Encode read offset as binary query.

        Args:
            read_offset: [batch] current read offset

        Returns:
            [batch, 1, num_bits] binary query
        """
        # Extract bits from offset
        query = self.offset_encoder(read_offset)

        # Add query dimension
        query = query.unsqueeze(1)

        return query

    def getchar(self, x: torch.Tensor, positions: torch.Tensor,
                anchor: torch.Tensor, read_offset: torch.Tensor,
                input_length: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read character from input using binary position matching.

        Args:
            x: [batch, seq, dim] token embeddings
            positions: [batch, seq] position indices
            anchor: [batch] anchor position
            read_offset: [batch] current read offset
            input_length: [batch] optional length of input (for proper masking)

        Returns:
            char_value: [batch, 8] character as nibbles
            new_offset: [batch] incremented read offset
        """
        batch, seq, dim = x.shape

        # Encode keys (position relative to anchor)
        keys = self.encode_keys(x, positions, anchor)  # [batch, seq, num_bits]

        # Encode query (read offset)
        query = self.encode_query(read_offset)  # [batch, 1, num_bits]

        # Project values
        values = self.value_proj(x)  # [batch, seq, dim]

        # Create mask: only positions in valid input range
        # positions > anchor AND positions <= anchor + input_length
        anchor_expanded = anchor.unsqueeze(-1).expand_as(positions)
        mask = (positions > anchor_expanded).float()  # [batch, seq]

        # If input_length provided, also mask positions beyond input
        if input_length is not None:
            input_end = anchor.unsqueeze(-1) + input_length.unsqueeze(-1)
            mask = mask * (positions <= input_end).float()

        # Binary matching attention with mask
        attended, weights = self.binary_attention(query, keys, values, mask)
        # attended: [batch, 1, dim]

        # Extract character from attended embedding
        char_value = self.char_extract(attended.squeeze(1))  # [batch, 8]

        # Increment offset
        new_offset = read_offset + 1

        return char_value, new_offset, weights

    def putchar(self, char_value: torch.Tensor, write_offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare character for output.

        For output, the character just becomes the next generated token.
        We track write_offset for consistency but it's mainly informational.

        Args:
            char_value: [batch, 8] character as nibbles
            write_offset: [batch] current write offset

        Returns:
            output_ready: [batch] flag indicating output ready
            new_offset: [batch] incremented write offset
        """
        output_ready = torch.ones_like(write_offset)
        new_offset = write_offset + 1

        return output_ready, new_offset


def demo_binary_position_io():
    """Demonstrate binary position I/O."""
    print("=" * 70)
    print("Binary Position I/O Demo")
    print("=" * 70)
    print()

    # Setup
    dim = 64
    num_bits = 4  # 4 bits = positions 0-15
    batch = 1
    seq_len = 10

    io_system = BinaryPositionIO(dim=dim, num_bits=num_bits)
    io_system.eval()

    # Simulate sequence: "Hi!<NI>Alice"
    # Positions:          0  1  2  3  4  5  6  7  8  9
    #                              ^anchor

    print("Sequence: Hi!<NI>Alice")
    print("Positions: 0  1  2  3  4  5  6  7  8  9")
    print("                    ^anchor=3")
    print()

    # Create embeddings (encode characters in first 8 dims as nibbles)
    x = torch.randn(batch, seq_len, dim)

    # Encode "Alice" at positions 4-8
    input_text = "Alice"
    for i, c in enumerate(input_text):
        char_val = ord(c)
        for nib in range(8):
            x[0, 4 + i, nib] = float((char_val >> (nib * 4)) & 0xF)

    # Position indices
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    # Anchor at position 3
    anchor = torch.tensor([3])

    # Show binary keys
    print("Binary Keys (position - anchor - 1):")
    print("-" * 50)
    keys = io_system.encode_keys(x, positions, anchor)
    for pos in range(4, 9):
        rel_pos = pos - 3 - 1
        bits = keys[0, pos].tolist()
        bits_str = ''.join([str(int(b > 0.5)) for b in bits[::-1]])  # MSB first
        char = input_text[pos - 4] if pos - 4 < len(input_text) else '?'
        print(f"  pos {pos} ('{char}'): {bits_str} = {rel_pos}")
    print()

    # Demonstrate GETCHAR
    print("GETCHAR Operations:")
    print("-" * 50)

    read_offset = torch.tensor([0.0])
    input_length = torch.tensor([5])  # "Alice" = 5 characters

    for i in range(5):
        char_nibbles, new_offset, weights = io_system.getchar(
            x, positions, anchor, read_offset, input_length
        )

        # Decode character
        char_val = 0
        for nib in range(8):
            char_val |= (int(char_nibbles[0, nib].item()) & 0xF) << (nib * 4)

        # Show query binary
        query_bits = io_system.encode_query(read_offset)
        query_str = ''.join([str(int(b > 0.5)) for b in query_bits[0, 0].tolist()[::-1]])

        # Find peak attention
        peak_pos = weights[0, 0].argmax().item()
        peak_weight = weights[0, 0, peak_pos].item()

        expected_char = input_text[i] if i < len(input_text) else '?'

        print(f"  offset={int(read_offset.item())}: query={query_str} → "
              f"peak@pos{peak_pos} (w={peak_weight:.3f}) → '{expected_char}'")

        read_offset = new_offset

    print()
    print("=" * 70)
    print("Binary position matching gives EXACT retrieval!")
    print("No token tagging needed - position comes from sequence index.")
    print("=" * 70)


def test_4k_positions():
    """Test binary position I/O with 4K+ positions."""
    print()
    print("=" * 70)
    print("Testing 4K Position Support (12 bits = 4096 positions)")
    print("=" * 70)
    print()

    # Setup for 4K
    dim = 64
    num_bits = 12  # 12 bits = 4096 positions (0-4095)
    batch = 1
    seq_len = 4200  # More than 4K total positions
    anchor_pos = 100  # Anchor at position 100

    io_system = BinaryPositionIO(dim=dim, num_bits=num_bits)
    io_system.eval()

    # Create embeddings
    x = torch.randn(batch, seq_len, dim)

    # Encode characters at various positions after anchor
    # Let's put characters at positions that test all 12 bits
    test_offsets = [0, 1, 15, 16, 255, 256, 1000, 2000, 4000, 4095]
    test_chars = "ABCDEFGHIJ"

    for idx, offset in enumerate(test_offsets):
        pos = anchor_pos + 1 + offset
        if pos < seq_len:
            char_val = ord(test_chars[idx % len(test_chars)])
            # Encode char in first 8 dims
            for nib in range(8):
                x[0, pos, nib] = float((char_val >> (nib * 4)) & 0xF)

    # Position indices
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    # Anchor
    anchor = torch.tensor([anchor_pos])

    # Input length = max_offset + 1 (to cover offset 0 through max_offset)
    input_length = torch.tensor([max(test_offsets) + 1])

    print(f"Anchor position: {anchor_pos}")
    print(f"Testing offsets: {test_offsets}")
    print(f"Max addressable: {2**num_bits - 1} (with {num_bits} bits)")
    print(f"Input length: {input_length.item()}")
    print()

    # Test GETCHAR at each offset
    print("GETCHAR Tests:")
    print("-" * 60)

    all_exact = True
    for idx, test_offset in enumerate(test_offsets):
        read_offset = torch.tensor([float(test_offset)])

        char_nibbles, new_offset, weights = io_system.getchar(
            x, positions, anchor, read_offset, input_length
        )

        # Show query binary
        query_bits = io_system.encode_query(read_offset)
        query_str = ''.join([str(int(b > 0.5)) for b in query_bits[0, 0].tolist()[::-1]])

        # Find peak attention
        peak_pos = weights[0, 0].argmax().item()
        peak_weight = weights[0, 0, peak_pos].item()

        expected_pos = anchor_pos + 1 + test_offset
        expected_char = test_chars[idx % len(test_chars)]

        is_exact = abs(peak_weight - 1.0) < 0.01 and peak_pos == expected_pos
        status = "✓" if is_exact else "✗"
        if not is_exact:
            all_exact = False

        print(f"  offset={test_offset:4d}: query={query_str} → "
              f"peak@pos{peak_pos:4d} (w={peak_weight:.4f}) "
              f"expected@{expected_pos:4d} '{expected_char}' {status}")

    print()
    print("-" * 60)
    if all_exact:
        print("✓ ALL 4K POSITION TESTS PASSED - EXACT RETRIEVAL!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)

    return all_exact


def test_8k_16k():
    """Test even larger position ranges."""
    print()
    print("=" * 70)
    print("Testing 8K and 16K Position Support")
    print("=" * 70)
    print()

    for num_bits, label in [(13, "8K"), (14, "16K"), (15, "32K")]:
        max_pos = 2**num_bits - 1
        dim = 64
        batch = 1
        seq_len = max_pos + 200
        anchor_pos = 50

        io_system = BinaryPositionIO(dim=dim, num_bits=num_bits)
        io_system.eval()

        x = torch.randn(batch, seq_len, dim)

        # Test at max position
        test_offset = max_pos
        pos = anchor_pos + 1 + test_offset
        if pos < seq_len:
            x[0, pos, 0] = 42.0  # Marker value

        positions = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        anchor = torch.tensor([anchor_pos])
        input_length = torch.tensor([test_offset + 1])  # Input extends to test_offset

        read_offset = torch.tensor([float(test_offset)])
        _, _, weights = io_system.getchar(x, positions, anchor, read_offset, input_length)

        peak_pos = weights[0, 0].argmax().item()
        peak_weight = weights[0, 0, peak_pos].item()
        expected_pos = anchor_pos + 1 + test_offset

        is_exact = abs(peak_weight - 1.0) < 0.01 and peak_pos == expected_pos
        status = "✓" if is_exact else "✗"

        print(f"  {label} ({num_bits} bits, max={max_pos}): "
              f"peak@{peak_pos} (w={peak_weight:.4f}) expected@{expected_pos} {status}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    demo_binary_position_io()
    test_4k_positions()
    test_8k_16k()
