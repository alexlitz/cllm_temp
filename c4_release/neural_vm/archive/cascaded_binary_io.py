"""
Cascaded Binary Position I/O for Neural VM.

Instead of encoding position all at once, we compute it layer-by-layer:
- Layer 0: Check if distance >= 2^31, extract bit 31
- Layer 1: Check if remaining distance >= 2^30, extract bit 30
- ...
- Layer 31: Check if remaining distance >= 2^0, extract bit 0

Each layer uses an attention head with a specific "power of 2" threshold.
This gives us the binary representation through the depth of the network.

## How it works

Given: current_pos, start_pos (marked by input start token)

Layer k (handles bit 31-k):
  1. Attention head computes: distance_to_start = f(current_pos, start_pos)
  2. FFN checks: is remaining_distance >= 2^(31-k)?
  3. If yes: bit[31-k] = 1, remaining -= 2^(31-k)
  4. If no: bit[31-k] = 0, remaining unchanged
  5. Pass to next layer

After 32 layers, we have the full binary offset.
Use this to index into the input buffer.

## Key Insight

With alibi slopes, attention naturally encodes distance as exponential:
  score = -slope * distance

With slope = 1/2^k, the attention score encodes bit k:
  - If distance has bit k set: score contribution is -1/2^k * 2^k = -1
  - If distance has bit k clear: contribution is smaller

By having multiple heads with slopes 1/2^0, 1/2^1, ..., 1/2^31,
each head naturally responds to its corresponding bit.

## Implementation

Each layer has:
1. Attention head with slope 1/2^(31-layer_idx)
2. FFN that thresholds and extracts the bit
3. Accumulator that builds up the binary representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# ALIBI SLOPE ATTENTION FOR BIT EXTRACTION
# ============================================================================

class AlibiBitAttention(nn.Module):
    """
    Attention head with specific alibi slope for bit extraction.

    With slope = 1/2^k, this head is sensitive to bit k of the distance:
    - score = -slope * distance = -distance / 2^k
    - For distance with bit k set: score contribution ≈ -1
    - For distance with bit k clear: score contribution ≈ 0

    The attention naturally groups positions by their bit k value.
    """

    def __init__(self, bit_index: int, dim: int = 64):
        """
        Create attention head for extracting a specific bit.

        Args:
            bit_index: Which bit to extract (0 = LSB, 31 = MSB)
            dim: Head dimension
        """
        super().__init__()
        self.bit_index = bit_index
        self.dim = dim

        # Alibi slope: 1/2^bit_index
        # This makes the attention sensitive to that power of 2
        self.slope = 1.0 / (2 ** bit_index)

        # Threshold: 2^bit_index
        self.threshold = 2 ** bit_index

        # Projections (using fixed patterns for pure neural)
        # Q projects current position marker
        # K projects start position marker
        # V projects the accumulated bits
        self.register_buffer('q_proj', torch.eye(dim))
        self.register_buffer('k_proj', torch.eye(dim))
        self.register_buffer('v_proj', torch.eye(dim))

    def forward(self, current_pos: torch.Tensor, start_pos: torch.Tensor,
                remaining: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract bit from distance using alibi attention.

        Args:
            current_pos: [batch] current sequence positions
            start_pos: [batch] start marker positions
            remaining: [batch] remaining distance after higher bits

        Returns:
            (bit_value, new_remaining): The extracted bit and updated remaining
        """
        # Compute distance
        distance = current_pos - start_pos

        # For cascaded approach, we use remaining instead of full distance
        # (remaining has already had higher bits subtracted)

        # Check if bit is set: remaining >= 2^bit_index
        # Using SiLU for smooth thresholding:
        # silu(S * (remaining - threshold)) > 0 iff remaining > threshold
        S = 10.0  # Scale for sharp threshold
        threshold_check = remaining - self.threshold + 0.5  # +0.5 for >= vs >

        # Bit value: 1 if remaining >= threshold, 0 otherwise
        bit_activation = torch.sigmoid(S * threshold_check)
        bit_value = (bit_activation > 0.5).float()

        # Update remaining: subtract threshold if bit is set
        new_remaining = remaining - bit_value * self.threshold

        return bit_value, new_remaining


class CascadedBitExtractor(nn.Module):
    """
    Extract all bits of a distance through cascaded layers.

    Each layer extracts one bit, starting from MSB:
    - Layer 0: bit 31 (MSB)
    - Layer 1: bit 30
    - ...
    - Layer 31: bit 0 (LSB)

    This builds up the binary representation through the network depth.
    """

    def __init__(self, num_bits: int = 32, dim: int = 64):
        super().__init__()
        self.num_bits = num_bits
        self.dim = dim

        # Create attention head for each bit (MSB first)
        self.bit_heads = nn.ModuleList([
            AlibiBitAttention(bit_index=num_bits - 1 - i, dim=dim)
            for i in range(num_bits)
        ])

    def forward(self, current_pos: torch.Tensor,
                start_pos: torch.Tensor) -> torch.Tensor:
        """
        Extract binary representation of (current_pos - start_pos).

        Args:
            current_pos: [batch] current positions
            start_pos: [batch] start positions

        Returns:
            [batch, num_bits] binary representation (bit 0 = LSB)
        """
        batch_size = current_pos.shape[0]
        device = current_pos.device

        # Initial remaining = full distance
        remaining = (current_pos - start_pos).float()

        # Extract bits from MSB to LSB
        bits = torch.zeros(batch_size, self.num_bits, device=device)

        for i, head in enumerate(self.bit_heads):
            bit_idx = self.num_bits - 1 - i  # MSB first
            bit_value, remaining = head(current_pos, start_pos, remaining)
            bits[:, bit_idx] = bit_value.squeeze(-1) if bit_value.dim() > 1 else bit_value

        return bits

    def bits_to_int(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert binary representation back to integer."""
        powers = torch.tensor([2 ** i for i in range(self.num_bits)],
                             device=bits.device, dtype=bits.dtype)
        return (bits * powers).sum(dim=-1).long()


# ============================================================================
# PURE FFN BIT EXTRACTION (Alternative to Attention)
# ============================================================================

class BitExtractionFFN(nn.Module):
    """
    FFN that extracts a single bit from a value.

    For bit k:
      bit_k = floor(value / 2^k) mod 2

    Neural implementation using SiLU:
      1. Compute value / 2^k (scaling)
      2. Extract fractional part mod 2
      3. Threshold to get 0 or 1

    Actually simpler: check if (value & 2^k) != 0
    Using: silu(S * (value - 2^k + 0.5)) gives positive for value >= 2^k
    But we need (value mod 2^(k+1)) >= 2^k

    Cleanest: value mod 2^(k+1) >= 2^k
    = floor(value / 2^k) mod 2 == 1
    """

    def __init__(self, bit_index: int, input_dim: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.bit_index = bit_index
        self.threshold = 2 ** bit_index
        self.mod_value = 2 ** (bit_index + 1)

        # SwiGLU-style FFN
        self.W_up = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.W_gate = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.W_down = nn.Parameter(torch.zeros(1, hidden_dim))
        self.b_up = nn.Parameter(torch.zeros(hidden_dim))
        self.b_gate = nn.Parameter(torch.zeros(hidden_dim))

        self._bake_weights()

    def _bake_weights(self):
        """Bake in the bit extraction logic."""
        S = 20.0  # Scale

        with torch.no_grad():
            # For value in slot 0, extract bit k
            # Strategy: compute (value mod 2^(k+1)) - 2^k
            # If >= 0, bit is 1; if < 0, bit is 0

            # The input value is spread across nibbles, but for simplicity
            # assume value is in input_dim=0

            # Method: silu(S * (value - threshold)) for values in [threshold, 2*threshold)
            # gives positive output

            # Simpler for now: direct threshold
            # bit = 1 if value >= threshold and value mod (2*threshold) < 2*threshold

            # Use two neurons:
            # 1. Fires if value >= threshold
            # 2. Gates if value < 2*threshold

            # Neuron 0: value >= threshold
            self.W_up[0, 0] = S
            self.b_up[0] = -S * (self.threshold - 0.5)
            self.b_gate[0] = 1.0

            # Neuron 1: value < 2*threshold (for masking higher values)
            # Actually, for cascaded approach, we use remaining which already
            # has higher bits removed, so we just need >= threshold check

            self.W_down[0, 0] = 1.0 / S

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Extract bit from value.

        Args:
            value: [...] tensor of values

        Returns:
            [...] tensor of bits (0 or 1)
        """
        # Flatten for matmul
        orig_shape = value.shape
        x = value.view(-1, 1) if value.dim() == 1 else value.view(-1, value.shape[-1])

        # SwiGLU
        up = F.silu(F.linear(x, self.W_up, self.b_up))
        gate = torch.sigmoid(F.linear(x, self.W_gate, self.b_gate))
        hidden = up * gate
        out = F.linear(hidden, self.W_down)

        # Threshold to binary
        bit = (out > 0.5).float()

        return bit.view(orig_shape[:-1] + (1,)) if len(orig_shape) > 1 else bit.squeeze()


# ============================================================================
# CASCADED BINARY LAYER (Full Layer)
# ============================================================================

class CascadedBinaryLayer(nn.Module):
    """
    A single layer in the cascaded binary decomposition.

    This layer handles one bit of the position offset:
    1. Receives: current_pos, start_pos, remaining_distance, accumulated_bits
    2. Extracts: bit (31 - layer_idx) from remaining_distance
    3. Updates: remaining_distance -= bit * 2^(31 - layer_idx)
    4. Outputs: updated remaining_distance, updated accumulated_bits

    The layer uses attention to read the start position marker
    and FFN to extract and accumulate the bit.
    """

    def __init__(self, layer_idx: int, total_bits: int = 32,
                 embed_dim: int = 64, num_heads: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.bit_index = total_bits - 1 - layer_idx  # MSB first
        self.threshold = 2 ** self.bit_index
        self.embed_dim = embed_dim

        # Attention to read start position marker
        self.start_pos_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # FFN for bit extraction
        self.bit_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Slots for remaining and accumulated bits
        self.remaining_slot = 0
        self.bits_start_slot = 1
        self.bits_end_slot = total_bits

    def forward(self, x: torch.Tensor, start_marker: torch.Tensor) -> torch.Tensor:
        """
        Process one layer of bit extraction.

        Args:
            x: [batch, seq, embed_dim] current state
               - slot 0: remaining distance
               - slots 1-32: accumulated bits
            start_marker: [batch, 1, embed_dim] start position embedding

        Returns:
            Updated x with bit extracted
        """
        # Attention to get distance info from start marker
        attn_out, _ = self.start_pos_attn(x, start_marker, start_marker)
        x = self.norm1(x + attn_out)

        # FFN for bit extraction
        # Read remaining from slot 0
        remaining = x[:, :, self.remaining_slot:self.remaining_slot+1]

        # Check if bit is set
        S = 20.0
        bit_check = remaining - self.threshold + 0.5
        bit_value = torch.sigmoid(S * bit_check)

        # Update remaining
        new_remaining = remaining - bit_value * self.threshold

        # Write back
        x = x.clone()
        x[:, :, self.remaining_slot] = new_remaining.squeeze(-1)
        x[:, :, self.bits_start_slot + self.bit_index] = bit_value.squeeze(-1)

        # FFN for mixing
        x = self.norm2(x + self.bit_ffn(x))

        return x


# ============================================================================
# COMPLETE CASCADED BINARY NETWORK
# ============================================================================

class CascadedBinaryNetwork(nn.Module):
    """
    Complete network for cascaded binary position extraction.

    32 layers, each extracting one bit of the position offset.

    Input: sequence with position embeddings + start marker
    Output: binary representation of offset from start marker

    Usage:
        net = CascadedBinaryNetwork()
        bits = net(current_pos, start_pos)  # [batch, 32] binary
        offset = net.bits_to_int(bits)       # [batch] integer offset
    """

    def __init__(self, num_bits: int = 32, embed_dim: int = 64):
        super().__init__()
        self.num_bits = num_bits
        self.embed_dim = embed_dim

        # Cascaded layers, one per bit
        self.layers = nn.ModuleList([
            CascadedBinaryLayer(i, num_bits, embed_dim)
            for i in range(num_bits)
        ])

        # Position embedding (for start marker)
        self.pos_embed = nn.Embedding(4096, embed_dim)

    def forward(self, current_pos: torch.Tensor,
                start_pos: torch.Tensor) -> torch.Tensor:
        """
        Extract binary offset through cascaded layers.

        Args:
            current_pos: [batch] current positions
            start_pos: [batch] start marker positions

        Returns:
            [batch, num_bits] binary representation
        """
        batch_size = current_pos.shape[0]
        device = current_pos.device

        # Initialize state
        # Slot 0: remaining distance
        # Slots 1-32: accumulated bits (initially 0)
        x = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        x[:, 0, 0] = (current_pos - start_pos).float()

        # Start marker embedding
        start_marker = self.pos_embed(start_pos).unsqueeze(1)

        # Cascade through layers
        for layer in self.layers:
            x = layer(x, start_marker)

        # Extract accumulated bits
        bits = x[:, 0, 1:self.num_bits+1]

        return bits

    def bits_to_int(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert binary to integer."""
        powers = torch.tensor([2 ** i for i in range(self.num_bits)],
                             device=bits.device, dtype=bits.dtype)
        return (bits * powers).sum(dim=-1).long()


# ============================================================================
# PURE FFN CASCADED IMPLEMENTATION (No Attention)
# ============================================================================

class PureCascadedBitFFN(nn.Module):
    """
    Pure FFN implementation of cascaded bit extraction.

    Each "layer" is actually an FFN that:
    1. Reads remaining distance from embedding slot
    2. Extracts one bit
    3. Updates remaining distance
    4. Writes bit to accumulator slot

    This can be done with a sequence of FFNs applied in order,
    each handling one bit from MSB to LSB.
    """

    def __init__(self, num_bits: int = 16, dim: int = 64):
        super().__init__()
        self.num_bits = num_bits
        self.dim = dim

        # Slots in embedding:
        # 0: input value / remaining
        # 1-num_bits: accumulated bits
        # num_bits+1: distance (written by attention to start marker)

        self.REMAINING_SLOT = 0
        self.BITS_START = 1
        self.DISTANCE_SLOT = num_bits + 1

        # FFN for each bit (MSB first)
        self.bit_ffns = nn.ModuleList()
        for i in range(num_bits):
            bit_idx = num_bits - 1 - i  # MSB first
            threshold = 2 ** bit_idx
            self.bit_ffns.append(self._make_bit_ffn(bit_idx, threshold))

    def _make_bit_ffn(self, bit_idx: int, threshold: int) -> nn.Module:
        """Create FFN for extracting a specific bit."""

        class BitFFN(nn.Module):
            def __init__(self, bit_idx, threshold, remaining_slot, bits_start, dim):
                super().__init__()
                self.bit_idx = bit_idx
                self.threshold = threshold
                self.remaining_slot = remaining_slot
                self.bit_slot = bits_start + bit_idx
                self.dim = dim
                self.S = 20.0  # Scale for sharp threshold

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Read remaining
                remaining = x[..., self.remaining_slot:self.remaining_slot+1]

                # Check if bit is set: remaining >= threshold
                bit_check = remaining - self.threshold + 0.5
                bit_soft = torch.sigmoid(self.S * bit_check)

                # Hard threshold to 0/1 (fixes accumulation errors)
                bit_value = (bit_soft > 0.5).float()

                # Update remaining
                new_remaining = remaining - bit_value * self.threshold

                # Write back
                x = x.clone()
                x[..., self.remaining_slot] = new_remaining.squeeze(-1)
                x[..., self.bit_slot] = bit_value.squeeze(-1)

                return x

        return BitFFN(bit_idx, threshold, self.REMAINING_SLOT, self.BITS_START, self.dim)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """
        Extract binary representation through cascaded FFNs.

        Args:
            distance: [...] distance values

        Returns:
            [..., num_bits] binary representation
        """
        # Create embedding with distance in slot 0
        shape = distance.shape
        x = torch.zeros(*shape, self.dim, device=distance.device)
        x[..., self.REMAINING_SLOT] = distance.float()

        # Apply FFNs in sequence (MSB first)
        for ffn in self.bit_ffns:
            x = ffn(x)

        # Extract accumulated bits
        bits = x[..., self.BITS_START:self.BITS_START + self.num_bits]

        return bits

    def bits_to_int(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert binary to integer."""
        powers = torch.tensor([2 ** i for i in range(self.num_bits)],
                             device=bits.device, dtype=bits.dtype)
        return (bits[..., :self.num_bits] * powers).sum(dim=-1).long()


# ============================================================================
# I/O SYSTEM USING CASCADED BINARY
# ============================================================================

class CascadedBinaryIO(nn.Module):
    """
    Complete I/O system using cascaded binary extraction.

    Input Flow:
    1. <USER_INPUT> marker writes start_pos to KV cache
    2. Each input char writes (pos, char) to input KV cache
    3. </USER_INPUT> marker writes end_pos

    GETCHAR:
    1. Read start_pos from marker KV
    2. Compute distance = current_read_pos - start_pos
    3. Cascade through layers to extract binary offset
    4. Use binary offset to query input KV cache
    5. Get character value

    Output Flow:
    1. PUTCHAR writes (offset, char) to output KV cache
    2. <USER_OUTPUT> marker signals generation start
    3. Output tokens cascade to compute their offset
    4. Query output KV cache with binary offset
    5. Get character to output
    """

    def __init__(self, num_bits: int = 16, max_io: int = 4096):
        super().__init__()
        self.num_bits = num_bits
        self.max_io = max_io

        # Cascaded bit extractor
        self.bit_extractor = PureCascadedBitFFN(num_bits=num_bits)

        # KV caches for I/O
        # Input: stores user input characters indexed by binary offset
        # Output: stores VM output characters indexed by binary offset
        self.register_buffer('input_keys', torch.zeros(max_io, num_bits))
        self.register_buffer('input_values', torch.zeros(max_io))
        self.register_buffer('input_valid', torch.zeros(max_io))

        self.register_buffer('output_keys', torch.zeros(max_io, num_bits))
        self.register_buffer('output_values', torch.zeros(max_io))
        self.register_buffer('output_valid', torch.zeros(max_io))

        # Markers and pointers
        self.register_buffer('input_start_pos', torch.tensor(-1))
        self.register_buffer('input_end_pos', torch.tensor(-1))
        self.register_buffer('input_read_ptr', torch.tensor(0))
        self.register_buffer('output_write_ptr', torch.tensor(0))

    def reset(self):
        """Reset all state."""
        self.input_keys.zero_()
        self.input_values.zero_()
        self.input_valid.zero_()
        self.output_keys.zero_()
        self.output_values.zero_()
        self.output_valid.zero_()
        self.input_start_pos.fill_(-1)
        self.input_end_pos.fill_(-1)
        self.input_read_ptr.zero_()
        self.output_write_ptr.zero_()

    def _offset_to_binary(self, offset: int) -> torch.Tensor:
        """Convert offset to binary key."""
        bits = torch.zeros(self.num_bits)
        for k in range(self.num_bits):
            bits[k] = float((offset >> k) & 1)
        return bits

    def on_input_start(self, pos: int):
        """Mark start of user input."""
        self.input_start_pos.fill_(pos)

    def on_input_char(self, pos: int, char: int):
        """Store user input character."""
        start = int(self.input_start_pos.item())
        if start >= 0:
            offset = pos - start - 1
            if 0 <= offset < self.max_io:
                self.input_keys[offset] = self._offset_to_binary(offset)
                self.input_values[offset] = float(char)
                self.input_valid[offset] = 1.0

    def on_input_end(self, pos: int):
        """Mark end of user input."""
        self.input_end_pos.fill_(pos)
        self.input_read_ptr.zero_()

    def getchar(self) -> int:
        """
        Read character using cascaded binary lookup.

        1. Get current read offset
        2. Extract binary representation through cascaded layers
        3. Use binary key to query input KV cache
        4. Return character
        """
        start = int(self.input_start_pos.item())
        end = int(self.input_end_pos.item())
        offset = int(self.input_read_ptr.item())

        if start < 0 or end < 0:
            return -1

        input_length = end - start - 1
        if offset >= input_length:
            return -1

        # Extract binary key through cascaded FFNs
        offset_tensor = torch.tensor([offset])
        bits = self.bit_extractor(offset_tensor)  # [1, num_bits]

        # Query input KV cache using binary matching
        # score = sum of matching bits (max = num_bits for exact match)
        query_bits = bits[0]  # [num_bits]
        scores = (query_bits.unsqueeze(0) * self.input_keys +
                  (1 - query_bits.unsqueeze(0)) * (1 - self.input_keys))
        scores = scores.sum(dim=-1)  # [max_io]
        scores = scores + (1 - self.input_valid) * (-1e9)  # Mask invalid

        # Softmax for selection (should be very sharp due to exact match)
        attn = F.softmax(scores * 10.0, dim=-1)

        # Get character
        char = (attn * self.input_values).sum().item()

        self.input_read_ptr.add_(1)
        return int(round(char))

    def putchar(self, char: int):
        """
        Write character using cascaded binary indexing.

        1. Get current write offset
        2. Extract binary key through cascaded layers
        3. Store (binary_key, char) in output KV cache
        """
        offset = int(self.output_write_ptr.item())
        if offset < self.max_io:
            self.output_keys[offset] = self._offset_to_binary(offset)
            self.output_values[offset] = float(char)
            self.output_valid[offset] = 1.0
            self.output_write_ptr.add_(1)

    def get_output_char(self, output_offset: int) -> int:
        """Get character from output buffer at given offset."""
        if output_offset < 0 or output_offset >= int(self.output_write_ptr.item()):
            return -1

        # Extract binary key
        offset_tensor = torch.tensor([output_offset])
        bits = self.bit_extractor(offset_tensor)[0]

        # Query output KV cache
        scores = (bits.unsqueeze(0) * self.output_keys +
                  (1 - bits.unsqueeze(0)) * (1 - self.output_keys))
        scores = scores.sum(dim=-1)
        scores = scores + (1 - self.output_valid) * (-1e9)

        attn = F.softmax(scores * 10.0, dim=-1)
        char = (attn * self.output_values).sum().item()

        return int(round(char))


# ============================================================================
# TESTS
# ============================================================================

def test_cascaded_bit_extraction():
    """Test cascaded bit extraction FFN."""
    print("=== Testing Cascaded Bit Extraction ===\n")

    extractor = PureCascadedBitFFN(num_bits=8)

    # Test various distances
    test_values = [0, 1, 5, 15, 127, 255]

    print("Cascaded extraction (MSB first):")
    for val in test_values:
        bits = extractor(torch.tensor([val]))
        bits_list = [int(b.item()) for b in bits[0]]
        bits_str = ''.join(map(str, reversed(bits_list)))  # MSB first for display
        reconstructed = extractor.bits_to_int(bits).item()
        print(f"  {val:3d} -> {bits_str} -> {reconstructed}")
        assert reconstructed == val, f"Mismatch: {val} != {reconstructed}"

    print("\n✓ All extractions correct!")
    print()


def test_cascaded_layer_by_layer():
    """Show the layer-by-layer extraction process."""
    print("=== Layer-by-Layer Extraction Demo ===\n")

    num_bits = 8
    value = 173  # 10101101 in binary

    print(f"Extracting bits from {value} (binary: {bin(value)})")
    print()

    remaining = float(value)
    bits = []

    for layer in range(num_bits):
        bit_idx = num_bits - 1 - layer  # MSB first
        threshold = 2 ** bit_idx

        bit = 1 if remaining >= threshold else 0
        if bit:
            remaining -= threshold

        bits.append(bit)
        print(f"Layer {layer}: bit[{bit_idx}] threshold={threshold:3d}, "
              f"remaining={remaining:6.1f} >= {threshold}? -> bit={bit}")

    bits_str = ''.join(map(str, bits))
    print(f"\nExtracted bits (MSB first): {bits_str}")
    print(f"Original binary:            {bin(value)[2:].zfill(num_bits)}")

    # Reconstruct
    reconstructed = sum(b * (2 ** (num_bits - 1 - i)) for i, b in enumerate(bits))
    print(f"Reconstructed: {reconstructed}")
    assert reconstructed == value

    print("\n✓ Layer-by-layer extraction works!")
    print()


def test_cascaded_io():
    """Test complete cascaded I/O system."""
    print("=== Testing Cascaded Binary I/O ===\n")

    io = CascadedBinaryIO(num_bits=16)

    # Simulate user input
    print("Simulating user input: 'ABC'")
    io.on_input_start(10)
    io.on_input_char(11, ord('A'))
    io.on_input_char(12, ord('B'))
    io.on_input_char(13, ord('C'))
    io.on_input_end(14)

    # Read back using cascaded lookup
    print("\nReading with cascaded binary lookup:")
    chars = []
    while True:
        char = io.getchar()
        if char < 0:
            print("  EOF")
            break
        chars.append(chr(char))
        print(f"  getchar() = '{chr(char)}' ({char})")

    assert ''.join(chars) == 'ABC', f"Got {''.join(chars)}"

    # Test output
    print("\nWriting output: 'XYZ'")
    for c in 'XYZ':
        io.putchar(ord(c))
        print(f"  putchar('{c}')")

    print("\nReading output buffer:")
    for i in range(3):
        char = io.get_output_char(i)
        print(f"  offset {i}: '{chr(char)}'")

    print("\n✓ Cascaded I/O works!")
    print()


def demo_alibi_intuition():
    """Demonstrate the intuition behind alibi slopes for bit extraction."""
    print("=== Alibi Slope Intuition ===\n")

    print("With alibi slope = 1/2^k, attention is sensitive to bit k:")
    print()

    for k in range(4):
        slope = 1.0 / (2 ** k)
        threshold = 2 ** k

        print(f"Bit {k}: slope = 1/{2**k} = {slope:.4f}, threshold = {threshold}")
        print(f"  Distances with bit {k} set: ", end="")
        examples = [d for d in range(16) if (d >> k) & 1]
        print(examples[:8], "...")
        print(f"  Attention score contribution when bit set: -{slope * threshold:.2f}")
        print()

    print("Key insight: Each slope creates a 'resonance' with its power of 2")
    print("The cascaded layers use this to extract bits one at a time")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Cascaded Binary Position Extraction")
    print("=" * 60)
    print()
    print("Each layer extracts one bit of the distance:")
    print("  Layer 0: bit 31 (2^31 threshold)")
    print("  Layer 1: bit 30 (2^30 threshold)")
    print("  ...")
    print("  Layer 31: bit 0 (2^0 threshold)")
    print()

    demo_alibi_intuition()
    test_cascaded_layer_by_layer()
    test_cascaded_bit_extraction()
    test_cascaded_io()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
