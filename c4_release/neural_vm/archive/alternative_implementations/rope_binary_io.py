"""
RoPE Binary Position I/O for Neural VM.

This implements I/O entirely through the attention mechanism using RoPE
with binary thetas (theta_k = 2^k) to encode position directly in binary.

Key insight: With theta_k = 2^k, position is encoded in binary:
- cos(2^k * pos) encodes bit k of position
- This allows direct offset computation without log!

## User Input Flow

1. <USER_INPUT> marker token:
   - Writes to INPUT_MARKER head: (fixed_key, pos_encoding)
   - This marks the start position of user input

2. Each input character token:
   - Writes to INPUT_DATA head: (binary_pos_key, char_value)
   - Position is relative to INPUT_MARKER

3. </USER_INPUT> marker:
   - Writes end position
   - VM can now compute input length

4. VM GETCHAR operation:
   - Queries INPUT_MARKER head to get start_pos
   - Computes offset = current_read_index
   - Queries INPUT_DATA head with binary encoding of offset
   - Gets character value

## User Output Flow

1. VM PUTCHAR operation:
   - Writes to OUTPUT_BUFFER head: (binary_offset_key, char_value)
   - Offset = current_write_index (increments each putchar)

2. <USER_OUTPUT> token:
   - Signals start of output generation
   - Each subsequent token queries OUTPUT_BUFFER with its offset

3. Output tokens:
   - Compute offset = my_pos - output_start_pos
   - Query OUTPUT_BUFFER head with binary encoding of offset
   - Attend and get character to output

## Binary Position Encoding (RoPE-style)

For position p, dimension k:
  q_k = cos(2^k * p)
  k_k = cos(2^k * p)  (same for stored key)

Dot product of encodings:
  sum_k cos(2^k * p1) * cos(2^k * p2)

For p1 == p2: sum of cos^2 = dim/2 (maximum)
For p1 != p2: lower due to destructive interference

This gives direct binary matching without exponentials or log!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from .embedding import E
from .neural_state import VMState
from .base_layers import PureFFN


# ============================================================================
# BINARY ROPE POSITION ENCODING
# ============================================================================

class BinaryRoPE(nn.Module):
    """
    RoPE with binary thetas: theta_k = 2^k

    This encodes position as binary:
    - Position 5 (binary 101) has:
      - Bit 0 set: cos(2^0 * 5) = cos(5)
      - Bit 1 clear: cos(2^1 * 5) = cos(10)
      - Bit 2 set: cos(2^2 * 5) = cos(20)

    Key property: cos(2^k * p) oscillates at frequency 2^k
    - For bit k of position p, if set: contributes positively
    - The pattern uniquely identifies each position

    Offset computation:
    - Don't need log! Just XOR the binary encodings
    - Or use the dot product which peaks when positions match
    """

    def __init__(self, dim: int = 16, max_positions: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

        # Thetas: 2^k for k = 0, 1, 2, ..., dim-1
        # We use dim/2 frequencies (each gives cos and sin)
        self.num_freqs = dim // 2
        thetas = torch.tensor([2.0 ** k for k in range(self.num_freqs)])
        self.register_buffer('thetas', thetas)

        # Precompute encodings for all positions
        positions = torch.arange(max_positions).unsqueeze(1)  # [max_pos, 1]
        angles = positions * thetas.unsqueeze(0)  # [max_pos, num_freqs]

        # Interleave cos and sin: [cos(θ0*p), sin(θ0*p), cos(θ1*p), sin(θ1*p), ...]
        cos_enc = torch.cos(angles)  # [max_pos, num_freqs]
        sin_enc = torch.sin(angles)  # [max_pos, num_freqs]

        # Stack interleaved
        encoding = torch.zeros(max_positions, dim)
        encoding[:, 0::2] = cos_enc
        encoding[:, 1::2] = sin_enc

        self.register_buffer('position_encoding', encoding)

        # Also store pure binary encoding (simpler, more direct)
        binary_enc = torch.zeros(max_positions, dim)
        for k in range(dim):
            # bit k of position: 0 -> -1, 1 -> +1
            bit_k = ((positions.squeeze() >> k) & 1).float()
            binary_enc[:, k] = 2 * bit_k - 1
        self.register_buffer('binary_encoding', binary_enc)

    def encode_rope(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get RoPE encoding for positions.

        Args:
            positions: [batch] or [batch, seq] integer positions

        Returns:
            [batch, dim] or [batch, seq, dim] RoPE encodings
        """
        return self.position_encoding[positions.long()]

    def encode_binary(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get direct binary encoding (-1/+1 per bit).

        Args:
            positions: [batch] or [...] integer positions

        Returns:
            [..., dim] binary encodings
        """
        return self.binary_encoding[positions.long()]

    def compute_offset_binary(self, pos_a: torch.Tensor, pos_b: torch.Tensor) -> torch.Tensor:
        """
        Compute offset = pos_a - pos_b using binary encodings.

        This is simple subtraction - no log needed!
        """
        return pos_a - pos_b

    def decode_binary(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Decode binary encoding back to position.

        Args:
            encoding: [..., dim] binary encodings (-1/+1)

        Returns:
            [...] positions
        """
        bits = (encoding > 0).float()
        powers = torch.tensor([2 ** k for k in range(self.dim)],
                             device=encoding.device, dtype=encoding.dtype)
        return (bits * powers).sum(dim=-1).long()

    def match_score(self, query_pos: torch.Tensor, key_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute match score between positions using binary encoding.

        Perfect match (same position) = dim
        Different positions = lower score

        Args:
            query_pos: [batch] query positions
            key_pos: [num_keys] key positions

        Returns:
            [batch, num_keys] match scores
        """
        q_enc = self.encode_binary(query_pos)  # [batch, dim]
        k_enc = self.encode_binary(key_pos)    # [num_keys, dim]

        # Dot product: same position gives dim, different gives less
        # (2*bit - 1) * (2*bit - 1) = 1 if same bit, -1 if different
        # Sum over bits: dim - 2*(hamming_distance)
        return torch.matmul(q_enc, k_enc.T)  # [batch, num_keys]


# ============================================================================
# KV HEADS FOR I/O
# ============================================================================

class IOKVHead(nn.Module):
    """
    A KV cache head for I/O operations.

    Uses binary position encoding as keys, character values as values.

    This is the core data structure for:
    - INPUT_DATA: stores user input characters
    - OUTPUT_BUFFER: stores VM output for user to read
    """

    def __init__(self, max_entries: int = 4096, value_dim: int = 8, key_dim: int = 16):
        super().__init__()
        self.max_entries = max_entries
        self.value_dim = value_dim
        self.key_dim = key_dim

        self.encoder = BinaryRoPE(dim=key_dim, max_positions=max_entries)

        # KV storage
        # Keys: binary position encodings
        # Values: character values (as nibbles or full values)
        self.register_buffer('keys', torch.zeros(max_entries, key_dim))
        self.register_buffer('values', torch.zeros(max_entries, value_dim))
        self.register_buffer('valid', torch.zeros(max_entries))  # 1 if entry valid
        self.register_buffer('write_ptr', torch.tensor(0))

    def reset(self):
        """Clear the KV head."""
        self.keys.zero_()
        self.values.zero_()
        self.valid.zero_()
        self.write_ptr.zero_()

    def write(self, position: int, value: torch.Tensor):
        """
        Write value at position using binary key.

        Args:
            position: Integer position (offset from start)
            value: [value_dim] tensor
        """
        if position < self.max_entries:
            key = self.encoder.encode_binary(torch.tensor([position]))[0]
            self.keys[position] = key
            self.values[position] = value
            self.valid[position] = 1.0
            if position >= self.write_ptr.item():
                self.write_ptr.fill_(position + 1)

    def write_char(self, position: int, char: int):
        """Write single character at position."""
        value = torch.zeros(self.value_dim)
        value[0] = float(char)  # Store char in first slot
        self.write(position, value)

    def read(self, position: torch.Tensor) -> torch.Tensor:
        """
        Read value at position using binary key matching.

        Uses attention with softmax1 for sparse lookup.

        Args:
            position: [batch] query positions

        Returns:
            [batch, value_dim] values
        """
        query_enc = self.encoder.encode_binary(position)  # [batch, key_dim]

        # Compute attention scores
        # score[i,j] = dot(query[i], key[j])
        # Perfect match = key_dim, mismatch = lower
        scores = torch.matmul(query_enc.float(), self.keys.T)  # [batch, max_entries]

        # Apply validity mask (invalid entries get -inf)
        scores = scores + (1 - self.valid) * (-1e9)

        # Softmax with temperature for sharp selection
        # With binary encoding, perfect match has score = key_dim
        # Scale to make softmax selective
        temperature = 0.1
        attn = F.softmax(scores / temperature, dim=-1)  # [batch, max_entries]

        # Weighted sum of values
        output = torch.matmul(attn, self.values)  # [batch, value_dim]

        return output

    def read_char(self, position: int) -> int:
        """Read single character at position."""
        pos_tensor = torch.tensor([position])
        value = self.read(pos_tensor)
        return int(value[0, 0].item())


# ============================================================================
# INPUT MARKER HEAD
# ============================================================================

class InputMarkerHead(nn.Module):
    """
    Tracks user input start/end positions.

    When <USER_INPUT> token appears:
    - Writes its sequence position to this head
    - Uses fixed key (always matches) with high alibi slope
    - New markers overwrite old ones

    When </USER_INPUT> token appears:
    - Writes end position

    VM can query this to get input region bounds.
    """

    def __init__(self, key_dim: int = 16):
        super().__init__()
        self.key_dim = key_dim

        # Fixed keys for start/end markers
        # These are constant keys that always match their respective queries
        start_key = torch.ones(key_dim)
        end_key = -torch.ones(key_dim)
        self.register_buffer('start_key', start_key)
        self.register_buffer('end_key', end_key)

        # Stored marker positions
        self.register_buffer('input_start_pos', torch.tensor(-1))
        self.register_buffer('input_end_pos', torch.tensor(-1))
        self.register_buffer('output_start_pos', torch.tensor(-1))

        # Binary encoder for position
        self.encoder = BinaryRoPE(dim=key_dim, max_positions=4096)

    def mark_input_start(self, position: int):
        """Called when <USER_INPUT> token processed."""
        self.input_start_pos.fill_(position)

    def mark_input_end(self, position: int):
        """Called when </USER_INPUT> token processed."""
        self.input_end_pos.fill_(position)

    def mark_output_start(self, position: int):
        """Called when <USER_OUTPUT> token processed."""
        self.output_start_pos.fill_(position)

    def get_input_bounds(self) -> Tuple[int, int]:
        """Get start and end positions of user input."""
        return (
            int(self.input_start_pos.item()),
            int(self.input_end_pos.item())
        )

    def get_input_length(self) -> int:
        """Get length of user input."""
        start, end = self.get_input_bounds()
        if start < 0 or end < 0:
            return 0
        return end - start - 1  # Exclude markers

    def in_input_mode(self, current_pos: int) -> bool:
        """Check if we're currently in input mode (between markers)."""
        start = int(self.input_start_pos.item())
        end = int(self.input_end_pos.item())
        return start >= 0 and (end < 0 or current_pos < end)


# ============================================================================
# NEURAL I/O SYSTEM (Attention-Based)
# ============================================================================

class AttentionBasedIO(nn.Module):
    """
    Complete I/O system using attention with binary RoPE.

    This replaces external handlers with pure attention operations:

    ## User Input
    1. <USER_INPUT> → marks start position
    2. Each char token → writes (binary_key, char) to INPUT_DATA head
    3. </USER_INPUT> → marks end position
    4. VM GETCHAR → queries INPUT_DATA with offset, gets char

    ## User Output
    1. VM PUTCHAR → writes (binary_key, char) to OUTPUT_BUFFER head
    2. <USER_OUTPUT> → marks output generation start
    3. Each output token → queries OUTPUT_BUFFER with offset, gets char

    All operations are attention-based, no external handlers needed.
    """

    def __init__(self, max_input: int = 4096, max_output: int = 4096):
        super().__init__()

        # Position/marker tracking
        self.marker_head = InputMarkerHead()

        # Input KV head: stores user input characters
        self.input_data = IOKVHead(max_entries=max_input, value_dim=8)

        # Output KV head: stores VM output for generation
        self.output_buffer = IOKVHead(max_entries=max_output, value_dim=8)

        # Read/write pointers
        self.register_buffer('input_read_ptr', torch.tensor(0))
        self.register_buffer('output_write_ptr', torch.tensor(0))

        # State
        self.register_buffer('in_input_mode', torch.tensor(0.0))
        self.register_buffer('in_output_mode', torch.tensor(0.0))

    def reset(self):
        """Reset all I/O state."""
        self.marker_head.input_start_pos.fill_(-1)
        self.marker_head.input_end_pos.fill_(-1)
        self.marker_head.output_start_pos.fill_(-1)
        self.input_data.reset()
        self.output_buffer.reset()
        self.input_read_ptr.zero_()
        self.output_write_ptr.zero_()
        self.in_input_mode.zero_()
        self.in_output_mode.zero_()

    # =========== Input Processing ===========

    def on_input_start(self, sequence_pos: int):
        """
        Called when <USER_INPUT> marker token is processed.

        This token appears in the user message, marking where input begins.
        """
        self.marker_head.mark_input_start(sequence_pos)
        self.in_input_mode.fill_(1.0)
        self.input_data.reset()

    def on_input_token(self, sequence_pos: int, char_value: int):
        """
        Called for each character token in user input.

        Writes to INPUT_DATA head with:
        - Key: binary encoding of offset from input_start
        - Value: character value
        """
        if self.in_input_mode > 0.5:
            start_pos = int(self.marker_head.input_start_pos.item())
            offset = sequence_pos - start_pos - 1  # -1 to skip marker
            if offset >= 0:
                self.input_data.write_char(offset, char_value)

    def on_input_end(self, sequence_pos: int):
        """Called when </USER_INPUT> marker token is processed."""
        self.marker_head.mark_input_end(sequence_pos)
        self.in_input_mode.fill_(0.0)
        self.input_read_ptr.zero_()  # Reset read pointer for VM

    def getchar(self) -> int:
        """
        VM GETCHAR operation.

        Queries INPUT_DATA head with current read offset.
        Returns character or -1 (EOF).
        """
        offset = int(self.input_read_ptr.item())
        input_length = self.marker_head.get_input_length()

        if offset >= input_length:
            return -1  # EOF

        char = self.input_data.read_char(offset)
        self.input_read_ptr.add_(1)
        return char

    # =========== Output Processing ===========

    def putchar(self, char: int):
        """
        VM PUTCHAR operation.

        Writes to OUTPUT_BUFFER head with:
        - Key: binary encoding of write offset
        - Value: character value
        """
        offset = int(self.output_write_ptr.item())
        self.output_buffer.write_char(offset, char)
        self.output_write_ptr.add_(1)

    def on_output_start(self, sequence_pos: int):
        """
        Called when <USER_OUTPUT> marker token is processed.

        Marks where output generation begins.
        Output tokens will read from OUTPUT_BUFFER.
        """
        self.marker_head.mark_output_start(sequence_pos)
        self.in_output_mode.fill_(1.0)

    def get_output_char(self, output_token_pos: int) -> int:
        """
        Get character for an output token.

        Output tokens attend to OUTPUT_BUFFER using their offset
        from the output start position.

        Args:
            output_token_pos: Sequence position of the output token

        Returns:
            Character to output, or -1 if past buffer
        """
        output_start = int(self.marker_head.output_start_pos.item())
        if output_start < 0:
            return -1

        offset = output_token_pos - output_start - 1  # -1 to skip marker
        if offset < 0 or offset >= int(self.output_write_ptr.item()):
            return -1

        return self.output_buffer.read_char(offset)

    def get_output_length(self) -> int:
        """Get current output buffer length."""
        return int(self.output_write_ptr.item())


# ============================================================================
# NEURAL FFNs FOR I/O OPERATIONS
# ============================================================================

class GetcharFromKVFFN(PureFFN):
    """
    GETCHAR using KV cache attention.

    When GETCHAR opcode is active:
    1. Read input_read_ptr from embedding
    2. Compute binary key for that offset
    3. Query INPUT_DATA head
    4. Write character to RESULT
    5. Increment input_read_ptr

    This is the neural version of getchar() that reads from
    the user input KV head.
    """

    def __init__(self, io_system: AttentionBasedIO):
        super().__init__(VMState.DIM, hidden_dim=16)
        self.io = io_system

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply getchar operation."""
        # Check if GETCHAR opcode is active
        if x[..., E.OP_START + 64].max() > 0.5:  # GETCHAR opcode
            # Read character using the IO system
            char = self.io.getchar()

            # Write to RESULT
            x = x.clone()
            x[..., E.RESULT] = float(char) if char >= 0 else 255.0  # -1 as 0xFF

        return x


class PutcharToKVFFN(PureFFN):
    """
    PUTCHAR using KV cache attention.

    When PUTCHAR opcode is active:
    1. Read character from NIB_A
    2. Compute binary key for output_write_ptr
    3. Write (key, char) to OUTPUT_BUFFER head
    4. Increment output_write_ptr
    """

    def __init__(self, io_system: AttentionBasedIO):
        super().__init__(VMState.DIM, hidden_dim=8)
        self.io = io_system

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply putchar operation."""
        # Check if PUTCHAR opcode is active
        if x[..., E.OP_START + 65].max() > 0.5:  # PUTCHAR opcode
            # Get character from NIB_A
            char = int(x[..., E.NIB_A].max().item())

            # Write using the IO system
            self.io.putchar(char)

        return x


# ============================================================================
# THINK TAG I/O (for character-by-character output)
# ============================================================================

class ThinkTagIO:
    """
    Handles output through think tags.

    As the user described: "Output is simply written to the output this
    involves ending and restarting the think tags for each character."

    Output format:
        </think>H<think>
        </think>e<think>
        </think>l<think>
        ...

    The think tags and their content can be filtered out, making the
    output seamless.

    This class generates the proper token sequence for output.
    """

    THINK_START = "<think>"
    THINK_END = "</think>"

    def __init__(self):
        self.in_think = True  # Start in think mode
        self.output_chars = []

    def putchar(self, char: int) -> str:
        """
        Generate output for a single character.

        Returns the token sequence to output.
        """
        if 0 <= char < 128:
            c = chr(char)
            self.output_chars.append(c)

            # End think, output char, start think
            return f"{self.THINK_END}{c}{self.THINK_START}"
        return ""

    def get_all_output(self) -> str:
        """Get all output characters (without think tags)."""
        return ''.join(self.output_chars)

    def wrap_output(self, text: str) -> str:
        """Wrap text in think tag toggle sequence."""
        result = []
        for c in text:
            result.append(f"{self.THINK_END}{c}{self.THINK_START}")
        return ''.join(result)


# ============================================================================
# INTEGRATED I/O SYSTEM
# ============================================================================

class NeuralIOComplete(nn.Module):
    """
    Complete neural I/O system integrating all components.

    Input: Read directly from user messages via KV attention
    Output: Written via think tag toggling

    All operations happen within the attention mechanism:
    1. Position encoded with binary RoPE (theta_k = 2^k)
    2. User input stored in INPUT_DATA KV head
    3. VM output stored in OUTPUT_BUFFER KV head
    4. Markers track region boundaries

    No external handlers needed - pure transformer operations.
    """

    def __init__(self, max_io: int = 4096):
        super().__init__()

        # Attention-based I/O
        self.attn_io = AttentionBasedIO(max_input=max_io, max_output=max_io)

        # Think tag output
        self.think_io = ThinkTagIO()

        # FFNs for VM operations
        self.getchar_ffn = GetcharFromKVFFN(self.attn_io)
        self.putchar_ffn = PutcharToKVFFN(self.attn_io)

        # Current sequence position
        self.register_buffer('current_pos', torch.tensor(0))

    def reset(self):
        """Reset all I/O state."""
        self.attn_io.reset()
        self.think_io = ThinkTagIO()
        self.current_pos.zero_()

    def process_token(self, token_type: str, char_value: int = 0) -> Optional[str]:
        """
        Process a token during generation.

        Args:
            token_type: 'input_start', 'input_char', 'input_end',
                       'output_start', 'output_char', 'normal'
            char_value: Character value for input/output chars

        Returns:
            Output string if any (for output chars with think tags)
        """
        pos = int(self.current_pos.item())
        self.current_pos.add_(1)

        if token_type == 'input_start':
            self.attn_io.on_input_start(pos)
            return None

        elif token_type == 'input_char':
            self.attn_io.on_input_token(pos, char_value)
            return None

        elif token_type == 'input_end':
            self.attn_io.on_input_end(pos)
            return None

        elif token_type == 'output_start':
            self.attn_io.on_output_start(pos)
            return None

        elif token_type == 'output_char':
            # This is for output tokens reading from buffer
            char = self.attn_io.get_output_char(pos)
            if char >= 0:
                return self.think_io.putchar(char)
            return None

        return None

    def vm_getchar(self) -> int:
        """VM calls this to read input."""
        return self.attn_io.getchar()

    def vm_putchar(self, char: int) -> str:
        """
        VM calls this to write output.

        Returns the think-tag-wrapped output string.
        """
        self.attn_io.putchar(char)
        return self.think_io.putchar(char)


# ============================================================================
# TESTS
# ============================================================================

def test_binary_rope():
    """Test binary RoPE position encoding."""
    print("=== Testing Binary RoPE ===\n")

    rope = BinaryRoPE(dim=16, max_positions=256)

    # Test encoding
    positions = torch.tensor([0, 1, 2, 3, 5, 7, 15, 16, 255])

    print("Binary encodings (-1/+1):")
    for pos in positions.tolist():
        enc = rope.encode_binary(torch.tensor([pos]))[0]
        bits = ''.join(['1' if b > 0 else '0' for b in enc[:8].tolist()])
        print(f"  Position {pos:3d}: {bits}... = {bin(pos)}")

    # Test match scores
    print("\nMatch scores (same pos should have max score):")
    q = torch.tensor([5])
    k = torch.tensor([5, 6, 7, 4, 0])
    scores = rope.match_score(q, k)
    for i, p in enumerate(k.tolist()):
        print(f"  score(5, {p}) = {scores[0, i].item():.1f}")

    # Test decoding
    print("\nDecode test:")
    for pos in [0, 5, 15, 255]:
        enc = rope.encode_binary(torch.tensor([pos]))
        decoded = rope.decode_binary(enc)
        print(f"  {pos} -> encode -> decode = {decoded.item()}")

    print()


def test_io_kv_head():
    """Test I/O KV head read/write."""
    print("=== Testing IO KV Head ===\n")

    head = IOKVHead(max_entries=64, value_dim=8)

    # Write some characters
    test_str = "Hello"
    print(f"Writing: '{test_str}'")
    for i, c in enumerate(test_str):
        head.write_char(i, ord(c))

    # Read back
    print("Reading back:")
    for i in range(len(test_str)):
        char = head.read_char(i)
        print(f"  Position {i}: '{chr(char)}' (code {char})")

    # Test non-sequential read
    print("\nNon-sequential reads:")
    for pos in [4, 0, 2]:
        char = head.read_char(pos)
        print(f"  Position {pos}: '{chr(char)}'")

    print()


def test_attention_io():
    """Test complete attention-based I/O."""
    print("=== Testing Attention-Based I/O ===\n")

    io = AttentionBasedIO()

    # Simulate user input sequence
    print("Simulating user input: 'Hi!'")
    io.on_input_start(sequence_pos=10)  # <USER_INPUT> at pos 10
    io.on_input_token(sequence_pos=11, char_value=ord('H'))
    io.on_input_token(sequence_pos=12, char_value=ord('i'))
    io.on_input_token(sequence_pos=13, char_value=ord('!'))
    io.on_input_end(sequence_pos=14)    # </USER_INPUT> at pos 14

    print(f"Input bounds: {io.marker_head.get_input_bounds()}")
    print(f"Input length: {io.marker_head.get_input_length()}")

    # VM reads input
    print("\nVM reading input (GETCHAR):")
    while True:
        char = io.getchar()
        if char < 0:
            print("  EOF")
            break
        print(f"  getchar() = '{chr(char)}' ({char})")

    # VM writes output
    print("\nVM writing output (PUTCHAR):")
    for c in "Bye":
        io.putchar(ord(c))
        print(f"  putchar('{c}')")

    print(f"Output buffer length: {io.get_output_length()}")

    # Simulate output tokens reading
    print("\nOutput tokens reading buffer:")
    io.on_output_start(sequence_pos=100)  # <USER_OUTPUT> at pos 100
    for i in range(3):
        char = io.get_output_char(output_token_pos=101 + i)
        if char >= 0:
            print(f"  Token at pos {101+i}: '{chr(char)}'")

    print()


def test_think_tag_io():
    """Test think tag output formatting."""
    print("=== Testing Think Tag I/O ===\n")

    think_io = ThinkTagIO()

    print("Output sequence for 'Hi':")
    for c in "Hi":
        output = think_io.putchar(ord(c))
        print(f"  putchar('{c}') -> '{output}'")

    print(f"\nAll output (without tags): '{think_io.get_all_output()}'")

    print()


def test_complete_io():
    """Test complete neural I/O system."""
    print("=== Testing Complete Neural I/O ===\n")

    io = NeuralIOComplete()

    # Process input tokens
    print("Processing input tokens:")
    io.process_token('input_start')
    for c in "AB":
        io.process_token('input_char', ord(c))
    io.process_token('input_end')

    # VM reads and echoes
    print("\nVM echo loop:")
    outputs = []
    while True:
        char = io.vm_getchar()
        if char < 0:
            break
        output = io.vm_putchar(char)
        outputs.append(output)
        print(f"  Read '{chr(char)}', output: {repr(output)}")

    # Process output tokens
    print("\nOutput token generation:")
    io.process_token('output_start')
    for i in range(2):
        result = io.process_token('output_char')
        if result:
            print(f"  Output token {i}: {repr(result)}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("RoPE Binary Position I/O")
    print("=" * 60)
    print()
    print("Key innovation: RoPE with theta_k = 2^k encodes position in binary")
    print("This allows direct offset computation without log!")
    print()
    print("I/O Flow:")
    print("  Input:  User tokens -> INPUT_DATA KV head -> VM GETCHAR")
    print("  Output: VM PUTCHAR -> OUTPUT_BUFFER KV head -> Output tokens")
    print()

    test_binary_rope()
    test_io_kv_head()
    test_attention_io()
    test_think_tag_io()
    test_complete_io()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
