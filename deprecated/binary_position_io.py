"""
Binary Position I/O for LLM-Native Neural VM.

This module implements I/O that works within an LLM context:
- Input comes from user messages (tokens after input marker)
- Output goes to model's non-thinking output (displayed to user)
- Memory positions are encoded as binary distance from BOS token
- Attention uses binary matching for precise position addressing

Architecture:
1. Token positions encoded as binary bits in embedding
2. Memory addressing via binary attention matching
3. Input stream = user tokens after <INPUT> marker
4. Output stream = generated tokens (model output)

Binary Position Encoding:
- Position p encoded as bits: bit_k = (p >> k) & 1
- 16 bits supports positions 0 to 65535
- Query encodes target address, keys encode actual positions
- Perfect match gives attention weight 1.0, any mismatch → 0.0
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention


# =============================================================================
# BINARY POSITION ENCODING
# =============================================================================

class BinaryPositionEncoder(nn.Module):
    """
    Encode positions as binary bits in the embedding.

    Each position p is encoded as:
        bit_k = floor(p / 2^k) mod 2

    This gives a unique binary signature for each position.
    """

    # Embedding slots for binary position bits (16 bits = 64K positions)
    NUM_BITS = 16
    BIT_START = 96  # Start after existing E.DIM slots

    def __init__(self):
        super().__init__()
        # Precompute powers of 2
        self.register_buffer('powers', torch.tensor([2**k for k in range(self.NUM_BITS)], dtype=torch.float32))

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add binary position encoding to embedding.

        Args:
            x: [batch, seq_len, dim] embedding tensor
            positions: [seq_len] position indices (default: 0, 1, 2, ...)

        Returns:
            x with binary bits written to BIT_START:BIT_START+NUM_BITS
        """
        batch, seq_len, dim = x.shape

        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # Ensure we have enough dimensions
        if dim < self.BIT_START + self.NUM_BITS:
            # Expand embedding dimension
            new_x = torch.zeros(batch, seq_len, self.BIT_START + self.NUM_BITS, device=x.device)
            new_x[:, :, :dim] = x
            x = new_x

        # Encode each position as binary
        for k in range(self.NUM_BITS):
            # bit_k = floor(pos / 2^k) mod 2
            bit_values = ((positions.unsqueeze(0) / self.powers[k]).floor() % 2)
            x[:, :, self.BIT_START + k] = bit_values

        return x


# =============================================================================
# BINARY ADDRESS MATCHING ATTENTION
# =============================================================================

class BinaryMatchAttention(nn.Module):
    """
    Attention that matches positions using binary encoding.

    Query encodes a target address as binary bits.
    Keys encode actual position indices as binary bits.
    Attention weight = product of bit matches = 1 only when all bits match.

    This gives precise O(1) lookup to any position.
    """

    def __init__(self, num_bits: int = 16):
        super().__init__()
        self.num_bits = num_bits
        self.bit_start = BinaryPositionEncoder.BIT_START

    def forward(self, x: torch.Tensor, query_addr: int) -> torch.Tensor:
        """
        Attend to a specific position using binary address matching.

        Args:
            x: [batch, seq_len, dim] with binary position bits encoded
            query_addr: Target address to read from

        Returns:
            Value at the target position (via attention)
        """
        batch, seq_len, dim = x.shape

        # Encode query address as binary
        query_bits = torch.zeros(self.num_bits, device=x.device)
        for k in range(self.num_bits):
            query_bits[k] = float((query_addr >> k) & 1)

        # Get key bits from each position
        key_bits = x[:, :, self.bit_start:self.bit_start + self.num_bits]  # [batch, seq, num_bits]

        # Compute match: product of (1 - |query_k - key_k|)
        # When bits match: 1 - 0 = 1
        # When bits differ: 1 - 1 = 0
        match = 1.0 - torch.abs(key_bits - query_bits.unsqueeze(0).unsqueeze(0))  # [batch, seq, num_bits]

        # Product across bits: only 1.0 when ALL bits match
        attention_weights = match.prod(dim=-1)  # [batch, seq]

        # Apply attention (should be one-hot to target position)
        # Return weighted sum (effectively reads from matched position)
        values = x[:, :, :E.DIM]  # Get value content (not position bits)
        output = torch.einsum('bs,bsd->bd', attention_weights, values)

        return output


class BinaryMatchAttentionPure(PureAttention):
    """
    Pure attention layer for binary position matching.

    Bakes binary matching into attention weights.
    Query is constructed from embedding slots (address to read).
    Keys are position encodings.
    """

    def __init__(self, addr_slot: int = E.TEMP, num_bits: int = 16):
        """
        Args:
            addr_slot: Embedding slot containing address to read
            num_bits: Number of bits for position encoding
        """
        super().__init__(E.DIM + num_bits)
        self.addr_slot = addr_slot
        self.num_bits = num_bits
        self.bit_start = BinaryPositionEncoder.BIT_START

    def _bake_weights(self):
        """Bake binary matching into attention weights."""
        S = E.SCALE
        with torch.no_grad():
            # Query: extract binary bits from address slot
            # The address in addr_slot needs to be converted to binary

            # For each bit k: Q_k reads bit k of the address
            for k in range(self.num_bits):
                # Q projects address slot through power-of-2 extraction
                # bit_k = floor(addr / 2^k) mod 2
                # This requires computation, so we use a different approach:
                # Store address bits in dedicated slots

                # Query copies from bit slots (if address already binary encoded)
                self.W_Q[k, self.bit_start + k] = 1.0

                # Key copies position bits
                self.W_K[k, self.bit_start + k] = 1.0

            # Value copies all slots (to retrieve content at matched position)
            for i in range(E.DIM):
                self.W_V[i, i] = 1.0


# =============================================================================
# LLM-NATIVE I/O STREAMS
# =============================================================================

class LLMIOState:
    """
    State for LLM-native I/O.

    Tracks:
    - BOS position (start of sequence)
    - Input marker position (<INPUT>)
    - Current input read position
    - Output buffer position
    """

    def __init__(self):
        self.bos_position = 0
        self.input_marker_position = -1  # Position of <INPUT> token
        self.input_read_position = 0     # Next char to read from input
        self.output_position = 0         # Next output position
        self.input_exhausted = False
        self.program_ended = False
        self.exit_code = 0

    def set_input_marker(self, pos: int):
        """Mark where user input begins."""
        self.input_marker_position = pos
        self.input_read_position = pos + 1  # Start after marker

    def get_next_input_position(self) -> int:
        """Get position of next input character."""
        return self.input_read_position

    def advance_input(self):
        """Move to next input character."""
        self.input_read_position += 1


class GetcharBinaryFFN(PureFFN):
    """
    GETCHAR that reads from input stream using binary position.

    1. Reads current input position from IO state slots
    2. Encodes position as binary query
    3. Uses binary attention to read character at that position
    4. Advances input position
    """

    # Extended embedding slots for binary I/O
    INPUT_POS_BITS = 112  # Binary encoding of input read position
    OUTPUT_POS = 128      # Current output position

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=32)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # When GETCHAR active:
            # 1. Signal that we need input from binary-addressed position
            # 2. The attention layer will do the actual read

            # Set IO_NEED_INPUT
            self.W_up[0, E.OP_START + Opcode.GETCHAR] = S
            self.W_gate[0, E.IO_INPUT_READY] = -1.0
            self.b_gate[0] = 1.0
            self.W_down[E.IO_NEED_INPUT, 0] = 1.0 / S

            # Copy input position bits to query slots for attention
            # (Binary attention will use these to read from correct position)
            for k in range(16):
                row = k + 1
                if row < 32:
                    self.W_up[row, E.OP_START + Opcode.GETCHAR] = S
                    # This would copy from INPUT_POS_BITS + k if we had those slots
                    self.b_gate[row] = 1.0


class PutcharBinaryFFN(PureFFN):
    """
    PUTCHAR that writes to output stream.

    In LLM context, this emits a character to the model's output.
    The character becomes part of the generated token sequence.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Copy character to IO_CHAR
            self.W_up[0, E.OP_START + Opcode.PUTCHAR] = S
            self.W_gate[0, E.NIB_A] = 1.0
            self.W_down[E.IO_CHAR, 0] = 1.0 / S

            # Set output ready flag
            self.W_up[1, E.OP_START + Opcode.PUTCHAR] = S
            self.b_gate[1] = 1.0
            self.W_down[E.IO_OUTPUT_READY, 1] = 1.0 / S


# =============================================================================
# BINARY MEMORY ADDRESSING
# =============================================================================

class BinaryAddressEncoder(nn.Module):
    """
    Encode a 32-bit address as 32 binary bits in the embedding.

    Used for memory operations that need to address specific positions.
    """

    NUM_ADDR_BITS = 32
    ADDR_BIT_START = 96  # After E.DIM

    def encode_address(self, embedding: torch.Tensor, addr: int, slot_start: int = None):
        """
        Encode address as binary bits in embedding.

        Args:
            embedding: [batch, seq, dim] tensor
            addr: Address to encode
            slot_start: Starting slot for bits (default: ADDR_BIT_START)
        """
        if slot_start is None:
            slot_start = self.ADDR_BIT_START

        for k in range(self.NUM_ADDR_BITS):
            bit = (addr >> k) & 1
            embedding[:, :, slot_start + k] = float(bit)

    def decode_address(self, embedding: torch.Tensor, pos: int = 0, slot_start: int = None) -> int:
        """
        Decode address from binary bits in embedding.

        Args:
            embedding: [batch, seq, dim] tensor
            pos: Sequence position to read from
            slot_start: Starting slot for bits

        Returns:
            Decoded address
        """
        if slot_start is None:
            slot_start = self.ADDR_BIT_START

        addr = 0
        for k in range(self.NUM_ADDR_BITS):
            bit = int(embedding[0, pos, slot_start + k].item() > 0.5)
            addr |= (bit << k)
        return addr


class MemoryReadAttention(PureAttention):
    """
    Read from memory using binary address matching.

    The address is encoded as binary bits in the query.
    Each memory position has its address encoded as binary bits in the key.
    Perfect bit match → attention weight 1.0, copy that position's value.
    """

    def __init__(self, num_bits: int = 32, addr_slot_start: int = 96):
        super().__init__(E.DIM + num_bits)
        self.num_bits = num_bits
        self.addr_slot_start = addr_slot_start

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # Query: project address bits
            for k in range(self.num_bits):
                self.W_Q[k, self.addr_slot_start + k] = S

            # Key: project position bits (each position's address)
            for k in range(self.num_bits):
                self.W_K[k, self.addr_slot_start + k] = S

            # Value: project content (the memory value at each position)
            # Memory value stored in NIB_A for simplicity
            self.W_V[0, E.NIB_A] = 1.0


class MemoryWriteAttention(PureAttention):
    """
    Write to memory using binary address matching.

    Similar to read, but updates the value at the matched position.
    In an autoregressive LLM context, "writing" means the value
    will be available for future reads from that position.
    """

    def __init__(self, num_bits: int = 32, addr_slot_start: int = 96):
        super().__init__(E.DIM + num_bits)
        self.num_bits = num_bits
        self.addr_slot_start = addr_slot_start

    def _bake_weights(self):
        # Similar structure to read
        S = E.SCALE
        with torch.no_grad():
            for k in range(self.num_bits):
                self.W_Q[k, self.addr_slot_start + k] = S
                self.W_K[k, self.addr_slot_start + k] = S

            # Value is the data to write
            self.W_V[0, E.RESULT] = 1.0


# =============================================================================
# LLM I/O HANDLER
# =============================================================================

class LLMNativeIOHandler:
    """
    I/O Handler for LLM-native operation.

    Input: Comes from user messages (tokenized text after <INPUT> marker)
    Output: Goes to model's generated output (non-thinking)

    Memory Model:
    - Sequence positions from BOS are memory addresses
    - Binary encoding allows O(1) addressing via attention
    - Input stream starts at input_marker_position + 1
    - Output stream is the model's generation
    """

    def __init__(self):
        self.state = LLMIOState()
        self.position_encoder = BinaryPositionEncoder()
        self.address_encoder = BinaryAddressEncoder()
        self.output_buffer = []  # Characters to output
        self.input_buffer = []   # Characters from user input
        self.input_pos = 0

    def set_input(self, text: str):
        """Set the input text (from user message)."""
        self.input_buffer = list(text)
        self.input_pos = 0

    def get_output(self) -> str:
        """Get the accumulated output."""
        return ''.join(self.output_buffer)

    def process_getchar(self, embedding: torch.Tensor) -> Optional[int]:
        """
        Process GETCHAR: read next character from input.

        Returns character code, or -1 if input exhausted.
        """
        if self.input_pos < len(self.input_buffer):
            char = ord(self.input_buffer[self.input_pos])
            self.input_pos += 1

            # Write character to embedding
            embedding[0, 0, E.IO_CHAR] = float(char)
            embedding[0, 0, E.IO_INPUT_READY] = 1.0
            embedding[0, 0, E.IO_NEED_INPUT] = 0.0

            return char
        else:
            # EOF
            self.state.input_exhausted = True
            embedding[0, 0, E.IO_CHAR] = 255.0  # -1 as unsigned
            embedding[0, 0, E.IO_INPUT_READY] = 1.0
            embedding[0, 0, E.IO_NEED_INPUT] = 0.0
            return -1

    def process_putchar(self, embedding: torch.Tensor) -> str:
        """
        Process PUTCHAR: emit character to output.

        Returns the character as a string.
        """
        char_code = int(embedding[0, 0, E.IO_CHAR].item()) & 0xFF
        char = chr(char_code)
        self.output_buffer.append(char)

        # Clear output ready flag
        embedding[0, 0, E.IO_OUTPUT_READY] = 0.0

        return char

    def check_io(self, embedding: torch.Tensor) -> Optional[str]:
        """
        Check embedding for I/O requests and handle them.

        Returns output character if PUTCHAR, None otherwise.
        """
        # Check for PUTCHAR
        if embedding[0, 0, E.IO_OUTPUT_READY].item() > 0.5:
            return self.process_putchar(embedding)

        # Check for GETCHAR
        if embedding[0, 0, E.IO_NEED_INPUT].item() > 0.5:
            self.process_getchar(embedding)
            return None

        # Check for EXIT
        if embedding[0, 0, E.IO_PROGRAM_END].item() > 0.5:
            self.state.program_ended = True
            self.state.exit_code = int(embedding[0, 0, E.IO_EXIT_CODE].item())
            return None

        return None

    def encode_positions(self, embedding: torch.Tensor) -> torch.Tensor:
        """Add binary position encoding to embedding."""
        return self.position_encoder(embedding)


# =============================================================================
# DEMO
# =============================================================================

def demo_binary_position_io():
    """Demonstrate binary position I/O."""
    print("=" * 60)
    print("Binary Position I/O Demo")
    print("=" * 60)
    print()

    # Create handler
    handler = LLMNativeIOHandler()
    handler.set_input("Hello World!")

    # Create embedding
    embedding = torch.zeros(1, E.NUM_POSITIONS, E.DIM + 32)  # Extra space for position bits

    # Add binary position encoding
    embedding = handler.encode_positions(embedding)

    print("Binary position encoding (first 8 positions):")
    for pos in range(8):
        bits = [int(embedding[0, pos, BinaryPositionEncoder.BIT_START + k].item()) for k in range(8)]
        print(f"  Position {pos}: {bits} = {sum(b << k for k, b in enumerate(bits))}")

    print()
    print("Simulating GETCHAR loop:")

    # Simulate reading input
    for i in range(5):
        embedding[0, 0, E.IO_NEED_INPUT] = 1.0
        handler.check_io(embedding)
        char = int(embedding[0, 0, E.IO_CHAR].item())
        print(f"  GETCHAR -> {char} ('{chr(char)}')")

    print()
    print("Simulating PUTCHAR loop:")

    # Simulate writing output
    for c in "Test":
        embedding[0, 0, E.IO_CHAR] = float(ord(c))
        embedding[0, 0, E.IO_OUTPUT_READY] = 1.0
        output = handler.check_io(embedding)
        print(f"  PUTCHAR('{c}') -> '{output}'")

    print(f"\nOutput buffer: '{handler.get_output()}'")

    print()
    print("Binary address matching:")

    # Test binary address matching
    matcher = BinaryMatchAttention(num_bits=8)

    # Create test embedding with values at different positions
    test_emb = torch.zeros(1, 16, E.DIM + 16)
    test_emb = handler.position_encoder(test_emb)

    # Store different values at different positions
    for pos in range(16):
        test_emb[0, pos, E.RESULT] = float(pos * 10)  # Value = position * 10

    # Query for specific positions
    for query_addr in [0, 5, 10, 15]:
        result = matcher(test_emb, query_addr)
        print(f"  Read address {query_addr}: value = {result[0, E.RESULT].item():.0f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    demo_binary_position_io()
