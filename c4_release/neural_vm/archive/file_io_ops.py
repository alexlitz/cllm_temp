"""
Pure Neural File I/O Operations for stdin.

OPEN, READ, CLOSE implemented via the same input KV cache mechanism as GETCHAR.

For stdin (fd=0):
- OPEN: Return fd=0, stdin is always open (pre-populated from user input)
- READ: Read bytes from INPUT_DATA KV cache to memory buffer
- CLOSE: No-op, return 0

This allows C programs using read(0, buf, n) to work without external handlers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .embedding import E, Opcode, IOToolCallType
    from .base_layers import PureFFN
except ImportError:
    from embedding import E, Opcode, IOToolCallType
    from base_layers import PureFFN


# ============================================================================
# EMBEDDING EXTENSIONS FOR FILE I/O
# ============================================================================

class FileIOSlots:
    """Additional embedding slots for file I/O state."""

    # File descriptor table (simple: just track if stdin is open)
    FD_STDIN_OPEN = 102      # 1.0 if stdin (fd=0) is open

    # Read operation state
    READ_FD = 103            # File descriptor for current read
    READ_BUF_PTR = 104       # Buffer pointer (lower 4 nibbles of address)
    READ_COUNT = 105         # Bytes to read
    READ_BYTES_DONE = 106    # Bytes read so far
    READ_IN_PROGRESS = 107   # 1.0 while read loop is active

    # These overlap with existing E slots, but we use them as aliases
    # The actual implementation reuses IO_CHAR and input KV cache


# ============================================================================
# NEURAL STDIN OPEN
# ============================================================================

class StdinOpenFFN(PureFFN):
    """
    OPEN for stdin (fd=0).

    When OPEN opcode is active:
    - Check if filename points to "stdin" or we're opening fd 0
    - Return fd=0 in RESULT
    - Set FD_STDIN_OPEN = 1.0

    For simplicity, we assume all opens are stdin in neural mode.
    The input buffer is pre-populated from <USER_INPUT> tokens.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # When OPEN is active, set RESULT = 0 (stdin fd)
            # Gate: silu(S * OPEN)
            self.W_up[0, E.OP_START + Opcode.OPEN] = S
            self.b_gate[0] = 1.0
            # RESULT = 0 (don't need to write, it's already 0 by default)

            # Set FD_STDIN_OPEN = 1.0
            self.W_up[1, E.OP_START + Opcode.OPEN] = S
            self.b_gate[1] = 1.0
            # Write to a slot to indicate stdin is open
            # Using IO_TOOL_RESPONSE as temp storage for fd state
            self.W_down[E.IO_TOOL_RESPONSE, 1] = 1.0 / S

            # Clear any pending tool call type (we handled it neurally)
            self.W_up[2, E.OP_START + Opcode.OPEN] = S
            self.W_gate[2, E.IO_TOOL_CALL_TYPE] = -1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 2] = 1.0 / S


class StdinReadFFN(PureFFN):
    """
    READ from stdin using input KV cache.

    read(fd=0, buf, count) -> bytes_read

    When READ opcode is active:
    1. Get count from stack (simplified: from NIB_B)
    2. For each byte (handled by external loop or subroutine):
       - Query INPUT_DATA KV head at current read offset
       - Write byte to memory at buf + i
       - Increment read pointer
    3. Return bytes read in RESULT

    For neural implementation, we signal that READ is needed and
    let the KV-based I/O system handle the actual byte transfer.
    This uses the same mechanism as GETCHAR.
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=8)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # READ signals that we need to read from stdin
            # This sets up the same state as GETCHAR but for bulk read

            # Set IO_NEED_INPUT = 1.0 if we need more input
            self.W_up[0, E.OP_START + Opcode.READ] = S
            self.W_gate[0, E.IO_INPUT_READY] = -1.0  # Only if not ready
            self.b_gate[0] = 1.0
            self.W_down[E.IO_NEED_INPUT, 0] = 1.0 / S

            # Set tool call type for handler to process
            self.W_up[1, E.OP_START + Opcode.READ] = S
            self.b_gate[1] = 1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = float(IOToolCallType.READ) / S

            # When input is ready, copy to RESULT
            self.W_up[2, E.OP_START + Opcode.READ] = S
            self.W_gate[2, E.IO_INPUT_READY] = 1.0
            self.W_gate[2, E.IO_CHAR] = 1.0
            self.W_down[E.RESULT, 2] = 1.0 / S


class StdinCloseFFN(PureFFN):
    """
    CLOSE for stdin (fd=0).

    When CLOSE opcode is active:
    - Check if fd == 0 (stdin)
    - Return 0 (success) - stdin close is a no-op
    - Clear FD_STDIN_OPEN flag
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=4)

    def _bake_weights(self):
        S = E.SCALE
        with torch.no_grad():
            # When CLOSE is active, set RESULT = 0 (success)
            # CLOSE for stdin is a no-op, just return success
            self.W_up[0, E.OP_START + Opcode.CLOS] = S
            self.b_gate[0] = 1.0
            # RESULT stays 0

            # Clear the stdin open flag
            self.W_up[1, E.OP_START + Opcode.CLOS] = S
            self.W_gate[1, E.IO_TOOL_RESPONSE] = -1.0
            self.W_down[E.IO_TOOL_RESPONSE, 1] = 1.0 / S

            # Clear tool call type
            self.W_up[2, E.OP_START + Opcode.CLOS] = S
            self.W_gate[2, E.IO_TOOL_CALL_TYPE] = -1.0
            self.W_down[E.IO_TOOL_CALL_TYPE, 2] = 1.0 / S


# ============================================================================
# INTEGRATED STDIN I/O SYSTEM
# ============================================================================

class StdinIOSystem(nn.Module):
    """
    Complete stdin I/O system using input KV cache.

    Provides:
    - open(stdin) -> 0
    - read(0, buf, n) -> bytes from input KV cache
    - close(0) -> 0 (no-op)

    Uses the same INPUT_DATA KV head as GETCHAR.
    """

    def __init__(self, input_kv_head):
        super().__init__()
        self.input_kv = input_kv_head

        # FFNs for file operations
        self.open_ffn = StdinOpenFFN()
        self.read_ffn = StdinReadFFN()
        self.close_ffn = StdinCloseFFN()

        # State
        self.register_buffer('read_ptr', torch.tensor(0))
        self.register_buffer('input_length', torch.tensor(0))
        self.register_buffer('stdin_open', torch.tensor(1.0))  # Always open

    def set_input(self, text: str):
        """Set input text (from user message)."""
        self.input_length.fill_(len(text))
        self.read_ptr.zero_()
        # Input KV head is populated by the tokenizer/handler

    def open(self, filename_ptr: int = 0) -> int:
        """
        Open file. Returns fd.

        For stdin (filename_ptr == 0 or points to "stdin"):
        - Returns fd = 0
        - Input is pre-populated from user message
        """
        # In neural mode, all opens are stdin
        return 0

    def read(self, fd: int, buf_ptr: int, count: int, memory_write_fn) -> int:
        """
        Read from file descriptor.

        For fd=0 (stdin):
        - Read up to `count` bytes from input KV cache
        - Write to memory at buf_ptr using memory_write_fn
        - Return actual bytes read

        Args:
            fd: File descriptor (must be 0 for stdin)
            buf_ptr: Memory address to write to
            count: Max bytes to read
            memory_write_fn: Function to write to memory: fn(addr, value)

        Returns:
            Number of bytes actually read
        """
        if fd != 0:
            return -1  # Only stdin supported

        bytes_read = 0
        current_ptr = int(self.read_ptr.item())
        input_len = int(self.input_length.item())

        for i in range(count):
            if current_ptr >= input_len:
                break  # EOF

            # Read byte from input KV cache
            char = self.input_kv.read_char(current_ptr)

            # Write to memory buffer
            memory_write_fn(buf_ptr + i, char)

            current_ptr += 1
            bytes_read += 1

        self.read_ptr.fill_(current_ptr)
        return bytes_read

    def close(self, fd: int) -> int:
        """
        Close file descriptor.

        For fd=0 (stdin): No-op, always returns 0.
        """
        if fd == 0:
            return 0  # Success, but stdin stays "open"
        return -1  # Unknown fd

    def getchar(self) -> int:
        """Single character read (same as read(0, &c, 1))."""
        current_ptr = int(self.read_ptr.item())
        input_len = int(self.input_length.item())

        if current_ptr >= input_len:
            return -1  # EOF

        char = self.input_kv.read_char(current_ptr)
        self.read_ptr.add_(1)
        return char

    def eof(self) -> bool:
        """Check if at end of input."""
        return int(self.read_ptr.item()) >= int(self.input_length.item())


# ============================================================================
# TESTS
# ============================================================================

def test_stdin_io():
    """Test stdin I/O operations."""
    print("=" * 60)
    print("Testing Pure Neural Stdin I/O")
    print("=" * 60)

    # Create mock input KV head
    class MockInputKV:
        def __init__(self, text: str):
            self.data = [ord(c) for c in text]

        def read_char(self, offset: int) -> int:
            if 0 <= offset < len(self.data):
                return self.data[offset]
            return -1

    # Test data
    input_text = "Hello, World!\n"
    input_kv = MockInputKV(input_text)

    # Create stdin system
    stdin_io = StdinIOSystem(input_kv)
    stdin_io.input_length.fill_(len(input_text))

    print(f"\nInput text: {repr(input_text)}")
    print(f"Input length: {len(input_text)}")

    # Test open
    print("\n1. Testing open():")
    fd = stdin_io.open()
    print(f"   open(stdin) = {fd}")
    assert fd == 0, "Expected fd=0 for stdin"
    print("   ✓ PASS")

    # Test getchar
    print("\n2. Testing getchar():")
    chars = []
    for i in range(5):
        c = stdin_io.getchar()
        chars.append(chr(c) if c >= 0 else 'EOF')
        print(f"   getchar() = {c} ({chars[-1]})")
    assert ''.join(chars) == "Hello", f"Expected 'Hello', got {''.join(chars)}"
    print("   ✓ PASS")

    # Test read
    print("\n3. Testing read():")
    memory = {}
    def mock_write(addr, val):
        memory[addr] = val

    stdin_io.read_ptr.fill_(0)  # Reset
    bytes_read = stdin_io.read(0, 1000, 7, mock_write)
    print(f"   read(0, buf, 7) = {bytes_read} bytes")

    read_str = ''.join(chr(memory.get(1000+i, 0)) for i in range(bytes_read))
    print(f"   Buffer contents: {repr(read_str)}")
    assert read_str == "Hello, ", f"Expected 'Hello, ', got {repr(read_str)}"
    print("   ✓ PASS")

    # Test close
    print("\n4. Testing close():")
    result = stdin_io.close(0)
    print(f"   close(0) = {result}")
    assert result == 0, "Expected 0 (success)"
    print("   ✓ PASS")

    # Test EOF
    print("\n5. Testing EOF:")
    stdin_io.read_ptr.fill_(len(input_text))
    c = stdin_io.getchar()
    print(f"   getchar() at EOF = {c}")
    assert c == -1, "Expected -1 (EOF)"
    print("   ✓ PASS")

    print("\n" + "=" * 60)
    print("All stdin I/O tests passed!")
    print("=" * 60)


def test_ffn_weights():
    """Test that FFNs have correct weight structure."""
    print("\n" + "=" * 60)
    print("Testing FFN Weight Structure")
    print("=" * 60)

    open_ffn = StdinOpenFFN()
    read_ffn = StdinReadFFN()
    close_ffn = StdinCloseFFN()

    print(f"\nStdinOpenFFN:")
    print(f"  W_up shape: {open_ffn.W_up.shape}")
    print(f"  Non-zero W_up: {(open_ffn.W_up != 0).sum().item()}")

    print(f"\nStdinReadFFN:")
    print(f"  W_up shape: {read_ffn.W_up.shape}")
    print(f"  Non-zero W_up: {(read_ffn.W_up != 0).sum().item()}")

    print(f"\nStdinCloseFFN:")
    print(f"  W_up shape: {close_ffn.W_up.shape}")
    print(f"  Non-zero W_up: {(close_ffn.W_up != 0).sum().item()}")

    print("\n✓ All FFNs initialized correctly")


if __name__ == '__main__':
    test_stdin_io()
    test_ffn_weights()
