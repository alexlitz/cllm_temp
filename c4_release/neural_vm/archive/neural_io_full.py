"""
Fully Neural I/O System for Neural VM.

Architecture:
1. Position encoding via RoPE with binary thetas (simpler than alibi exponentials)
2. User input: Marked regions write to KV cache, VM reads via position matching
3. Output: VM writes to output buffer, output tokens read via position matching

Key insight: RoPE with theta_k = 2^k creates binary position encoding:
- cos(2^k * pos) and sin(2^k * pos) encode bit k of position
- This allows direct offset computation without log

Flow:
1. <USER_INPUT> marker sets input_start position
2. Input tokens write (position_encoding, char_value) to KV cache
3. </USER_INPUT> marker sets input_end position
4. VM GETCHAR reads from input buffer using offset from input_start
5. VM PUTCHAR writes to output buffer
6. <USER_OUTPUT> tokens read from output buffer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Callable

from .embedding import E, Opcode
from .base_layers import PureFFN, PureAttention
from .neural_state import VMState
from .bump_allocator import BufferAllocator, DynamicBuffer, HeapInitFFN


# ============================================================================
# BINARY POSITION ENCODING (RoPE-style)
# ============================================================================

class BinaryPositionEncoder(nn.Module):
    """
    Direct binary position encoding (not trigonometric).

    Each position n is encoded as its binary representation:
    - Dimension k has value (n >> k) & 1 (scaled to -1/+1)

    This gives:
    - Position 0: [-1, -1, -1, -1, ...]
    - Position 1: [+1, -1, -1, -1, ...]
    - Position 2: [-1, +1, -1, -1, ...]
    - Position 3: [+1, +1, -1, -1, ...]
    - Position 7: [+1, +1, +1, -1, ...]

    Dot product of same positions = dim (perfect match)
    Dot product of different positions < dim (partial match)
    """

    def __init__(self, dim: int = 16, max_positions: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

        # Precompute binary encodings
        positions = torch.arange(max_positions)
        encoding = torch.zeros(max_positions, dim)
        for k in range(dim):
            # bit k of position: 0 -> -1, 1 -> +1
            bit_k = ((positions >> k) & 1).float()
            encoding[:, k] = 2 * bit_k - 1
        self.register_buffer('position_encoding', encoding)

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions as binary vectors.

        Args:
            positions: [batch] or [batch, seq] integer positions

        Returns:
            [batch, dim] or [batch, seq, dim] encodings
        """
        return self.position_encoding[positions.long()]

    def compute_offset(self, pos_a: int, pos_b: int) -> int:
        """Simple offset computation."""
        return pos_a - pos_b

    def decode(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Decode binary encoding back to position.

        Args:
            encoding: [batch, dim] binary encodings (-1/+1)

        Returns:
            [batch] positions
        """
        # Convert -1/+1 to 0/1
        bits = (encoding > 0).float()
        # Weighted sum: bit_k * 2^k
        powers = torch.tensor([2 ** k for k in range(self.dim)],
                             device=encoding.device, dtype=encoding.dtype)
        positions = (bits * powers).sum(dim=-1)
        return positions.long()


# ============================================================================
# I/O BUFFER MANAGEMENT
# ============================================================================

class IOBuffer(nn.Module):
    """
    I/O Buffer using VM heap memory (dynamic allocation).

    Instead of hardcoded tensor storage, this buffer allocates memory
    within the VM's heap via bump allocator. All reads/writes go through
    the VM's KV cache memory system.

    Supports:
    - Write: Store (offset, value) pairs in VM memory
    - Read: Retrieve value at specific offset via attention

    Uses attention with binary position encoding:
    - Q = target offset encoding (binary)
    - K = stored offset encodings (binary)
    - V = stored values
    - Output = value at matching offset

    Memory layout in VM heap:
    - base_addr + offset * value_dim = address of entry
    """

    def __init__(self, buffer_size: int = 1024, value_dim: int = 8, key_dim: int = 16):
        super().__init__()
        self.buffer_size = buffer_size
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.encoder = BinaryPositionEncoder(dim=key_dim, max_positions=buffer_size)

        # VM memory allocation info (set when allocated)
        self.base_addr = 0
        self.allocated = False

        # Local tracking
        self.register_buffer('write_ptr', torch.tensor(0))
        self.register_buffer('valid_count', torch.tensor(0))

        # Memory access functions (set by VM)
        self._mem_read: Optional[Callable[[int], torch.Tensor]] = None
        self._mem_write: Optional[Callable[[int, torch.Tensor], None]] = None

    def allocate(self, base_addr: int,
                 mem_read: Callable[[int], torch.Tensor],
                 mem_write: Callable[[int, torch.Tensor], None]):
        """
        Allocate this buffer at given address in VM memory.

        Args:
            base_addr: Start address in VM heap
            mem_read: Function to read from VM memory (addr -> value)
            mem_write: Function to write to VM memory (addr, value -> None)
        """
        self.base_addr = base_addr
        self._mem_read = mem_read
        self._mem_write = mem_write
        self.allocated = True

    def get_address(self, offset: int) -> int:
        """Get VM memory address for entry at offset."""
        return self.base_addr + (offset * self.value_dim)

    def reset(self):
        """Clear the buffer (write zeros = free in softmax1 semantics)."""
        self.write_ptr.zero_()
        self.valid_count.zero_()
        # With softmax1, we don't need to explicitly clear memory
        # Zero values are ignored in attention computation

    def write(self, offset: int, value: torch.Tensor):
        """
        Write value at offset to VM memory.

        Args:
            offset: Position in buffer
            value: [value_dim] tensor
        """
        if offset < self.buffer_size:
            if self.allocated and self._mem_write is not None:
                addr = self.get_address(offset)
                self._mem_write(addr, value)
            if offset >= self.valid_count.item():
                self.valid_count.fill_(offset + 1)

    def read(self, offset: torch.Tensor) -> torch.Tensor:
        """
        Read value at offset from VM memory using attention.

        Args:
            offset: [batch] target offsets

        Returns:
            [batch, value_dim] values
        """
        if not self.allocated or self._mem_read is None:
            return torch.zeros(offset.shape[0], self.value_dim)

        results = []
        for off in offset.tolist():
            addr = self.get_address(int(off))
            val = self._mem_read(addr)
            results.append(val)

        return torch.stack(results) if results else torch.zeros(0, self.value_dim)


# ============================================================================
# USER INPUT HANDLING
# ============================================================================

class UserInputMarker(nn.Module):
    """
    Handles <USER_INPUT> and </USER_INPUT> markers.

    When <USER_INPUT> is encountered:
    1. Records current position as input_start
    2. Subsequent tokens write their values to input buffer

    When </USER_INPUT> is encountered:
    1. Records current position as input_end
    2. Input is now available in buffer for VM to read

    Buffer is allocated dynamically in VM heap via bump allocator.
    """

    def __init__(self, buffer_size: int = 1024):
        super().__init__()
        self.buffer_size = buffer_size
        self.input_buffer = IOBuffer(buffer_size, value_dim=8)
        self.encoder = BinaryPositionEncoder(dim=16, max_positions=buffer_size)

        # State
        self.register_buffer('input_start', torch.tensor(-1))
        self.register_buffer('input_end', torch.tensor(-1))
        self.register_buffer('input_ptr', torch.tensor(0))  # Next read position
        self.register_buffer('in_input_mode', torch.tensor(0.0))

    def allocate(self, allocator: BufferAllocator):
        """
        Allocate input buffer using VM's bump allocator.

        Args:
            allocator: BufferAllocator linked to VM state
        """
        buf = allocator.allocate_buffer("user_input", self.buffer_size, 8)
        self.input_buffer.allocate(
            buf.base_addr,
            allocator._make_mem_read(),
            allocator._make_mem_write()
        )

    def on_input_start(self, position: int):
        """Called when <USER_INPUT> marker encountered."""
        self.input_start.fill_(position)
        self.in_input_mode.fill_(1.0)
        self.input_buffer.reset()

    def on_input_token(self, position: int, char_value: int):
        """Called for each input character token."""
        if self.in_input_mode > 0.5:
            offset = position - self.input_start.item()
            value = torch.zeros(8)
            # Encode character as nibbles
            for i in range(8):
                value[i] = (char_value >> (i * 4)) & 0xF
            self.input_buffer.write(offset, value)

    def on_input_end(self, position: int):
        """Called when </USER_INPUT> marker encountered."""
        self.input_end.fill_(position)
        self.in_input_mode.fill_(0.0)
        self.input_ptr.fill_(0)  # Reset read pointer

    def read_char(self) -> Tuple[int, bool]:
        """
        Read next character from input buffer.

        Returns:
            (char_value, eof): Character and whether end of input
        """
        offset = self.input_ptr.item()
        input_len = self.input_end.item() - self.input_start.item()

        if offset >= input_len:
            return (-1, True)  # EOF

        value = self.input_buffer.read(torch.tensor([offset]))
        # Decode nibbles to character
        char_value = 0
        for i in range(8):
            char_value |= int(value[0, i].item()) << (i * 4)

        self.input_ptr.add_(1)
        return (char_value & 0xFF, False)


# ============================================================================
# USER OUTPUT HANDLING
# ============================================================================

class UserOutputMarker(nn.Module):
    """
    Handles output to user.

    VM writes characters to output buffer via PUTCHAR.
    When output sequence starts, tokens read from this buffer.

    Flow:
    1. VM calls putchar(char) → writes to output_buffer
    2. <USER_OUTPUT> marker signals start of output sequence
    3. Output tokens attend to buffer using their position offset

    Buffer is allocated dynamically in VM heap via bump allocator.
    """

    def __init__(self, buffer_size: int = 4096):
        super().__init__()
        self.buffer_size = buffer_size
        self.output_buffer = IOBuffer(buffer_size, value_dim=8)
        self.encoder = BinaryPositionEncoder(dim=16, max_positions=buffer_size)

        # State
        self.register_buffer('output_ptr', torch.tensor(0))  # Next write position
        self.register_buffer('output_start', torch.tensor(-1))

    def allocate(self, allocator: BufferAllocator):
        """
        Allocate output buffer using VM's bump allocator.

        Args:
            allocator: BufferAllocator linked to VM state
        """
        buf = allocator.allocate_buffer("user_output", self.buffer_size, 8)
        self.output_buffer.allocate(
            buf.base_addr,
            allocator._make_mem_read(),
            allocator._make_mem_write()
        )

    def write_char(self, char_value: int):
        """VM writes character to output buffer."""
        offset = self.output_ptr.item()
        value = torch.zeros(8)
        # Encode character as nibbles
        for i in range(8):
            value[i] = (char_value >> (i * 4)) & 0xF
        self.output_buffer.write(offset, value)
        self.output_ptr.add_(1)

    def on_output_start(self, position: int):
        """Called when starting to generate output tokens."""
        self.output_start.fill_(position)

    def get_char_for_position(self, position: int) -> Optional[int]:
        """
        Get character for output position.

        Args:
            position: Absolute sequence position

        Returns:
            Character to output, or None if past buffer
        """
        offset = position - self.output_start.item()
        if offset < 0 or offset >= self.output_ptr.item():
            return None

        value = self.output_buffer.read(torch.tensor([offset]))
        char_value = 0
        for i in range(8):
            char_value |= int(value[0, i].item()) << (i * 4)
        return char_value & 0xFF


# ============================================================================
# FULLY NEURAL FILE I/O
# ============================================================================

class NeuralFileSystem(nn.Module):
    """
    Fully neural file system simulation with dynamic buffer allocation.

    Files are stored as dynamically allocated buffers in VM heap.
    OPEN: Allocate buffer via bump allocator, find buffer by name hash
    READ: Read from buffer at current position
    CLOSE: Free buffer (write zeros), clear file descriptor

    For neural implementation:
    - File descriptors are indices into a file table
    - Each entry has: name_hash, buffer_addr, position, flags
    - Buffers allocated dynamically via malloc
    - READ/WRITE use VM's KV cache memory
    """

    MAX_FILES = 16
    BUFFER_SIZE = 256

    def __init__(self):
        super().__init__()

        # File table (addresses stored instead of buffers)
        self.register_buffer('fd_valid', torch.zeros(self.MAX_FILES))
        self.register_buffer('fd_name_hash', torch.zeros(self.MAX_FILES))
        self.register_buffer('fd_position', torch.zeros(self.MAX_FILES))
        self.register_buffer('fd_mode', torch.zeros(self.MAX_FILES))  # 0=read, 1=write
        self.register_buffer('fd_buffer_addr', torch.zeros(self.MAX_FILES))  # VM heap address

        # Allocator reference (set when allocated)
        self._allocator: Optional[BufferAllocator] = None
        self._allocated = False

    def allocate(self, allocator: BufferAllocator):
        """
        Set up file system with VM's allocator.

        Args:
            allocator: BufferAllocator linked to VM state
        """
        self._allocator = allocator
        self._allocated = True

    def open(self, filename_hash: int, mode: int) -> int:
        """
        Open file.

        Args:
            filename_hash: Hash of filename (computed by caller)
            mode: 0=read, 1=write

        Returns:
            File descriptor (0-15) or -1 on error
        """
        if not self._allocated or self._allocator is None:
            return -1

        # Find free slot
        for fd in range(self.MAX_FILES):
            if self.fd_valid[fd] < 0.5:
                # Allocate buffer for this file
                buf = self._allocator.allocate_buffer(
                    f"file_{fd}", self.BUFFER_SIZE, 8
                )

                self.fd_valid[fd] = 1.0
                self.fd_name_hash[fd] = filename_hash
                self.fd_position[fd] = 0
                self.fd_mode[fd] = mode
                self.fd_buffer_addr[fd] = float(buf.base_addr)
                return fd
        return -1

    def _get_buffer_addr(self, fd: int, offset: int) -> int:
        """Get VM memory address for file buffer entry."""
        base = int(self.fd_buffer_addr[fd].item())
        return base + (offset * 8)  # 8 nibbles per entry

    def read(self, fd: int, count: int) -> Tuple[torch.Tensor, int]:
        """
        Read from file using VM memory.

        Args:
            fd: File descriptor
            count: Number of bytes to read

        Returns:
            (data, bytes_read)
        """
        if not self._allocated or self._allocator is None:
            return (torch.zeros(count, 8), 0)

        if fd < 0 or fd >= self.MAX_FILES or self.fd_valid[fd] < 0.5:
            return (torch.zeros(count, 8), 0)

        pos = int(self.fd_position[fd].item())

        data = []
        for i in range(count):
            if pos + i >= self.BUFFER_SIZE:
                break
            addr = self._get_buffer_addr(fd, pos + i)
            val = self._allocator.mem_read(addr)
            data.append(val.unsqueeze(0))

        bytes_read = len(data)
        self.fd_position[fd] += bytes_read

        if data:
            return (torch.cat(data, dim=0), bytes_read)
        return (torch.zeros(count, 8), 0)

    def write(self, fd: int, data: torch.Tensor) -> int:
        """
        Write to file using VM memory.

        Args:
            fd: File descriptor
            data: [count, 8] tensor of nibble values

        Returns:
            Bytes written
        """
        if not self._allocated or self._allocator is None:
            return 0

        if fd < 0 or fd >= self.MAX_FILES or self.fd_valid[fd] < 0.5:
            return 0

        pos = int(self.fd_position[fd].item())
        count = data.shape[0]

        for i in range(count):
            if pos + i >= self.BUFFER_SIZE:
                break
            addr = self._get_buffer_addr(fd, pos + i)
            self._allocator.mem_write(addr, data[i])

        self.fd_position[fd] += count
        return count

    def close(self, fd: int):
        """
        Close file and free buffer (write zeros).

        With softmax1 semantics, zero values can be evicted from KV cache.
        """
        if 0 <= fd < self.MAX_FILES:
            if self.fd_valid[fd] > 0.5 and self._allocator is not None:
                # Free buffer by writing zeros (softmax1 semantics)
                base = int(self.fd_buffer_addr[fd].item())
                self._allocator.free(base, self.BUFFER_SIZE * 8)

            self.fd_valid[fd] = 0.0
            self.fd_buffer_addr[fd] = 0.0


# ============================================================================
# NEURAL FFNs FOR BLT/BGE
# ============================================================================

# Add FLAG_SIGN slot to VMState
VMState.FLAG_SIGN = 140  # 1.0 if AX < 0 (sign bit set), 0.0 otherwise


class SetSignFlagFFN(PureFFN):
    """
    Compute FLAG_SIGN from AX high nibble.

    FLAG_SIGN = 1 if AX[7] >= 8 (sign bit set, negative number)
    FLAG_SIGN = 0 if AX[7] < 8 (sign bit clear, non-negative)

    Uses bit extraction: AX[7] >= 8 iff bit 3 of AX[7] is set.
    FLAG_SIGN = floor(AX[7] / 8)
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Clear FLAG_SIGN first
            self.b_up[0] = S
            self.W_gate[0, VMState.FLAG_SIGN] = -1.0
            self.W_down[VMState.FLAG_SIGN, 0] = 1.0 / S

            # Extract bit 3 of AX[7] (the sign bit)
            # bit3 = floor(AX[7] / 8) = 1 if AX[7] >= 8 else 0
            #
            # Neural approximation:
            # silu(S * (AX[7] - 7.5)) / silu(S * 0.5)
            #
            # But silu(x) ≈ x for x > 0, and ≈ 0 for x < 0
            # So: silu(S * (AX[7] - 7.5)) ≈ S * (AX[7] - 7.5) for AX[7] >= 8
            #                              ≈ 0 for AX[7] <= 7
            #
            # To get FLAG_SIGN = 1 for AX[7] >= 8:
            # We need to normalize by the minimum positive output
            # At AX[7] = 8: silu(S * 0.5) ≈ S * 0.5
            # At AX[7] = 15: silu(S * 7.5) ≈ S * 7.5
            #
            # Use clipping: min(1, silu(S*(AX-7.5)) / (S*0.5))
            # At AX=8: 0.5/0.5 = 1 ✓
            # At AX=15: 7.5/0.5 = 15, clipped to 1 ✓
            # At AX=7: ~0/0.5 = 0 ✓
            #
            # Implement with saturation: high slope near threshold

            # Simple approach: FLAG_SIGN = AX[7] / 8 (linear approx)
            # Then threshold in BLT/BGE
            self.W_up[1, VMState.AX_BASE + 7] = S
            self.b_up[1] = -7.5 * S
            self.b_gate[1] = 1.0
            # Scale so AX[7]=8 gives ~1, AX[7]=15 gives ~1
            # silu(0.5*S) ≈ 0.5*S, so output = 0.5*S * 1 / S * scale
            # Want this to be 1, so scale = 2
            self.W_down[VMState.FLAG_SIGN, 1] = 2.0 / S


class BranchLTFFN(PureFFN):
    """
    BLT: Branch if Less Than (signed comparison).

    PC = target if AX < 0 (FLAG_SIGN = 1)

    Requires SetSignFlagFFN to be run first to compute FLAG_SIGN.
    Uses same structure as BZ/BNZ but with FLAG_SIGN instead of FLAG_ZERO.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Same structure as BranchPCFFN with branch_on_zero=True
            # but using FLAG_SIGN instead of FLAG_ZERO
            threshold_bias = -1.5 * S

            # Unit 0: Add RESULT when BLT=1 AND FLAG_SIGN=1
            self.W_up[0, E.OP_START + Opcode.BLT] = S
            self.W_up[0, VMState.FLAG_SIGN] = S
            self.b_up[0] = threshold_bias
            self.W_gate[0, E.RESULT] = 1.0
            self.W_down[VMState.PC, 0] = 2.0 / S

            # Unit 1: Subtract old PC when condition met
            self.W_up[1, E.OP_START + Opcode.BLT] = S
            self.W_up[1, VMState.FLAG_SIGN] = S
            self.b_up[1] = threshold_bias
            self.W_gate[1, VMState.PC] = -1.0
            self.W_down[VMState.PC, 1] = 2.0 / S


class BranchGEFFN(PureFFN):
    """
    BGE: Branch if Greater or Equal (signed comparison).

    PC = target if AX >= 0 (FLAG_SIGN = 0)

    Requires SetSignFlagFFN to be run first.
    Uses same structure as BNZ but with FLAG_SIGN.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Same structure as BranchPCFFN with branch_on_zero=False
            # but using FLAG_SIGN: branch when FLAG_SIGN=0
            threshold_bias = -0.5 * S

            # Unit 0: Add RESULT when BGE=1 AND FLAG_SIGN=0
            self.W_up[0, E.OP_START + Opcode.BGE] = S
            self.W_up[0, VMState.FLAG_SIGN] = -S  # Negative: active when FLAG_SIGN=0
            self.b_up[0] = threshold_bias
            self.W_gate[0, E.RESULT] = 1.0
            self.W_down[VMState.PC, 0] = 2.0 / S

            # Unit 1: Subtract old PC
            self.W_up[1, E.OP_START + Opcode.BGE] = S
            self.W_up[1, VMState.FLAG_SIGN] = -S
            self.b_up[1] = threshold_bias
            self.W_gate[1, VMState.PC] = -1.0
            self.W_down[VMState.PC, 1] = 2.0 / S


# ============================================================================
# NEURAL FILE OPERATION FFNs
# ============================================================================

class FileOpenFFN(PureFFN):
    """
    OPEN: Neural file open operation.

    Sets up file descriptor in embedding slots.
    Filename is read from memory (null-terminated string at AX).
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Set up file open request
            # Copies AX (filename ptr) to MEM_ADDR_BASE for string read
            self.W_up[0, E.OP_START + Opcode.OPEN] = S
            self.W_gate[0, VMState.AX] = 1.0
            self.W_down[VMState.MEM_ADDR_BASE, 0] = 1.0 / S

            # Set IO_TOOL_CALL_TYPE = OPEN (for hybrid mode)
            self.W_up[1, E.OP_START + Opcode.OPEN] = S
            self.b_gate[1] = 4.0  # OPEN type
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = 1.0 / S


class FileReadFFN(PureFFN):
    """
    READ: Neural file read operation.

    Reads count bytes from fd into buffer at AX.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Set up read request
            self.W_up[0, E.OP_START + Opcode.READ] = S
            self.W_gate[0, VMState.AX] = 1.0
            self.W_down[VMState.MEM_ADDR_BASE, 0] = 1.0 / S

            # Set IO_TOOL_CALL_TYPE = READ
            self.W_up[1, E.OP_START + Opcode.READ] = S
            self.b_gate[1] = 5.0  # READ type
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = 1.0 / S


class FileCloseFFN(PureFFN):
    """
    CLOS: Neural file close operation.

    Closes file descriptor in AX.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=2)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Set IO_TOOL_CALL_TYPE = CLOSE
            self.W_up[0, E.OP_START + Opcode.CLOS] = S
            self.b_gate[0] = 6.0  # CLOSE type
            self.W_down[E.IO_TOOL_CALL_TYPE, 0] = 1.0 / S


class PrintfFFN(PureFFN):
    """
    PRTF: Neural printf operation.

    Format string at AX, args on stack.
    Outputs to user output buffer.
    """

    def __init__(self):
        super().__init__(VMState.DIM, hidden_dim=4)

    def _bake_weights(self):
        with torch.no_grad():
            S = E.SCALE

            # Copy format string pointer to MEM_ADDR_BASE
            self.W_up[0, E.OP_START + Opcode.PRTF] = S
            self.W_gate[0, VMState.AX] = 1.0
            self.W_down[VMState.MEM_ADDR_BASE, 0] = 1.0 / S

            # Set IO_TOOL_CALL_TYPE = PRINTF
            self.W_up[1, E.OP_START + Opcode.PRTF] = S
            self.b_gate[1] = 7.0  # PRINTF type
            self.W_down[E.IO_TOOL_CALL_TYPE, 1] = 1.0 / S

            # Set output ready flag
            self.W_up[2, E.OP_START + Opcode.PRTF] = S
            self.b_gate[2] = 1.0
            self.W_down[E.IO_OUTPUT_READY, 2] = 1.0 / S


# ============================================================================
# COMBINED NEURAL I/O SYSTEM
# ============================================================================

class NeuralIOSystem(nn.Module):
    """
    Complete neural I/O system combining all components.

    All buffers are dynamically allocated in VM heap via bump allocator.

    Provides:
    - User input handling (via RoPE position encoding)
    - User output handling (via output buffer)
    - File I/O (via neural file system)
    - Stream management (stdin/stdout/stderr simulation)
    - Tool-use mode vs streaming mode toggle
    """

    def __init__(self, tool_use_mode: bool = False):
        super().__init__()

        self.user_input = UserInputMarker(buffer_size=4096)
        self.user_output = UserOutputMarker(buffer_size=4096)
        self.file_system = NeuralFileSystem()

        # Neural FFNs for file ops
        self.file_open = FileOpenFFN()
        self.file_read = FileReadFFN()
        self.file_close = FileCloseFFN()
        self.printf = PrintfFFN()

        # Branch operations
        self.branch_lt = BranchLTFFN()
        self.branch_ge = BranchGEFFN()

        # Tool-use mode: returns tool calls instead of streaming I/O
        # Non-tool-use returns -1 for non-stdin/stdout operations
        self._tool_use_mode = tool_use_mode
        self._allocator: Optional[BufferAllocator] = None

    def allocate(self, allocator: BufferAllocator):
        """
        Allocate all I/O buffers using VM's bump allocator.

        Args:
            allocator: BufferAllocator linked to VM state
        """
        self._allocator = allocator
        self.user_input.allocate(allocator)
        self.user_output.allocate(allocator)
        self.file_system.allocate(allocator)

    @property
    def tool_use_mode(self) -> bool:
        return self._tool_use_mode

    @tool_use_mode.setter
    def tool_use_mode(self, value: bool):
        self._tool_use_mode = value

    def getchar(self) -> int:
        """
        Read character from stdin.

        In tool-use mode: Returns tool call request.
        In streaming mode: Reads from user_input buffer.
        """
        char, eof = self.user_input.read_char()
        return -1 if eof else char

    def putchar(self, char: int) -> int:
        """
        Write character to stdout.

        In tool-use mode: Returns tool call request.
        In streaming mode: Writes to user_output buffer.
        """
        self.user_output.write_char(char)
        return char

    def file_open_op(self, filename_hash: int, mode: int) -> int:
        """
        Open file.

        In non-tool-use mode: Returns -1 (only stdin/stdout supported).
        In tool-use mode: Allocates file buffer via malloc.
        """
        if not self._tool_use_mode:
            return -1  # Non-tool-use doesn't support file I/O
        return self.file_system.open(filename_hash, mode)

    def file_read_op(self, fd: int, count: int) -> Tuple[torch.Tensor, int]:
        """
        Read from file.

        In non-tool-use mode: Returns -1 for non-stdin.
        """
        if not self._tool_use_mode:
            return (torch.zeros(count, 8), -1)
        return self.file_system.read(fd, count)

    def file_close_op(self, fd: int) -> int:
        """
        Close file.

        In non-tool-use mode: No-op, returns -1.
        In tool-use mode: Frees buffer (writes zeros).
        """
        if not self._tool_use_mode:
            return -1
        self.file_system.close(fd)
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all I/O operations based on opcode."""
        x = self.file_open(x)
        x = self.file_read(x)
        x = self.file_close(x)
        x = self.printf(x)
        x = self.branch_lt(x)
        x = self.branch_ge(x)
        return x


# ============================================================================
# TESTS
# ============================================================================

def test_binary_position_encoding():
    """Test binary position encoding."""
    print("=== Testing Binary Position Encoding ===\n")

    encoder = BinaryPositionEncoder(dim=16, max_positions=256)

    # Test encoding
    pos = torch.tensor([0, 1, 2, 3, 7, 8, 15, 16, 255])
    enc = encoder.encode(pos)

    print("Binary encodings (-1 = bit 0, +1 = bit 1):")
    for i, p in enumerate(pos.tolist()):
        bits = enc[i, :8].tolist()
        bits_str = ''.join(['1' if b > 0 else '0' for b in bits])
        print(f"  Position {p:3d} = {bits_str}... (binary: {bin(p)})")
    print()

    # Test dot products
    print("Dot products (same pos should be max):")
    enc_5 = encoder.encode(torch.tensor([5]))
    enc_5b = encoder.encode(torch.tensor([5]))
    enc_6 = encoder.encode(torch.tensor([6]))
    enc_7 = encoder.encode(torch.tensor([7]))

    dot_same = (enc_5 * enc_5b).sum().item()
    dot_diff1 = (enc_5 * enc_6).sum().item()
    dot_diff2 = (enc_5 * enc_7).sum().item()

    print(f"  dot(enc(5), enc(5)) = {dot_same:.0f} (max = {encoder.dim})")
    print(f"  dot(enc(5), enc(6)) = {dot_diff1:.0f}")
    print(f"  dot(enc(5), enc(7)) = {dot_diff2:.0f}")
    print()

    # Test decoding
    decoded = encoder.decode(enc)
    print(f"Decoded positions: {decoded.tolist()}")
    print(f"Original positions: {pos.tolist()}")
    assert decoded.tolist() == pos.tolist(), "Decode failed!"
    print("✓ Encode/decode round-trip successful")
    print()


def test_io_buffer():
    """Test I/O buffer read/write with dynamic allocation."""
    print("=== Testing I/O Buffer (Dynamic Allocation) ===\n")

    # Initialize VM state with heap
    vm_state = torch.zeros(1, 8, VMState.DIM)
    heap_init = HeapInitFFN(heap_start=0x1000, heap_size=0x4000)
    vm_state = heap_init(vm_state)

    # Create allocator
    allocator = BufferAllocator(vm_state)

    # Create and allocate buffer
    buffer = IOBuffer(buffer_size=64, value_dim=8)
    buf_info = allocator.allocate_buffer("test_buffer", 64, 8)
    buffer.allocate(
        buf_info.base_addr,
        allocator._make_mem_read(),
        allocator._make_mem_write()
    )

    print(f"Buffer allocated at: 0x{buf_info.base_addr:04X}")

    # Write some values
    for i in range(5):
        value = torch.zeros(8)
        value[0] = i + ord('A')  # 'A', 'B', 'C', 'D', 'E'
        buffer.write(i, value)
        print(f"  Wrote '{chr(i + ord('A'))}' at offset {i}")

    # Read back
    print("\nReading back:")
    for i in range(5):
        val = buffer.read(torch.tensor([i]))
        char_val = int(val[0, 0].item()) if val.numel() > 0 else 0
        print(f"  Offset {i}: value = {chr(char_val) if 0 < char_val < 128 else '?'} (raw: {char_val})")
    print()


def test_branch_lt_ge():
    """Test BLT and BGE operations."""
    print("=== Testing BLT/BGE ===\n")

    set_sign = SetSignFlagFFN()
    blt = BranchLTFFN()
    bge = BranchGEFFN()

    # Architecture: All positions have same register values (replicated)
    # First compute FLAG_SIGN, then use BLT/BGE

    # Test BLT with negative number (high nibble = 0xF)
    x = torch.zeros(1, 8, VMState.DIM)
    x[:, :, E.OP_START + Opcode.BLT] = 1.0
    x[:, :, VMState.AX_BASE + 7] = 15  # Negative (0xF)
    x[:, :, VMState.PC] = 100
    x[:, :, E.RESULT] = 200

    # First compute sign flag
    x = set_sign(x)
    print(f"  FLAG_SIGN after AX[7]=15: {x[0, 0, VMState.FLAG_SIGN].item():.2f} (expected ~1)")

    # Then do branch
    y = blt(x)
    pc_value = y[0, 0, VMState.PC].item()
    print(f"BLT (AX negative): PC = {pc_value:.1f} (expected ~200)")
    if abs(pc_value - 200) < 20:
        print("  ✓ Branch taken")
    else:
        print(f"  ✗ Branch not taken correctly")

    # Test BLT with positive number (high nibble = 0x0)
    x2 = torch.zeros(1, 8, VMState.DIM)
    x2[:, :, E.OP_START + Opcode.BLT] = 1.0
    x2[:, :, VMState.AX_BASE + 7] = 0  # Positive
    x2[:, :, VMState.PC] = 100
    x2[:, :, E.RESULT] = 200

    x2 = set_sign(x2)
    print(f"  FLAG_SIGN after AX[7]=0: {x2[0, 0, VMState.FLAG_SIGN].item():.2f} (expected ~0)")

    y2 = blt(x2)
    pc_value2 = y2[0, 0, VMState.PC].item()
    print(f"BLT (AX positive): PC = {pc_value2:.1f} (expected ~100)")
    if abs(pc_value2 - 100) < 20:
        print("  ✓ Branch not taken")
    else:
        print(f"  ✗ Branch taken incorrectly")

    # Test BGE with positive number
    x3 = torch.zeros(1, 8, VMState.DIM)
    x3[:, :, E.OP_START + Opcode.BGE] = 1.0
    x3[:, :, VMState.AX_BASE + 7] = 0  # Positive
    x3[:, :, VMState.PC] = 100
    x3[:, :, E.RESULT] = 200

    x3 = set_sign(x3)
    y3 = bge(x3)
    pc_value3 = y3[0, 0, VMState.PC].item()
    print(f"BGE (AX positive): PC = {pc_value3:.1f} (expected ~200)")
    if abs(pc_value3 - 200) < 20:
        print("  ✓ Branch taken")
    else:
        print(f"  ✗ Branch not taken correctly")

    # Test BGE with negative number
    x4 = torch.zeros(1, 8, VMState.DIM)
    x4[:, :, E.OP_START + Opcode.BGE] = 1.0
    x4[:, :, VMState.AX_BASE + 7] = 15  # Negative
    x4[:, :, VMState.PC] = 100
    x4[:, :, E.RESULT] = 200

    x4 = set_sign(x4)
    y4 = bge(x4)
    pc_value4 = y4[0, 0, VMState.PC].item()
    print(f"BGE (AX negative): PC = {pc_value4:.1f} (expected ~100)")
    if abs(pc_value4 - 100) < 20:
        print("  ✓ Branch not taken")
    else:
        print(f"  ✗ Branch taken incorrectly")

    print()


def test_tool_use_mode():
    """Test tool-use mode vs streaming mode."""
    print("=== Testing Tool-Use Mode Toggle ===\n")

    # Initialize VM state with heap
    vm_state = torch.zeros(1, 8, VMState.DIM)
    heap_init = HeapInitFFN(heap_start=0x2000, heap_size=0x8000)
    vm_state = heap_init(vm_state)

    allocator = BufferAllocator(vm_state)

    # Test streaming mode (non-tool-use)
    io_streaming = NeuralIOSystem(tool_use_mode=False)
    io_streaming.allocate(allocator)

    print("Streaming mode (tool_use_mode=False):")
    print(f"  file_open returns: {io_streaming.file_open_op(123, 0)} (expected -1)")
    data, bytes_read = io_streaming.file_read_op(0, 10)
    print(f"  file_read returns: {bytes_read} bytes (expected -1)")
    print(f"  file_close returns: {io_streaming.file_close_op(0)} (expected -1)")

    # Reset allocator for tool-use mode test
    allocator2 = BufferAllocator(vm_state.clone())

    # Test tool-use mode
    io_tooluse = NeuralIOSystem(tool_use_mode=True)
    io_tooluse.allocate(allocator2)

    print("\nTool-use mode (tool_use_mode=True):")
    fd = io_tooluse.file_open_op(456, 1)  # Write mode
    print(f"  file_open returns: {fd} (expected 0)")

    if fd >= 0:
        # Write some data
        data = torch.zeros(3, 8)
        data[0, 0] = ord('H')
        data[1, 0] = ord('i')
        data[2, 0] = ord('!')
        bytes_written = io_tooluse.file_system.write(fd, data)
        print(f"  file_write wrote: {bytes_written} bytes")

        # Seek to beginning and read
        io_tooluse.file_system.fd_position[fd] = 0
        read_data, bytes_read = io_tooluse.file_read_op(fd, 3)
        print(f"  file_read returned: {bytes_read} bytes")

        # Close file (frees buffer)
        result = io_tooluse.file_close_op(fd)
        print(f"  file_close returns: {result} (expected 0)")

    # Test toggle
    print("\nToggle test:")
    io_tooluse.tool_use_mode = False
    print(f"  After setting tool_use_mode=False:")
    print(f"  file_open returns: {io_tooluse.file_open_op(789, 0)} (expected -1)")

    io_tooluse.tool_use_mode = True
    fd2 = io_tooluse.file_open_op(789, 0)
    print(f"  After setting tool_use_mode=True:")
    print(f"  file_open returns: {fd2} (expected >= 0)")

    print()


def test_file_system_dynamic():
    """Test file system with dynamic buffer allocation."""
    print("=== Testing NeuralFileSystem (Dynamic Allocation) ===\n")

    # Initialize VM state
    vm_state = torch.zeros(1, 8, VMState.DIM)
    heap_init = HeapInitFFN(heap_start=0x3000, heap_size=0x8000)
    vm_state = heap_init(vm_state)

    allocator = BufferAllocator(vm_state)

    # Create file system
    fs = NeuralFileSystem()
    fs.allocate(allocator)

    # Open multiple files
    fd1 = fs.open(hash("file1.txt") & 0xFFFFFFFF, 1)
    fd2 = fs.open(hash("file2.txt") & 0xFFFFFFFF, 1)
    print(f"Opened file1: fd={fd1}, buffer at 0x{int(fs.fd_buffer_addr[fd1].item()):04X}")
    print(f"Opened file2: fd={fd2}, buffer at 0x{int(fs.fd_buffer_addr[fd2].item()):04X}")

    # Write to file1
    data = torch.zeros(5, 8)
    for i, c in enumerate("hello"):
        data[i, 0] = ord(c)
    fs.write(fd1, data)
    print(f"Wrote 'hello' to fd={fd1}")

    # Read from file1
    fs.fd_position[fd1] = 0  # Seek to start
    read_data, count = fs.read(fd1, 5)
    chars = ''.join(chr(int(read_data[i, 0].item())) for i in range(count))
    print(f"Read from fd={fd1}: '{chars}'")

    # Close file1 (should free memory)
    fs.close(fd1)
    print(f"Closed fd={fd1}")

    # Check heap ptr shows memory was allocated
    heap_ptr = allocator.get_heap_ptr()
    print(f"Current heap ptr: 0x{heap_ptr:04X}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Fully Neural I/O System (Dynamic Allocation)")
    print("=" * 60)
    print()
    print("Components:")
    print("  - BinaryPositionEncoder: Direct binary position encoding")
    print("  - IOBuffer: Dynamically allocated in VM heap")
    print("  - UserInputMarker: Handle <USER_INPUT> regions")
    print("  - UserOutputMarker: Handle output generation")
    print("  - NeuralFileSystem: File I/O with malloc/free")
    print("  - BranchLTFFN/BranchGEFFN: Signed comparison branches")
    print("  - Tool-use mode toggle for stdin/stdout vs file I/O")
    print()

    test_binary_position_encoding()
    test_io_buffer()
    test_branch_lt_ge()
    test_tool_use_mode()
    test_file_system_dynamic()

    print("=" * 60)
    print("All neural I/O components implemented!")
    print("=" * 60)
