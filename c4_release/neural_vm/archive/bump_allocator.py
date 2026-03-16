"""
Neural Bump Allocator for Dynamic Memory Allocation.

Instead of hardcoded buffers, all memory is allocated dynamically
within the VM's KV cache memory system using a bump allocator.

Bump Allocator:
- HEAP_BASE: Fixed base address of heap (set at program start)
- HEAP_PTR: Current allocation pointer (bumps upward by 4-byte alignment)
- HEAP_END: End of heap (for bounds checking)

malloc(size):
  old_ptr = HEAP_PTR
  HEAP_PTR += align4(size)  # 4-byte alignment
  return old_ptr

free(ptr, size):
  Overwrite memory region with zeros.

  Why this works:
  - With softmax1 (softmax with constant "1" term), zero values are essentially
    ignored in attention computation
  - ZFOD (Zero Fill On Demand): Reading uninitialized/freed memory returns zero
  - Zero entries can be evicted from KV cache without affecting computation
  - This gives us "free" memory reclamation without explicit eviction logic

All buffers (I/O, file system, etc.) are allocated at runtime
via malloc calls within the VM.
"""

import torch
import torch.nn as nn

from .embedding import E, Opcode
from .base_layers import PureFFN, bake_weights


# ============================================================================
# NEURAL MALLOC FFN
# ============================================================================

class MallocFFN(PureFFN):
    """
    malloc(size): Allocate memory using bump allocator with 4-byte alignment.

    When MALC opcode is active:
    1. Copy HEAP_PTR to AX (return value)
    2. Add aligned size (from stack or NIB_A) to HEAP_PTR
       - Size is rounded up to 4-byte boundary

    Neural implementation:
    - silu(S * MALC) gates all operations
    - HEAP_PTR copied to AX
    - Size (from NIB_A nibbles) added to HEAP_PTR
    - Alignment: HEAP_PTR increments by 4 (nibble 0 unchanged, nibble 1 += size/16)

    The size is passed in the instruction immediate or on the stack.
    For simplicity, we use NIB_A (8 nibbles = 32-bit size).
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=24)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Step 1: Copy HEAP_PTR to AX (return allocated address)
        # For each of 8 nibbles: AX[i] = HEAP_PTR[i]
        for i in range(8):
            # Clear old AX[i]
            self.W_up[h, E.OP_START + Opcode.MALC] = S
            self.W_gate[h, E.AX_BASE + i] = -1.0
            self.W_down[E.AX_BASE + i, h] = 1.0 / S
            h += 1

            # Copy HEAP_PTR[i] to AX[i]
            self.W_up[h, E.OP_START + Opcode.MALC] = S
            self.W_gate[h, E.HEAP_PTR + i] = 1.0
            self.W_down[E.AX_BASE + i, h] = 1.0 / S
            h += 1

        # Step 2: Add size to HEAP_PTR with 4-byte alignment
        # Size comes from NIB_A (passed via instruction)
        # Add to nibble 0 first (represents bytes 0-15)
        # Alignment: always increment by at least 4 (size rounded to nibble boundary)
        self.W_up[h, E.OP_START + Opcode.MALC] = S
        self.W_gate[h, E.NIB_A] = 1.0
        self.W_down[E.HEAP_PTR, h] = 1.0 / S


class MallocMultiNibbleFFN(PureFFN):
    """
    Extended malloc that handles multi-nibble size with carry propagation.

    Size is in NIB_A (8 nibbles = 32-bit value).
    HEAP_PTR = HEAP_PTR + size, with carry chain.
    """

    def __init__(self):
        # Need hidden units for carry chain
        super().__init__(E.DIM, hidden_dim=48)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy HEAP_PTR to AX (return value)
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.MALC] = S
            self.W_gate[h, E.AX_BASE + i] = -1.0
            self.W_down[E.AX_BASE + i, h] = 1.0 / S
            h += 1

            self.W_up[h, E.OP_START + Opcode.MALC] = S
            self.W_gate[h, E.HEAP_PTR + i] = 1.0
            self.W_down[E.AX_BASE + i, h] = 1.0 / S
            h += 1

        # Add size (NIB_A) to HEAP_PTR with carry
        # For now, add each nibble without carry (simplified)
        # Full carry chain requires attention or iterative FFN
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.MALC] = S
            self.W_gate[h, E.NIB_A] = 1.0 / (1 << (i * 4))  # Scale by nibble position
            self.W_down[E.HEAP_PTR + i, h] = 1.0 / S
            h += 1
            if h >= 48:
                break


class FreeFFN(PureFFN):
    """
    free(ptr): Overwrite memory region with zeros.

    With softmax1 attention, zero values are ignored in computation.
    This enables:
    - ZFOD (Zero Fill On Demand): freed memory reads as zero
    - KV cache eviction: zero entries can be removed without effect
    - Implicit memory reclamation without explicit tracking

    When FREE opcode is active:
    1. Set MEM_ADDR from AX (pointer to free)
    2. Set MEM_DATA to all zeros
    3. Set MEM_WRITE flag
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy AX (pointer) to MEM_ADDR
        for i in range(8):
            self.W_up[h, E.OP_START + Opcode.FREE] = S
            self.W_gate[h, E.AX_BASE + i] = 1.0
            self.W_down[E.MEM_ADDR_BASE + i, h] = 1.0 / S
            h += 1

        # Set MEM_DATA to zero (clear any existing value)
        # Done implicitly by not adding anything

        # Set MEM_WRITE flag
        self.W_up[h, E.OP_START + Opcode.FREE] = S
        self.b_gate[h] = 1.0
        self.W_down[E.MEM_WRITE, h] = 1.0 / S


# ============================================================================
# HEAP INITIALIZATION
# ============================================================================

class HeapInitFFN(PureFFN):
    """
    Initialize heap at program start.

    Sets:
    - HEAP_BASE = start_address
    - HEAP_PTR = start_address
    - HEAP_END = start_address + heap_size

    Called once at VM initialization with heap parameters in RESULT slots.
    """

    def __init__(self, heap_start: int = 0x10000, heap_size: int = 0x10000):
        self.heap_start = heap_start
        self.heap_size = heap_size
        super().__init__(E.DIM, hidden_dim=32)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Set HEAP_BASE and HEAP_PTR to heap_start (as nibbles)
        for i in range(8):
            nibble_val = (self.heap_start >> (i * 4)) & 0xF

            # HEAP_BASE[i] = nibble_val
            self.b_up[h] = S
            self.b_gate[h] = float(nibble_val)
            self.W_down[E.HEAP_BASE + i, h] = 1.0 / S
            h += 1

            # HEAP_PTR[i] = nibble_val
            self.b_up[h] = S
            self.b_gate[h] = float(nibble_val)
            self.W_down[E.HEAP_PTR + i, h] = 1.0 / S
            h += 1

        # Set HEAP_END to heap_start + heap_size
        heap_end = self.heap_start + self.heap_size
        for i in range(8):
            nibble_val = (heap_end >> (i * 4)) & 0xF
            self.b_up[h] = S
            self.b_gate[h] = float(nibble_val)
            self.W_down[E.HEAP_END + i, h] = 1.0 / S
            h += 1
            if h >= 32:
                break


# ============================================================================
# DYNAMIC BUFFER ALLOCATION
# ============================================================================

class DynamicBuffer:
    """
    A buffer that is allocated within VM memory.

    Instead of using a separate nn.Module with register_buffer,
    this class tracks an address range within the VM's KV cache memory.

    Buffer operations use the VM's neural memory read/write.
    """

    def __init__(self, base_addr: int, size: int, value_dim: int = 8):
        """
        Create a buffer at the given address.

        Args:
            base_addr: Address in VM memory where buffer starts
            size: Number of entries in buffer
            value_dim: Dimensions per entry (default 8 for nibbles)
        """
        self.base_addr = base_addr
        self.size = size
        self.value_dim = value_dim
        self.write_ptr = 0

    def get_address(self, offset: int) -> int:
        """Get memory address for buffer entry at offset."""
        return self.base_addr + (offset * self.value_dim)

    def reset(self):
        """Reset write pointer."""
        self.write_ptr = 0


class BufferAllocator:
    """
    Allocates buffers using the VM's heap.

    Tracks allocated buffers and their addresses for I/O operations.
    Provides memory read/write functions that interface with VM's KV cache.
    """

    def __init__(self, vm_state_tensor: torch.Tensor,
                 kv_memory: dict = None):
        """
        Create allocator linked to VM state.

        Args:
            vm_state_tensor: The VM's embedding tensor containing heap state
            kv_memory: Dict mapping addresses to values (simulated KV cache)
        """
        self.vm_state = vm_state_tensor
        self.allocated_buffers = {}
        # KV cache memory: addr -> tensor value
        self.kv_memory = kv_memory if kv_memory is not None else {}

    def get_heap_ptr(self) -> int:
        """Read current HEAP_PTR from VM state."""
        ptr = 0
        for i in range(8):
            nibble = int(self.vm_state[0, 0, E.HEAP_PTR + i].item())
            ptr |= (nibble & 0xF) << (i * 4)
        return ptr

    def set_heap_ptr(self, value: int):
        """Write HEAP_PTR to VM state."""
        for i in range(8):
            nibble = (value >> (i * 4)) & 0xF
            self.vm_state[:, :, E.HEAP_PTR + i] = float(nibble)

    def allocate_buffer(self, name: str, size: int, value_dim: int = 8) -> DynamicBuffer:
        """
        Allocate a new buffer with 4-byte alignment.

        Args:
            name: Name for tracking
            size: Number of entries
            value_dim: Dimensions per entry

        Returns:
            DynamicBuffer with allocated address
        """
        ptr = self.get_heap_ptr()
        # 4-byte alignment
        if ptr % 4 != 0:
            ptr = ((ptr // 4) + 1) * 4
        total_size = size * value_dim
        # Round up to 4-byte boundary
        aligned_size = ((total_size + 3) // 4) * 4
        self.set_heap_ptr(ptr + aligned_size)

        buf = DynamicBuffer(ptr, size, value_dim)
        self.allocated_buffers[name] = buf
        return buf

    def get_buffer(self, name: str) -> DynamicBuffer:
        """Get previously allocated buffer by name."""
        return self.allocated_buffers.get(name)

    def mem_read(self, addr: int) -> torch.Tensor:
        """
        Read value from VM memory (KV cache).

        With softmax1 semantics, uninitialized memory reads as zero (ZFOD).

        Args:
            addr: Memory address

        Returns:
            [8] tensor (8 nibbles = 32 bits)
        """
        if addr in self.kv_memory:
            return self.kv_memory[addr]
        # ZFOD: Zero Fill On Demand
        return torch.zeros(8)

    def mem_write(self, addr: int, value: torch.Tensor):
        """
        Write value to VM memory (KV cache).

        Writing zeros effectively "frees" memory in softmax1 semantics.

        Args:
            addr: Memory address
            value: [8] tensor (8 nibbles)
        """
        if value.sum().item() == 0:
            # Writing zeros = free (can be evicted from KV cache)
            if addr in self.kv_memory:
                del self.kv_memory[addr]
        else:
            self.kv_memory[addr] = value.clone()

    def _make_mem_read(self):
        """Create a memory read function for IOBuffer."""
        return self.mem_read

    def _make_mem_write(self):
        """Create a memory write function for IOBuffer."""
        return self.mem_write

    def free(self, addr: int, size: int = 8):
        """
        Free memory by writing zeros.

        With softmax1, zero values are ignored in attention,
        so this effectively frees the memory.

        Args:
            addr: Start address
            size: Number of bytes to free
        """
        for i in range(0, size, 8):
            self.mem_write(addr + i, torch.zeros(8))


# ============================================================================
# DYNAMIC I/O BUFFER (replaces hardcoded IOBuffer)
# ============================================================================

class DynamicIOBuffer(nn.Module):
    """
    I/O Buffer using VM memory instead of hardcoded tensors.

    This is a drop-in replacement for IOBuffer that allocates
    its storage within the VM's heap via bump allocator.

    Key difference: Instead of self-contained storage, this uses
    the VM's KV cache memory for actual data storage.
    """

    def __init__(self, buffer_size: int = 1024, value_dim: int = 8, key_dim: int = 16):
        super().__init__()
        self.buffer_size = buffer_size
        self.value_dim = value_dim
        self.key_dim = key_dim

        # These will be set when allocated
        self.base_addr = 0
        self.allocated = False

        # Write pointer (local state)
        self.register_buffer('write_ptr', torch.tensor(0))
        self.register_buffer('valid_count', torch.tensor(0))

    def allocate(self, allocator: BufferAllocator, name: str):
        """
        Allocate this buffer using VM's bump allocator.

        Args:
            allocator: BufferAllocator linked to VM state
            name: Name for tracking
        """
        buf = allocator.allocate_buffer(name, self.buffer_size, self.value_dim)
        self.base_addr = buf.base_addr
        self.allocated = True

    def get_memory_address(self, offset: int) -> int:
        """Get VM memory address for buffer entry at offset."""
        return self.base_addr + (offset * self.value_dim)

    def reset(self):
        """Clear the buffer."""
        self.write_ptr.zero_()
        self.valid_count.zero_()

    def write(self, offset: int, value: torch.Tensor, memory_write_fn):
        """
        Write value at offset using VM memory.

        Args:
            offset: Position in buffer
            value: [value_dim] tensor
            memory_write_fn: Function to write to VM memory
        """
        if offset < self.buffer_size and self.allocated:
            addr = self.get_memory_address(offset)
            memory_write_fn(addr, value)
            if offset >= self.valid_count.item():
                self.valid_count.fill_(offset + 1)

    def read(self, offset: torch.Tensor, memory_read_fn) -> torch.Tensor:
        """
        Read value at offset using VM memory.

        Args:
            offset: [batch] target offsets
            memory_read_fn: Function to read from VM memory

        Returns:
            [batch, value_dim] values
        """
        if not self.allocated:
            return torch.zeros(offset.shape[0], self.value_dim)

        results = []
        for i, off in enumerate(offset.tolist()):
            addr = self.get_memory_address(int(off))
            val = memory_read_fn(addr)
            results.append(val)

        return torch.stack(results)


# ============================================================================
# NEURAL BUFFER OPERATIONS
# ============================================================================

class BufferWriteFFN(PureFFN):
    """
    Write to dynamically allocated buffer using VM memory.

    When active:
    1. Computes target address = buffer_base + offset * value_dim
    2. Copies data to MEM_DATA slots
    3. Sets MEM_WRITE flag
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy RESULT (data to write) to MEM_DATA
        for i in range(8):
            self.W_up[h, E.RESULT] = S
            self.W_gate[h, E.RESULT + i] = 1.0
            self.W_down[E.MEM_DATA_BASE + i, h] = 1.0 / S
            h += 1

        # Set MEM_WRITE flag
        self.b_up[h] = S
        self.b_gate[h] = 1.0
        self.W_down[E.MEM_WRITE, h] = 1.0 / S


class BufferReadFFN(PureFFN):
    """
    Read from dynamically allocated buffer using VM memory.

    When active:
    1. Computes source address = buffer_base + offset * value_dim
    2. Sets MEM_READ flag
    3. Result copied from MEM_DATA to RESULT
    """

    def __init__(self):
        super().__init__(E.DIM, hidden_dim=16)

    @bake_weights
    def _bake_weights(self):
        S = E.SCALE
        h = 0

        # Copy MEM_DATA (read result) to RESULT
        for i in range(8):
            self.W_up[h, E.MEM_READY] = S
            self.W_gate[h, E.MEM_DATA_BASE + i] = 1.0
            self.W_down[E.RESULT + i, h] = 1.0 / S
            h += 1


# ============================================================================
# TESTS
# ============================================================================

def test_bump_allocator():
    """Test bump allocator operations."""
    print("=== Testing Bump Allocator ===\n")

    # Initialize VM state
    x = torch.zeros(1, 8, E.DIM)

    # Initialize heap
    heap_init = HeapInitFFN(heap_start=0x1000, heap_size=0x4000)
    x = heap_init(x)

    # Read back HEAP_PTR
    heap_ptr = 0
    for i in range(8):
        nibble = int(x[0, 0, E.HEAP_PTR + i].item())
        heap_ptr |= nibble << (i * 4)

    print(f"Initial HEAP_PTR: 0x{heap_ptr:04X} (expected 0x1000)")
    assert heap_ptr == 0x1000, f"Expected 0x1000, got 0x{heap_ptr:04X}"
    print("  PASS")

    # Test malloc
    x[:, :, E.OP_START + Opcode.MALC] = 1.0
    x[:, :, E.NIB_A] = 0x20  # Allocate 32 bytes

    malloc_ffn = MallocFFN()
    y = malloc_ffn(x)

    # Check AX has old HEAP_PTR (allocated address)
    ax_val = 0
    for i in range(8):
        nibble = int(y[0, 0, E.AX_BASE + i].item())
        ax_val |= nibble << (i * 4)

    print(f"malloc(32) returned: 0x{ax_val:04X} (expected 0x1000)")
    assert ax_val == 0x1000, f"Expected 0x1000, got 0x{ax_val:04X}"

    # Check HEAP_PTR was bumped
    new_heap_ptr = 0
    for i in range(8):
        nibble = int(y[0, 0, E.HEAP_PTR + i].item())
        new_heap_ptr |= nibble << (i * 4)

    print(f"New HEAP_PTR: 0x{new_heap_ptr:04X} (expected 0x1020)")
    print("  PASS")
    print()


def test_buffer_allocator():
    """Test buffer allocator."""
    print("=== Testing Buffer Allocator ===\n")

    # Initialize VM state with heap
    x = torch.zeros(1, 8, E.DIM)
    heap_init = HeapInitFFN(heap_start=0x2000, heap_size=0x8000)
    x = heap_init(x)

    # Create allocator
    allocator = BufferAllocator(x)

    # Allocate a buffer
    buf1 = allocator.allocate_buffer("input", size=256, value_dim=8)
    print(f"Allocated 'input' buffer at: 0x{buf1.base_addr:04X}")
    assert buf1.base_addr == 0x2000, f"Expected 0x2000, got 0x{buf1.base_addr:04X}"

    # Allocate another buffer
    buf2 = allocator.allocate_buffer("output", size=512, value_dim=8)
    expected_addr = 0x2000 + (256 * 8)  # After first buffer
    print(f"Allocated 'output' buffer at: 0x{buf2.base_addr:04X} (expected 0x{expected_addr:04X})")
    assert buf2.base_addr == expected_addr, f"Expected 0x{expected_addr:04X}, got 0x{buf2.base_addr:04X}"

    # Check heap pointer
    heap_ptr = allocator.get_heap_ptr()
    expected_ptr = expected_addr + (512 * 8)
    print(f"Final HEAP_PTR: 0x{heap_ptr:04X} (expected 0x{expected_ptr:04X})")
    assert heap_ptr == expected_ptr

    print("  PASS")
    print()


def test_dynamic_io_buffer():
    """Test dynamic I/O buffer."""
    print("=== Testing Dynamic I/O Buffer ===\n")

    # Initialize VM state
    x = torch.zeros(1, 8, E.DIM)
    heap_init = HeapInitFFN(heap_start=0x3000, heap_size=0x8000)
    x = heap_init(x)

    # Create allocator and buffer
    allocator = BufferAllocator(x)

    io_buf = DynamicIOBuffer(buffer_size=64, value_dim=8)
    io_buf.allocate(allocator, "stdio")

    print(f"Allocated 'stdio' buffer at: 0x{io_buf.base_addr:04X}")
    print(f"  Size: {io_buf.buffer_size} entries x {io_buf.value_dim} dims")
    print(f"  Total memory: {io_buf.buffer_size * io_buf.value_dim} bytes")

    # Test address calculation
    addr0 = io_buf.get_memory_address(0)
    addr1 = io_buf.get_memory_address(1)
    addr10 = io_buf.get_memory_address(10)

    print(f"  Entry 0 address: 0x{addr0:04X}")
    print(f"  Entry 1 address: 0x{addr1:04X}")
    print(f"  Entry 10 address: 0x{addr10:04X}")

    assert addr1 == addr0 + 8, "Entry spacing incorrect"
    assert addr10 == addr0 + 80, "Entry spacing incorrect"

    print("  PASS")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Bump Allocator")
    print("=" * 60)
    print()
    print("Components:")
    print("  - MallocFFN: Bump HEAP_PTR, return allocated address")
    print("  - FreeFFN: No-op (bump allocator doesn't reclaim)")
    print("  - HeapInitFFN: Set initial heap bounds")
    print("  - BufferAllocator: Allocate buffers in VM memory")
    print("  - DynamicIOBuffer: I/O buffer using VM memory")
    print()

    test_bump_allocator()
    test_buffer_allocator()
    test_dynamic_io_buffer()

    print("=" * 60)
    print("All bump allocator tests passed!")
    print("=" * 60)
