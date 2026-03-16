"""
Memory Subroutines for Neural VM.

Implements memset, memcmp, memcpy using the neural ALU primitives.
These are loop-based operations that use the existing load/store ops.

MSET (memset): Fill memory region with value
MCMP (memcmp): Compare two memory regions
MCPY (memcpy): Copy memory region (not in C4, added as extension)

Note: These are "subroutines" that execute multiple ALU operations
rather than single FFN passes. They use the neural ALU for the
underlying ADD, SUB, and comparison operations.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .embedding import E, Opcode


class MemorySubroutines:
    """
    Memory subroutine implementations using neural ALU.

    These operate on a memory array (externally provided) and use
    the neural ALU for address arithmetic and comparisons.
    """

    def __init__(self, alu):
        """
        Args:
            alu: The neural ALU instance (SparseMoEALU or MultiNibbleALU)
        """
        self.alu = alu

    def _encode_op(self, opcode: int, a: int, b: int) -> torch.Tensor:
        """Encode operands for ALU operation."""
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
            x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
            x[0, i, E.OP_START + opcode] = 1.0
            x[0, i, E.POS] = float(i)
        return x

    def _decode_result(self, x: torch.Tensor) -> int:
        """Decode result from ALU output."""
        result = 0
        for i in range(E.NUM_POSITIONS):
            nib = int(round(x[0, i, E.RESULT].item()))
            nib = max(0, min(15, nib))
            result |= (nib << (i * 4))
        return result

    def _alu_add(self, a: int, b: int) -> int:
        """32-bit addition using ALU."""
        x = self._encode_op(Opcode.ADD, a, b)
        y = self.alu(x)
        return self._decode_result(y) & 0xFFFFFFFF

    def _alu_sub(self, a: int, b: int) -> int:
        """32-bit subtraction using ALU."""
        x = self._encode_op(Opcode.SUB, a, b)
        y = self.alu(x)
        return self._decode_result(y) & 0xFFFFFFFF

    def _alu_lt(self, a: int, b: int) -> int:
        """32-bit less-than comparison using ALU."""
        x = self._encode_op(Opcode.LT, a, b)
        y = self.alu(x)
        return self._decode_result(y)

    def memset(self, memory: bytearray, ptr: int, val: int, size: int) -> int:
        """
        memset(ptr, val, size) - Fill memory region with value.

        Args:
            memory: Memory array to modify
            ptr: Start address
            val: Value to fill (0-255)
            size: Number of bytes to fill

        Returns:
            ptr (for chaining)
        """
        val = val & 0xFF
        for i in range(size):
            addr = self._alu_add(ptr, i)
            if 0 <= addr < len(memory):
                memory[addr] = val
        return ptr

    def memcmp(self, memory: bytearray, a_ptr: int, b_ptr: int, size: int) -> int:
        """
        memcmp(a, b, size) - Compare two memory regions.

        Args:
            memory: Memory array to read from
            a_ptr: First region address
            b_ptr: Second region address
            size: Number of bytes to compare

        Returns:
            0 if equal, <0 if a<b, >0 if a>b
        """
        for i in range(size):
            addr_a = self._alu_add(a_ptr, i)
            addr_b = self._alu_add(b_ptr, i)

            val_a = memory[addr_a] if 0 <= addr_a < len(memory) else 0
            val_b = memory[addr_b] if 0 <= addr_b < len(memory) else 0

            # Use ALU subtraction for comparison
            diff = self._alu_sub(val_a, val_b)

            # Check if difference is non-zero (treating as signed for proper comparison)
            if diff != 0:
                # Convert to signed interpretation
                if diff > 0x7FFFFFFF:  # Negative in 2's complement
                    return -1
                else:
                    return 1

        return 0

    def memcpy(self, memory: bytearray, dest: int, src: int, size: int) -> int:
        """
        memcpy(dest, src, size) - Copy memory region.

        Args:
            memory: Memory array to read/write
            dest: Destination address
            src: Source address
            size: Number of bytes to copy

        Returns:
            dest (for chaining)

        Note: Handles overlapping regions correctly (copies backward if needed).
        """
        # Check for overlap and direction
        if dest > src and dest < self._alu_add(src, size):
            # Overlapping, copy backward
            for i in range(size - 1, -1, -1):
                src_addr = self._alu_add(src, i)
                dest_addr = self._alu_add(dest, i)
                if 0 <= src_addr < len(memory) and 0 <= dest_addr < len(memory):
                    memory[dest_addr] = memory[src_addr]
        else:
            # Non-overlapping or dest before src, copy forward
            for i in range(size):
                src_addr = self._alu_add(src, i)
                dest_addr = self._alu_add(dest, i)
                if 0 <= src_addr < len(memory) and 0 <= dest_addr < len(memory):
                    memory[dest_addr] = memory[src_addr]

        return dest


class MsetFFN(nn.Module):
    """
    Neural FFN layer that sets up MSET operation parameters.

    Extracts ptr, val, size from stack (NIB_A positions) and stores
    them in TEMP slots for the subroutine loop.

    Note: Actual loop execution happens in the subroutine wrapper.
    """

    def __init__(self):
        super().__init__()
        # MSET parameter extraction handled by external subroutine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSET is a subroutine - actual work done by MemorySubroutines
        return x


class McmpFFN(nn.Module):
    """
    Neural FFN layer that sets up MCMP operation parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MCMP is a subroutine
        return x


class McpyFFN(nn.Module):
    """
    Neural FFN layer that sets up MCPY operation parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MCPY is a subroutine
        return x


class VMWithMemorySubroutines:
    """
    VM wrapper that handles memory subroutines.

    Intercepts MSET, MCMP, MCPY opcodes and executes the
    corresponding subroutine using the neural ALU.
    """

    def __init__(self, alu, memory_size: int = 65536):
        """
        Args:
            alu: The neural ALU instance
            memory_size: Size of memory in bytes (default 64KB)
        """
        self.alu = alu
        self.memory = bytearray(memory_size)
        self.subroutines = MemorySubroutines(alu)

    def execute_mset(self, ptr: int, val: int, size: int) -> int:
        """Execute MSET (memset) subroutine."""
        return self.subroutines.memset(self.memory, ptr, val, size)

    def execute_mcmp(self, a_ptr: int, b_ptr: int, size: int) -> int:
        """Execute MCMP (memcmp) subroutine."""
        return self.subroutines.memcmp(self.memory, a_ptr, b_ptr, size)

    def execute_mcpy(self, dest: int, src: int, size: int) -> int:
        """Execute MCPY (memcpy) subroutine."""
        return self.subroutines.memcpy(self.memory, dest, src, size)

    def read_byte(self, addr: int) -> int:
        """Read byte from memory."""
        if 0 <= addr < len(self.memory):
            return self.memory[addr]
        return 0

    def write_byte(self, addr: int, val: int):
        """Write byte to memory."""
        if 0 <= addr < len(self.memory):
            self.memory[addr] = val & 0xFF

    def read_int(self, addr: int) -> int:
        """Read 32-bit int from memory (little-endian)."""
        result = 0
        for i in range(4):
            if 0 <= addr + i < len(self.memory):
                result |= self.memory[addr + i] << (i * 8)
        return result

    def write_int(self, addr: int, val: int):
        """Write 32-bit int to memory (little-endian)."""
        for i in range(4):
            if 0 <= addr + i < len(self.memory):
                self.memory[addr + i] = (val >> (i * 8)) & 0xFF


def demo_memory_subroutines():
    """Demonstrate memory subroutines."""
    print("=" * 60)
    print("Memory Subroutines Demo")
    print("=" * 60)
    print()

    # Import ALU
    try:
        from .sparse_moe_alu import SparseMoEALU
        alu = SparseMoEALU()
        print("Using SparseMoEALU")
    except ImportError:
        # Fallback to simple implementation
        print("SparseMoEALU not available, using mock ALU")

        class MockALU:
            def __call__(self, x):
                # Simple mock that returns ADD result
                result = 0
                for i in range(8):
                    a = int(x[0, i, 0].item())
                    b = int(x[0, i, 1].item())
                    result |= ((a + b) & 0xF) << (i * 4)
                x[0, 0, 5] = result & 0xF
                for i in range(8):
                    x[0, i, 5] = float((result >> (i * 4)) & 0xF)
                return x

        alu = MockALU()

    vm = VMWithMemorySubroutines(alu)

    # Demo MSET
    print("1. Testing memset:")
    print(f"   memset(addr=0x100, val=0xAB, size=8)")
    vm.execute_mset(0x100, 0xAB, 8)
    values = [vm.read_byte(0x100 + i) for i in range(8)]
    print(f"   Result: {[hex(v) for v in values]}")
    print()

    # Demo MCPY
    print("2. Testing memcpy:")
    # First set up source data
    for i in range(4):
        vm.write_byte(0x200 + i, 0x10 + i)
    print(f"   Source at 0x200: {[hex(vm.read_byte(0x200 + i)) for i in range(4)]}")
    vm.execute_mcpy(0x300, 0x200, 4)
    print(f"   memcpy(dest=0x300, src=0x200, size=4)")
    print(f"   Result at 0x300: {[hex(vm.read_byte(0x300 + i)) for i in range(4)]}")
    print()

    # Demo MCMP
    print("3. Testing memcmp:")
    # Equal regions
    vm.execute_mset(0x400, 0x55, 4)
    vm.execute_mset(0x410, 0x55, 4)
    result = vm.execute_mcmp(0x400, 0x410, 4)
    print(f"   memcmp(0x400, 0x410, 4) = {result} (expected 0 for equal)")

    # Different regions
    vm.write_byte(0x400, 0x60)  # Make first byte larger
    result = vm.execute_mcmp(0x400, 0x410, 4)
    print(f"   memcmp(0x400, 0x410, 4) = {result} (expected >0 for a>b)")

    vm.write_byte(0x400, 0x40)  # Make first byte smaller
    result = vm.execute_mcmp(0x400, 0x410, 4)
    print(f"   memcmp(0x400, 0x410, 4) = {result} (expected <0 for a<b)")
    print()

    print("=" * 60)


if __name__ == "__main__":
    demo_memory_subroutines()
