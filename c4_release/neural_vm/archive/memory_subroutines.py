"""
Memory Subroutines for Neural VM.

MSET (memset): Fill memory region with a value
MCMP (memcmp): Compare two memory regions

These are implemented as subroutines that use existing memory operations
(LI, SI, LC, SC) in a loop pattern. They're not pure neural layers -
they're executed by the VM controller using the neural ALU and memory.

The subroutines work with the KV-cache based memory system.
"""

import torch
from typing import Callable, Optional

from .embedding import E, Opcode


class MemorySubroutine:
    """
    Base class for memory subroutines.

    Subroutines are executed by the VM controller, not as neural layers.
    They use the existing neural ALU and KV memory operations.
    """

    def __init__(self, memory_read: Callable[[int], int], memory_write: Callable[[int, int], None]):
        """
        Args:
            memory_read: Function to read from memory (addr -> value)
            memory_write: Function to write to memory (addr, value -> None)
        """
        self.mem_read = memory_read
        self.mem_write = memory_write


class MsetSubroutine(MemorySubroutine):
    """
    memset(ptr, val, size) - Fill memory region with a value.

    C signature: void *memset(void *ptr, int value, size_t size);

    Implementation:
    1. Read args from stack: ptr, val, size
    2. Loop size times: write val at ptr+i
    3. Return ptr in AX
    """

    def execute(self, ptr: int, val: int, size: int) -> int:
        """
        Execute memset.

        Args:
            ptr: Start address
            val: Value to fill (byte)
            size: Number of bytes to fill

        Returns:
            ptr (for chaining)
        """
        val = val & 0xFF  # Ensure byte
        for i in range(size):
            self.mem_write(ptr + i, val)
        return ptr

    def execute_from_stack(self, stack_read: Callable[[int], int], sp: int) -> tuple[int, int]:
        """
        Execute memset with args from stack.

        Stack layout (before call):
            [sp+0]: ptr
            [sp+8]: val
            [sp+16]: size

        Returns:
            (result, new_sp)
        """
        ptr = stack_read(sp)
        val = stack_read(sp + 8)
        size = stack_read(sp + 16)

        result = self.execute(ptr, val, size)
        return result, sp + 24  # Pop 3 args


class McmpSubroutine(MemorySubroutine):
    """
    memcmp(a, b, size) - Compare two memory regions.

    C signature: int memcmp(const void *a, const void *b, size_t size);

    Implementation:
    1. Read args from stack: a, b, size
    2. Loop size times: compare a[i] vs b[i]
    3. Return first difference (a[i] - b[i]) or 0 if equal
    """

    def execute(self, a_ptr: int, b_ptr: int, size: int) -> int:
        """
        Execute memcmp.

        Args:
            a_ptr: First memory region
            b_ptr: Second memory region
            size: Number of bytes to compare

        Returns:
            0 if equal, negative if a < b, positive if a > b
        """
        for i in range(size):
            a_byte = self.mem_read(a_ptr + i) & 0xFF
            b_byte = self.mem_read(b_ptr + i) & 0xFF
            if a_byte != b_byte:
                return a_byte - b_byte
        return 0

    def execute_from_stack(self, stack_read: Callable[[int], int], sp: int) -> tuple[int, int]:
        """
        Execute memcmp with args from stack.

        Stack layout (before call):
            [sp+0]: a
            [sp+8]: b
            [sp+16]: size

        Returns:
            (result, new_sp)
        """
        a_ptr = stack_read(sp)
        b_ptr = stack_read(sp + 8)
        size = stack_read(sp + 16)

        result = self.execute(a_ptr, b_ptr, size)
        return result, sp + 24  # Pop 3 args


class McpySubroutine(MemorySubroutine):
    """
    memcpy(dest, src, size) - Copy memory region.

    C signature: void *memcpy(void *dest, const void *src, size_t size);

    Implementation:
    1. Read args from stack: dest, src, size
    2. Loop size times: copy src[i] to dest[i]
    3. Return dest in AX
    """

    def execute(self, dest: int, src: int, size: int) -> int:
        """
        Execute memcpy.

        Args:
            dest: Destination address
            src: Source address
            size: Number of bytes to copy

        Returns:
            dest (for chaining)
        """
        for i in range(size):
            val = self.mem_read(src + i)
            self.mem_write(dest + i, val)
        return dest

    def execute_from_stack(self, stack_read: Callable[[int], int], sp: int) -> tuple[int, int]:
        """
        Execute memcpy with args from stack.

        Stack layout (before call):
            [sp+0]: dest
            [sp+8]: src
            [sp+16]: size

        Returns:
            (result, new_sp)
        """
        dest = stack_read(sp)
        src = stack_read(sp + 8)
        size = stack_read(sp + 16)

        result = self.execute(dest, src, size)
        return result, sp + 24  # Pop 3 args


class StrlenSubroutine(MemorySubroutine):
    """
    strlen(s) - Get string length.

    C signature: size_t strlen(const char *s);
    """

    def execute(self, s_ptr: int) -> int:
        """
        Execute strlen.

        Args:
            s_ptr: String pointer

        Returns:
            Length (not including null terminator)
        """
        length = 0
        while self.mem_read(s_ptr + length) != 0:
            length += 1
            if length > 1000000:  # Safety limit
                break
        return length


class StrcmpSubroutine(MemorySubroutine):
    """
    strcmp(a, b) - Compare two strings.

    C signature: int strcmp(const char *a, const char *b);
    """

    def execute(self, a_ptr: int, b_ptr: int) -> int:
        """
        Execute strcmp.

        Args:
            a_ptr: First string
            b_ptr: Second string

        Returns:
            0 if equal, negative if a < b, positive if a > b
        """
        i = 0
        while True:
            a_char = self.mem_read(a_ptr + i) & 0xFF
            b_char = self.mem_read(b_ptr + i) & 0xFF

            if a_char != b_char:
                return a_char - b_char
            if a_char == 0:  # Both reached null
                return 0

            i += 1
            if i > 1000000:  # Safety limit
                break
        return 0


class MemorySubroutineHandler:
    """
    Handler for memory subroutines in the VM.

    Integrates with the VM controller to execute subroutines
    when their opcodes are encountered.
    """

    def __init__(self, memory_read: Callable[[int], int], memory_write: Callable[[int, int], None]):
        self.mset = MsetSubroutine(memory_read, memory_write)
        self.mcmp = McmpSubroutine(memory_read, memory_write)
        self.mcpy = McpySubroutine(memory_read, memory_write)
        self.strlen = StrlenSubroutine(memory_read, memory_write)
        self.strcmp = StrcmpSubroutine(memory_read, memory_write)

    def handle_opcode(self, opcode: int, stack_read: Callable[[int], int], sp: int) -> Optional[tuple[int, int]]:
        """
        Handle a subroutine opcode.

        Args:
            opcode: The opcode number
            stack_read: Function to read from stack
            sp: Current stack pointer

        Returns:
            (result, new_sp) if handled, None if not a subroutine opcode
        """
        if opcode == Opcode.MSET:
            return self.mset.execute_from_stack(stack_read, sp)
        elif opcode == Opcode.MCMP:
            return self.mcmp.execute_from_stack(stack_read, sp)
        # MCPY would need a new opcode
        return None


def test_memory_subroutines():
    """Test memory subroutines."""
    # Simple in-memory storage for testing
    memory = {}

    def mem_read(addr: int) -> int:
        return memory.get(addr, 0)

    def mem_write(addr: int, val: int):
        memory[addr] = val & 0xFF

    print("=== Memory Subroutine Tests ===\n")

    # Test MSET
    print("MSET test:")
    mset = MsetSubroutine(mem_read, mem_write)
    mset.execute(0x1000, 0xAB, 10)
    result = [memory.get(0x1000 + i, 0) for i in range(10)]
    print(f"  memset(0x1000, 0xAB, 10) = {[hex(x) for x in result]}")
    assert all(x == 0xAB for x in result), "MSET failed"
    print("  ✓ PASS")

    # Test MCMP (equal)
    print("\nMCMP test (equal):")
    for i in range(5):
        mem_write(0x2000 + i, 0x11)
        mem_write(0x3000 + i, 0x11)
    mcmp = McmpSubroutine(mem_read, mem_write)
    result = mcmp.execute(0x2000, 0x3000, 5)
    print(f"  memcmp([0x11]*5, [0x11]*5, 5) = {result}")
    assert result == 0, "MCMP equal failed"
    print("  ✓ PASS")

    # Test MCMP (not equal)
    print("\nMCMP test (not equal):")
    mem_write(0x3002, 0x22)  # Make one byte different
    result = mcmp.execute(0x2000, 0x3000, 5)
    print(f"  memcmp with difference at byte 2: {result}")
    assert result == 0x11 - 0x22, "MCMP not equal failed"
    print("  ✓ PASS")

    # Test MCPY
    print("\nMCPY test:")
    for i in range(5):
        mem_write(0x4000 + i, i + 1)
    mcpy = McpySubroutine(mem_read, mem_write)
    mcpy.execute(0x5000, 0x4000, 5)
    result = [memory.get(0x5000 + i, 0) for i in range(5)]
    print(f"  memcpy result: {result}")
    assert result == [1, 2, 3, 4, 5], "MCPY failed"
    print("  ✓ PASS")

    # Test strlen
    print("\nSTRLEN test:")
    for i, c in enumerate("hello"):
        mem_write(0x6000 + i, ord(c))
    mem_write(0x6005, 0)  # Null terminator
    strlen = StrlenSubroutine(mem_read, mem_write)
    result = strlen.execute(0x6000)
    print(f"  strlen('hello') = {result}")
    assert result == 5, "STRLEN failed"
    print("  ✓ PASS")

    # Test strcmp
    print("\nSTRCMP test:")
    for i, c in enumerate("abc"):
        mem_write(0x7000 + i, ord(c))
    mem_write(0x7003, 0)
    for i, c in enumerate("abd"):
        mem_write(0x8000 + i, ord(c))
    mem_write(0x8003, 0)
    strcmp = StrcmpSubroutine(mem_read, mem_write)
    result = strcmp.execute(0x7000, 0x8000)
    print(f"  strcmp('abc', 'abd') = {result}")
    assert result < 0, "STRCMP failed"
    print("  ✓ PASS")

    print("\n✓ All memory subroutine tests passed!")


if __name__ == "__main__":
    test_memory_subroutines()
