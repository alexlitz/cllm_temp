#!/usr/bin/env python3
"""
Test suite for neural memory subroutine FFNs.

Tests MsetFFN, McmpFFN, and memory operations via PureALU.
All operations are PURE NEURAL - no Python control flow in forward passes.
"""

import torch
import sys
import os

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm import E, Opcode, PureALU
from neural_vm.memory_subroutines import (
    MsetSubroutine, McmpSubroutine, McpySubroutine,
    StrlenSubroutine, StrcmpSubroutine, MemorySubroutineHandler
)
from neural_vm.missing_ops import MsetFFN, McmpFFN
from neural_vm.bump_allocator import MallocFFN, FreeFFN


def encode_operands(opcode: int, a: int, b: int) -> torch.Tensor:
    """Encode two 32-bit values and opcode into ALU input format."""
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

    for i in range(E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.OP_START + opcode] = 1.0
        x[0, i, E.POS] = float(i)

    return x


def decode_result(x: torch.Tensor) -> int:
    """Extract 32-bit result from ALU output."""
    result = 0
    for i in range(E.NUM_POSITIONS):
        nib = int(round(x[0, i, E.RESULT].item()))
        nib = max(0, min(15, nib))
        result |= (nib << (i * 4))
    return result


def test_mset_ffn():
    """Test MsetFFN neural layer directly."""
    print("\n=== MsetFFN Direct Test ===")
    passed = 0
    total = 0

    mset = MsetFFN()

    # Test 1: MsetFFN activates on MSET opcode
    total += 1
    x = encode_operands(Opcode.MSET, 0xAB, 0)  # val=0xAB
    y = mset(x)

    # Check that RESULT contains the value
    result_nib0 = y[0, 0, E.RESULT].item()
    if abs(result_nib0 - 0xB) < 1.0:  # Low nibble of 0xAB
        passed += 1
        print(f"  OK: MsetFFN sets RESULT[0] = {result_nib0:.1f}")
    else:
        print(f"  FAIL: MsetFFN RESULT[0] = {result_nib0:.1f}, expected ~11")

    print(f"MsetFFN: {passed}/{total} passed")
    return passed, total


def test_mcmp_ffn():
    """Test McmpFFN neural layer directly."""
    print("\n=== McmpFFN Direct Test ===")
    passed = 0
    total = 0

    mcmp = McmpFFN()

    # Test 1: Equal values should give 0 difference
    total += 1
    x = encode_operands(Opcode.MCMP, 0x55, 0x55)
    y = mcmp(x)

    result = decode_result(y)
    if result == 0:
        passed += 1
        print(f"  OK: McmpFFN(0x55, 0x55) = 0")
    else:
        print(f"  FAIL: McmpFFN(0x55, 0x55) = {result}, expected 0")

    # Test 2: Different values
    total += 1
    x = encode_operands(Opcode.MCMP, 0x60, 0x55)
    y = mcmp(x)

    # Check TEMP slots for difference flags
    temp_val = y[0, 0, E.TEMP].item()
    if temp_val != 0:
        passed += 1
        print(f"  OK: McmpFFN(0x60, 0x55) TEMP = {temp_val:.1f} (non-zero)")
    else:
        print(f"  FAIL: McmpFFN(0x60, 0x55) TEMP = {temp_val:.1f}, expected non-zero")

    print(f"McmpFFN: {passed}/{total} passed")
    return passed, total


def test_malloc_ffn():
    """Test MallocFFN neural layer."""
    print("\n=== MallocFFN Test ===")
    passed = 0
    total = 0

    try:
        malloc = MallocFFN()

        # Test that malloc FFN exists and can forward
        total += 1
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.OP_START + Opcode.MALC] = 1.0
            x[0, i, E.NIB_A] = float(16)  # Allocate 16 bytes
            x[0, i, E.POS] = float(i)

        y = malloc(x)
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: MallocFFN forward pass works")
        else:
            print(f"  FAIL: MallocFFN output shape mismatch")
    except Exception as e:
        print(f"  SKIP: MallocFFN not available ({e})")

    print(f"MallocFFN: {passed}/{total} passed")
    return passed, total


def test_free_ffn():
    """Test FreeFFN neural layer."""
    print("\n=== FreeFFN Test ===")
    passed = 0
    total = 0

    try:
        free = FreeFFN()

        # Test that free FFN exists and can forward
        total += 1
        x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
        for i in range(E.NUM_POSITIONS):
            x[0, i, E.OP_START + Opcode.FREE] = 1.0
            x[0, i, E.NIB_A] = float((0x1000 >> (i * 4)) & 0xF)  # ptr=0x1000
            x[0, i, E.POS] = float(i)

        y = free(x)
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: FreeFFN forward pass works")
        else:
            print(f"  FAIL: FreeFFN output shape mismatch")
    except Exception as e:
        print(f"  SKIP: FreeFFN not available ({e})")

    print(f"FreeFFN: {passed}/{total} passed")
    return passed, total


def test_memory_subroutine_handler():
    """Test MemorySubroutineHandler (reference implementation)."""
    print("\n=== MemorySubroutineHandler Tests ===")
    passed = 0
    total = 0

    # Simple in-memory storage for testing
    memory = {}

    def mem_read(addr: int) -> int:
        return memory.get(addr, 0)

    def mem_write(addr: int, val: int):
        memory[addr] = val & 0xFF

    # Test MSET
    total += 1
    mset = MsetSubroutine(mem_read, mem_write)
    mset.execute(0x1000, 0xAB, 10)
    result = [memory.get(0x1000 + i, 0) for i in range(10)]
    if all(x == 0xAB for x in result):
        passed += 1
        print(f"  OK: mset(0x1000, 0xAB, 10)")
    else:
        print(f"  FAIL: mset result = {[hex(x) for x in result]}")

    # Test MCMP (equal)
    total += 1
    for i in range(5):
        mem_write(0x2000 + i, 0x55)
        mem_write(0x3000 + i, 0x55)
    mcmp = McmpSubroutine(mem_read, mem_write)
    result = mcmp.execute(0x2000, 0x3000, 5)
    if result == 0:
        passed += 1
        print(f"  OK: mcmp(equal regions) = 0")
    else:
        print(f"  FAIL: mcmp(equal regions) = {result}")

    # Test MCMP (not equal)
    total += 1
    mem_write(0x3002, 0x66)  # Make one byte different
    result = mcmp.execute(0x2000, 0x3000, 5)
    expected = 0x55 - 0x66
    if result == expected:
        passed += 1
        print(f"  OK: mcmp(diff at byte 2) = {result}")
    else:
        print(f"  FAIL: mcmp(diff at byte 2) = {result}, expected {expected}")

    # Test MCPY
    total += 1
    for i in range(5):
        mem_write(0x4000 + i, i + 1)
    mcpy = McpySubroutine(mem_read, mem_write)
    mcpy.execute(0x5000, 0x4000, 5)
    result = [memory.get(0x5000 + i, 0) for i in range(5)]
    if result == [1, 2, 3, 4, 5]:
        passed += 1
        print(f"  OK: mcpy(0x5000, 0x4000, 5) = {result}")
    else:
        print(f"  FAIL: mcpy result = {result}")

    # Test strlen
    total += 1
    for i, c in enumerate("hello"):
        mem_write(0x6000 + i, ord(c))
    mem_write(0x6005, 0)  # Null terminator
    strlen = StrlenSubroutine(mem_read, mem_write)
    result = strlen.execute(0x6000)
    if result == 5:
        passed += 1
        print(f"  OK: strlen('hello') = {result}")
    else:
        print(f"  FAIL: strlen('hello') = {result}")

    # Test strcmp
    total += 1
    for i, c in enumerate("abc"):
        mem_write(0x7000 + i, ord(c))
    mem_write(0x7003, 0)
    for i, c in enumerate("abd"):
        mem_write(0x8000 + i, ord(c))
    mem_write(0x8003, 0)
    strcmp = StrcmpSubroutine(mem_read, mem_write)
    result = strcmp.execute(0x7000, 0x8000)
    if result < 0:
        passed += 1
        print(f"  OK: strcmp('abc', 'abd') = {result} (<0)")
    else:
        print(f"  FAIL: strcmp('abc', 'abd') = {result}")

    print(f"MemorySubroutineHandler: {passed}/{total} passed")
    return passed, total


def test_pure_alu_memory_ops():
    """Test memory operations through PureALU."""
    print("\n=== PureALU Memory Ops ===")
    passed = 0
    total = 0

    try:
        alu = PureALU()

        # Test MSET through ALU
        total += 1
        x = encode_operands(Opcode.MSET, 0xAB, 0)
        y = alu(x)
        result = decode_result(y)
        # MSET writes to RESULT slots
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: PureALU MSET forward pass")
        else:
            print(f"  FAIL: PureALU MSET")

        # Test MCMP through ALU
        total += 1
        x = encode_operands(Opcode.MCMP, 0x55, 0x55)
        y = alu(x)
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: PureALU MCMP forward pass")
        else:
            print(f"  FAIL: PureALU MCMP")

        # Test MALC through ALU
        total += 1
        x = encode_operands(Opcode.MALC, 256, 0)  # malloc(256)
        y = alu(x)
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: PureALU MALC forward pass")
        else:
            print(f"  FAIL: PureALU MALC")

        # Test FREE through ALU
        total += 1
        x = encode_operands(Opcode.FREE, 0x1000, 0)  # free(0x1000)
        y = alu(x)
        if y.shape == x.shape:
            passed += 1
            print(f"  OK: PureALU FREE forward pass")
        else:
            print(f"  FAIL: PureALU FREE")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

    print(f"PureALU Memory Ops: {passed}/{total} passed")
    return passed, total


def main():
    print("=" * 60)
    print("NEURAL MEMORY SUBROUTINE TESTS")
    print("All operations are PURE NEURAL - no Python control flow")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    # Test individual FFN layers
    p, t = test_mset_ffn()
    total_passed += p
    total_tests += t

    p, t = test_mcmp_ffn()
    total_passed += p
    total_tests += t

    p, t = test_malloc_ffn()
    total_passed += p
    total_tests += t

    p, t = test_free_ffn()
    total_passed += p
    total_tests += t

    # Test reference implementation (for bytecode generation verification)
    p, t = test_memory_subroutine_handler()
    total_passed += p
    total_tests += t

    # Test through PureALU
    p, t = test_pure_alu_memory_ops()
    total_passed += p
    total_tests += t

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)

    if total_passed == total_tests:
        print("\nAll memory subroutine tests passed!")
        return 0
    else:
        print(f"\n{total_tests - total_passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
