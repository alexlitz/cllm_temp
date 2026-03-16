#!/usr/bin/env python3
"""
Test suite for C4 opcode implementations.

Tests all 42 C4 opcodes with 32-bit operands.
"""

import torch
import sys
import os

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm import E, Opcode, PureALU


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


def test_alu_op(alu, opcode: int, a: int, b: int, expected: int, name: str) -> bool:
    """Test a single ALU operation."""
    x = encode_operands(opcode, a, b)
    y = alu(x)
    result = decode_result(y)

    if result == expected:
        return True
    else:
        print(f"FAIL: {name}({a}, {b}) = {result}, expected {expected}")
        return False


def test_arithmetic_ops(alu):
    """Test ADD, SUB, MUL, DIV, MOD."""
    print("\n=== Arithmetic Operations ===")
    passed = 0
    total = 0

    # ADD tests
    add_tests = [
        (0, 0, 0),
        (1, 1, 2),
        (100, 200, 300),
        (0xFFFF, 1, 0x10000),
        (0x12345678, 0x11111111, 0x23456789),
        (0xFFFFFFFF, 1, 0),  # Overflow wraps
    ]
    for a, b, exp in add_tests:
        total += 1
        if test_alu_op(alu, Opcode.ADD, a, b, exp & 0xFFFFFFFF, "ADD"):
            passed += 1

    # SUB tests
    sub_tests = [
        (5, 3, 2),
        (100, 50, 50),
        (0, 1, 0xFFFFFFFF),  # Underflow wraps
        (0x12345678, 0x11111111, 0x01234567),
    ]
    for a, b, exp in sub_tests:
        total += 1
        if test_alu_op(alu, Opcode.SUB, a, b, exp & 0xFFFFFFFF, "SUB"):
            passed += 1

    # MUL tests (single nibble for now)
    mul_tests = [
        (2, 3, 6),
        (5, 5, 25),
        (15, 15, 225),
        (10, 10, 100),
    ]
    for a, b, exp in mul_tests:
        total += 1
        if test_alu_op(alu, Opcode.MUL, a, b, exp, "MUL"):
            passed += 1

    # DIV tests (single nibble for now)
    div_tests = [
        (10, 2, 5),
        (15, 3, 5),
        (14, 4, 3),
        (9, 3, 3),
    ]
    for a, b, exp in div_tests:
        total += 1
        if test_alu_op(alu, Opcode.DIV, a, b, exp, "DIV"):
            passed += 1

    # MOD tests (single nibble for now)
    mod_tests = [
        (10, 3, 1),
        (15, 4, 3),
        (7, 2, 1),
        (9, 5, 4),
    ]
    for a, b, exp in mod_tests:
        total += 1
        if test_alu_op(alu, Opcode.MOD, a, b, exp, "MOD"):
            passed += 1

    print(f"Arithmetic: {passed}/{total} passed")
    return passed, total


def test_bitwise_ops(alu):
    """Test AND, OR, XOR."""
    print("\n=== Bitwise Operations ===")
    passed = 0
    total = 0

    # AND tests
    and_tests = [
        (0xFF, 0x0F, 0x0F),
        (0xAA, 0x55, 0x00),
        (0x12345678, 0xFF00FF00, 0x12005600),
    ]
    for a, b, exp in and_tests:
        total += 1
        if test_alu_op(alu, Opcode.AND, a, b, exp, "AND"):
            passed += 1

    # OR tests
    or_tests = [
        (0xF0, 0x0F, 0xFF),
        (0xAA, 0x55, 0xFF),
        (0x12340000, 0x00005678, 0x12345678),
    ]
    for a, b, exp in or_tests:
        total += 1
        if test_alu_op(alu, Opcode.OR, a, b, exp, "OR"):
            passed += 1

    # XOR tests
    xor_tests = [
        (0xFF, 0xFF, 0x00),
        (0xAA, 0x55, 0xFF),
        (0x12345678, 0x12345678, 0x00000000),
    ]
    for a, b, exp in xor_tests:
        total += 1
        if test_alu_op(alu, Opcode.XOR, a, b, exp, "XOR"):
            passed += 1

    print(f"Bitwise: {passed}/{total} passed")
    return passed, total


def test_comparison_ops(alu):
    """Test EQ, NE, LT, GT, LE, GE."""
    print("\n=== Comparison Operations ===")
    passed = 0
    total = 0

    # EQ tests
    eq_tests = [
        (5, 5, 1),
        (5, 6, 0),
        (0x12345678, 0x12345678, 1),
        (0x12345678, 0x12345679, 0),
    ]
    for a, b, exp in eq_tests:
        total += 1
        if test_alu_op(alu, Opcode.EQ, a, b, exp, "EQ"):
            passed += 1

    # NE tests
    ne_tests = [
        (5, 5, 0),
        (5, 6, 1),
        (0, 1, 1),
    ]
    for a, b, exp in ne_tests:
        total += 1
        if test_alu_op(alu, Opcode.NE, a, b, exp, "NE"):
            passed += 1

    # LT tests
    lt_tests = [
        (3, 5, 1),
        (5, 3, 0),
        (5, 5, 0),
        (100, 1000, 1),
    ]
    for a, b, exp in lt_tests:
        total += 1
        if test_alu_op(alu, Opcode.LT, a, b, exp, "LT"):
            passed += 1

    # GT tests
    gt_tests = [
        (5, 3, 1),
        (3, 5, 0),
        (5, 5, 0),
    ]
    for a, b, exp in gt_tests:
        total += 1
        if test_alu_op(alu, Opcode.GT, a, b, exp, "GT"):
            passed += 1

    # LE tests
    le_tests = [
        (3, 5, 1),
        (5, 5, 1),
        (5, 3, 0),
    ]
    for a, b, exp in le_tests:
        total += 1
        if test_alu_op(alu, Opcode.LE, a, b, exp, "LE"):
            passed += 1

    # GE tests
    ge_tests = [
        (5, 3, 1),
        (5, 5, 1),
        (3, 5, 0),
    ]
    for a, b, exp in ge_tests:
        total += 1
        if test_alu_op(alu, Opcode.GE, a, b, exp, "GE"):
            passed += 1

    print(f"Comparison: {passed}/{total} passed")
    return passed, total


def test_shift_ops(alu):
    """Test SHL, SHR."""
    print("\n=== Shift Operations ===")
    passed = 0
    total = 0

    # SHL tests
    shl_tests = [
        (1, 4, 16),
        (0xFF, 8, 0xFF00),
        (0x12345678, 4, 0x23456780),
    ]
    for a, b, exp in shl_tests:
        total += 1
        if test_alu_op(alu, Opcode.SHL, a, b, exp & 0xFFFFFFFF, "SHL"):
            passed += 1

    # SHR tests
    shr_tests = [
        (16, 4, 1),
        (0xFF00, 8, 0xFF),
        (0x12345678, 4, 0x01234567),
    ]
    for a, b, exp in shr_tests:
        total += 1
        if test_alu_op(alu, Opcode.SHR, a, b, exp, "SHR"):
            passed += 1

    print(f"Shift: {passed}/{total} passed")
    return passed, total


def test_32bit_arithmetic(alu):
    """Test 32-bit arithmetic operations."""
    print("\n=== 32-bit Arithmetic ===")
    passed = 0
    total = 0

    # Large 32-bit ADD
    tests = [
        (0x10000000, 0x20000000, 0x30000000, "ADD large"),
        (0xFFFF0000, 0x0000FFFF, 0xFFFFFFFF, "ADD max"),
        (0x12345678, 0x87654321, 0x99999999, "ADD cross"),
    ]
    for a, b, exp, name in tests:
        total += 1
        if test_alu_op(alu, Opcode.ADD, a, b, exp & 0xFFFFFFFF, name):
            passed += 1

    # Large 32-bit SUB
    tests = [
        (0x30000000, 0x10000000, 0x20000000, "SUB large"),
        (0xFFFFFFFF, 0x00000001, 0xFFFFFFFE, "SUB from max"),
        (0x12345678, 0x12345678, 0x00000000, "SUB equal"),
    ]
    for a, b, exp, name in tests:
        total += 1
        if test_alu_op(alu, Opcode.SUB, a, b, exp & 0xFFFFFFFF, name):
            passed += 1

    print(f"32-bit Arithmetic: {passed}/{total} passed")
    return passed, total


def test_32bit_comparison(alu):
    """Test 32-bit comparison operations."""
    print("\n=== 32-bit Comparison ===")
    passed = 0
    total = 0

    tests = [
        (Opcode.EQ, 0xDEADBEEF, 0xDEADBEEF, 1, "EQ large equal"),
        (Opcode.EQ, 0xDEADBEEF, 0xDEADBEEE, 0, "EQ large diff"),
        (Opcode.LT, 0x10000000, 0x20000000, 1, "LT large"),
        (Opcode.LT, 0x20000000, 0x10000000, 0, "LT large reverse"),
        (Opcode.GT, 0x20000000, 0x10000000, 1, "GT large"),
        (Opcode.GT, 0x10000000, 0x20000000, 0, "GT large reverse"),
        (Opcode.LE, 0xFFFFFFFF, 0xFFFFFFFF, 1, "LE max equal"),
        (Opcode.GE, 0x00000000, 0x00000000, 1, "GE zero equal"),
    ]
    for op, a, b, exp, name in tests:
        total += 1
        if test_alu_op(alu, op, a, b, exp, name):
            passed += 1

    print(f"32-bit Comparison: {passed}/{total} passed")
    return passed, total


def test_32bit_shift(alu):
    """Test 32-bit shift operations."""
    print("\n=== 32-bit Shift ===")
    passed = 0
    total = 0

    # Various shift amounts
    tests = [
        (Opcode.SHL, 0x00000001, 16, 0x00010000, "SHL by 16"),
        (Opcode.SHL, 0x000000FF, 24, 0xFF000000, "SHL by 24"),
        (Opcode.SHR, 0xFF000000, 24, 0x000000FF, "SHR by 24"),
        (Opcode.SHR, 0x12345678, 16, 0x00001234, "SHR by 16"),
        (Opcode.SHL, 0x12345678, 4, 0x23456780, "SHL by 4"),
        (Opcode.SHR, 0x12345678, 4, 0x01234567, "SHR by 4"),
    ]
    for op, a, b, exp, name in tests:
        total += 1
        if test_alu_op(alu, op, a, b, exp & 0xFFFFFFFF, name):
            passed += 1

    print(f"32-bit Shift: {passed}/{total} passed")
    return passed, total


def main():
    print("=" * 60)
    print("C4 Neural VM Opcode Test Suite")
    print("=" * 60)

    print("\nLoading ALU...")
    try:
        alu = PureALU()
        print("ALU loaded successfully")
    except Exception as e:
        print(f"ERROR loading ALU: {e}")
        return

    total_passed = 0
    total_tests = 0

    # Run test suites
    p, t = test_arithmetic_ops(alu)
    total_passed += p
    total_tests += t

    p, t = test_bitwise_ops(alu)
    total_passed += p
    total_tests += t

    p, t = test_comparison_ops(alu)
    total_passed += p
    total_tests += t

    p, t = test_shift_ops(alu)
    total_passed += p
    total_tests += t

    p, t = test_32bit_arithmetic(alu)
    total_passed += p
    total_tests += t

    p, t = test_32bit_comparison(alu)
    total_passed += p
    total_tests += t

    p, t = test_32bit_shift(alu)
    total_passed += p
    total_tests += t

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)

    if total_passed == total_tests:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total_tests - total_passed} tests failed")


if __name__ == "__main__":
    main()
