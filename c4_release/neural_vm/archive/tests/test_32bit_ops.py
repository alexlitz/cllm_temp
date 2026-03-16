#!/usr/bin/env python3
"""
Test suite for 32-bit MUL, DIV, MOD operations.

Tests both the simple schoolbook/long-division implementations
and the neural ALU wrapper approach.
"""

import torch
import sys
import os

# Add parent directories for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from neural_vm import E, Opcode, PureALU
from neural_vm.multi_nibble_ops import (
    MultiNibbleALU, mul_schoolbook, div_long,
    encode_operands, decode_result
)


def test_schoolbook_mul():
    """Test schoolbook multiplication (pure Python, no neural ALU)."""
    print("\n=== Schoolbook MUL Tests ===")
    passed = 0
    total = 0

    tests = [
        # Small numbers
        (2, 3, 6),
        (5, 5, 25),
        (15, 15, 225),
        (10, 10, 100),
        # Multi-nibble (16+ results)
        (16, 2, 32),
        (255, 2, 510),
        (256, 256, 65536),
        # Larger numbers
        (1000, 1000, 1000000),
        (12345, 67, 827115),
        (0xFFFF, 0xFFFF, 0xFFFE0001),
        # Edge cases
        (0, 12345, 0),
        (12345, 0, 0),
        (1, 0xFFFFFFFF, 0xFFFFFFFF),
        # Large 32-bit
        (0x12345678, 2, 0x2468ACF0),
        (0x10000, 0x10000, 0),  # Overflow wraps
        (0x1000, 0x1000, 0x1000000),
    ]

    for a, b, expected in tests:
        result = mul_schoolbook(a, b, None)  # No ALU needed for schoolbook
        expected_masked = expected & 0xFFFFFFFF
        total += 1
        if result == expected_masked:
            passed += 1
        else:
            print(f"FAIL: schoolbook_mul({a:#x}, {b:#x}) = {result:#x}, expected {expected_masked:#x}")

    print(f"Schoolbook MUL: {passed}/{total} passed")
    return passed, total


def test_long_div():
    """Test long division (pure Python, no neural ALU)."""
    print("\n=== Long DIV Tests ===")
    passed = 0
    total = 0

    tests = [
        # Small numbers
        (10, 2, 5),
        (15, 3, 5),
        (14, 4, 3),
        (9, 3, 3),
        # Multi-nibble dividends
        (100, 7, 14),
        (1000, 33, 30),
        (65535, 256, 255),
        # Large numbers
        (0x12345678, 0x1234, 0x10004),
        (0xFFFFFFFF, 2, 0x7FFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF, 1),
        (0xFFFFFFFF, 0x10000, 0xFFFF),
        # Edge cases
        (0, 5, 0),
        (5, 0, 0),  # Div by zero returns 0
        (1, 1, 1),
        (100, 100, 1),
    ]

    for a, b, expected in tests:
        quotient, _ = div_long(a, b)
        total += 1
        if quotient == expected:
            passed += 1
        else:
            print(f"FAIL: div_long({a:#x}, {b:#x}) = {quotient:#x}, expected {expected:#x}")

    print(f"Long DIV: {passed}/{total} passed")
    return passed, total


def test_long_mod():
    """Test long modulo (pure Python, no neural ALU)."""
    print("\n=== Long MOD Tests ===")
    passed = 0
    total = 0

    tests = [
        # Small numbers
        (10, 3, 1),
        (15, 4, 3),
        (7, 2, 1),
        (9, 5, 4),
        # Multi-nibble
        (100, 7, 2),
        (1000, 33, 10),
        (65535, 256, 255),
        # Large numbers
        (0x12345678, 0x1000, 0x678),
        (0xFFFFFFFF, 2, 1),
        (0xFFFFFFFF, 0x10000, 0xFFFF),
        # Edge cases
        (0, 5, 0),
        (5, 5, 0),
        (100, 100, 0),
    ]

    for a, b, expected in tests:
        _, remainder = div_long(a, b)
        total += 1
        if remainder == expected:
            passed += 1
        else:
            print(f"FAIL: mod_long({a:#x}, {b:#x}) = {remainder:#x}, expected {expected:#x}")

    print(f"Long MOD: {passed}/{total} passed")
    return passed, total


def test_neural_alu_mul(alu):
    """Test 32-bit MUL through neural ALU wrapper."""
    print("\n=== Neural ALU 32-bit MUL Tests ===")
    passed = 0
    total = 0

    multi_alu = MultiNibbleALU(alu)

    tests = [
        # Small numbers (would work with single-nibble too)
        (2, 3, 6),
        (5, 5, 25),
        (10, 10, 100),
        # Multi-nibble results
        (16, 16, 256),
        (100, 100, 10000),
        (255, 255, 65025),
        # Larger numbers
        (1000, 1000, 1000000),
        (12345, 67, 827115),
        # 32-bit range
        (0x1000, 0x100, 0x100000),
        (0x10000, 0x10, 0x100000),
    ]

    for a, b, expected in tests:
        result = multi_alu.mul_32bit(a, b)
        expected_masked = expected & 0xFFFFFFFF
        total += 1
        if result == expected_masked:
            passed += 1
            print(f"OK: mul_32bit({a}, {b}) = {result}")
        else:
            print(f"FAIL: mul_32bit({a}, {b}) = {result}, expected {expected_masked}")

    print(f"Neural ALU MUL: {passed}/{total} passed")
    return passed, total


def test_neural_alu_div(alu):
    """Test 32-bit DIV through neural ALU wrapper."""
    print("\n=== Neural ALU 32-bit DIV Tests ===")
    passed = 0
    total = 0

    multi_alu = MultiNibbleALU(alu)

    tests = [
        # Small numbers
        (10, 2, 5),
        (15, 3, 5),
        (100, 10, 10),
        # Multi-nibble
        (1000, 100, 10),
        (65536, 256, 256),
        (10000, 33, 303),
        # Larger numbers
        (0x100000, 0x100, 0x1000),
        (0x1000000, 0x1000, 0x1000),
    ]

    for a, b, expected in tests:
        result = multi_alu.div_32bit(a, b)
        total += 1
        if result == expected:
            passed += 1
            print(f"OK: div_32bit({a}, {b}) = {result}")
        else:
            print(f"FAIL: div_32bit({a}, {b}) = {result}, expected {expected}")

    print(f"Neural ALU DIV: {passed}/{total} passed")
    return passed, total


def test_neural_alu_mod(alu):
    """Test 32-bit MOD through neural ALU wrapper."""
    print("\n=== Neural ALU 32-bit MOD Tests ===")
    passed = 0
    total = 0

    multi_alu = MultiNibbleALU(alu)

    tests = [
        # Small numbers
        (10, 3, 1),
        (15, 4, 3),
        (100, 7, 2),
        # Multi-nibble
        (1000, 33, 10),
        (65535, 256, 255),
        (10000, 100, 0),
        # Larger numbers
        (0x12345678, 0x1000, 0x678),
    ]

    for a, b, expected in tests:
        result = multi_alu.mod_32bit(a, b)
        total += 1
        if result == expected:
            passed += 1
            print(f"OK: mod_32bit({a}, {b}) = {result}")
        else:
            print(f"FAIL: mod_32bit({a}, {b}) = {result}, expected {expected}")

    print(f"Neural ALU MOD: {passed}/{total} passed")
    return passed, total


def test_alu_primitives(alu):
    """Verify the primitive operations work correctly for multi-nibble wrapper."""
    print("\n=== Neural ALU Primitive Tests ===")
    passed = 0
    total = 0

    multi_alu = MultiNibbleALU(alu)

    # Test ADD
    add_tests = [(100, 200, 300), (0xFFFF, 1, 0x10000), (0x12345678, 0x11111111, 0x23456789)]
    for a, b, exp in add_tests:
        result = multi_alu.add(a, b)
        total += 1
        if result == (exp & 0xFFFFFFFF):
            passed += 1
        else:
            print(f"FAIL: add({a:#x}, {b:#x}) = {result:#x}, expected {exp:#x}")

    # Test SUB
    sub_tests = [(300, 100, 200), (0x10000, 1, 0xFFFF), (0x12345678, 0x11111111, 0x01234567)]
    for a, b, exp in sub_tests:
        result = multi_alu.sub(a, b)
        total += 1
        if result == (exp & 0xFFFFFFFF):
            passed += 1
        else:
            print(f"FAIL: sub({a:#x}, {b:#x}) = {result:#x}, expected {exp:#x}")

    # Test GE
    ge_tests = [(5, 3, 1), (3, 5, 0), (5, 5, 1), (0x10000, 0xFFFF, 1), (0xFFFF, 0x10000, 0)]
    for a, b, exp in ge_tests:
        result = multi_alu.ge(a, b)
        total += 1
        if result == exp:
            passed += 1
        else:
            print(f"FAIL: ge({a:#x}, {b:#x}) = {result}, expected {exp}")

    # Test SHL
    shl_tests = [(1, 4, 16), (0xFF, 8, 0xFF00), (0x12345678, 4, 0x23456780)]
    for a, b, exp in shl_tests:
        result = multi_alu.shl(a, b)
        total += 1
        if result == (exp & 0xFFFFFFFF):
            passed += 1
        else:
            print(f"FAIL: shl({a:#x}, {b}) = {result:#x}, expected {exp:#x}")

    # Test SHR
    shr_tests = [(16, 4, 1), (0xFF00, 8, 0xFF), (0x12345678, 4, 0x01234567)]
    for a, b, exp in shr_tests:
        result = multi_alu.shr(a, b)
        total += 1
        if result == exp:
            passed += 1
        else:
            print(f"FAIL: shr({a:#x}, {b}) = {result:#x}, expected {exp:#x}")

    print(f"Primitives: {passed}/{total} passed")
    return passed, total


def main():
    print("=" * 60)
    print("32-bit MUL/DIV/MOD Test Suite")
    print("=" * 60)

    total_passed = 0
    total_tests = 0

    # Pure Python tests (verify algorithms)
    p, t = test_schoolbook_mul()
    total_passed += p
    total_tests += t

    p, t = test_long_div()
    total_passed += p
    total_tests += t

    p, t = test_long_mod()
    total_passed += p
    total_tests += t

    # Neural ALU tests
    print("\n" + "=" * 60)
    print("Loading Neural ALU...")
    try:
        alu = PureALU()
        print("ALU loaded successfully")

        p, t = test_alu_primitives(alu)
        total_passed += p
        total_tests += t

        p, t = test_neural_alu_mul(alu)
        total_passed += p
        total_tests += t

        p, t = test_neural_alu_div(alu)
        total_passed += p
        total_tests += t

        p, t = test_neural_alu_mod(alu)
        total_passed += p
        total_tests += t

    except Exception as e:
        print(f"ERROR loading ALU: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
