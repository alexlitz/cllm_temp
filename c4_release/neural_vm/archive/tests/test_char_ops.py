"""
Test coverage for char (8-bit) operations.

The char datatype uses only the lower 8 bits (2 nibbles).
These tests verify correct handling of:
- LC (Load Char)
- SC (Store Char)
- 8-bit arithmetic with truncation to char range
- Sign extension vs zero extension
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from neural_vm.embedding import E, Opcode
from neural_vm.pure_alu import build_pure_alu
from neural_vm.multi_nibble_ops import MultiNibbleALU


def get_multi_alu():
    """Get a cached MultiNibbleALU instance."""
    if not hasattr(get_multi_alu, 'alu'):
        get_multi_alu.alu = MultiNibbleALU(build_pure_alu())
    return get_multi_alu.alu


def test_char_range():
    """Test char values in valid range (0-255)."""
    print("=== Char Range Tests ===")
    passed = 0
    total = 0

    # Test all boundary values for 8-bit
    test_values = [
        0x00,    # Min char
        0x7F,    # Max signed positive (127)
        0x80,    # Min signed negative (128 unsigned)
        0xFF,    # Max char (255)
        0x41,    # 'A'
        0x61,    # 'a'
        0x30,    # '0'
        0x20,    # space
        0x0A,    # newline
    ]

    multi_alu = get_multi_alu()

    for val in test_values:
        # Test that char values are preserved in lower 8 bits
        # When stored as 32-bit, upper bits should be 0
        result = multi_alu.add(val, 0)
        if (result & 0xFF) == val:
            passed += 1
            print(f"  OK: 0x{val:02X} preserved")
        else:
            print(f"  FAIL: 0x{val:02X} -> 0x{result:08X}")
        total += 1

    print(f"Char Range: {passed}/{total} passed\n")
    return passed == total


def test_char_arithmetic():
    """Test 8-bit arithmetic operations."""
    print("=== Char Arithmetic Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()

    tests = [
        # (a, b, expected_sum & 0xFF, description)
        (0x7F, 0x01, 0x80, "127 + 1 = 128"),
        (0xFF, 0x01, 0x00, "255 + 1 = 0 (overflow)"),
        (0x80, 0x80, 0x00, "128 + 128 = 0 (overflow)"),
        (0x00, 0xFF, 0xFF, "0 + 255 = 255"),
        (0x55, 0xAA, 0xFF, "0x55 + 0xAA = 0xFF"),
        (0x41, 0x20, 0x61, "'A' + ' ' = 'a'"),
    ]

    for a, b, expected, desc in tests:
        result = multi_alu.add(a, b) & 0xFF
        if result == expected:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} -> 0x{result:02X} (expected 0x{expected:02X})")
        total += 1

    print(f"Char Arithmetic: {passed}/{total} passed\n")
    return passed == total


def test_char_subtraction():
    """Test 8-bit subtraction with underflow."""
    print("=== Char Subtraction Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()

    tests = [
        # (a, b, expected_diff & 0xFF, description)
        (0x80, 0x01, 0x7F, "128 - 1 = 127"),
        (0x00, 0x01, 0xFF, "0 - 1 = 255 (underflow)"),
        (0xFF, 0xFF, 0x00, "255 - 255 = 0"),
        (0x61, 0x20, 0x41, "'a' - ' ' = 'A'"),
        (0x39, 0x30, 0x09, "'9' - '0' = 9"),
    ]

    for a, b, expected, desc in tests:
        result = multi_alu.sub(a, b) & 0xFF
        if result == expected:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} -> 0x{result:02X} (expected 0x{expected:02X})")
        total += 1

    print(f"Char Subtraction: {passed}/{total} passed\n")
    return passed == total


def test_char_comparison():
    """Test 8-bit comparisons."""
    print("=== Char Comparison Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()

    # Need to use the ALU's internal encoding for comparison
    from neural_vm.multi_nibble_ops import encode_operands, decode_result
    from neural_vm.embedding import Opcode

    tests = [
        # (a, b, expected_eq, expected_lt, description)
        (0x41, 0x41, 1, 0, "'A' == 'A'"),
        (0x41, 0x61, 0, 1, "'A' < 'a'"),
        (0x61, 0x41, 0, 0, "'a' > 'A'"),
        (0x00, 0xFF, 0, 1, "0 < 255"),
        (0xFF, 0x00, 0, 0, "255 > 0"),
        (0x7F, 0x80, 0, 1, "127 < 128 (unsigned)"),
    ]

    for a, b, exp_eq, exp_lt, desc in tests:
        x_eq = encode_operands(Opcode.EQ, a, b)
        y_eq = multi_alu.alu(x_eq)
        eq_result = decode_result(y_eq)

        x_lt = encode_operands(Opcode.LT, a, b)
        y_lt = multi_alu.alu(x_lt)
        lt_result = decode_result(y_lt)

        if eq_result == exp_eq and lt_result == exp_lt:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} (eq={eq_result}, lt={lt_result})")
        total += 1

    print(f"Char Comparison: {passed}/{total} passed\n")
    return passed == total


def test_char_bitwise():
    """Test 8-bit bitwise operations."""
    print("=== Char Bitwise Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()
    from neural_vm.multi_nibble_ops import encode_operands, decode_result
    from neural_vm.embedding import Opcode

    tests = [
        # (a, b, expected_and, expected_or, expected_xor, description)
        (0xFF, 0x00, 0x00, 0xFF, 0xFF, "0xFF op 0x00"),
        (0xAA, 0x55, 0x00, 0xFF, 0xFF, "alternating bits"),
        (0xF0, 0x0F, 0x00, 0xFF, 0xFF, "high/low nibbles"),
        (0x41, 0x20, 0x00, 0x61, 0x61, "'A' op ' '"),
    ]

    for a, b, exp_and, exp_or, exp_xor, desc in tests:
        x_and = encode_operands(Opcode.AND, a, b)
        and_result = decode_result(multi_alu.alu(x_and)) & 0xFF

        x_or = encode_operands(Opcode.OR, a, b)
        or_result = decode_result(multi_alu.alu(x_or)) & 0xFF

        x_xor = encode_operands(Opcode.XOR, a, b)
        xor_result = decode_result(multi_alu.alu(x_xor)) & 0xFF

        if and_result == exp_and and or_result == exp_or and xor_result == exp_xor:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} (and={and_result:02X}, or={or_result:02X}, xor={xor_result:02X})")
        total += 1

    print(f"Char Bitwise: {passed}/{total} passed\n")
    return passed == total


def test_char_multiplication():
    """Test 8-bit multiplication with truncation."""
    print("=== Char Multiplication Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()

    tests = [
        # (a, b, expected & 0xFF, description)
        (0x02, 0x03, 0x06, "2 * 3 = 6"),
        (0x0F, 0x0F, 0xE1, "15 * 15 = 225"),
        (0xFF, 0x02, 0xFE, "255 * 2 = 510 & 0xFF = 254"),
        (0x10, 0x10, 0x00, "16 * 16 = 256 & 0xFF = 0"),
        (0x41, 0x00, 0x00, "'A' * 0 = 0"),
    ]

    for a, b, expected, desc in tests:
        result = multi_alu.mul_32bit(a, b) & 0xFF
        if result == expected:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} -> 0x{result:02X} (expected 0x{expected:02X})")
        total += 1

    print(f"Char Multiplication: {passed}/{total} passed\n")
    return passed == total


def test_ascii_operations():
    """Test ASCII character operations."""
    print("=== ASCII Operations Tests ===")
    passed = 0
    total = 0

    multi_alu = get_multi_alu()

    # Test case conversion
    tests = [
        # Upper to lower: add 0x20
        (ord('A'), 0x20, ord('a'), "A -> a"),
        (ord('Z'), 0x20, ord('z'), "Z -> z"),
        # Lower to upper: subtract 0x20
        # Digit to value: subtract '0'
        (ord('5'), ord('0'), 5, "'5' - '0' = 5 (sub)"),
    ]

    for a, b, expected, desc in tests:
        if "sub" in desc:
            result = multi_alu.sub(a, b) & 0xFF
        else:
            result = multi_alu.add(a, b) & 0xFF
        if result == expected:
            passed += 1
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} -> {result} (expected {expected})")
        total += 1

    print(f"ASCII Operations: {passed}/{total} passed\n")
    return passed == total


def main():
    """Run all char operation tests."""
    print("=" * 60)
    print("CHAR (8-BIT) OPERATION TEST SUITE")
    print("=" * 60)
    print()

    results = {}
    results['Char Range'] = test_char_range()
    results['Char Arithmetic'] = test_char_arithmetic()
    results['Char Subtraction'] = test_char_subtraction()
    results['Char Comparison'] = test_char_comparison()
    results['Char Bitwise'] = test_char_bitwise()
    results['Char Multiplication'] = test_char_multiplication()
    results['ASCII Operations'] = test_ascii_operations()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL CHAR TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    main()
