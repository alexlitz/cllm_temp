#!/usr/bin/env python3
"""
Comprehensive test suite for V6 Neural VM.
"""
import sys
sys.path.insert(0, '/Users/alexlitz/Dropbox/Docs/misc/llm_math/c4_release')

from pure_gen_vm_v6 import NeuralVMv6, Opcode, E, print_opcode_table


def test_add():
    """Test ADD operation."""
    print("\n=== ADD Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (1, 1, 2),
        (5, 3, 8),
        (15, 1, 16),
        (255, 1, 256),
        (100, 200, 300),
        (65535, 1, 65536),
        (0xFFFFFFFF, 1, 0),  # Overflow
        (12345, 54321, 66666),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.ADD)
        expected = expected & 0xFFFFFFFF
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} + {b} = {result} (expected {expected}) {status}")

    print(f"ADD: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_sub():
    """Test SUB operation."""
    print("\n=== SUB Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (5, 3, 2),
        (100, 50, 50),
        (256, 1, 255),
        (0, 1, 0xFFFFFFFF),  # Underflow
        (1000, 999, 1),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.SUB)
        expected = expected & 0xFFFFFFFF
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} - {b} = {result} (expected {expected}) {status}")

    print(f"SUB: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_mul():
    """Test MUL operation."""
    print("\n=== MUL Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (1, 1, 1),
        (6, 7, 42),
        (12, 12, 144),
        (15, 15, 225),
        (255, 255, 65025),
        (1000, 1000, 1000000),
        (12345, 6789, 83810205),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.MUL)
        expected = expected & 0xFFFFFFFF
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} * {b} = {result} (expected {expected}) {status}")

    print(f"MUL: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_div():
    """Test DIV operation."""
    print("\n=== DIV Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 1, 0),
        (1, 1, 1),
        (42, 7, 6),
        (100, 10, 10),
        (7, 2, 3),       # Truncation
        (14, 15, 0),     # a < b
        (1000, 33, 30),
        (65535, 255, 257),
        (999999, 123, 8130),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.DIV)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} / {b} = {result} (expected {expected}) {status}")

    print(f"DIV: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_mod():
    """Test MOD operation."""
    print("\n=== MOD Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 1, 0),
        (1, 1, 0),
        (42, 7, 0),
        (100, 7, 2),
        (17, 5, 2),
        (1000, 33, 10),
        (65535, 256, 255),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.MOD)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} % {b} = {result} (expected {expected}) {status}")

    print(f"MOD: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_and():
    """Test AND operation."""
    print("\n=== AND Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (0xFF, 0xFF, 0xFF),
        (0xFF, 0x0F, 0x0F),
        (0xAAAA, 0x5555, 0),
        (0xFFFF, 0xFFFF, 0xFFFF),
        (0x12345678, 0xF0F0F0F0, 0x10305070),
        (0xFFFFFFFF, 0, 0),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.AND)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  0x{a:X} & 0x{b:X} = 0x{result:X} (expected 0x{expected:X}) {status}")

    print(f"AND: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_or():
    """Test OR operation."""
    print("\n=== OR Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (0xFF, 0, 0xFF),
        (0xF0, 0x0F, 0xFF),
        (0xAAAA, 0x5555, 0xFFFF),
        (0x1234, 0x5678, 0x567C),
        (0xFFFFFFFF, 0, 0xFFFFFFFF),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.OR)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  0x{a:X} | 0x{b:X} = 0x{result:X} (expected 0x{expected:X}) {status}")

    print(f"OR: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_xor():
    """Test XOR operation."""
    print("\n=== XOR Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (0xFF, 0, 0xFF),
        (0xFF, 0xFF, 0),
        (0xAAAA, 0x5555, 0xFFFF),
        (0x1234, 0x5678, 0x444C),
        (0xFFFFFFFF, 0xFFFFFFFF, 0),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.XOR)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  0x{a:X} ^ 0x{b:X} = 0x{result:X} (expected 0x{expected:X}) {status}")

    print(f"XOR: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_eq():
    """Test EQ operation."""
    print("\n=== EQ Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 1),
        (1, 1, 1),
        (0, 1, 0),
        (1, 0, 0),
        (12345, 12345, 1),
        (12345, 12346, 0),
        (0xFFFFFFFF, 0xFFFFFFFF, 1),
        (0xFFFFFFFF, 0, 0),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.EQ)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} == {b} = {result} (expected {expected}) {status}")

    print(f"EQ: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_ne():
    """Test NE operation."""
    print("\n=== NE Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (12345, 12345, 0),
        (12345, 12346, 1),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.NE)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} != {b} = {result} (expected {expected}) {status}")

    print(f"NE: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_lt():
    """Test LT operation."""
    print("\n=== LT Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (5, 10, 1),
        (10, 5, 0),
        (100, 100, 0),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.LT)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} < {b} = {result} (expected {expected}) {status}")

    print(f"LT: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_gt():
    """Test GT operation."""
    print("\n=== GT Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 1),
        (5, 10, 0),
        (10, 5, 1),
        (100, 100, 0),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.GT)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} > {b} = {result} (expected {expected}) {status}")

    print(f"GT: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_le():
    """Test LE operation."""
    print("\n=== LE Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (5, 10, 1),
        (10, 5, 0),
        (100, 100, 1),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.LE)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} <= {b} = {result} (expected {expected}) {status}")

    print(f"LE: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_ge():
    """Test GE operation."""
    print("\n=== GE Tests ===")
    vm = NeuralVMv6()

    tests = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 1),
        (5, 10, 0),
        (10, 5, 1),
        (100, 100, 1),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.GE)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} >= {b} = {result} (expected {expected}) {status}")

    print(f"GE: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_shl():
    """Test SHL operation."""
    print("\n=== SHL Tests ===")
    vm = NeuralVMv6()

    tests = [
        (1, 0, 1),
        (1, 1, 2),
        (1, 4, 16),
        (1, 8, 256),
        (1, 16, 65536),
        (0xFF, 8, 0xFF00),
        (1, 31, 0x80000000),
        (0xFFFFFFFF, 1, 0xFFFFFFFE),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.SHL)
        expected = expected & 0xFFFFFFFF
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} << {b} = {result} (expected {expected}) {status}")

    print(f"SHL: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_shr():
    """Test SHR operation."""
    print("\n=== SHR Tests ===")
    vm = NeuralVMv6()

    tests = [
        (1, 0, 1),
        (2, 1, 1),
        (16, 4, 1),
        (256, 8, 1),
        (0xFF00, 8, 0xFF),
        (0x80000000, 31, 1),
        (0xFFFFFFFF, 1, 0x7FFFFFFF),
    ]

    passed = 0
    for a, b, expected in tests:
        result = vm.compute(a, b, Opcode.SHR)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {a} >> {b} = {result} (expected {expected}) {status}")

    print(f"SHR: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_c4_behavior():
    """Test that behavior matches c4 compiler expectations."""
    print("\n=== C4 Compatibility Tests ===")
    vm = NeuralVMv6()

    # C4 uses 32-bit signed integers, but we work with unsigned
    # Division by zero should return 0 (handled in code)
    # Modulo by zero should return 0

    tests = [
        # Basic operations
        ("6 * 7", Opcode.MUL, 6, 7, 42),
        ("100 + 200", Opcode.ADD, 100, 200, 300),
        ("1000 / 33", Opcode.DIV, 1000, 33, 30),
        ("1000 % 33", Opcode.MOD, 1000, 33, 10),
        # Bitwise
        ("0xFF & 0x0F", Opcode.AND, 0xFF, 0x0F, 0x0F),
        ("0xF0 | 0x0F", Opcode.OR, 0xF0, 0x0F, 0xFF),
        ("0xFF ^ 0xFF", Opcode.XOR, 0xFF, 0xFF, 0),
        # Shifts
        ("1 << 4", Opcode.SHL, 1, 4, 16),
        ("16 >> 4", Opcode.SHR, 16, 4, 1),
        # Comparisons
        ("5 == 5", Opcode.EQ, 5, 5, 1),
        ("5 != 6", Opcode.NE, 5, 6, 1),
        ("3 < 7", Opcode.LT, 3, 7, 1),
    ]

    passed = 0
    for name, op, a, b, expected in tests:
        result = vm.compute(a, b, op)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  {name} = {result} (expected {expected}) {status}")

    print(f"C4 Compat: {passed}/{len(tests)} passed")
    return passed == len(tests)


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("V6 COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    print_opcode_table()

    results = []

    # Arithmetic
    results.append(("ADD", test_add()))
    results.append(("SUB", test_sub()))
    results.append(("MUL", test_mul()))
    results.append(("DIV", test_div()))
    results.append(("MOD", test_mod()))

    # Bitwise
    results.append(("AND", test_and()))
    results.append(("OR", test_or()))
    results.append(("XOR", test_xor()))

    # Comparisons
    results.append(("EQ", test_eq()))
    results.append(("NE", test_ne()))
    results.append(("LT", test_lt()))
    results.append(("GT", test_gt()))
    results.append(("LE", test_le()))
    results.append(("GE", test_ge()))

    # Shifts
    results.append(("SHL", test_shl()))
    results.append(("SHR", test_shr()))

    # C4 compatibility
    results.append(("C4 Compat", test_c4_behavior()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
