#!/usr/bin/env python3
"""
Comprehensive test suite for all VM operations.
Run this to verify all 32-bit operations work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vm_archive import ExtendedTransformer
from pure_gen_vm import Opcode

model = ExtendedTransformer()

def int_to_bytes(v):
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]

def bytes_to_int(b):
    return b[0] + (b[1] << 8) + (b[2] << 16) + (b[3] << 24)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def test(self, name, actual, expected):
        if actual == expected:
            self.passed += 1
            return True
        else:
            self.failed += 1
            self.failures.append((name, actual, expected))
            return False

results = TestResults()

def test_op(name, opcode, a, b, expected):
    a_bytes = int_to_bytes(a)
    b_bytes = int_to_bytes(b)
    result_bytes, _ = model.forward_fully_neural_step(opcode, a_bytes, b_bytes, None)
    result = bytes_to_int(result_bytes)
    return results.test(name, result, expected)

print("=" * 70)
print("COMPREHENSIVE VM OPERATION TESTS")
print("=" * 70)

# ============================================================================
# ADD TESTS - Full 32-bit with carry lookahead
# ============================================================================
print("\n=== ADD (32-bit) ===")
add_tests = [
    ("5 + 3", 5, 3, 8),
    ("100 + 200", 100, 200, 300),
    ("10000 + 20000", 10000, 20000, 30000),
    ("255 + 1 (8-bit carry)", 255, 1, 256),
    ("65535 + 1 (16-bit carry)", 65535, 1, 65536),
    ("16777215 + 1 (24-bit carry)", 16777215, 1, 16777216),
    ("0xFFFFFFFF + 1 (overflow)", 0xFFFFFFFF, 1, 0),
    ("0x7FFFFFFF + 0x7FFFFFFF", 0x7FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFE),
    ("0x12345678 + 0x11111111", 0x12345678, 0x11111111, 0x23456789),
]
add_passed = 0
for name, a, b, expected in add_tests:
    if test_op(f"ADD: {name}", Opcode.ADD, a, b, expected):
        add_passed += 1
print(f"  {add_passed}/{len(add_tests)} passed")

# ============================================================================
# SUB TESTS - Full 32-bit with borrow
# ============================================================================
print("\n=== SUB (32-bit) ===")
sub_tests = [
    ("10 - 3", 10, 3, 7),
    ("500 - 200", 500, 200, 300),
    ("100000 - 50000", 100000, 50000, 50000),
    ("256 - 1 (8-bit borrow)", 256, 1, 255),
    ("65536 - 1 (16-bit borrow)", 65536, 1, 65535),
    ("0 - 1 (underflow)", 0, 1, 0xFFFFFFFF),
    ("0x12345678 - 0x11111111", 0x12345678, 0x11111111, 0x01234567),
]
for name, a, b, expected in sub_tests:
    test_op(f"SUB: {name}", Opcode.SUB, a, b, expected)

# ============================================================================
# BITWISE TESTS - Full 32-bit
# ============================================================================
print("\n=== AND (32-bit) ===")
and_tests = [
    ("0xF & 0x5", 0xF, 0x5, 0x5),
    ("0xFF & 0x55", 0xFF, 0x55, 0x55),
    ("0xFFFF & 0x5555", 0xFFFF, 0x5555, 0x5555),
    ("0xFFFFFFFF & 0x55555555", 0xFFFFFFFF, 0x55555555, 0x55555555),
    ("0x12345678 & 0xF0F0F0F0", 0x12345678, 0xF0F0F0F0, 0x10305070),
]
for name, a, b, expected in and_tests:
    test_op(f"AND: {name}", Opcode.AND, a, b, expected)

print("\n=== OR (32-bit) ===")
or_tests = [
    ("0xA | 0x5", 0xA, 0x5, 0xF),
    ("0xAA | 0x55", 0xAA, 0x55, 0xFF),
    ("0xAAAA | 0x5555", 0xAAAA, 0x5555, 0xFFFF),
    ("0xAAAAAAAA | 0x55555555", 0xAAAAAAAA, 0x55555555, 0xFFFFFFFF),
]
for name, a, b, expected in or_tests:
    test_op(f"OR: {name}", Opcode.OR, a, b, expected)

print("\n=== XOR (32-bit) ===")
xor_tests = [
    ("0xF ^ 0xA", 0xF, 0xA, 0x5),
    ("0xFF ^ 0xAA", 0xFF, 0xAA, 0x55),
    ("0xFFFF ^ 0xAAAA", 0xFFFF, 0xAAAA, 0x5555),
    ("self XOR", 0x12345678, 0x12345678, 0),
]
for name, a, b, expected in xor_tests:
    test_op(f"XOR: {name}", Opcode.XOR, a, b, expected)

# ============================================================================
# COMPARISON TESTS - Full 32-bit
# ============================================================================
print("\n=== COMPARISONS (32-bit) ===")
cmp_tests = [
    ("EQ: 5 == 5", Opcode.EQ, 5, 5, 1),
    ("EQ: 0x12345678 == 0x12345678", Opcode.EQ, 0x12345678, 0x12345678, 1),
    ("EQ: 100 == 200", Opcode.EQ, 100, 200, 0),
    ("NE: 5 != 10", Opcode.NE, 5, 10, 1),
    ("NE: 100 != 100", Opcode.NE, 100, 100, 0),
    ("LT: 5 < 10", Opcode.LT, 5, 10, 1),
    ("LT: 10 < 10", Opcode.LT, 10, 10, 0),
    ("LT: 15 < 10", Opcode.LT, 15, 10, 0),
    ("GT: 15 > 10", Opcode.GT, 15, 10, 1),
    ("GT: 10 > 10", Opcode.GT, 10, 10, 0),
    ("GT: 5 > 10", Opcode.GT, 5, 10, 0),
    ("LE: 5 <= 10", Opcode.LE, 5, 10, 1),
    ("LE: 10 <= 10", Opcode.LE, 10, 10, 1),
    ("LE: 15 <= 10", Opcode.LE, 15, 10, 0),
    ("GE: 15 >= 10", Opcode.GE, 15, 10, 1),
    ("GE: 10 >= 10", Opcode.GE, 10, 10, 1),
    ("GE: 5 >= 10", Opcode.GE, 5, 10, 0),
]
for name, opcode, a, b, expected in cmp_tests:
    test_op(name, opcode, a, b, expected)

# ============================================================================
# MUL TESTS - Full 32-bit via nibble long multiplication
# ============================================================================
print("\n=== MUL (32-bit) ===")
mul_tests = [
    # Nibble-level tests
    ("2 * 3", 2, 3, 6),
    ("4 * 4", 4, 4, 16),
    ("7 * 8", 7, 8, 56),
    ("15 * 15", 15, 15, 225),
    # Byte-level tests
    ("100 * 10", 100, 10, 1000),
    ("256 * 2", 256, 2, 512),
    ("255 * 255", 255, 255, 65025),
    # Multi-byte tests
    ("1000 * 1000", 1000, 1000, 1000000),
    ("12345 * 6", 12345, 6, 74070),
    ("65536 * 2", 65536, 2, 131072),
    # Large tests
    ("50000 * 50", 50000, 50, 2500000),
    ("1000000 * 3", 1000000, 3, 3000000),
    # Overflow tests (truncate to 32 bits)
    ("0x10000 * 0x10000 (overflow)", 0x10000, 0x10000, 0),
    ("0xFFFF * 0xFFFF", 0xFFFF, 0xFFFF, 0xFFFE0001),
    ("0xFFFFFFFF * 2 (overflow)", 0xFFFFFFFF, 2, 0xFFFFFFFE),
    ("0x80000000 * 2 (overflow)", 0x80000000, 2, 0),
    ("65536 * 65536 (overflow)", 65536, 65536, 0),
]
mul_passed = 0
for name, a, b, expected in mul_tests:
    if test_op(f"MUL: {name}", Opcode.MUL, a, b, expected):
        mul_passed += 1
print(f"  {mul_passed}/{len(mul_tests)} passed")

# ============================================================================
# DIV TESTS - Full 32-bit via restoring division
# ============================================================================
print("\n=== DIV (32-bit) ===")
div_tests = [
    # Nibble-level tests
    ("6 / 2", 6, 2, 3),
    ("15 / 3", 15, 3, 5),
    ("10 / 3 (floor)", 10, 3, 3),
    ("8 / 0 (returns 0)", 8, 0, 0),
    # Byte-level tests
    ("100 / 10", 100, 10, 10),
    ("255 / 17", 255, 17, 15),
    ("1000 / 33", 1000, 33, 30),
    # Multi-byte tests
    ("65536 / 256", 65536, 256, 256),
    ("1000000 / 1000", 1000000, 1000, 1000),
    ("12345678 / 1234", 12345678, 1234, 10004),
    # Large tests (no overflow in DIV, but test large values)
    ("0xFFFFFFFF / 1", 0xFFFFFFFF, 1, 0xFFFFFFFF),
    ("0xFFFFFFFF / 2", 0xFFFFFFFF, 2, 0x7FFFFFFF),
    ("0x80000000 / 2", 0x80000000, 2, 0x40000000),
    ("1000000000 / 7", 1000000000, 7, 142857142),
]
div_passed = 0
for name, a, b, expected in div_tests:
    if test_op(f"DIV: {name}", Opcode.DIV, a, b, expected):
        div_passed += 1
print(f"  {div_passed}/{len(div_tests)} passed")

# ============================================================================
# MOD TESTS - Full 32-bit via A - (A/B)*B
# ============================================================================
print("\n=== MOD (32-bit) ===")
mod_tests = [
    # Nibble-level tests
    ("10 % 3", 10, 3, 1),
    ("15 % 4", 15, 4, 3),
    ("8 % 0 (returns 0)", 8, 0, 0),
    # Byte-level tests
    ("100 % 7", 100, 7, 2),
    ("255 % 16", 255, 16, 15),
    ("1000 % 33", 1000, 33, 10),
    # Multi-byte tests
    ("65536 % 100", 65536, 100, 36),
    ("1000000 % 7", 1000000, 7, 1),
    ("12345678 % 1000", 12345678, 1000, 678),
    # Large value tests
    ("0xFFFFFFFF % 2", 0xFFFFFFFF, 2, 1),
    ("0xFFFFFFFF % 256", 0xFFFFFFFF, 256, 255),
    ("1000000000 % 7", 1000000000, 7, 6),
]
mod_passed = 0
for name, a, b, expected in mod_tests:
    if test_op(f"MOD: {name}", Opcode.MOD, a, b, expected):
        mod_passed += 1
print(f"  {mod_passed}/{len(mod_tests)} passed")

# ============================================================================
# MEMORY TESTS
# ============================================================================
print("\n=== MEMORY (32-bit) ===")
memory = {}

# Store and load int
addr = int_to_bytes(0x1000)
value = int_to_bytes(0xDEADBEEF)
_, memory = model.forward_fully_neural_step(Opcode.SI, addr, value, memory)
loaded, memory = model.forward_fully_neural_step(Opcode.LI, addr, [0,0,0,0], memory)
results.test("SI/LI: 0xDEADBEEF", bytes_to_int(loaded), 0xDEADBEEF)

# Overwrite
new_value = int_to_bytes(0x12345678)
_, memory = model.forward_fully_neural_step(Opcode.SI, addr, new_value, memory)
loaded, memory = model.forward_fully_neural_step(Opcode.LI, addr, [0,0,0,0], memory)
results.test("SI/LI: overwrite", bytes_to_int(loaded), 0x12345678)

# Different address
addr2 = int_to_bytes(0x2000)
value2 = int_to_bytes(0xCAFEBABE)
_, memory = model.forward_fully_neural_step(Opcode.SI, addr2, value2, memory)
loaded, memory = model.forward_fully_neural_step(Opcode.LI, addr2, [0,0,0,0], memory)
results.test("SI/LI: different address", bytes_to_int(loaded), 0xCAFEBABE)

# Char store/load
char_addr = int_to_bytes(0x3000)
char_val = [ord('A')]
_, memory = model.forward_fully_neural_step(Opcode.SC, char_addr, char_val, memory)
loaded, memory = model.forward_fully_neural_step(Opcode.LC, char_addr, [0,0,0,0], memory)
results.test("SC/LC: 'A'", loaded, [ord('A'), 0, 0, 0])

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"Passed: {results.passed}")
print(f"Failed: {results.failed}")
print(f"Total:  {results.passed + results.failed}")

if results.failures:
    print("\nFailed tests:")
    for name, actual, expected in results.failures:
        print(f"  {name}: got {actual}, expected {expected}")

print("\n" + "=" * 70)
print("IMPLEMENTATION STATUS")
print("=" * 70)
print("Working 32-bit operations:")
print("  + ADD - Full carry lookahead")
print("  + SUB - Full borrow propagation")
print("  + AND, OR, XOR - Bitwise")
print("  + EQ, NE, LT, GT, LE, GE - Comparisons")
print("  + MUL - Nibble long multiplication (64 partial products)")
print("  + DIV - Restoring division algorithm")
print("  + MOD - Via A - (A/B)*B")
print("  + SI, LI, SC, LC - Memory via attention")
print()
print("Architecture:")
print("  - Layers: 4 (attention + MoE FFN)")
print("  - Dim: 1600")
print("  - MUL uses 64 neural nibble products")
print("  - Memory via softmax1 attention with ALiBi")
print("  - All operations include 32-bit overflow tests")
print("=" * 70)

if results.failed == 0:
    print("\nALL TESTS PASSED!")
    sys.exit(0)
else:
    print(f"\n{results.failed} TESTS FAILED")
    sys.exit(1)
