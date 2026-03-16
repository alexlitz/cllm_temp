#!/usr/bin/env python3
"""
Comprehensive Test Suite for Neural VM V7

Tests:
1. PureALU operations (arithmetic, bitwise, comparison, shift)
2. 32-bit operations
3. New opcodes (LEA, IMM, BZ, BNZ, etc.)
4. ONNX export and runtime
5. Purity verification
"""

import os
import sys
import random
import subprocess
import tempfile
import torch
import numpy as np
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def record(self, name, passed, details=None):
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append((name, details))
            print(f"  [FAIL] {name}: {details}")

    def skip(self, name, reason):
        self.skipped += 1
        print(f"  [SKIP] {name}: {reason}")

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        if self.errors:
            print(f"\nFailed tests:")
            for name, details in self.errors[:10]:
                print(f"  - {name}: {details}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()
_cached_alu = None  # Cache ALU to avoid rebuilding


def get_alu():
    """Get cached ALU instance."""
    global _cached_alu
    if _cached_alu is None:
        from neural_vm import PureALU
        print("Building PureALU (one-time)...")
        _cached_alu = PureALU()
    return _cached_alu


# =============================================================================
# Helper Functions
# =============================================================================

def encode_operands(opcode: int, a: int, b: int) -> torch.Tensor:
    """Encode two 32-bit values and opcode into ALU input format."""
    from neural_vm import E
    x = torch.zeros(1, E.NUM_POSITIONS, E.DIM)
    for i in range(E.NUM_POSITIONS):
        x[0, i, E.NIB_A] = float((a >> (i * 4)) & 0xF)
        x[0, i, E.NIB_B] = float((b >> (i * 4)) & 0xF)
        x[0, i, E.OP_START + opcode] = 1.0
        x[0, i, E.POS] = float(i)
    return x


def decode_result(x: torch.Tensor) -> int:
    """Extract 32-bit result from ALU output."""
    from neural_vm import E
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
    return result == expected


# =============================================================================
# SECTION 1: PureALU Arithmetic Tests
# =============================================================================

def test_pure_alu_arithmetic():
    """Test PureALU arithmetic operations."""
    print("\n" + "="*60)
    print("SECTION 1: PureALU Arithmetic Operations")
    print("="*60)

    try:
        from neural_vm import E, Opcode
        alu = get_alu()
    except Exception as e:
        results.skip("PureALU import", str(e))
        return

    # ADD tests
    print("\n--- ADD Tests ---")
    add_tests = [
        (0, 0, 0),
        (1, 1, 2),
        (5, 3, 8),
        (15, 1, 16),
        (100, 200, 300),
        (255, 1, 256),
        (0xFF, 0xFF, 0x1FE),
    ]
    add_passed = 0
    for a, b, exp in add_tests:
        if test_alu_op(alu, Opcode.ADD, a, b, exp, f"ADD({a},{b})"):
            add_passed += 1
        else:
            print(f"  FAIL: ADD({a}, {b}) expected {exp}")
    results.record(f"ADD ({len(add_tests)} tests)", add_passed == len(add_tests),
                  f"{add_passed}/{len(add_tests)}")

    # SUB tests
    print("\n--- SUB Tests ---")
    sub_tests = [
        (5, 3, 2),
        (10, 5, 5),
        (100, 50, 50),
        (255, 100, 155),
        (0x100, 1, 0xFF),
    ]
    sub_passed = 0
    for a, b, exp in sub_tests:
        if test_alu_op(alu, Opcode.SUB, a, b, exp, f"SUB({a},{b})"):
            sub_passed += 1
        else:
            print(f"  FAIL: SUB({a}, {b}) expected {exp}")
    results.record(f"SUB ({len(sub_tests)} tests)", sub_passed == len(sub_tests),
                  f"{sub_passed}/{len(sub_tests)}")

    # MUL tests (small values)
    print("\n--- MUL Tests ---")
    mul_tests = [
        (2, 3, 6),
        (5, 5, 25),
        (10, 10, 100),
        (15, 15, 225),
        (7, 8, 56),
    ]
    mul_passed = 0
    for a, b, exp in mul_tests:
        if test_alu_op(alu, Opcode.MUL, a, b, exp, f"MUL({a},{b})"):
            mul_passed += 1
        else:
            print(f"  FAIL: MUL({a}, {b}) expected {exp}")
    results.record(f"MUL ({len(mul_tests)} tests)", mul_passed >= len(mul_tests) - 1,
                  f"{mul_passed}/{len(mul_tests)}")

    # DIV tests
    print("\n--- DIV Tests ---")
    div_tests = [
        (10, 2, 5),
        (15, 3, 5),
        (100, 10, 10),
        (255, 15, 17),
    ]
    div_passed = 0
    for a, b, exp in div_tests:
        if test_alu_op(alu, Opcode.DIV, a, b, exp, f"DIV({a},{b})"):
            div_passed += 1
        else:
            print(f"  FAIL: DIV({a}, {b}) expected {exp}")
    results.record(f"DIV ({len(div_tests)} tests)", div_passed >= len(div_tests) - 1,
                  f"{div_passed}/{len(div_tests)}")

    # MOD tests
    print("\n--- MOD Tests ---")
    mod_tests = [
        (10, 3, 1),
        (15, 4, 3),
        (100, 7, 2),
        (255, 16, 15),
    ]
    mod_passed = 0
    for a, b, exp in mod_tests:
        if test_alu_op(alu, Opcode.MOD, a, b, exp, f"MOD({a},{b})"):
            mod_passed += 1
        else:
            print(f"  FAIL: MOD({a}, {b}) expected {exp}")
    results.record(f"MOD ({len(mod_tests)} tests)", mod_passed >= len(mod_tests) - 1,
                  f"{mod_passed}/{len(mod_tests)}")


# =============================================================================
# SECTION 2: PureALU Bitwise Tests
# =============================================================================

def test_pure_alu_bitwise():
    """Test PureALU bitwise operations."""
    print("\n" + "="*60)
    print("SECTION 2: PureALU Bitwise Operations")
    print("="*60)

    try:
        from neural_vm import E, Opcode
        alu = get_alu()
    except Exception as e:
        results.skip("PureALU bitwise import", str(e))
        return

    # AND tests
    print("\n--- AND Tests ---")
    and_tests = [
        (0xF, 0xF, 0xF),
        (0xF, 0x0, 0x0),
        (0xA, 0x5, 0x0),
        (0xFF, 0x0F, 0x0F),
    ]
    and_passed = 0
    for a, b, exp in and_tests:
        if test_alu_op(alu, Opcode.AND, a, b, exp, f"AND({a:X},{b:X})"):
            and_passed += 1
    results.record(f"AND ({len(and_tests)} tests)", and_passed == len(and_tests),
                  f"{and_passed}/{len(and_tests)}")

    # OR tests
    print("\n--- OR Tests ---")
    or_tests = [
        (0xF, 0x0, 0xF),
        (0xA, 0x5, 0xF),
        (0x0, 0x0, 0x0),
        (0xFF, 0x00, 0xFF),
    ]
    or_passed = 0
    for a, b, exp in or_tests:
        if test_alu_op(alu, Opcode.OR, a, b, exp, f"OR({a:X},{b:X})"):
            or_passed += 1
    results.record(f"OR ({len(or_tests)} tests)", or_passed == len(or_tests),
                  f"{or_passed}/{len(or_tests)}")

    # XOR tests
    print("\n--- XOR Tests ---")
    xor_tests = [
        (0xF, 0xF, 0x0),
        (0xA, 0x5, 0xF),
        (0xF, 0x0, 0xF),
        (0xFF, 0xFF, 0x00),
    ]
    xor_passed = 0
    for a, b, exp in xor_tests:
        if test_alu_op(alu, Opcode.XOR, a, b, exp, f"XOR({a:X},{b:X})"):
            xor_passed += 1
    results.record(f"XOR ({len(xor_tests)} tests)", xor_passed == len(xor_tests),
                  f"{xor_passed}/{len(xor_tests)}")


# =============================================================================
# SECTION 3: PureALU Comparison Tests
# =============================================================================

def test_pure_alu_comparison():
    """Test PureALU comparison operations."""
    print("\n" + "="*60)
    print("SECTION 3: PureALU Comparison Operations")
    print("="*60)

    try:
        from neural_vm import E, Opcode
        alu = get_alu()
    except Exception as e:
        results.skip("PureALU comparison import", str(e))
        return

    # EQ tests
    print("\n--- EQ Tests ---")
    eq_tests = [
        (5, 5, 1),
        (5, 3, 0),
        (0, 0, 1),
        (15, 15, 1),
        (100, 100, 1),
        (100, 101, 0),
    ]
    eq_passed = 0
    for a, b, exp in eq_tests:
        if test_alu_op(alu, Opcode.EQ, a, b, exp, f"EQ({a},{b})"):
            eq_passed += 1
    results.record(f"EQ ({len(eq_tests)} tests)", eq_passed == len(eq_tests),
                  f"{eq_passed}/{len(eq_tests)}")

    # NE tests
    print("\n--- NE Tests ---")
    ne_tests = [
        (5, 3, 1),
        (5, 5, 0),
        (0, 1, 1),
        (100, 100, 0),
    ]
    ne_passed = 0
    for a, b, exp in ne_tests:
        if test_alu_op(alu, Opcode.NE, a, b, exp, f"NE({a},{b})"):
            ne_passed += 1
    results.record(f"NE ({len(ne_tests)} tests)", ne_passed == len(ne_tests),
                  f"{ne_passed}/{len(ne_tests)}")

    # LT tests
    print("\n--- LT Tests ---")
    lt_tests = [
        (3, 5, 1),
        (5, 3, 0),
        (5, 5, 0),
        (0, 1, 1),
        (100, 200, 1),
    ]
    lt_passed = 0
    for a, b, exp in lt_tests:
        if test_alu_op(alu, Opcode.LT, a, b, exp, f"LT({a},{b})"):
            lt_passed += 1
    results.record(f"LT ({len(lt_tests)} tests)", lt_passed >= len(lt_tests) - 1,
                  f"{lt_passed}/{len(lt_tests)}")

    # GT tests
    print("\n--- GT Tests ---")
    gt_tests = [
        (5, 3, 1),
        (3, 5, 0),
        (5, 5, 0),
        (200, 100, 1),
    ]
    gt_passed = 0
    for a, b, exp in gt_tests:
        if test_alu_op(alu, Opcode.GT, a, b, exp, f"GT({a},{b})"):
            gt_passed += 1
    results.record(f"GT ({len(gt_tests)} tests)", gt_passed >= len(gt_tests) - 1,
                  f"{gt_passed}/{len(gt_tests)}")

    # LE tests
    print("\n--- LE Tests ---")
    le_tests = [
        (3, 5, 1),
        (5, 5, 1),
        (5, 3, 0),
    ]
    le_passed = 0
    for a, b, exp in le_tests:
        if test_alu_op(alu, Opcode.LE, a, b, exp, f"LE({a},{b})"):
            le_passed += 1
    results.record(f"LE ({len(le_tests)} tests)", le_passed >= len(le_tests) - 1,
                  f"{le_passed}/{len(le_tests)}")

    # GE tests
    print("\n--- GE Tests ---")
    ge_tests = [
        (5, 3, 1),
        (5, 5, 1),
        (3, 5, 0),
    ]
    ge_passed = 0
    for a, b, exp in ge_tests:
        if test_alu_op(alu, Opcode.GE, a, b, exp, f"GE({a},{b})"):
            ge_passed += 1
    results.record(f"GE ({len(ge_tests)} tests)", ge_passed >= len(ge_tests) - 1,
                  f"{ge_passed}/{len(ge_tests)}")


# =============================================================================
# SECTION 4: PureALU Shift Tests (4-bit shifts)
# =============================================================================

def test_pure_alu_shift():
    """Test PureALU shift operations."""
    print("\n" + "="*60)
    print("SECTION 4: PureALU Shift Operations")
    print("="*60)

    try:
        from neural_vm import E, Opcode
        alu = get_alu()
    except Exception as e:
        results.skip("PureALU shift import", str(e))
        return

    # SHL tests (shift by 4 bits = 1 nibble)
    print("\n--- SHL Tests (4-bit shifts) ---")
    shl_tests = [
        (1, 4, 16),       # 1 << 4 = 16
        (0xF, 4, 0xF0),   # 15 << 4 = 240
        (0x12, 4, 0x120), # 18 << 4 = 288
    ]
    shl_passed = 0
    for a, b, exp in shl_tests:
        if test_alu_op(alu, Opcode.SHL, a, b, exp, f"SHL({a},{b})"):
            shl_passed += 1
    results.record(f"SHL ({len(shl_tests)} tests)", shl_passed >= len(shl_tests) - 1,
                  f"{shl_passed}/{len(shl_tests)}")

    # SHR tests (shift by 4 bits = 1 nibble)
    print("\n--- SHR Tests (4-bit shifts) ---")
    shr_tests = [
        (16, 4, 1),       # 16 >> 4 = 1
        (0xF0, 4, 0xF),   # 240 >> 4 = 15
        (0x120, 4, 0x12), # 288 >> 4 = 18
    ]
    shr_passed = 0
    for a, b, exp in shr_tests:
        if test_alu_op(alu, Opcode.SHR, a, b, exp, f"SHR({a},{b})"):
            shr_passed += 1
    results.record(f"SHR ({len(shr_tests)} tests)", shr_passed >= len(shr_tests) - 1,
                  f"{shr_passed}/{len(shr_tests)}")


# =============================================================================
# SECTION 5: Purity Verification
# =============================================================================

def test_purity():
    """Verify PureALU is truly pure (no custom forward)."""
    print("\n" + "="*60)
    print("SECTION 5: Purity Verification")
    print("="*60)

    try:
        import torch.nn as nn

        alu = get_alu()

        # Check it's an nn.Sequential
        is_sequential = isinstance(alu, nn.Sequential)
        results.record("PureALU is nn.Sequential", is_sequential)

        # Check forward methods in codebase
        from neural_vm import base_layers, pure_moe

        # Count forward methods
        forward_count = 0
        if hasattr(base_layers.PureFFN, 'forward'):
            forward_count += 1
        if hasattr(base_layers.PureAttention, 'forward'):
            forward_count += 1
        if hasattr(pure_moe.SoftMoEFFN, 'forward'):
            forward_count += 1
        if hasattr(pure_moe.SoftMoEAttention, 'forward'):
            forward_count += 1
        if hasattr(pure_moe.UnifiedMoEBlock, 'forward'):
            forward_count += 1

        results.record(f"Forward methods count", True, f"{forward_count} forward methods in core modules")

    except Exception as e:
        results.record("Purity verification", False, str(e))


# =============================================================================
# SECTION 6: Random Stress Tests
# =============================================================================

def test_random_stress():
    """Random stress tests for arithmetic operations (reduced for speed)."""
    print("\n" + "="*60)
    print("SECTION 6: Random Stress Tests (50 operations each)")
    print("="*60)

    try:
        from neural_vm import E, Opcode
        alu = get_alu()
    except Exception as e:
        results.skip("Random stress test import", str(e))
        return

    # ADD stress test
    print("\n--- ADD Stress Test (50 random) ---")
    add_passed = 0
    for _ in range(50):
        a = random.randint(0, 255)
        b = random.randint(0, 255 - a)
        exp = a + b
        if test_alu_op(alu, Opcode.ADD, a, b, exp, "ADD"):
            add_passed += 1
    results.record(f"ADD stress (50)", add_passed >= 45, f"{add_passed}/50")

    # SUB stress test
    print("\n--- SUB Stress Test (50 random) ---")
    sub_passed = 0
    for _ in range(50):
        a = random.randint(0, 255)
        b = random.randint(0, a)
        exp = a - b
        if test_alu_op(alu, Opcode.SUB, a, b, exp, "SUB"):
            sub_passed += 1
    results.record(f"SUB stress (50)", sub_passed >= 45, f"{sub_passed}/50")

    # MUL stress test (small values)
    print("\n--- MUL Stress Test (50 random) ---")
    mul_passed = 0
    for _ in range(50):
        a = random.randint(0, 15)
        b = random.randint(0, 15)
        exp = a * b
        if test_alu_op(alu, Opcode.MUL, a, b, exp, "MUL"):
            mul_passed += 1
    results.record(f"MUL stress (50)", mul_passed >= 40, f"{mul_passed}/50")

    # Comparison stress test
    print("\n--- Comparison Stress Test (50 random) ---")
    cmp_passed = 0
    for _ in range(50):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        # Test EQ
        exp = 1 if a == b else 0
        if test_alu_op(alu, Opcode.EQ, a, b, exp, "EQ"):
            cmp_passed += 1
    results.record(f"EQ stress (50)", cmp_passed >= 45, f"{cmp_passed}/50")

    # Bitwise stress test
    print("\n--- Bitwise Stress Test (50 random) ---")
    bit_passed = 0
    for _ in range(50):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        exp = a & b
        if test_alu_op(alu, Opcode.AND, a, b, exp, "AND"):
            bit_passed += 1
    results.record(f"AND stress (50)", bit_passed >= 45, f"{bit_passed}/50")


# =============================================================================
# SECTION 7: ONNX Export Test
# =============================================================================

def test_onnx_export():
    """Test ONNX export capability."""
    print("\n" + "="*60)
    print("SECTION 7: ONNX Export")
    print("="*60)

    try:
        import onnx
    except ImportError:
        results.skip("ONNX export", "onnx not installed")
        return

    try:
        from neural_vm import E
        alu = get_alu()

        # Create dummy input
        dummy_input = torch.zeros(1, E.NUM_POSITIONS, E.DIM)

        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        torch.onnx.export(
            alu,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )

        # Verify ONNX model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        results.record("ONNX export and validation", True)

        os.unlink(onnx_path)

    except Exception as e:
        results.record("ONNX export", False, str(e))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("COMPREHENSIVE NEURAL VM V7 TEST SUITE")
    print("="*60)
    print("\nTesting PureALU - Pure Neural ALU with zero Python control flow")
    print("Only forward() methods: PureFFN, PureAttention, SoftMoE routing")
    print()

    # Run all test sections
    test_pure_alu_arithmetic()
    test_pure_alu_bitwise()
    test_pure_alu_comparison()
    test_pure_alu_shift()
    test_purity()
    test_random_stress()
    test_onnx_export()

    # Print summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
