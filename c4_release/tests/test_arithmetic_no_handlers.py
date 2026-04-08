#!/usr/bin/env python3
"""
Regression test for Layer 6 head allocation fix.

This test verifies that arithmetic operations (ADD, SUB, MUL, DIV, MOD)
work correctly WITHOUT Python handlers, using only the neural weights.

Background:
  The original code had `_set_layer6_relay_heads()` overwriting heads 2-3
  that were configured by `_set_layer6_attn()`. This broke JMP/JSR control
  flow operations. The fix was to disable `_set_layer6_relay_heads()`.

  This test ensures:
  1. Arithmetic operations work via Layer 8/9 FFN (AX_CARRY path preserved)
  2. The fix doesn't break basic arithmetic functionality

Related Issues:
  - docs/ACTUAL_FIX_AX_CARRY.md
  - docs/FIX_SUMMARY.md

Date: 2026-04-08
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Opcode
import torch


def test_arithmetic_operations():
    """Test basic arithmetic operations without Python handlers."""

    test_cases = [
        # (source_code, expected_result, description)
        ("return 42;", 42, "Literal constant"),
        ("return 10 + 32;", 42, "Addition"),
        ("return 50 - 8;", 42, "Subtraction"),
        ("return 6 * 7;", 42, "Multiplication"),
        ("return 84 / 2;", 42, "Division"),
        ("return 100 % 58;", 42, "Modulo"),

        # More complex expressions
        ("return 20 + 20 + 2;", 42, "Chained addition"),
        ("return 100 - 50 - 8;", 42, "Chained subtraction"),
        ("return 2 * 3 * 7;", 42, "Chained multiplication"),
        ("return (10 + 5) * 2 + 12;", 42, "Mixed operations"),

        # Edge cases
        ("return 0 + 42;", 42, "Addition with zero"),
        ("return 42 - 0;", 42, "Subtraction of zero"),
        ("return 42 * 1;", 42, "Multiplication by one"),
        ("return 42 / 1;", 42, "Division by one"),
    ]

    print("="*80)
    print("ARITHMETIC OPERATIONS TEST (without handlers)")
    print("="*80)
    print(f"\nTesting {len(test_cases)} cases...\n")

    passed = 0
    failed = 0
    errors = 0
    failed_tests = []

    # Create runner once and reuse it
    runner = AutoregressiveVMRunner()

    # Remove arithmetic operation handlers to force neural implementation
    handlers_removed = []
    for op in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD]:
        if op in runner._func_call_handlers:
            del runner._func_call_handlers[op]
            handlers_removed.append(op.name if hasattr(op, 'name') else str(op))

    print(f"Removed handlers: {', '.join(handlers_removed)}")
    print("Using neural implementation only\n")

    for i, (code, expected, desc) in enumerate(test_cases, 1):
        full_code = f"int main() {{ {code} }}"

        try:
            bytecode, data = compile_c(full_code)
            output, exit_code = runner.run(bytecode, max_steps=50)

            if output == expected and exit_code == 0:
                print(f"  [{i:2d}] ✓ PASS: {desc:30s} ({code})")
                passed += 1
            else:
                print(f"  [{i:2d}] ✗ FAIL: {desc:30s} (expected {expected}, got {output}, exit {exit_code})")
                failed += 1
                failed_tests.append((desc, expected, output, code))

        except Exception as e:
            print(f"  [{i:2d}] ✗ ERROR: {desc:30s} ({str(e)[:50]})")
            errors += 1
            failed_tests.append((desc, expected, f"ERROR: {e}", code))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  Passed: {passed}/{len(test_cases)}")
    print(f"  Failed: {failed}/{len(test_cases)}")
    print(f"  Errors: {errors}/{len(test_cases)}")

    if failed_tests:
        print("\nFailed tests:")
        for desc, expected, result, code in failed_tests:
            print(f"  - {desc}: expected {expected}, got {result}")
            print(f"    Code: {code}")

    success_rate = (passed / len(test_cases)) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return success if all tests passed
    return failed == 0 and errors == 0


if __name__ == '__main__':
    success = test_arithmetic_operations()
    sys.exit(0 if success else 1)
