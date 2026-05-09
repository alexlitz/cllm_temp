#!/usr/bin/env python3
"""
Run the 1000+ test suite against the C4 Transformer VM (pure neural path).

Usage:
    python tests/run_1000_tests.py          # Run all tests via AutoregressiveVMRunner
    python tests/run_1000_tests.py --quick  # Run first 100 tests
    python tests/run_1000_tests.py --limit N  # Run first N tests
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_suite_1000 import generate_test_programs, get_quick_tests, CATEGORY_COUNTS
from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c


def _build_runner():
    runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
    runner._func_call_handlers = {}
    runner._syscall_handlers = {}
    return runner


def run_tests(tests, verbose=False, max_steps=2000):
    """Run tests against the pure-neural autoregressive runner."""

    runner = _build_runner()
    vm_name = "AutoregressiveVMRunner(pure_neural=True)"

    passed = 0
    failed = 0
    errors = 0
    failed_tests = []

    start_time = time.time()

    for i, (source, expected, desc) in enumerate(tests):
        try:
            bytecode, data = compile_c(source)
            runner._memory = {}
            runner._mem_history = {}
            runner._mem_access_order = []
            _, result = runner.run(bytecode, data or b"", max_steps=max_steps)

            if result == expected:
                passed += 1
                if verbose:
                    print(f"  [{i+1:4d}] PASS: {desc}")
            else:
                failed += 1
                failed_tests.append((desc, expected, result))
                if verbose:
                    print(f"  [{i+1:4d}] FAIL: {desc} (expected {expected}, got {result})")
        except Exception as e:
            errors += 1
            failed_tests.append((desc, expected, f"ERROR: {e}"))
            if verbose:
                print(f"  [{i+1:4d}] ERROR: {desc} - {e}")

        # Progress indicator
        if not verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(tests)} ({elapsed:.1f}s, "
                  f"pass={passed}, fail={failed}, err={errors})")

    elapsed = time.time() - start_time

    return {
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'total': len(tests),
        'elapsed': elapsed,
        'failed_tests': failed_tests,
        'vm_name': vm_name,
    }


def main():
    parser = argparse.ArgumentParser(description='Run 1000+ test suite (pure neural)')
    parser.add_argument('--quick', action='store_true', help='Run first 100 tests only')
    parser.add_argument('--limit', type=int, default=None, help='Run only first N tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show each test result')
    parser.add_argument('--category', type=str, help='Run only tests from specific category')
    parser.add_argument('--max-steps', type=int, default=2000, help='Max VM steps per test')
    args = parser.parse_args()

    print("=" * 60)
    print("C4 TRANSFORMER VM - 1000+ TEST SUITE")
    print("=" * 60)
    print()

    # Generate tests
    if args.quick:
        tests = get_quick_tests()
        print(f"Running QUICK test suite ({len(tests)} tests)")
    else:
        tests = generate_test_programs()
        print(f"Running FULL test suite ({len(tests)} tests)")

    # Filter by category if specified
    if args.category:
        original_count = len(tests)
        tests = [(s, e, d) for s, e, d in tests if args.category.lower() in d.lower()]
        print(f"Filtered to category '{args.category}': {len(tests)} tests (from {original_count})")

    if args.limit is not None:
        tests = tests[:args.limit]
        print(f"Limited to first {len(tests)} tests")

    print()
    print("Category breakdown:")
    for cat, count in CATEGORY_COUNTS.items():
        print(f"  {cat}: {count}")
    print()

    print("Using AutoregressiveVMRunner(pure_neural=True) - pure neural path")
    print("-" * 60)

    results = run_tests(tests, verbose=args.verbose, max_steps=args.max_steps)

    # Report results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  VM: {results['vm_name']}")
    print(f"  Total tests: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Success rate: {results['passed']/results['total']*100:.1f}%")
    print(f"  Time: {results['elapsed']:.2f}s")
    print(f"  Tests/sec: {results['total']/results['elapsed']:.1f}")

    # Show failed tests
    if results['failed_tests'] and not args.verbose:
        print()
        print("Failed tests (first 10):")
        for desc, expected, got in results['failed_tests'][:10]:
            print(f"  - {desc}: expected {expected}, got {got}")
        if len(results['failed_tests']) > 10:
            print(f"  ... and {len(results['failed_tests']) - 10} more")

    print()
    if results['failed'] == 0 and results['errors'] == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"SOME TESTS FAILED ({results['failed']} failures, {results['errors']} errors)")

    return 0 if results['failed'] == 0 and results['errors'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
