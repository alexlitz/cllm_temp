#!/usr/bin/env python3
"""
Comprehensive Test Runner

Orchestrates testing across all backends and test categories.
Provides summary comparison showing consistency across implementations.

Usage:
    python tests/run_comprehensive_tests.py --all-modes      # All backends
    python tests/run_comprehensive_tests.py --backends       # Just backends
    python tests/run_comprehensive_tests.py --features       # Just features
    python tests/run_comprehensive_tests.py --validation     # KV cache validation
"""

import sys
import os
import argparse
import time
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_suite_1000 import generate_test_programs, CATEGORY_COUNTS
from tests.vm_runners import create_runner


def run_backend_tests(mode: str, test_count: int = 100) -> Dict:
    """Run tests on a single backend."""
    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} backend")
    print(f"{'='*60}\n")

    tests = generate_test_programs()[:test_count]

    try:
        runner = create_runner(mode)
    except ValueError as e:
        print(f"ERROR: {e}")
        return {'mode': mode, 'available': False, 'error': str(e)}

    if not runner.setup():
        print(f"ERROR: Failed to initialize {mode} backend")
        return {'mode': mode, 'available': False, 'error': 'Setup failed'}

    passed = 0
    failed = 0
    errors = 0
    start_time = time.time()

    for i, (source, expected, desc) in enumerate(tests):
        try:
            result, _ = runner.run_program(source, max_steps=10000)
            if result == expected:
                passed += 1
            else:
                failed += 1
        except Exception:
            errors += 1

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{test_count}")

    elapsed = time.time() - start_time
    runner.cleanup()

    return {
        'mode': mode,
        'available': True,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'total': test_count,
        'elapsed': elapsed,
    }


def run_all_backends(test_count: int = 100):
    """Run tests across all available backends."""
    print("\n" + "="*60)
    print("COMPREHENSIVE BACKEND TESTING")
    print("="*60)
    print(f"\nRunning {test_count} tests across all backends...")
    print()

    modes = ['fast', 'transformer', 'onnx', 'c-runtime', 'bundler']
    results = []

    for mode in modes:
        result = run_backend_tests(mode, test_count)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("BACKEND COMPARISON")
    print("="*70)
    print()
    print(f"{'Backend':<15} {'Status':<12} {'Passed':<10} {'Failed':<10} {'Time (s)':<10}")
    print("-"*70)

    for result in results:
        if not result['available']:
            print(f"{result['mode']:<15} {'UNAVAILABLE':<12} {'-':<10} {'-':<10} {'-':<10}")
        else:
            status = "✓ OK" if result['failed'] == 0 and result['errors'] == 0 else "✗ FAIL"
            print(f"{result['mode']:<15} {status:<12} {result['passed']:<10} "
                  f"{result['failed']:<10} {result['elapsed']:<10.2f}")

    # Check consistency
    print()
    available_results = [r for r in results if r['available']]
    if len(available_results) > 1:
        passed_counts = [r['passed'] for r in available_results]
        if len(set(passed_counts)) == 1:
            print("✓ All available backends produce consistent results")
        else:
            print("✗ WARNING: Backends produce different results!")
            print(f"  Passed counts: {dict(zip([r['mode'] for r in available_results], passed_counts))}")

    return results


def run_category_tests(category: str, test_count: int = 50):
    """Run tests from a specific category."""
    print(f"\n{'='*60}")
    print(f"Testing Category: {category.upper()}")
    print(f"{'='*60}\n")

    all_tests = generate_test_programs()
    category_tests = [(s, e, d) for s, e, d in all_tests if category in d.lower()][:test_count]

    print(f"Found {len(category_tests)} tests matching '{category}'")

    if not category_tests:
        print("No tests found for this category")
        return

    runner = create_runner('fast')
    if not runner.setup():
        print("ERROR: Failed to initialize runner")
        return

    passed = 0
    failed = 0

    for i, (source, expected, desc) in enumerate(category_tests):
        try:
            result, _ = runner.run_program(source, max_steps=10000)
            if result == expected:
                passed += 1
                print(f"  ✓ {desc}")
            else:
                failed += 1
                print(f"  ✗ {desc} (expected {expected}, got {result})")
        except Exception as e:
            failed += 1
            print(f"  ✗ {desc} (ERROR: {e})")

    runner.cleanup()

    print(f"\nResults: {passed}/{len(category_tests)} passed ({passed/len(category_tests)*100:.1f}%)")


def run_validation_tests():
    """Run validation tests (KV cache, etc.)."""
    print("\n" + "="*60)
    print("VALIDATION TESTS")
    print("="*60)
    print()

    tests_to_run = [
        ('KV Cache Correctness', 'tests/test_kv_cache_correctness.py'),
        ('KV Cache Performance', 'tests/benchmark_kv_cache.py'),
    ]

    for name, test_path in tests_to_run:
        print(f"\nRunning: {name}")
        print("-"*40)

        full_path = os.path.join(os.path.dirname(__file__), '..', test_path)
        if not os.path.exists(full_path):
            print(f"  ✗ Test file not found: {test_path}")
            continue

        print(f"  Note: Run manually: python {test_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive test runner')
    parser.add_argument('--all-modes', action='store_true',
                        help='Test all backends (fast, transformer, onnx, c, bundler)')
    parser.add_argument('--backends', action='store_true',
                        help='Test all backends (alias for --all-modes)')
    parser.add_argument('--features', action='store_true',
                        help='Test new features (long-context, I/O, tool-calling)')
    parser.add_argument('--validation', action='store_true',
                        help='Run validation tests (KV cache)')
    parser.add_argument('--category', type=str,
                        help='Run tests from specific category')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of tests to run (default: 100)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("C4 COMPREHENSIVE TEST SUITE")
    print("="*60)
    print()
    print(f"Total test suite size: {sum(CATEGORY_COUNTS.values())} tests")
    print()
    print("Categories:")
    for cat, count in CATEGORY_COUNTS.items():
        print(f"  {cat}: {count}")
    print()

    # Run requested tests
    if args.all_modes or args.backends:
        run_all_backends(args.count)

    elif args.features:
        print("Testing new features...")
        for category in ['long_context', 'conversational_io', 'tool_calling']:
            run_category_tests(category, 50)

    elif args.validation:
        run_validation_tests()

    elif args.category:
        run_category_tests(args.category, args.count)

    else:
        # Default: show help
        print("Usage:")
        print("  --all-modes    Test all backends")
        print("  --features     Test new features")
        print("  --validation   Run validation tests")
        print("  --category X   Test specific category")
        print()
        print("Examples:")
        print("  python tests/run_comprehensive_tests.py --all-modes --count 50")
        print("  python tests/run_comprehensive_tests.py --category long_context")
        print("  python tests/run_comprehensive_tests.py --features")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
