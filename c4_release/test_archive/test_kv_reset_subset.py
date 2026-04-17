#!/usr/bin/env python3
"""
Quick subset test to verify KV cache reset fix on diverse program sizes.
"""

import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from src.speculator import FastLogicalVM
from neural_vm.batch_runner import BatchedSpeculativeRunner


def group_tests_by_steps(tests):
    """Group tests by execution step count."""
    by_steps = defaultdict(list)
    fast_vm = FastLogicalVM()

    for source, expected, desc in tests:
        try:
            bytecode, data = compile_c(source)
            fast_vm.reset()
            fast_vm.load(bytecode, data)
            fast_vm.run()
            step_count = fast_vm.steps
            by_steps[step_count].append((source, expected, desc, bytecode, data))
        except Exception as e:
            pass

    return by_steps


def run_subset_test():
    """Run a quick subset of tests to verify KV cache reset fix."""

    print("=" * 70)
    print("KV CACHE RESET VERIFICATION - SUBSET TEST")
    print("=" * 70)
    print()

    # Generate all tests, then sample from different step counts
    all_tests = list(generate_test_programs())
    print(f"Total available tests: {len(all_tests)}")

    # Group by steps
    print("Grouping tests by execution step count...")
    by_steps = group_tests_by_steps(all_tests)
    print(f"Found {len(by_steps)} unique step counts")

    # Sample from different step groups to test variety
    step_counts_sorted = sorted(by_steps.keys())

    # Pick representative step counts: min, low, med, high, max
    selected_steps = [
        step_counts_sorted[0],                    # minimum
        step_counts_sorted[len(step_counts_sorted)//4],     # 25th percentile
        step_counts_sorted[len(step_counts_sorted)//2],     # median
        step_counts_sorted[3*len(step_counts_sorted)//4],   # 75th percentile
        step_counts_sorted[-1],                   # maximum
    ]

    print(f"\nSelected step counts for testing: {selected_steps}")
    print()

    # Create runner with KV cache
    print("Initializing batched runner with KV cache...")
    runner = BatchedSpeculativeRunner(
        batch_size=32,
        use_kv_cache=True,
        kv_cache_max_tokens=2048,
        use_sparse=True,
    )
    print()

    print("=" * 70)
    print("Running subset tests...")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    errors = 0
    total_processed = 0

    start_time = time.time()

    for step_count in selected_steps:
        tests = by_steps[step_count]

        # Take first 5 tests from each group
        tests = tests[:5]

        # Reset KV cache between different step count groups
        if runner.kv_cache is not None:
            runner.kv_cache.reset()

        print(f"[Steps {step_count}] {len(tests)} tests...", end=" ", flush=True)

        # Extract bytecodes and data
        bytecodes = [bc for _, _, _, bc, _ in tests]
        data_list = [d for _, _, _, _, d in tests]
        expected_results = [exp for _, exp, _, _, _ in tests]

        try:
            # Run batch with KV cache
            results = runner.run_batch(bytecodes, data_list, max_steps=10000)

            # Check results
            for i, ((out, result), expected) in enumerate(zip(results, expected_results)):
                total_processed += 1
                if result == expected:
                    passed += 1
                else:
                    failed += 1

            print(f"Done ({passed}/{total_processed} passed so far)")

        except Exception as e:
            errors += len(tests)
            total_processed += len(tests)
            print(f"ERROR: {str(e)[:80]}")

    elapsed = time.time() - start_time

    # Get cache statistics
    cache_stats = None
    if runner.kv_cache is not None:
        cache_stats = runner.kv_cache.get_total_stats()

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total tests: {total_processed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.2f}s")
    print()

    if cache_stats:
        print("KV Cache Statistics:")
        print(f"  Tokens cached: {cache_stats['tokens_cached']:,}")
        print(f"  Tokens evicted: {cache_stats['tokens_evicted']:,}")
        print(f"  Cache hits: {cache_stats['cache_hits']:,}")
        print(f"  Current total size: {cache_stats['current_total_size']:,}")
        print()

    runner_stats = runner.get_stats()
    print("Validation statistics:")
    print(f"  Validations: {runner_stats['validations']}")
    print(f"  Mismatches: {runner_stats['mismatches']}")
    print(f"  Match rate: {runner_stats['match_rate']:.1%}")
    print()

    if errors == 0 and failed == 0:
        print("✓ SUCCESS: KV cache reset fix verified across diverse program sizes!")
        return 0
    elif errors > 0:
        print(f"✗ FAILURE: {errors} errors occurred")
        return 1
    else:
        print(f"⚠ WARNING: {failed} tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_subset_test())
