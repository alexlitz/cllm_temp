#!/usr/bin/env python3
"""
Run 1000+ tests with KV cache eviction enabled.
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
    """Group tests by execution step count for uniform batch processing."""
    by_steps = defaultdict(list)
    fast_vm = FastLogicalVM()

    for source, expected, desc in tests:
        try:
            bytecode, data = compile_c(source)
            # Run through Fast VM to count actual execution steps
            fast_vm.reset()
            fast_vm.load(bytecode, data)
            fast_vm.run()
            step_count = fast_vm.steps  # Number of VM steps executed
            by_steps[step_count].append((source, expected, desc, bytecode, data))
        except Exception as e:
            print(f"Warning: Failed to compile {desc}: {e}")

    return by_steps


def run_tests_with_kv_cache(batch_size=32, use_kv_cache=True, verbose=False):
    """Run tests with KV cache eviction."""

    print("=" * 70)
    print("C4 TRANSFORMER VM - 1000+ TESTS WITH KV CACHE EVICTION")
    print("=" * 70)
    print()

    # Generate tests
    all_tests = list(generate_test_programs())
    print(f"Total tests: {len(all_tests)}")
    print(f"Batch size: {batch_size}")
    print(f"KV cache: {'ENABLED' if use_kv_cache else 'DISABLED'}")
    print()

    # Group by execution step count
    print("Grouping tests by execution step count...")
    by_steps = group_tests_by_steps(all_tests)
    print(f"Found {len(by_steps)} unique step counts")

    # Show step distribution
    total_tests_grouped = sum(len(tests) for tests in by_steps.values())
    print(f"Successfully grouped: {total_tests_grouped}/{len(all_tests)} tests")
    print()

    # Create runner with KV cache
    print("Initializing batched runner...")
    runner = BatchedSpeculativeRunner(
        batch_size=batch_size,
        use_kv_cache=use_kv_cache,
        kv_cache_max_tokens=2048,
        use_sparse=True,
    )
    print()

    print("=" * 70)
    print("Running tests...")
    print("=" * 70)
    print()

    passed = 0
    failed = 0
    errors = 0
    total_processed = 0

    start_time = time.time()

    # Process each step count group
    for step_idx, (step_count, tests) in enumerate(sorted(by_steps.items())):
        # Reset KV cache between test groups to avoid state pollution
        if runner.kv_cache is not None:
            runner.kv_cache.reset()

        # Adaptive batch size based on step count to avoid OOM
        # Memory usage ∝ batch_size × step_count × 35 tokens
        adaptive_batch_size = min(batch_size, max(4, 20000 // (step_count * 35)))

        print(f"[Steps {step_count}] {len(tests)} tests (batch={adaptive_batch_size})...", end=" ", flush=True)

        # Process in batches
        for batch_start in range(0, len(tests), adaptive_batch_size):
            batch = tests[batch_start:batch_start + adaptive_batch_size]

            # Extract bytecodes and data
            bytecodes = [bc for _, _, _, bc, _ in batch]
            data_list = [d for _, _, _, _, d in batch]
            expected_results = [exp for _, exp, _, _, _ in batch]

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
                        if verbose:
                            _, _, desc, _, _ = batch[i]
                            print(f"\n  ✗ {desc}: got {result}, expected {expected}")

            except Exception as e:
                # Batch failed
                errors += len(batch)
                total_processed += len(batch)
                if verbose:
                    print(f"\n  ERROR: {str(e)[:100]}")

        print(f"Done ({total_processed}/{len(all_tests)})")

    elapsed = time.time() - start_time

    # Get cache statistics if enabled
    cache_stats = None
    if use_kv_cache and runner.kv_cache is not None:
        cache_stats = runner.kv_cache.get_total_stats()

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total tests: {len(all_tests)}")
    print(f"Processed: {total_processed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Speed: {total_processed/elapsed:.1f} tests/second")
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

    if passed == len(all_tests):
        print("✓ ALL TESTS PASSED!")
    elif failed > 0:
        print(f"✗ {failed} tests failed")

    if errors > 0:
        print(f"⚠ {errors} errors")

    return 0 if (failed == 0 and errors == 0) else 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--no-cache', action='store_true', help='Disable KV cache')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    sys.exit(run_tests_with_kv_cache(
        batch_size=args.batch_size,
        use_kv_cache=not args.no_cache,
        verbose=args.verbose
    ))
