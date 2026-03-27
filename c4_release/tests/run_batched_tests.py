#!/usr/bin/env python3
"""
Run test suite with GPU + batched validation.

This is MUCH faster than sequential validation:
- Groups tests by bytecode length
- Runs 32-64 programs in parallel on GPU
- Expected speedup: 320-1000x vs CPU sequential
"""

import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_suite_1000 import generate_test_programs
from src.baked_c4 import BakedC4Transformer
from src.speculator import ValidationError, FastLogicalVM
from src.compiler import compile_c


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


def run_batched_tests(batch_size=32, verbose=False):
    """Run tests with GPU batching."""

    print("=" * 60)
    print("C4 TRANSFORMER VM - BATCHED GPU VALIDATION")
    print("=" * 60)
    print()

    # Generate tests
    all_tests = list(generate_test_programs())
    print(f"Total tests: {len(all_tests)}")
    print(f"Batch size: {batch_size}")
    print()

    # Group by execution step count
    print("Grouping tests by execution step count...")
    by_steps = group_tests_by_steps(all_tests)
    print(f"Found {len(by_steps)} unique step counts")
    print()

    # Create VM with batching
    print("Initializing batched validation...")
    c4 = BakedC4Transformer(use_speculator=True)

    # Enable batching
    c4.speculator.use_batching = True
    c4.speculator.batch_size = batch_size

    print()
    print("=" * 60)
    print("Running tests with GPU + batching...")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    errors = 0
    total_processed = 0

    start_time = time.time()

    # Process each step count group
    for step_idx, (step_count, tests) in enumerate(sorted(by_steps.items())):
        # Adaptive batch size based on step count to avoid OOM
        # Memory usage ∝ batch_size × step_count × 35 tokens
        # Target: Keep total tokens per batch < ~20,000 (conservative for GPU memory)
        adaptive_batch_size = min(batch_size, max(8, 20000 // (step_count * 35)))

        print(f"[Steps {step_count}] {len(tests)} tests (batch={adaptive_batch_size})...", end=" ", flush=True)

        # Process in batches
        for batch_start in range(0, len(tests), adaptive_batch_size):
            batch = tests[batch_start:batch_start + adaptive_batch_size]

            # Extract bytecodes and data
            bytecodes = [bc for _, _, _, bc, _ in batch]
            data_list = [d for _, _, _, _, d in batch]
            expected_results = [exp for _, exp, _, _, _ in batch]
            descriptions = [desc for _, _, desc, _, _ in batch]

            try:
                # Run batch with validation
                results = c4.speculator.run_batch(bytecodes, data_list)

                # Check results
                for i, (result, expected, desc) in enumerate(zip(results, expected_results, descriptions)):
                    total_processed += 1
                    if result == expected:
                        passed += 1
                        if verbose:
                            print(f"  ✓ {desc}")
                    else:
                        failed += 1
                        if verbose:
                            print(f"  ✗ {desc}: got {result}, expected {expected}")

            except ValidationError as e:
                # Batch failed validation
                failed += len(batch)
                total_processed += len(batch)
                if verbose:
                    print(f"\n  Batch validation failed:")
                    print(f"    {str(e).split(chr(10))[0]}")
                break  # Stop on first validation failure

            except Exception as e:
                errors += len(batch)
                total_processed += len(batch)
                print(f"\n  ERROR in batch: {type(e).__name__}: {e}")
                break

        # Progress
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"Done ({total_processed}/{len(all_tests)}, {rate:.1f} tests/sec)")

        if failed > 0 or errors > 0:
            break  # Stop on first failure

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total tests: {len(all_tests)}")
    print(f"Processed: {total_processed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
    print(f"Speed: {total_processed/elapsed:.1f} tests/second")
    print()

    # Show validation stats
    stats = c4.speculator.get_stats()
    print("Validation statistics:")
    print(f"  Validations: {stats['validations']}")
    print(f"  Mismatches: {stats['mismatches']}")
    print(f"  Match rate: {stats['match_rate']*100:.1f}%")
    print()

    if failed == 0 and errors == 0:
        print("✓ ALL TESTS PASSED!")
        print("  Neural VM matches Fast VM perfectly!")
    else:
        print(f"✗ {failed} tests failed, {errors} errors")
        print("  This is expected if neural VM is not fully working")

    return 0 if failed == 0 and errors == 0 else 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run tests with GPU batching')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    sys.exit(run_batched_tests(
        batch_size=args.batch_size,
        verbose=args.verbose
    ))
