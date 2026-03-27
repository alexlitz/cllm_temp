#!/usr/bin/env python3
"""
Fast test that compaction/sparsification don't change results.

Strategy: Use a single model, run tests, then optimize it, run tests again.
This is much faster than creating new models for each configuration.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def test_consistency():
    """Test consistency across different optimization configurations."""
    print("=" * 70)
    print("OPTIMIZATION CONSISTENCY TEST (Fast Version)")
    print("=" * 70)
    print()

    # Test programs
    test_programs = [
        ("int main() { return 42; }", 42, "simple"),
        ("int main() { return 5 + 7; }", 12, "addition"),
        ("int main() { return 10 * 4; }", 40, "multiplication"),
        ("int main() { return (3 + 4) * 5; }", 35, "expression"),
    ]

    # Compile programs
    print(f"Compiling {len(test_programs)} test programs...")
    bytecodes = []
    data_list = []
    expected_results = []

    for source, expected, desc in test_programs:
        bytecode, data = compile_c(source)
        bytecodes.append(bytecode)
        data_list.append(data)
        expected_results.append(expected)
        print(f"  ✓ {desc}")

    print()

    # Create runner - start with NO optimizations
    print("Creating batched runner (baseline - no optimizations)...")
    runner = BatchedSpeculativeRunner(
        batch_size=10,
        use_kv_cache=False,
        use_sparse=False,  # Start without sparse
    )
    print()

    # Test 1: Baseline (no optimizations)
    print("=" * 70)
    print("TEST 1: Baseline (no optimizations)")
    print("=" * 70)

    results_baseline = runner.run_batch(bytecodes, data_list, max_steps=1000)
    baseline_numeric = [res for (out, res) in results_baseline]

    passed = sum(1 for res, exp in zip(baseline_numeric, expected_results) if res == exp)
    print(f"Results: {passed}/{len(bytecodes)} passed")
    for i, (res, exp, desc) in enumerate(zip(baseline_numeric, expected_results, [d for _, _, d in test_programs])):
        status = "✓" if res == exp else "✗"
        print(f"  {status} {desc}: got {res}, expected {exp}")
    print()

    # Test 2: Apply compaction
    print("=" * 70)
    print("TEST 2: After applying compaction")
    print("=" * 70)
    print("Applying compaction to existing model...")
    runner.model.compact(block_size=32)
    runner.model.compact_moe()

    results_compact = runner.run_batch(bytecodes, data_list, max_steps=1000)
    compact_numeric = [res for (out, res) in results_compact]

    passed = sum(1 for res, exp in zip(compact_numeric, expected_results) if res == exp)
    print(f"Results: {passed}/{len(bytecodes)} passed")
    for i, (res, exp, desc) in enumerate(zip(compact_numeric, expected_results, [d for _, _, d in test_programs])):
        status = "✓" if res == exp else "✗"
        print(f"  {status} {desc}: got {res}, expected {exp}")
    print()

    # Test 3: Apply sparsification (on top of compaction)
    print("=" * 70)
    print("TEST 3: After applying sparsification (on top of compaction)")
    print("=" * 70)
    print("Applying sparsification to compacted model...")
    runner.model.sparsify()

    results_both = runner.run_batch(bytecodes, data_list, max_steps=1000)
    both_numeric = [res for (out, res) in results_both]

    passed = sum(1 for res, exp in zip(both_numeric, expected_results) if res == exp)
    print(f"Results: {passed}/{len(bytecodes)} passed")
    for i, (res, exp, desc) in enumerate(zip(both_numeric, expected_results, [d for _, _, d in test_programs])):
        status = "✓" if res == exp else "✗"
        print(f"  {status} {desc}: got {res}, expected {exp}")
    print()

    # Compare results
    print("=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)
    print()

    all_match = True

    print("Baseline vs Compaction:")
    if baseline_numeric == compact_numeric:
        print("  ✓ IDENTICAL")
    else:
        print("  ✗ DIFFERENT")
        all_match = False
        for i, (r1, r2, desc) in enumerate(zip(baseline_numeric, compact_numeric, [d for _, _, d in test_programs])):
            if r1 != r2:
                print(f"    {desc}: baseline={r1}, compacted={r2}")
    print()

    print("Baseline vs Compaction+Sparse:")
    if baseline_numeric == both_numeric:
        print("  ✓ IDENTICAL")
    else:
        print("  ✗ DIFFERENT")
        all_match = False
        for i, (r1, r2, desc) in enumerate(zip(baseline_numeric, both_numeric, [d for _, _, d in test_programs])):
            if r1 != r2:
                print(f"    {desc}: baseline={r1}, optimized={r2}")
    print()

    print("Compaction vs Compaction+Sparse:")
    if compact_numeric == both_numeric:
        print("  ✓ IDENTICAL")
    else:
        print("  ✗ DIFFERENT")
        all_match = False
        for i, (r1, r2, desc) in enumerate(zip(compact_numeric, both_numeric, [d for _, _, d in test_programs])):
            if r1 != r2:
                print(f"    {desc}: compacted={r1}, both={r2}")
    print()

    print("=" * 70)
    if all_match:
        print("✓ SUCCESS: All optimization configurations produce IDENTICAL results!")
        print()
        print("  This means:")
        print("  - Weight compaction preserves functionality")
        print("  - Sparse matrix optimization preserves functionality")
        print("  - Combined optimizations preserve functionality")
        print("  - You can safely use these optimizations")
        return 0
    else:
        print("✗ FAILURE: Some configurations produce DIFFERENT results!")
        print("  This indicates a bug in optimization implementations!")
        return 1


if __name__ == '__main__':
    sys.exit(test_consistency())
