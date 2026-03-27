#!/usr/bin/env python3
"""
Test that compaction/sparsification don't change results.

Verifies that optimization configurations produce identical outputs:
- Baseline (no optimizations)
- With sparse weights
- With compaction
- With both sparse + compaction (full optimization)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def test_consistency():
    """Test consistency across different optimization configurations."""
    print("=" * 70)
    print("OPTIMIZATION CONSISTENCY TEST")
    print("=" * 70)
    print()

    # Test programs - simple to run fast
    test_programs = [
        ("int main() { return 42; }", 42, "simple"),
        ("int main() { return 5 + 7; }", 12, "addition"),
        ("int main() { return 10 * 4; }", 40, "multiplication"),
        ("int main() { return (3 + 4) * 5; }", 35, "expression"),
        ("int main() { int x; x = 10; return x + 5; }", 15, "variable"),
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

    print(f"\nTotal programs: {len(bytecodes)}")
    print()

    # Configuration matrix to test
    # NOTE: KV cache is NOT tested here because it's a runtime optimization
    # that doesn't affect correctness (already verified in other tests)
    configs = [
        ("Baseline (no optimizations)", {
            'use_sparse': False,
            'compact': False,
        }),
        ("With sparse only", {
            'use_sparse': True,
            'compact': False,
        }),
        ("With compaction only", {
            'use_sparse': False,
            'compact': True,
        }),
        ("Full optimization (sparse + compact)", {
            'use_sparse': True,
            'compact': True,
        }),
    ]

    results_by_config = {}

    print("=" * 70)
    print("Running tests with different configurations...")
    print("=" * 70)
    print()

    for config_name, config in configs:
        print(f"{config_name}:")
        print(f"  Sparse: {config['use_sparse']}")
        print(f"  Compact: {config['compact']}")

        # Create runner with specific configuration
        runner = BatchedSpeculativeRunner(
            batch_size=10,
            use_kv_cache=False,  # Disable to isolate compaction/sparse effects
            use_sparse=False,     # We'll control this manually
        )

        # Apply optimizations based on config
        if config['compact']:
            print("  Applying compaction...")
            runner.model.compact(block_size=32)
            runner.model.compact_moe()

        if config['use_sparse']:
            print("  Applying sparsification...")
            runner.model.sparsify()

        try:
            results = runner.run_batch(
                bytecodes=bytecodes,
                data_list=data_list,
                max_steps=1000
            )

            # Extract just the numeric results
            numeric_results = [res for (out, res) in results]
            results_by_config[config_name] = numeric_results

            # Check correctness
            passed = sum(1 for res, exp in zip(numeric_results, expected_results) if res == exp)
            print(f"  Results: {passed}/{len(bytecodes)} passed")

            for i, (res, exp, desc) in enumerate(zip(numeric_results, expected_results, [d for _, _, d in test_programs])):
                status = "✓" if res == exp else "✗"
                print(f"    {status} {desc}: got {res}, expected {exp}")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)[:100]}")
            results_by_config[config_name] = None

        print()

    # Compare all configurations
    print("=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)
    print()

    baseline_name = "Baseline (no optimizations)"
    baseline_results = results_by_config.get(baseline_name)

    if baseline_results is None:
        print(f"✗ Baseline failed to run - cannot check consistency")
        return 1

    all_match = True
    for config_name, results in results_by_config.items():
        if config_name == baseline_name:
            continue

        if results is None:
            print(f"✗ {config_name}: FAILED TO RUN")
            all_match = False
            continue

        if results == baseline_results:
            print(f"✓ {config_name}: IDENTICAL to baseline")
        else:
            print(f"✗ {config_name}: DIFFERENT from baseline!")
            all_match = False
            for i, (r1, r2, desc) in enumerate(zip(baseline_results, results, [d for _, _, d in test_programs])):
                if r1 != r2:
                    print(f"    {desc}: baseline={r1}, optimized={r2}")

    print()
    print("=" * 70)
    if all_match:
        print("✓ SUCCESS: All optimization configurations produce IDENTICAL results!")
        print("  Compaction and sparsification are numerically consistent!")
        print()
        print("  This means:")
        print("  - Weight compaction preserves functionality")
        print("  - Sparse matrix optimization preserves functionality")
        print("  - Combined optimizations preserve functionality")
        print("  - You can safely use these optimizations for faster/smaller models")
        return 0
    else:
        print("✗ FAILURE: Some configurations produce DIFFERENT results!")
        print("  This indicates a bug in optimization implementations!")
        return 1


if __name__ == '__main__':
    sys.exit(test_consistency())
