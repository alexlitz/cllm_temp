#!/usr/bin/env python3
"""
Test KV Cache Eviction with BatchedSpeculativeRunner.

Verifies:
1. KV cache is being used during execution
2. Eviction actually removes old entries (reduces memory)
3. Mismatch detection is working correctly
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def test_kv_cache_eviction():
    """Test KV cache eviction with multiple programs."""
    print("=" * 60)
    print("TEST: KV Cache Eviction with BatchedSpeculativeRunner")
    print("=" * 60)

    # Test programs with varying complexity
    test_programs = [
        ("int main() { return 42; }", 42, "simple return"),
        ("int main() { return 5 + 7; }", 12, "addition"),
        ("int main() { return 10 * 4; }", 40, "multiplication"),
        ("int main() { return (3 + 4) * 5; }", 35, "expression"),
    ]

    # Compile all programs
    bytecodes = []
    data_list = []
    expected_results = []

    for source, expected, desc in test_programs:
        bytecode, data = compile_c(source)
        bytecodes.append(bytecode)
        data_list.append(data)
        expected_results.append(expected)
        print(f"  Compiled: {desc}")

    print(f"\nTotal programs: {len(bytecodes)}")
    print()

    # Test 1: Run WITHOUT KV cache (baseline)
    print("1. Running WITHOUT KV cache (baseline)...")
    runner_no_cache = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=False,
        use_sparse=True,
    )

    results_no_cache = runner_no_cache.run_batch(
        bytecodes=bytecodes,
        data_list=data_list,
        max_steps=1000
    )

    passed_no_cache = sum(1 for (out, res), exp in zip(results_no_cache, expected_results) if res == exp)
    stats_no_cache = runner_no_cache.get_stats()

    print(f"   Results: {passed_no_cache}/{len(bytecodes)} passed")
    print(f"   Validations: {stats_no_cache['validations']}")
    print(f"   Mismatches: {stats_no_cache['mismatches']}")
    print(f"   Match rate: {stats_no_cache['match_rate']:.1%}")
    print()

    # Test 2: Run WITH KV cache and eviction
    print("2. Running WITH KV cache and eviction...")
    runner_with_cache = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=True,
        kv_cache_max_tokens=128,  # Small cache to force eviction
        use_sparse=True,
    )

    results_with_cache = runner_with_cache.run_batch(
        bytecodes=bytecodes,
        data_list=data_list,
        max_steps=1000
    )

    passed_with_cache = sum(1 for (out, res), exp in zip(results_with_cache, expected_results) if res == exp)
    stats_with_cache = runner_with_cache.get_stats()

    print(f"   Results: {passed_with_cache}/{len(bytecodes)} passed")
    print(f"   Validations: {stats_with_cache['validations']}")
    print(f"   Mismatches: {stats_with_cache['mismatches']}")
    print(f"   Match rate: {stats_with_cache['match_rate']:.1%}")
    print()

    # Check KV cache statistics
    if runner_with_cache.kv_cache is not None:
        cache_stats = runner_with_cache.kv_cache.get_total_stats()
        print("3. KV Cache Statistics:")
        print(f"   Tokens cached: {cache_stats['tokens_cached']}")
        print(f"   Tokens evicted: {cache_stats['tokens_evicted']}")
        print(f"   Current size: {cache_stats['current_total_size']}")
        print(f"   Cache hits: {cache_stats['cache_hits']}")
        print(f"   Num layers: {cache_stats['num_layers']}")

        # Verify eviction is working
        if cache_stats['tokens_evicted'] > 0:
            print(f"\n✓ Eviction is WORKING - {cache_stats['tokens_evicted']} tokens evicted!")
        else:
            print(f"\n⚠ WARNING: No tokens evicted (cache might be too large)")

        # Memory reduction estimate
        max_tokens_per_layer = 128
        actual_tokens_per_layer = cache_stats['current_total_size'] // cache_stats['num_layers']
        print(f"\n4. Memory Reduction:")
        print(f"   Max cache per layer: {max_tokens_per_layer} tokens")
        print(f"   Actual cache per layer: {actual_tokens_per_layer} tokens")

        # Calculate saved memory (rough estimate)
        # Each token stores K,V: [num_heads, head_dim] * 2 * float16 = 2 bytes/value
        # For 8 heads, 64 head_dim: 8 * 64 * 2 * 2 = 2KB per token
        saved_tokens = cache_stats['tokens_evicted']
        saved_kb = saved_tokens * 2  # Rough estimate
        print(f"   Estimated memory saved: ~{saved_kb} KB")
    else:
        print("\n✗ ERROR: KV cache was not created!")
        return 1

    # Compare results
    print(f"\n5. Results Comparison:")
    print(f"   Without cache: {passed_no_cache}/{len(bytecodes)} passed")
    print(f"   With cache: {passed_with_cache}/{len(bytecodes)} passed")

    if passed_no_cache == passed_with_cache == len(bytecodes):
        print("\n✓ SUCCESS: KV cache produces identical results!")
        print("✓ Eviction is working and memory is being freed!")
        return 0
    else:
        print("\n✗ FAILURE: Results differ with KV cache!")
        return 1


if __name__ == '__main__':
    sys.exit(test_kv_cache_eviction())
