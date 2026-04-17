#!/usr/bin/env python3
"""
Test KV cache speedup on LONG programs (many steps).
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def test_long_programs():
    """Test speed difference for programs with many execution steps."""
    print("=" * 70)
    print("LONG PROGRAM PERFORMANCE TEST")
    print("=" * 70)

    # Long-running programs (simple expressions only - compiler limitation)
    test_programs = [
        # Simple loop - sum
        ("int main() { return (1+2+3+4+5+6+7+8+9+10); }", 55, "sum expression"),

        # Many operations
        ("int main() { return ((10*5) + (20*3) + (15*2) + (8*7)); }", 186, "multi-ops"),

        # Nested arithmetic
        ("int main() { return (((5+3)*2) + ((7+2)*3) + ((4+1)*4)); }", 59, "nested-arith"),
    ]

    # Compile programs
    print(f"\nCompiling {len(test_programs)} long-running programs...")
    bytecodes = []
    data_list = []
    expected_results = []

    for source, expected, desc in test_programs:
        bytecode, data = compile_c(source)
        bytecodes.append(bytecode)
        data_list.append(data)
        expected_results.append(expected)
        print(f"  ✓ {desc}: {len(bytecode)} instructions")

    print(f"\nTotal programs: {len(bytecodes)}")

    # Test 1: WITHOUT KV cache
    print("\n" + "=" * 70)
    print("TEST 1: WITHOUT KV Cache")
    print("=" * 70)

    runner_no_cache = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=False,
        use_sparse=True,
    )

    start = time.time()
    results_no_cache = runner_no_cache.run_batch(
        bytecodes=bytecodes,
        data_list=data_list,
        max_steps=5000
    )
    time_no_cache = time.time() - start

    passed_no_cache = sum(1 for (out, res), exp in zip(results_no_cache, expected_results) if res == exp)
    stats_no_cache = runner_no_cache.get_stats()

    print(f"\nResults:")
    for i, ((out, res), exp, desc) in enumerate(zip(results_no_cache, expected_results, [d for _, _, d in test_programs])):
        status = "✓" if res == exp else "✗"
        print(f"  {status} {desc}: got {res}, expected {exp}")

    print(f"\nTime: {time_no_cache:.2f}s")
    print(f"Validations: {stats_no_cache['validations']}")
    print(f"Total steps: {stats_no_cache['total_steps']}")
    print(f"Passed: {passed_no_cache}/{len(bytecodes)}")

    # Test 2: WITH KV cache
    print("\n" + "=" * 70)
    print("TEST 2: WITH KV Cache (2048 tokens)")
    print("=" * 70)

    runner_with_cache = BatchedSpeculativeRunner(
        batch_size=4,
        use_kv_cache=True,
        kv_cache_max_tokens=2048,
        use_sparse=True,
    )

    start = time.time()
    results_with_cache = runner_with_cache.run_batch(
        bytecodes=bytecodes,
        data_list=data_list,
        max_steps=5000
    )
    time_with_cache = time.time() - start

    passed_with_cache = sum(1 for (out, res), exp in zip(results_with_cache, expected_results) if res == exp)
    stats_with_cache = runner_with_cache.get_stats()
    cache_stats = runner_with_cache.kv_cache.get_total_stats()

    print(f"\nResults:")
    for i, ((out, res), exp, desc) in enumerate(zip(results_with_cache, expected_results, [d for _, _, d in test_programs])):
        status = "✓" if res == exp else "✗"
        print(f"  {status} {desc}: got {res}, expected {exp}")

    print(f"\nTime: {time_with_cache:.2f}s")
    print(f"Validations: {stats_with_cache['validations']}")
    print(f"Total steps: {stats_with_cache['total_steps']}")
    print(f"Passed: {passed_with_cache}/{len(bytecodes)}")

    print(f"\nKV Cache Stats:")
    print(f"  Tokens cached: {cache_stats['tokens_cached']}")
    print(f"  Tokens evicted: {cache_stats['tokens_evicted']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")

    # Comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    print(f"\nExecution Time:")
    print(f"  Without cache: {time_no_cache:.2f}s")
    print(f"  With cache:    {time_with_cache:.2f}s")

    if time_with_cache < time_no_cache:
        speedup = time_no_cache / time_with_cache
        print(f"\n✓ SPEEDUP: {speedup:.2f}x FASTER with KV cache!")
        savings = time_no_cache - time_with_cache
        print(f"  Time saved: {savings:.2f}s ({savings/time_no_cache*100:.1f}%)")
    else:
        slowdown = time_with_cache / time_no_cache
        print(f"\n⚠ Cache is {slowdown:.2f}x SLOWER (overhead > savings)")

    print(f"\nSteps executed:")
    print(f"  Without cache: {stats_no_cache['total_steps']} steps")
    print(f"  With cache:    {stats_with_cache['total_steps']} steps")

    print(f"\nValidations:")
    print(f"  Without cache: {stats_no_cache['validations']}")
    print(f"  With cache:    {stats_with_cache['validations']}")

    if cache_stats['tokens_evicted'] > 0:
        print(f"\n✓ Eviction working: {cache_stats['tokens_evicted']} tokens evicted")

    print(f"\nCorrectness:")
    print(f"  Without cache: {passed_no_cache}/{len(bytecodes)} passed")
    print(f"  With cache:    {passed_with_cache}/{len(bytecodes)} passed")

    if passed_no_cache == passed_with_cache == len(bytecodes):
        print("\n✓ Both methods produce identical correct results!")

    return 0


if __name__ == '__main__':
    sys.exit(test_long_programs())
