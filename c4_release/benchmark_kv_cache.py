#!/usr/bin/env python3
"""
Benchmark KV cache eviction performance.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner


def benchmark_kv_cache():
    """Benchmark execution speed with and without KV cache."""
    print("=" * 70)
    print("KV CACHE EVICTION PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Test programs with varying complexity
    test_programs = [
        ("int main() { return 42; }", 42, "simple return"),
        ("int main() { return 5 + 7; }", 12, "addition"),
        ("int main() { return 10 * 4; }", 40, "multiplication"),
        ("int main() { return (3 + 4) * 5; }", 35, "expression"),
        ("int main() { return 100 - 42; }", 58, "subtraction"),
        ("int main() { return 12 / 3; }", 4, "division"),
        ("int main() { return 15 + 10; }", 25, "addition2"),
        ("int main() { return 10 + 20; }", 30, "addition3"),
    ]

    # Compile all programs
    print(f"\nCompiling {len(test_programs)} test programs...")
    bytecodes = []
    data_list = []
    expected_results = []

    for source, expected, desc in test_programs:
        bytecode, data = compile_c(source)
        bytecodes.append(bytecode)
        data_list.append(data)
        expected_results.append(expected)

    print(f"✓ Compiled {len(bytecodes)} programs")

    # Warm up GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Benchmark 1: WITHOUT KV cache
    print("\n" + "=" * 70)
    print("BENCHMARK 1: WITHOUT KV Cache")
    print("=" * 70)

    runner_no_cache = BatchedSpeculativeRunner(
        batch_size=8,
        use_kv_cache=False,
        use_sparse=True,
    )

    # Warm-up run
    _ = runner_no_cache.run_batch(bytecodes, data_list, max_steps=1000)

    # Timed runs
    num_runs = 5
    times_no_cache = []

    for i in range(num_runs):
        runner_no_cache = BatchedSpeculativeRunner(
            batch_size=8,
            use_kv_cache=False,
            use_sparse=True,
        )

        start = time.time()
        results = runner_no_cache.run_batch(bytecodes, data_list, max_steps=1000)
        elapsed = time.time() - start
        times_no_cache.append(elapsed)

        passed = sum(1 for (out, res), exp in zip(results, expected_results) if res == exp)
        print(f"  Run {i+1}: {elapsed:.4f}s ({passed}/{len(bytecodes)} passed)")

    avg_no_cache = sum(times_no_cache) / len(times_no_cache)
    stats_no_cache = runner_no_cache.get_stats()

    print(f"\nAverage time: {avg_no_cache:.4f}s")
    print(f"Validations: {stats_no_cache['validations']}")
    print(f"Total steps: {stats_no_cache['total_steps']}")

    # Benchmark 2: WITH KV cache (2048 tokens)
    print("\n" + "=" * 70)
    print("BENCHMARK 2: WITH KV Cache (2048 tokens/layer)")
    print("=" * 70)

    runner_with_cache = BatchedSpeculativeRunner(
        batch_size=8,
        use_kv_cache=True,
        kv_cache_max_tokens=2048,
        use_sparse=True,
    )

    # Warm-up run
    _ = runner_with_cache.run_batch(bytecodes, data_list, max_steps=1000)

    # Timed runs
    times_with_cache = []

    for i in range(num_runs):
        runner_with_cache = BatchedSpeculativeRunner(
            batch_size=8,
            use_kv_cache=True,
            kv_cache_max_tokens=2048,
            use_sparse=True,
        )

        start = time.time()
        results = runner_with_cache.run_batch(bytecodes, data_list, max_steps=1000)
        elapsed = time.time() - start
        times_with_cache.append(elapsed)

        passed = sum(1 for (out, res), exp in zip(results, expected_results) if res == exp)
        print(f"  Run {i+1}: {elapsed:.4f}s ({passed}/{len(bytecodes)} passed)")

    avg_with_cache = sum(times_with_cache) / len(times_with_cache)
    stats_with_cache = runner_with_cache.get_stats()
    cache_stats = runner_with_cache.kv_cache.get_total_stats()

    print(f"\nAverage time: {avg_with_cache:.4f}s")
    print(f"Validations: {stats_with_cache['validations']}")
    print(f"Total steps: {stats_with_cache['total_steps']}")
    print(f"\nKV Cache Stats:")
    print(f"  Tokens cached: {cache_stats['tokens_cached']}")
    print(f"  Tokens evicted: {cache_stats['tokens_evicted']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")

    # Benchmark 3: WITH KV cache (128 tokens - aggressive eviction)
    print("\n" + "=" * 70)
    print("BENCHMARK 3: WITH KV Cache (128 tokens/layer - aggressive)")
    print("=" * 70)

    times_small_cache = []

    for i in range(num_runs):
        runner_small_cache = BatchedSpeculativeRunner(
            batch_size=8,
            use_kv_cache=True,
            kv_cache_max_tokens=128,
            use_sparse=True,
        )

        start = time.time()
        results = runner_small_cache.run_batch(bytecodes, data_list, max_steps=1000)
        elapsed = time.time() - start
        times_small_cache.append(elapsed)

        passed = sum(1 for (out, res), exp in zip(results, expected_results) if res == exp)
        print(f"  Run {i+1}: {elapsed:.4f}s ({passed}/{len(bytecodes)} passed)")

    avg_small_cache = sum(times_small_cache) / len(times_small_cache)
    cache_stats_small = runner_small_cache.kv_cache.get_total_stats()

    print(f"\nAverage time: {avg_small_cache:.4f}s")
    print(f"\nKV Cache Stats:")
    print(f"  Tokens cached: {cache_stats_small['tokens_cached']}")
    print(f"  Tokens evicted: {cache_stats_small['tokens_evicted']}")
    print(f"  Cache hits: {cache_stats_small['cache_hits']}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    print(f"\nExecution Time (average of {num_runs} runs):")
    print(f"  No cache:        {avg_no_cache:.4f}s  (baseline)")
    print(f"  Cache 2048:      {avg_with_cache:.4f}s  ({avg_with_cache/avg_no_cache:.2f}x)")
    print(f"  Cache 128:       {avg_small_cache:.4f}s  ({avg_small_cache/avg_no_cache:.2f}x)")

    speedup_2048 = avg_no_cache / avg_with_cache
    speedup_128 = avg_no_cache / avg_small_cache

    if speedup_2048 > 1.0:
        print(f"\n✓ KV cache (2048) is {speedup_2048:.2f}x FASTER")
    elif speedup_2048 < 1.0:
        print(f"\n⚠ KV cache (2048) is {1/speedup_2048:.2f}x SLOWER")
    else:
        print(f"\n= KV cache (2048) has SAME speed")

    if speedup_128 > 1.0:
        print(f"✓ KV cache (128) is {speedup_128:.2f}x FASTER")
    elif speedup_128 < 1.0:
        print(f"⚠ KV cache (128) is {1/speedup_128:.2f}x SLOWER")
    else:
        print(f"= KV cache (128) has SAME speed")

    print(f"\nMemory Impact:")
    print(f"  Cache 2048: {cache_stats['tokens_evicted']} tokens evicted")
    print(f"  Cache 128:  {cache_stats_small['tokens_evicted']} tokens evicted (more aggressive)")

    print(f"\nConclusion:")
    if speedup_2048 > 0.95:  # Within 5% is good
        print(f"  ✓ KV cache maintains performance while reducing memory")
    else:
        print(f"  ⚠ KV cache has overhead ({1/speedup_2048:.1%} slower)")
        print(f"    Trade-off: Reduced memory for slightly lower speed")

    return 0


if __name__ == '__main__':
    sys.exit(benchmark_kv_cache())
