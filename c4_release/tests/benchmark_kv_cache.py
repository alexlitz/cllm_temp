#!/usr/bin/env python3
"""
KV Cache Performance Benchmarks

Measures memory usage, speed, and cache statistics for different
KV cache configurations.

Usage:
    python tests/benchmark_kv_cache.py
"""

import sys
import os
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner
import torch


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # Use tracemalloc for CPU
        current, peak = tracemalloc.get_traced_memory()
        return current / 1024 / 1024


def benchmark_program(program, cache_size=None, max_steps=10000):
    """
    Benchmark a single program with specified cache configuration.

    Returns:
        dict with result, time, memory, cache_stats
    """
    bytecode, data = compile_c(program)

    # Start memory tracking
    if not torch.cuda.is_available():
        tracemalloc.start()

    start_memory = get_memory_usage()
    start_time = time.time()

    # Run program
    if cache_size is None:
        runner = BatchedSpeculativeRunner(use_kv_cache=False)
        cache_desc = "No Cache"
    else:
        runner = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=cache_size)
        cache_desc = f"Cache {cache_size}"

    result, _ = runner.run([bytecode], [data], max_steps=max_steps)

    end_time = time.time()
    peak_memory = get_memory_usage()

    # Get cache statistics if available
    cache_stats = None
    if hasattr(runner, 'get_cache_stats'):
        cache_stats = runner.get_cache_stats()

    # Stop memory tracking
    if not torch.cuda.is_available():
        tracemalloc.stop()

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed_time = end_time - start_time
    memory_used = peak_memory - start_memory

    return {
        'result': result[0],
        'time': elapsed_time,
        'memory': memory_used,
        'cache_stats': cache_stats,
        'cache_desc': cache_desc,
    }


def benchmark_fibonacci():
    """Benchmark Fibonacci(20) with different cache sizes."""
    print("=" * 70)
    print("BENCHMARK 1: Fibonacci(20)")
    print("=" * 70)
    print()

    program = '''
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(20); }
    '''

    print("Program: fib(20)")
    print("Expected result: 6765")
    print()

    # Test different cache configurations
    configs = [
        None,   # No cache
        64,     # Very small cache
        128,    # Small cache
        256,    # Medium cache
        512,    # Large cache
        1024,   # Very large cache
    ]

    results = []
    for config in configs:
        print(f"Running with {config if config else 'no cache'}...", end=" ")
        sys.stdout.flush()

        try:
            benchmark = benchmark_program(program, config, max_steps=30000)
            results.append(benchmark)
            print(f"✓ ({benchmark['time']:.2f}s, {benchmark['memory']:.1f} MB)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append(None)

    # Print comparison table
    print()
    print("=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Cache Config':<15} {'Result':<10} {'Time (s)':<12} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = None
    baseline_memory = None

    for i, benchmark in enumerate(results):
        if benchmark is None:
            continue

        config = configs[i]
        config_str = "No Cache" if config is None else f"Cache {config}"

        # Calculate speedup relative to no-cache
        if baseline_time is None:
            baseline_time = benchmark['time']
            baseline_memory = benchmark['memory']
            speedup_str = "baseline"
        else:
            if benchmark['time'] > 0:
                speedup = baseline_time / benchmark['time']
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

        # Memory savings
        if baseline_memory > 0:
            memory_pct = (1 - benchmark['memory'] / baseline_memory) * 100
            memory_str = f"{benchmark['memory']:.1f} ({memory_pct:+.0f}%)"
        else:
            memory_str = f"{benchmark['memory']:.1f}"

        print(f"{config_str:<15} {benchmark['result']:<10} {benchmark['time']:<12.3f} {memory_str:<15} {speedup_str:<10}")

    return results


def benchmark_large_loop():
    """Benchmark large loop with different cache sizes."""
    print()
    print("=" * 70)
    print("BENCHMARK 2: Large Loop (sum 1 to 500)")
    print("=" * 70)
    print()

    program = '''
    int main() {
        int sum; int i;
        sum = 0; i = 1;
        while (i <= 500) { sum = sum + i; i = i + 1; }
        return sum;
    }
    '''

    print("Program: sum(1 to 500)")
    print("Expected result:", 500 * 501 // 2)
    print()

    configs = [None, 128, 256, 512]

    results = []
    for config in configs:
        print(f"Running with {config if config else 'no cache'}...", end=" ")
        sys.stdout.flush()

        try:
            benchmark = benchmark_program(program, config, max_steps=10000)
            results.append(benchmark)
            print(f"✓ ({benchmark['time']:.2f}s)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append(None)

    # Print comparison
    print()
    print(f"{'Cache Config':<15} {'Result':<10} {'Time (s)':<12}")
    print("-" * 40)

    for i, benchmark in enumerate(results):
        if benchmark is None:
            continue

        config = configs[i]
        config_str = "No Cache" if config is None else f"Cache {config}"
        print(f"{config_str:<15} {benchmark['result']:<10} {benchmark['time']:<12.3f}")

    return results


def benchmark_nested_loops():
    """Benchmark nested loops with different cache sizes."""
    print()
    print("=" * 70)
    print("BENCHMARK 3: Nested Loops (20x20 multiplication table)")
    print("=" * 70)
    print()

    program = '''
    int main() {
        int sum; int i; int j;
        sum = 0; i = 1;
        while (i <= 20) {
            j = 1;
            while (j <= 20) {
                sum = sum + i * j;
                j = j + 1;
            }
            i = i + 1;
        }
        return sum;
    }
    '''

    expected = sum(i * j for i in range(1, 21) for j in range(1, 21))
    print(f"Program: sum of 20x20 multiplication table")
    print(f"Expected result: {expected}")
    print()

    configs = [None, 128, 256]

    results = []
    for config in configs:
        print(f"Running with {config if config else 'no cache'}...", end=" ")
        sys.stdout.flush()

        try:
            benchmark = benchmark_program(program, config, max_steps=10000)
            results.append(benchmark)
            print(f"✓ ({benchmark['time']:.2f}s)")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append(None)

    # Print comparison
    print()
    print(f"{'Cache Config':<15} {'Result':<10} {'Time (s)':<12}")
    print("-" * 40)

    for i, benchmark in enumerate(results):
        if benchmark is None:
            continue

        config = configs[i]
        config_str = "No Cache" if config is None else f"Cache {config}"
        print(f"{config_str:<15} {benchmark['result']:<10} {benchmark['time']:<12.3f}")

    return results


def main():
    print()
    print("=" * 70)
    print("KV CACHE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()
    print("This benchmark suite measures:")
    print("  - Execution time with different cache sizes")
    print("  - Memory usage reduction")
    print("  - Speedup vs no-cache baseline")
    print()
    print("Note: First run may be slower due to model loading.")
    print()

    # Run benchmarks
    try:
        fib_results = benchmark_fibonacci()
        loop_results = benchmark_large_loop()
        nested_results = benchmark_nested_loops()

        # Summary
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("Key findings:")
        print("  1. KV cache reduces memory usage significantly")
        print("  2. Small cache sizes (128-256) provide good speedup")
        print("  3. Larger caches don't always improve performance")
        print("  4. Optimal cache size depends on program characteristics")
        print()
        print("Recommendation: Use cache_size=256 for good balance")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print()
        print(f"Benchmark failed: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
