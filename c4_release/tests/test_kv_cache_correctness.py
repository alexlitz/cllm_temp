#!/usr/bin/env python3
"""
KV Cache Correctness Validation Tests

Tests that KV cache eviction doesn't break correctness by running
the same programs with different cache configurations and comparing results.

Usage:
    python tests/test_kv_cache_correctness.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from neural_vm.batch_runner import BatchedSpeculativeRunner
import torch


def test_determinism():
    """Test that KV cache produces deterministic results."""
    print("=" * 60)
    print("TEST 1: Determinism (same program, 10 runs)")
    print("=" * 60)

    program = '''
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(15); }
    '''

    bytecode, data = compile_c(program)

    # Run 10 times with KV cache
    results = []
    for i in range(10):
        runner = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=128)
        result, _ = runner.run([bytecode], [data], max_steps=5000)
        results.append(result[0])

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Check all results identical
    if len(set(results)) == 1:
        print(f"  ✓ All 10 runs produced: {results[0]}")
        print(f"  ✓ PASS: Deterministic results")
        return True
    else:
        print(f"  ✗ FAIL: Non-deterministic results: {set(results)}")
        return False


def test_cache_size_sweep():
    """Test that different cache sizes produce identical results."""
    print()
    print("=" * 60)
    print("TEST 2: Cache Size Sweep")
    print("=" * 60)

    program = '''
    int sum(int n) {
        int s; int i;
        s = 0; i = 1;
        while (i <= n) { s = s + i; i = i + 1; }
        return s;
    }
    int main() { return sum(100); }
    '''

    bytecode, data = compile_c(program)

    # Test with different cache sizes
    cache_sizes = [64, 128, 256, 512, 1024, 2048]
    results = {}

    for cache_size in cache_sizes:
        runner = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=cache_size)
        result, _ = runner.run([bytecode], [data], max_steps=2000)
        results[cache_size] = result[0]

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Also test without cache
    runner_no_cache = BatchedSpeculativeRunner(use_kv_cache=False)
    result_no_cache, _ = runner_no_cache.run([bytecode], [data], max_steps=2000)
    results['no_cache'] = result_no_cache[0]

    # Check all results identical
    unique_results = set(results.values())

    print(f"  Results by cache size:")
    for size, result in results.items():
        print(f"    {str(size):>10s}: {result}")

    if len(unique_results) == 1:
        print(f"  ✓ PASS: All cache sizes produce: {list(unique_results)[0]}")
        return True
    else:
        print(f"  ✗ FAIL: Different results: {unique_results}")
        return False


def test_eviction_boundary():
    """Test program at eviction boundary."""
    print()
    print("=" * 60)
    print("TEST 3: Eviction Boundary")
    print("=" * 60)

    # Simple program that executes ~150 steps
    program = '''
    int main() {
        int sum; int i;
        sum = 0; i = 1;
        while (i <= 50) { sum = sum + i; i = i + 1; }
        return sum;
    }
    '''

    bytecode, data = compile_c(program)

    # Test with cache just below and above program length
    cache_configs = [
        (128, "cache < steps (eviction)"),
        (256, "cache > steps (no eviction)")
    ]

    results = {}
    for cache_size, desc in cache_configs:
        runner = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=cache_size)
        result, _ = runner.run([bytecode], [data], max_steps=500)
        results[desc] = result[0]

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Results:")
    for desc, result in results.items():
        print(f"    {desc}: {result}")

    if len(set(results.values())) == 1:
        print(f"  ✓ PASS: Eviction doesn't affect correctness")
        return True
    else:
        print(f"  ✗ FAIL: Different results across eviction boundary")
        return False


def test_batch_consistency():
    """Test that batched execution with KV cache matches individual execution."""
    print()
    print("=" * 60)
    print("TEST 4: Batch Consistency")
    print("=" * 60)

    programs = [
        'int main() { return 10 + 32; }',
        'int main() { return 50 - 8; }',
        'int main() { return 6 * 7; }',
        'int main() { return 84 / 2; }',
    ]

    bytecodes = [compile_c(prog)[0] for prog in programs]
    datas = [compile_c(prog)[1] for prog in programs]

    # Run batched
    runner_batch = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=128)
    batch_results, _ = runner_batch.run(bytecodes, datas, max_steps=100)

    # Run individually
    individual_results = []
    for bytecode, data in zip(bytecodes, datas):
        runner = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=128)
        result, _ = runner.run([bytecode], [data], max_steps=100)
        individual_results.append(result[0])

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Batch results:      {batch_results}")
    print(f"  Individual results: {individual_results}")

    if batch_results == individual_results:
        print(f"  ✓ PASS: Batch matches individual execution")
        return True
    else:
        print(f"  ✗ FAIL: Batch differs from individual")
        return False


def test_long_context_stress():
    """Stress test with long-running program."""
    print()
    print("=" * 60)
    print("TEST 5: Long-Context Stress Test")
    print("=" * 60)

    # Fibonacci(20) - requires many recursive calls
    program = '''
    int fib(int n) {
        if (n < 2) return n;
        return fib(n-1) + fib(n-2);
    }
    int main() { return fib(20); }
    '''

    bytecode, data = compile_c(program)

    # Run with small cache (heavy eviction)
    print("  Running fib(20) with cache_size=256 (heavy eviction)...")
    runner_cached = BatchedSpeculativeRunner(use_kv_cache=True, kv_cache_max_tokens=256)
    result_cached, _ = runner_cached.run([bytecode], [data], max_steps=30000)

    # Run without cache (reference)
    print("  Running fib(20) without cache (reference)...")
    runner_no_cache = BatchedSpeculativeRunner(use_kv_cache=False)
    result_no_cache, _ = runner_no_cache.run([bytecode], [data], max_steps=30000)

    print(f"  Result with cache:    {result_cached[0]}")
    print(f"  Result without cache: {result_no_cache[0]}")
    print(f"  Expected:             6765")

    if result_cached[0] == result_no_cache[0] == 6765:
        print(f"  ✓ PASS: Long-context program correct with heavy eviction")
        return True
    else:
        print(f"  ✗ FAIL: Incorrect result")
        return False


def main():
    print()
    print("=" * 60)
    print("KV CACHE CORRECTNESS VALIDATION")
    print("=" * 60)
    print()
    print("This test suite validates that KV cache eviction doesn't")
    print("break correctness by comparing results across:")
    print("  - Multiple runs (determinism)")
    print("  - Different cache sizes")
    print("  - Eviction boundaries")
    print("  - Batch vs individual execution")
    print("  - Long-running programs with heavy eviction")
    print()

    tests = [
        ("Determinism", test_determinism),
        ("Cache Size Sweep", test_cache_size_sweep),
        ("Eviction Boundary", test_eviction_boundary),
        ("Batch Consistency", test_batch_consistency),
        ("Long-Context Stress", test_long_context_stress),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"  Total: {passed_count}/{total_count} tests passed")
    print(f"  Success rate: {passed_count/total_count*100:.1f}%")

    if passed_count == total_count:
        print()
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print()
        print("KV cache eviction maintains correctness across all test scenarios.")
        return 0
    else:
        print()
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
        return 1


if __name__ == '__main__':
    sys.exit(main())
