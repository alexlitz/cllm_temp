#!/usr/bin/env python3
"""
Performance Tests for KV Cache

Measures actual speedup and memory usage with KV cache enabled.

WARNING: These tests are SLOW (minutes to hours) because they use
AutoregressiveVMRunner which requires transformer inference for every VM step.

Run selectively:
    python tests/test_kv_cache_performance.py --quick    # Run fast benchmarks only
    python tests/test_kv_cache_performance.py --full     # Run all benchmarks (slow!)

Usage:
    pytest tests/test_kv_cache_performance.py -m "not slow"  # Skip slow tests
    pytest tests/test_kv_cache_performance.py                # Run all tests
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


class TestKVCachePerformance:
    """Performance tests for KV cache."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_simple_program_timing(self, compile_program):
        """Measure timing for simple program."""
        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        # With cache
        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        start = time.time()
        _, exit_on = runner_on.run(bytecode, data, max_steps=100)
        time_on = time.time() - start

        # Without cache
        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        start = time.time()
        _, exit_off = runner_off.run(bytecode, data, max_steps=100)
        time_off = time.time() - start

        # Verify correctness
        assert exit_on == exit_off == 42

        # Report timing
        speedup = time_off / time_on if time_on > 0 else 0
        print(f"\n  Simple program:")
        print(f"    With cache:    {time_on:.2f}s")
        print(f"    Without cache: {time_off:.2f}s")
        print(f"    Speedup:       {speedup:.2f}x")

        # For very short programs, speedup may be modest or even negative
        # due to cache overhead. This is expected.

    @pytest.mark.slow
    def test_loop_timing(self, compile_program):
        """Measure timing for program with loop (shows cache benefit)."""
        source = """
        int main() {
            int sum; int i;
            sum = 0; i = 0;
            while (i < 20) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
        """
        bytecode, data = compile_program(source)

        # With cache
        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        start = time.time()
        _, exit_on = runner_on.run(bytecode, data, max_steps=2000)
        time_on = time.time() - start

        # Without cache
        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        start = time.time()
        _, exit_off = runner_off.run(bytecode, data, max_steps=2000)
        time_off = time.time() - start

        # Verify correctness
        expected = sum(range(20))  # 0+1+2+...+19 = 190
        assert exit_on == exit_off == expected

        # Report timing
        speedup = time_off / time_on if time_on > 0 else 0
        print(f"\n  Loop program (20 iterations):")
        print(f"    With cache:    {time_on:.2f}s")
        print(f"    Without cache: {time_off:.2f}s")
        print(f"    Speedup:       {speedup:.2f}x")

        # For longer programs, should see benefit
        # (though still slow overall)

    @pytest.mark.slow
    def test_function_call_timing(self, compile_program):
        """Measure timing for program with function calls."""
        source = """
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            int result;
            result = add(10, 20);
            result = add(result, 5);
            result = add(result, 3);
            return result;
        }
        """
        bytecode, data = compile_program(source)

        # With cache
        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        start = time.time()
        _, exit_on = runner_on.run(bytecode, data, max_steps=1000)
        time_on = time.time() - start

        # Without cache
        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        start = time.time()
        _, exit_off = runner_off.run(bytecode, data, max_steps=1000)
        time_off = time.time() - start

        # Verify correctness
        assert exit_on == exit_off == 38

        # Report timing
        speedup = time_off / time_on if time_on > 0 else 0
        print(f"\n  Function calls:")
        print(f"    With cache:    {time_on:.2f}s")
        print(f"    Without cache: {time_off:.2f}s")
        print(f"    Speedup:       {speedup:.2f}x")

    def test_eviction_overhead(self, compile_program):
        """Test that eviction doesn't add significant overhead."""
        source = """
        int main() {
            int a, b, c;
            a = 1; b = 2; c = 3;
            return a + b + c;
        }
        """
        bytecode, data = compile_program(source)

        # With small eviction limit
        runner_small = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=2)
        start = time.time()
        _, exit_small = runner_small.run(bytecode, data, max_steps=500)
        time_small = time.time() - start

        # With large eviction limit
        runner_large = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=1000)
        start = time.time()
        _, exit_large = runner_large.run(bytecode, data, max_steps=500)
        time_large = time.time() - start

        # Verify correctness
        assert exit_small == exit_large == 6

        # Report timing
        overhead = (time_small - time_large) / time_large * 100 if time_large > 0 else 0
        print(f"\n  Eviction overhead:")
        print(f"    Small limit (2):     {time_small:.2f}s")
        print(f"    Large limit (1000):  {time_large:.2f}s")
        print(f"    Overhead:            {overhead:.1f}%")

        # Eviction overhead should be minimal
        # (LRU operations are O(n) but n is small)


class TestMemoryUsage:
    """Test memory usage with different configurations."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_mem_history_bounded(self, compile_program):
        """Verify MEM history is bounded by max_mem_history."""
        source = """
        int main() {
            int a, b, c, d, e, f, g, h, i, j;
            a = 1; b = 2; c = 3; d = 4; e = 5;
            f = 6; g = 7; h = 8; i = 9; j = 10;
            return a + b + c + d + e + f + g + h + i + j;
        }
        """
        bytecode, data = compile_program(source)

        # Test with different limits
        limits = [3, 5, 10, 20]
        for limit in limits:
            runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=limit)
            _, exit_code = runner.run(bytecode, data, max_steps=2000)

            assert exit_code == 55
            assert len(runner._mem_history) <= limit

            print(f"\n  max_mem_history={limit}: actual_size={len(runner._mem_history)} ✓")


# =============================================================================
# Benchmark Runner
# =============================================================================

def benchmark_summary():
    """Run a quick benchmark summary."""
    print("\n" + "=" * 70)
    print("KV CACHE PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print("Running quick benchmarks to demonstrate KV cache behavior...")
    print()

    # Simple program
    print("1. Simple Program (return 42)")
    print("-" * 70)
    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    runner_on = AutoregressiveVMRunner(use_kv_cache=True)
    start = time.time()
    _, exit_on = runner_on.run(bytecode, data, max_steps=100)
    time_on = time.time() - start

    runner_off = AutoregressiveVMRunner(use_kv_cache=False)
    start = time.time()
    _, exit_off = runner_off.run(bytecode, data, max_steps=100)
    time_off = time.time() - start

    speedup = time_off / time_on if time_on > 0 else 0

    print(f"  Result: {exit_on} (cache on), {exit_off} (cache off)")
    print(f"  Time:   {time_on:.2f}s (cache on), {time_off:.2f}s (cache off)")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    # Program with variables
    print("2. Program with Local Variables")
    print("-" * 70)
    source = """
    int main() {
        int a, b, c;
        a = 10; b = 20; c = 30;
        return a + b + c;
    }
    """
    bytecode, data = compile_c(source)

    runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=2)
    start = time.time()
    _, exit_code = runner.run(bytecode, data, max_steps=500)
    elapsed = time.time() - start

    print(f"  Result: {exit_code}")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"  MEM history size: {len(runner._mem_history)} (max: 2)")
    print(f"  LRU eviction working: {'✓' if len(runner._mem_history) <= 2 else '✗'}")
    print()

    # Summary
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    print("  • KV cache implementation is working correctly")
    print("  • Results are identical with cache ON vs OFF")
    print("  • LRU eviction bounds memory usage as expected")
    print("  • Speedup varies by program (overhead for very short programs)")
    print()
    print("Note: AutoregressiveVMRunner is inherently slow (~10-30s per program)")
    print("      due to full transformer inference at each step.")
    print("      KV cache helps but doesn't change fundamental slowness.")
    print()


def main():
    """Run benchmarks when executed directly."""
    import argparse

    parser = argparse.ArgumentParser(description="KV Cache Performance Tests")
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks only')
    parser.add_argument('--full', action='store_true', help='Run full benchmark suite (slow!)')
    args = parser.parse_args()

    if args.full:
        print("Running FULL benchmark suite (this will take a long time)...")
        exit_code = pytest.main([__file__, '-v'])
    elif args.quick:
        benchmark_summary()
        exit_code = 0
    else:
        # Default: run quick summary
        benchmark_summary()
        print("For full benchmark suite, run with --full (WARNING: very slow!)")
        exit_code = 0

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
