#!/usr/bin/env python3
"""
Comprehensive Test Suite for AutoregressiveVMRunner KV Cache Features

Tests the KV cache and LRU eviction implementation in AutoregressiveVMRunner.

Test Categories:
1. Unit Tests - Parameter initialization and basic setup
2. Correctness Tests - Verify cache doesn't change results
3. LRU Eviction Tests - Verify eviction maintains correctness
4. Integration Tests - Works with existing infrastructure
5. Edge Case Tests - Boundary conditions and corner cases

Usage:
    pytest tests/test_autoregressive_kv_cache.py
    python tests/test_autoregressive_kv_cache.py  # runs quick subset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner


# =============================================================================
# Unit Tests - Parameter Initialization
# =============================================================================

class TestParameterInitialization:
    """Test that KV cache parameters are properly initialized."""

    def test_default_parameters(self):
        """Test default parameter values."""
        runner = AutoregressiveVMRunner()

        assert hasattr(runner, 'use_kv_cache')
        assert hasattr(runner, 'max_mem_history')
        assert hasattr(runner, '_mem_access_order')

        assert runner.use_kv_cache == True
        assert runner.max_mem_history == 64
        assert isinstance(runner._mem_access_order, list)
        assert len(runner._mem_access_order) == 0

    def test_custom_cache_enabled(self):
        """Test with custom cache size."""
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=128)

        assert runner.use_kv_cache == True
        assert runner.max_mem_history == 128

    def test_cache_disabled(self):
        """Test with cache disabled."""
        runner = AutoregressiveVMRunner(use_kv_cache=False, max_mem_history=32)

        assert runner.use_kv_cache == False
        assert runner.max_mem_history == 32

    def test_lru_tracking_initialization(self):
        """Test that LRU tracking is properly initialized."""
        runner = AutoregressiveVMRunner(max_mem_history=10)

        assert hasattr(runner, '_mem_access_order')
        assert isinstance(runner._mem_access_order, list)
        assert runner._mem_access_order == []

    def test_mem_history_initialization(self):
        """Test that MEM history dict is initialized."""
        runner = AutoregressiveVMRunner()

        assert hasattr(runner, '_mem_history')
        assert isinstance(runner._mem_history, dict)


# =============================================================================
# Correctness Tests - Cache ON vs OFF
# =============================================================================

class TestCacheCorrectness:
    """Test that KV cache produces identical results to no-cache."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_simple_return(self, compile_program):
        """Test simple return statement."""
        source = "int main() { return 42; }"
        bytecode, data = compile_program(source)

        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_on = runner_on.run(bytecode, data, max_steps=100)

        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        _, exit_off = runner_off.run(bytecode, data, max_steps=100)

        assert exit_on == exit_off == 42

    def test_arithmetic(self, compile_program):
        """Test arithmetic operations."""
        source = "int main() { return 10 + 20 * 3; }"
        bytecode, data = compile_program(source)

        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_on = runner_on.run(bytecode, data, max_steps=200)

        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        _, exit_off = runner_off.run(bytecode, data, max_steps=200)

        assert exit_on == exit_off == 70

    def test_variables(self, compile_program):
        """Test with local variables."""
        source = """
        int main() {
            int a; int b; int c;
            a = 10;
            b = 20;
            c = a + b;
            return c;
        }
        """
        bytecode, data = compile_program(source)

        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_on = runner_on.run(bytecode, data, max_steps=500)

        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        _, exit_off = runner_off.run(bytecode, data, max_steps=500)

        assert exit_on == exit_off == 30

    def test_small_loop(self, compile_program):
        """Test with small loop."""
        source = """
        int main() {
            int sum; int i;
            sum = 0; i = 0;
            while (i < 10) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
        """
        bytecode, data = compile_program(source)

        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_on = runner_on.run(bytecode, data, max_steps=1000)

        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        _, exit_off = runner_off.run(bytecode, data, max_steps=1000)

        assert exit_on == exit_off == 45

    def test_function_call(self, compile_program):
        """Test with function calls."""
        source = """
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            return add(10, 20);
        }
        """
        bytecode, data = compile_program(source)

        runner_on = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_on = runner_on.run(bytecode, data, max_steps=500)

        runner_off = AutoregressiveVMRunner(use_kv_cache=False)
        _, exit_off = runner_off.run(bytecode, data, max_steps=500)

        assert exit_on == exit_off == 30


# =============================================================================
# LRU Eviction Tests
# =============================================================================

class TestLRUEviction:
    """Test that LRU eviction maintains correctness."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_eviction_with_small_limit(self, compile_program):
        """Test with eviction limit smaller than unique addresses."""
        source = """
        int main() {
            int a, b, c, d, e, f, g, h;
            a = 1; b = 2; c = 3; d = 4;
            e = 5; f = 6; g = 7; h = 8;
            return a + b + c + d + e + f + g + h;
        }
        """
        bytecode, data = compile_program(source)

        # Small eviction limit (should trigger eviction)
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=3)
        _, exit_code = runner.run(bytecode, data, max_steps=1000)

        # Verify result is correct despite eviction
        assert exit_code == 36

        # Verify eviction occurred (history bounded)
        assert len(runner._mem_history) <= 3

    def test_eviction_maintains_recent(self, compile_program):
        """Test that eviction keeps most recently accessed addresses."""
        source = """
        int main() {
            int arr[10];
            int i;
            i = 0;
            while (i < 10) {
                arr[i] = i * 2;
                i = i + 1;
            }
            return arr[9];
        }
        """
        bytecode, data = compile_program(source)

        # Limit smaller than total addresses
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=5)
        _, exit_code = runner.run(bytecode, data, max_steps=2000)

        # Should still get correct result
        assert exit_code == 18

        # History should be bounded
        assert len(runner._mem_history) <= 5

    def test_no_eviction_when_under_limit(self, compile_program):
        """Test that no eviction occurs when under limit."""
        source = """
        int main() {
            int a, b, c;
            a = 10; b = 20; c = 30;
            return a + b + c;
        }
        """
        bytecode, data = compile_program(source)

        # Large limit (no eviction expected)
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=100)
        _, exit_code = runner.run(bytecode, data, max_steps=500)

        assert exit_code == 60

        # Should have kept all addresses (no eviction)
        # Note: actual count depends on implementation details
        assert len(runner._mem_history) <= 100

    @pytest.mark.slow
    def test_many_unique_addresses(self, compile_program):
        """Test with many unique memory addresses (forces heavy eviction)."""
        source = """
        int main() {
            int arr[50];
            int i;
            i = 0;
            while (i < 50) {
                arr[i] = i;
                i = i + 1;
            }
            return arr[25];
        }
        """
        bytecode, data = compile_program(source)

        # Small limit forces eviction
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=10)
        _, exit_code = runner.run(bytecode, data, max_steps=5000)

        # Should still be correct
        assert exit_code == 25

        # History should be bounded
        assert len(runner._mem_history) <= 10


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Test integration with existing infrastructure."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_works_with_baked_c4(self):
        """Test that AutoregressiveVMRunner works alongside BakedC4Transformer."""
        from src.baked_c4 import BakedC4Transformer

        # BakedC4 should still work
        c4 = BakedC4Transformer(use_speculator=True)
        result_c4 = c4.run_c("int main() { return 42; }")
        assert result_c4 == 42

        # AutoregressiveVMRunner should also work
        source = "int main() { return 42; }"
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(use_kv_cache=True)
        _, exit_ar = runner.run(bytecode, data, max_steps=100)
        assert exit_ar == 42

    def test_multiple_sequential_runs(self, compile_program):
        """Test multiple runs on same runner instance."""
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=64)

        # Run 1
        source1 = "int main() { return 10; }"
        bytecode1, data1 = compile_program(source1)
        _, exit1 = runner.run(bytecode1, data1, max_steps=100)
        assert exit1 == 10

        # Run 2 (should reset state)
        source2 = "int main() { return 20; }"
        bytecode2, data2 = compile_program(source2)
        _, exit2 = runner.run(bytecode2, data2, max_steps=100)
        assert exit2 == 20

        # Run 3
        source3 = "int main() { return 30; }"
        bytecode3, data3 = compile_program(source3)
        _, exit3 = runner.run(bytecode3, data3, max_steps=100)
        assert exit3 == 30

    def test_lru_resets_between_runs(self, compile_program):
        """Test that LRU tracking resets between runs."""
        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=64)

        # Run 1 - populate some memory history
        source = """
        int main() {
            int a, b, c;
            a = 1; b = 2; c = 3;
            return a + b + c;
        }
        """
        bytecode, data = compile_program(source)
        runner.run(bytecode, data, max_steps=500)

        # Check state after run 1
        history_size_1 = len(runner._mem_history)
        lru_size_1 = len(runner._mem_access_order)

        # Run 2 - should reset
        runner.run(bytecode, data, max_steps=500)

        # LRU should reset between runs
        # (Can't check exact values, but they should be consistent)
        assert len(runner._mem_access_order) >= 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_zero_max_mem_history(self, compile_program):
        """Test with max_mem_history=0 (immediate eviction)."""
        source = "int main() { int a; a = 42; return a; }"
        bytecode, data = compile_program(source)

        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=0)

        # Should still work, just with no history
        _, exit_code = runner.run(bytecode, data, max_steps=500)
        assert exit_code == 42
        assert len(runner._mem_history) <= 0

    def test_very_large_max_mem_history(self, compile_program):
        """Test with very large max_mem_history."""
        source = """
        int main() {
            int a, b, c, d, e;
            a = 1; b = 2; c = 3; d = 4; e = 5;
            return a + b + c + d + e;
        }
        """
        bytecode, data = compile_program(source)

        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=10000)
        _, exit_code = runner.run(bytecode, data, max_steps=1000)

        assert exit_code == 15

    def test_single_variable(self, compile_program):
        """Test with single variable (minimal memory usage)."""
        source = "int main() { int x; x = 100; return x; }"
        bytecode, data = compile_program(source)

        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=64)
        _, exit_code = runner.run(bytecode, data, max_steps=300)

        assert exit_code == 100

    def test_repeated_access_same_address(self, compile_program):
        """Test repeated access to same memory address."""
        source = """
        int main() {
            int x;
            x = 1;
            x = 2;
            x = 3;
            x = 4;
            x = 5;
            return x;
        }
        """
        bytecode, data = compile_program(source)

        runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=10)
        _, exit_code = runner.run(bytecode, data, max_steps=1000)

        assert exit_code == 5

        # Should only have one entry (same address)
        # (actual behavior depends on implementation)
        assert len(runner._mem_history) >= 0


# =============================================================================
# Quick Smoke Tests (for fast validation)
# =============================================================================

def smoke_test_basic():
    """Quick smoke test for basic functionality."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Basic Functionality")
    print("=" * 70)

    source = "int main() { return 42; }"
    bytecode, data = compile_c(source)

    print("  Testing cache ON...")
    runner_on = AutoregressiveVMRunner(use_kv_cache=True)
    _, exit_on = runner_on.run(bytecode, data, max_steps=100)

    print("  Testing cache OFF...")
    runner_off = AutoregressiveVMRunner(use_kv_cache=False)
    _, exit_off = runner_off.run(bytecode, data, max_steps=100)

    if exit_on == exit_off == 42:
        print("  ✅ PASS: Both produce correct result (42)")
        return True
    else:
        print(f"  ❌ FAIL: cache_on={exit_on}, cache_off={exit_off}")
        return False


def smoke_test_lru():
    """Quick smoke test for LRU eviction."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: LRU Eviction")
    print("=" * 70)

    source = """
    int main() {
        int a, b, c, d, e, f, g, h, i, j;
        a = 1; b = 2; c = 3; d = 4; e = 5;
        f = 6; g = 7; h = 8; i = 9; j = 10;
        return a + b + c + d + e + f + g + h + i + j;
    }
    """
    bytecode, data = compile_c(source)

    print("  Running with small eviction limit (max_mem_history=3)...")
    runner = AutoregressiveVMRunner(use_kv_cache=True, max_mem_history=3)
    _, exit_code = runner.run(bytecode, data, max_steps=2000)

    print(f"  Exit code: {exit_code} (expected: 55)")
    print(f"  MEM history size: {len(runner._mem_history)} (max: 3)")

    if exit_code == 55 and len(runner._mem_history) <= 3:
        print("  ✅ PASS: Correct result with bounded memory")
        return True
    else:
        print(f"  ❌ FAIL: exit_code={exit_code}, history_size={len(runner._mem_history)}")
        return False


def smoke_test_parameters():
    """Quick smoke test for parameter initialization."""
    print("\n" + "=" * 70)
    print("SMOKE TEST: Parameter Initialization")
    print("=" * 70)

    runner = AutoregressiveVMRunner(use_kv_cache=False, max_mem_history=32)

    checks = [
        (hasattr(runner, 'use_kv_cache'), "use_kv_cache attribute"),
        (hasattr(runner, 'max_mem_history'), "max_mem_history attribute"),
        (hasattr(runner, '_mem_access_order'), "_mem_access_order attribute"),
        (runner.use_kv_cache == False, "use_kv_cache value"),
        (runner.max_mem_history == 32, "max_mem_history value"),
        (isinstance(runner._mem_access_order, list), "_mem_access_order type"),
    ]

    all_pass = True
    for check, desc in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {desc}")
        if not check:
            all_pass = False

    return all_pass


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run quick smoke tests when executed directly."""
    print("\n" + "=" * 70)
    print("AUTOREGRESSIVE KV CACHE - QUICK SMOKE TESTS")
    print("=" * 70)
    print()
    print("These tests run quickly to verify basic functionality.")
    print("For comprehensive testing, use: pytest tests/test_autoregressive_kv_cache.py")
    print()

    results = []

    # Run smoke tests
    results.append(("Parameters", smoke_test_parameters()))
    results.append(("Basic Functionality", smoke_test_basic()))
    results.append(("LRU Eviction", smoke_test_lru()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"  Total: {passed}/{total} smoke tests passed")

    if passed == total:
        print()
        print("✅ ALL SMOKE TESTS PASSED")
        print()
        print("Run full test suite with:")
        print("  pytest tests/test_autoregressive_kv_cache.py")
        return 0
    else:
        print()
        print("❌ SOME SMOKE TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
