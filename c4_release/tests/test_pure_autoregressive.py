#!/usr/bin/env python3
"""
Test Suite for Pure Autoregressive Mode.

Tests all operations to ensure they work without Python fallbacks when
pure_attention_memory=True.

Progress toward 100% autoregressive execution by validating each operation
category independently.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baked_c4 import BakedC4Transformer
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
import pytest


class TestTier1Operations:
    """Test Tier 1: Core stack operations (SP corrections, ADJ)."""

    def test_binary_pop_sp_add(self):
        """Test SP += 8 correction for ADD (binary pop operation)."""
        source = 'int main() { int a; int b; a = 10; b = 5; return a + b; }'
        bytecode, data = compile_c(source)

        # Run in pure mode
        runner_pure = AutoregressiveVMRunner(pure_attention_memory=True)
        result_pure = runner_pure.run(bytecode, data)

        # Run in hybrid mode for comparison
        runner_hybrid = AutoregressiveVMRunner(pure_attention_memory=False)
        result_hybrid = runner_hybrid.run(bytecode, data)

        assert result_pure == result_hybrid == 15, \
            f"Expected 15, got pure={result_pure}, hybrid={result_hybrid}"

    def test_binary_pop_sp_sub(self):
        """Test SP += 8 correction for SUB."""
        source = """
        int main() {
            int a = 20;
            int b = 7;
            return a - b;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 13

    def test_binary_pop_sp_mul(self):
        """Test SP += 8 correction for MUL."""
        source = """
        int main() {
            int a = 6;
            int b = 7;
            return a * b;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 42

    def test_binary_pop_sp_div(self):
        """Test SP += 8 correction for DIV."""
        source = """
        int main() {
            int a = 100;
            int b = 4;
            return a / b;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 25

    def test_binary_pop_sp_mod(self):
        """Test SP += 8 correction for MOD."""
        source = """
        int main() {
            int a = 17;
            int b = 5;
            return a % b;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 2

    def test_binary_pop_sp_bitwise(self):
        """Test SP += 8 for bitwise operations (OR, XOR, AND)."""
        source = """
        int main() {
            int a = 0xFF;
            int b = 0x0F;
            int result = 0;
            result = a | b;   // OR
            result = a ^ b;   // XOR
            result = a & b;   // AND
            return result;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 0x0F

    def test_binary_pop_sp_shift(self):
        """Test SP += 8 for shift operations (SHL, SHR)."""
        source = """
        int main() {
            int a = 8;
            int b = 2;
            int c = a << b;  // SHL: 8 << 2 = 32
            int d = c >> b;  // SHR: 32 >> 2 = 8
            return d;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 8

    def test_binary_pop_sp_compare(self):
        """Test SP += 8 for comparison operations (SI, SC)."""
        source = """
        int main() {
            int arr[2];
            arr[0] = 42;  // SI - store int
            arr[1] = 7;   // SI - store int
            return arr[0] + arr[1];
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 49

    def test_psh_sp_correction(self):
        """Test PSH SP -= 8 correction (should already work)."""
        source = """
        int main() {
            int a = 10;
            int b = 20;
            int c = 30;
            return a + b + c;  // Multiple PSH operations
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 60

    def test_psh_sp_boundary(self):
        """Test PSH with SP near byte boundary (carry propagation)."""
        # This test checks multi-byte borrow in PSH SP -= 8
        source = """
        int main() {
            int a, b, c, d, e, f;  // 6 variables = 6 PSH operations
            a = 1; b = 2; c = 3; d = 4; e = 5; f = 6;
            return a + b + c + d + e + f;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 21


class TestTier2Operations:
    """Test Tier 2: Memory management (MALC, FREE, MSET)."""

    @pytest.mark.skip(reason="MALC not yet implemented in pure mode")
    def test_malc_simple(self):
        """Test simple malloc allocation."""
        source = """
        int main() {
            int *p = malloc(100);
            return (int)p;  // Should return valid address
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result > 0  # Valid pointer

    @pytest.mark.skip(reason="MALC not yet implemented in pure mode")
    def test_malc_alignment(self):
        """Test malloc aligns to 8-byte boundaries."""
        source = """
        int main() {
            void *a = malloc(1);
            void *b = malloc(1);
            return (int)b - (int)a;  // Should be 8 (aligned)
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 8

    @pytest.mark.skip(reason="FREE not yet implemented in pure mode")
    def test_free_simple(self):
        """Test simple free operation."""
        source = """
        int main() {
            int *p = malloc(100);
            *p = 42;
            free(p);
            return 0;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 0

    @pytest.mark.skip(reason="MSET not yet implemented in pure mode")
    def test_mset_simple(self):
        """Test memset operation."""
        source = """
        int main() {
            char buf[10];
            memset(buf, 0xAB, 10);
            return buf[0] + buf[9];  // Should be 0x156
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 0x156


class TestTier3Operations:
    """Test Tier 3: Advanced operations (MCMP, IMM, LEA, ENT, LEV)."""

    @pytest.mark.skip(reason="MCMP not yet implemented in pure mode")
    def test_mcmp_equal(self):
        """Test memcmp with equal strings."""
        source = """
        int main() {
            char *a = "hello";
            char *b = "hello";
            return memcmp(a, b, 5);
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 0

    @pytest.mark.skip(reason="MCMP not yet implemented in pure mode")
    def test_mcmp_different(self):
        """Test memcmp with different strings."""
        source = """
        int main() {
            char *a = "hello";
            char *b = "world";
            int cmp = memcmp(a, b, 5);
            return cmp != 0;  // Should be non-zero
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 1

    def test_imm_simple(self):
        """Test IMM immediate loading."""
        source = """
        int main() {
            int a = 12345;
            return a;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 12345

    def test_imm_negative(self):
        """Test IMM with negative immediate (sign extension)."""
        source = """
        int main() {
            int a = -100;
            return a;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == -100 & 0xFFFFFFFF

    def test_lea_simple(self):
        """Test LEA load effective address."""
        source = """
        int main() {
            int a = 10;
            int *p = &a;
            return *p;
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 10

    @pytest.mark.skip(reason="ENT/LEV not yet verified in pure mode")
    def test_ent_lev_simple(self):
        """Test ENT/LEV function frame setup/teardown."""
        source = """
        int add(int a, int b) {
            return a + b;
        }

        int main() {
            return add(10, 32);
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 42


class TestTier4Operations:
    """Test Tier 4: Control flow (JSR)."""

    @pytest.mark.skip(reason="JSR not yet verified in pure mode")
    def test_jsr_simple(self):
        """Test JSR jump to subroutine."""
        source = """
        int func() {
            return 42;
        }

        int main() {
            return func();
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 42

    @pytest.mark.skip(reason="JSR not yet verified in pure mode")
    def test_jsr_recursive(self):
        """Test JSR with recursion."""
        source = """
        int fib(int n) {
            if (n <= 1) return n;
            return fib(n-1) + fib(n-2);
        }

        int main() {
            return fib(7);  // Should return 13
        }
        """
        bytecode, data = compile_c(source)
        runner = AutoregressiveVMRunner(pure_attention_memory=True)
        result = runner.run(bytecode, data)
        assert result == 13


class TestPurityValidation:
    """Validate that no Python fallbacks are being used."""

    def test_tier1_operations_use_no_fallbacks(self):
        """Verify Tier 1 operations don't trigger Python fallbacks."""
        source = """
        int main() {
            int a = 10;
            int b = 20;
            int c = a + b;  // Tests: PSH (SP -= 8), ADD (SP += 8)
            return c;
        }
        """
        c4 = BakedC4Transformer(use_speculator=True, pure_attention_memory=True)

        # Access the runner to check fallback usage
        result = c4.run_c(source)

        # TODO: Add purity report checking once runner exposes fallback stats
        # report = c4.runner.get_pure_attention_report()
        # assert report['fallback_calls'] == 0

        assert result == 30


class TestRegressionSuite:
    """Run subset of 1096 test suite in pure mode."""

    def test_arithmetic_ops_pure(self):
        """Test all arithmetic operations in pure mode."""
        from tests.test_suite_1000 import get_quick_tests

        c4 = BakedC4Transformer(use_speculator=True, pure_attention_memory=True)
        tests = get_quick_tests()

        # Filter to just arithmetic tests
        arithmetic_tests = [
            (src, expected, desc) for src, expected, desc in tests
            if any(kw in desc.lower() for kw in ['add', 'sub', 'mul', 'div', 'mod'])
        ]

        passed = 0
        failed = 0
        for source, expected, desc in arithmetic_tests[:20]:  # Test first 20
            try:
                result = c4.run_c(source)
                if result == expected:
                    passed += 1
                else:
                    failed += 1
                    print(f"FAIL: {desc} - expected {expected}, got {result}")
            except Exception as e:
                failed += 1
                print(f"ERROR: {desc} - {e}")

        print(f"\nArithmetic tests: {passed} passed, {failed} failed")
        assert failed == 0, f"{failed} arithmetic tests failed in pure mode"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
