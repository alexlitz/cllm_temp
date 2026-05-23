#!/usr/bin/env python3
"""
Integration Test for KV Cache in Main Test Suite

Quick test that can be added to the main test suite to verify
AutoregressiveVMRunner works with KV cache enabled.

This test is designed to run fast (< 30 seconds) as part of CI/CD.

Usage:
    pytest tests/test_kv_cache_integration.py
    python tests/test_kv_cache_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode


def _make_bytecode(ops):
    bytecode = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bytecode.append(opcode | (imm << 8))
        else:
            bytecode.append(op)
    return bytecode


class TestKVCacheIntegration:
    """Integration tests for KV cache in main test suite."""

    @pytest.fixture
    def compile_program(self):
        """Fixture to compile C programs."""
        return compile_c

    def test_smoke_autoregressive_vm(self, compile_program):
        """Smoke test for AutoregressiveVMRunner with KV cache."""
        bytecode = _make_bytecode([(Opcode.IMM, 42), Opcode.EXIT])

        runner = AutoregressiveVMRunner(
            pure_neural=True,
            trust_neural_alu=True,
            use_kv_cache=True,
            max_mem_history=64,
        )
        _, exit_code = runner.run(bytecode, b"", max_steps=10)

        assert exit_code == 42

    def test_kv_cache_on_vs_off(self, compile_program):
        """Verify cache ON and OFF produce same result."""
        bytecode = _make_bytecode([
            (Opcode.IMM, 10),
            Opcode.PSH,
            (Opcode.IMM, 20),
            Opcode.ADD,
            Opcode.EXIT,
        ])

        runner_on = AutoregressiveVMRunner(
            pure_neural=True,
            trust_neural_alu=True,
            use_kv_cache=True,
        )
        _, exit_on = runner_on.run(bytecode, b"", max_steps=20)

        runner_off = AutoregressiveVMRunner(
            pure_neural=True,
            trust_neural_alu=True,
            use_kv_cache=False,
        )
        _, exit_off = runner_off.run(bytecode, b"", max_steps=20)

        assert exit_on == exit_off == 30

    def test_lru_eviction_basic(self, compile_program):
        """Small-history setting must not break KV-cached neural execution."""
        bytecode = _make_bytecode([
            (Opcode.IMM, 1),
            Opcode.PSH,
            (Opcode.IMM, 2),
            Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 3),
            Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 4),
            Opcode.ADD,
            Opcode.PSH,
            (Opcode.IMM, 5),
            Opcode.ADD,
            Opcode.EXIT,
        ])

        runner = AutoregressiveVMRunner(
            pure_neural=True,
            trust_neural_alu=True,
            use_kv_cache=True,
            max_mem_history=2,
        )
        _, exit_code = runner.run(bytecode, b"", max_steps=60)

        assert exit_code == 15
        assert runner._kv_cache_obj is not None

    def test_parameters_initialized(self):
        """Verify parameters are properly initialized."""
        runner = AutoregressiveVMRunner(use_kv_cache=False, max_mem_history=32)

        assert runner.use_kv_cache == False
        assert runner.max_mem_history == 32
        assert hasattr(runner, '_mem_access_order')
        assert isinstance(runner._mem_access_order, list)


def main():
    """Run tests when executed directly."""
    print("\n" + "=" * 70)
    print("KV CACHE INTEGRATION TESTS")
    print("=" * 70)
    print()

    # Run with pytest
    exit_code = pytest.main([__file__, '-v'])

    print()
    if exit_code == 0:
        print("✅ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
