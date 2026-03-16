"""Ultra-fast opcode tests using speculative batch execution.

Uses UltraBatchRunner which:
1. Executes 256 programs in parallel via batch dimension
2. Uses DraftVM for fast Python execution
3. Validates all programs in ONE transformer forward pass per step

Expected speedup: 100x+ over original tests
"""

import unittest
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.batch_runner_v2 import UltraBatchRunner, UltraBatchRunnerCached, run_batch_ultra


def run_programs_batch_ultra(bytecodes_list, batch_size=256):
    """Run multiple programs using ultra-fast speculative batch execution.

    Returns: list of exit codes
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    runner = UltraBatchRunner(batch_size=batch_size, device=device)
    return runner.run_batch(bytecodes_list)


# =============================================================================
# Shared model cache
# =============================================================================

_shared_model = None

def _get_model():
    global _shared_model
    if _shared_model is None:
        _shared_model = AutoregressiveVM()
        set_vm_weights(_shared_model)
        _shared_model.compact(block_size=32)
        _shared_model.compact_moe()
        if torch.cuda.is_available():
            _shared_model.cuda()
    return _shared_model


# =============================================================================
# Caches for fast test execution
# =============================================================================

_imm_cache = None
_mul_cache = None
_binop_cache = None


def _get_imm_cache():
    """Cache IMM v; EXIT results for v in 0..255."""
    global _imm_cache
    if _imm_cache is None:
        bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]
        _imm_cache = run_programs_batch_ultra(bytecodes, batch_size=256)
    return _imm_cache


def _get_mul_cache():
    """Cache MUL results for common values."""
    global _mul_cache
    if _mul_cache is None:
        # Generate all pairs of values 0-15
        pairs = [(a, b) for a in range(16) for b in range(16)]
        bytecodes = [
            [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8), Opcode.MUL, Opcode.EXIT]
            for a, b in pairs
        ]
        results = run_programs_batch_ultra(bytecodes, batch_size=256)
        _mul_cache = {pairs[i]: results[i] for i in range(len(pairs))}
    return _mul_cache


def _get_binop_cache(op):
    """Cache binary op results for common values."""
    key = f'_binop_{op}_cache'
    cache = globals().get(key)
    if cache is None:
        values = [0, 1, 5, 10, 15, 100, 127, 128, 255]
        pairs = [(a, b) for a in values for b in values]
        bytecodes = [
            [Opcode.IMM | (a << 8), Opcode.PSH, Opcode.IMM | (b << 8), op, Opcode.EXIT]
            for a, b in pairs
        ]
        results = run_programs_batch_ultra(bytecodes, batch_size=len(bytecodes))
        cache = {pairs[i]: results[i] for i in range(len(pairs))}
        globals()[key] = cache
    return cache


# =============================================================================
# IMM Exit Code Tests (256 tests)
# =============================================================================

class TestIMMExitCodes(unittest.TestCase):
    """Test IMM immediate value loading and exit codes."""

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_imm_cache()

    def test_imm_000_exit(self):
        self.assertEqual(self.cache[0], 0)

    def test_imm_001_exit(self):
        self.assertEqual(self.cache[1], 1)

    def test_imm_002_exit(self):
        self.assertEqual(self.cache[2], 2)

    def test_imm_042_exit(self):
        self.assertEqual(self.cache[42], 42)

    def test_imm_127_exit(self):
        self.assertEqual(self.cache[127], 127)

    def test_imm_128_exit(self):
        self.assertEqual(self.cache[128], 128)

    def test_imm_255_exit(self):
        self.assertEqual(self.cache[255], 255)


# =============================================================================
# Multiplication Tests
# =============================================================================

class TestMultiplication(unittest.TestCase):
    """Test MUL opcode."""

    @classmethod
    def setUpClass(cls):
        cls.cache = _get_mul_cache()

    def test_mul_0_0(self):
        self.assertEqual(self.cache[(0, 0)], 0)

    def test_mul_0_255(self):
        self.assertEqual(self.cache[(0, 255)], 0)

    def test_mul_1_1(self):
        self.assertEqual(self.cache[(1, 1)], 1)

    def test_mul_2_3(self):
        self.assertEqual(self.cache[(2, 3)], 6)

    def test_mul_5_5(self):
        self.assertEqual(self.cache[(5, 5)], 25)

    def test_mul_10_10(self):
        self.assertEqual(self.cache[(10, 10)], 100)

    def test_mul_15_15(self):
        self.assertEqual(self.cache[(15, 15)], 225)

    def test_mul_6_7(self):
        self.assertEqual(self.cache[(6, 7)], 42)


# =============================================================================
# Binary Operation Tests
# =============================================================================

class TestBinaryOps(unittest.TestCase):
    """Test binary operations: ADD, SUB, MUL, DIV, MOD, AND, OR, XOR."""

    def test_add_0_0(self):
        cache = _get_binop_cache(Opcode.ADD)
        self.assertEqual(cache[(0, 0)], 0)

    def test_add_1_1(self):
        cache = _get_binop_cache(Opcode.ADD)
        self.assertEqual(cache[(1, 1)], 2)

    def test_add_100_50(self):
        cache = _get_binop_cache(Opcode.ADD)
        self.assertEqual(cache[(100, 50)], 150)

    def test_sub_10_3(self):
        cache = _get_binop_cache(Opcode.SUB)
        self.assertEqual(cache[(10, 3)], 7)

    def test_sub_100_100(self):
        cache = _get_binop_cache(Opcode.SUB)
        self.assertEqual(cache[(100, 100)], 0)

    def test_mul_3_4(self):
        cache = _get_binop_cache(Opcode.MUL)
        self.assertEqual(cache[(3, 4)], 12)

    def test_mul_15_17(self):
        cache = _get_binop_cache(Opcode.MUL)
        self.assertEqual(cache[(15, 17)], 255)

    def test_div_10_3(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(10, 3)], 3)

    def test_div_0_5(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(0, 5)], 0)

    def test_div_100_10(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(100, 10)], 10)

    def test_mod_10_3(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(10, 3)], 1)

    def test_mod_255_16(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(255, 16)], 15)

    def test_and_255_15(self):
        cache = _get_binop_cache(Opcode.AND)
        self.assertEqual(cache[(255, 15)], 15)

    def test_or_0_0(self):
        cache = _get_binop_cache(Opcode.OR)
        self.assertEqual(cache[(0, 0)], 0)

    def test_or_240_15(self):
        cache = _get_binop_cache(Opcode.OR)
        self.assertEqual(cache[(240, 15)], 255)

    def test_xor_255_255(self):
        cache = _get_binop_cache(Opcode.XOR)
        self.assertEqual(cache[(255, 255)], 0)

    def test_xor_240_15(self):
        cache = _get_binop_cache(Opcode.XOR)
        self.assertEqual(cache[(240, 15)], 255)


# =============================================================================
# Division/Modulo Tests
# =============================================================================

class TestDivMod(unittest.TestCase):
    """Test DIV and MOD opcodes with edge cases."""

    def test_div_3_5(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(3, 5)], 0)

    def test_div_7_7(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(7, 7)], 1)

    def test_div_42_0(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(42, 0)], 0)

    def test_div_255_1(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(255, 1)], 255)

    def test_div_200_50(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(200, 50)], 4)

    def test_div_255_16(self):
        cache = _get_binop_cache(Opcode.DIV)
        self.assertEqual(cache[(255, 16)], 15)

    def test_mod_0_5(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(0, 5)], 0)

    def test_mod_3_5(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(3, 5)], 3)

    def test_mod_7_7(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(7, 7)], 0)

    def test_mod_42_0(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(42, 0)], 0)

    def test_mod_100_10(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(100, 10)], 0)

    def test_mod_200_50(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(200, 50)], 0)

    def test_mod_255_1(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(255, 1)], 0)

    def test_mod_255_16(self):
        cache = _get_binop_cache(Opcode.MOD)
        self.assertEqual(cache[(255, 16)], 15)


# =============================================================================
# Comparison Tests
# =============================================================================

class TestComparisons(unittest.TestCase):
    """Test comparison operations: EQ, NE, LT, GT, LE, GE."""

    def test_eq_5_5(self):
        cache = _get_binop_cache(Opcode.EQ)
        self.assertEqual(cache[(5, 5)], 1)

    def test_eq_5_3(self):
        cache = _get_binop_cache(Opcode.EQ)
        self.assertEqual(cache[(5, 3)], 0)

    def test_ne_5_3(self):
        cache = _get_binop_cache(Opcode.NE)
        self.assertEqual(cache[(5, 3)], 1)

    def test_ne_5_5(self):
        cache = _get_binop_cache(Opcode.NE)
        self.assertEqual(cache[(5, 5)], 0)

    def test_lt_3_5(self):
        cache = _get_binop_cache(Opcode.LT)
        self.assertEqual(cache[(3, 5)], 1)

    def test_lt_5_3(self):
        cache = _get_binop_cache(Opcode.LT)
        self.assertEqual(cache[(5, 3)], 0)

    def test_gt_5_3(self):
        cache = _get_binop_cache(Opcode.GT)
        self.assertEqual(cache[(5, 3)], 1)

    def test_gt_3_5(self):
        cache = _get_binop_cache(Opcode.GT)
        self.assertEqual(cache[(3, 5)], 0)

    def test_le_5_5(self):
        cache = _get_binop_cache(Opcode.LE)
        self.assertEqual(cache[(5, 5)], 1)

    def test_le_5_3(self):
        cache = _get_binop_cache(Opcode.LE)
        self.assertEqual(cache[(5, 3)], 0)

    def test_ge_5_5(self):
        cache = _get_binop_cache(Opcode.GE)
        self.assertEqual(cache[(5, 5)], 1)

    def test_ge_3_5(self):
        cache = _get_binop_cache(Opcode.GE)
        self.assertEqual(cache[(3, 5)], 0)


# =============================================================================
# Performance Benchmark (run as part of tests)
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Benchmark test to measure speedup."""

    def test_batch_256_immediate(self):
        """Test running 256 IMM programs in a single batch."""
        import time

        bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        start = time.time()
        results = run_programs_batch_ultra(bytecodes, batch_size=256)
        elapsed = time.time() - start

        # Verify correctness
        for v, result in enumerate(results):
            self.assertEqual(result, v)

        print(f"\n256 IMM programs in {elapsed*1000:.1f}ms")
        print(f"  Average: {elapsed*1000/256:.3f}ms per program")
        print(f"  Device: {device}")

        # Should be very fast with batching
        if device == 'cuda':
            self.assertLess(elapsed, 5.0, "256 programs should complete in <5s on GPU")
        else:
            self.assertLess(elapsed, 30.0, "256 programs should complete in <30s on CPU")


if __name__ == '__main__':
    unittest.main()
