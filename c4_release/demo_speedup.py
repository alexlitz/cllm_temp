#!/usr/bin/env python3
"""
Demonstrate C4 VM Performance Improvements

This script shows:
1. DraftVM (pure Python) - baseline fastest
2. SpeculativeRunner - uses DraftVM + transformer validation
3. Original Runner - pure transformer (slow)
4. GPU speedup

Expected results:
- DraftVM: ~0.01ms
- SpeculativeRunner: ~1-5s (10-35x faster than pure transformer)
- Original Runner: ~50-300s (baseline)
"""

import time
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.fast_runner import SpeculativeRunner
from neural_vm.speculative import DraftVM

# Test bytecode: fibonacci(10) = 55
FIB_BYTECODE = """
int fib(int n) {
    if (n < 2) return n;
    return fib(n-1) + fib(n-2);
}
int main() { return fib(10); }
"""

# Simple test: 6 * 7 = 42
MUL_BYTECODE = [
    Opcode.IMM | (6 << 8),
    Opcode.PSH,
    Opcode.IMM | (7 << 8),
    Opcode.MUL,
    Opcode.EXIT
]


def time_it(fn, label, warmup=1):
    """Time a function."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Time it
    start = time.time()
    result = fn()
    elapsed = time.time() - start

    print(f"{label:30s} {elapsed*1000:8.3f}ms  result={result}")
    return elapsed, result


def test_draft_vm():
    """Test DraftVM - pure Python, fastest."""
    print("\n" + "="*60)
    print("DRAFT VM (Pure Python) - Baseline")
    print("="*60)

    def run():
        vm = DraftVM(MUL_BYTECODE)
        while vm.step():
            pass
        return vm.ax

    time_it(run, "DraftVM 6*7")


def test_speculative_runner():
    """Test SpeculativeRunner - DraftVM + transformer validation."""
    print("\n" + "="*60)
    print("SPECULATIVE RUNNER (DraftVM + Transformer)")
    print("="*60)

    runner = SpeculativeRunner(validate_every=1)

    def run():
        runner.draft_vm = None  # Reset
        output, result = runner.run(MUL_BYTECODE, b'', [])
        return result

    elapsed, result = time_it(run, "SpeculativeRunner 6*7")
    stats = runner.get_stats()
    print(f"  Steps: {stats['total_steps']}")
    print(f"  Validations: {stats['validations']}")
    print(f"  Match rate: {stats['match_rate']*100:.1f}%")
    return elapsed


def test_original_runner():
    """Test original runner - pure transformer."""
    print("\n" + "="*60)
    print("ORIGINAL RUNNER (Pure Transformer)")
    print("="*60)
    print("WARNING: This is VERY slow (~50-300s)")

    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    runner.model.compact(block_size=32)
    runner.model.compact_moe()

    # Warmup
    runner.run(MUL_BYTECODE, b'', [])

    def run():
        output, result = runner.run(MUL_BYTECODE, b'', [])
        return result

    return time_it(run, "OriginalRunner 6*7")


def test_gpu():
    """Test GPU execution."""
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU not available")
        print("="*60)
        return None

    print("\n" + "="*60)
    print("GPU EXECUTION")
    print("="*60)

    runner = SpeculativeRunner(validate_every=1)
    runner.model.cuda()

    def run():
        runner.draft_vm = None  # Reset
        output, result = runner.run(MUL_BYTECODE, b'', [])
        return result

    return time_it(run, "GPU SpeculativeRunner 6*7")


def main():
    """Run all tests."""
    print("="*60)
    print("C4 VM PERFORMANCE DEMONSTRATION")
    print("="*60)

    # Test DraftVM
    test_draft_vm()

    # Test SpeculativeRunner
    spec_time = test_speculative_runner()

    # Test GPU
    gpu_time = test_gpu()

    # Test original runner (commented out because it's too slow)
    # orig_time = test_original_runner()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SpeculativeRunner: {spec_time*1000:.1f}ms")
    if gpu_time:
        print(f"GPU Speculative:  {gpu_time*1000:.1f}ms")
        print(f"GPU speedup:      {spec_time/gpu_time:.2f}x")
    print(f"\nSpeculativeRunner uses:")
    print("  - DraftVM for fast Python execution")
    print("  - Transformer validates 35 tokens in 1 forward pass")
    print("  - Expected 10-35x speedup over pure transformer")
    print(f"\nTo run original (slow) runner, uncomment test_original_runner()")


if __name__ == "__main__":
    main()
