#!/usr/bin/env python3
"""
Simple batched performance test for C4 VM.

Tests 256 programs running in parallel using:
- DraftVM (Python baseline)
- UltraBatchRunner with speculative execution
"""

import time
import sys
sys.path.insert(0, 'neural_vm')

from neural_vm.speculative import DraftVM
from neural_vm.batch_runner_v2 import run_batch_ultra
from neural_vm.embedding import Opcode


def test_draft_vm_baseline():
    """Test DraftVM baseline (pure Python)."""
    print("=" * 60)
    print("DRAFT VM (Python Baseline) - 256 programs")
    print("=" * 60)

    # Create 256 simple programs: IMM v; EXIT
    bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]

    start = time.time()
    results = []
    for bc in bytecodes:
        vm = DraftVM(bc)
        while vm.step():
            pass
        results.append(vm.ax)
    elapsed = time.time() - start

    # Verify correctness
    for v, result in enumerate(results):
        assert result == v, f"Expected {v}, got {result}"

    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Average: {elapsed*1000/256:.3f}ms per program")
    return elapsed


def test_ultra_batch():
    """Test UltraBatchRunner with speculative execution."""
    print("\n" + "=" * 60)
    print("ULTRA BATCH RUNNER - 256 programs in parallel")
    print("=" * 60)

    # Create 256 simple programs: IMM v; EXIT
    bytecodes = [[Opcode.IMM | (v << 8), Opcode.EXIT] for v in range(256)]

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    start = time.time()
    results = run_batch_ultra(bytecodes, batch_size=256)
    elapsed = time.time() - start

    # Verify correctness
    for v, result in enumerate(results):
        assert result == v, f"Expected {v}, got {result}"

    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Average: {elapsed*1000/256:.3f}ms per program")
    return elapsed


def test_mul_batch():
    """Test multiplication operations in batch."""
    print("\n" + "=" * 60)
    print("MUL TEST - 256 multiplication programs")
    print("=" * 60)

    # Create 256 multiplication programs
    bytecodes = []
    expected = []
    for a in range(16):
        for b in range(16):
            bc = [
                Opcode.IMM | (a << 8),
                Opcode.PSH,
                Opcode.IMM | (b << 8),
                Opcode.MUL,
                Opcode.EXIT
            ]
            bytecodes.append(bc)
            expected.append(a * b)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    start = time.time()
    results = run_batch_ultra(bytecodes, batch_size=256)
    elapsed = time.time() - start

    # Verify correctness
    for i, (result, exp) in enumerate(zip(results, expected)):
        assert result == exp, f"Program {i}: expected {exp}, got {result}"

    print(f"Time: {elapsed*1000:.1f}ms")
    print(f"Average: {elapsed*1000/256:.3f}ms per program")
    return elapsed


def main():
    print("C4 VM BATCHED PERFORMANCE TEST")
    print()

    draft_time = test_draft_vm_baseline()
    ultra_time = test_ultra_batch()
    mul_time = test_mul_batch()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DraftVM (serial):     {draft_time*1000:8.1f}ms for 256 programs")
    print(f"UltraBatchRunner:     {ultra_time*1000:8.1f}ms for 256 programs")
    print(f"MUL (256 programs):    {mul_time*1000:8.1f}ms")
    print()
    print(f"Speedup: {draft_time/ultra_time:.1f}x")


if __name__ == "__main__":
    main()
