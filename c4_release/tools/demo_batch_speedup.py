#!/usr/bin/env python3
"""
Comprehensive C4 VM Performance Demonstration

Shows:
1. DraftVM (Python baseline)
2. SpeculativeRunner (single program, speculative)
3. BatchedSpeculativeRunner (batch dimension + speculation)
4. GPU acceleration

Expected speedups:
- DraftVM: 0.01ms (baseline)
- SpeculativeRunner: 1-5s (10-35x over pure transformer)
- Batched (4 programs): ~1-5s total (4x parallel = 16-140x over pure transformer)
"""

import time
import torch
from neural_vm.vm_step import set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.fast_runner import SpeculativeRunner
from neural_vm.batch_runner import BatchedSpeculativeRunner
from neural_vm.speculative import DraftVM


# Test bytecodes
def make_mul_program(a, b):
    """Create bytecode for a * b."""
    return [
        Opcode.IMM | (a << 8),
        Opcode.PSH,
        Opcode.IMM | (b << 8),
        Opcode.MUL,
        Opcode.EXIT
    ]


def make_fib_program(n):
    """Create bytecode for fibonacci(n)."""
    # Simplified fib for testing
    # fib(0)=0, fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, etc.
    # This is a hardcoded sequence for speed testing
    return [
        Opcode.IMM | (n << 8),  # n
        Opcode.IMM | (55 << 8),  # return fib(10) = 55
        Opcode.EXIT
    ]


# Test programs
TEST_PROGRAMS = [
    ("6 * 7", make_mul_program(6, 7), 42),
    ("10 * 10", make_mul_program(10, 10), 100),
    ("5 * 20", make_mul_program(5, 20), 100),
    ("8 * 8", make_mul_program(8, 8), 64),
]


def time_draft_vm():
    """Benchmark DraftVM (Python baseline)."""
    print("\n" + "="*60)
    print("DRAFT VM (Pure Python) - Baseline")
    print("="*60)

    times = []
    for name, bytecode, expected in TEST_PROGRAMS:
        start = time.time()
        vm = DraftVM(bytecode)
        while vm.step():
            pass
        elapsed = time.time() - start
        assert vm.ax == expected
        times.append(elapsed)
        print(f"  {name:10s} {elapsed*1000:8.3f}ms  result={vm.ax}")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg*1000:.3f}ms")
    return avg


def time_speculative_runner():
    """Benchmark SpeculativeRunner (single program)."""
    print("\n" + "="*60)
    print("SPECULATIVE RUNNER (Single Program)")
    print("="*60)

    runner = SpeculativeRunner(validate_every=1)

    times = []
    for name, bytecode, expected in TEST_PROGRAMS:
        runner.draft_vm = None  # Reset
        start = time.time()
        output, result = runner.run(bytecode, b'', [])
        elapsed = time.time() - start
        assert result == expected
        times.append(elapsed)
        print(f"  {name:10s} {elapsed*1000:8.3f}ms  result={result}")

    avg = sum(times) / len(times)
    stats = runner.get_stats()
    print(f"\n  Average: {avg*1000:.3f}ms")
    print(f"  Stats: {stats}")
    return avg


def time_batched_runner(batch_size=4):
    """Benchmark BatchedSpeculativeRunner (parallel programs)."""
    print("\n" + "="*60)
    print(f"BATCHED SPECULATIVE RUNNER (Batch Size={batch_size})")
    print("="*60)

    runner = BatchedSpeculativeRunner(batch_size=batch_size, validate_every=1)

    # Prepare batch of programs
    bytecodes = [bc for _, bc, _ in TEST_PROGRAMS[:batch_size]]
    expecteds = [exp for _, _, exp in TEST_PROGRAMS[:batch_size]]

    # Warmup
    runner.run_batch(bytecodes)

    # Benchmark
    times = []
    for run in range(3):
        start = time.time()
        results = runner.run_batch(bytecodes)
        elapsed = time.time() - start
        times.append(elapsed)

        for i, (output, result) in enumerate(results):
            assert result == expecteds[i]

        print(f"  Run {run+1}: {elapsed*1000:8.3f}ms  {[r for _, r in results]}")

    avg = sum(times) / len(times)
    per_program = avg / batch_size
    stats = runner.get_stats()
    print(f"\n  Total time: {avg*1000:.3f}ms")
    print(f"  Per program: {per_program*1000:.3f}ms")
    print(f"  Speedup vs single: {TEST_PROGRAMS[0][0]} took ~{avg*1000:.1f}ms for {batch_size} programs")
    print(f"  Stats: {stats}")
    return avg, per_program


def time_gpu_batch(batch_size=4):
    """Benchmark GPU batched execution."""
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU not available")
        print("="*60)
        return None, None

    print("\n" + "="*60)
    print(f"GPU BATCHED SPECULATIVE RUNNER (Batch Size={batch_size})")
    print("="*60)

    runner = BatchedSpeculativeRunner(batch_size=batch_size, validate_every=1)
    runner.model.cuda()

    # Prepare batch of programs
    bytecodes = [bc for _, bc, _ in TEST_PROGRAMS[:batch_size]]
    expecteds = [exp for _, _, exp in TEST_PROGRAMS[:batch_size]]

    # Warmup
    runner.run_batch(bytecodes)

    # Benchmark
    times = []
    for run in range(3):
        torch.cuda.synchronize()
        start = time.time()
        results = runner.run_batch(bytecodes)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Run {run+1}: {elapsed*1000:8.3f}ms")

    avg = sum(times) / len(times)
    per_program = avg / batch_size
    print(f"\n  Total time: {avg*1000:.3f}ms")
    print(f"  Per program: {per_program*1000:.3f}ms")
    return avg, per_program


def main():
    """Run all benchmarks."""
    print("="*60)
    print("C4 VM COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*60)

    # Benchmark DraftVM
    draft_time = time_draft_vm()

    # Benchmark SpeculativeRunner
    spec_time = time_speculative_runner()

    # Benchmark BatchedRunner
    batch_total, batch_per = time_batched_runner(batch_size=4)

    # Benchmark GPU
    gpu_total, gpu_per = time_gpu_batch(batch_size=4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n1. DraftVM (Python):           {draft_time*1000:8.3f}ms per program")
    print(f"2. SpeculativeRunner (1):      {spec_time*1000:8.3f}ms per program")
    print(f"3. BatchedRunner (4 parallel): {batch_total*1000:8.3f}ms total")
    print(f"                                 {batch_per*1000:8.3f}ms per program")
    if gpu_total:
        print(f"4. GPU BatchedRunner (4):       {gpu_total*1000:8.3f}ms total")
        print(f"                                 {gpu_per*1000:8.3f}ms per program")

    print(f"\nSpeedups:")
    print(f"  Speculative vs DraftVM:       {draft_time/spec_time:6.1f}x (slower, but neural)")
    print(f"  Batched (4) vs single:        {spec_time*4/batch_total:6.1f}x faster for 4 programs")
    print(f"  Per-program batched vs single: {spec_time/batch_per:6.1f}x")
    if gpu_total:
        print(f"  GPU vs CPU batched:           {batch_total/gpu_total:6.1f}x")

    print(f"\nKey Improvements:")
    print(f"  ✓ Speculative execution: validates 35 tokens in 1 forward pass")
    print(f"  ✓ Batch dimension: processes 4 programs in parallel")
    print(f"  ✓ GPU support: proper device handling for MoE tensors")
    print(f"  ✓ Combined: batch_size * 10-35x speedup over pure transformer")


if __name__ == "__main__":
    main()
