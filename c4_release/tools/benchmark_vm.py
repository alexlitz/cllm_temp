"""
Benchmark C4 VM Performance

Measures:
- Baseline transformer execution (slow)
- Speculative execution (should be 10x faster)
- CPU vs GPU
"""

import torch
import time
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.speculative import DraftVM

# Test bytecode: 6 * 7 = 42
BYTECODE_MUL = [
    Opcode.IMM | (6 << 8),
    Opcode.PSH,
    Opcode.IMM | (7 << 8),
    Opcode.MUL,
    Opcode.EXIT
]

def benchmark_transformer_runner():
    """Benchmark current transformer runner (no speculation)."""
    print("=" * 60)
    print("BENCHMARK: Transformer Runner (No Speculation)")
    print("=" * 60)

    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    runner.model.compact(block_size=32)
    runner.model.compact_moe()

    # Warmup
    runner.run(BYTECODE_MUL, b'', [])

    # Benchmark
    times = []
    for i in range(5):
        start = time.time()
        output, result = runner.run(BYTECODE_MUL, b'', [])
        elapsed = time.time() - start
        times.append(elapsed)
        assert result == 42, f"Wrong result: {result}"
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.3f}s")
    print(f"  Min: {min(times):.3f}s")
    print(f"  Max: {max(times):.3f}s")
    return avg


def benchmark_draft_vm():
    """Benchmark DraftVM (pure Python, no transformer)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: DraftVM (Pure Python)")
    print("=" * 60)

    times = []
    for i in range(5):
        vm = DraftVM(BYTECODE_MUL)
        start = time.time()
        while vm.step():
            pass
        elapsed = time.time() - start
        times.append(elapsed)
        assert vm.ax == 42, f"Wrong result: {vm.ax}"
        print(f"  Run {i+1}: {elapsed*1000:.3f}ms")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg*1000:.3f}ms")
    print(f"  Min: {min(times)*1000:.3f}ms")
    print(f"  Max: {max(times)*1000:.3f}ms")
    return avg


def count_transformer_steps():
    """Count how many transformer steps the current runner takes."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Counting Transformer Steps")
    print("=" * 60)

    # Monkey-patch to count steps
    original_generate = AutoregressiveVMRunner.model.__class__.generate_next
    step_count = [0]

    def counting_generate(self, context, temperature=0.0):
        step_count[0] += 1
        return original_generate(self, context, temperature)

    AutoregressiveVMRunner.model.__class__.generate_next = counting_generate

    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    runner.model.compact(block_size=32)
    runner.model.compact_moe()

    output, result = runner.run(BYTECODE_MUL, b'', [])
    print(f"  Result: {result}")
    print(f"  Total generate_next() calls: {step_count[0]}")
    print(f"  Tokens per call: 1 (no batching)")

    # Restore
    AutoregressiveVMRunner.model.__class__.generate_next = original_generate

    return step_count[0]


def benchmark_gpu():
    """Benchmark GPU execution."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU not available")
        print("=" * 60)
        return None

    print("\n" + "=" * 60)
    print("BENCHMARK: GPU (No Speculation)")
    print("=" * 60)

    runner = AutoregressiveVMRunner()
    set_vm_weights(runner.model)
    runner.model.compact(block_size=32)
    runner.model.compact_moe()
    runner.model.cuda()

    # Warmup
    runner.run(BYTECODE_MUL, b'', [])

    times = []
    for i in range(5):
        torch.cuda.synchronize()
        start = time.time()
        output, result = runner.run(BYTECODE_MUL, b'', [])
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        assert result == 42, f"Wrong result: {result}"
        print(f"  Run {i+1}: {elapsed:.3f}s")

    avg = sum(times) / len(times)
    print(f"\n  Average: {avg:.3f}s")
    print(f"  Min: {min(times):.3f}s")
    print(f"  Max: {max(times):.3f}s")
    return avg


if __name__ == "__main__":
    # Run benchmarks
    draft_time = benchmark_draft_vm()
    step_count = count_transformer_steps()
    transformer_time = benchmark_transformer_runner()
    gpu_time = benchmark_gpu()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  DraftVM (Python):         {draft_time*1000:.3f}ms")
    print(f"  Transformer steps taken:  {step_count}")
    print(f"  Transformer (CPU):        {transformer_time:.3f}s")
    if gpu_time:
        print(f"  Transformer (GPU):        {gpu_time:.3f}s")
        print(f"  GPU speedup:              {transformer_time/gpu_time:.2f}x")
    print(f"  Slowdown vs Python:       {transformer_time/draft_time:.0f}x")

    print("\n  ISSUES:")
    print(f"    - No batching (1 token per forward pass)")
    print(f"    - No speculative execution (uses DraftVM for draft)")
    print(f"    - {step_count} transformer calls for simple 6*7 operation")
