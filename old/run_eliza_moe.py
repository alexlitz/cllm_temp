#!/usr/bin/env python3
"""
Run ELIZA chatbot through the full MoE Neural VM.

This demonstrates running a text-based conversational AI program
through a neural network that simulates a processor.
"""

import sys
import os
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c4_compiler_full import Compiler
from c4_moe_vm import C4MoEVM


def main():
    print("=" * 70)
    print("  ELIZA CHATBOT via FULL MoE NEURAL VM")
    print("=" * 70)
    print()
    print("This runs the classic 1966 ELIZA chatbot through a neural network")
    print("that simulates a processor using Mixture of Experts routing.")
    print()
    print("All arithmetic operations use transformer primitives:")
    print("  - Multiplications: SwiGLU (silu(a)*b + silu(-a)*(-b))")
    print("  - Divisions: Newton-Raphson via MUL only")
    print("  - Bitwise ops: Attention-based table lookup")
    print()

    # Read ELIZA source (uses printf("%s", ...) for MoE VM compatibility)
    c_file = os.path.join(os.path.dirname(__file__), "eliza_moe_fixed.c")
    with open(c_file, 'r') as f:
        source = f.read()

    print("Compiling ELIZA to bytecode...")
    start = time.time()
    compiler = Compiler()
    bytecode, data_segment = compiler.compile(source)
    compile_time = time.time() - start
    print(f"Compiled in {compile_time:.2f}s ({len(bytecode)} instructions)")
    print()

    print("Executing through MoE Neural VM...")
    print("-" * 70)

    vm = C4MoEVM()
    vm.load(bytecode, data_segment)

    start = time.time()
    # Use fast=True for reasonable speed while still using MoE for MUL/DIV
    exit_code, output, stats = vm.run(max_steps=10000000, fast=True)
    exec_time = time.time() - start

    # Print ELIZA output
    print(output)
    print("-" * 70)
    print()

    print("=" * 70)
    print("  EXECUTION REPORT")
    print("=" * 70)
    print(f"Exit code: {exit_code}")
    print(f"Execution time: {exec_time:.2f}s")
    print(f"Total steps: {stats['steps']:,}")
    print()

    # Operation counts from expert usage
    print("NEURAL OPERATION COUNTS (from expert routing):")
    expert_usage = stats.get('expert_usage', None)

    if expert_usage is not None:
        import torch
        if isinstance(expert_usage, torch.Tensor):
            mul_count = int(expert_usage[27].item()) if len(expert_usage) > 27 else 0
            div_count = int(expert_usage[28].item()) if len(expert_usage) > 28 else 0
            add_count = int(expert_usage[25].item()) + int(expert_usage[26].item()) if len(expert_usage) > 26 else 0
        else:
            mul_count = expert_usage.get(27, 0)
            div_count = expert_usage.get(28, 0)
            add_count = expert_usage.get(25, 0) + expert_usage.get(26, 0)
    else:
        mul_count = 0
        div_count = 0
        add_count = 0

    print(f"  Multiplications (SwiGLU): {mul_count:,}")
    print(f"  Divisions:                {div_count:,}")
    print(f"  Additions/Subtractions:   {add_count:,}")
    print()

    # FLOP estimation
    swiglu_flops = mul_count * 10
    div_flops = div_count * 50
    total_flops = swiglu_flops + div_flops + add_count

    if total_flops > 0:
        print("ESTIMATED FLOPs:")
        print(f"  SwiGLU multiplications: {swiglu_flops:,}")
        print(f"  Divisions:              {div_flops:,}")
        print(f"  Other operations:       {add_count:,}")
        print(f"  TOTAL:                  {total_flops:,}")
        if exec_time > 0:
            print(f"  Throughput:             {total_flops / exec_time / 1e3:.2f} KFLOPs/s")

    print("=" * 70)


if __name__ == "__main__":
    main()
