#!/usr/bin/env python3
"""
Run Mandelbrot through full MoE Neural VM with I/O support.

Uses the MoEVM with soft attention for instruction fetch/memory
and MoE routing for operation dispatch. All arithmetic is neural.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.moe_vm import MoEVM, MoEALU
from src.compiler import compile_c


class MoEVMWithIO(MoEVM):
    """MoE VM extended with I/O syscalls and operation counting."""

    SYS_GETCHAR = 64
    SYS_PUTCHAR = 65

    def __init__(self, top_k=1):
        super().__init__(top_k=top_k)
        self.stdout_bytes = []
        self.stdin_data = ""
        self.stdin_pos = 0

        # Operation counters
        self.mul_count = 0
        self.div_count = 0
        self.add_count = 0
        self.attention_ops = 0

    def reset(self):
        super().reset()
        self.stdout_bytes = []
        self.stdin_pos = 0

    def step(self) -> torch.Tensor:
        """Execute one step with I/O syscall handling."""
        # Fetch instruction
        op_enc, imm_enc = self._fetch()
        self.attention_ops += 1  # Count attention for fetch

        # Check for I/O syscalls (hard check on opcode)
        op_idx = int(torch.argmax(op_enc).item())

        if op_idx == self.SYS_PUTCHAR:
            # Get character from stack
            stack_val = self._mem_load(self.state.sp)
            self.attention_ops += 1  # Count attention for memory load
            c = self.alu.decode(stack_val) & 0xFF
            self.stdout_bytes.append(c)

            # Update SP (pop the argument)
            self.state.sp = self.alu.add(self.state.sp, self.eight)
            self.add_count += 1

            # Advance PC
            self.state.pc = self.alu.add(self.state.pc, self.eight)
            self.add_count += 1

            return self.state.halted

        elif op_idx == self.SYS_GETCHAR:
            if self.stdin_pos < len(self.stdin_data):
                c = ord(self.stdin_data[self.stdin_pos])
                self.stdin_pos += 1
            else:
                c = 0xFFFFFFFF  # EOF
            self.state.ax = self.alu.encode(c)
            self.state.pc = self.alu.add(self.state.pc, self.eight)
            self.add_count += 1
            return self.state.halted

        # Track operations in experts
        if op_idx == self.OP_MUL:
            self.mul_count += 1
        elif op_idx == self.OP_DIV:
            self.div_count += 1
        elif op_idx in (self.OP_ADD, self.OP_SUB):
            self.add_count += 1

        # Use parent step for other operations
        return super().step()

    def run(self, max_steps=10000000, verbose_interval=100000):
        """Run with progress reporting."""
        steps = 0
        last_report = 0

        while steps < max_steps:
            halted = self.step()
            if halted > 0.5:
                break
            steps += 1

            if verbose_interval and steps - last_report >= verbose_interval:
                print(f"\rSteps: {steps:,} | Muls: {self.mul_count:,} | "
                      f"Output bytes: {len(self.stdout_bytes):,}", end="", flush=True)
                last_report = steps

        return self.alu.decode(self.state.ax)


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_moe_mandelbrot.py WIDTH HEIGHT OUTPUT.png")
        print("Example: python run_moe_mandelbrot.py 32 32 test.png")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    output = sys.argv[3]

    print("=" * 70)
    print("  MANDELBROT via FULL MoE NEURAL VM")
    print("=" * 70)
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output}")
    print()

    # Read the Mandelbrot C source (putchar version)
    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file, 'r') as f:
        source = f.read()

    # Modify dimensions
    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling C to bytecode...")
    start = time.time()
    bytecode, data = compile_c(source)
    compile_time = time.time() - start
    print(f"Compiled in {compile_time:.2f}s ({len(bytecode)} instructions)")
    print()

    print("Executing through MoE Neural VM...")
    vm = MoEVMWithIO(top_k=1)
    vm.load_bytecode(bytecode, data)

    start = time.time()
    result = vm.run(max_steps=100000000, verbose_interval=500000)
    exec_time = time.time() - start

    print()
    print()
    print("=" * 70)
    print("  EXECUTION REPORT")
    print("=" * 70)
    print(f"Execution time: {exec_time:.2f}s")
    print(f"Output bytes: {len(vm.stdout_bytes):,}")
    print()
    print("NEURAL OPERATION COUNTS:")
    print(f"  Multiplications (SwiGLU):     {vm.mul_count:,}")
    print(f"  Divisions:                    {vm.div_count:,}")
    print(f"  Additions/Subtractions:       {vm.add_count:,}")
    print(f"  Attention operations:         {vm.attention_ops:,}")
    print()

    # FLOP estimation
    # MoE attention: ~hidden_dim * seq_len FLOPs per attention
    # SwiGLU: ~10 FLOPs (silu computation + multiply + add)
    # Division: ~50 FLOPs (iterative)
    # Each MoE step also has routing softmax
    swiglu_flops = vm.mul_count * 10
    div_flops = vm.div_count * 50
    attention_flops = vm.attention_ops * 1000  # ~1K FLOPs per attention op
    add_flops = vm.add_count
    total_flops = swiglu_flops + div_flops + attention_flops + add_flops

    print("ESTIMATED FLOPs:")
    print(f"  SwiGLU multiplications:  {swiglu_flops:,}")
    print(f"  Divisions:               {div_flops:,}")
    print(f"  Attention operations:    {attention_flops:,}")
    print(f"  Additions:               {add_flops:,}")
    print(f"  TOTAL:                   {total_flops:,}")
    print(f"  Throughput:              {total_flops / exec_time / 1e6:.2f} MFLOPs/s")
    print()

    # Save PNG
    if vm.stdout_bytes:
        with open(output, 'wb') as f:
            f.write(bytes(vm.stdout_bytes))
        print(f"Saved {len(vm.stdout_bytes):,} bytes to {output}")

        import subprocess
        result = subprocess.run(['file', output], capture_output=True, text=True)
        print(f"File type: {result.stdout.strip()}")
    else:
        print("ERROR: No output bytes generated!")

    print("=" * 70)


if __name__ == "__main__":
    main()
