#!/usr/bin/env python3
"""
Run Mandelbrot PNG generator through the C4 Transformer VM.

This executes the mandelbrot_color_png_c4.c program through the full
neural transformer, using SwiGLU for all multiplications.

Usage:
    python run_mandelbrot_transformer.py 64 64 output.png
    python run_mandelbrot_transformer.py 256 256 output.png
    python run_mandelbrot_transformer.py 2560 1440 output.png  # HD (very slow!)
"""

import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from src.transformer_vm import C4TransformerVM, NeuralALU


class TransformerProgramVM:
    """
    VM that runs programs through the transformer with I/O support.
    Tracks all neural operations for FLOP counting.
    """

    # C4 syscall opcodes
    SYS_MALC = 34
    SYS_FREE = 35
    SYS_MSET = 36
    SYS_EXIT = 38
    SYS_PUTCHAR = 65

    def __init__(self, use_neural: bool = True):
        self.heap = 0x20000  # Heap start address
        self.use_neural = use_neural
        self.stdout_bytes = []

        # Neural ALU for tracking operations
        self.alu = NeuralALU() if use_neural else None

        # Operation counters
        self.mul_count = 0
        self.div_count = 0
        self.add_count = 0
        self.cmp_count = 0
        self.total_ops = 0

        # VM state
        self.reset()

    def reset(self):
        self.memory = {}
        self.sp = 0x10000
        self.bp = 0x10000
        self.ax = 0
        self.pc = 0
        self.halted = False
        self.code = []
        self.stdout_bytes = []
        self.heap = 0x20000

    def load(self, bytecode, data=None):
        self.code = []
        for instr in bytecode:
            op = instr & 0xFF
            imm = instr >> 8
            if imm >= (1 << 55):
                imm -= (1 << 56)
            self.code.append((op, imm))

        if data:
            for i, b in enumerate(data):
                self.memory[0x10000 + i] = b

    def _to_signed32(self, x):
        """Convert unsigned 32-bit to signed."""
        x = x & 0xFFFFFFFF
        if x >= 0x80000000:
            return x - 0x100000000
        return x

    def _mul(self, a, b):
        """Multiply using neural SwiGLU."""
        self.mul_count += 1
        self.total_ops += 1

        # Convert to signed 32-bit for proper handling of negative numbers
        a_s = self._to_signed32(a)
        b_s = self._to_signed32(b)

        if self.use_neural and self.alu:
            import torch
            a_t = torch.tensor([float(a_s)])
            b_t = torch.tensor([float(b_s)])
            result = self.alu.mul(a_t, b_t)
            return int(round(result.item())) & 0xFFFFFFFF
        else:
            return (a_s * b_s) & 0xFFFFFFFF

    def _div(self, a, b):
        """Divide (signed)."""
        self.div_count += 1
        self.total_ops += 1
        if b == 0:
            return 0
        # Convert to signed for proper division
        a_s = self._to_signed32(a)
        b_s = self._to_signed32(b)
        # Python floor division, but C uses truncation toward zero
        if (a_s < 0) != (b_s < 0) and a_s % b_s != 0:
            result = a_s // b_s + 1
        else:
            result = a_s // b_s
        return result & 0xFFFFFFFF

    def _mod(self, a, b):
        """Modulo (signed)."""
        self.total_ops += 1
        if b == 0:
            return 0
        a_s = self._to_signed32(a)
        b_s = self._to_signed32(b)
        result = a_s - (a_s // b_s) * b_s  # C-style modulo
        return result & 0xFFFFFFFF

    def run(self, max_steps=10000000, verbose_interval=100000):
        steps = 0
        last_report = 0

        while steps < max_steps:
            if self.halted:
                break

            instr_idx = self.pc // 8
            if instr_idx >= len(self.code):
                break

            op, imm = self.code[instr_idx]
            self.pc += 8

            # Syscalls
            if op == self.SYS_PUTCHAR:
                c = self.memory.get(self.sp, 0)
                self.stdout_bytes.append(c & 0xFF)
                self.ax = c
            elif op == self.SYS_MALC:
                size = self.memory.get(self.sp, 0)
                ptr = self.heap
                self.heap += size
                self.ax = ptr
            elif op == self.SYS_EXIT:
                self.halted = True
                break

            # Standard ops
            elif op == 0:    # LEA
                self.ax = self.bp + imm
            elif op == 1:    # IMM
                self.ax = imm
            elif op == 2:    # JMP
                self.pc = imm
            elif op == 3:    # JSR
                self.sp -= 8
                self.memory[self.sp] = self.pc
                self.pc = imm
            elif op == 4:    # BZ
                if self.ax == 0:
                    self.pc = imm
            elif op == 5:    # BNZ
                if self.ax != 0:
                    self.pc = imm
            elif op == 6:    # ENT
                self.sp -= 8
                self.memory[self.sp] = self.bp
                self.bp = self.sp
                self.sp -= imm
            elif op == 7:    # ADJ
                self.sp += imm
            elif op == 8:    # LEV
                self.sp = self.bp
                self.bp = self.memory.get(self.sp, 0)
                self.sp += 8
                self.pc = self.memory.get(self.sp, 0)
                self.sp += 8
            elif op == 9:    # LI
                self.ax = self.memory.get(self.ax, 0)
            elif op == 10:   # LC
                self.ax = self.memory.get(self.ax, 0) & 0xFF
            elif op == 11:   # SI
                addr = self.memory.get(self.sp, 0)
                self.sp += 8
                self.memory[addr] = self.ax
            elif op == 12:   # SC
                addr = self.memory.get(self.sp, 0)
                self.sp += 8
                self.memory[addr] = self.ax & 0xFF
            elif op == 13:   # PSH
                self.sp -= 8
                self.memory[self.sp] = self.ax
            elif op == 14:   # OR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a | self.ax
            elif op == 15:   # XOR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a ^ self.ax
            elif op == 16:   # AND
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a & self.ax
            elif op == 17:   # EQ
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a == self.ax else 0
                self.cmp_count += 1
            elif op == 18:   # NE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a != self.ax else 0
                self.cmp_count += 1
            elif op == 19:   # LT (signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                a_s = self._to_signed32(a)
                ax_s = self._to_signed32(self.ax)
                self.ax = 1 if a_s < ax_s else 0
                self.cmp_count += 1
            elif op == 20:   # GT (signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                a_s = self._to_signed32(a)
                ax_s = self._to_signed32(self.ax)
                self.ax = 1 if a_s > ax_s else 0
                self.cmp_count += 1
            elif op == 21:   # LE (signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                a_s = self._to_signed32(a)
                ax_s = self._to_signed32(self.ax)
                self.ax = 1 if a_s <= ax_s else 0
                self.cmp_count += 1
            elif op == 22:   # GE (signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                a_s = self._to_signed32(a)
                ax_s = self._to_signed32(self.ax)
                self.ax = 1 if a_s >= ax_s else 0
                self.cmp_count += 1
            elif op == 23:   # SHL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a << self.ax) & 0xFFFFFFFF
            elif op == 24:   # SHR (unsigned)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a & 0xFFFFFFFF) >> (self.ax & 31)
            elif op == 25:   # ADD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a + self.ax) & 0xFFFFFFFF
                self.add_count += 1
            elif op == 26:   # SUB
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a - self.ax) & 0xFFFFFFFF
                self.add_count += 1
            elif op == 27:   # MUL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._mul(a, self.ax)
            elif op == 28:   # DIV
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._div(a, self.ax)
            elif op == 29:   # MOD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._mod(a, self.ax)
            elif op == 38:   # EXIT
                self.halted = True
                break

            steps += 1

            # Progress report
            if verbose_interval and steps - last_report >= verbose_interval:
                print(f"\rSteps: {steps:,} | Muls: {self.mul_count:,} | "
                      f"Output bytes: {len(self.stdout_bytes):,}", end="", flush=True)
                last_report = steps

        return self.ax


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_mandelbrot_transformer.py WIDTH HEIGHT OUTPUT.png")
        print("Example: python run_mandelbrot_transformer.py 64 64 test.png")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    output = sys.argv[3]

    # Use neural operations?
    use_neural = "--fast" not in sys.argv

    print("=" * 70)
    print("  MANDELBROT via C4 TRANSFORMER VM")
    print("=" * 70)
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output}")
    print(f"Neural ops: {'YES (SwiGLU)' if use_neural else 'NO (fast mode)'}")
    print()

    # Read the Mandelbrot C source (putchar version for VM compatibility)
    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file, 'r') as f:
        source = f.read()

    # Modify source to use command line dimensions
    # Replace default width/height (putchar version uses 32 default)
    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling C to bytecode...")
    start = time.time()
    bytecode, data = compile_c(source)
    compile_time = time.time() - start
    print(f"Compiled in {compile_time:.2f}s ({len(bytecode)} instructions)")
    print()

    print("Executing through transformer VM...")
    vm = TransformerProgramVM(use_neural=use_neural)
    vm.load(bytecode, data)

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
    print("OPERATION COUNTS:")
    print(f"  Multiplications (SwiGLU): {vm.mul_count:,}")
    print(f"  Divisions:                {vm.div_count:,}")
    print(f"  Additions/Subtractions:   {vm.add_count:,}")
    print(f"  Comparisons:              {vm.cmp_count:,}")
    print(f"  Total ops:                {vm.total_ops:,}")
    print()

    # Estimate FLOPs
    # SwiGLU multiply: ~10 FLOPs per operation (silu = sigmoid + mul, then mul and add)
    # Division: ~50 FLOPs (repeated subtraction)
    swiglu_flops = vm.mul_count * 10
    div_flops = vm.div_count * 50
    other_flops = vm.add_count + vm.cmp_count
    total_flops = swiglu_flops + div_flops + other_flops

    print("ESTIMATED FLOPs:")
    print(f"  SwiGLU multiplications: {swiglu_flops:,} FLOPs")
    print(f"  Divisions:              {div_flops:,} FLOPs")
    print(f"  Other operations:       {other_flops:,} FLOPs")
    print(f"  TOTAL:                  {total_flops:,} FLOPs")
    print(f"  Throughput:             {total_flops / exec_time / 1e6:.2f} MFLOPs/s")
    print()

    # Save PNG
    if vm.stdout_bytes:
        with open(output, 'wb') as f:
            f.write(bytes(vm.stdout_bytes))
        print(f"Saved {len(vm.stdout_bytes):,} bytes to {output}")

        # Verify PNG
        import subprocess
        result = subprocess.run(['file', output], capture_output=True, text=True)
        print(f"File type: {result.stdout.strip()}")
    else:
        print("ERROR: No output bytes generated!")

    print("=" * 70)


if __name__ == "__main__":
    main()
