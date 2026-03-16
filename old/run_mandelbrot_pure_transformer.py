#!/usr/bin/env python3
"""
Run Mandelbrot PNG generator through the PURE TRANSFORMER VM.

This executes the mandelbrot_color_png_c4.c program using:
- ALL arithmetic through NeuralALU (SwiGLU multiplication, neural division, etc.)
- ALL comparisons through neural compare
- ALL bitwise ops through neural bitwise
- Only control flow and memory remain as Python (necessary for I/O)

Usage:
    python run_mandelbrot_pure_transformer.py 32 32 output.png
    python run_mandelbrot_pure_transformer.py 64 64 output.png
"""

import sys
import os
import time
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from src.transformer_vm import NeuralALU


class PureTransformerProgramVM:
    """
    VM that runs programs using the NeuralALU for ALL arithmetic.
    Only memory/IO and control flow remain as Python.
    """

    # C4 syscall opcodes
    SYS_MALC = 34
    SYS_FREE = 35
    SYS_MSET = 36
    SYS_EXIT = 38
    SYS_PUTCHAR = 65

    def __init__(self):
        self.heap = 0x20000  # Heap start address
        self.stdout_bytes = []

        # Neural ALU for ALL operations
        self.alu = NeuralALU()

        # Helper to decode neural tensor to int
        def decode_int(x):
            """Decode 4 one-hot bytes to int."""
            val = 0
            for i in range(4):
                val += int(torch.argmax(x[i]).item()) << (i * 8)
            return val
        self._decode_neural = decode_int

        # Operation counters
        self.neural_ops = {
            'add': 0,
            'sub': 0,
            'mul': 0,
            'div': 0,
            'mod': 0,
            'and': 0,
            'or': 0,
            'xor': 0,
            'shl': 0,
            'shr': 0,
            'cmp': 0,
        }
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

    def _neural_add(self, a, b):
        """Add using neural ALU."""
        self.neural_ops['add'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        result = self.alu.add(a_enc, b_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_sub(self, a, b):
        """Subtract using neural ALU."""
        self.neural_ops['sub'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        result = self.alu.subtract(a_enc, b_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_mul(self, a, b):
        """Multiply using SwiGLU."""
        self.neural_ops['mul'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        result = self.alu.multiply(a_enc, b_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_div(self, a, b):
        """Divide using neural binary long division."""
        self.neural_ops['div'] += 1
        self.total_ops += 1
        if b == 0:
            return 0
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        result = self.alu.divide(a_enc, b_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_mod(self, a, b):
        """Modulo using neural division."""
        self.neural_ops['mod'] += 1
        self.total_ops += 1
        if b == 0:
            return 0
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        quot = self.alu.divide(a_enc, b_enc)
        prod = self.alu.multiply(quot, b_enc)
        result = self.alu.subtract(a_enc, prod)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_and(self, a, b):
        """Bitwise AND using neural ALU."""
        self.neural_ops['and'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(a)
        b_enc = self.alu._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'and')
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_or(self, a, b):
        """Bitwise OR using neural ALU."""
        self.neural_ops['or'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(a)
        b_enc = self.alu._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'or')
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_xor(self, a, b):
        """Bitwise XOR using neural ALU."""
        self.neural_ops['xor'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(a)
        b_enc = self.alu._encode_int(b)
        result = self.alu.bitwise_op(a_enc, b_enc, 'xor')
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_shl(self, a, shift):
        """Shift left using neural ALU."""
        self.neural_ops['shl'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(a)
        shift_enc = self.alu._encode_int(shift & 31)
        result = self.alu.neural_shift_left(a_enc, shift_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_shr(self, a, shift):
        """Shift right (unsigned) using neural ALU."""
        self.neural_ops['shr'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(a & 0xFFFFFFFF)
        shift_enc = self.alu._encode_int(shift & 31)
        result = self.alu.neural_shift_right(a_enc, shift_enc)
        return self._decode_neural(result) & 0xFFFFFFFF

    def _neural_compare(self, a, b):
        """Compare using neural ALU. Returns (lt, eq, gt) as bools."""
        self.neural_ops['cmp'] += 1
        self.total_ops += 1
        a_enc = self.alu._encode_int(self._to_signed32(a))
        b_enc = self.alu._encode_int(self._to_signed32(b))
        lt, eq, gt = self.alu.compare(a_enc, b_enc)
        return (lt[0].item() > 0.5, eq[0].item() > 0.5, gt[0].item() > 0.5)

    def _decode_int(self, x):
        """Helper to decode neural tensor to int."""
        return self.alu._decode_int(x)

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

            # Standard ops - ALL using neural ALU
            elif op == 0:    # LEA
                self.ax = self._neural_add(self.bp, imm)
            elif op == 1:    # IMM
                self.ax = imm & 0xFFFFFFFF
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
                self.sp = self._neural_sub(self.sp, imm)
            elif op == 7:    # ADJ
                self.sp = self._neural_add(self.sp, imm)
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
            elif op == 14:   # OR (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_or(a, self.ax)
            elif op == 15:   # XOR (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_xor(a, self.ax)
            elif op == 16:   # AND (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_and(a, self.ax)
            elif op == 17:   # EQ (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 1 if eq else 0
            elif op == 18:   # NE (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 0 if eq else 1
            elif op == 19:   # LT (neural signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 1 if lt else 0
            elif op == 20:   # GT (neural signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 1 if gt else 0
            elif op == 21:   # LE (neural signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 1 if (lt or eq) else 0
            elif op == 22:   # GE (neural signed)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                lt, eq, gt = self._neural_compare(a, self.ax)
                self.ax = 1 if (gt or eq) else 0
            elif op == 23:   # SHL (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_shl(a, self.ax)
            elif op == 24:   # SHR (neural unsigned)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_shr(a, self.ax)
            elif op == 25:   # ADD (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_add(a, self.ax)
            elif op == 26:   # SUB (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_sub(a, self.ax)
            elif op == 27:   # MUL (neural SwiGLU)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_mul(a, self.ax)
            elif op == 28:   # DIV (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_div(a, self.ax)
            elif op == 29:   # MOD (neural)
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = self._neural_mod(a, self.ax)
            elif op == 38:   # EXIT
                self.halted = True
                break

            steps += 1

            # Progress report
            if verbose_interval and steps - last_report >= verbose_interval:
                print(f"\rSteps: {steps:,} | Neural ops: {self.total_ops:,} | "
                      f"Output bytes: {len(self.stdout_bytes):,}", end="", flush=True)
                last_report = steps

        return self.ax


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_mandelbrot_pure_transformer.py WIDTH HEIGHT OUTPUT.png")
        print("Example: python run_mandelbrot_pure_transformer.py 32 32 test.png")
        sys.exit(1)

    width = int(sys.argv[1])
    height = int(sys.argv[2])
    output = sys.argv[3]

    print("=" * 70)
    print("  MANDELBROT via PURE TRANSFORMER VM")
    print("  ALL arithmetic through NeuralALU (SwiGLU, neural division, etc.)")
    print("=" * 70)
    print(f"Resolution: {width}x{height}")
    print(f"Output: {output}")
    print()

    # Read the Mandelbrot C source (putchar version for VM compatibility)
    c_file = os.path.join(os.path.dirname(__file__), "mandelbrot_putchar_c4.c")
    with open(c_file, 'r') as f:
        source = f.read()

    # Modify source to use command line dimensions
    source = source.replace("width = 32;", f"width = {width};")
    source = source.replace("height = 32;", f"height = {height};")

    print("Compiling C to bytecode...")
    start = time.time()
    bytecode, data = compile_c(source)
    compile_time = time.time() - start
    print(f"Compiled in {compile_time:.2f}s ({len(bytecode)} instructions)")
    print()

    print("Executing through PURE transformer VM...")
    print("(All arithmetic uses NeuralALU - SwiGLU mul, neural division, etc.)")
    print()

    vm = PureTransformerProgramVM()
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
    print("NEURAL OPERATION COUNTS:")
    for op, count in sorted(vm.neural_ops.items()):
        if count > 0:
            print(f"  {op:12s}: {count:>12,}")
    print(f"  {'TOTAL':12s}: {vm.total_ops:>12,}")
    print()

    # Estimate FLOPs
    # Each neural op involves FFN tables and softmax
    # Add/Sub: ~20 FLOPs (nibble add with carry, 8 nibbles)
    # Mul (SwiGLU): ~10 FLOPs
    # Div (neural binary): ~1000 FLOPs (32 iterations of compare/subtract)
    # Mod: ~2000 FLOPs (div + mul + sub)
    # Compare: ~50 FLOPs (subtract + checks)
    # Bitwise: ~40 FLOPs (8 nibble ops)
    # Shift: ~200 FLOPs (5 conditional shifts)

    flops_per_op = {
        'add': 20, 'sub': 20, 'mul': 10, 'div': 1000, 'mod': 2000,
        'cmp': 50, 'and': 40, 'or': 40, 'xor': 40, 'shl': 200, 'shr': 200
    }

    total_flops = sum(vm.neural_ops[op] * flops_per_op.get(op, 10)
                      for op in vm.neural_ops)

    print("ESTIMATED FLOPs:")
    for op, count in sorted(vm.neural_ops.items()):
        if count > 0:
            op_flops = count * flops_per_op.get(op, 10)
            print(f"  {op:12s}: {op_flops:>15,} FLOPs")
    print(f"  {'TOTAL':12s}: {total_flops:>15,} FLOPs")
    print(f"  Throughput:   {total_flops / exec_time / 1e6:>15.2f} MFLOPs/s")
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
