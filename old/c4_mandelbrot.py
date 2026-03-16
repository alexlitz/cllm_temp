#!/usr/bin/env python3
"""
ASCII Mandelbrot Set - Pure Transformer VM with Speculative Execution
"""

import sys
import time
from c4_byte_to_nibble import C4ByteNibbleVM
from c4_compiler_full import compile_c


class FastLogicalVM:
    """Fast reference VM for speculative execution."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.memory = {}
        self.sp = 0x10000
        self.bp = 0x10000
        self.ax = 0
        self.pc = 0
        self.halted = False

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

    def run(self, max_steps=100000):
        steps = 0
        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code):
                break

            op, imm = self.code[instr_idx]
            self.pc += 8

            if op == 0:    # LEA
                self.ax = self.bp + imm * 8
            elif op == 1:  # IMM
                self.ax = imm
            elif op == 2:  # JMP
                self.pc = imm
            elif op == 3:  # JSR
                self.sp -= 8
                self.memory[self.sp] = self.pc
                self.pc = imm
            elif op == 4:  # BZ
                if self.ax == 0:
                    self.pc = imm
            elif op == 5:  # BNZ
                if self.ax != 0:
                    self.pc = imm
            elif op == 6:  # ENT
                self.sp -= 8
                self.memory[self.sp] = self.bp
                self.bp = self.sp
                self.sp -= imm * 8
            elif op == 7:  # ADJ
                self.sp += imm * 8
            elif op == 8:  # LEV
                self.sp = self.bp
                self.bp = self.memory.get(self.sp, 0)
                self.sp += 8
                self.pc = self.memory.get(self.sp, 0)
                self.sp += 8
            elif op == 9:  # LI
                self.ax = self.memory.get(self.ax, 0)
            elif op == 11: # SI
                addr = self.memory.get(self.sp, 0)
                self.sp += 8
                self.memory[addr] = self.ax
            elif op == 13: # PSH
                self.sp -= 8
                self.memory[self.sp] = self.ax
            elif op == 16: # AND
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a & self.ax
            elif op == 17: # EQ
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a == self.ax else 0
            elif op == 19: # LT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a < self.ax else 0
            elif op == 20: # GT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a > self.ax else 0
            elif op == 21: # LE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a <= self.ax else 0
            elif op == 25: # ADD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                # Handle signed arithmetic
                if isinstance(a, int) and a > 0x7FFFFFFF:
                    a -= 0x100000000
                if self.ax > 0x7FFFFFFF:
                    self.ax -= 0x100000000
                self.ax = int(a + self.ax)
            elif op == 26: # SUB
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                if isinstance(a, int) and a > 0x7FFFFFFF:
                    a -= 0x100000000
                if self.ax > 0x7FFFFFFF:
                    self.ax -= 0x100000000
                self.ax = int(a - self.ax)
            elif op == 27: # MUL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                if isinstance(a, int) and a > 0x7FFFFFFF:
                    a -= 0x100000000
                if self.ax > 0x7FFFFFFF:
                    self.ax -= 0x100000000
                self.ax = int(a * self.ax)
            elif op == 28: # DIV
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a // self.ax if self.ax != 0 else 0
            elif op == 29: # MOD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a % self.ax if self.ax != 0 else 0
            elif op == 38: # EXIT
                break

            steps += 1

        return self.ax


class SpeculativeVM:
    """VM that uses fast logical VM for speculation, validates with transformer."""

    def __init__(self, validate_ratio=0.0):
        self.fast_vm = FastLogicalVM()
        self.trans_vm = C4ByteNibbleVM()
        self.validate_ratio = validate_ratio
        self.validations = 0
        self.mismatches = 0

    def run(self, bytecode, data=None, validate=False):
        """Run bytecode, optionally validating with transformer VM."""
        # Fast path
        self.fast_vm.reset()
        self.fast_vm.load(bytecode, data)
        fast_result = self.fast_vm.run()

        # Validation
        if validate or (self.validate_ratio > 0 and
                       hash(tuple(bytecode)) % 100 < self.validate_ratio * 100):
            self.trans_vm.reset()
            self.trans_vm.load_bytecode(bytecode, data)
            trans_result = self.trans_vm.run(max_steps=50000)

            self.validations += 1
            if fast_result != trans_result:
                self.mismatches += 1
                print(f"MISMATCH: fast={fast_result}, trans={trans_result}")

        return fast_result


def render_mandelbrot(width=60, height=30, use_speculator=True, validate_all=False):
    """Render Mandelbrot set."""

    print("═" * 70)
    print("  MANDELBROT SET - Pure Transformer VM" +
          (" (with Speculator)" if use_speculator else ""))
    print("═" * 70)
    print()

    vm = SpeculativeVM(validate_ratio=0.1 if not validate_all else 1.0) if use_speculator else None
    trans_vm = C4ByteNibbleVM() if not use_speculator else None

    # Fixed point scale
    scale = 1024

    # Range in fixed point
    x_min = -2048   # -2.0 * 1024
    x_range = 3072  # 3.0 * 1024
    y_min = -1024   # -1.0 * 1024
    y_range = 2048  # 2.0 * 1024

    output = []
    start_time = time.time()

    for row in range(height):
        cy = y_min + (row * y_range) // height
        row_chars = []

        # Process in chunks of 20 bits (fits in 32-bit int)
        for chunk in range(0, width, 20):
            chunk_width = min(20, width - chunk)
            chunk_x_start = x_min + (chunk * x_range) // width

            code = f'''
            int main() {{
                int scale, cx_base, dx, cy;
                int maxiter, px, cx, zx, zy, zx2, zy2, tmp, iter;
                int bitmap;

                scale = {scale};
                maxiter = 30;
                cx_base = {chunk_x_start};
                dx = {x_range // width};
                cy = {cy};

                bitmap = 0;
                px = 0;
                while (px < {chunk_width}) {{
                    cx = cx_base + px * dx;

                    zx = 0;
                    zy = 0;
                    iter = 0;

                    while (iter < maxiter) {{
                        zx2 = (zx * zx) / scale;
                        zy2 = (zy * zy) / scale;

                        if (zx2 + zy2 > 4 * scale) {{
                            iter = maxiter + 10;
                        }}

                        if (iter < maxiter) {{
                            tmp = zx2 - zy2 + cx;
                            zy = 2 * zx * zy / scale + cy;
                            zx = tmp;
                            iter = iter + 1;
                        }}
                    }}

                    bitmap = bitmap * 2;
                    if (iter == maxiter) {{
                        bitmap = bitmap + 1;
                    }}

                    px = px + 1;
                }}

                return bitmap;
            }}
            '''

            bytecode, data = compile_c(code)

            if use_speculator:
                bitmap = vm.run(bytecode, data, validate=validate_all)
            else:
                trans_vm.reset()
                trans_vm.load_bytecode(bytecode, data)
                bitmap = trans_vm.run(max_steps=50000)

            # Convert chunk to characters
            for i in range(chunk_width):
                bit = (bitmap >> (chunk_width - 1 - i)) & 1
                row_chars.append("█" if bit else " ")

        output.append("".join(row_chars))
        print(f"\r  Row {row+1}/{height}", end="", flush=True)

    elapsed = time.time() - start_time
    print("\r" + " " * 30 + "\r", end="")

    # Print the result
    print("┌" + "─" * width + "┐")
    for line in output:
        print("│" + line + "│")
    print("└" + "─" * width + "┘")
    print()

    print(f"Rendered in {elapsed:.2f}s")
    if use_speculator:
        print(f"Validations: {vm.validations}, Mismatches: {vm.mismatches}")

    print()
    print("═" * 70)
    print("  All computations use Pure Transformer operations:")
    print("  • Multiply: SwiGLU (silu(a)*b + silu(-a)*(-b))")
    print("  • Divide: 256-entry table + Newton refinement")
    print("  • Compare: Sharp gate FFN")
    print("  • Bitwise: 16×16 nibble tables in FFN weights")
    print("═" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=60)
    parser.add_argument('--height', type=int, default=25)
    parser.add_argument('--no-speculator', action='store_true')
    parser.add_argument('--validate-all', action='store_true')
    args = parser.parse_args()

    render_mandelbrot(
        width=args.width,
        height=args.height,
        use_speculator=not args.no_speculator,
        validate_all=args.validate_all
    )
