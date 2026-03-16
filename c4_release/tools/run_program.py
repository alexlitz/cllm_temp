#!/usr/bin/env python3
"""
Run C programs through the Transformer VM.

Reads stdin, executes C code, writes stdout, exits with program's return code.

Usage:
    # Evaluate expression
    python run_program.py -e '6 * 7'
    # Returns exit code 42

    # Run C code
    python run_program.py -c 'int main() { return 123; }'
    # Returns exit code 123

    # Run from file (recommended for complex programs)
    echo "HAL" | python run_program.py caesar.c
    # Output: IBM

    # Simple inline code (no != or ! due to shell escaping)
    python run_program.py -c 'int main() { putchar(72); putchar(105); return 0; }'
    # Output: Hi

Note: Shell escaping issues with '!' character - use files for code with '!='
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.speculator import FastLogicalVM
from src.compiler import compile_c


class ProgramVM(FastLogicalVM):
    """VM with stdin/stdout support and operation counting."""

    SYS_GETCHAR = 64
    SYS_PUTCHAR = 65

    def __init__(self, stdin_data: str = "", count_ops: bool = False):
        super().__init__()
        self.stdin_data = stdin_data
        self.stdin_pos = 0
        self.stdout_data = []
        self.count_ops = count_ops
        # Operation counters
        self.mul_count = 0
        self.div_count = 0
        self.add_count = 0
        self.cmp_count = 0
        self.bitwise_count = 0

    def _getchar(self) -> int:
        if self.stdin_pos >= len(self.stdin_data):
            return -1 & 0xFFFFFFFF  # EOF as unsigned
        c = ord(self.stdin_data[self.stdin_pos])
        self.stdin_pos += 1
        return c

    def _putchar(self, c: int):
        self.stdout_data.append(chr(c & 0xFF))

    def get_stdout(self) -> str:
        return ''.join(self.stdout_data)

    def run(self, max_steps: int = 1000000) -> int:
        steps = 0

        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code) or self.halted:
                break

            op, imm = self.code[instr_idx]
            self.pc += 8

            # I/O syscalls
            if op == self.SYS_GETCHAR:
                self.ax = self._getchar()
            elif op == self.SYS_PUTCHAR:
                c = self.memory.get(self.sp, 0)
                self._putchar(c)
                self.ax = c

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
            elif op == 18:   # NE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = 1 if a != self.ax else 0
            elif op == 19:   # LT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                # Signed comparison
                if a >= 0x80000000:
                    a -= 0x100000000
                ax = self.ax
                if ax >= 0x80000000:
                    ax -= 0x100000000
                self.ax = 1 if a < ax else 0
            elif op == 20:   # GT
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                if a >= 0x80000000:
                    a -= 0x100000000
                ax = self.ax
                if ax >= 0x80000000:
                    ax -= 0x100000000
                self.ax = 1 if a > ax else 0
            elif op == 21:   # LE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                if a >= 0x80000000:
                    a -= 0x100000000
                ax = self.ax
                if ax >= 0x80000000:
                    ax -= 0x100000000
                self.ax = 1 if a <= ax else 0
            elif op == 22:   # GE
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                if a >= 0x80000000:
                    a -= 0x100000000
                ax = self.ax
                if ax >= 0x80000000:
                    ax -= 0x100000000
                self.ax = 1 if a >= ax else 0
            elif op == 23:   # SHL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a << self.ax) & 0xFFFFFFFF
            elif op == 24:   # SHR
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a >> self.ax
            elif op == 25:   # ADD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a + self.ax) & 0xFFFFFFFF
                if self.count_ops:
                    self.add_count += 1
            elif op == 26:   # SUB
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a - self.ax) & 0xFFFFFFFF
                if self.count_ops:
                    self.add_count += 1
            elif op == 27:   # MUL
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = (a * self.ax) & 0xFFFFFFFF
                if self.count_ops:
                    self.mul_count += 1
            elif op == 28:   # DIV
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a // self.ax if self.ax != 0 else 0
                if self.count_ops:
                    self.div_count += 1
            elif op == 29:   # MOD
                a = self.memory.get(self.sp, 0)
                self.sp += 8
                self.ax = a % self.ax if self.ax != 0 else 0
                if self.count_ops:
                    self.div_count += 1  # MOD uses division
            elif op == 38:   # EXIT
                self.halted = True
                break

            steps += 1

        return self.ax


def run_program(source: str, stdin_data: str = "", count_ops: bool = False) -> tuple:
    """Run C program, return (exit_code, stdout, vm)."""
    bytecode, data = compile_c(source)
    vm = ProgramVM(stdin_data, count_ops=count_ops)
    vm.load(bytecode, data)
    exit_code = vm.run()
    return exit_code, vm.get_stdout(), vm


def main():
    parser = argparse.ArgumentParser(description='Run C programs through Transformer VM')
    parser.add_argument('file', nargs='?', help='C source file')
    parser.add_argument('-c', '--code', help='C code string')
    parser.add_argument('-e', '--expr', help='Expression to evaluate')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show exit code')
    parser.add_argument('--stats', action='store_true', help='Show operation counts')

    args = parser.parse_args()

    # Get source
    if args.expr:
        source = f'int main() {{ return {args.expr}; }}'
    elif args.code:
        source = args.code
    elif args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        parser.print_help()
        sys.exit(1)

    # Get stdin
    stdin_data = ""
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read()

    # Debug
    if os.environ.get('DEBUG'):
        print(f"Source: {repr(source)}", file=sys.stderr)
        print(f"Stdin: {repr(stdin_data)}", file=sys.stderr)

    # Run
    try:
        exit_code, stdout, vm = run_program(source, stdin_data, count_ops=args.stats)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if stdout:
        sys.stdout.write(stdout)

    if args.verbose:
        print(f"Exit code: {exit_code}", file=sys.stderr)

    if args.stats:
        print(file=sys.stderr)
        print("=== OPERATION COUNTS ===", file=sys.stderr)
        print(f"Multiplications: {vm.mul_count:,}", file=sys.stderr)
        print(f"Divisions:       {vm.div_count:,}", file=sys.stderr)
        print(f"Add/Sub:         {vm.add_count:,}", file=sys.stderr)
        # FLOP estimation
        # SwiGLU multiply: ~10 FLOPs
        # Division: ~50 FLOPs (Newton-Raphson)
        swiglu_flops = vm.mul_count * 10
        div_flops = vm.div_count * 50
        total_flops = swiglu_flops + div_flops + vm.add_count
        print(f"Estimated FLOPs: {total_flops:,}", file=sys.stderr)

    sys.exit(exit_code % 256)


if __name__ == '__main__':
    main()
