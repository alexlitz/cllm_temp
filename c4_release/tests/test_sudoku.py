#!/usr/bin/env python3
"""
Test Sudoku Solver on C4 VM

Compiles and runs a backtracking sudoku solver using the C4 compiler
and an extended VM with malloc/free support. Tests against
Arto Inkala's "World's Hardest Sudoku" (2012).
"""

import sys
import os
import time
from collections import defaultdict
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compiler import compile_c
from src.io_support import IOExtendedVM


class SudokuVM(IOExtendedVM):
    """
    Extended VM with malloc, free, and printf support.

    Adds system calls:
    - MALC (op 34): malloc - simple bump allocator
    - FREE (op 35): free - no-op
    - PRTF (op 33): printf - format string with %d, %s, %c
    """

    def __init__(self):
        super().__init__()
        self.heap_ptr = 0x200000
        self.step_count = 0

    def reset(self):
        super().reset()
        self.heap_ptr = 0x200000
        self.step_count = 0

    def _read_string(self, addr):
        """Read null-terminated string from memory."""
        chars = []
        while True:
            c = self.memory.get(addr, 0)
            if c == 0:
                break
            chars.append(chr(c & 0xFF))
            addr += 1
        return ''.join(chars)

    def _handle_printf(self, argc):
        """Handle printf syscall with format string substitution."""
        fmt_addr = self.memory.get(self.sp + (argc - 1) * 8, 0)
        fmt = self._read_string(fmt_addr)

        result = []
        arg_idx = 1
        i = 0
        while i < len(fmt):
            if fmt[i] == '%' and i + 1 < len(fmt):
                spec = fmt[i + 1]
                if spec == '%':
                    result.append('%')
                    i += 2
                    continue

                val = self.memory.get(self.sp + (argc - 1 - arg_idx) * 8, 0) if arg_idx < argc else 0

                if spec == 'd':
                    result.append(str(val))
                elif spec == 'c':
                    result.append(chr(val & 0xFF))
                elif spec == 's':
                    result.append(self._read_string(val))
                else:
                    result.append('%' + spec)

                arg_idx += 1
                i += 2
            else:
                result.append(fmt[i])
                i += 1

        self.io.stdout.write(''.join(result))

    # Superinstruction opcodes (> 100 to avoid collision)
    # Fusing the top instruction pairs eliminates ~41% of dispatch overhead
    SUPER_LEA_LI = 101       # LEA(x) + LI  → ax = mem[bp + x]
    SUPER_LI_PSH = 102       # LI + PSH     → sp-=8; mem[sp] = mem[ax]; ax = mem[ax]
    SUPER_PSH_IMM = 103      # PSH + IMM(x) → sp-=8; mem[sp] = ax; ax = x
    SUPER_LEA_LI_PSH = 104   # LEA+LI+PSH   → sp-=8; mem[sp] = mem[bp+x]
    SUPER_PSH_LEA = 105      # PSH + LEA(x) → sp-=8; mem[sp] = ax; ax = bp+x
    SUPER_IMM_EQ = 106       # IMM(x) + EQ  → ax = 1 if mem[sp] == x else 0; sp+=8
    SUPER_IMM_AND = 107      # IMM(x) + AND → ax = mem[sp] & x; sp+=8
    SUPER_IMM_MUL = 108      # IMM(x) + MUL → ax = mem[sp] * x; sp+=8
    SUPER_EQ_BZ = 109        # EQ + BZ(t)   → if mem[sp]==ax: pc=t; sp+=8; ax=result
    SUPER_LEA_PSH = 110      # LEA(x) + PSH → sp-=8; mem[sp] = bp+x

    @staticmethod
    def _fuse_bytecode(code):
        """Pre-process bytecode to create superinstructions.

        Scans for common instruction pairs/triples and fuses them into
        single dispatch entries. Returns (ops, imms, imms2) arrays.
        imms2 holds the second immediate for fused pairs that need it.
        """
        n = len(code)
        ops = []
        imms = []
        imms2 = []  # Secondary immediate for some superinstructions
        # Map from original PC index to fused index
        pc_map = {}
        i = 0

        while i < n:
            pc_map[i] = len(ops)
            op0, imm0 = code[i]

            # Try 3-instruction fusion: LEA + LI + PSH
            if i + 2 < n and op0 == 0:  # LEA
                op1, imm1 = code[i + 1]
                op2, imm2 = code[i + 2]
                if op1 == 9 and op2 == 13:  # LI + PSH
                    ops.append(104)  # SUPER_LEA_LI_PSH
                    imms.append(imm0)
                    imms2.append(0)
                    i += 3
                    continue

            # Try 2-instruction fusions
            if i + 1 < n:
                op1, imm1 = code[i + 1]

                if op0 == 0 and op1 == 9:  # LEA + LI
                    ops.append(101)
                    imms.append(imm0)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 9 and op1 == 13:  # LI + PSH
                    ops.append(102)
                    imms.append(0)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 13 and op1 == 1:  # PSH + IMM
                    ops.append(103)
                    imms.append(imm1)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 13 and op1 == 0:  # PSH + LEA
                    ops.append(105)
                    imms.append(imm1)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 1 and op1 == 17:  # IMM + EQ
                    ops.append(106)
                    imms.append(imm0)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 1 and op1 == 16:  # IMM + AND
                    ops.append(107)
                    imms.append(imm0)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 1 and op1 == 27:  # IMM + MUL
                    ops.append(108)
                    imms.append(imm0)
                    imms2.append(0)
                    i += 2
                    continue
                if op0 == 0 and op1 == 13:  # LEA + PSH
                    ops.append(110)
                    imms.append(imm0)
                    imms2.append(0)
                    i += 2
                    continue

            # No fusion - emit as-is
            ops.append(op0)
            imms.append(imm0)
            imms2.append(0)
            i += 1

        # Remap jump targets from original PC to fused PC
        # Original PC = original_index * 8; fused PC = fused_index * 8
        for j in range(len(ops)):
            op = ops[j]
            # Opcodes with jump targets in imm
            if op in (2, 3, 4, 5):  # JMP, JSR, BZ, BNZ
                orig_idx = imms[j] // 8
                if orig_idx in pc_map:
                    imms[j] = pc_map[orig_idx] * 8

        return ops, imms, pc_map

    def run(self, max_steps=100000000):
        """Execute bytecode - heavily optimized with superinstructions.

        Optimizations:
        1. All state as local variables (no self.x per step)
        2. Flat bytecode arrays with superinstruction fusion
        3. defaultdict(int) for zero-default memory
        4. Top instruction pairs fused (41% of all instructions)
        5. Opcodes ordered by frequency for faster dispatch
        """
        from collections import defaultdict
        mem = defaultdict(int, self.memory)
        code = self.code
        code_len = len(code)

        # Fuse bytecode into superinstructions
        ops, imms, pc_map = self._fuse_bytecode(code)
        fused_len = len(ops)

        sp = self.sp
        bp = self.bp
        ax = self.ax
        pc = 0  # Start at fused PC 0
        heap_ptr = self.heap_ptr
        stdout_write = self.io.stdout.write
        steps = 0

        while steps < max_steps:
            idx = pc >> 3
            if idx >= fused_len:
                break

            op = ops[idx]
            pc += 8

            # --- Superinstructions (most frequent first) ---
            if op == 104:  # SUPER_LEA_LI_PSH (10.6% of original pairs)
                v = mem[bp + imms[idx]]
                sp -= 8
                mem[sp] = v
                ax = v
                steps += 2
            elif op == 103: # SUPER_PSH_IMM (16.6%)
                sp -= 8
                mem[sp] = ax
                ax = imms[idx]
                steps += 1
            elif op == 102: # SUPER_LI_PSH (13.7%)
                ax = mem[ax]
                sp -= 8
                mem[sp] = ax
                steps += 1
            elif op == 101: # SUPER_LEA_LI (10.6%)
                ax = mem[bp + imms[idx]]
                steps += 1
            elif op == 105: # SUPER_PSH_LEA (6.0%)
                sp -= 8
                mem[sp] = ax
                ax = bp + imms[idx]
                steps += 1
            elif op == 106: # SUPER_IMM_EQ (4.3%)
                ax = 1 if mem[sp] == imms[idx] else 0
                sp += 8
                steps += 1
            elif op == 107: # SUPER_IMM_AND (2.8%)
                ax = mem[sp] & imms[idx]
                sp += 8
                steps += 1
            elif op == 108: # SUPER_IMM_MUL (2.6%)
                ax = mem[sp] * imms[idx]
                sp += 8
                steps += 1
            elif op == 110: # SUPER_LEA_PSH (4.8%)
                v = bp + imms[idx]
                sp -= 8
                mem[sp] = v
                ax = v
                steps += 1

            # --- Regular opcodes ---
            elif op == 1:  # IMM
                ax = imms[idx]
            elif op == 9:  # LI
                ax = mem[ax]
            elif op == 13: # PSH
                sp -= 8
                mem[sp] = ax
            elif op == 25: # ADD
                ax = mem[sp] + ax
                sp += 8
            elif op == 0:  # LEA
                ax = bp + imms[idx]
            elif op == 4:  # BZ
                if ax == 0:
                    pc = imms[idx]
            elif op == 5:  # BNZ
                if ax != 0:
                    pc = imms[idx]
            elif op == 17: # EQ
                ax = 1 if mem[sp] == ax else 0
                sp += 8
            elif op == 11: # SI
                mem[mem[sp]] = ax
                sp += 8
            elif op == 16: # AND
                ax = mem[sp] & ax
                sp += 8
            elif op == 14: # OR
                ax = mem[sp] | ax
                sp += 8
            elif op == 7:  # ADJ
                sp += imms[idx]
            elif op == 26: # SUB
                ax = mem[sp] - ax
                sp += 8
            elif op == 19: # LT
                ax = 1 if mem[sp] < ax else 0
                sp += 8
            elif op == 23: # SHL
                ax = mem[sp] << ax
                sp += 8
            elif op == 3:  # JSR
                sp -= 8
                mem[sp] = pc
                pc = imms[idx]
            elif op == 8:  # LEV
                sp = bp
                bp = mem[sp]
                sp += 8
                pc = mem[sp]
                sp += 8
            elif op == 6:  # ENT
                sp -= 8
                mem[sp] = bp
                bp = sp
                sp -= imms[idx]
            elif op == 2:  # JMP
                pc = imms[idx]
            elif op == 27: # MUL
                ax = mem[sp] * ax
                sp += 8
            elif op == 28: # DIV
                ax = mem[sp] // ax if ax != 0 else 0
                sp += 8
            elif op == 29: # MOD
                ax = mem[sp] % ax if ax != 0 else 0
                sp += 8
            elif op == 10: # LC
                ax = mem[ax] & 0xFF
            elif op == 18: # NE
                ax = 1 if mem[sp] != ax else 0
                sp += 8
            elif op == 20: # GT
                ax = 1 if mem[sp] > ax else 0
                sp += 8
            elif op == 21: # LE
                ax = 1 if mem[sp] <= ax else 0
                sp += 8
            elif op == 22: # GE
                ax = 1 if mem[sp] >= ax else 0
                sp += 8
            elif op == 15: # XOR
                ax = mem[sp] ^ ax
                sp += 8
            elif op == 24: # SHR
                ax = mem[sp] >> ax
                sp += 8
            elif op == 12: # SC
                mem[mem[sp]] = ax & 0xFF
                sp += 8
            elif op == 65: # PUTCHAR
                stdout_write(chr(mem[sp] & 0xFF))
                ax = mem[sp]
            elif op == 34: # MALC
                size = mem[sp]
                ax = heap_ptr
                heap_ptr += size
                if heap_ptr & 7:
                    heap_ptr += 8 - (heap_ptr & 7)
            elif op == 35: # FREE
                pass
            elif op == 38: # EXIT
                break
            elif op == 33: # PRTF
                self.sp = sp
                self.memory = dict(mem)
                next_idx = pc >> 3
                argc = 1
                if next_idx < fused_len:
                    if ops[next_idx] == 7:
                        argc = imms[next_idx] >> 3
                self._handle_printf(argc)
                ax = 0

            steps += 1

        # Write back state
        self.sp = sp
        self.bp = bp
        self.ax = ax
        self.pc = pc
        self.heap_ptr = heap_ptr
        self.memory = dict(mem)
        self.step_count = steps
        return ax


def main():
    # Read the C source
    src_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'demos', 'sudoku.c'
    )
    with open(src_path) as f:
        source = f.read()

    print("=" * 60)
    print("  Sudoku Solver - C4 VM (Optimized)")
    print("=" * 60)
    print()
    print("Puzzle: Arto Inkala's 'World's Hardest Sudoku' (2012)")
    print()
    print("  8 . . | . . . | . . .")
    print("  . . 3 | 6 . . | . . .")
    print("  . 7 . | . 9 . | 2 . .")
    print("  ------+-------+------")
    print("  . 5 . | . . 7 | . . .")
    print("  . . . | . 4 5 | 7 . .")
    print("  . . . | 1 . . | . 3 .")
    print("  ------+-------+------")
    print("  . . 1 | . . . | . 6 8")
    print("  . . 8 | 5 . . | . 1 .")
    print("  . 9 . | . . . | 4 . .")
    print()

    # Compile
    print("Compiling...", end=" ", flush=True)
    start = time.time()
    bytecode, data = compile_c(source)
    compile_time = time.time() - start
    print(f"done ({compile_time*1000:.1f}ms, {len(bytecode)} instructions)")

    # Run
    print("Solving...", end=" ", flush=True)
    vm = SudokuVM()
    vm.load(bytecode, data)

    start = time.time()
    result = vm.run(max_steps=100000000)
    solve_time = time.time() - start

    output = vm.get_stdout()
    steps_per_sec = vm.step_count / solve_time if solve_time > 0 else 0
    print(f"done ({solve_time:.2f}s, {vm.step_count:,} VM steps, {steps_per_sec/1e6:.1f}M steps/sec)")
    print()

    if result == 1:
        print("SOLVED! Output:")
        print()
        for line in output.strip().split('\n'):
            print(f"  {line}")
        print()

        # Verify solution
        expected = [
            "8 1 2 7 5 3 6 4 9",
            "9 4 3 6 8 2 1 7 5",
            "6 7 5 4 9 1 2 8 3",
            "1 5 4 2 3 7 8 9 6",
            "3 6 9 8 4 5 7 2 1",
            "2 8 7 1 6 9 5 3 4",
            "5 2 1 9 7 4 3 6 8",
            "4 3 8 5 2 6 9 1 7",
            "7 9 6 3 1 8 4 5 2",
        ]

        actual_lines = output.strip().split('\n')
        all_match = True
        for i, (exp, act) in enumerate(zip(expected, actual_lines)):
            if exp != act:
                print(f"  Row {i} MISMATCH: expected '{exp}', got '{act}'")
                all_match = False

        if all_match and len(actual_lines) == 9:
            print("VERIFIED: Solution matches known answer!")
        else:
            print("VERIFICATION FAILED")
    else:
        print("FAILED: No solution found")
        print(f"  Output: {repr(output)}")
        print(f"  ax={result}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
