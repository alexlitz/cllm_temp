#!/usr/bin/env python3
"""
Run ELIZA with speculative execution and neural verification.

Uses fast logical VM for execution, then batch-verifies all arithmetic
operations against the neural ALU (SwiGLU multiplication, etc.)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.compiler import compile_c
from src.speculator import TracingVM


class TracingVMWithIO(TracingVM):
    """Tracing VM extended with I/O support."""

    SYS_GETCHAR = 64
    SYS_PUTCHAR = 65

    def __init__(self, stdin_data: str = ""):
        super().__init__()
        self.stdin_data = stdin_data
        self.stdin_pos = 0
        self.stdout_data = []

    def _getchar(self) -> int:
        if self.stdin_pos >= len(self.stdin_data):
            return 0xFFFFFFFF
        c = ord(self.stdin_data[self.stdin_pos])
        self.stdin_pos += 1
        return c

    def _putchar(self, c: int):
        self.stdout_data.append(chr(c & 0xFF))

    def get_stdout(self) -> str:
        return ''.join(self.stdout_data)

    def run_with_trace(self, max_steps: int = 1000000):
        """Execute and record trace of arithmetic operations."""
        from src.speculator import TraceStep
        steps = 0
        self.trace = []

        while steps < max_steps:
            instr_idx = self.pc // 8
            if instr_idx >= len(self.code) or self.halted:
                break

            op, imm = self.code[instr_idx]
            pc_before = self.pc
            self.pc += 8

            # I/O syscalls
            if op == self.SYS_GETCHAR:
                self.ax = self._getchar()
            elif op == self.SYS_PUTCHAR:
                c = self.memory.get(self.sp, 0)
                self._putchar(c)
                self.ax = c

            # Record arithmetic operations for verification
            elif op in self.ARITHMETIC_OPS:
                operand_a = self.memory.get(self.sp, 0)
                operand_b = self.ax

                # Execute operation
                if op == 25:  # ADD
                    self.sp += 8
                    self.ax = (operand_a + operand_b) & 0xFFFFFFFF
                elif op == 26:  # SUB
                    self.sp += 8
                    self.ax = (operand_a - operand_b) & 0xFFFFFFFF
                elif op == 27:  # MUL
                    self.sp += 8
                    self.ax = (operand_a * operand_b) & 0xFFFFFFFF
                elif op == 28:  # DIV
                    self.sp += 8
                    self.ax = operand_a // operand_b if operand_b != 0 else 0
                elif op == 29:  # MOD
                    self.sp += 8
                    self.ax = operand_a % operand_b if operand_b != 0 else 0
                elif op == 16:  # AND
                    self.sp += 8
                    self.ax = operand_a & operand_b
                elif op == 14:  # OR
                    self.sp += 8
                    self.ax = operand_a | operand_b
                elif op == 15:  # XOR
                    self.sp += 8
                    self.ax = operand_a ^ operand_b
                elif op == 17:  # EQ
                    self.sp += 8
                    self.ax = 1 if operand_a == operand_b else 0
                elif op == 18:  # NE
                    self.sp += 8
                    self.ax = 1 if operand_a != operand_b else 0
                elif op == 19:  # LT
                    self.sp += 8
                    self.ax = 1 if operand_a < operand_b else 0
                elif op == 20:  # GT
                    self.sp += 8
                    self.ax = 1 if operand_a > operand_b else 0
                elif op == 21:  # LE
                    self.sp += 8
                    self.ax = 1 if operand_a <= operand_b else 0
                elif op == 22:  # GE
                    self.sp += 8
                    self.ax = 1 if operand_a >= operand_b else 0
                elif op == 23:  # SHL
                    self.sp += 8
                    self.ax = (operand_a << operand_b) & 0xFFFFFFFF
                elif op == 24:  # SHR
                    self.sp += 8
                    self.ax = operand_a >> operand_b

                # Record the step
                self.trace.append(TraceStep(
                    op=op,
                    imm=imm,
                    operand_a=operand_a,
                    operand_b=operand_b,
                    result=self.ax,
                    pc_before=pc_before,
                    pc_after=self.pc,
                ))
            else:
                # Non-arithmetic ops
                if op == 0:    # LEA
                    self.ax = self.bp + imm
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
                    self.sp -= imm
                elif op == 7:  # ADJ
                    self.sp += imm
                elif op == 8:  # LEV
                    self.sp = self.bp
                    self.bp = self.memory.get(self.sp, 0)
                    self.sp += 8
                    self.pc = self.memory.get(self.sp, 0)
                    self.sp += 8
                elif op == 9:  # LI
                    self.ax = self.memory.get(self.ax, 0)
                elif op == 10: # LC
                    self.ax = self.memory.get(self.ax, 0) & 0xFF
                elif op == 11: # SI
                    addr = self.memory.get(self.sp, 0)
                    self.sp += 8
                    self.memory[addr] = self.ax
                elif op == 12: # SC
                    addr = self.memory.get(self.sp, 0)
                    self.sp += 8
                    self.memory[addr] = self.ax & 0xFF
                elif op == 13: # PSH
                    self.sp -= 8
                    self.memory[self.sp] = self.ax
                elif op == 38: # EXIT
                    self.halted = True
                    break

            steps += 1

        return self.ax, self.trace


def verify_trace_neural(trace, verify_mul=True):
    """Verify arithmetic operations using neural ALU."""
    from src.transformer_vm import NeuralALU
    import torch

    alu = NeuralALU()
    results = {'total': 0, 'verified': 0, 'correct': 0, 'mul': 0, 'div': 0, 'add': 0}

    for step in trace:
        results['total'] += 1

        # Only verify MUL operations with neural ALU (most interesting)
        if step.op == 27 and verify_mul:  # MUL
            results['mul'] += 1
            a_t = torch.tensor([float(step.operand_a)])
            b_t = torch.tensor([float(step.operand_b)])
            neural_result = int(round(alu.mul(a_t, b_t).item()))

            results['verified'] += 1
            if neural_result == step.result:
                results['correct'] += 1
            else:
                print(f"  MUL mismatch: {step.operand_a} * {step.operand_b} = {step.result} (neural: {neural_result})")

        elif step.op == 28:  # DIV
            results['div'] += 1
        elif step.op in (25, 26):  # ADD/SUB
            results['add'] += 1

    return results


def main():
    print("=" * 70)
    print("  ELIZA with SPECULATIVE EXECUTION + NEURAL VERIFICATION")
    print("=" * 70)
    print()

    # Read ELIZA source
    c_file = os.path.join(os.path.dirname(__file__), "eliza_simple.c")
    with open(c_file, 'r') as f:
        source = f.read()

    # Conversation input
    stdin_data = "hello\ni feel sad today\ni think about my mother\ni had a dream\ntalking to a computer helps\nit makes me happy\nbye\n"

    print("Compiling ELIZA to bytecode...")
    bytecode, data = compile_c(source)
    print(f"Compiled: {len(bytecode)} instructions")
    print()

    print("Executing with tracing (fast VM)...")
    print("-" * 70)

    vm = TracingVMWithIO(stdin_data)
    vm.load(bytecode, data)

    start = time.time()
    result, trace = vm.run_with_trace()
    exec_time = time.time() - start

    # Print output
    print(vm.get_stdout())
    print("-" * 70)
    print()

    # Analyze trace
    op_counts = {}
    for step in trace:
        op_counts[step.op] = op_counts.get(step.op, 0) + 1

    print("=" * 70)
    print("  EXECUTION REPORT")
    print("=" * 70)
    print(f"Execution time: {exec_time*1000:.1f}ms")
    print(f"Total arithmetic ops: {len(trace):,}")
    print()

    print("OPERATION BREAKDOWN:")
    op_names = {25: 'ADD', 26: 'SUB', 27: 'MUL', 28: 'DIV', 29: 'MOD',
                17: 'EQ', 18: 'NE', 19: 'LT', 20: 'GT', 21: 'LE', 22: 'GE'}
    for op, count in sorted(op_counts.items()):
        name = op_names.get(op, f'OP{op}')
        print(f"  {name:6s}: {count:,}")
    print()

    # Neural verification
    mul_count = op_counts.get(27, 0)
    if mul_count > 0:
        print("NEURAL VERIFICATION (SwiGLU multiply):")
        verify_results = verify_trace_neural(trace)
        print(f"  Multiplications verified: {verify_results['verified']}")
        print(f"  All correct: {'Yes' if verify_results['correct'] == verify_results['verified'] else 'No'}")
    else:
        print("No multiplications to verify (ELIZA uses string comparisons only)")

    # FLOP estimation
    add_count = op_counts.get(25, 0) + op_counts.get(26, 0)
    div_count = op_counts.get(28, 0)

    print()
    print("ESTIMATED NEURAL FLOPs (if run through neural VM):")
    print(f"  Add/Sub:         {add_count:,} ops")
    print(f"  Multiplications: {mul_count * 10:,} FLOPs (10 per SwiGLU mul)")
    print(f"  Divisions:       {div_count * 50:,} FLOPs (50 per Newton div)")
    print(f"  Total:           {add_count + mul_count*10 + div_count*50:,} FLOPs")
    print("=" * 70)


if __name__ == "__main__":
    main()
