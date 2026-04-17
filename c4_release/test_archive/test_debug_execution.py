#!/usr/bin/env python3
"""Debug execution to see what's happening."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode

print("Compiling: int main() { return 42; }")
bytecode, syms = compile_c('int main() { return 42; }')

print(f"\nBytecode ({len(bytecode)} instructions):")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(op, f"UNK({op})")
    print(f"  {i*5:3d}: {op_name:8s} {imm}")

print(f"\nSymbols: {syms}")

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Patch runner to print first 5 steps
original_handler_jsr = runner._handler_jsr
step_count = [0]

def debug_jsr(context, output):
    step_count[0] += 1
    if step_count[0] <= 3:
        pc = runner._extract_register(context, runner.model.Token.REG_PC)
        ax = runner._extract_register(context, runner.model.Token.REG_AX)
        sp = runner._extract_register(context, runner.model.Token.REG_SP)
        print(f"  Step {step_count[0]}: JSR at PC={pc}, AX={ax}, SP={sp}")
    original_handler_jsr(context, output)

runner._handler_jsr = debug_jsr

print("\nRunning (max_steps=10)...")
result = runner.run(bytecode, max_steps=10)
print(f"\nResult: {result}")
