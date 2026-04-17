#!/usr/bin/env python3
"""Trace with override debugging."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Patch handler to see if it's called
original_handler_jsr = runner._handler_jsr
jsr_calls = []

def debug_jsr(context, output):
    exec_pc = runner._exec_pc()
    exec_idx = exec_pc // 5
    instr = runner._bytecode[exec_idx]
    target = instr >> 8
    
    pc_before = runner._extract_register(context, Token.REG_PC)
    jsr_calls.append(f"JSR handler: exec_pc={exec_pc}, target={target}, PC_before={pc_before}")
    
    original_handler_jsr(context, output)
    
    pc_after = runner._extract_register(context, Token.REG_PC)
    jsr_calls.append(f"  → PC_after={pc_after}")

runner._handler_jsr = debug_jsr

print("Running program...")
result = runner.run(bytecode, max_steps=10)
print(f"\nResult: {result}")

print(f"\nJSR handler calls: {len(jsr_calls)}")
for call in jsr_calls:
    print(f"  {call}")
