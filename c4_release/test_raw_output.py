#!/usr/bin/env python3
"""Check raw transformer output before handlers."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.constants import INSTR_WIDTH

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

for step_num in range(3):
    print(f"\n=== Step {step_num} ===")
    
    # Generate step
    for i in range(40):
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        if next_token == Token.STEP_END:
            break
    
    # Extract BEFORE any handlers
    pc_raw = runner._extract_register(context, Token.REG_PC)
    print(f"RAW transformer output PC: {pc_raw}")
    
    # Check exec_pc
    exec_pc = runner._exec_pc()
    exec_idx = exec_pc // INSTR_WIDTH
    if 0 <= exec_idx < len(bytecode):
        exec_op = bytecode[exec_idx] & 0xFF
        print(f"Executed instruction: op={exec_op} at PC={exec_pc}")
    
    # Update _last_pc  
    runner._last_pc = pc_raw

print("\nConclusion: The transformer is outputting these PC values.")
print("If they're wrong, it's a model/weights issue, not a runner issue.")
