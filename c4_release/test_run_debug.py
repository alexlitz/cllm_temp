#!/usr/bin/env python3
"""Test run with debug output."""
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

# Manually run the loop with debug
runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])
output = []

max_steps = 5
step_count = 0

for i in range(max_steps * Token.STEP_TOKENS):
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    
    if next_token == Token.STEP_END:
        step_count += 1
        print(f"\n=== STEP_END #{step_count} ===")
        
        # Extract PC
        pc = runner._extract_register(context, Token.REG_PC)
        print(f"Output PC: {pc}")
        
        # Check output opcode
        if pc is not None:
            instr_idx = pc // INSTR_WIDTH
            if 0 <= instr_idx < len(bytecode):
                op = bytecode[instr_idx] & 0xFF
                print(f"Output op: {op} (idx={instr_idx})")
        
        # Check exec_pc
        exec_pc = runner._exec_pc()
        exec_idx = exec_pc // INSTR_WIDTH
        print(f"Exec PC: {exec_pc} (idx={exec_idx})")
        
        if 0 <= exec_idx < len(bytecode):
            exec_op = bytecode[exec_idx] & 0xFF
            print(f"Exec op: {exec_op}")
            
            # Check if it's in func_call_handlers
            func_handler = runner._func_call_handlers.get(exec_op)
            print(f"Has func_handler: {func_handler is not None}")
            if func_handler:
                print(f"Calling handler...")
                func_handler(context, output)
                pc_after = runner._extract_register(context, Token.REG_PC)
                print(f"PC after handler: {pc_after}")
        
        # Update _last_pc
        runner._last_pc = pc
        
        if step_count >= max_steps:
            break
    
    elif next_token == Token.HALT:
        print("\nHALT")
        break

print(f"\nCompleted {step_count} steps")
