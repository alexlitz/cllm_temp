#!/usr/bin/env python3
"""Trace execution step by step."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode

bytecode, _ = compile_c('int main() { return 42; }')

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Build initial context
runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

print(f"Initial context length: {len(context)}")

for step in range(5):
    print(f"\n=== Step {step} ===")
    
    # Generate one full step (up to 35 tokens)
    for token_idx in range(40):  # Extra buffer
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        
        if next_token == Token.STEP_END or next_token == Token.HALT:
            token_name = "STEP_END" if next_token == Token.STEP_END else "HALT"
            print(f"  {token_name} at token {token_idx}")
            
            # Extract registers
            pc = runner._extract_register(context, Token.REG_PC)
            ax = runner._extract_register(context, Token.REG_AX)
            sp = runner._extract_register(context, Token.REG_SP)
            
            print(f"  Output: PC={pc}, AX={ax}, SP={sp}")
            
            # Check exec_pc
            exec_pc = runner._exec_pc()
            exec_idx = exec_pc // 5
            print(f"  Exec PC: {exec_pc} (idx={exec_idx})")
            
            if 0 <= exec_idx < len(bytecode):
                exec_op = bytecode[exec_idx] & 0xFF
                exec_imm = bytecode[exec_idx] >> 8
                op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(exec_op, f"UNK({exec_op})")
                print(f"  Executed: {op_name} {exec_imm}")
                
                # Check if handler would be called
                if exec_op in runner._func_call_handlers:
                    print(f"  → Would call {op_name} handler")
            
            # Update _last_pc
            runner._last_pc = pc
            
            if next_token == Token.HALT:
                print(f"\nHALT detected, final AX={ax}")
                print(f"Final result: ('', {ax if ax is not None else 0})")
                sys.exit(0)
            break
    else:
        print("  ERROR: No STEP_END/HALT in 40 tokens!")
        break

print(f"\nCompleted 5 steps without HALT")
