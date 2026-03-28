#!/usr/bin/env python3
"""Trace handler dispatch."""
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

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])
output = []

print("Manually running first 2 steps with debug...")

for step_num in range(2):
    print(f"\n=== Step {step_num} ===")
    
    # Generate tokens
    for token_idx in range(40):
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        
        if next_token == Token.STEP_END:
            print(f"  STEP_END at token {token_idx}")
            
            # Check what run() would do
            pc = runner._extract_register(context, Token.REG_PC)
            print(f"  Output PC: {pc}")
            
            if pc is not None:
                instr_idx = pc // 5
                print(f"  Output instr_idx: {instr_idx}")
                
                if 0 <= instr_idx < len(bytecode):
                    op = bytecode[instr_idx] & 0xFF
                    op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(op, f"?{op}")
                    print(f"  Output op: {op_name}")
            
            # Check exec_pc dispatch
            exec_pc = runner._exec_pc()
            exec_idx = exec_pc // 5
            print(f"  Exec PC: {exec_pc}, idx: {exec_idx}")
            
            if 0 <= exec_idx < len(bytecode):
                exec_op = bytecode[exec_idx] & 0xFF
                op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(exec_op, f"?{exec_op}")
                print(f"  Exec op: {op_name}")
                
                func_handler = runner._func_call_handlers.get(exec_op)
                print(f"  Has func_handler: {func_handler is not None}")
                
                if func_handler:
                    print(f"  → Calling {op_name} handler")
                    func_handler(context, output)
                    print(f"  → Handler complete")
            
            # Update tracking
            runner._last_pc = pc
            break
        elif next_token == Token.HALT:
            print(f"  HALT")
            break
    else:
        print(f"  ERROR: No terminator")
        break

print(f"\nFinal PC: {runner._last_pc}")
