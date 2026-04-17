#!/usr/bin/env python3
"""Trace execution with fixed INSTR_WIDTH."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode
from neural_vm.constants import INSTR_WIDTH

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    pc = i * INSTR_WIDTH
    op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(op, f"?{op}")
    print(f"  PC={pc:3d} (idx={i}): {op_name:6s} {imm}")

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

for step_num in range(6):
    print(f"\n=== Step {step_num} ===")
    
    # Generate tokens for this step
    step_tokens = []
    for i in range(40):
        next_token = runner.model.generate_next(context)
        context.append(next_token)
        step_tokens.append(next_token)
        
        if next_token == Token.STEP_END:
            break
        elif next_token == Token.HALT:
            print("HALT detected!")
            break
    
    if next_token == Token.STEP_END:
        # Extract state
        pc = runner._extract_register(context, Token.REG_PC)
        ax = runner._extract_register(context, Token.REG_AX)
        sp = runner._extract_register(context, Token.REG_SP)
        
        # Determine what instruction was executed
        exec_pc = runner._exec_pc()
        exec_idx = exec_pc // INSTR_WIDTH
        
        if 0 <= exec_idx < len(bytecode):
            exec_instr = bytecode[exec_idx]
            exec_op = exec_instr & 0xFF
            exec_imm = exec_instr >> 8
            op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(exec_op, f"?{exec_op}")
            print(f"  Executed: PC={exec_pc} {op_name} {exec_imm}")
        else:
            print(f"  Executed: PC={exec_pc} (out of bounds)")
        
        print(f"  Output:   PC={pc}, AX={ax}, SP={sp}")
        print(f"  Tokens:   {len(step_tokens)} tokens")
        
        # Update tracking
        runner._last_pc = pc
    elif next_token == Token.HALT:
        ax = runner._extract_register(context, Token.REG_AX)
        print(f"  Final AX={ax}")
        break
    else:
        print(f"  ERROR: No terminator in 40 tokens")
        break

print(f"\nFinal result: AX={ax if 'ax' in locals() else 'unknown'}")
