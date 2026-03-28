#!/usr/bin/env python3
"""Detailed execution trace."""
import sys
sys.path.insert(0, '.')
import torch

from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
from neural_vm.embedding import Opcode

code = 'int main() { return 42; }'
bytecode, _ = compile_c(code)

print("Bytecode:")
for i, instr in enumerate(bytecode):
    op = instr & 0xFF
    imm = instr >> 8
    op_name = {v: k for k, v in vars(Opcode).items() if isinstance(v, int)}.get(op, f"UNK({op})")
    print(f"  {i*5:3d}: {op_name:8s} {imm}")

runner = AutoregressiveVMRunner()
if torch.cuda.is_available():
    runner.model.cuda()

# Build context
runner._bytecode = bytecode
context = runner._build_context(bytecode, b'', [])

print(f"\nInitial context: {len(context)} tokens")
print(f"Context: {context}")

# Generate first step and check what we get
print("\n=== Generating Step 0 ===")
for i in range(40):
    next_token = runner.model.generate_next(context)
    context.append(next_token)
    
    token_name = None
    if next_token == Token.STEP_END:
        token_name = "STEP_END"
    elif next_token == Token.HALT:
        token_name = "HALT"
    elif next_token < 256:
        token_name = f"BYTE({next_token})"
    elif next_token >= Token.REG_PC and next_token <= Token.STACK0:
        token_names = {Token.REG_PC: "REG_PC", Token.REG_AX: "REG_AX", 
                      Token.REG_SP: "REG_SP", Token.REG_BP: "REG_BP", Token.STACK0: "STACK0"}
        token_name = token_names.get(next_token, f"REG({next_token})")
    elif next_token == Token.MEM:
        token_name = "MEM"
    else:
        token_name = f"TOKEN({next_token})"
    
    print(f"  [{i}] {next_token:3d} = {token_name}")
    
    if next_token == Token.STEP_END:
        print("\nExtracting registers from step:")
        pc = runner._extract_register(context, Token.REG_PC)
        ax = runner._extract_register(context, Token.REG_AX)
        sp = runner._extract_register(context, Token.REG_SP)
        bp = runner._extract_register(context, Token.REG_BP)
        stack0 = runner._extract_register(context, Token.STACK0)
        print(f"  PC={pc}, AX={ax}, SP={sp}, BP={bp}, STACK0={stack0}")
        break
    elif next_token == Token.HALT:
        print("\nHALT detected!")
        break
else:
    print("\nERROR: No STEP_END/HALT in 40 tokens")

print(f"\nFinal context length: {len(context)}")
