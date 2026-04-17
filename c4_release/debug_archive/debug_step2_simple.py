#!/usr/bin/env python3
"""Simple debug of step 2 tokens."""

import torch
from neural_vm.vm_step import AutoregressiveVM, Token, _SetDim as BD
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

# Token names
TOKEN_NAMES = {
    257: "REG_PC",
    258: "REG_AX",
    259: "REG_SP",
    260: "REG_BP",
    261: "REG_STACK0",
    262: "STEP_END",
    263: "HALT",
    264: "REG_MEM",
}

bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26
    Opcode.EXIT,              # Instr 1: Never reached
    Opcode.NOP,               # Instr 2: Padding
    Opcode.IMM | (42 << 8),   # Instr 3: Target at PC=26
    Opcode.EXIT,              # Instr 4: Exit
]

runner = AutoregressiveVMRunner()
model = runner.model

# Build initial context
context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context length: {len(context)}")

# Generate step 1 (JSR)
print("\n=== Step 1 (JSR) ===")
step1_tokens = []
for i in range(35):
    token = model.generate_next(context)
    context.append(token)
    step1_tokens.append(token)
    name = TOKEN_NAMES.get(token, str(token))
    if i < 25:
        print(f"  Token {i+1}: {token:3d} ({name})")
    if token == Token.STEP_END:
        break

# Parse step 1
print("\nStep 1 summary:")
for i, t in enumerate(step1_tokens):
    if t == Token.REG_PC:
        pc_bytes = step1_tokens[i+1:i+5]
        pc_val = sum(b << (j*8) for j, b in enumerate(pc_bytes))
        print(f"  PC = {pc_val} (bytes: {list(pc_bytes)})")
    if t == Token.REG_AX:
        ax_bytes = step1_tokens[i+1:i+5]
        ax_val = sum(b << (j*8) for j, b in enumerate(ax_bytes))
        print(f"  AX = {ax_val} (bytes: {list(ax_bytes)})")
    if t == Token.REG_SP:
        sp_bytes = step1_tokens[i+1:i+5]
        sp_val = sum(b << (j*8) for j, b in enumerate(sp_bytes))
        print(f"  SP = {sp_val} (bytes: {list(sp_bytes)})")

# Generate step 2 (should be IMM 42)
print("\n=== Step 2 (should execute IMM 42) ===")
step2_tokens = []
for i in range(35):
    token = model.generate_next(context)
    context.append(token)
    step2_tokens.append(token)
    name = TOKEN_NAMES.get(token, str(token))
    if i < 25:
        print(f"  Token {i+1}: {token:3d} ({name})")
    if token == Token.STEP_END or token == Token.HALT:
        break

# Parse step 2
print("\nStep 2 summary:")
for i, t in enumerate(step2_tokens):
    if t == Token.REG_PC:
        pc_bytes = step2_tokens[i+1:i+5]
        pc_val = sum(b << (j*8) for j, b in enumerate(pc_bytes))
        print(f"  PC = {pc_val} (bytes: {list(pc_bytes)}) - expected 34")
    if t == Token.REG_AX:
        ax_bytes = step2_tokens[i+1:i+5]
        ax_val = sum(b << (j*8) for j, b in enumerate(ax_bytes))
        print(f"  AX = {ax_val} (bytes: {list(ax_bytes)}) - expected 42")
    if t == Token.REG_SP:
        sp_bytes = step2_tokens[i+1:i+5]
        sp_val = sum(b << (j*8) for j, b in enumerate(sp_bytes))
        print(f"  SP = {sp_val}")
    if t == Token.REG_STACK0:
        s0_bytes = step2_tokens[i+1:i+5]
        s0_val = sum(b << (j*8) for j, b in enumerate(s0_bytes))
        print(f"  STACK0 = {s0_val}")

# Final result
print("\n" + "="*60)
print("RESULT:")
for i, t in enumerate(step2_tokens):
    if t == Token.REG_PC:
        pc_bytes = step2_tokens[i+1:i+5]
        pc_val = sum(b << (j*8) for j, b in enumerate(pc_bytes))
        if pc_val == 34:
            print(f"  PC: PASS (34)")
        else:
            print(f"  PC: FAIL ({pc_val} != 34)")
    if t == Token.REG_AX:
        ax_bytes = step2_tokens[i+1:i+5]
        ax_val = sum(b << (j*8) for j, b in enumerate(ax_bytes))
        if ax_val == 42:
            print(f"  AX: PASS (42)")
        else:
            print(f"  AX: FAIL ({ax_val} != 42)")
