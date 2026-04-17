#!/usr/bin/env python3
"""Trace JSR execution step by step."""

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import Token

bytecode = [
    Opcode.JSR | (26 << 8),  # Instr 0: Jump to PC=26
    Opcode.EXIT,              # Instr 1: Never reached
    Opcode.NOP,               # Instr 2: Padding
    Opcode.IMM | (42 << 8),   # Instr 3: Target
    Opcode.EXIT,              # Instr 4: Exit
]

print("Creating runner...")
runner = AutoregressiveVMRunner()

# Patch to trace execution
original_run = runner.run

def traced_run(bytecode, data, **kwargs):
    # Trace the context generation
    context = runner._build_context(bytecode, data, [], "")
    model = runner.model

    print(f"\nInitial context length: {len(context)}")
    print(f"Initial context: {context[:20]}...")

    # Generate step 1
    print("\n=== Step 1 (JSR) ===")
    for i in range(35):
        token = model.generate_next(context)
        context.append(token)
        if i < 10 or token >= 256:
            name = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "STACK0", 262: "STEP_END"}.get(token, str(token))
            print(f"  Token {i+1}: {token} ({name})")
        if token == Token.STEP_END:
            break

    # Extract step 1 PC
    pc_marker_idx = None
    for i, t in enumerate(context):
        if t == 257:  # PC marker
            pc_marker_idx = i
    if pc_marker_idx:
        pc_bytes = context[pc_marker_idx+1:pc_marker_idx+5]
        pc_value = sum(b << (i*8) for i, b in enumerate(pc_bytes))
        print(f"  Step 1 PC bytes: {pc_bytes} = {pc_value}")

    # Generate step 2
    print("\n=== Step 2 (should be IMM 42) ===")
    for i in range(35):
        token = model.generate_next(context)
        context.append(token)
        if i < 15 or token >= 256:
            name = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "STACK0", 262: "STEP_END", 263: "HALT"}.get(token, str(token))
            print(f"  Token {i+1}: {token} ({name})")
        if token == Token.HALT:
            # Get AX value
            ax_idx = None
            for j in range(len(context)-36, len(context)):
                if context[j] == 258:  # AX marker
                    ax_idx = j
                    break
            if ax_idx:
                ax_bytes = context[ax_idx+1:ax_idx+5]
                ax_value = sum(b << (i*8) for i, b in enumerate(ax_bytes))
                print(f"  Step 2 AX bytes: {ax_bytes} = {ax_value}")
            break
        if token == Token.STEP_END:
            break

    # Now run for real
    return original_run(bytecode, data, **kwargs)

runner.run = traced_run
result = runner.run(bytecode, b"", max_steps=5)
print(f"\nFinal result: {result}")
