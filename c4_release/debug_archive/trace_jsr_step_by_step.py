#!/usr/bin/env python3
"""Trace JSR execution token by token through the first step."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token

bytecode = [Opcode.JSR | (26 << 8)]
runner = AutoregressiveVMRunner()
model = runner.model

context = runner._build_context(bytecode, b"", [], "")
print(f"Initial context: {context}")
print(f"Expected tokens: CODE_START, JSR(3), imm(26), padding...")

# Generate tokens one by one and check for PC marker
print("\nGenerating step tokens...")
for i in range(10):
    next_token = model.generate_next(context)
    context.append(next_token)
    print(f"Token {i}: {next_token}", end="")

    if next_token == Token.REG_PC:
        print(" ← PC MARKER FOUND!")
        pc_marker_idx = len(context) - 1

        # Now generate the next few tokens (PC bytes)
        for j in range(4):
            next_token = model.generate_next(context)
            context.append(next_token)
            print(f"Token {i+j+1}: {next_token} (PC byte {j})")

        # Compute PC value
        pc_bytes = context[pc_marker_idx+1:pc_marker_idx+5]
        pc_value = sum([b << (k*8) for k, b in enumerate(pc_bytes)])
        print(f"\nPC value: {pc_value}")
        print(f"Expected: 26 if JSR worked, 10 if failed")

        if pc_value == 26:
            print("✓ JSR WORKED!")
        elif pc_value == 10:
            print("✗ JSR FAILED - PC advanced normally")
        else:
            print(f"? Unexpected PC: {pc_value}")

        break
    else:
        print()
