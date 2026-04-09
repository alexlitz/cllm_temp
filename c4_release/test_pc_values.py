#!/usr/bin/env python3
"""Check what PC values are generated."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import Token
import inspect

# Simple program: IMM 42, EXIT
bytecode = [1 | (42 << 8), 34 | (0 << 8)]

sig = inspect.signature(AutoregressiveVMRunner.__init__)
if 'conversational_io' in sig.parameters:
    runner = AutoregressiveVMRunner(conversational_io=False)
else:
    runner = AutoregressiveVMRunner()

# Track all generated tokens
generated_tokens = []
original_gen = runner.model.generate_next

def track_gen(ctx):
    tok = original_gen(ctx)
    generated_tokens.append(tok)
    return tok

runner.model.generate_next = track_gen

print("Running IMM 42, EXIT...")
output, exit_code = runner.run(bytecode, b'', [], max_steps=3)

print(f"\nGenerated {len(generated_tokens)} tokens")

# Find REG_PC markers and the following 4 bytes
print("\nPC values in context:")
for i, tok in enumerate(generated_tokens):
    if tok == Token.REG_PC:
        if i + 4 < len(generated_tokens):
            pc_bytes = generated_tokens[i+1:i+5]
            pc_value = sum((b & 0xFF) << (j*8) for j, b in enumerate(pc_bytes))
            print(f"  Step {i//35}: REG_PC followed by bytes {pc_bytes} → 0x{pc_value:08x} (PC={pc_value})")
            if pc_value == 0:
                print(f"    ✗ PC stuck at 0!")
            elif pc_value == 6:  # PC_OFFSET + INSTR_WIDTH = 2 + 4 = 6
                print(f"    ✓ PC advanced correctly!")
