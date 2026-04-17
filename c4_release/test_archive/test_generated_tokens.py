#!/usr/bin/env python3
"""Check what tokens are generated for AX bytes."""

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

print(f"\nExit code: {exit_code} (0x{exit_code:08x})")
print(f"\nGenerated {len(generated_tokens)} tokens")

# Find REG_AX markers and the following 4 bytes
print("\nAX values in context:")
for i, tok in enumerate(generated_tokens):
    if tok == Token.REG_AX:
        if i + 4 < len(generated_tokens):
            ax_bytes = generated_tokens[i+1:i+5]
            ax_value = sum((b & 0xFF) << (j*8) for j, b in enumerate(ax_bytes))
            print(f"  Position {i}: REG_AX followed by bytes {ax_bytes} → 0x{ax_value:08x} ({ax_value})")
