#\!/usr/bin/env python3
"""Verify actual exit code extraction."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

# Build context manually (no runner)
from neural_vm.run_vm import AutoregressiveVMRunner
runner = AutoregressiveVMRunner()
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
context = runner._build_context(bytecode, b'', [])

print("Running IMM 42, EXIT...\n")

# Generate both steps
for step in range(2):
    for i in range(50):
        tok = model.generate_next(context)
        context.append(tok)
        if tok == Token.STEP_END or tok == Token.HALT:
            print(f"Step {step} complete ({'HALT' if tok == Token.HALT else 'STEP_END'})")
            break

# Extract exit code using runner method
exit_code = runner._extract_register(context, 260)  # REG_AX = 260
print(f"\nExtracted exit code: {exit_code}")

if exit_code == 42:
    print("✓ CORRECT exit code\!")
elif exit_code == 0x00010000:
    print(f"✗ Exit code is {exit_code} (byte order issue)")
else:
    print(f"✗ Exit code is {exit_code}")
