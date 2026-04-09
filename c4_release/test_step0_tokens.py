#!/usr/bin/env python3
"""Check what tokens are generated in step 0."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.embedding import Opcode
from neural_vm.run_vm import AutoregressiveVMRunner

print("Building model...")
model = AutoregressiveVM()
set_vm_weights(model)
model = model.cuda()
model.eval()

runner = AutoregressiveVMRunner()
runner.model = model
bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]

context = runner._build_context(bytecode, b'', [])
initial_len = len(context)
print(f"Initial context: {initial_len} tokens\n")

print("Generating step 0...")
for i in range(100):
    tok = model.generate_next(context)
    context.append(tok)
    print(f"Token {i}: {tok} ({hex(tok) if tok < 256 else tok})")
    if tok == Token.STEP_END or tok == Token.HALT:
        print(f"\nStep 0 complete with {tok}")
        break
    if i > 50:
        print("\nToo many tokens, stopping")
        break

print(f"\nTotal tokens generated: {len(context) - initial_len}")
