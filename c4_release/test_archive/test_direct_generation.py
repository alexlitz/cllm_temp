#!/usr/bin/env python3
"""Test using direct generation like test_opcodes.py does."""

import sys
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from neural_vm.run_vm import AutoregressiveVMRunner
from src.compiler import compile_c

print("1. Compiling...", flush=True)
code = 'int main() { return 42; }'
bytecode, data = compile_c(code)

print("2. Creating model and setting weights...", flush=True)
model = AutoregressiveVM(n_layers=17)
set_vm_weights(model)

print("3. Building context...", flush=True)
runner = AutoregressiveVMRunner()
runner.model = model
context = runner._build_context(bytecode, data, [])
print(f"   Context length: {len(context)}", flush=True)

print("4. Generating tokens (max 200)...", flush=True)
generated = []
for i in range(200):
    if i % 10 == 0:
        print(f"   Step {i}...", flush=True)
    tok = model.generate_next(context)
    context.append(tok)
    generated.append(tok)
    if tok == Token.HALT:
        print(f"   HALT at step {i}", flush=True)
        break

print(f"5. Generated {len(generated)} tokens", flush=True)

# Extract exit code from AX register
exit_code = 0
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_AX and i + 4 < len(context):
        exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
        break

print(f"\n✅ Exit code: {exit_code} (expected: 42)")
