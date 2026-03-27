#!/usr/bin/env python3
"""Test neural VM with limited steps like in test_opcodes.py."""

import sys
from src.compiler import compile_c
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.vm_step import set_vm_weights, Token

print("Testing Neural VM with limited steps")
print("=" * 60)

source = "int main() { return 0; }"
bytecode, data = compile_c(source)

print(f"Source: {source}")
print(f"Bytecode: {bytecode}")
print()

print("Creating runner...")
sys.stdout.flush()
runner = AutoregressiveVMRunner()
set_vm_weights(runner.model)
# Compact calls removed - causing AttributeError
# runner.model.compact(block_size=32)
# runner.model.compact_moe()
print("Runner created")
sys.stdout.flush()

print("Building context...")
sys.stdout.flush()
context = runner._build_context(bytecode, data or b"", [])
print(f"Context length: {len(context)}")
sys.stdout.flush()

print("Generating tokens (max 5 steps * 35 tokens/step + 10 = 185 tokens)...")
sys.stdout.flush()

max_steps = 5
generated = []
for i in range(max_steps * Token.STEP_TOKENS + 10):
    if i % 10 == 0:
        print(f"  Token {i}...")
        sys.stdout.flush()
    tok = runner.model.generate_next(context)
    context.append(tok)
    generated.append(tok)
    if tok == Token.HALT:
        print(f"  HALT at token {i}")
        sys.stdout.flush()
        break

print(f"Generated {len(generated)} tokens")
sys.stdout.flush()

# Extract exit code
exit_code = 0
for i in range(len(context) - 1, -1, -1):
    if context[i] == Token.REG_AX and i + 4 < len(context):
        exit_code = sum(context[i + 1 + j] << (j * 8) for j in range(4))
        break

print()
print(f"Exit code: {exit_code}")
print(f"Expected: 0")
print(f"Match: {exit_code == 0}")
