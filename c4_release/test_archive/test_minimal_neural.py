#!/usr/bin/env python3
"""Minimal test with debug output."""

import sys
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token
from src.compiler import compile_c

print("Creating model...")
sys.stdout.flush()
model = AutoregressiveVM()

print("Setting weights...")
sys.stdout.flush()
set_vm_weights(model)

print("Compiling test program...")
sys.stdout.flush()
source = "int main() { return 0; }"
bytecode, data = compile_c(source)
print(f"Bytecode: {bytecode}")

print("Building context...")
sys.stdout.flush()

# Build minimal context (from run_vm.py logic)
context = []
context.append(Token.CODE_START)
for instr in bytecode:
    op = instr & 0xFF
    imm = instr >> 8
    context.extend([op, imm & 0xFF, (imm >> 8) & 0xFF, (imm >> 16) & 0xFF])
context.append(Token.CODE_END)

print(f"Context length: {len(context)}")
print(f"Context: {context[:20]}...")

print("\nAttempting to generate 1 token...")
sys.stdout.flush()

try:
    # Add timeout via signal
    import signal
    def timeout_handler(signum, frame):
        print("\nTIMEOUT in generate_next!")
        sys.exit(124)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

    print("Calling model.generate_next()...")
    sys.stdout.flush()

    tok = model.generate_next(context)

    signal.alarm(0)  # Cancel alarm
    print(f"Success! Generated token: {tok}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
