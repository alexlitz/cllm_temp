#!/usr/bin/env python3
"""Test just the transformer VM."""

import sys
import signal

def timeout_handler(signum, frame):
    print("\n  TIMEOUT!")
    sys.exit(124)

from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c

print("Creating C4TransformerVM...")
sys.stdout.flush()
vm = C4TransformerVM()
print("Created successfully")
print(f"Using neural VM: {vm._use_neural_vm}")
sys.stdout.flush()

source = "int main() { return 0; }"
bytecode, data = compile_c(source)

print(f"\nTest: {source}")
print(f"Bytecode: {bytecode}")
sys.stdout.flush()

print("\nResetting VM...")
sys.stdout.flush()
vm.reset()
print("Reset complete")
sys.stdout.flush()

print("Loading bytecode...")
sys.stdout.flush()
vm.load_bytecode(bytecode, data)
print("Bytecode loaded")
sys.stdout.flush()

print("\nSetting 30 second timeout and running...")
sys.stdout.flush()
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

result = vm.run(max_steps=100)

signal.alarm(0)  # Cancel alarm
print(f"Result: {result}")
print(f"Type: {type(result)}")

if isinstance(result, tuple):
    output, exit_code = result
    print(f"  Output: {output!r}")
    print(f"  Exit code: {exit_code}")
