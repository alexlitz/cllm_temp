#!/usr/bin/env python3
"""Test Fast VM and Neural VM separately with short timeout."""

import sys
import signal

def timeout_handler(signum, frame):
    print("\n  TIMEOUT!")
    sys.exit(124)

from src.speculator import FastLogicalVM
from src.compiler import compile_c

print("Testing Fast VM")
print("=" * 60)

source = "int main() { return 0; }"
bytecode, data = compile_c(source)
print(f"Source: {source}")
print(f"Bytecode: {bytecode}")
print()

print("Fast VM:")
fast_vm = FastLogicalVM()
fast_vm.load(bytecode, data)
result = fast_vm.run()
print(f"  Result: {result}")
print(f"  Type: {type(result)}")
print()

print("=" * 60)
print("Testing Transformer VM (Neural)")
print("=" * 60)

from src.transformer_vm import C4TransformerVM

print("Creating C4TransformerVM...")
sys.stdout.flush()
transformer_vm = C4TransformerVM()
print("Created")
sys.stdout.flush()

print("Resetting...")
sys.stdout.flush()
transformer_vm.reset()
print("Reset complete")
sys.stdout.flush()

print("Loading bytecode...")
sys.stdout.flush()
transformer_vm.load_bytecode(bytecode, data)
print("Bytecode loaded")
sys.stdout.flush()

print("Setting 30 second timeout...")
sys.stdout.flush()
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("Running transformer...")
sys.stdout.flush()
result = transformer_vm.run()

signal.alarm(0)  # Cancel alarm
print(f"  Result: {result}")
print(f"  Type: {type(result)}")

if isinstance(result, tuple):
    output, exit_code = result
    print(f"  Output: {output!r}")
    print(f"  Exit code: {exit_code}")
