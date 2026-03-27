#!/usr/bin/env python3
"""Debug what the Fast VM and Neural VM actually return."""

from src.speculator import FastLogicalVM
from src.transformer_vm import C4TransformerVM
from src.compiler import compile_c

print("Debugging result comparison")
print("=" * 60)

code = "int main() { return 42; }"
bytecode, data = compile_c(code)

print(f"Test: {code}")
print(f"Bytecode: {bytecode}")
print()

# Fast VM
print("Fast VM:")
fast_vm = FastLogicalVM()
fast_vm.load(bytecode, data)
fast_result = fast_vm.run()
print(f"  Result: {fast_result}")
print(f"  Type: {type(fast_result)}")
print()

# Transformer VM
print("Transformer VM (Neural):")
transformer_vm = C4TransformerVM()
transformer_vm.reset()
transformer_vm.load_bytecode(bytecode, data)
trans_result = transformer_vm.run()
print(f"  Result: {trans_result}")
print(f"  Type: {type(trans_result)}")
print()

# Comparison
print("Comparison:")
print(f"  fast_result == trans_result: {fast_result == trans_result}")
print(f"  fast_result: {fast_result!r}")
print(f"  trans_result: {trans_result!r}")
print()

if isinstance(trans_result, tuple):
    output, exit_code = trans_result
    print("  trans_result is a TUPLE!")
    print(f"    output: {output!r}")
    print(f"    exit_code: {exit_code!r}")
    print()
    print(f"  Comparing fast_result ({fast_result}) vs exit_code ({exit_code}):")
    print(f"    Match: {fast_result == exit_code}")
