#!/usr/bin/env python3
"""Simple test to check if return 0 matches."""

from src.speculator import FastLogicalVM
from src.transformer_vm import C4TransformerVM
from neural_vm.vm_step import set_vm_weights
from src.compiler import compile_c

print("Simple Match Test")
print("=" * 60)

# Test: return 0
source = "int main() { return 0; }"
bytecode, data = compile_c(source)

print(f"Test: {source}")
print(f"Bytecode length: {len(bytecode)}")
print()

# Fast VM
print("Fast VM:")
fast_vm = FastLogicalVM()
fast_vm.load(bytecode, data)
fast_result = fast_vm.run()
print(f"  Result: {fast_result}")
print(f"  Type: {type(fast_result)}")
print()

# Transformer VM with limited steps
print("Transformer VM (max 20 steps):")
transformer_vm = C4TransformerVM()

# Set weights
try:
    set_vm_weights(transformer_vm.model)
    transformer_vm.model.compact(block_size=32)
    transformer_vm.model.compact_moe()
    print("  Weights loaded and compacted")
except Exception as e:
    print(f"  Warning: {e}")

transformer_vm.reset()
transformer_vm.load_bytecode(bytecode, data)

try:
    trans_result = transformer_vm.run(max_steps=20)
    print(f"  Result: {trans_result}")
    print(f"  Type: {type(trans_result)}")

    # Extract exit code
    if isinstance(trans_result, tuple):
        trans_output, trans_exit_code = trans_result
        print(f"  Output: {trans_output!r}")
        print(f"  Exit code: {trans_exit_code}")
    else:
        trans_exit_code = trans_result
        print(f"  Exit code: {trans_exit_code}")

    print()
    print("Comparison:")
    print(f"  Fast result: {fast_result}")
    print(f"  Neural exit code: {trans_exit_code}")
    print(f"  Match: {fast_result == trans_exit_code}")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
