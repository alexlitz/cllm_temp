#!/usr/bin/env python3
"""Check if bytecode is being compiled correctly."""

from src.compiler import compile_c
from src.baked_c4 import BakedC4Transformer

# Test compilation
code = "int main() { return 42; }"
bytecode, data = compile_c(code)

print(f"Source: {code}")
print(f"Bytecode: {bytecode}")
print(f"Data: {data}")
print(f"Bytecode length: {len(bytecode)}")

# Test with fast VM
from src.speculator import FastLogicalVM
fast_vm = FastLogicalVM()
fast_vm.load(bytecode, data)
result = fast_vm.run()
print(f"\nFast VM result: {result}")
print(f"Expected: 42")

# Test with transformer VM
model = BakedC4Transformer(use_speculator=False)
print(f"\nTransformer VM has _use_neural_vm: {model.transformer_vm._use_neural_vm}")
model.transformer_vm.reset()
model.transformer_vm.load_bytecode(bytecode, data)
print(f"Has _neural_bytecode: {hasattr(model.transformer_vm, '_neural_bytecode')}")
if hasattr(model.transformer_vm, '_neural_bytecode'):
    print(f"_neural_bytecode length: {len(model.transformer_vm._neural_bytecode)}")
    print(f"First few instructions: {model.transformer_vm._neural_bytecode[:5]}")

result = model.transformer_vm.run(max_steps=1000)
print(f"\nTransformer VM result: {result}")
