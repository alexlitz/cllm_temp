#!/usr/bin/env python3
"""Test MUL weight extraction."""

from neural_vm.alu_weight_extractor import ALUWeightExtractor
from neural_vm.embedding import Opcode

# Create extractor
extractor = ALUWeightExtractor()

# Extract MUL weights
print("Extracting MUL weights...")
mul_weights = extractor.extract_mul_weights(opcode=Opcode.MUL)

print(f"Number of layers: {len(mul_weights.layers)}")

for i, layer_weights in enumerate(mul_weights.layers):
    if layer_weights is None:
        print(f"Layer {i}: Non-FFN layer (skipped)")
    else:
        W_up = layer_weights['W_up']
        W_down = layer_weights['W_down']
        print(f"Layer {i}: W_up shape {W_up.shape}, W_down shape {W_down.shape}")

print("\nExtraction successful!")
