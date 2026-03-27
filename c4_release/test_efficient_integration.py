"""Test comprehensive efficient ALU integration."""

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim
from neural_vm.efficient_alu_integrated import integrate_efficient_alu

print("="*70)
print("Testing Comprehensive Efficient ALU Integration")
print("="*70)

# Build model
print("\nBuilding model...")
model = AutoregressiveVM()

# Get dimension constants
BD = _SetDim
S = 100.0

# Integrate efficient ALU
stats = integrate_efficient_alu(model, S, BD)

print("\n" + "="*70)
print("Integration Complete!")
print("="*70)

# Test forward pass
print("\nTesting forward pass...")
try:
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        output = model(x)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
