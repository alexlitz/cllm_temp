"""Test SHIFT-only integration first."""

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim,  set_vm_weights
from neural_vm.efficient_alu_integrated import EfficientALU_L13

print("="*70)
print("Testing SHIFT Integration")
print("="*70)

# Build model with standard weights
print("\nBuilding model...")
model = AutoregressiveVM()
set_vm_weights(model)

BD = _SetDim
S = 100.0

# Count L13 params before
before = sum((p != 0).sum().item() for p in model.blocks[13].ffn.parameters())
print(f"\nL13 params before: {before:,}")

# Replace L13 with efficient SHIFT
print("Replacing L13 FFN with efficient SHIFT...")
model.blocks[13].ffn = EfficientALU_L13(S, BD)

# Count after
after_layers = model.blocks[13].ffn.shl_layers + model.blocks[13].ffn.shr_layers
after = sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in after_layers)
print(f"L13 params after: {after:,}")
print(f"Savings: {before - after:,} ({(before-after)/before*100:.1f}%)")

# Test forward pass
print("\nTesting forward pass...")
try:
    x = torch.randint(0, 256, (1, 10))
    with torch.no_grad():
        output = model(x)
    print(f"✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
