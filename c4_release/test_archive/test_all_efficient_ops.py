"""Test all efficient ALU operations integration."""

import torch
from neural_vm.vm_step import AutoregressiveVM, _SetDim, set_vm_weights
from neural_vm.efficient_alu_integrated import EfficientALU_L8_L9, EfficientALU_L13

print("="*70)
print("Testing All Efficient ALU Operations")
print("="*70)

# Build model with standard weights
print("\nBuilding model...")
model = AutoregressiveVM()
set_vm_weights(model)

BD = _SetDim
S = 100.0

# Measure before
before = {}
for i in [8, 9, 13]:
    before[i] = sum((p != 0).sum().item() for p in model.blocks[i].ffn.parameters())

print("\nBefore integration:")
for i in sorted(before.keys()):
    print(f"  L{i}: {before[i]:,} params")
print(f"  Total: {sum(before.values()):,} params")

# Replace with efficient implementations
print("\nIntegrating efficient operations...")

# L8-L9: ADD/SUB (combined into L8, leave L9 for now)
print("  L8: EfficientALU_L8_L9 (ADD/SUB)")
model.blocks[8].ffn = EfficientALU_L8_L9(S, BD)

# L13: SHIFT
print("  L13: EfficientALU_L13 (SHL/SHR)")
model.blocks[13].ffn = EfficientALU_L13(S, BD)

# Measure after
after = {}
for i in [8, 9, 13]:
    ffn = model.blocks[i].ffn
    if hasattr(ffn, 'add_layers'):
        params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                    for layer in ffn.add_layers + ffn.sub_layers)
    elif hasattr(ffn, 'shl_layers'):
        params = sum(sum((p != 0).sum().item() for p in layer.parameters())
                    for layer in ffn.shl_layers + ffn.shr_layers)
    else:
        params = sum((p != 0).sum().item() for p in ffn.parameters())
    after[i] = params

print("\nAfter integration:")
for i in sorted(after.keys()):
    print(f"  L{i}: {after[i]:,} params")
print(f"  Total: {sum(after.values()):,} params")

print("\n" + "="*70)
print("Savings:")
for i in sorted(before.keys()):
    savings = before[i] - after[i]
    pct = savings / before[i] * 100 if before[i] > 0 else 0
    print(f"  L{i}: {savings:,} params ({pct:.1f}%)")
total_savings = sum(before.values()) - sum(after.values())
total_pct = total_savings / sum(before.values()) * 100
print(f"  Total: {total_savings:,} params ({total_pct:.1f}%)")

# Test forward pass
print("\n" + "="*70)
print("Testing forward pass...")
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

print("\n" + "="*70)
