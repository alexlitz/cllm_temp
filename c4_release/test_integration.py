"""Test integration of efficient SHIFT into vm_step.py"""

import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights

print("="*70)
print("Testing Efficient SHIFT Integration")
print("="*70)

print("\nBuilding model with efficient SHIFT...")
model = AutoregressiveVM()
set_vm_weights(model)

# Count parameters in L13 FFN
print("\n--- L13 FFN Parameters ---")
ffn13 = model.blocks[13].ffn
if hasattr(ffn13, 'shl_layers'):
    # Efficient implementation
    shl_params = sum(sum((p != 0).sum().item() if p.is_sparse else (p != 0).sum().item()
                         for p in layer.parameters())
                     for layer in ffn13.shl_layers)
    shr_params = sum(sum((p != 0).sum().item() if p.is_sparse else (p != 0).sum().item()
                         for p in layer.parameters())
                     for layer in ffn13.shr_layers)
    total_params = shl_params + shr_params
    print(f"✓ L13 FFN is EfficientShiftFFN")
    print(f"  SHL params: {shl_params:,}")
    print(f"  SHR params: {shr_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"  Previous: 36,864")
    print(f"  Savings: {36864 - total_params:,} ({(36864-total_params)/36864*100:.1f}%)")
else:
    # Old implementation
    print(f"✗ L13 FFN is still PureFFN (integration failed)")
    import sys
    sys.exit(1)

# Count total model parameters
print("\n--- Total Model Parameters ---")
total = 0
for name, param in model.named_parameters():
    if param.is_sparse:
        count = (param.to_dense() != 0).sum().item()
    else:
        count = (param != 0).sum().item()
    total += count

print(f"Total non-zero params: {total:,}")
print(f"Previous total: ~141,740")
print(f"Expected new total: ~110,500 (141,740 - 31,240)")
print(f"Actual savings: {141740 - total:,}")

# Test forward pass
print("\n--- Testing Forward Pass ---")
try:
    # Create a small input
    x = torch.randint(0, 256, (1, 10))  # [batch=1, seq_len=10]

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
print("✓ Efficient SHIFT Integration Test Complete")
print("="*70)
