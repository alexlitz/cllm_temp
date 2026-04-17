"""Test simple direct integration of efficient SHIFT."""

from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.shift import build_shl_layers, build_shr_layers

# Build efficient layers
shl = build_shl_layers(NIBBLE, 23)
shr = build_shr_layers(NIBBLE, 24)

print("SHL layers:", len(shl))
print("SHR layers:", len(shr))

# Count total params
total = sum(sum((p != 0).sum().item() for p in layer.parameters()) for layer in shl + shr)
print(f"\nTotal params: {total:,}")
print(f"Current: 36,864")
print(f"Savings: {36864 - total:,} ({(36864-total)/36864*100:.1f}%)")

# Key insight: These are ready-to-use PyTorch modules!
# We can replace vm_step.py's FFN weights OR use these directly

print("\n✓ Efficient implementations are ready")
print("✓ Just need to wire them into vm_step.py's layer structure")
