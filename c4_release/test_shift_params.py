from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.shift import build_shl_layers, build_shr_layers

# Build efficient layers
shl_layers = build_shl_layers(NIBBLE, opcode=23)
shr_layers = build_shr_layers(NIBBLE, opcode=24)

print("SHL layers:")
for i, layer in enumerate(shl_layers):
    layer_params = sum((p != 0).sum().item() for p in layer.parameters())
    print(f"  Layer {i}: {layer_params:,} params")

print("\nSHR layers:")
for i, layer in enumerate(shr_layers):
    layer_params = sum((p != 0).sum().item() for p in layer.parameters())
    print(f"  Layer {i}: {layer_params:,} params")

total = sum((p != 0).sum().item() for p in shl_layers.parameters()) + \
        sum((p != 0).sum().item() for p in shr_layers.parameters())
print(f"\nTotal efficient SHIFT: {total:,} params")
print(f"Current SHIFT (lookup): 36,864 params")
print(f"Savings: {36864 - total:,} params ({(36864-total)/36864*100:.1f}%)")
