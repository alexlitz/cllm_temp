from neural_vm.alu.chunk_config import NIBBLE
from neural_vm.alu.ops.shift import build_shl_layers, build_shr_layers

shl = build_shl_layers(NIBBLE, opcode=23)
shr = build_shr_layers(NIBBLE, opcode=24)

print("SHL Layers:")
total_hidden = 0
for i, layer in enumerate(shl):
    if hasattr(layer, 'ffn'):
        if hasattr(layer.ffn, 'hidden_dim'):
            h = layer.ffn.hidden_dim
        else:
            h = layer.ffn.W_up.shape[0]
    else:
        if hasattr(layer, 'flat_ffn'):
            h = layer.flat_ffn.ffn.hidden_dim
        else:
            h = "unknown"
    params = sum((p != 0).sum().item() for p in layer.parameters())
    print(f"  Layer {i}: hidden={h}, params={params}")
    if isinstance(h, int):
        total_hidden += h

print(f"\nSHL total hidden units: {total_hidden}")

print("\nSHR Layers:")
total_hidden = 0
for i, layer in enumerate(shr):
    if hasattr(layer, 'ffn'):
        if hasattr(layer.ffn, 'hidden_dim'):
            h = layer.ffn.hidden_dim
        else:
            h = layer.ffn.W_up.shape[0]
    else:
        if hasattr(layer, 'flat_ffn'):
            h = layer.flat_ffn.ffn.hidden_dim
        else:
            h = "unknown"
    params = sum((p != 0).sum().item() for p in layer.parameters())
    print(f"  Layer {i}: hidden={h}, params={params}")
    if isinstance(h, int):
        total_hidden += h

print(f"\nSHR total hidden units: {total_hidden}")

print("\n" + "="*70)
print(f"Current SHIFT: 4,096 hidden units, 36,864 params")
print(f"Efficient SHIFT: {total_hidden * 2} total hidden units (approx), 5,624 params")
print(f"Savings: {36864 - 5624} params ({(36864-5624)/36864*100:.1f}%)")
