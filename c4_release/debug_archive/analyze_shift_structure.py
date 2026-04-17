"""Analyze the structure of efficient SHIFT to understand if we can simplify for 8-bit."""

from neural_vm.alu.chunk_config import NIBBLE, BYTE
from neural_vm.alu.ops.shift import build_shl_layers, build_shr_layers

print("="*70)
print("SHIFT IMPLEMENTATION ANALYSIS")
print("="*70)

for config_name, config in [("NIBBLE (4-bit chunks)", NIBBLE), ("BYTE (8-bit chunks)", BYTE)]:
    print(f"\n{config_name}:")
    print(f"  chunk_bits={config.chunk_bits}, num_positions={config.num_positions}")
    
    shl = build_shl_layers(config, opcode=23)
    print(f"\n  SHL has {len(shl)} layers:")
    for i, layer in enumerate(shl):
        layer_type = type(layer).__name__
        params = sum((p != 0).sum().item() for p in layer.parameters())
        print(f"    Layer {i} ({layer_type}): {params} params")

print("\n" + "="*70)
print("\nKEY INSIGHT:")
print("  BYTE config (chunk_bits=8) has only 1 layer (no precompute needed)")
print("  This could fit into a single vm_step.py FFN layer!")
print("\n  But BYTE treats value as 4 positions of 8-bit chunks (32-bit total)")
print("  We need 8-bit total value, which doesn't fit standard configs")
