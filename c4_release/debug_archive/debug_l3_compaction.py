"""Check if L3 FFN is compacted and verify unit 0."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
set_vm_weights(model)

l3_ffn = model.blocks[2].ffn

print("L3 FFN configuration:")
print(f"Hidden dim: {l3_ffn.hidden_dim}")
print(f"Dim: {l3_ffn.dim}")
print(f"W_up shape: {l3_ffn.W_up.shape}")
print(f"b_up shape: {l3_ffn.b_up.shape}")
print(f"b_gate shape: {l3_ffn.b_gate.shape}")

print(f"\nIs compacted: {hasattr(l3_ffn, '_compact_size')}")
if hasattr(l3_ffn, '_compact_size'):
    print(f"Compact size: {l3_ffn._compact_size}")

print(f"\nUnit 0 weights:")
print(f"W_up[0, MARK_PC={BD.MARK_PC}]: {l3_ffn.W_up[0, BD.MARK_PC].item():.3f}")
print(f"b_up[0]: {l3_ffn.b_up[0].item():.3f}")
print(f"b_gate[0]: {l3_ffn.b_gate[0].item():.3f}")

print(f"\nChecking first 5 units MARK_PC weights:")
for i in range(min(5, l3_ffn.W_up.shape[0])):
    w = l3_ffn.W_up[i, BD.MARK_PC].item()
    b_up = l3_ffn.b_up[i].item()
    b_gate = l3_ffn.b_gate[i].item()
    print(f"Unit {i}: W_up[MARK_PC]={w:.1f}, b_up={b_up:.1f}, b_gate={b_gate:.3f}")
