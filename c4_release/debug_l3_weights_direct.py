"""Check L3 FFN weights directly at the matrix level."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
set_vm_weights(model)

l3_ffn = model.blocks[2].ffn

print("L3 FFN Unit 0 (first-step PC default):")
print(f"  W_up matrix shape: {l3_ffn.up.weight.shape}")  # Should be [hidden_dim, input_dim]
print(f"  Accessing W_up as: [unit=0, dim=MARK_PC={BD.MARK_PC}]")
print(f"  PyTorch Linear weight is transposed, so actual access is: weight[0, {BD.MARK_PC}]")

# Direct weight access
w_up_weight = l3_ffn.up.weight  # Shape: [hidden_dim, input_dim] = [4096, 512]
print(f"\n  l3_ffn.up.weight[0, 0]: {w_up_weight[0, 0].item():.3f}")
print(f"  l3_ffn.up.weight[0, {BD.MARK_PC}]: {w_up_weight[0, BD.MARK_PC].item():.3f}")

# Check via property
print(f"\n  l3_ffn.W_up[0, 0]: {l3_ffn.W_up[0, 0].item():.3f}")
print(f"  l3_ffn.W_up[0, MARK_PC={BD.MARK_PC}]: {l3_ffn.W_up[0, BD.MARK_PC].item():.3f}")

# Check if any weights in unit 0 are non-zero
nonzero_count = (l3_ffn.W_up[0].abs() > 0.001).sum().item()
print(f"\n  Non-zero weights in unit 0: {nonzero_count} / {l3_ffn.W_up.shape[1]}")

if nonzero_count > 0:
    nonzero_indices = (l3_ffn.W_up[0].abs() > 0.001).nonzero(as_tuple=True)[0]
    print(f"  Non-zero indices: {nonzero_indices.tolist()[:10]}")  # First 10
    for idx in nonzero_indices[:5]:
        print(f"    W_up[0, {idx.item()}] = {l3_ffn.W_up[0, idx].item():.3f}")

print(f"\n  b_up[0]: {l3_ffn.b_up[0].item():.3f}")
print(f"  b_gate[0]: {l3_ffn.b_gate[0].item():.3f}")

# Check unit 2 (should also have MARK_PC weight)
print(f"\nL3 FFN Unit 2 (first-step PC default for HI nibble):")
print(f"  l3_ffn.W_up[2, MARK_PC={BD.MARK_PC}]: {l3_ffn.W_up[2, BD.MARK_PC].item():.3f}")
print(f"  b_up[2]: {l3_ffn.b_up[2].item():.3f}")
print(f"  b_gate[2]: {l3_ffn.b_gate[2].item():.3f}")
