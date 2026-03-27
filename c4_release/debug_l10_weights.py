"""Check if Layer 10 head 1 weights are present after compaction."""
import torch
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD

model = AutoregressiveVM()
set_vm_weights(model)

print("=== BEFORE COMPACTION ===")
attn = model.blocks[10].attn
HD = 64  # head_dim
base = HD  # head 1

print(f"W_q[base + 0, BD.HAS_SE] = {attn.W_q[base + 0, BD.HAS_SE].item():.2f}")
print(f"W_q[base + 33, BD.HAS_SE] = {attn.W_q[base + 33, BD.HAS_SE].item():.2f}")
print(f"W_q[base + 33, BD.H1 + 1] = {attn.W_q[base + 33, BD.H1 + 1].item():.2f}")

model.compact(block_size=32)
model.compact_moe()

print("\n=== AFTER COMPACTION ===")
attn = model.blocks[10].attn
print(f"W_q shape: {attn.W_q.shape}")
print(f"W_q[base + 0, BD.HAS_SE] = {attn.W_q[base + 0, BD.HAS_SE].item():.2f}")
print(f"W_q[base + 33, BD.HAS_SE] = {attn.W_q[base + 33, BD.HAS_SE].item():.2f}")
print(f"W_q[base + 33, BD.H1 + 1] = {attn.W_q[base + 33, BD.H1 + 1].item():.2f}")

# Check if the weights are actually being used
print("\n=== CHECK ACTUAL USAGE ===")
print(f"Number of heads: {attn.num_heads}")
print(f"Head dim: {attn.head_dim}")
