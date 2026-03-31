"""Check Layer 5 head 4 weights."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim

BD = _SetDim

model = AutoregressiveVM()
set_vm_weights(model)

attn5 = model.blocks[5].attn
HD = attn5.head_dim
base = 4 * HD

print(f"Layer 5 head 4 weights:")
print(f"Head dim: {HD}")
print(f"Base: {base}")
print()

print("Q weights:")
q_ax = attn5.W_q[base, BD.MARK_AX].item()
q_se = attn5.W_q[base, BD.HAS_SE].item()
print(f"  W_q[{base}, MARK_AX] = {q_ax:.4f}")
print(f"  W_q[{base}, HAS_SE] = {q_se:.4f}")
print()

print("K weights:")
k_pc = attn5.W_k[base, BD.MARK_PC].item()
print(f"  W_k[{base}, MARK_PC] = {k_pc:.4f}")
print()

print("V weights (should copy FETCH):")
for k in [0, 8]:
    w_v = attn5.W_v[base + k, BD.FETCH_LO + k].item()
    print(f"  W_v[{base + k}, FETCH_LO[{k}]] = {w_v:.4f}")
print()

print("O weights (should write to FETCH at AX marker):")
for k in [0, 8]:
    w_o = attn5.W_o[BD.FETCH_LO + k, base + k].item()
    print(f"  W_o[FETCH_LO[{k}], {base + k}] = {w_o:.4f}")
