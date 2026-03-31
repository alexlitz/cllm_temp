"""Check Layer 6 head 6 weights."""
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

attn6 = model.blocks[6].attn

HD = attn6.head_dim
base = 6 * HD

print(f"Head dimension: {HD}")
print(f"Base offset for head 6: {base}")
print()

print("W_v weights for FETCH_LO:")
for k in [0, 8]:
    w_v = attn6.W_v[base + k, BD.FETCH_LO + k].item()
    print(f"  W_v[{base + k}, BD.FETCH_LO[{k}]] = {w_v:.4f}")
print()

print("W_o weights for FETCH_LO:")
for k in [0, 8]:
    w_o = attn6.W_o[BD.FETCH_LO + k, base + k].item()
    print(f"  W_o[BD.FETCH_LO[{k}], {base + k}] = {w_o:.4f}")
print()

print("W_q weights:")
q_ax = attn6.W_q[base, BD.MARK_AX].item()
q_se = attn6.W_q[base, BD.HAS_SE].item()
print(f"  W_q[{base}, BD.MARK_AX] = {q_ax:.4f}")
print(f"  W_q[{base}, BD.HAS_SE] = {q_se:.4f}")
print()

print("W_k weights:")
k_pc = attn6.W_k[base, BD.MARK_PC].item()
print(f"  W_k[{base}, BD.MARK_PC] = {k_pc:.4f}")
