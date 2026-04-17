"""Check Layer 6 head 4 weights."""
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
base = 4 * HD

print(f"Head dimension: {HD}")
print(f"Base offset for head 4: {base}")
print()

print("W_v weights for FETCH_LO (should be identity):")
for k in [0, 8]:
    w_v = attn6.W_v[base + k, BD.FETCH_LO + k].item()
    print(f"  W_v[{base + k}, BD.FETCH_LO[{k}]] = {w_v:.4f}")
print()

print("W_o weights for FETCH_LO (should be identity):")
for k in [0, 8]:
    w_o = attn6.W_o[BD.FETCH_LO + k, base + k].item()
    print(f"  W_o[BD.FETCH_LO[{k}], {base + k}] = {w_o:.4f}")
print()

# Check if there are any unexpected weights
print("Checking for cross-talk in W_v (should all be zero except diagonal):")
for k in range(16):
    for j in range(16):
        if k != j:
            w_v = attn6.W_v[base + k, BD.FETCH_LO + j].item()
            if abs(w_v) > 0.01:
                print(f"  W_v[{base + k}, BD.FETCH_LO[{j}]] = {w_v:.4f} (cross-talk!)")

print()
print("Checking for cross-talk in W_o (should all be zero except diagonal):")
for k in range(16):
    for j in range(16):
        if k != j:
            w_o = attn6.W_o[BD.FETCH_LO + k, base + j].item()
            if abs(w_o) > 0.01:
                print(f"  W_o[BD.FETCH_LO[{k}], {base + j}] = {w_o:.4f} (cross-talk!)")
