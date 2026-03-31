"""Check all Layer 5 heads that write to FETCH."""
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

print("Checking which heads write to FETCH_LO[0] and FETCH_LO[8]:")
print()

for head in range(8):
    base = head * HD
    writes_to_fetch = False

    for v_pos in range(HD):
        for fetch_idx in [0, 8]:
            w_o = attn5.W_o[BD.FETCH_LO + fetch_idx, base + v_pos].item()
            if abs(w_o) > 0.01:
                if not writes_to_fetch:
                    print(f"Head {head}:")
                    writes_to_fetch = True
                print(f"  W_o[FETCH_LO[{fetch_idx}], {base + v_pos}] = {w_o:.4f}")

    if writes_to_fetch:
        print()
