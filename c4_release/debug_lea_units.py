"""Check if LEA cleanup units exist."""
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

ffn6 = model.blocks[6].ffn

print("Checking units 850-913 for LEA:")
print()

# Check units that write to AX_CARRY_LO[0]
for unit in range(850, 914):
    w_down = ffn6.W_down[BD.AX_CARRY_LO + 0, unit].item()
    if abs(w_down) > 0.01:
        b_up = ffn6.b_up[unit].item()
        w_up_lea = ffn6.W_up[unit, BD.OP_LEA].item()
        w_up_ax = ffn6.W_up[unit, BD.MARK_AX].item()
        b_gate = ffn6.b_gate[unit].item()

        print(f"Unit {unit}:")
        print(f"  W_down[AX_CARRY_LO[0], {unit}] = {w_down:.4f}")
        print(f"  b_up[{unit}] = {b_up:.4f}")
        print(f"  W_up[{unit}, OP_LEA] = {w_up_lea:.4f}")
        print(f"  W_up[{unit}, MARK_AX] = {w_up_ax:.4f}")
        print(f"  b_gate[{unit}] = {b_gate:.4f}")
        print()
