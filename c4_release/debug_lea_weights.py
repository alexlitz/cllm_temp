"""Check LEA unit gate weights."""
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

print("Checking LEA AX_CARRY_LO override units (850-865):")
print()

for unit in range(850, 866):
    k = unit - 850
    w_down = ffn6.W_down[BD.AX_CARRY_LO + k, unit].item()
    b_up = ffn6.b_up[unit].item()
    w_up_lea = ffn6.W_up[unit, BD.OP_LEA].item()
    w_up_ax = ffn6.W_up[unit, BD.MARK_AX].item()
    w_gate_fetch = ffn6.W_gate[unit, BD.FETCH_LO + k].item()
    b_gate = ffn6.b_gate[unit].item()

    print(f"Unit {unit} (FETCH_LO[{k}] -> AX_CARRY_LO[{k}]):")
    print(f"  W_up[{unit}, OP_LEA] = {w_up_lea:.4f}")
    print(f"  W_up[{unit}, MARK_AX] = {w_up_ax:.4f}")
    print(f"  b_up[{unit}] = {b_up:.4f}")
    print(f"  W_gate[{unit}, FETCH_LO[{k}]] = {w_gate_fetch:.4f}")
    print(f"  b_gate[{unit}] = {b_gate:.4f}")
    print(f"  W_down[AX_CARRY_LO[{k}], {unit}] = {w_down:.4f}")
    print()
