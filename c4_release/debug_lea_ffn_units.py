"""Check Layer 6 FFN units for LEA."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

# Force module reload
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

print("Layer 6 FFN shape:")
print(f"  W_up: {ffn6.W_up.shape}")
print(f"  W_gate: {ffn6.W_gate.shape}")
print(f"  W_down: {ffn6.W_down.shape}")
print()

# Check units 850-900 (should include cleanup + override)
print("Checking units 850-900 for LEA:")
for unit in range(850, 900):
    # Check if this unit writes to AX_CARRY_LO[0]
    if abs(ffn6.W_down[BD.AX_CARRY_LO + 0, unit].item()) > 0.1:
        print(f"\nUnit {unit} writes to AX_CARRY_LO[0]:")
        print(f"  W_down[AX_CARRY_LO[0], {unit}] = {ffn6.W_down[BD.AX_CARRY_LO + 0, unit].item():.4f}")
        print(f"  W_up[{unit}, OP_LEA] = {ffn6.W_up[unit, BD.OP_LEA].item():.4f}")
        print(f"  W_up[{unit}, MARK_AX] = {ffn6.W_up[unit, BD.MARK_AX].item():.4f}")
        print(f"  b_up[{unit}] = {ffn6.b_up[unit].item():.4f}")
        print(f"  b_gate[{unit}] = {ffn6.b_gate[unit].item():.4f}")
        if abs(ffn6.W_gate[unit, BD.FETCH_LO + 0].item()) > 0.1:
            print(f"  W_gate[{unit}, FETCH_LO[0]] = {ffn6.W_gate[unit, BD.FETCH_LO + 0].item():.4f}")
        if abs(ffn6.W_gate[unit, BD.AX_CARRY_LO + 0].item()) > 0.1:
            print(f"  W_gate[{unit}, AX_CARRY_LO[0]] = {ffn6.W_gate[unit, BD.AX_CARRY_LO + 0].item():.4f}")
