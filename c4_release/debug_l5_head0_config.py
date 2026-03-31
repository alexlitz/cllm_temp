"""Check Layer 5 head 0 configuration."""
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
base = 0 * HD

print(f"Layer 5 head 0 configuration:")
print(f"Head dim: {HD}")
print()

# Check Q weights
print("Q weights (should query TEMP for PC+1, gated at AX marker):")
q_mark_ax = attn5.W_q[base + 32, BD.MARK_AX].item()
print(f"  W_q[{base + 32}, MARK_AX] = {q_mark_ax:.4f}")

# Check if there's an anti-leakage gate
GATE = 33
q_gate_ax = attn5.W_q[base + GATE, BD.MARK_AX].item()
q_gate_const = attn5.W_q[base + GATE, BD.CONST].item()
print(f"  W_q[{base + GATE}, MARK_AX] = {q_gate_ax:.4f} (anti-leakage)")
print(f"  W_q[{base + GATE}, CONST] = {q_gate_const:.4f} (anti-leakage)")
print()

# Check O weights
print("O weights (should write to FETCH at query position):")
for k in [0, 8]:
    w_o = attn5.W_o[BD.FETCH_LO + k, base + 32 + k].item()
    print(f"  W_o[FETCH_LO[{k}], {base + 32 + k}] = {w_o:.4f}")
