"""Debug Layer 6 head 4 attention output."""
import sys
sys.path.insert(0, '/home/alexlitz/Documents/misc/c4_release/c4_release')

for mod in list(sys.modules.keys()):
    if 'neural_vm' in mod:
        del sys.modules[mod]

import torch
torch.set_num_threads(1)
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim
from neural_vm.embedding import Opcode
from neural_vm.speculative import DraftVM

BD = _SetDim

bytecode = [Opcode.LEA | (8 << 8), Opcode.EXIT]

def build_context(bc):
    tokens = [Token.CODE_START]
    for instr in bc:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4):
            tokens.append((imm >> (i * 8)) & 0xFF)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
expected_tokens = draft.draft_tokens()

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

input_tokens = context + expected_tokens[0:6]
ax_marker_pos = len(input_tokens) - 1

# Check head 4 V/O weights
attn6 = model.blocks[6].attn
HD = attn6.head_dim
base = 4 * HD

print(f"Head 4 V/O weights (offset 40-71):")
for k in [0, 8]:
    w_v = attn6.W_v[base + 40 + k, BD.FETCH_LO + k].item()
    w_o = attn6.W_o[BD.FETCH_LO + k, base + 40 + k].item()
    print(f"  V[{base + 40 + k}, FETCH_LO[{k}]] = {w_v:.4f}")
    print(f"  O[FETCH_LO[{k}], {base + 40 + k}] = {w_o:.4f}")
print()

# Check Q weights for AX marker activation
w_q_ax = attn6.W_q[base, BD.MARK_AX].item()
w_q_se = attn6.W_q[base, BD.HAS_SE].item()
w_k_pc = attn6.W_k[base, BD.MARK_PC].item()
print(f"Head 4 Q/K weights:")
print(f"  Q[{base}, MARK_AX] = {w_q_ax:.4f}")
print(f"  Q[{base}, HAS_SE] = {w_q_se:.4f}")
print(f"  K[{base}, MARK_PC] = {w_k_pc:.4f}")
print()

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 0-5
    for i in range(6):
        x = model.blocks[i](x)

    # Compute Layer 6 head 4 attention manually
    attn_out = model.blocks[6].attn.forward(x)

    print(f"After Layer 6 attention (all heads), AX marker FETCH:")
    x_after_attn = x + attn_out
    for k in range(16):
        val = x_after_attn[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")
