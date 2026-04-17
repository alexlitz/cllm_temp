"""Check FETCH after attention vs after FFN in Layer 5."""
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

# Teacher forcing up to AX marker
input_tokens = context + expected_tokens[0:6]

ax_marker_pos = len(input_tokens) - 1

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 0-4
    for i in range(5):
        x = model.blocks[i](x)

    print(f"Before Layer 5 attention, AX marker FETCH:")
    for k in range(16):
        val = x[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")
    print()

    # Run Layer 5 attention only
    attn_output = model.blocks[5].attn(x)

    print(f"Layer 5 attention output (no residual), AX marker FETCH:")
    for k in range(16):
        val = attn_output[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")
    print()

    x_after_attn = x + attn_output  # Residual connection

    print(f"After Layer 5 attention (with residual), AX marker FETCH:")
    for k in range(16):
        val = x_after_attn[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")
    print()

    # Run Layer 5 FFN
    x_ffn_out = model.blocks[5].ffn(x_after_attn)
    x_after_ffn = x_after_attn + x_ffn_out

    print(f"After Layer 5 FFN (full layer 5), AX marker FETCH:")
    for k in range(16):
        val = x_after_ffn[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")
