"""Check which LEA units are actually firing."""
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

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))

    # Run through Layer 5
    for i in range(6):
        x = model.blocks[i](x)

    # Run Layer 6 attention
    x_attn = model.blocks[6].attn(x)
    x_post_attn = x + x_attn

    # Check what the FFN sees
    print("Input to Layer 6 FFN at AX marker:")
    print(f"OP_LEA: {x_post_attn[0, ax_marker_pos, BD.OP_LEA].item():.4f}")
    print(f"MARK_AX: {x_post_attn[0, ax_marker_pos, BD.MARK_AX].item():.4f}")
    print("\nFETCH_LO:")
    for k in range(16):
        val = x_post_attn[0, ax_marker_pos, BD.FETCH_LO + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_LO[{k}] = {val:.4f}")

    print("\nFETCH_HI:")
    for k in range(16):
        val = x_post_attn[0, ax_marker_pos, BD.FETCH_HI + k].item()
        if abs(val) > 0.01:
            print(f"  FETCH_HI[{k}] = {val:.4f}")

    # Manually compute LEA unit activations (units 850-881)
    S = 16.0
    ffn = model.blocks[6].ffn
    print("\n\nLEA unit activations (FETCH_LO units 850-865):")
    for k in range(16):
        unit = 850 + k
        # W_up activation
        w_up_act = (S * x_post_attn[0, ax_marker_pos, BD.OP_LEA].item() +
                    S * x_post_attn[0, ax_marker_pos, BD.MARK_AX].item() +
                    ffn.b_up[unit].item())
        # W_gate activation
        w_gate_act = (100.0 * x_post_attn[0, ax_marker_pos, BD.FETCH_LO + k].item() +
                      ffn.b_gate[unit].item())
        # SwiGLU output
        silu_val = w_up_act / (1.0 + torch.exp(-w_up_act).item())
        sigmoid_val = 1.0 / (1.0 + torch.exp(-w_gate_act).item())
        output = silu_val * sigmoid_val
        if abs(output) > 0.01 or k == 8:
            print(f"  Unit {unit} (k={k}): W_up={w_up_act:.2f}, W_gate={w_gate_act:.2f}, "
                  f"silu={silu_val:.2f}, sigmoid={sigmoid_val:.4f}, output={output:.2f}")

    print("\n\nLEA unit activations (FETCH_HI units 866-881):")
    for k in range(16):
        unit = 866 + k
        w_up_act = (S * x_post_attn[0, ax_marker_pos, BD.OP_LEA].item() +
                    S * x_post_attn[0, ax_marker_pos, BD.MARK_AX].item() +
                    ffn.b_up[unit].item())
        w_gate_act = (100.0 * x_post_attn[0, ax_marker_pos, BD.FETCH_HI + k].item() +
                      ffn.b_gate[unit].item())
        silu_val = w_up_act / (1.0 + torch.exp(-w_up_act).item())
        sigmoid_val = 1.0 / (1.0 + torch.exp(-w_gate_act).item())
        output = silu_val * sigmoid_val
        if abs(output) > 0.01 or k == 0:
            print(f"  Unit {unit} (k={k}): W_up={w_up_act:.2f}, W_gate={w_gate_act:.2f}, "
                  f"silu={silu_val:.2f}, sigmoid={sigmoid_val:.4f}, output={output:.2f}")
