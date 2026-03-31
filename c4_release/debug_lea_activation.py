"""Debug LEA unit activations."""
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

with torch.no_grad():
    x = model.embed(torch.tensor([input_tokens], dtype=torch.long))
    pos = len(input_tokens) - 1

    # Run through Layer 5
    for i in range(6):
        x = model.blocks[i](x)

    print("Before Layer 6:")
    print(f"  OP_LEA: {x[0, pos, BD.OP_LEA].item():.4f}")
    print(f"  MARK_AX: {x[0, pos, BD.MARK_AX].item():.4f}")
    print(f"  FETCH_LO[0]: {x[0, pos, BD.FETCH_LO + 0].item():.4f}")
    print(f"  FETCH_LO[8]: {x[0, pos, BD.FETCH_LO + 8].item():.4f}")
    print(f"  AX_CARRY_LO[0]: {x[0, pos, BD.AX_CARRY_LO + 0].item():.4f}")
    print(f"  AX_CARRY_LO[8]: {x[0, pos, BD.AX_CARRY_LO + 8].item():.4f}")
    print()

    # Manually compute unit 850 activation (FETCH_LO[0] -> AX_CARRY_LO[0])
    ffn6 = model.blocks[6].ffn
    unit = 850

    w_up_lea = ffn6.W_up[unit, BD.OP_LEA].item()
    w_up_ax = ffn6.W_up[unit, BD.MARK_AX].item()
    b_up = ffn6.b_up[unit].item()

    op_lea_val = x[0, pos, BD.OP_LEA].item()
    mark_ax_val = x[0, pos, BD.MARK_AX].item()

    up_input = op_lea_val * w_up_lea + mark_ax_val * w_up_ax + b_up
    up_activation = torch.nn.functional.silu(torch.tensor(up_input)).item()

    print(f"Unit 850 (FETCH_LO[0] -> AX_CARRY_LO[0]):")
    print(f"  up_input = {op_lea_val:.4f}*{w_up_lea:.1f} + {mark_ax_val:.4f}*{w_up_ax:.1f} + {b_up:.1f}")
    print(f"  up_input = {up_input:.4f}")
    print(f"  silu(up_input) = {up_activation:.4f}")

    w_gate_fetch0 = ffn6.W_gate[unit, BD.FETCH_LO + 0].item()
    b_gate = ffn6.b_gate[unit].item()
    fetch0_val = x[0, pos, BD.FETCH_LO + 0].item()

    gate_input = fetch0_val * w_gate_fetch0 + b_gate
    gate_activation = torch.sigmoid(torch.tensor(gate_input)).item()

    print(f"  gate_input = {fetch0_val:.4f}*{w_gate_fetch0:.1f} + {b_gate:.1f}")
    print(f"  gate_input = {gate_input:.4f}")
    print(f"  sigmoid(gate_input) = {gate_activation:.4f}")

    output = up_activation * gate_activation
    print(f"  output = {up_activation:.4f} * {gate_activation:.4f} = {output:.4f}")

    w_down = ffn6.W_down[BD.AX_CARRY_LO + 0, unit].item()
    contribution = output * w_down
    print(f"  contribution to AX_CARRY_LO[0] = {output:.4f} * {w_down:.4f} = {contribution:.4f}")
    print()

    # Manually compute unit 858 activation (FETCH_LO[8] -> AX_CARRY_LO[8])
    unit = 858

    w_up_lea = ffn6.W_up[unit, BD.OP_LEA].item()
    w_up_ax = ffn6.W_up[unit, BD.MARK_AX].item()
    b_up = ffn6.b_up[unit].item()

    up_input = op_lea_val * w_up_lea + mark_ax_val * w_up_ax + b_up
    up_activation = torch.nn.functional.silu(torch.tensor(up_input)).item()

    print(f"Unit 858 (FETCH_LO[8] -> AX_CARRY_LO[8]):")
    print(f"  up_input = {op_lea_val:.4f}*{w_up_lea:.1f} + {mark_ax_val:.4f}*{w_up_ax:.1f} + {b_up:.1f}")
    print(f"  up_input = {up_input:.4f}")
    print(f"  silu(up_input) = {up_activation:.4f}")

    w_gate_fetch8 = ffn6.W_gate[unit, BD.FETCH_LO + 8].item()
    b_gate = ffn6.b_gate[unit].item()
    fetch8_val = x[0, pos, BD.FETCH_LO + 8].item()

    gate_input = fetch8_val * w_gate_fetch8 + b_gate
    gate_activation = torch.sigmoid(torch.tensor(gate_input)).item()

    print(f"  gate_input = {fetch8_val:.4f}*{w_gate_fetch8:.1f} + {b_gate:.1f}")
    print(f"  gate_input = {gate_input:.4f}")
    print(f"  sigmoid(gate_input) = {gate_activation:.4f}")

    output = up_activation * gate_activation
    print(f"  output = {up_activation:.4f} * {gate_activation:.4f} = {output:.4f}")

    w_down = ffn6.W_down[BD.AX_CARRY_LO + 8, unit].item()
    contribution = output * w_down
    print(f"  contribution to AX_CARRY_LO[8] = {output:.4f} * {w_down:.4f} = {contribution:.4f}")
