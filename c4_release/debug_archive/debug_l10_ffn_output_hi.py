#!/usr/bin/env python3
"""Debug which L10 FFN units write to OUTPUT_HI[1]."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

bytecode = [Opcode.JMP | (16 << 8)]

def build_context(bytecode):
    tokens = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        tokens.append(op)
        for i in range(4): tokens.append((imm >> (i * 8)) & 0xFF)
        for _ in range(3): tokens.append(0)
    tokens.append(Token.CODE_END)
    tokens.append(Token.DATA_START)
    tokens.append(Token.DATA_END)
    return tokens

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

context = build_context(bytecode)
draft = DraftVM(bytecode)
draft.step()
tokens = draft.draft_tokens()

ctx = context + tokens[:2]
pc_byte0_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    for i in range(10):
        x = model.blocks[i](x)

    x = model.blocks[10].attn(x)

    # Get L10 FFN
    ffn = model.blocks[10].ffn
    W_down = ffn.W_down.data

    # Find units that write to OUTPUT_HI[1]
    active_units = (W_down[BD.OUTPUT_HI + 1, :].abs() > 0).nonzero().squeeze(-1)
    print(f"Units writing to OUTPUT_HI[1]: {active_units.tolist()}")

    inp = x[0, pc_byte0_pos, :]

    for unit in active_units.tolist():
        W_up_row = ffn.W_up[unit, :].data
        W_gate_row = ffn.W_gate[unit, :].data
        b_up = ffn.b_up[unit].item()
        b_gate = ffn.b_gate[unit].item()
        W_down_val = W_down[BD.OUTPUT_HI + 1, unit].item()

        up_val = (W_up_row @ inp).item() + b_up
        gate_val = (W_gate_row @ inp).item() + b_gate
        silu_val = torch.nn.functional.silu(torch.tensor(up_val)).item()
        output = silu_val * gate_val * W_down_val

        if abs(output) > 0.1:
            print(f"\n=== Unit {unit} ===")
            print(f"  up = {up_val:.4f}")
            print(f"  gate = {gate_val:.4f}")
            print(f"  silu(up) = {silu_val:.4f}")
            print(f"  W_down = {W_down_val:.4f}")
            print(f"  output = {output:.4f}")

            # Show input dim values
            up_dims = (W_up_row.abs() > 0.1).nonzero().squeeze(-1)
            gate_dims = (W_gate_row.abs() > 0.1).nonzero().squeeze(-1)
            print(f"  W_up dims: {up_dims.tolist() if up_dims.numel() > 0 else []}")
            print(f"  W_gate dims: {gate_dims.tolist() if gate_dims.numel() > 0 else []}")

            for d in up_dims.tolist()[:5]:
                print(f"    W_up[{d}]={W_up_row[d].item():.1f}, x[{d}]={inp[d].item():.4f}")
            for d in gate_dims.tolist()[:5]:
                if d not in up_dims.tolist():
                    print(f"    W_gate[{d}]={W_gate_row[d].item():.1f}, x[{d}]={inp[d].item():.4f}")
