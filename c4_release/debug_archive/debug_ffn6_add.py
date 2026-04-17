#!/usr/bin/env python3
"""Debug which L6 FFN units fire at AX marker during ADD 10+32."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (10 << 8), Opcode.PSH, Opcode.IMM | (32 << 8), Opcode.ADD, Opcode.EXIT]

def build_context(bytecode, data=b''):
    context = [Token.CODE_START]
    for instr in bytecode:
        op = instr & 0xFF
        imm = instr >> 8
        context.append(op)
        for i in range(IMMEDIATE_SIZE):
            context.append((imm >> (i * 8)) & 0xFF)
        for _ in range(PADDING_SIZE):
            context.append(0)
    context.append(Token.CODE_END)
    context.append(Token.DATA_START)
    context.extend(list(data))
    context.append(Token.DATA_END)
    return context

def main():
    model = AutoregressiveVM()
    set_vm_weights(model)
    model.eval()

    context = build_context(BYTECODE)
    draft = DraftVM(BYTECODE)

    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    draft.step()
    step4_tokens = draft.draft_tokens()
    context_for_ax = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_ax) - 1

    token_ids = torch.tensor([context_for_ax], dtype=torch.long)

    print("=" * 80)
    print(f"Debugging L6 FFN (blocks[6].ffn) at AX marker (pos {ax_marker_pos}) for ADD")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(6):
            x = model.blocks[i](x)

        attn_out = model.blocks[6].attn(x)
        x_after_attn = x + attn_out

        ffn6 = model.blocks[6].ffn
        x_ax = x_after_attn[0, ax_marker_pos, :]

        print(f"\nRelevant input dimensions at AX marker:")
        dims_to_check = [
            ('MARK_AX', BD.MARK_AX),
            ('MARK_PC', BD.MARK_PC),
            ('IS_BYTE', BD.IS_BYTE),
            ('OP_ADD', BD.OP_ADD),
            ('OP_IMM', BD.OP_IMM),
            ('OP_EXIT', BD.OP_EXIT),
        ]
        for name, dim in dims_to_check:
            print(f"  {name}: {x_ax[dim].item():.4f}")

        print(f"\n" + "=" * 80)
        print(f"Top 10 FFN units with highest OUTPUT_LO contribution:")
        print("=" * 80)

        S = 100.0
        num_units = ffn6.W_up.shape[0]
        contributions = []

        for unit in range(num_units):
            output_lo_write = ffn6.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, unit].abs().max().item()
            if output_lo_write < 0.01:
                continue

            up_val = (ffn6.W_up[unit, :] @ x_ax + ffn6.b_up[unit]).item()
            gate_val = (ffn6.W_gate[unit, :] @ x_ax + ffn6.b_gate[unit]).item()

            if up_val > -50:  # Only consider potentially active units
                silu_up = up_val * torch.sigmoid(torch.tensor(up_val)).item() if up_val > -10 else 0
                sig_gate = torch.sigmoid(torch.tensor(gate_val)).item()
                activation = silu_up * sig_gate

                if activation > 0.01:
                    total_contrib = 0
                    for k in range(16):
                        w = ffn6.W_down[BD.OUTPUT_LO + k, unit].item()
                        total_contrib += abs(activation * w)
                    contributions.append((unit, up_val, gate_val, activation, total_contrib))

        contributions.sort(key=lambda x: -x[4])
        for unit, up_val, gate_val, activation, total_contrib in contributions[:10]:
            print(f"\nUnit {unit}:")
            print(f"  up={up_val:.2f}, gate={gate_val:.2f}, activation={activation:.4f}")
            print(f"  total |contribution| to OUTPUT_LO: {total_contrib:.4f}")

            # Show what OUTPUT_LO slots it writes to
            writes_to = []
            for k in range(16):
                w = ffn6.W_down[BD.OUTPUT_LO + k, unit].item()
                if abs(w) > 0.01:
                    writes_to.append((k, w))
            print(f"  writes to: {writes_to}")

if __name__ == "__main__":
    main()
