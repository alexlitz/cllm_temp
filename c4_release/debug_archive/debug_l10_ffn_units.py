#!/usr/bin/env python3
"""Debug which L10 FFN units fire to produce OUTPUT_LO[1] = 15.0 for ADD."""
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
    print(f"Debug L10 FFN units writing to OUTPUT_LO[1] for ADD")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(10):
            x = model.blocks[i](x)

        # Run attention first
        x = model.blocks[10].attn(x)

        ffn = model.blocks[10].ffn
        x_ax = x[0, ax_marker_pos, :]

        # Check relevant input dimensions
        print(f"\nInput dimensions at AX marker:")
        print(f"  MARK_AX: {x_ax[BD.MARK_AX].item():.4f}")
        print(f"  OP_ADD: {x_ax[BD.OP_ADD].item():.4f}")
        print(f"  OP_NE: {x_ax[BD.OP_NE].item():.4f}")
        print(f"  OP_GT: {x_ax[BD.OP_GT].item():.4f}")
        print(f"  OP_GE: {x_ax[BD.OP_GE].item():.4f}")

        S = 100.0
        num_units = ffn.W_up.shape[0]
        contributions = []

        for unit in range(num_units):
            # Check if this unit writes to OUTPUT_LO[1]
            w_out = ffn.W_down[BD.OUTPUT_LO + 1, unit].item()
            if abs(w_out) < 0.001:
                continue

            # Compute activation
            up_val = (ffn.W_up[unit, :] @ x_ax + ffn.b_up[unit]).item()
            gate_val = (ffn.W_gate[unit, :] @ x_ax + ffn.b_gate[unit]).item()

            if up_val > -30:  # Consider potentially active
                silu_up = up_val * torch.sigmoid(torch.tensor(up_val)).item() if up_val > -10 else 0
                sig_gate = torch.sigmoid(torch.tensor(gate_val)).item()
                activation = silu_up * sig_gate
                contrib = activation * w_out

                if abs(contrib) > 0.01:
                    contributions.append((unit, up_val, gate_val, activation, w_out, contrib))

        print(f"\n{len(contributions)} units contribute to OUTPUT_LO[1]:")
        contributions.sort(key=lambda x: -abs(x[5]))

        total = 0
        for unit, up_val, gate_val, activation, w_out, contrib in contributions[:20]:
            print(f"\nUnit {unit}:")
            print(f"  up={up_val:.2f}, gate={gate_val:.2f}, activation={activation:.4f}")
            print(f"  W_down[OUTPUT_LO[1]] = {w_out:.4f}")
            print(f"  contribution = {contrib:.4f}")
            total += contrib

            # Show what dimensions drive the up projection
            up_contributors = []
            for dim in range(ffn.W_up.shape[1]):
                w = ffn.W_up[unit, dim].item()
                v = x_ax[dim].item()
                prod = w * v
                if abs(prod) > 10:
                    dim_name = f"dim{dim}"
                    for attr in dir(BD):
                        if not attr.startswith('_') and getattr(BD, attr) == dim:
                            dim_name = attr
                            break
                    up_contributors.append((dim_name, w, v, prod))
            if up_contributors:
                print(f"  up drivers: {up_contributors[:4]}")

        print(f"\nTotal from top 20 units: {total:.4f}")

if __name__ == "__main__":
    main()
