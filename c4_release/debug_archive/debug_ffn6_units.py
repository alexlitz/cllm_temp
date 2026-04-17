#!/usr/bin/env python3
"""Debug which L6 FFN units fire at AX marker during MUL 6*7."""
import torch
import sys
sys.path.insert(0, '.')

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, Token, _SetDim as BD
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE

BYTECODE = [Opcode.IMM | (6 << 8), Opcode.PSH, Opcode.IMM | (7 << 8), Opcode.MUL, Opcode.EXIT]

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
    context_for_pred = context + step4_tokens[:6]
    ax_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)

    print("=" * 80)
    print(f"Debugging L6 FFN (blocks[6].ffn) at AX marker (pos {ax_marker_pos})")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)

        # Run through blocks 0-5
        for i in range(6):
            x = model.blocks[i](x)

        # Run blocks[6] attention only
        attn_out = model.blocks[6].attn(x)
        x_after_attn = x + attn_out  # residual

        # Get input to FFN at AX marker position
        ffn6 = model.blocks[6].ffn
        x_ax = x_after_attn[0, ax_marker_pos, :]

        print(f"\nChecking relevant input dimensions at AX marker:")
        dims_to_check = [
            ('MARK_AX', BD.MARK_AX),
            ('MARK_PC', BD.MARK_PC),
            ('IS_BYTE', BD.IS_BYTE),
            ('HAS_SE', BD.HAS_SE),
            ('OP_MUL', BD.OP_MUL),
            ('OP_IMM', BD.OP_IMM),
            ('OP_EXIT', BD.OP_EXIT),
            ('OP_NOP', BD.OP_NOP),
            ('OP_JMP', BD.OP_JMP),
        ]
        for name, dim in dims_to_check:
            print(f"  {name}: {x_ax[dim].item():.4f}")

        # Check AX_CARRY values
        print(f"\nAX_CARRY_LO at AX marker:")
        for k in range(16):
            v = x_ax[BD.AX_CARRY_LO + k].item()
            if abs(v) > 0.01:
                print(f"  AX_CARRY_LO[{k}]: {v:.4f}")

        # Manually compute FFN activations for units that write to OUTPUT_LO
        print(f"\n" + "=" * 80)
        print(f"FFN units writing to OUTPUT_LO that fire at AX marker:")
        print("=" * 80)

        S = 100.0  # scale factor
        num_units = ffn6.W_up.shape[0]

        for unit in range(num_units):
            # Check if this unit writes to OUTPUT_LO
            output_lo_write = ffn6.W_down[BD.OUTPUT_LO:BD.OUTPUT_LO+16, unit].abs().max().item()
            if output_lo_write < 0.01:
                continue

            # Compute up projection
            up_val = (ffn6.W_up[unit, :] @ x_ax + ffn6.b_up[unit]).item()
            # Compute gate projection
            gate_val = (ffn6.W_gate[unit, :] @ x_ax + ffn6.b_gate[unit]).item()

            # SiLU on up, sigmoid on gate
            silu_up = up_val * torch.sigmoid(torch.tensor(up_val)).item() if up_val > -10 else 0
            sig_gate = torch.sigmoid(torch.tensor(gate_val)).item()

            # Combined activation
            activation = silu_up * sig_gate

            if activation > 0.01:
                # Which OUTPUT_LO does it write to?
                writes_to = []
                for k in range(16):
                    w = ffn6.W_down[BD.OUTPUT_LO + k, unit].item()
                    if abs(w) > 0.01:
                        contribution = activation * w
                        writes_to.append((k, w, contribution))

                if writes_to:
                    # Debug: what drives the up projection?
                    up_contributors = []
                    for dim in range(ffn6.W_up.shape[1]):
                        w = ffn6.W_up[unit, dim].item()
                        v = x_ax[dim].item()
                        contrib = w * v
                        if abs(contrib) > 0.1:
                            # Try to get dim name
                            dim_name = f"dim{dim}"
                            for attr in dir(BD):
                                if not attr.startswith('_') and getattr(BD, attr) == dim:
                                    dim_name = attr
                                    break
                            up_contributors.append((dim_name, w, v, contrib))

                    print(f"\nUnit {unit}:")
                    print(f"  up={up_val:.2f}, gate={gate_val:.2f}, silu_up={silu_up:.2f}, sig_gate={sig_gate:.2f}")
                    print(f"  activation = {activation:.4f}")
                    print(f"  bias_up = {ffn6.b_up[unit].item():.2f}, bias_gate = {ffn6.b_gate[unit].item():.2f}")
                    print(f"  up contributors: {up_contributors[:5]}")
                    print(f"  writes to OUTPUT_LO: {writes_to}")

        # Show final OUTPUT_LO values after FFN
        x_after_ffn = model.blocks[6](x)
        print(f"\n" + "=" * 80)
        print(f"Final OUTPUT_LO at AX marker after blocks[6]:")
        for k in range(16):
            v = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

if __name__ == "__main__":
    main()
