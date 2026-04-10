#!/usr/bin/env python3
"""Trace raw FFN delta (before MARK_AX gate) for ADD 10+32."""
import torch
import torch.nn.functional as F
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
    print(f"Tracing raw FFN delta for ADD at AX marker")
    print("=" * 80)

    with torch.no_grad():
        x = model.embed(token_ids)
        for i in range(6):
            x = model.blocks[i](x)

        # Run blocks[6] attention (with residual)
        attn = model.blocks[6].attn
        attn_out = attn(x)  # This already includes residual
        x_after_attn = attn_out

        # Manually compute FFN delta (before MARK_AX gate)
        ffn = model.blocks[6].ffn
        up = F.linear(x_after_attn, ffn.W_up, ffn.b_up)
        gate = F.linear(x_after_attn, ffn.W_gate, ffn.b_gate)
        hidden = F.silu(up) * gate
        delta = F.linear(hidden, ffn.W_down)

        print(f"\nAt AX marker (pos {ax_marker_pos}):")
        print(f"MARK_AX = {x_after_attn[0, ax_marker_pos, BD.MARK_AX].item():.4f}")

        print(f"\nRaw delta (before gate) at OUTPUT_LO:")
        for k in range(16):
            v = delta[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # Apply MARK_AX gate
        mark_ax_gate = x_after_attn[..., BD.MARK_AX:BD.MARK_AX+1]
        delta_gated = delta.clone()
        delta_gated[..., BD.OUTPUT_LO:BD.OUTPUT_LO+16] *= mark_ax_gate
        delta_gated[..., BD.OUTPUT_HI:BD.OUTPUT_HI+16] *= mark_ax_gate

        print(f"\nGated delta (after × MARK_AX) at OUTPUT_LO:")
        for k in range(16):
            v = delta_gated[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # Final output
        x_after_ffn = x_after_attn + delta_gated
        print(f"\nFinal OUTPUT_LO after blocks[6]:")
        for k in range(16):
            v = x_after_ffn[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

        # But what does the actual FFN output look like?
        print(f"\n--- Using actual FFN forward ---")
        actual_output = model.blocks[6].ffn(x_after_attn)
        print(f"OUTPUT_LO after actual FFN forward:")
        for k in range(16):
            v = actual_output[0, ax_marker_pos, BD.OUTPUT_LO + k].item()
            if abs(v) > 0.01:
                print(f"  OUTPUT_LO[{k}]: {v:.4f}")

if __name__ == "__main__":
    main()
