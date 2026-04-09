#!/usr/bin/env python3
"""Debug FFN units writing to OUTPUT_HI[15]."""

import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
import torch.nn.functional as F

BYTECODE = [Opcode.IMM | (1 << 8), Opcode.PSH, Opcode.IMM | (0 << 8), Opcode.MUL, Opcode.EXIT]

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

    # Run 3 steps
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    # Build context up to BP marker
    context_for_pred = context + step4_tokens[:16]
    bp_marker_pos = len(context_for_pred) - 1

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    # Run through L0-L5
    with torch.no_grad():
        for i in range(6):
            x = model.blocks[i](x)

    # Get input to L6 FFN (after L6 attention)
    block = model.blocks[6]
    x_after_attn = block.attn(x)

    ffn = block.ffn
    output_hi_15_idx = BD.OUTPUT_HI + 15

    # Find units with non-zero W_down[OUTPUT_HI+15, :]
    print(f"=== FFN units writing to OUTPUT_HI[15] (dim {output_hi_15_idx}) ===")
    w_down = ffn.W_down.data
    w_up = ffn.W_up.data
    w_gate = ffn.W_gate.data
    b_up = ffn.b_up.data
    b_gate = ffn.b_gate.data

    for u in range(w_down.shape[1]):
        weight = w_down[output_hi_15_idx, u].item()
        if abs(weight) > 1e-6:
            # Compute hidden activation for this unit
            up_val = (w_up[u] @ x_after_attn[0, bp_marker_pos]).item() + b_up[u].item()
            gate_val = (w_gate[u] @ x_after_attn[0, bp_marker_pos]).item() + b_gate[u].item()
            hidden_val = F.silu(torch.tensor(up_val)).item() * gate_val
            contrib = weight * hidden_val

            if abs(contrib) > 1e-4:
                print(f"\nUnit {u}:")
                print(f"  W_down[OUTPUT_HI+15, {u}] = {weight:.4f}")
                print(f"  up = {up_val:.4f}")
                print(f"  gate = {gate_val:.4f}")
                print(f"  hidden = silu({up_val:.2f}) * {gate_val:.4f} = {hidden_val:.4f}")
                print(f"  contrib = {weight:.4f} * {hidden_val:.4f} = {contrib:.4f}")

                # Find what W_up reads
                up_inputs = (w_up[u].abs() > 1e-6).nonzero(as_tuple=True)[0]
                if len(up_inputs) > 0:
                    print(f"  W_up reads dims: {up_inputs.tolist()[:10]}")

                # Find what W_gate reads
                gate_inputs = (w_gate[u].abs() > 1e-6).nonzero(as_tuple=True)[0]
                if len(gate_inputs) > 0:
                    print(f"  W_gate reads dims: {gate_inputs.tolist()[:10]}")

                # Print actual input values at active dims
                print(f"\n  Input values at BP marker:")
                for dim in up_inputs.tolist()[:10]:
                    val = x_after_attn[0, bp_marker_pos, dim].item()
                    weight = w_up[u, dim].item()
                    print(f"    dim {dim}: val={val:.4f}, weight={weight:.4f}, contrib={val*weight:.4f}")
                print(f"  b_up = {b_up[u].item():.4f}")

                print(f"\n  Gate input values:")
                for dim in gate_inputs.tolist()[:10]:
                    val = x_after_attn[0, bp_marker_pos, dim].item()
                    weight = w_gate[u, dim].item()
                    print(f"    dim {dim}: val={val:.4f}, weight={weight:.4f}")

if __name__ == "__main__":
    main()
