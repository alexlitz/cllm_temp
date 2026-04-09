#!/usr/bin/env python3
"""Debug L6 output at BP marker position."""

import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_vm.vm_step import AutoregressiveVM, set_vm_weights, _SetDim as BD, Token
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode
from neural_vm.constants import IMMEDIATE_SIZE, PADDING_SIZE
from neural_vm.kv_cache_eviction import softmax1

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

    # Run 3 steps to get to step 4
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    # Step 4
    draft.step()
    step4_tokens = draft.draft_tokens()

    # Build context up to BP marker
    target_token = 16  # BP byte 0
    context_for_pred = context + step4_tokens[:target_token]
    bp_marker_pos = len(context_for_pred) - 1

    print(f"=== Debugging L6 at BP marker (pos {bp_marker_pos}) ===")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)

    # Run through L0-L5
    with torch.no_grad():
        for i in range(6):
            x = model.blocks[i](x)

    print(f"\n=== Input to L6 at BP marker ===")
    print(f"OUTPUT_LO[0]: {x[0, bp_marker_pos, BD.OUTPUT_LO].item():.4f}")
    print(f"MARK_BP: {x[0, bp_marker_pos, BD.MARK_BP].item():.4f}")
    print(f"MARK_SP: {x[0, bp_marker_pos, BD.MARK_SP].item():.4f}")
    print(f"MARK_AX: {x[0, bp_marker_pos, BD.MARK_AX].item():.4f}")
    print(f"HAS_SE: {x[0, bp_marker_pos, BD.HAS_SE].item():.4f}")

    # Now trace L6 attention
    block = model.blocks[6]
    attn = block.attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    with torch.no_grad():
        Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
        V = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
        causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask
        attn_weights = softmax1(scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        out_flat = out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = F.linear(out_flat, attn.W_o)

    print(f"\n=== L6 Attention output at BP marker ===")
    print(f"OUTPUT_LO[0] delta from attention: {attn_out[0, bp_marker_pos, BD.OUTPUT_LO].item():.4f}")

    # Check each head
    for h in range(H):
        h_out_start = h * HD
        h_out_end = (h + 1) * HD
        h_out = out_flat[0, bp_marker_pos, h_out_start:h_out_end]

        # Check if this head is contributing to OUTPUT_LO
        # W_o[OUTPUT_LO, h*HD:(h+1)*HD] maps head h output to OUTPUT_LO
        w_o_slice = attn.W_o[BD.OUTPUT_LO, h_out_start:h_out_end]
        contrib = (h_out * w_o_slice).sum().item()

        if abs(contrib) > 0.01:
            print(f"  Head {h}: OUTPUT_LO contrib = {contrib:.4f}")

            # Where is this head attending?
            h_weights = attn_weights[0, h, bp_marker_pos, :]
            top_weights, top_pos = h_weights.topk(5)
            print(f"    Top attended: {[(p.item(), f'{w.item():.4f}') for p, w in zip(top_pos, top_weights)]}")

    # Check L6 FFN
    x_after_attn = x + attn_out
    ffn = block.ffn

    # FFN uses SwiGLU: h1 = W1a*x, gate = W1b*x, output = W2*(h1 * sigmoid(gate))
    # Or simpler: output = ffn(x_after_attn)
    ffn_out = ffn(x_after_attn)

    print(f"\n=== L6 FFN output at BP marker ===")
    ffn_delta = ffn_out[0, bp_marker_pos, BD.OUTPUT_LO].item()
    print(f"OUTPUT_LO[0] delta from FFN: {ffn_delta:.4f}")

    # Check total L6 output
    x_after_l6 = x + attn_out + ffn_out
    print(f"OUTPUT_LO[0] after L6: {x_after_l6[0, bp_marker_pos, BD.OUTPUT_LO].item():.4f}")

    # Check FFN internal activations
    print(f"\n=== L6 FFN internal at BP marker ===")
    h1 = F.linear(x_after_attn, ffn.W1)
    gate = torch.sigmoid(F.linear(x_after_attn, ffn.W1b) if hasattr(ffn, 'W1b') else h1)
    h1_gated = h1 * gate if hasattr(ffn, 'W1b') else F.relu(h1)

    # Check W1 input dimensions that matter
    print(f"\n=== L6 FFN input dims at BP marker ===")
    for dim_name in ['MARK_BP', 'MARK_SP', 'MARK_AX', 'HAS_SE', 'CONST', 'OP_PSH']:
        dim_idx = getattr(BD, dim_name, None)
        if dim_idx is not None:
            val = x_after_attn[0, bp_marker_pos, dim_idx].item()
            if abs(val) > 0.01:
                print(f"  {dim_name}: {val:.4f}")

    # Find which hidden units contribute to OUTPUT_LO
    print(f"\n=== FFN hidden units contributing to OUTPUT_LO[0] ===")
    h1_active = (h1_gated[0, bp_marker_pos].abs() > 0.01).nonzero().flatten()
    print(f"  Active count: {len(h1_active)}")
    for hu in h1_active.tolist()[:10]:
        h1_val = h1_gated[0, bp_marker_pos, hu].item()
        w2_contrib = (ffn.W2[BD.OUTPUT_LO, hu] * h1_val).item()
        if abs(w2_contrib) > 0.01:
            print(f"  HU {hu}: h1_gated={h1_val:.2f}, contrib to OUTPUT_LO[0]={w2_contrib:.2f}")

if __name__ == "__main__":
    main()
