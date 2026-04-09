#!/usr/bin/env python3
"""Debug L6 OUTPUT_HI[15] at BP marker."""

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

    print(f"=== Before L6 at BP marker (pos {bp_marker_pos}) ===")
    print(f"OUTPUT_HI[15]: {x[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    # Trace L6 attention
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

    print(f"\n=== L6 Attention output ===")
    attn_delta = attn_out[0, bp_marker_pos, BD.OUTPUT_HI + 15].item()
    print(f"OUTPUT_HI[15] delta from attention: {attn_delta:.4f}")

    # Trace L6 FFN
    x_after_attn = x + attn_out
    ffn = block.ffn
    ffn_out = ffn(x_after_attn)

    print(f"\n=== L6 FFN output ===")
    ffn_delta = ffn_out[0, bp_marker_pos, BD.OUTPUT_HI + 15].item()
    print(f"OUTPUT_HI[15] delta from FFN: {ffn_delta:.4f}")

    # Run full L6 and compare
    x_before_l6 = x.clone()
    x_after_l6 = block(x)
    full_delta = (x_after_l6 - x_before_l6)[0, bp_marker_pos, BD.OUTPUT_HI + 15].item()
    print(f"\n=== Full L6 block delta ===")
    print(f"OUTPUT_HI[15] delta from full L6: {full_delta:.4f}")
    print(f"OUTPUT_HI[15] after L6: {x_after_l6[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    # Check for post_ops
    print(f"\n=== L6 post_ops ===")
    print(f"Number of post_ops: {len(block.post_ops)}")
    for i, op in enumerate(block.post_ops):
        print(f"  post_op {i}: {type(op).__name__}")

    # Trace step by step using actual modules
    print(f"\n=== Step-by-step L6 trace ===")
    x_input = x.clone()
    print(f"Before attn: OUTPUT_HI[15] = {x_input[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    x_after_attn_module = block.attn(x_input)
    print(f"After attn module: OUTPUT_HI[15] = {x_after_attn_module[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    # Compare my manual x_after_attn with module output
    diff = (x_after_attn - x_after_attn_module).abs().max()
    print(f"Max diff between manual and module attn output: {diff.item():.6f}")

    x_after_ffn_module = block.ffn(x_after_attn_module)
    print(f"After ffn module: OUTPUT_HI[15] = {x_after_ffn_module[0, bp_marker_pos, BD.OUTPUT_HI + 15].item():.4f}")

    # Check what's different in the ffn input
    print(f"\n=== FFN input comparison at BP marker ===")
    print(f"EMBED_HI[15] in manual x_after_attn: {x_after_attn[0, bp_marker_pos, BD.EMBED_HI + 15].item():.4f}")
    print(f"EMBED_HI[15] in module attn output: {x_after_attn_module[0, bp_marker_pos, BD.EMBED_HI + 15].item():.4f}")

    # Check all EMBED_HI values in module attn output
    print(f"\n=== Module attn output EMBED_HI at BP marker ===")
    embed_hi = x_after_attn_module[0, bp_marker_pos, BD.EMBED_HI:BD.EMBED_HI+16].tolist()
    print(f"EMBED_HI[0:16]: {[f'{v:.4f}' for v in embed_hi]}")

    # The issue: L6 FFN reads EMBED_HI and writes to OUTPUT_HI
    # Check if EMBED_HI[15] > 0 somewhere in the FFN input
    print(f"\n=== EMBED_HI[15] after L3 (carry-forward) ===")
    # L3 copies EMBED_HI from BP byte 0 positions, which should all have EMBED_HI[15]=0
    # But maybe the attention after L3 is corrupting it?

    # Check what FFN is reading
    if ffn_delta > 0.1:
        print(f"\n=== L6 FFN inputs at BP marker ===")
        for dim_name in ['MARK_BP', 'MARK_SP', 'MARK_STACK0', 'IS_BYTE', 'EMBED_HI+15']:
            if dim_name == 'EMBED_HI+15':
                val = x_after_attn[0, bp_marker_pos, BD.EMBED_HI + 15].item()
            else:
                dim_idx = getattr(BD, dim_name)
                val = x_after_attn[0, bp_marker_pos, dim_idx].item()
            print(f"  {dim_name}: {val:.4f}")

        # The issue might be in the SP/BP/STACK0 identity carry
        # It reads EMBED_HI → OUTPUT_HI, and if EMBED_HI[15] is wrong...
        print(f"\n=== EMBED_HI at BP marker (should be 0 for marker token 260) ===")
        for k in range(16):
            val = x_after_attn[0, bp_marker_pos, BD.EMBED_HI + k].item()
            if abs(val) > 0.01:
                print(f"  EMBED_HI[{k}]: {val:.4f}")

if __name__ == "__main__":
    main()
