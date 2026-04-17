#!/usr/bin/env python3
"""Debug Layer 2 attention vs FFN effect on BYTE_INDEX_0."""
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

ctx = context + tokens[:2]  # PC marker + PC byte 0
pc_byte1_pos = len(ctx) - 1

token_ids = torch.tensor([ctx], dtype=torch.long)

with torch.no_grad():
    x = model.embed(token_ids)
    x = model.blocks[0](x)  # Layer 1

    print(f"After blocks[0] (Layer 1):")
    print(f"  BYTE_INDEX_0: {x[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_1: {x[0, pc_byte1_pos, BD.BYTE_INDEX_1].item():.4f}")

    # Manually apply Layer 2 in parts
    # NOTE: attn and ffn include residual connections
    block1 = model.blocks[1]
    x_pre = x.clone()

    # Apply attention (includes residual)
    x_after_attn = block1.attn(x_pre)
    attn_delta = x_after_attn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item() - x_pre[0, pc_byte1_pos, BD.BYTE_INDEX_0].item()

    print(f"\nAfter blocks[1].attn:")
    print(f"  BYTE_INDEX_0: {x_after_attn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_0 delta: {attn_delta:.4f}")

    # Apply FFN (includes residual)
    x_after_ffn = block1.ffn(x_after_attn)
    ffn_delta = x_after_ffn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item() - x_after_attn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item()

    print(f"\nAfter blocks[1].ffn:")
    print(f"  BYTE_INDEX_0: {x_after_ffn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
    print(f"  BYTE_INDEX_0 delta from FFN: {ffn_delta:.4f}")

    # Check post_ops
    if block1.post_ops:
        x_after_post = x_after_ffn.clone()
        for op in block1.post_ops:
            x_after_post = op(x_after_post)
        post_delta = x_after_post[0, pc_byte1_pos, BD.BYTE_INDEX_0].item() - x_after_ffn[0, pc_byte1_pos, BD.BYTE_INDEX_0].item()
        print(f"\nAfter post_ops:")
        print(f"  BYTE_INDEX_0: {x_after_post[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")
        print(f"  BYTE_INDEX_0 delta from post_ops: {post_delta:.4f}")

    # Full block for comparison
    x_full = block1(x_pre)
    print(f"\nFull blocks[1]:")
    print(f"  BYTE_INDEX_0: {x_full[0, pc_byte1_pos, BD.BYTE_INDEX_0].item():.4f}")

    # Now trace what Layer 2 FFN is doing
    # There's a unit in L2 FFN that fires: L1H4[BP] + IS_BYTE - 1.5 > 0 gated by 1-H1[BP]
    # At PC byte 1, the distance from BP marker...
    BP_I = 3

    print(f"\n=== Layer 2 FFN debug ===")
    # What's L1H4[BP] at PC byte 1?
    l1h4_bp = x_pre[0, pc_byte1_pos, BD.L1H4 + BP_I].item()
    is_byte = x_pre[0, pc_byte1_pos, BD.IS_BYTE].item()
    h1_bp = x_pre[0, pc_byte1_pos, BD.H1 + BP_I].item()
    print(f"  L1H4[BP]: {l1h4_bp:.4f}  (d<=6.5 from BP)")
    print(f"  IS_BYTE: {is_byte:.4f}")
    print(f"  H1[BP]: {h1_bp:.4f}  (d<=4.5 from BP)")
    print(f"  Unit fires if: L1H4[BP] + IS_BYTE > 1.5 AND H1[BP] < 1")
    print(f"    {l1h4_bp + is_byte:.4f} > 1.5 ? {l1h4_bp + is_byte > 1.5}")
    print(f"    gate = 1 - H1[BP] = {1 - h1_bp:.4f}")

    # Check all marker distances - find what marker is closest
    print(f"\n  All threshold heads at PC byte 1 pos ({pc_byte1_pos}):")
    MARKER_NAMES = ["PC", "AX", "SP", "BP", "MEM", "STACK0", "STEP_END"]
    for mi, mname in enumerate(MARKER_NAMES):
        l1h4 = x_pre[0, pc_byte1_pos, BD.L1H4 + mi].item()
        h1 = x_pre[0, pc_byte1_pos, BD.H1 + mi].item()
        if l1h4 > 0.1 or h1 > 0.1:
            print(f"    {mname}: L1H4={l1h4:.2f} (d<=6.5), H1={h1:.2f} (d<=4.5)")
