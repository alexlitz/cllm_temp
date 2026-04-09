#\!/usr/bin/env python3
"""Debug L15 attention for STACK0 byte 0."""

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
    for step in range(3):
        draft.step()
        context.extend(draft.draft_tokens())

    step_start = len(context) - 35
    target_token = 21
    context_for_pred = context[:step_start + target_token]
    query_pos = len(context_for_pred) - 1

    print(f"Query position: {query_pos} (STACK0 marker)")

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    
    # Run through layers 0-14 to get input to L15
    x = model.embed(token_ids)
    with torch.no_grad():
        for i in range(15):
            x = model.blocks[i](x)

    print(f"\n=== Input to L15 at STACK0 marker (pos {query_pos}) ===")
    print(f"OUTPUT_LO[0]: {x[0, query_pos, BD.OUTPUT_LO].item():.4f}")
    print(f"OUTPUT_LO[1]: {x[0, query_pos, BD.OUTPUT_LO+1].item():.4f}")
    print(f"MARK_STACK0: {x[0, query_pos, BD.MARK_STACK0].item():.4f}")

    # Now trace L15 attention
    attn = model.blocks[15].attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    with torch.no_grad():
        Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
        V = F.linear(x, attn.W_v).view(B, S, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale

        # Causal mask
        causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask

        attn_weights = softmax1(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out_flat = out.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = F.linear(out_flat, attn.W_o)

    print(f"\n=== L15 Head 0 analysis (STACK0 lookup) ===")
    h = 0
    weights = attn_weights[0, h, query_pos, :]
    top_weights, top_positions = weights.topk(5)
    print(f"Top 5 attended positions: {top_positions.tolist()}")
    print(f"Attention weights: {[f'{w:.4f}' for w in top_weights.tolist()]}")

    # Check what values are at the top positions
    print(f"\nV values at top positions (slots 32-35 = CLEAN_EMBED_LO[0:4]):")
    for pos in top_positions.tolist()[:3]:
        v_slots = V[0, h, pos, 32:36].tolist()
        tok = context_for_pred[pos]
        print(f"  Pos {pos} (token {tok}): V[32:36] = {[f'{v:.3f}' for v in v_slots]}")

    print(f"\n=== L15 attention contribution to OUTPUT_LO ===")
    print(f"OUTPUT_LO[0] delta: {attn_out[0, query_pos, BD.OUTPUT_LO].item():.4f}")
    print(f"OUTPUT_LO[1] delta: {attn_out[0, query_pos, BD.OUTPUT_LO+1].item():.4f}")

    # Check out_flat for head 0 slots 32 (CLEAN_EMBED_LO[0])
    h0_out = out_flat[0, query_pos, 0:HD]
    print(f"\nHead 0 output slots 32-35: {h0_out[32:36].tolist()}")

    # Find MEM sections in context
    print(f"\n=== MEM sections in context ===")
    for i, tok in enumerate(context_for_pred):
        if tok == Token.MEM:
            mem_tokens = context_for_pred[i:i+9]
            addr = mem_tokens[1] | (mem_tokens[2] << 8) | (mem_tokens[3] << 16) | (mem_tokens[4] << 24)
            val = mem_tokens[5] | (mem_tokens[6] << 8) | (mem_tokens[7] << 16) | (mem_tokens[8] << 24)
            mem_store = x[0, i, BD.MEM_STORE].item()
            print(f"  Position {i}: MEM addr={addr:#x}, val={val:#x}, MEM_STORE={mem_store:.2f}")

if __name__ == "__main__":
    main()
