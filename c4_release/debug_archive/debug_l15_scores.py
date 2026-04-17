#\!/usr/bin/env python3
"""Debug L15 attention scores for STACK0 byte 0."""

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

    token_ids = torch.tensor([context_for_pred], dtype=torch.long)
    x = model.embed(token_ids)
    with torch.no_grad():
        for i in range(15):
            x = model.blocks[i](x)

    attn = model.blocks[15].attn
    B, S, D = x.shape
    H = attn.num_heads
    HD = attn.head_dim

    with torch.no_grad():
        Q = F.linear(x, attn.W_q).view(B, S, H, HD).transpose(1, 2)
        K = F.linear(x, attn.W_k).view(B, S, H, HD).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale

    print(f"=== L15 Head 0 raw scores at STACK0 marker ===")
    h = 0
    h_scores = scores[0, h, query_pos, :]
    
    # Key positions in Step 2 MEM section (addr=0xfff8, val=1)
    mem_start = 104
    print(f"\nStep 2 MEM section positions:")
    for i, name in enumerate(["MEM", "a0", "a1", "a2", "a3", "v0", "v1", "v2", "v3"]):
        pos = mem_start + i
        score = h_scores[pos].item()
        tok = context_for_pred[pos]
        print(f"  {name} (pos {pos}, tok={tok}): score={score:.2f}")

    # Check Q vector at query position for key dims
    print(f"\n=== Q vector at query pos {query_pos} (STACK0 marker) ===")
    q = Q[0, h, query_pos, :]
    print(f"Dim 0 (bias): Q[0] = {q[0].item():.2f}")
    print(f"Dim 1 (store): Q[1] = {q[1].item():.2f}")
    print(f"Dim 2 (ZFOD): Q[2] = {q[2].item():.2f}")
    print(f"Dim 3 (byte): Q[3] = {q[3].item():.2f}")
    print(f"Dims 4-27 (addr): Q[4:8] = {q[4:8].tolist()}")

    # Check K vector at val byte 0 (pos 109) vs val byte 3 (pos 112)
    print(f"\n=== K vectors at val byte positions ===")
    for pos in [109, 112]:
        k = K[0, h, pos, :]
        print(f"Position {pos} (token {context_for_pred[pos]}):")
        print(f"  K[0] (bias): {k[0].item():.2f}")
        print(f"  K[1] (store): {k[1].item():.2f}")
        print(f"  K[2] (ZFOD): {k[2].item():.2f}")
        print(f"  K[3] (byte): {k[3].item():.2f}")
        print(f"  K[4:8] (addr): {k[4:8].tolist()}")

    # Check MEM_STORE at various positions
    print(f"\n=== MEM_STORE at key positions ===")
    for pos in [104, 109, 112]:
        ms = x[0, pos, BD.MEM_STORE].item()
        print(f"  Position {pos}: MEM_STORE = {ms:.2f}")

if __name__ == "__main__":
    main()
