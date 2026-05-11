#!/usr/bin/env python3
"""Print raw block 3 (=L3 carry-forward post-fix) head 0 attention scores."""
import torch
from neural_vm.run_vm import AutoregressiveVMRunner
from neural_vm.embedding import Opcode
from neural_vm.vm_step import _SetDim as BD, Token


runner = AutoregressiveVMRunner(trust_neural_alu=True, pure_neural=True)
runner._func_call_handlers = {}; runner._syscall_handlers = {}
runner._memory = {}; runner._mem_history = {}; runner._mem_access_order = []
model = runner.model
model.eval()
device = next(model.parameters()).device

prog = [(Opcode.IMM, 1), (Opcode.BZ, 4), (Opcode.IMM, 7), Opcode.EXIT]
bytecode = []
for p in prog:
    if isinstance(p, tuple): op, imm = p; bytecode.append((imm<<8)|op)
    else: bytecode.append(p)

context = runner._build_context(bytecode, b"", [], b"")
sp0 = 0x10000

def append_step(ctx, pc, ax):
    ctx.append(Token.REG_PC)
    for i in range(4): ctx.append((pc >> (i*8)) & 0xFF)
    ctx.append(Token.REG_AX)
    for i in range(4): ctx.append((ax >> (i*8)) & 0xFF)
    ctx.append(Token.REG_SP)
    for i in range(4): ctx.append((sp0 >> (i*8)) & 0xFF)
    ctx.append(Token.REG_BP); ctx.extend([0]*4)
    ctx.append(Token.STACK0); ctx.extend([0]*4)
    ctx.append(Token.MEM); ctx.extend([0]*8)
    ctx.append(Token.STEP_END)

append_step(context, pc=10, ax=1)
pos = len(context)
context.append(Token.REG_PC)
token_ids = torch.tensor([context], dtype=torch.long, device=device)
BLOCK_IDX = 3

with torch.no_grad():
    x = model.embed(token_ids, active_opcode=model._active_opcode)
    for i in range(BLOCK_IDX):
        x = model.blocks[i](x)
    attn = model.blocks[BLOCK_IDX].attn
    W_q = attn.W_q if not attn.W_q.is_sparse else attn.W_q.to_dense()
    W_k = attn.W_k if not attn.W_k.is_sparse else attn.W_k.to_dense()
    HD = W_q.shape[0] // attn.num_heads

    Q_full = torch.nn.functional.linear(x, W_q)
    K_full = torch.nn.functional.linear(x, W_k)
    base = 0
    Q_h = Q_full[0, pos, base:base+HD]
    K_full_h = K_full[0, :, base:base+HD]
    scores_per_pos = (Q_h.unsqueeze(0) * K_full_h).sum(-1) / (HD ** 0.5)

    slope = attn.alibi_slopes[0].item() if attn.alibi_slopes is not None else 0.0
    distances = (torch.arange(scores_per_pos.shape[0], device=device) - pos).abs().float()
    alibi_term = -slope * distances
    causal_mask = torch.zeros_like(scores_per_pos)
    causal_mask[torch.arange(scores_per_pos.shape[0], device=device) > pos] = float('-inf')

    scores_final = scores_per_pos + alibi_term + causal_mask
    max_val = torch.max(scores_final.max(), torch.tensor(0.0, device=device))
    exp_scores = torch.exp(scores_final - max_val)
    exp_anchor = torch.exp(-max_val)
    sum_exp = exp_scores.sum() + exp_anchor
    attn_w = exp_scores / sum_exp

    print(f"alibi_slope[head0]={slope}", flush=True)
    print(f"\nScores at all interesting positions (Q at pos={pos}, target marker):", flush=True)

    for special_pos, label in [(37, "step0 PC byte 0 SRC"), (pos, "self/query")]:
        sc_q = scores_per_pos[special_pos].item()
        al = alibi_term[special_pos].item()
        w = attn_w[special_pos].item()
        print(f"  pos={special_pos} ({label}): score_raw={sc_q:+.3f} alibi={al:+.2f} score+alibi={sc_q+al:+.3f} attn_wt={w:.4f}", flush=True)

    top10 = attn_w.topk(10)
    print(f"\nTop 10 attended positions:", flush=True)
    for k in range(10):
        p = top10.indices[k].item()
        w = top10.values[k].item()
        sc = scores_per_pos[p].item()
        al = alibi_term[p].item()
        tok = context[p] if p < len(context) else "?"
        elo_vals = x[0, p, BD.EMBED_LO:BD.EMBED_LO+16]
        elo_argmax = elo_vals.argmax().item()
        elo_val = elo_vals.max().item()
        print(f"  pos={p:3d} tok={tok:3} w={w:.4f} score={sc:+.2f} alibi={al:+.2f} EMBED_LO[{elo_argmax}]={elo_val:.2f}", flush=True)

    # Now check post-L3-attn EMBED_LO at pos 71
    x_post = model.blocks[BLOCK_IDX].attn(x)
    elo_at_pos = x_post[0, pos, BD.EMBED_LO:BD.EMBED_LO+16]
    elo_argmax_at_pos = elo_at_pos.argmax().item()
    elo_val_at_pos = elo_at_pos.max().item()
    print(f"\nAfter L3 attn (block {BLOCK_IDX}) at PC marker pos {pos}: EMBED_LO[{elo_argmax_at_pos}]={elo_val_at_pos:.3f}", flush=True)
    print(f"Expected: EMBED_LO[10]=1.0 (carrying PC=10 from previous step)", flush=True)
