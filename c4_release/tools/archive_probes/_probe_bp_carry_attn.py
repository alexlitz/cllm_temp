#!/usr/bin/env python3
"""Inspect the BP carry head (L13 block16 head8) attention weights from the ENT
val predictor rows. spec_k=0. Usage: python ... <id> <maxsteps> <ent_step>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_rows(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], []).append(i); i += 1; continue
        i += 1
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ent_step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    head = 8; blk = 16
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    rows = step_rows(ctx, pl)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    mem_idx = rows[ent_step]["MEM"][0]
    bp_prev = rows[ent_step - 1]["BP"][0]

    # Capture the residual entering block `blk` (output of block blk-1).
    resid = probe.model.forward(padded, stop_after_block=blk - 1)[0]  # [seq, d]
    b = probe.model.blocks[blk]
    attn = b.attn
    HD = attn.W_q.shape[0] // attn.num_heads
    Wq = attn.W_q[head*HD:(head+1)*HD]   # [HD, d]
    Wk = attn.W_k[head*HD:(head+1)*HD]
    # LayerNorm if present
    x = resid
    if hasattr(b, "ln1") and b.ln1 is not None:
        x = b.ln1(x.unsqueeze(0))[0]
    elif hasattr(attn, "norm") and attn.norm is not None:
        x = attn.norm(x.unsqueeze(0))[0]
    Q = x @ Wq.T   # [seq, HD]
    K = x @ Wk.T
    scores = Q @ K.T   # [seq, seq]
    # ALiBi: positive slope -> recency. Approx by adding slope * (j - i)? We just
    # show raw scores + the top-attended positions per query row.
    for k in range(4):
        qpos = mem_idx + 4 + k
        sc = scores[qpos].clone()
        # causal mask (can only attend <= qpos)
        sc[qpos+1:] = -1e9
        top = torch.topk(sc, 6)
        print(f"val{k} pred qpos={qpos}: top K positions (raw score):")
        for s, p in zip(top.values.tolist(), top.indices.tolist()):
            lbl = ""
            if bp_prev <= p <= bp_prev+4:
                lbl = f" <-- PREV BP byte{p-bp_prev-1}" if p > bp_prev else " <-- PREV BP marker"
            print(f"    pos={p} tok={ctx[p]:3d} score={s:.1f}{lbl}")


if __name__ == "__main__":
    main()
