#!/usr/bin/env python3
"""Find which rows the L14 attention (block 17) writes nonzero OUTPUT, and
which head/source. Compares ENT step vs PSH step. Computes the per-head O
contribution to OUTPUT_LO/HI at every row in the step span.

Usage: python tools/_probe_l14_fire.py [id] [step]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch, torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 268: "STACK0", 261: "MEM"}
BLK = 17


def step_spans(ctx, pl):
    spans = []; i = pl; start = pl
    while i < len(ctx):
        if ctx[i] == SE:
            spans.append((start, i)); start = i + 1
        i += 1
    if start < len(ctx):
        spans.append((start, len(ctx)))
    return spans


def dense(w):
    if w.is_sparse or (hasattr(w, "layout") and "sparse" in str(w.layout)):
        w = w.to_dense()
    return w.float().contiguous()


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=20)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    a, b = spans[step]
    mem_pos = next((p for p in range(a, b) if ctx[p] == 261), None)

    block = probe.model.blocks[BLK]
    attn = block.attn
    captured = {}
    h = attn.register_forward_pre_hook(lambda m, inp: captured.__setitem__("x", inp[0].detach()))
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    _ = probe.model.forward(padded, stop_after_block=BLK)
    h.remove()
    x = captured["x"][0].float()
    S, D = x.shape
    H = attn.num_heads; HD = attn.head_dim
    Wq, Wk, Wv, Wo = dense(attn.W_q), dense(attn.W_k), dense(attn.W_v), dense(attn.W_o)
    Q = F.linear(x, Wq).view(S, H, HD)
    K = F.linear(x, Wk).view(S, H, HD)
    V = F.linear(x, Wv).view(S, H, HD)
    slopes = attn.alibi_slopes
    pos_ids = torch.arange(S, device=x.device).float()
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]; CE_LO = dp["CLEAN_EMBED_LO"]; CE_HI = dp["CLEAN_EMBED_HI"]

    def dist_for(pos, hh):
        q = Q[pos, hh]
        scores = (K[:, hh, :] @ q) * attn.scale
        if slopes is not None:
            scores = scores - slopes[hh] * (pos_ids[pos] - pos_ids).abs()
        scores = scores.clone(); scores[pos+1:] = float("-inf")
        m = scores.max(); ex = torch.exp(scores - m)
        return ex / (ex.sum() + torch.exp(-m))

    def dbyte(row, lo, hi):
        ln = int(torch.argmax(row[lo:lo+16]).item()); hn = int(torch.argmax(row[hi:hi+16]).item())
        return (hn << 4) | ln, round(float(row[lo+ln]),2), round(float(row[hi+hn]),2)

    print(f"id{pid} {desc} step={step} span[{a}:{b}] MEM@{mem_pos} addr={ctx[mem_pos+1:mem_pos+5]} val={ctx[mem_pos+5:mem_pos+9]}")
    print("Per-row: which value head (4-7) writes nonzero OUTPUT and from where")
    for pos in range(a, b):
        for hh in range(4, 8):
            w = dist_for(pos, hh)
            wv = (w.unsqueeze(1) * V[:, hh, :]).sum(0)
            head_out = Wo[:, hh*HD:(hh+1)*HD] @ wv
            ob, lov, hiv = dbyte(head_out, OL, OH)
            mag = max(abs(lov), abs(hiv))
            if mag > 0.3:
                topw, topi = w.topk(3)
                tops = [(int(j), round(float(wj),2), REGS.get(ctx[int(j)], f"t{ctx[int(j)]}")) for wj,j in zip(topw,topi) if wj>0.05]
                srcbytes = []
                for j,_,_ in tops:
                    bce,_,_ = dbyte(x[j], CE_LO, CE_HI); srcbytes.append((j,bce))
                print(f"  row{pos}({REGS.get(ctx[pos],'t'+str(ctx[pos]))}) h{hh}: O-byte={ob} (mag={mag:.1f}) top={tops} src_CE={srcbytes}")


if __name__ == "__main__":
    main()
