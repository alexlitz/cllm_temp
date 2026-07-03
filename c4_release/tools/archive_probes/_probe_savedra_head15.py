#!/usr/bin/env python3
"""Probe L15 head 15 (savedra_pc) attention at the LEV PC marker: does it attend
the JSR AX-marker row and deliver LOOKAHEAD_PC (=90)?

Usage: python tools/_probe_savedra_head15.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}

def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out

@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    pcm = sm[lev].get("PC")
    print(f"STEP_TOKENS={int(Token.STEP_TOKENS)} steps={len(sm)} lev_pc_marker={pcm}")
    L15 = [i for i, b in enumerate(model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 16][0]
    blk = model.blocks[L15]; attn = blk.attn
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    xin = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    x = xin
    if getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None:
        x = blk.attn_norm(xin.unsqueeze(0))[0]
    Wq = attn.W_q.to_dense() if (attn.W_q.is_sparse or attn.W_q.layout != torch.strided) else attn.W_q
    Wk = attn.W_k.to_dense() if (attn.W_k.is_sparse or attn.W_k.layout != torch.strided) else attn.W_k
    Wv = attn.W_v.to_dense() if (attn.W_v.is_sparse or attn.W_v.layout != torch.strided) else attn.W_v
    H = attn.num_heads; HD = Wq.shape[0] // H
    Q = (x @ Wq.float().t()).view(-1, H, HD).transpose(0, 1)
    K = (x @ Wk.float().t()).view(-1, H, HD).transpose(0, 1)
    scale = HD ** -0.5
    h = 15
    qv = Q[h, pcm]
    # ALiBi slope for this head
    slope = None
    for a in ("alibi_slopes", "alibi_slope"):
        s = getattr(attn, a, None)
        if s is not None:
            try: slope = float(s[h]) if hasattr(s, "__getitem__") else float(s)
            except Exception: slope = None
            break
    print(f"L15={L15} HD={HD} scale={scale:.4f} alibi_slope={slope}")
    # score every key row up to pcm
    rows = []
    for p in range(pcm + 1):
        sc = float((qv * K[h, p]).sum().item()) * scale
        if slope:
            sc += -slope * (pcm - p)
        opjsr = float(xin[p, dimp["OP_JSR"]].item()) if "OP_JSR" in dimp else 0
        max_ = float(xin[p, dimp["MARK_AX"]].item()) if "MARK_AX" in dimp else 0
        rows.append((sc, p, ctx[p], opjsr, max_))
    rows.sort(reverse=True)
    print(f"{'key':>4} {'tok':>4} {'score':>12} OP_JSR MARK_AX")
    for sc, p, tok, opjsr, mark in rows[:10]:
        print(f"{p:>4} {tok:>4} {sc:>12.1f} {opjsr:>5.1f} {mark:>6.1f}")
    import torch.nn.functional as Fn
    sc_all = torch.tensor([r[0] for r in rows])
    w = Fn.softmax(torch.cat([sc_all, torch.zeros(1)]), dim=0)[:-1]
    wi = int(torch.argmax(w).item())
    print(f"softmax1 winner: key={rows[wi][1]} tok={rows[wi][2]} w={w[wi]:.3f}")
    # what LOOKAHEAD_PC does the winner carry?
    def hot(p, base): return int(torch.argmax(xin[p, base:base+16]).item())
    wp = rows[wi][1]
    if "LOOKAHEAD_PC_LO" in dimp:
        lpc = hot(wp, dimp["LOOKAHEAD_PC_LO"]) | (hot(wp, dimp["LOOKAHEAD_PC_HI"])<<4)
        lv = float(xin[wp, dimp["LOOKAHEAD_PC_LO"]:dimp["LOOKAHEAD_PC_LO"]+16].max())
        print(f"winner LOOKAHEAD_PC = {lpc} (0x{lpc:02x}) max_act={lv:.2f}")

if __name__ == "__main__":
    main()
