#!/usr/bin/env python3
"""At the func LI query, list head-0's TOP causal-masked rows with full content
(clval, MEM_VAL_B0/B1/B2/B3, MEM_STORE/MSAV, ADDR_B0 nibbles, marker, token),
AND explicitly the genuine value row(s) carrying value `val`, so we can see why
the spurious same-address row out-scores the genuine store.

Usage: python tools/_probe_func_winnerrow.py <id> <li_step> [maxsteps] [val]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT); RAX = int(Token.REG_AX)


def _find_l15(model, dimp):
    opli = dimp.get("OP_LI_RELAY"); cand = []
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        wq = wq.to_dense() if (wq.is_sparse or wq.is_sparse_csr) else wq
        if opli is not None and wq.shape[1] > opli and abs(float(wq[:, opli].abs().max())) > 1000:
            cand.append((bi, blk.attn.num_heads))
    cand.sort(key=lambda t: -t[1]); return cand[0][0] if cand else None


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); li_step = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    val = int(sys.argv[4]) if len(sys.argv) > 4 else 57
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t in (STEP_END, HALT):
            steps.append((s, i)); s = i + 1
    L15 = _find_l15(model, dimp)
    st, en = steps[li_step]
    axm = next(i for i in range(st, en+1) if ctx[i] == RAX)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    S = resid.shape[0]
    blk = model.blocks[L15]; attn = blk.attn
    xin = resid
    if getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None:
        xin = blk.attn_norm(resid.unsqueeze(0))[0]
    H = attn.num_heads
    Wq = attn.W_q.to_dense() if (attn.W_q.is_sparse or attn.W_q.layout != torch.strided) else attn.W_q
    Wk = attn.W_k.to_dense() if (attn.W_k.is_sparse or attn.W_k.layout != torch.strided) else attn.W_k
    HD = Wq.shape[0] // H
    Q = (xin @ Wq.float().t()).view(S, H, HD).transpose(0, 1)
    K = (xin @ Wk.float().t()).view(S, H, HD).transpose(0, 1)
    scale = HD ** -0.5
    pos = torch.arange(S, device=resid.device).float()
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    slopes = attn.alibi_slopes.float() if getattr(attn, "alibi_slopes", None) is not None else None
    cl_lo = dimp.get("CLEAN_EMBED_LO"); cl_hi = dimp.get("CLEAN_EMBED_HI")
    a0lo = dimp["ADDR_B0_LO"]; a0hi = dimp["ADDR_B0_HI"]
    def clval(p):
        lo = int(torch.argmax(resid[p, cl_lo:cl_lo+16]).item())
        hi = int(torch.argmax(resid[p, cl_hi:cl_hi+16]).item())
        return lo | (hi << 4)
    def nz(p, base):
        sub = resid[p, base:base+16]
        return [k for k in range(16) if float(sub[k]) > 0.3]
    h = 0
    sc = (Q[h, axm] @ K[h].t()) * scale
    if slopes is not None:
        sc = sc - slopes[h] * dist[axm]
    sc_masked = sc.clone()
    sc_masked[axm+1:] = float("-inf")  # causal
    print(f"id{pid} {desc} LI step={li_step} L15={L15} AX_query={axm} emit_b0={ctx[axm+1]} slope={slopes[h] if slopes is not None else None}")
    def dump(p, tag=""):
        cv = clval(p)
        b = [round(float(resid[p, dimp[f'MEM_VAL_B{j}']]), 2) if f'MEM_VAL_B{j}' in dimp else 0 for j in range(4)]
        msv = round(float(resid[p, dimp['MEM_STORE_AT_VAL']]), 2) if 'MEM_STORE_AT_VAL' in dimp else 0
        ms2 = round(float(resid[p, dimp['MEM_STORE']]), 2) if 'MEM_STORE' in dimp else 0
        d = float(dist[axm, p])
        print(f"   {tag}pos{p:4d} sc={float(sc[p]):10.1f} dist={d:5.0f} tok={ctx[p]:3d} clval={cv:3d} B={b} MSAV={msv} MSTORE={ms2} a0lo={nz(p,a0lo)} a0hi={nz(p,a0hi)}")
    print("\n-- TOP 12 causal-masked rows by score --")
    top = torch.topk(sc_masked, 12)
    for p in top.indices.tolist():
        dump(p)
    print(f"\n-- genuine rows carrying value {val} on CLEAN (and their score) --")
    for p in range(axm+1):
        if clval(p) == val:
            dump(p, tag="VAL ")
    # softmax1 winner
    m = torch.maximum(sc_masked.max(), torch.zeros((), device=sc.device))
    ex = torch.exp(sc_masked - m); denom = ex.sum() + torch.exp(-m)
    w = ex / denom
    wtop = torch.topk(w, 3)
    print(f"\n-- softmax1 winner: pos {int(wtop.indices[0])} w={float(wtop.values[0]):.3f}, "
          f"2nd pos {int(wtop.indices[1])} w={float(wtop.values[1]):.3f} --")


if __name__ == "__main__":
    main()
