#!/usr/bin/env python3
"""Deep probe of the LI step for a func_* program: (1) where does the value
70 live in the residual at the L15 input (which rows carry CLEAN_EMBED=70 /
MEM_VAL=70 / STACK0=70)?  (2) what does the L15 LI head-0 attend to at the LI
query row?  spec_k=0, hook-free, all flags read from env.

Usage: python tools/_probe_func_li_value.py <id> <li_step> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
import torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT); RAX = int(Token.REG_AX)
NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


def _find_l15(model, dimp):
    """The L15 ATTENTION block: the one whose attn has the OP_LI_RELAY W_q
    signature AND >=14 heads (the memory-lookup attention, not the FFN block)."""
    opli = dimp.get("OP_LI_RELAY")
    cand = []
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        wq = wq.to_dense() if (wq.is_sparse or wq.is_sparse_csr) else wq
        if opli is not None and wq.shape[1] > opli and abs(float(wq[:, opli].abs().max())) > 1000:
            cand.append((bi, blk.attn.num_heads))
    # pick the widest-head candidate (the LI/LC memory lookup attention)
    if not cand:
        return None
    cand.sort(key=lambda t: -t[1])
    return cand[0][0]


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); li_step = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    val = int(sys.argv[4]) if len(sys.argv) > 4 else 70
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    rev = {v: k for k, v in dimp.items()}
    ctx = probe._final_context(bc, max_steps=ms)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t in (STEP_END, HALT):
            steps.append((s, i)); s = i + 1
    L15 = _find_l15(model, dimp)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    S = resid.shape[0]
    st, en = steps[li_step]
    axm = next(i for i in range(st, en + 1) if ctx[i] == RAX)
    print(f"id{pid} {desc} exp={exp}  LI step={li_step}  L15 block={L15} "
          f"num_heads={model.blocks[L15].attn.num_heads}  AX_marker={axm}  emit_b0={ctx[axm+1]}")

    # (1) scan rows (up to LI query) for which carry the value `val` in any band
    print(f"\n--- rows carrying value {val} (0x{val:02x}) up to LI query @{axm} ---")
    cl_lo = dimp.get("CLEAN_EMBED_LO"); cl_hi = dimp.get("CLEAN_EMBED_HI")
    for p in range(axm + 1):
        tok = ctx[p]
        marks = [n.replace("MARK_", "") for n in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0")
                 if dimp.get(n) is not None and resid[p, dimp[n]].item() > 0.5]
        hits = []
        if tok == val:
            hits.append(f"TOKEN={val}")
        # CLEAN_EMBED one-hot decode
        if cl_lo is not None and cl_hi is not None:
            lo = int(torch.argmax(resid[p, cl_lo:cl_lo + 16]).item())
            hi = int(torch.argmax(resid[p, cl_hi:cl_hi + 16]).item())
            clv = lo | (hi << 4)
            lo_mag = float(resid[p, cl_lo + lo].item())
            if clv == val and lo_mag > 0.3:
                hits.append(f"CLEAN={clv}({lo_mag:.1f})")
        # MEM_VAL bands
        for n in ("MEM_VAL_B0",):
            d = dimp.get(n)
            if d is not None and abs(resid[p, d].item()) > 0.3:
                hits.append(f"{n}={resid[p,d].item():.1f}")
        # STACK0 byte0
        for n in ("STACK0_BYTE0",):
            d = dimp.get(n)
            if d is not None and abs(resid[p, d].item() - val) < 0.6:
                hits.append(f"{n}~={val}")
        mstore = dimp.get("MEM_STORE")
        ms_v = float(resid[p, mstore].item()) if mstore is not None else 0.0
        if hits:
            print(f"  pos{p} tok={tok}({NAMES.get(tok,tok)}) marks={marks} MEM_STORE={ms_v:.2f}  {hits}")

    # (2) L15 attention for the LI query row, all heads, top-5 keys.
    # Replicate the real softmax1 + ALiBi + causal math.
    blk = model.blocks[L15]; attn = blk.attn
    xin = resid
    if getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None:
        xin = blk.attn_norm(resid.unsqueeze(0))[0]
    H = attn.num_heads
    Wq = attn.W_q.to_dense() if (attn.W_q.is_sparse or attn.W_q.layout != torch.strided) else attn.W_q
    Wk = attn.W_k.to_dense() if (attn.W_k.is_sparse or attn.W_k.layout != torch.strided) else attn.W_k
    HD = Wq.shape[0] // H
    Q = (xin @ Wq.float().t()).view(S, H, HD).transpose(0, 1)  # [H,S,HD]
    K = (xin @ Wk.float().t()).view(S, H, HD).transpose(0, 1)
    scale = HD ** -0.5
    pos = torch.arange(S, device=resid.device).float()
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # [S,S]
    causal = torch.triu(torch.full((S, S), float("-inf"), device=resid.device), diagonal=1)
    slopes = attn.alibi_slopes.float() if getattr(attn, "alibi_slopes", None) is not None else None
    cl_lo = dimp.get("CLEAN_EMBED_LO"); cl_hi = dimp.get("CLEAN_EMBED_HI")
    def clval(p):
        lo = int(torch.argmax(resid[p, cl_lo:cl_lo + 16]).item())
        hi = int(torch.argmax(resid[p, cl_hi:cl_hi + 16]).item())
        return lo | (hi << 4)
    print(f"\n--- L15 attention at LI query @{axm} (all heads top-5; softmax1) ---")
    for h in range(H):
        sc = (Q[h, axm] @ K[h].t()) * scale  # [S]
        if slopes is not None:
            sc = sc - slopes[h] * dist[axm]
        sc = sc + causal[axm]
        # softmax1: add a 0-anchor sink
        m = torch.maximum(sc.max(), torch.zeros((), device=sc.device))
        ex = torch.exp(sc - m); denom = ex.sum() + torch.exp(-m)
        w = ex / denom
        top = torch.topk(w, 5)
        if top.values[0] < 0.05:
            print(f"  h{h:2d}: (sink-dominated, max w={top.values[0]:.3f})")
            continue
        parts = []
        for wv, kk in zip(top.values.tolist(), top.indices.tolist()):
            kk = int(kk); tok = ctx[kk]
            marks = "+".join(n.replace("MARK_", "") for n in ("MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_MEM","MARK_STACK0")
                             if dimp.get(n) is not None and resid[kk, dimp[n]].item() > 0.5)
            parts.append(f"K{kk}(t{tok}/{marks}/cl{clval(kk)} w{wv:.2f})")
        print(f"  h{h:2d}: {'  '.join(parts)}")


if __name__ == "__main__":
    main()
