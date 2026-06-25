#!/usr/bin/env python3
"""Decompose L15 head-0's pre-softmax score at the func LI query row.

For the func_add LI step, dump head-0's RAW score (Q@K/sqrt + ALiBi) at:
  - the genuine value row (271 for func_add)
  - the spurious ADDR-byte cluster (292-295)
  - the AX query/self row
and the key residual dims that the CAM keys on (MARK_AX, ADDR_B0_LO/HI nibbles,
MEM_VAL_B0/B1, MEM_STORE_AT_VAL). Also dump the QUERY row's residual at the
gate dims (MARK_AX, ADDR_B0_LO/HI, OP_LI/OP_LI_RELAY) so we see whether the
query even carries the CAM/lift activation.

Usage: python tools/_probe_func_h0_score.py <id> <li_step> [maxsteps] [val]
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
NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


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
    axm = next(i for i in range(st, en + 1) if ctx[i] == RAX)
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

    print(f"id{pid} {desc}  LI step={li_step} L15={L15} HD={HD} AX_query={axm} emit_b0={ctx[axm+1]}")
    # Query row residual at gate dims
    gate_dims = ["MARK_AX", "MARK_STACK0", "OP_LI", "OP_LI_RELAY",
                 "ADDR_B0_LO", "ADDR_B0_HI", "CONST"]
    print("\n-- QUERY row", axm, "residual at gate dims --")
    for n in gate_dims:
        d = dimp.get(n)
        if d is None: continue
        if n in ("ADDR_B0_LO", "ADDR_B0_HI"):
            sub = resid[axm, d:d+16]
            nz = [(k, round(float(sub[k]), 2)) for k in range(16) if abs(float(sub[k])) > 0.1]
            print(f"   {n}: nz_nibbles={nz}")
        else:
            print(f"   {n}: {float(resid[axm, d]):.3f}")

    # Head-0 score at candidate rows
    h = 0
    sc = (Q[h, axm] @ K[h].t()) * scale
    if slopes is not None:
        sc = sc - slopes[h] * dist[axm]
    print(f"\n-- head-0 raw score (Q@K/sqrt + ALiBi) at candidate rows (sink anchor=0) --")
    # which rows carry value `val` on CLEAN, and value rows
    cl_lo = dimp.get("CLEAN_EMBED_LO"); cl_hi = dimp.get("CLEAN_EMBED_HI")
    mb0 = dimp.get("MEM_VAL_B0"); mb1 = dimp.get("MEM_VAL_B1")
    msav = dimp.get("MEM_STORE_AT_VAL"); mstore = dimp.get("MEM_STORE")
    def clval(p):
        lo = int(torch.argmax(resid[p, cl_lo:cl_lo+16]).item())
        hi = int(torch.argmax(resid[p, cl_hi:cl_hi+16]).item())
        return lo | (hi << 4)
    rows = list(range(st - 140, axm + 1))
    interesting = []
    for p in rows:
        if p < 0 or p > axm: continue
        cv = clval(p)
        b0 = float(resid[p, mb0]) if mb0 is not None else 0.0
        b1 = float(resid[p, mb1]) if mb1 is not None else 0.0
        msv = float(resid[p, msav]) if msav is not None else 0.0
        ms2 = float(resid[p, mstore]) if mstore is not None else 0.0
        a0lo = resid[p, dimp["ADDR_B0_LO"]:dimp["ADDR_B0_LO"]+16] if "ADDR_B0_LO" in dimp else None
        a0hi = resid[p, dimp["ADDR_B0_HI"]:dimp["ADDR_B0_HI"]+16] if "ADDR_B0_HI" in dimp else None
        lo_nz = [k for k in range(16) if a0lo is not None and float(a0lo[k]) > 0.3]
        hi_nz = [k for k in range(16) if a0hi is not None and float(a0hi[k]) > 0.3]
        # keep rows with clval==val, or MEM_VAL_B1/B0 nonzero, or near top score
        keep = (cv == val) or abs(b1) > 0.3 or abs(b0) > 0.3 or abs(msv) > 0.3
        if keep:
            interesting.append((float(sc[p]), p, cv, b0, b1, msv, ms2, lo_nz, hi_nz))
    # sort by score desc
    interesting.sort(key=lambda t: -t[0])
    print("   score    pos  clval  B0    B1    MSAV  MSTORE  a0lo_nz  a0hi_nz")
    for scv, p, cv, b0, b1, msv, ms2, lo_nz, hi_nz in interesting[:25]:
        print(f"   {scv:8.1f}  {p:4d}  {cv:5d}  {b0:4.1f}  {b1:4.1f}  {msv:4.1f}  {ms2:5.1f}   {lo_nz}  {hi_nz}")
    print(f"\n   top raw score overall (any row): {float(sc.max()):.1f} at pos {int(sc.argmax())}")
    print(f"   sink anchor = 0.0 -> head fires iff some row score > 0")


if __name__ == "__main__":
    main()
