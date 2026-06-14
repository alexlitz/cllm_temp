#!/usr/bin/env python3
"""FAST: precompute final context ONCE, then one forward per requested block
(stop_after_block), reading the residual at chosen rows. spec_k=0, hook-free.
Usage: python tools/_probe_li_resid2.py <id> <maxsteps> <li_step>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)


def step_markers(ctx, pl):
    out = []
    i = pl; cur = {}
    REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
            int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur[REGS[t]] = i; i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]); li = int(sys.argv[3])
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    print(f"id{pid} {desc} exp={exp} ctxlen={len(ctx)} steps={len(sm)} li_step={li}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)

    bands = {}
    for b in range(3):
        for n in ("LO","HI"):
            k=f"ADDR_B{b}_{n}";  bands[k]=dp.get(k)
    for k in ("MEM_STORE","MEM_VAL_B0","MEM_VAL_B1","OUTPUT_LO","OUTPUT_HI","ALU_LO","AX_FULL_LO","OP_LI","OP_LI_RELAY","MARK_AX","CLEAN_EMBED_LO"):
        bands[k]=dp.get(k)
    bands={k:v for k,v in bands.items() if v is not None}

    li_ax = sm[li].get("AX")
    li_mem = sm[li].get("MEM")
    # the PSH store row (step 3 for identity) — find MEM rows in earlier steps
    store_rows = [(s, sm[s]["MEM"]) for s in range(len(sm)) if "MEM" in sm[s]]
    nblk = len(probe.model.blocks)
    rows_of_interest = {f"LI(s{li}).AX": li_ax}
    if li_mem is not None: rows_of_interest[f"LI(s{li}).MEM"] = li_mem

    # Dump only a curated block set to stay fast.
    blocks = [13, 17, 18, 26, 27, 28, 30, 32, 34, 37, 47]
    blocks = [b for b in blocks if b < nblk]
    print("\n== LI-step AX row residual across blocks (value bands) ==")
    vb = ["OUTPUT_LO","OUTPUT_HI","ALU_LO","AX_FULL_LO","MEM_VAL_B0","MEM_STORE"]
    vb = [k for k in vb if k in bands]
    print("blk " + " ".join(f"{k:>10}" for k in vb))
    for blk in blocks:
        resid = probe.model.forward(padded, stop_after_block=blk)[0]
        row = resid[li_ax]
        print(f"{blk:3d} " + " ".join(f"{float(row[bands[k]]):10.2f}" for k in vb))

    # Also: at the LI AX row after block 13 (post mem-addr gather), dump the
    # ADDR query bits decoded to an address to confirm LI queries 65512.
    resid13 = probe.model.forward(padded, stop_after_block=min(17,nblk-1))[0]
    r = resid13[li_ax]
    print("\nLI-step AX-row ADDR query bands (post-block17):")
    for b in range(3):
        for n in ("LO","HI"):
            k=f"ADDR_B{b}_{n}"
            if k in bands:
                vals=[round(float(r[bands[k]+j]),1) for j in range(16)]
                print(f"  {k}: {vals}")


if __name__ == "__main__":
    main()
