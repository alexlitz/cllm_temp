#!/usr/bin/env python3
"""Block-by-block OUTPUT_LO[6] vs OUTPUT_LO[8] at the step-9 AX byte-0 row.

The token decode for 70 (0x46) vs 72 (0x48) is governed by OUTPUT_LO[6] vs
OUTPUT_LO[8] (head.weight dW=+/-5 on those two cells only). At the final block
they are TIED (~-0.0795) so argmax tie-breaks to 8 -> token 72. This walks the
blocks to find WHERE LO[6] loses its lead / LO[8] catches up, and which FFN
units at that block write the two cells.

Usage: python tools/_probe_step9_lo_trace.py [id] [step] [lo_a] [lo_b]
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

SE = int(Token.STEP_END); AX = int(Token.REG_AX)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    tstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    na = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    inv = {v: k for k, v in dp.items()}
    LO = dp["OUTPUT_LO"]
    da, db = LO + na, LO + nb
    ctx = probe._final_context(bc, max_steps=14)
    pl = len(probe._build_context(bc))
    steps = []; cur = []
    for p in range(pl, len(ctx)):
        cur.append(p)
        if ctx[p] == SE: steps.append(cur); cur = []
    axrow = None
    for p in steps[tstep]:
        if ctx[p] == AX: axrow = p + 1; break
    print(f"id{pid} {desc} step{tstep} AX-b0 row={axrow} LO[{na}]=dim{da} LO[{nb}]=dim{db}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    prev = (None, None)
    print(f"\nblk : LO[{na}]      LO[{nb}]      (lead = LO[{na}]-LO[{nb}])")
    for blk in range(nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][axrow]
        va, vb = r[da].item(), r[db].item()
        lead = va - vb
        flag = ""
        if prev[0] is not None:
            dlead = lead - (prev[0] - prev[1])
            if abs(dlead) > 0.05:
                flag = f"  <<< dlead={dlead:+.3f}"
        print(f"{blk:3d} : {va:+.4f}   {vb:+.4f}   lead={lead:+.4f}{flag}")
        prev = (va, vb)

    # at the block(s) where lead drops most, dump the FFN units writing da/db
    print("\n== which FFN units write LO[na]/LO[nb] across blocks (constant weight cols) ==")
    for blk in range(nblk):
        ffn = probe.model.blocks[blk].ffn
        def dense(w):
            return w.to_dense() if (w.is_sparse or getattr(w, "is_sparse_csr", False)) else w
        Wd = dense(ffn.W_down)
        ua = torch.nonzero(Wd[da].abs() > 1e-6).flatten().tolist()
        ub = torch.nonzero(Wd[db].abs() > 1e-6).flatten().tolist()
        if ua or ub:
            print(f"  blk{blk}: LO[{na}]<-units {[(u, round(Wd[da,u].item(),2)) for u in ua[:6]]}  "
                  f"LO[{nb}]<-units {[(u, round(Wd[db,u].item(),2)) for u in ub[:6]]}")


if __name__ == "__main__":
    main()
