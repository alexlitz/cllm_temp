#!/usr/bin/env python3
"""LM-head logit attribution for the step-9 AX byte-0 token (70 vs 72).

Free-runs (spec_k=0) to the step-9 AX byte-0 row, then for that row decomposes
the LM-head logit for token 72 (wrong) and 70 (right) per residual dim:
    contrib[d] = (head.weight[72,d] - head.weight[70,d]) * x_final[d]
so we see which residual dims push 72 over 70 at the FINAL block output. Then
walks block-by-block to find which block first makes 72 win.

Usage: python tools/_probe_step9_logit_attrib.py [id] [step]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); AX = int(Token.REG_AX)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    tstep = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    right = int(sys.argv[3]) if len(sys.argv) > 3 else 70
    wrong = int(sys.argv[4]) if len(sys.argv) > 4 else 72
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    inv = {v: k for k, v in dp.items()}
    ctx = probe._final_context(bc, max_steps=14)
    pl = len(probe._build_context(bc))
    steps = []; cur = []
    for p in range(pl, len(ctx)):
        cur.append(p)
        if ctx[p] == SE: steps.append(cur); cur = []
    axrow = None
    for p in steps[tstep]:
        if ctx[p] == AX: axrow = p + 1; break
    print(f"id{pid} {desc} exp={exp&0xff:#x} step{tstep} AX-b0 row={axrow} emitted={ctx[axrow]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    W = probe.model.head.weight  # [vocab, d_model] (possibly sparse COO)
    if W.is_sparse or getattr(W, "is_sparse_csr", False):
        W = W.to_dense()
    W = W.detach()
    dW = (W[wrong] - W[right])  # contrib to (wrong - right)

    Wwrong = W[wrong]; Wright = W[right]
    nblk = len(probe.model.blocks)
    do_scan = "--scan" in sys.argv
    # block-by-block: logit margin (wrong - right) at this row
    print(f"\nblock : logit[{wrong}]  logit[{right}]  margin(w-r)")
    prev_margin = None
    final_x = None
    blk_iter = range(nblk) if do_scan else [nblk - 1]
    for blk in blk_iter:
        r = probe.model.forward(padded, stop_after_block=blk)[0][axrow]
        lw = (Wwrong @ r).item(); lr = (Wright @ r).item()
        margin = lw - lr
        flag = ""
        if prev_margin is not None and (prev_margin <= 0) != (margin <= 0):
            flag = "  <<< FLIP"
        if blk >= nblk - 6 or flag or blk % 8 == 0 or not do_scan:
            print(f"{blk:3d} : {lw:+.2e}  {lr:+.2e}  {margin:+.2e}{flag}")
        prev_margin = margin
        if blk == nblk - 1:
            final_x = r

    # per-dim attribution at the final residual
    contrib = dW * final_x
    print(f"\n== top dims pushing margin (wrong-right) at FINAL block (margin={contrib.sum().item():+.2e}) ==")
    order = torch.argsort(contrib.abs(), descending=True)[:25]
    for d in order.tolist():
        nm = inv.get(d, str(d))
        print(f"  dim {d:4d} {nm:24s} x={final_x[d].item():+.3e}  dW={dW[d].item():+.3f}  contrib={contrib[d].item():+.3e}")


if __name__ == "__main__":
    main()
