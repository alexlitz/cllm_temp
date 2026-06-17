#!/usr/bin/env python3
"""Diagnose the STACK0-block SHORT-step desync (34-token step 3 in func_*).

The PSH-arg store step emits only 3 STACK0 value bytes then the MEM marker
(261), one byte early, so the step is 34 tokens and the fixed-35 slicer
desyncs. This probe builds the FREE-RUN context (byte-identical to run_batch
spec_k=0), locates the off24 row of the diverging step (the row that wrongly
predicts MEM-marker 261 instead of STACK0 byte-3 value 0), and reports the
LM-head logit attribution: which residual dims push token 261 up and token 0
down at that row, traced across blocks.

Usage: python tools/_probe_stack0_short.py <id> [maxsteps]
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

NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}
STEP_END = int(Token.STEP_END)
HALT = int(Token.HALT)
MEM_TOK = int(Token.MEM)  # 261


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device
    dp = model.embed._dim_positions; inv = {v: k for k, v in dp.items()}

    # Build the FREE-RUN context.
    recs = probe.probe(bc, max_steps=ms)
    positions = sorted(recs.keys())
    prompt_len = positions[0]
    toks = [recs[p]["token"] for p in positions]
    full = list(probe._build_context(bc)) + toks
    # Find the SHORT step (ntok != 35) and its off24 row.
    step = 0; start = prompt_len; i = prompt_len
    short_off24 = None; short_step = None
    cur_start = prompt_len
    counts = []
    j = prompt_len
    seg = []
    for p in positions:
        seg.append(p)
        t = recs[p]["token"]
        if t == STEP_END or t == HALT:
            n = len(seg)
            counts.append((step, n, seg[0]))
            if n == 34 and short_off24 is None:
                short_step = step
                short_off24 = seg[0] + 24   # off24 absolute position
            step += 1; seg = []
    print(f"id{pid} {desc} exp={exp} prompt_len={prompt_len}")
    print(f"  step token counts: {[(s, n) for s, n, _ in counts]}")
    if short_off24 is None:
        print("  no 34-token step found"); return
    print(f"  SHORT step={short_step}, off24 abs pos={short_off24} tok={full[short_off24]}({NAMES.get(full[short_off24])})")

    # LM-head logit attribution at the off24 LOGITS row (= off24-1, predicts off24 token).
    logit_row = short_off24 - 1
    padded = torch.tensor([full], dtype=torch.long, device=dev)
    logits = model.forward(padded)[0]
    row = logits[logit_row]
    tk = torch.topk(row, 6)
    pairs = [(int(t), NAMES.get(int(t), str(int(t))), round(float(v), 1))
             for v, t in zip(tk.values.tolist(), tk.indices.tolist())]
    print(f"  logits row {logit_row} predicts: top6={pairs}")
    print(f"    logit[0 (STACK0 val)]={float(row[0]):.1f}  logit[261 (MEM)]={float(row[MEM_TOK]):.1f}")

    # Which dims drive token-0 (STACK0 val) so negative? LM-head: logit0 = sum_d r[d]*W[0,d].
    W = model.head.weight  # [V, D]
    if W.is_sparse or getattr(W, "is_sparse_csr", False):
        W = W.to_dense()
    w0 = W[0].to_dense() if (W[0].is_sparse or getattr(W[0], "is_sparse_csr", False)) else W[0]
    nblocks = len(model.blocks)
    # Final residual (pre-head). stop_after_block reads up to the given block.
    def resid_row(blk):
        out = model.forward(padded, stop_after_block=blk)
        if out.is_sparse or getattr(out, "is_sparse_csr", False):
            out = out.to_dense()
        return out[0, logit_row]
    r_final = resid_row(nblocks - 1)  # [D]
    contrib0 = r_final * w0  # per-dim contribution to logit0
    order = torch.argsort(contrib0.abs(), descending=True)[:18]
    print("  FINAL residual @ off24 logit row — dims driving logit[token 0] (STACK0 val):")
    for d in order.tolist():
        nm = inv.get(d, f"dim{d}")
        print(f"    dim{d:4d} {nm:26s} resid={float(r_final[d]):+10.2f} "
              f"W[0,d]={float(w0[d]):+8.3f} contrib0={float(contrib0[d]):+10.2f}")

    # Trace the dominant suppressor dim across blocks.
    dom = int(order[0].item())
    print(f"  block-by-block residual of dom dim {dom} ({inv.get(dom, dom)}):")
    prev = 0.0
    for blk in range(nblocks):
        v = float(resid_row(blk)[dom])
        mark = "  <<<" if abs(v - prev) > 50 else ""
        print(f"    blk{blk:2d}: {v:+12.2f}  (delta {v - prev:+10.2f}){mark}")
        prev = v


if __name__ == "__main__":
    main()
