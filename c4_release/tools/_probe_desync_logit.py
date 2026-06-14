#!/usr/bin/env python3
"""Find the 0xFF over-emitter: at the row right after step-1's STEP_END, what
token does the LM head predict and at what logit? Trace the top OUTPUT/dim
driving it across blocks.

Usage: python tools/_probe_desync_logit.py [id] [after_step]
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

SE = int(Token.STEP_END)


def step_spans(ctx, pl):
    spans = []; i = pl; start = pl
    while i < len(ctx):
        if ctx[i] == SE:
            spans.append((start, i, i)); start = i + 1
        i += 1
    if start < len(ctx):
        spans.append((start, len(ctx), None))
    return spans


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    after = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions; inv = {v: k for k, v in dp.items()}
    ctx = probe._final_context(bc, max_steps=20)
    pl = len(probe._build_context(bc))
    spans = step_spans(ctx, pl)
    # SE position of step `after`
    se_pos = spans[after][1]
    print(f"id{pid} {desc} step{after} STEP_END@{se_pos}; next step starts at {se_pos+1} tok={ctx[se_pos+1]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    # logits at the SE row (predicts the FIRST token of next step)
    logits = probe.model.forward(padded)[0]  # [S, vocab]
    row = logits[se_pos]
    top = torch.topk(row, 5)
    print(f"  LM logits at SE row {se_pos}: top tokens = {[(int(i), round(float(v),1)) for v,i in zip(top.values, top.indices)]}")
    # If top-1 is 255, trace which block injects the magnitude into the residual
    nblk = len(probe.model.blocks)
    print("  residual max-magnitude dim at SE row across blocks:")
    prevmag = 0.0
    for blk in range(nblk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][se_pos]
        amax = int(torch.argmax(r.abs()).item()); amag = float(r[amax])
        mark = "  <<<" if abs(amag) > 10 * (abs(prevmag) + 1) else ""
        if blk == 0 or abs(amag) > abs(prevmag) * 3 + 1 or blk == nblk-1:
            print(f"    blk{blk:2d}: maxdim={inv.get(amax,amax)}={amag:.3e}{mark}")
        prevmag = amag


if __name__ == "__main__":
    main()
