#!/usr/bin/env python3
"""Compare the AX byte-0 decode at a GOOD step vs the BAD step (final residual).

For each requested step, dump (a) the full OUTPUT_LO/OUTPUT_HI 16-cell vectors at
the AX byte-0 row, (b) the top-5 vocab logits, and (c) the top dims by |logit
contribution| to the emitted token. Reveals which residual band actually carries
the byte value and why the good step decodes 70 but the bad step decodes 72.

Usage: python tools/_probe_axb0_decode.py [id] [step_good] [step_bad]
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
    tsteps = [int(s) for s in sys.argv[2:]] or [7, 9]
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    inv = {v: k for k, v in dp.items()}
    LO = dp["OUTPUT_LO"]; HI = dp["OUTPUT_HI"]
    ctx = probe._final_context(bc, max_steps=14)
    pl = len(probe._build_context(bc))
    steps = []; cur = []
    for p in range(pl, len(ctx)):
        cur.append(p)
        if ctx[p] == SE: steps.append(cur); cur = []
    W = probe.model.head.weight
    if W.is_sparse or getattr(W, "is_sparse_csr", False): W = W.to_dense()
    W = W.detach()
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    for tstep in tsteps:
        axrow = None
        for p in steps[tstep]:
            if ctx[p] == AX: axrow = p + 1; break
        r = probe.model.forward(padded, stop_after_block=nblk - 1)[0][axrow]
        logits = W @ r
        top = torch.topk(logits, 6)
        print(f"\n=== step{tstep} AX-b0 row={axrow} emitted={ctx[axrow]} ===")
        print("  OUTPUT_LO[0..15]:", [round(r[LO + i].item(), 3) for i in range(16)])
        print("  OUTPUT_HI[0..15]:", [round(r[HI + i].item(), 3) for i in range(16)])
        print("  top logits:", [(int(t), round(v, 4)) for t, v in zip(top.indices.tolist(), top.values.tolist())])
        # per-dim contribution to emitted token logit
        tok = int(top.indices[0])
        contrib = W[tok] * r
        order = torch.argsort(contrib.abs(), descending=True)[:12]
        print(f"  top dims -> logit[{tok}]:")
        for d in order.tolist():
            if abs(contrib[d].item()) < 1e-4: continue
            print(f"    dim {d:4d} {inv.get(d, str(d)):22s} x={r[d].item():+.3f} W={W[tok, d].item():+.3f} c={contrib[d].item():+.3f}")


if __name__ == "__main__":
    main()
