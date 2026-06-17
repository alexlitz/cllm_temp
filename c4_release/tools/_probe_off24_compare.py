#!/usr/bin/env python3
"""Compare the off24 row (STACK0 byte-3 emission) between a CLEAN 35-token step
and the BROKEN 34-token step, reading OUTPUT_LO/HI byte-0 (dims 69/85) plus the
top dims that differ. Uses the FREE-RUN context (byte-identical to run_batch).

Usage: python tools/_probe_off24_compare.py <id> <clean_step> <broken_step> [ms]
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
STEP_END = int(Token.STEP_END); HALT = int(Token.HALT)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); cs = int(sys.argv[2]); bs = int(sys.argv[3])
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    src, exp, desc = generate_test_programs()[pid]; bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device
    dp = model.embed._dim_positions; inv = {v: k for k, v in dp.items()}
    recs = probe.probe(bc, max_steps=ms)
    positions = sorted(recs.keys()); prompt_len = positions[0]
    full = list(probe._build_context(bc)) + [recs[p]["token"] for p in positions]
    # step start positions
    starts = {}; step = 0; seg0 = prompt_len
    for p in positions:
        if step not in starts:
            starts[step] = p
        if recs[p]["token"] in (STEP_END, HALT):
            step += 1
    padded = torch.tensor([full], dtype=torch.long, device=dev)
    nb = len(model.blocks)
    def rrow(blk, pos):
        out = model.forward(padded, stop_after_block=blk)
        if out.is_sparse or getattr(out, "is_sparse_csr", False): out = out.to_dense()
        return out[0, pos]
    for label, st in (("CLEAN", cs), ("BROKEN", bs)):
        base = starts[st]
        # off24 row = base+24; its LOGITS row = base+23
        off24_logit = base + 23
        r = rrow(nb - 1, off24_logit)
        print(f"{label} step{st}: off24 logit row={off24_logit} "
              f"OUTPUT_LO[0](d69)={float(r[69]):+.2f} OUTPUT_HI[0](d85)={float(r[85]):+.2f}")
        # show the L11 (block16) value of these dims
        r16 = rrow(16, off24_logit)
        print(f"    @block16(L11): d69={float(r16[69]):+.2f} d85={float(r16[85]):+.2f}")


if __name__ == "__main__":
    main()
