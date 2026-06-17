#!/usr/bin/env python3
"""FREE-RUN (spec_k=0, model-authoritative) per-step token counts for a func_*
program. This is the ground truth for the post-ENT 37-token desync: unlike the
teacher-forced _probe_func_tokens.py, this lets the MODEL emit its own tokens
(byte-identical to run_batch spec_k=0) so an over-emit (extra leading 0xFF/0x00
value bytes) is directly visible as a step with ntok != 35.

Usage: python tools/_probe_freerun_tokens.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}
SE = int(Token.STEP_TOKENS)  # 35
STEP_END = int(Token.STEP_END)


def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    recs = probe.probe(bc, max_steps=ms)
    prompt_len = min(recs.keys()) if recs else 0
    # Reconstruct the emitted stream in position order.
    positions = sorted(recs.keys())
    toks = [recs[p]["token"] for p in positions]
    print(f"id{pid} {desc} exp={exp} prompt_len={prompt_len} emitted={len(toks)}")
    # Split by STEP_END.
    step = 0; cur = []
    def flush(s, ts):
        n = len(ts)
        flag = ""
        if n != SE:
            flag = f"  <== {n} (DESYNC! expected {SE})" if n > SE else f"  <== {n} (SHORT)"
        lead = [NAMES.get(t, str(t)) for t in ts[:6]]
        print(f"  step {s:2d}: ntok={n}{flag}  lead={lead}")
    for t in toks:
        cur.append(t)
        if t == STEP_END:
            flush(step, cur); step += 1; cur = []
        if t == int(Token.HALT):
            flush(step, cur); cur = []; break
    if cur:
        flush(step, cur)


if __name__ == "__main__":
    main()
