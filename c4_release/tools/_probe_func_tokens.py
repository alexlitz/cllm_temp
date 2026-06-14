#!/usr/bin/env python3
"""Full token-stream dump per step for a func_* program. spec_k=0, hook-free.
Shows every marker + its byte payload so we can see PSH-store vs LI-load tokens.
Usage: python tools/_probe_func_tokens.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    print(f"id{pid} {desc} exp={exp} ctxlen={len(ctx)} prompt_len={pl}")
    SE = int(Token.STEP_END)
    # group by step
    i = pl; step = 0; toks = []
    def flush(s, ts):
        # render markers + payloads
        parts = []
        j = 0
        while j < len(ts):
            t = ts[j]
            nm = NAMES.get(t, str(t))
            if nm.startswith("REG_") or t in (268, 261):
                payload = ts[j+1:j+5]
                val = sum((b & 0xFF) << (8*k) for k, b in enumerate(payload))
                lbl = {268: "STACK0", 261: "MEM"}.get(t, nm.replace("REG_", ""))
                parts.append(f"{lbl}[{','.join(str(b) for b in payload)}]={val}")
                j += 5
            else:
                parts.append(nm)
                j += 1
        print(f"  step {s:2d}: " + "  ".join(parts))
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            flush(step, toks); toks = []; step += 1; i += 1; continue
        toks.append(t); i += 1
    if toks:
        flush(step, toks)


if __name__ == "__main__":
    main()
