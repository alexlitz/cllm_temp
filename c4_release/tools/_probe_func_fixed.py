#!/usr/bin/env python3
"""Per-step register trace using FIXED 35-token offsets (matches runner decode).
spec_k=0, hook-free. Usage: python tools/_probe_func_fixed.py <id> [maxsteps]
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

STEP = int(Token.STEP_TOKENS)  # 35
FIELDS = [("PC", 0), ("AX", 5), ("SP", 10), ("BP", 15), ("STACK0", 20)]


def val4(ctx, base):
    return sum((ctx[base + 1 + j] & 0xFF) << (8 * j) for j in range(4)) if base + 4 < len(ctx) else None


def main():
    ids = [int(x) for x in sys.argv[1:2]] or [550]
    pid = ids[0]
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    body = ctx[pl:]
    print(f"id{pid} {desc} exp={exp} body_tokens={len(body)} steps~{len(body)//STEP}")
    nsteps = len(body) // STEP
    for s in range(nsteps):
        b = s * STEP
        parts = []
        for nm, off in FIELDS:
            v = val4(body, b + off)
            parts.append(f"{nm}={v}")
        # MEM addr (26..29 -> base 25), val (30..33)
        addr = sum((body[b+26+j] & 0xFF) << (8*j) for j in range(4)) if b+29 < len(body) else None
        mval = sum((body[b+30+j] & 0xFF) << (8*j) for j in range(4)) if b+33 < len(body) else None
        parts.append(f"MEMa={addr}")
        parts.append(f"MEMv={mval}")
        # marker sanity: which token is at each marker offset
        markers = {0: body[b+0], 5: body[b+5], 10: body[b+10], 15: body[b+15], 20: body[b+20], 25: body[b+25], 34: body[b+34] if b+34 < len(body) else None}
        ok = (markers[0]==int(Token.REG_PC) and markers[5]==int(Token.REG_AX) and markers[20]==268 and markers[25]==261 and markers[34]==int(Token.STEP_END))
        print(f"  step {s:2d}: " + "  ".join(parts) + ("" if ok else f"   <DESYNC markers={markers}>"))


if __name__ == "__main__":
    main()
