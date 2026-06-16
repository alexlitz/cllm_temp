#!/usr/bin/env python3
"""FREE-RUN full token dump for ONE step of a func_* program (spec_k=0).
Shows every emitted token + its top-2 logits so we can see WHERE a step
gains/loses a token vs the canonical 35-token layout.

Usage: python tools/_probe_freerun_step.py <id> <step> [maxsteps]
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
STEP_END = int(Token.STEP_END)
HALT = int(Token.HALT)

# canonical 35-token step layout offsets
LAYOUT = {0: "REG_PC", 5: "REG_AX", 10: "REG_SP", 15: "REG_BP",
          20: "STACK0", 25: "MEM_MARK", 34: "STEP_END"}


def main():
    pid = int(sys.argv[1]); want = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    recs = probe.probe(bc, max_steps=ms)
    positions = sorted(recs.keys())
    print(f"id{pid} {desc} exp={exp}")
    # split into steps
    step = 0; cur = []
    for p in positions:
        t = recs[p]["token"]
        cur.append((p, t, recs[p]["top_k_logits"][:2]))
        if t == STEP_END or t == HALT:
            if step == want:
                print(f"  STEP {step}: ntok={len(cur)}")
                for off, (pp, tt, tk) in enumerate(cur):
                    lay = LAYOUT.get(off, "")
                    nm = NAMES.get(tt, str(tt))
                    tkstr = " ".join(f"{a}:{b:.0f}" for a, b in tk)
                    print(f"    off{off:2d} pos{pp} tok={tt}({nm}) {lay:9s} top=[{tkstr}]")
                return
            step += 1; cur = []
    print("step not found")


if __name__ == "__main__":
    main()
