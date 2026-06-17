#!/usr/bin/env python3
"""FULL teacher-forcing test for the multi-local step-6 LEA.

Builds a fully ORACLE-CORRECT context (prompt + DraftVM canonical 35-token
frames for every step) and reads what the model emits for each step's AX/PC by
argmax at the marker rows. If step-6 AX (the second-local LEA = BP-16 = 0xffe0)
is emitted CORRECTLY under full teacher-forcing but WRONG in free-run, the root
is cross-step framing drift. If wrong even under teacher-forcing, it's an
intrinsic per-step LEA-offset decode bug.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/probe_var_tf_full.py 275
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import contextlib
import io
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from neural_vm.batched_pure_neural import Token
from neural_vm.speculative import DraftVM

STEP = int(Token.STEP_TOKENS)


def build_tf_context(p, bc, n_steps):
    """Prompt + DraftVM canonical frames for steps 0..n_steps-1."""
    ctx = p._build_context(bc)
    plen = len(ctx)
    dv = DraftVM(list(bc))
    frames = []
    for k in range(n_steps):
        dv.step()
        fr = dv.draft_tokens()
        frames.append([int(t) for t in fr])
        ctx.extend(int(t) for t in fr)
    return ctx, plen, frames


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 275
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)

    from tools.probe_groundtruth import build_groundtruth_probe
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_groundtruth_probe()
    dev = p._device

    n_steps = 10
    ctx, plen, frames = build_tf_context(p, bc, n_steps)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    logits = p.model.forward(padded)[0]

    print(f"\n==== FULL TF id {idx}  {desc[:40]}  exp={exp} ====")
    ROLES = [("PC", 0), ("AX", 5), ("SP", 10), ("BP", 15), ("STACK0", 20)]
    for k in range(n_steps):
        base = plen + k * STEP
        def pred4(marker_off):
            return [int(logits[base + marker_off + j].argmax().item()) & 0xFF
                    for j in range(4)]
        def tf4(marker_off):
            return [frames[k][marker_off + 1 + j] & 0xFF for j in range(4)]
        def v(bs):
            return sum(b << (8 * j) for j, b in enumerate(bs))
        parts = []
        for name, off in ROLES:
            pr, tf = pred4(off), tf4(off)
            ok = "OK" if pr == tf else "XX"
            parts.append(f"{name} {v(pr):#06x}/{v(tf):#06x} {ok}")
        print(f" step{k:2d}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
