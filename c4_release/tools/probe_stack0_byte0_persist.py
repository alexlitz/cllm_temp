#!/usr/bin/env python3
"""Confirm the cross-step STACK0 byte-0 framing drift (Root 2).

Traces an ``if_gt``-style program and prints, per emitted VM step, the token
COUNT and the register block. A DRIFT step emits a spurious extra register
block (57 tokens instead of 35) because the STACK0 byte-0 emission one-hot is
corrupted on the carried comparison step, so a marker ([PC]) wins the argmax
at the STACK0 byte-0 predictor row and the runner re-emits the whole register
block. Value-dependent: both-nibbles-nonzero operands (0x11, 0x23) DRIFT;
one-nibble-zero (0x10, 0x20) stay CLEAN. ``--raw`` dumps the offending step's
tokens (the spurious [PC] register restart is visible).

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_byte0_persist.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "SE",
        268: "STACK0", 261: "MEM"}


def _tok(t):
    nm = MARK.get(t)
    return f"[{nm}]" if nm else f"{t:02x}"


def trace(probe, src, label, raw=False, max_steps=12):
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=max_steps)
    pl = len(probe._build_context(bc))
    out, ec = probe.emitted_result(bc, max_steps=max_steps)
    # split into steps at STEP_END
    steps = []
    cur = []
    i = pl
    while i < len(ctx):
        cur.append(ctx[i])
        if ctx[i] == int(Token.STEP_END):
            steps.append(cur)
            cur = []
        i += 1
    if cur:
        steps.append(cur)
    ntoks = [len(s) for s in steps]
    bad = [si for si, n in enumerate(ntoks) if n != 35]
    print(f"\n===== {label} :: {src!r} =====")
    print(f"  exit={ec}  ntoks={ntoks}  non35_steps={bad}")
    if raw:
        for si in bad:
            print(f"  step {si} ({ntoks[si]} tok): "
                  + " ".join(_tok(t) for t in steps[si]))


def main():
    raw = "--raw" in sys.argv
    probe = build_groundtruth_probe()
    cases = [
        ("CLEAN 16>32", "int main() { if (16 > 32) return 1; return 0; }"),
        ("DRIFT 17>35", "int main() { if (17 > 35) return 1; return 0; }"),
        ("DRIFT 35>43", "int main() { if (35 > 43) return 1; return 0; }"),
    ]
    for label, src in cases:
        trace(probe, src, label, raw=raw)


if __name__ == "__main__":
    main()
