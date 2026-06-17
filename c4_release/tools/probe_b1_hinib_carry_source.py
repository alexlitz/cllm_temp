#!/usr/bin/env python3
"""Find a high-nibble carry source at the CARRIED (step1) AX byte-1 row.

Diffs two edge_literal programs with the SAME low nibble but DIFFERENT high
nibble (5561 hi=1 vs 9647 hi=2, both lo=5) over EVERY registry band at the
step1 (carried, non-IMM) byte-1 predictor row. Any band whose cells differ
between the two carries the high nibble across the step. CPU-only (faithful).

Run: CUDA_VISIBLE_DEVICES="" python tools/probe_b1_hinib_carry_source.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps, STEP_TOKENS,
)


def residual_at(fwd, bc, data, step, off):
    ot = oracle_tape_and_steps(bc, data, max_steps=4)
    prompt = build_code_prompt(bc, data)
    full = prompt + ot.draft_tokens
    resid = fwd._residual_pre_head(full)
    return resid[len(prompt) + step * STEP_TOKENS + off]


def main():
    ctx = build_gate_context(verbose=True)
    dp = ctx.dim_positions
    fwd = ctx.fwd
    # next dim boundary for each name (to slice band widths)
    names = sorted(dp.items(), key=lambda kv: kv[1])
    width = {}
    for i, (nm, pos) in enumerate(names):
        nxt = names[i + 1][1] if i + 1 < len(names) else ctx.model.d_model
        width[nm] = nxt - pos

    bcA, dA = compile_c("int main() { return 5561; }")   # lo=5 hi=1
    bcB, dB = compile_c("int main() { return 9647; }")   # lo=5 hi=2
    for step in (0, 1):
        rA = residual_at(fwd, bcA, dA, step, 6)
        rB = residual_at(fwd, bcB, dB, step, 6)
        print(f"\n=== step{step} byte-1 predictor row: bands differing hi1 vs hi2 ===",
              flush=True)
        diffs = []
        for nm, pos in dp.items():
            w = min(max(width[nm], 1), 32)
            a = rA[pos:pos + w]
            b = rB[pos:pos + w]
            if a.numel() == 0:
                continue
            d = float((a - b).abs().max())
            if d > 0.3:
                ai = int(a.argmax()); bi = int(b.argmax())
                diffs.append((d, nm, ai, float(a[ai]), bi, float(b[bi])))
        diffs.sort(reverse=True)
        for d, nm, ai, av, bi, bv in diffs[:25]:
            print(f"  {nm:22s} maxdiff={d:7.2f}  hi1: am={ai} v={av:.2f}"
                  f"  hi2: am={bi} v={bv:.2f}", flush=True)


if __name__ == "__main__":
    main()
