#!/usr/bin/env python3
"""Attribute the if_var GT-FALSE AX byte-2 leak to residual dims + owning block.

The if_var GT-FALSE cluster (426/430/431/432 ...) fails full_trace because at
the BZ (branch-if-zero) step AND the following IMM (return-0) step the model
emits AX byte-2 = 0x01 / 0x05 instead of the required 0x00 (probed
``tools/probe_ifbool_step1_vcorr.py``). This poisons the autoregressive frame so
the return value decodes as 0x0001xxxx instead of 0x00000000.

This probe does the exact LM-head logit attribution (``logit[want=0x00] -
logit[got]``) at the AX byte-2 predictor row on those steps, so the wrong dims
the head reads to emit byte-2 are named. Single teacher-forced forward
(spec_k=0), no autoregressive decode — fast, and the per-step emission bug is
what the interp gate's flat decode already sees.

Run::

    CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 python tools/probe_ifvar_axbyte2_attrib.py \
        --id 426 --steps 11,12
"""
from __future__ import annotations
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
_ROOT = os.path.dirname(_PKG)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from c4_release.src.compiler import compile_c  # noqa: E402
from c4_release.tools import probe_lib as P  # noqa: E402
from c4_release.tests.test_suite_1000 import generate_test_programs  # noqa: E402


def _parse_ints(spec):
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--id", type=int, default=426)
    ap.add_argument("--steps", type=str, default="11,12",
                    help="VM steps (0-based) to attribute the AX byte-2 leak at")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--byte", type=int, default=2,
                    help="AX byte index to attribute (default 2)")
    args = ap.parse_args()

    tests = generate_test_programs()
    src, expected, desc = tests[args.id]
    bc, data = compile_c(src)
    print(f"id={args.id} {desc}  expect={expected}")
    print(f"src: {src}")

    probe = P.build_probe()
    namer = P.DimNamer(probe.model)
    trace = probe.probe(bc, max_steps=args.max_steps)
    ax_rows = P.register_marker_rows(trace, "REG_AX")
    print(f"AX marker rows (per step): {ax_rows[:16]}")

    steps = _parse_ints(args.steps)
    bk = args.byte
    for step in steps:
        if step >= len(ax_rows):
            print(f"  step {step}: no AX marker row (only {len(ax_rows)} steps)")
            continue
        pos = ax_rows[step] + bk  # byte-k predictor row
        want = 0x00
        # The emitted byte-k token: the trace records the argmax token at pos.
        got = trace.get(pos, {}).get("token")
        if got is None or got == want:
            got = 0x01  # fall back to the observed leak token
        attr = P.logit_attrib(probe, bc, pos, want=want, got=got,
                              max_steps=args.max_steps)
        print(f"\n=== step {step} AX byte-{bk} predictor pos {pos}  "
              f"want=0x{want:02x} got=0x{got:02x} ===")
        print(f"   logit[want]={attr.logit_want:.3f} logit[got]={attr.logit_got:.3f}"
              f"  diff(want-got)={attr.diff:.3f}")
        print("   top dims driving (logit_want - logit_got):")
        for di in attr.top(20):
            print(f"      dim {di:4d} {namer.name_for(di):28s} "
                  f"res={float(attr.residual[di]):9.3f} "
                  f"contrib={float(attr.contrib[di]):9.3f}")


if __name__ == "__main__":
    main()
