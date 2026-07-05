#!/usr/bin/env python3
"""Probe the JSR step-0 AX-passthrough leak (task #350).

At the entry JSR-to-main (bytecode idx 0, executed at step 0) the model runs
``_function_call_jsr_ax_passthrough_rules`` UNCONDITIONALLY: it copies
AX_CARRY_LO/HI -> OUTPUT_LO/HI at the AX marker. On step 0 there is no genuine
AX write, so a polluted AX_CARRY index is copied onto OUTPUT and corrupts the
step-0 AX register (the gcd/rec cluster wall).

Reads the AX marker row residual on the JSR step at:
  * block 7  (physical L5, BEFORE the function_call FFN) -> the AX_CARRY input
  * block 8  (physical L6, AFTER the JSR AX passthrough) -> the leaked OUTPUT

spec_k=0, hook-free, CPU-friendly. Dims resolved from the BUILT layout.

Run: python tools/probe_jsr_ax_leak.py [--ids 900,725,750]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from tools import probe_lib as P  # noqa: E402


FC_BLOCK = 8      # physical block hosting the L6 function_call FFN (AX passthrough)
PRE_BLOCK = 7     # physical L5, before the function_call FFN


def _pos(model, name):
    dp = P.dim_positions(model)
    v = dp.get(name)
    return None if v is None else int(v)


def _nib_decode(res, lo_n, hi_n):
    lo = [res.get(f"{lo_n}+{k}", 0.0) for k in range(16)]
    hi = [res.get(f"{hi_n}+{k}", 0.0) for k in range(16)]
    li = max(range(16), key=lambda k: lo[k]) if max(lo) > 0.3 else None
    hz = max(range(16), key=lambda k: hi[k]) if max(hi) > 0.3 else None
    val = (li | (hz << 4)) if (li is not None and hz is not None) else None
    return li, hz, val, max(lo), max(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="900,725,750")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    probe = P.build_probe()
    model = probe.model
    progs = generate_test_programs()

    PAIRS = (
        ("AX_CARRY_LO", "AX_CARRY_HI"),
        ("OUTPUT_LO", "OUTPUT_HI"),
    )
    SINGLE = ("MARK_AX", "OP_JSR", "OP_ENT", "TEMP", "IS_BYTE", "MARK_PC")

    dim_names = {}
    for lo_n, hi_n in PAIRS:
        for nm in (lo_n, hi_n):
            base = _pos(model, nm)
            if base is None:
                continue
            for k in range(16):
                dim_names[f"{nm}+{k}"] = base + k
    for nm in SINGLE:
        base = _pos(model, nm)
        if base is not None:
            dim_names[nm] = base

    for pid in ids:
        src, exp, desc = progs[pid]
        bc, _data = compile_c(src)
        # locate the AX marker on step 0 via the marker rows.
        trace = probe.probe(bc, max_steps=2)
        markers = P.register_marker_rows(trace, "REG_AX")
        if not markers:
            print(f"id {pid} ({desc}): no AX marker found")
            continue
        m0 = markers[0]
        print(f"\n=== id {pid}: {desc} (expected {exp}) ===")
        print(f"  step-0 AX marker pos = {m0}; entry token = JSR main")
        for label, blk in (("PRE (blk7,post-L5)", PRE_BLOCK),
                            ("POST(blk8,post-L6 AX-pass)", FC_BLOCK)):
            res = probe.residual_at(bc, block_idx=blk, position=m0,
                                    dim_names=dim_names, max_steps=2)
            axc = _nib_decode(res, "AX_CARRY_LO", "AX_CARRY_HI")
            out = _nib_decode(res, "OUTPUT_LO", "OUTPUT_HI")
            print(f"  -- {label} --")
            print(f"     OP_JSR={res.get('OP_JSR',0):.2f} OP_ENT={res.get('OP_ENT',0):.2f} "
                  f"MARK_AX={res.get('MARK_AX',0):.2f} MARK_PC={res.get('MARK_PC',0):.2f} "
                  f"IS_BYTE={res.get('IS_BYTE',0):.2f}")
            print(f"     AX_CARRY lo={axc[0]} hi={axc[1]} "
                  f"val={('0x%02x'%axc[2]) if axc[2] is not None else '--'} "
                  f"(mlo={axc[3]:.2f} mhi={axc[4]:.2f})")
            print(f"     OUTPUT   lo={out[0]} hi={out[1]} "
                  f"val={('0x%02x'%out[2]) if out[2] is not None else '--'} "
                  f"(mlo={out[3]:.2f} mhi={out[4]:.2f})")


if __name__ == "__main__":
    main()
