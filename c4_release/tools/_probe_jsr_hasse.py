#!/usr/bin/env python3
"""Efficient: read HAS_SE / AX_CARRY / OUTPUT at EVERY JSR AX-marker row.

One full-context build per program, then ONE forward at block 7 (pre AX-pass)
and ONE at block 8 (post). Prints per JSR step so we can see if HAS_SE
discriminates the entry JSR-to-main (polluted) from nested JSRs (legitimate).
Uses print(flush=True).
"""
from __future__ import annotations
import argparse, os, sys, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from tools import probe_lib as P  # noqa: E402

PRE_BLOCK = 7
FC_BLOCK = 8


def _nib(res, lo, hi):
    los = [res.get(f"{lo}+{k}", 0.0) for k in range(16)]
    his = [res.get(f"{hi}+{k}", 0.0) for k in range(16)]
    li = max(range(16), key=lambda k: los[k]) if max(los) > 0.3 else None
    hz = max(range(16), key=lambda k: his[k]) if max(his) > 0.3 else None
    val = (li | (hz << 4)) if (li is not None and hz is not None) else None
    return val, max(los), max(his)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="700,750,600,900")
    ap.add_argument("--max-steps", type=int, default=10)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    print("building probe...", flush=True)
    probe = P.build_probe()
    model = probe.model
    dp = P.dim_positions(model)
    progs = generate_test_programs()
    print("probe built.", flush=True)

    dim_names = {}
    for nm in ("AX_CARRY_LO", "AX_CARRY_HI", "OUTPUT_LO", "OUTPUT_HI"):
        base = dp.get(nm)
        if base is None:
            continue
        for k in range(16):
            dim_names[f"{nm}+{k}"] = int(base) + k
    for nm in ("MARK_AX", "OP_JSR", "OP_ENT", "HAS_SE", "IS_BYTE", "TEMP"):
        b = dp.get(nm)
        if b is not None:
            dim_names[nm] = int(b)

    for pid in ids:
        src, exp, desc = progs[pid]
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=args.max_steps)
        markers = P.register_marker_rows(trace, "REG_AX")
        print(f"\n=== id{pid}: {desc} (exp {exp}) — {len(markers)} AX markers ===",
              flush=True)
        for si, m in enumerate(markers):
            resP = probe.residual_at(bc, block_idx=PRE_BLOCK, position=m,
                                     dim_names=dim_names, max_steps=args.max_steps)
            if resP.get("OP_JSR", 0) < 2.0:
                continue
            resQ = probe.residual_at(bc, block_idx=FC_BLOCK, position=m,
                                     dim_names=dim_names, max_steps=args.max_steps)
            axc, amlo, amhi = _nib(resP, "AX_CARRY_LO", "AX_CARRY_HI")
            outv, omlo, omhi = _nib(resQ, "OUTPUT_LO", "OUTPUT_HI")
            kind = "ENTRY-to-main" if si == 0 else f"nested#{si}"
            print(f"  AXmark#{si} pos={m} [{kind}]: HAS_SE={resP.get('HAS_SE',0):+.3f} "
                  f"OP_JSR={resP.get('OP_JSR',0):.1f} OP_ENT={resP.get('OP_ENT',0):.1f} "
                  f"TEMP0={resP.get('TEMP',0):+.2f}", flush=True)
            print(f"     PRE  AX_CARRY val={('0x%02x'%axc) if axc is not None else '--'} "
                  f"(mlo={amlo:.2f} mhi={amhi:.2f})   "
                  f"POST OUTPUT val={('0x%02x'%outv) if outv is not None else '--'} "
                  f"(mlo={omlo:.2f} mhi={omhi:.2f})", flush=True)


if __name__ == "__main__":
    main()
