#!/usr/bin/env python3
"""Trace ALL JSR steps (entry + nested) in a program: read AX_CARRY / OUTPUT /
FETCH_HI / HAS_SE at the AX-marker on each JSR step, autoregressively.

Goal: find a discriminator that separates the polluted ENTRY JSR-to-main
(AX undefined) from a LEGITIMATE nested JSR (AX holds a computed value the
passthrough must preserve). Reads block 7 (pre AX-pass) and block 8 (post).
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

FC_BLOCK = 8
PRE_BLOCK = 7


def _nib(res, lo, hi):
    los = [res.get(f"{lo}+{k}", 0.0) for k in range(16)]
    his = [res.get(f"{hi}+{k}", 0.0) for k in range(16)]
    li = max(range(16), key=lambda k: los[k]) if max(los) > 0.3 else None
    hz = max(range(16), key=lambda k: his[k]) if max(his) > 0.3 else None
    val = (li | (hz << 4)) if (li is not None and hz is not None) else None
    return val, max(los), max(his)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="700,750,900")
    ap.add_argument("--max-steps", type=int, default=12)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    probe = P.build_probe()
    model = probe.model
    dp = P.dim_positions(model)
    progs = generate_test_programs()

    dim_names = {}
    for nm in ("AX_CARRY_LO", "AX_CARRY_HI", "OUTPUT_LO", "OUTPUT_HI",
               "FETCH_HI", "FETCH_LO"):
        base = dp.get(nm)
        if base is None:
            continue
        for k in range(16):
            dim_names[f"{nm}+{k}"] = int(base) + k
    for nm in ("MARK_AX", "OP_JSR", "OP_ENT", "HAS_SE", "IS_BYTE",
               "TEMP", "MARK_PC"):
        b = dp.get(nm)
        if b is not None:
            dim_names[nm] = int(b)

    for pid in ids:
        src, exp, desc = progs[pid]
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=args.max_steps)
        markers = P.register_marker_rows(trace, "REG_AX")
        print(f"\n=== id{pid}: {desc} (exp {exp}) — {len(markers)} AX markers ===")
        for si, m in enumerate(markers):
            resP = probe.residual_at(bc, block_idx=PRE_BLOCK, position=m,
                                     dim_names=dim_names,
                                     max_steps=args.max_steps)
            if resP.get("OP_JSR", 0) < 2.0:
                continue
            resQ = probe.residual_at(bc, block_idx=FC_BLOCK, position=m,
                                     dim_names=dim_names,
                                     max_steps=args.max_steps)
            axc, amlo, amhi = _nib(resP, "AX_CARRY_LO", "AX_CARRY_HI")
            outv, omlo, omhi = _nib(resQ, "OUTPUT_LO", "OUTPUT_HI")
            fhi = [resP.get(f"FETCH_HI+{k}", 0.0) for k in range(16)]
            flo = [resP.get(f"FETCH_LO+{k}", 0.0) for k in range(16)]
            fhi_arg = max(range(16), key=lambda k: fhi[k]) if max(fhi) > 0.3 else None
            flo_arg = max(range(16), key=lambda k: flo[k]) if max(flo) > 0.3 else None
            print(f"  AXmark#{si} pos={m}: HAS_SE={resP.get('HAS_SE',0):.2f} "
                  f"OP_JSR={resP.get('OP_JSR',0):.1f} OP_ENT={resP.get('OP_ENT',0):.1f}")
            print(f"     PRE  AX_CARRY val={('0x%02x'%axc) if axc is not None else '--'} "
                  f"(mlo={amlo:.2f} mhi={amhi:.2f})  "
                  f"FETCH_HI_arg={fhi_arg} FETCH_LO_arg={flo_arg}")
            print(f"     POST OUTPUT   val={('0x%02x'%outv) if outv is not None else '--'} "
                  f"(mlo={omlo:.2f} mhi={omhi:.2f})")


if __name__ == "__main__":
    main()
