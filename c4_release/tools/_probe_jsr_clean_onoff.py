#!/usr/bin/env python3
"""GATE-2 core check: does the NARROWED C4_JSR_AX_CLEAN band

  (a) still ZERO the polluted OUTPUT on the ENTRY JSR-to-main (gcd id900 step0),
  (b) PRESERVE the real caller AX on a NESTED JSR (rec_sum id750, n=0x0e)?

Reads OUTPUT at the AX-marker on the entry JSR step (#0) and a nested JSR step
after block 8 (post AX-passthrough+clean). Set C4_JSR_AX_CLEAN via env to
compare. print(flush=True).
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


def _nib(res, lo, hi):
    los = [res.get(f"{lo}+{k}", 0.0) for k in range(16)]
    his = [res.get(f"{hi}+{k}", 0.0) for k in range(16)]
    li = max(range(16), key=lambda k: los[k]) if max(los) > 0.3 else None
    hz = max(range(16), key=lambda k: his[k]) if max(his) > 0.3 else None
    val = (li | (hz << 4)) if (li is not None and hz is not None) else None
    return val, max(los), max(his)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()

    flag = os.environ.get("C4_JSR_AX_CLEAN", "0")
    print(f"C4_JSR_AX_CLEAN={flag}  building...", flush=True)
    probe = P.build_probe()
    dp = P.dim_positions(probe.model)
    progs = generate_test_programs()
    print("built.", flush=True)

    dim_names = {}
    for nm in ("AX_CARRY_LO", "AX_CARRY_HI", "OUTPUT_LO", "OUTPUT_HI"):
        base = dp.get(nm)
        for k in range(16):
            dim_names[f"{nm}+{k}"] = int(base) + k
    for nm in ("OP_JSR", "HAS_SE"):
        dim_names[nm] = int(dp.get(nm))

    # (900, "ENTRY-to-main gcd", marker index 0),
    # (750, "NESTED rec_sum", marker index that is a JSR after step 0)
    for pid, label in ((900, "gcd ENTRY"), (750, "rec_sum")):
        src, exp, desc = progs[pid]
        bc, _ = compile_c(src)
        markers = P.register_marker_rows(probe.probe(bc, max_steps=args.max_steps),
                                         "REG_AX")
        print(f"\n=== id{pid} {desc} (exp {exp}) ===", flush=True)
        for si, m in enumerate(markers):
            resQ = probe.residual_at(bc, block_idx=FC_BLOCK, position=m,
                                     dim_names=dim_names, max_steps=args.max_steps)
            if resQ.get("OP_JSR", 0) < 2.0:
                continue
            outv, omlo, omhi = _nib(resQ, "OUTPUT_LO", "OUTPUT_HI")
            axv, amlo, amhi = _nib(resQ, "AX_CARRY_LO", "AX_CARRY_HI")
            kind = "ENTRY(HAS_SE~0)" if si == 0 else f"nested#{si}(HAS_SE~1)"
            print(f"  AXmark#{si} [{kind}] HAS_SE={resQ.get('HAS_SE',0):+.3f} "
                  f"AX_CARRY val={('0x%02x'%axv) if axv is not None else '--'} "
                  f"-> OUTPUT val={('0x%02x'%outv) if outv is not None else '--'} "
                  f"(mlo={omlo:.2f} mhi={omhi:.2f})", flush=True)


if __name__ == "__main__":
    main()
