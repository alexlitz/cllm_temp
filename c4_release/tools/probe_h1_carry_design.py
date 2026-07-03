#!/usr/bin/env python3
"""Design the L4 byte-1 AX carry: what to read at prev byte-1 row, what L9 needs.

(A) At block 4 (where an L4 head reads), dump EMBED/CLEAN_EMBED + marker
    bits at the PREV step's byte-1 rows (the K target) and at the CURRENT
    step's byte-1 predictor row (the Q firing row), IMM and PSH steps.
(B) Confirm: does L9 regenerate the H1 one-hot if AX_CARRY_HI at the m+1
    row holds the byte1 value? Check whether AX_CARRY_HI on the fresh
    (IMM) step encodes byte1 as a nibble pair.

PORTED to ``tools/probe_lib`` — dims from the BUILT ``model.dim_positions``
(the old ``pos`` read the STALE static registry -> wrong cell post-widen), the
post-block row via ``probe_lib.residual_row``, markers via
``probe_lib.register_marker_rows``.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_h1_carry_design.py
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tools import probe_lib as P  # noqa: E402

GATES = ("MARK_AX", "MARK_SP", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1",
         "BYTE_INDEX_2", "HAS_SE", "MARK_PC", "MARK_STACK0")


def nib(r, base):
    if base is None:
        return None
    v = [float(r[base + k]) for k in range(16)]
    m = max(range(16), key=lambda k: v[k])
    return (m, round(v[m], 1))


def main():
    probe = P.build_probe()
    model = probe.model
    dp = P.dim_positions(model)

    def pos(nm):
        v = dp.get(nm)
        return None if v is None else int(v)

    L1H0, L1H1 = pos("L1H0"), pos("L1H1")
    EL, EH = pos("EMBED_LO"), pos("EMBED_HI")
    CEL, CEH = pos("CLEAN_EMBED_LO"), pos("CLEAN_EMBED_HI")

    bc, _ = compile_c("int main() { return 654 + 114; }")  # byte1=2
    trace = probe.probe(bc, max_steps=4)
    ms = P.register_marker_rows(trace, "REG_AX")

    # Block-4 residual for every row of interest (one truncated forward each).
    print("=== block 4 rows for L4 byte-1 carry design (654+114, byte1=2) ===")
    rows = [
        (ms[0],     "PREV(IMM) marker"),
        (ms[0] + 1, "PREV(IMM) b0-tok/b1-pred"),
        (ms[0] + 2, "PREV(IMM) b1-token(=0x02)"),
        (ms[1],     "CUR(PSH) marker"),
        (ms[1] + 1, "CUR(PSH) b0-tok/b1-pred <- Q fires here"),
        (ms[1] + 2, "CUR(PSH) b1-token"),
    ]
    for row, lbl in rows:
        r = P.residual_row(probe, bc, block_idx=4, position=row, max_steps=4)
        g = {nm: round(float(r[pos(nm)]), 1) for nm in GATES
             if pos(nm) is not None and abs(float(r[pos(nm)])) > 0.4}
        l1h1 = [round(float(r[L1H1 + k]), 1) for k in range(5)] if L1H1 else None
        l1h0 = [round(float(r[L1H0 + k]), 1) for k in range(5)] if L1H0 else None
        print(f"\n row {row} [{lbl}] tok={trace.get(row,{}).get('token')}")
        print(f"   gates={g}")
        print(f"   L1H1[:5]={l1h1}  L1H0[:5]={l1h0}")
        print(f"   EMBED nib lo={nib(r,EL)} hi={nib(r,EH)} | "
              f"CLEAN_EMBED nib lo={nib(r,CEL)} hi={nib(r,CEH)}")


if __name__ == "__main__":
    main()
