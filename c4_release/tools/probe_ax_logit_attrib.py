#!/usr/bin/env python3
"""Attribute the AX byte-1 emission logit to residual dims (no final norm).

logit[t] = head.weight[t] . residual_lastblock + head.bias[t].  At the byte-1
predictor row, compute per-dim contribution to (logit[want] - logit[got]) on
the IMM step (want=got=correct) and PSH step (want!=got, the bug), to find the
exact dims the LM head reads to emit AX byte 1.

PORTED to ``tools/probe_lib`` — the residual read, the dim->name reverse map
and the (W[want]-W[got])*resid attribution are all now shared library calls
(``probe_lib.residual_row`` / ``probe_lib.DimNamer`` / ``probe_lib.logit_attrib``).
The old hand-rolled ``name_for`` read the STALE ``build_default_registry_dynamic``
static slots, which mislabel every dim after the widen repack; ``DimNamer`` uses
the BUILT ``model.dim_positions`` so the labels are correct.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_ax_logit_attrib.py
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


def main():
    probe = P.build_probe()
    namer = P.DimNamer(probe.model)

    bc, _ = compile_c("int main() { return 654 + 114; }")  # AX byte1 = 0x02
    trace = probe.probe(bc, max_steps=4)
    ax_rows = P.register_marker_rows(trace, "REG_AX")

    for step, label in ((0, "IMM(correct=0x02)"), (1, "PSH(wrong->0x00)")):
        pos = ax_rows[step] + 1  # byte-1 predictor row
        attr = P.logit_attrib(probe, bc, pos, want=0x02, got=0x00, max_steps=4)
        print(f"\n=== {label} step{step} byte-1 predictor pos {pos} ===")
        print(f"   logit[0x02]={attr.logit_want:.3f}  "
              f"logit[0x00]={attr.logit_got:.3f}  "
              f"diff(02-00)={attr.diff:.3f}")
        print("   top dims driving (logit02 - logit00):")
        for di in attr.top(18):
            print(f"      dim {di:4d} {namer.name_for(di):26s} "
                  f"res={float(attr.residual[di]):8.3f} "
                  f"contrib={float(attr.contrib[di]):8.3f}")


if __name__ == "__main__":
    main()
