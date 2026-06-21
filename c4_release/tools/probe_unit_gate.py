#!/usr/bin/env python3
"""Print the W_gate / W_up input-dim weights (the gating CONDITIONS) for a set
of block-N FFN units, resolved to dim NAMES, so we can grep for the owning rule.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic,
)
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

BLK = int(os.environ.get("PROBE_BLK", "32"))
UNITS = [int(u) for u in os.environ.get("PROBE_UNITS", "1874,1876,3").split(",")]


def main():
    _, layout = compile_full_vm_dynamic()
    dp = layout.dim_positions
    inv = {}
    for name, base in dp.items():
        # base->name; also expand band names to +k by nearest base
        inv.setdefault(int(base), name)
    # Build a per-dim label that prefers exact then base+offset
    def label(d):
        if d in inv:
            return inv[d]
        # find nearest base below
        best = None
        for b, n in inv.items():
            if b <= d and (best is None or b > best[0]):
                best = (b, n)
        if best:
            return f"{best[1]}+{d - best[0]}"
        return f"dim{d}"

    probe = build_groundtruth_probe()
    ffn = probe.model.blocks[BLK].ffn
    Wg, Wu = ffn.W_gate, ffn.W_up
    bg = ffn.b_gate
    bu = ffn.b_up
    out_hi = dp["OUTPUT_HI_THIS_STEP"]
    Wd = ffn.W_down
    for u in UNITS:
        print(f"\n==== block {BLK} unit {u} ====")
        bgv = float(bg[u]) if bg is not None else 0.0
        buv = float(bu[u]) if bu is not None else 0.0
        print(f"  b_gate={bgv:+.3f}  b_up={buv:+.3f}")
        for label_w, W in (("W_gate", Wg), ("W_up", Wu)):
            row = W[u]
            nz = [(float(row[d]), int(d)) for d in range(row.shape[0])
                  if abs(float(row[d])) > 1e-6]
            nz.sort(key=lambda t: -abs(t[0]))
            print(f"  -- {label_w} nonzero ({len(nz)}) top 24 --")
            for v, d in nz[:24]:
                print(f"      {label(d):24s} = {v:+10.3f}")
        # output writes
        col = Wd[:, u]
        outs = [(float(col[d]), int(d)) for d in range(col.shape[0])
                if abs(float(col[d])) > 1e-6]
        outs.sort(key=lambda t: -abs(t[0]))
        print(f"  -- W_down writes (out dims) top 12 --")
        for v, d in outs[:12]:
            print(f"      {label(d):24s} = {v:+10.3f}")


if __name__ == "__main__":
    main()
