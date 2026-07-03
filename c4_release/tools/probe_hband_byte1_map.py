#!/usr/bin/env python3
"""Map the fresh-step byte-1 H-band one-hot: value v -> (H-band, offset).

The fresh IMM step emits byte1=v via head.weight[v, H<k>+off]=+5.0 (a positional
one-hot across H1..H7). Dump H0..H7 on the FRESH byte-1 predictor row for
v=0..40 and identify the single active (band,cell) per v. This tells us
EXACTLY which bands the value-general carry must copy + mirror.

PORTED to ``tools/probe_lib`` — H-band bases from the BUILT
``model.dim_positions`` (the old ``P(nm)`` read the STALE static registry ->
wrong cell post-widen), the residual row via ``probe_lib.residual_row`` and the
REG_AX marker via ``probe_lib.register_marker_rows``.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_hband_byte1_map.py
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.compiler import compile_c  # noqa: E402
from tools import probe_lib as PL  # noqa: E402


def main():
    probe = PL.build_probe()
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    last_block = len(model.blocks) - 1
    dp = PL.dim_positions(model)
    H_bases = {f"H{k}": int(dp[f"H{k}"]) for k in range(8) if f"H{k}" in dp}

    # First: which (band,cell) does the LM head READ to emit byte token v?
    # head.weight[v, dim] for dims in H0..H7. Find the +5.0 column per v.
    print("### LM-head emission columns per byte token v (H-bands) ###")
    print("  v : (band, off)  weight")
    vmap = {}
    for v in range(0, 40):
        best = None
        for k in range(8):
            base = H_bases.get(f"H{k}")
            if base is None:
                continue
            for off in range(7):
                w = float(W[v, base + off])
                if w > 2.0 and (best is None or w > best[2]):
                    best = (f"H{k}", off, w)
        vmap[v] = best
        if best:
            print(f"  {v:3d} : ({best[0]}, +{best[1]})  {best[2]:.1f}")
        else:
            print(f"  {v:3d} : (none in H-bands)")

    # Now: fresh-step residual H-band activation per v (does it match vmap?)
    print("\n### FRESH-step byte-1 predictor: active H cell per v ###")
    for v in (0, 2, 4, 5, 8, 11, 12, 15, 18, 19, 25, 30):
        a = v * 256 + 0x40
        bc, _ = compile_c(f"int main() {{ return {a}; }}")
        trace = probe.probe(bc, max_steps=2)
        ms = PL.register_marker_rows(trace, "REG_AX")
        m = ms[0]
        pos = m + 1
        em = trace.get(m + 2, {}).get("token")
        res = PL.residual_row(probe, bc, last_block, pos, max_steps=2)
        res = res.float().cpu()
        # report active cell in each band
        cells = []
        for k in range(8):
            base = H_bases.get(f"H{k}")
            if base is None:
                continue
            vals = [float(res[base + off]) for off in range(7)]
            am = max(range(7), key=lambda o: vals[o])
            if vals[am] > 3.0:
                cells.append(f"H{k}+{am}={vals[am]:.0f}")
        want = vmap.get(v)
        print(f"  v={v:3d} emitted=0x{(em or 0):02x} "
              f"LMcol={want[0]}+{want[1] if want else '?'}  active=[{' '.join(cells)}]")


if __name__ == "__main__":
    main()
