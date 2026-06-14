#!/usr/bin/env python3
"""Map the fresh-step byte-1 H-band one-hot: value v -> (H-band, offset).

The fresh IMM step emits byte1=v via head.weight[v, H<k>+off]=+5.0 (a positional
one-hot across H1..H7). Dump H0..H7 (registry 60..116) on the FRESH byte-1
predictor row for v=0..40 and identify the single active (band,cell) per v.
This tells us EXACTLY which bands the value-general carry must copy + mirror.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_hband_byte1_map.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.probe_groundtruth import GroundTruthProbe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic  # noqa: E402

_REG = build_default_registry_dynamic()


def P(nm):
    return int(_REG.slots[nm].start)


@torch.no_grad()
def residual_full(probe, bc, block_idx, position, max_steps):
    ctx = probe._final_context(bc, max_steps=max_steps)
    if position < 0:
        position = len(ctx) + position
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0, position]


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    last_block = len(model.blocks) - 1
    RAX = int(Token.REG_AX)
    H_bases = {f"H{k}": P(f"H{k}") for k in range(8)}

    # First: which (band,cell) does the LM head READ to emit byte token v?
    # head.weight[v, dim] for dims in H0..H7. Find the +5.0 column per v.
    print("### LM-head emission columns per byte token v (H-bands) ###")
    print("  v : (band, off)  weight")
    vmap = {}
    for v in range(0, 40):
        best = None
        for k in range(8):
            base = H_bases[f"H{k}"]
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
        ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        m = ms[0]
        pos = m + 1
        em = trace.get(m + 2, {}).get("token")
        res = residual_full(probe, bc, last_block, pos, 2)
        if res.is_sparse:
            res = res.to_dense()
        res = res.float().cpu()
        # report active cell in each band
        cells = []
        for k in range(8):
            base = H_bases[f"H{k}"]
            vals = [float(res[base + off]) for off in range(7)]
            am = max(range(7), key=lambda o: vals[o])
            if vals[am] > 3.0:
                cells.append(f"H{k}+{am}={vals[am]:.0f}")
        want = vmap.get(v)
        print(f"  v={v:3d} emitted=0x{(em or 0):02x} "
              f"LMcol={want[0]}+{want[1] if want else '?'}  active=[{' '.join(cells)}]")


if __name__ == "__main__":
    main()
