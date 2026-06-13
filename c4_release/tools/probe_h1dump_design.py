#!/usr/bin/env python3
"""Design probe for the H1_DUMP cross-step carry head.

Picks a host read-point block and dumps, across several add programs:
  (1) byte-1 predictor row signature candidates (Q-fire on CURRENT carried
      row + K-match on PREV fresh row): dims ~1.0 on ALL byte-1 predictor
      rows (fresh+carried, all programs), ~0 on neighbouring b0/b2 rows.
  (2) AX_CARRY sum gate (carried-vs-fresh) at the byte-1 predictor row.
  (3) prev-step (fresh) byte-1 predictor H1 band (the K target one-hot to
      V-copy), and current carried-step H1 band (should be empty).

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_h1dump_design.py [READBLOCK]
  READBLOCK = stop_after_block (host reads state AFTER this block).
  Default 12 (host = block 13).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

_REG = build_default_registry_dynamic()


def pos(nm):
    s = _REG.slots.get(nm)
    return None if s is None else int(s.start)


def name_for(p):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= p < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{p - slot.start}", slot.size)
    return best[0] if best else f"dim{p}"


H1 = pos("H1")
AXC_LO, AXC_HI = pos("AX_CARRY_LO"), pos("AX_CARRY_HI")

PROGRAMS = {
    "654+114(b1=2)": "int main() { return 654 + 114; }",
    "754+104(b1=2)": "int main() { return 754 + 104; }",
    "913+558(b1=3)": "int main() { return 913 + 558; }",
    "300+0(b1=1)":   "int main() { return 300 + 0; }",
    "1024+0(b1=4)":  "int main() { return 1024 + 0; }",
}


@torch.no_grad()
def fwd(probe, ctx, block):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block)[0]
    return x.to_dense() if x.is_sparse else x


def axc_sum(r):
    return sum(float(r[AXC_LO + k]) + float(r[AXC_HI + k]) for k in range(16))


def h1_band(r):
    return [round(float(r[H1 + k]), 1) for k in range(7)]


def main():
    stop = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)

    b1_rows, other_rows = [], []
    print(f"=== host reads after block {stop} (host = block {stop+1}) ===\n")
    for name, src in PROGRAMS.items():
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=4)
        ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        ctx = probe._final_context(bc, max_steps=4)
        X = fwd(probe, ctx, stop)
        rf, rc = X[ms[0] + 1], X[ms[1] + 1]
        print(f"{name}: axc fresh={axc_sum(rf):.1f} carried={axc_sum(rc):.1f} | "
              f"prevH1(K,fresh)={h1_band(rf)} curH1(carried)={h1_band(rc)}")
        for s in (0, 1):
            b1_rows.append(X[ms[s] + 1])
            other_rows.append(X[ms[s] + 0])
            other_rows.append(X[ms[s] + 2])

    D = b1_rows[0].shape[0]
    b1 = torch.stack(b1_rows)
    oth = torch.stack(other_rows)
    b1_min = b1.min(0).values
    b1_max = b1.max(0).values
    oth_maxabs = oth.abs().max(0).values
    print("\nByte-1 predictor row signature candidates "
          "(>0.5 on ALL b1 fresh+carried rows, <0.3 abs on all b0/b2):")
    for d in range(D):
        if float(b1_min[d]) > 0.5 and float(oth_maxabs[d]) < 0.3:
            print(f"  dim {d:4d} {name_for(d):26s} "
                  f"b1=[{float(b1_min[d]):.2f},{float(b1_max[d]):.2f}] "
                  f"b0b2max={float(oth_maxabs[d]):.2f}")


if __name__ == "__main__":
    main()
