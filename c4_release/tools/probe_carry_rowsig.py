#!/usr/bin/env python3
"""Dump the byte-1 predictor row signature candidates at a chosen block.

Find a row signature that:
  (a) is ~constant across all add programs on the byte-1 predictor row
      (= marker+1, the AX byte-0 token), so it fires the Q AND matches K;
  (b) is DISTINCT from the byte-0 predictor (marker) and byte-2 predictor
      (marker+2) rows so the head only fires on byte-1 rows.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_carry_rowsig.py [BLOCK]
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


def name_for(p):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= p < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{p - slot.start}", slot.size)
    return best[0] if best else f"dim{p}"


PROGRAMS = {
    "654+114": "int main() { return 654 + 114; }",
    "754+104": "int main() { return 754 + 104; }",
    "913+558": "int main() { return 913 + 558; }",
}


@torch.no_grad()
def fwd(probe, ctx, block):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block)[0]
    return x.to_dense() if x.is_sparse else x


def main():
    block = int(sys.argv[1]) if len(sys.argv) > 1 else 11  # read for block 12
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)

    # For each program collect the byte-1 predictor rows (fresh & carried)
    # and the neighbouring byte-0 / byte-2 predictor rows.
    # Find dims that are ~constant (>0.5) on ALL byte-1 predictor rows
    # across programs+steps, and ~0 on byte-0/byte-2 predictor rows.
    b1_rows = []   # list of residual vectors (byte-1 predictors)
    other_rows = []  # byte-0 + byte-2 predictors
    for name, src in PROGRAMS.items():
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=4)
        ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        ctx = probe._final_context(bc, max_steps=4)
        X = fwd(probe, ctx, block)
        for s in (0, 1):  # IMM fresh, PSH carried
            b1_rows.append(X[ms[s] + 1])     # byte-1 predictor
            other_rows.append(X[ms[s] + 0])  # byte-0 predictor (marker row)
            other_rows.append(X[ms[s] + 2])  # byte-2 predictor

    D = b1_rows[0].shape[0]
    b1 = torch.stack(b1_rows)       # [N1, D]
    oth = torch.stack(other_rows)   # [N2, D]
    b1_min = b1.min(0).values
    b1_max = b1.max(0).values
    oth_max_abs = oth.abs().max(0).values

    # Candidate signature dims: present (>0.5) on ALL byte-1 rows, near-0
    # (<0.3) on every byte-0/byte-2 row.
    print(f"=== block {block+1} read point (stop_after_block={block}) ===")
    print("Byte-1 predictor row signature candidates "
          "(>0.5 on all b1 rows, <0.3 abs on all b0/b2 rows):\n")
    cands = []
    for d in range(D):
        if float(b1_min[d]) > 0.5 and float(oth_max_abs[d]) < 0.3:
            cands.append(d)
    for d in cands:
        print(f"  dim {d:4d} {name_for(d):28s} "
              f"b1_range=[{float(b1_min[d]):.2f},{float(b1_max[d]):.2f}] "
              f"b0b2_maxabs={float(oth_max_abs[d]):.2f}")
    if not cands:
        print("  (none clean) — relaxing: dims >0.5 on all b1 rows:")
        for d in range(D):
            if float(b1_min[d]) > 0.5:
                print(f"  dim {d:4d} {name_for(d):28s} "
                      f"b1_range=[{float(b1_min[d]):.2f},{float(b1_max[d]):.2f}] "
                      f"b0b2_maxabs={float(oth_max_abs[d]):.2f}")


if __name__ == "__main__":
    main()
