#!/usr/bin/env python3
"""Design the L4 byte-1 AX carry: what to read at prev byte-1 row, what L9 needs.

(A) At block 4 (where an L4 head reads), dump EMBED/CLEAN_EMBED + marker
    bits at the PREV step's byte-1 rows (the K target) and at the CURRENT
    step's byte-1 predictor row (the Q firing row), IMM and PSH steps.
(B) Confirm: does L9 regenerate the H1 one-hot if AX_CARRY_HI at the m+1
    row holds the byte1 value? Check whether AX_CARRY_HI on the fresh
    (IMM) step encodes byte1 as a nibble pair.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_h1_carry_design.py
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


GATES = ("MARK_AX", "MARK_SP", "IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1",
         "BYTE_INDEX_2", "HAS_SE", "MARK_PC", "MARK_STACK0")
L1H0, L1H1 = pos("L1H0"), pos("L1H1")
EL, EH = pos("EMBED_LO"), pos("EMBED_HI")
CEL, CEH = pos("CLEAN_EMBED_LO"), pos("CLEAN_EMBED_HI")


@torch.no_grad()
def fwd(probe, ctx, block):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block)[0]
    return x.to_dense() if x.is_sparse else x


def nib(r, base):
    v = [float(r[base + k]) for k in range(16)]
    m = max(range(16), key=lambda k: v[k])
    return (m, round(v[m], 1))


def main():
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)
    bc, _ = compile_c("int main() { return 654 + 114; }")  # byte1=2
    trace = probe.probe(bc, max_steps=4)
    ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
    ctx = probe._final_context(bc, max_steps=4)
    X4 = fwd(probe, ctx, 4)

    # Step 0 (IMM/fresh) is the PREV step for step 1 (PSH).
    # Its rows: marker=ms[0]. byte tokens at ms[0]+1 (b0), +2 (b1 token=0x02).
    # Step 1 (PSH) predictor row = ms[1]+1 (where my Q must fire).
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
        r = X4[row]
        g = {nm: round(float(r[pos(nm)]), 1) for nm in GATES
             if pos(nm) is not None and abs(float(r[pos(nm)])) > 0.4}
        l1h1 = [round(float(r[L1H1 + k]), 1) for k in range(5)]
        l1h0 = [round(float(r[L1H0 + k]), 1) for k in range(5)]
        print(f"\n row {row} [{lbl}] tok={trace.get(row,{}).get('token')}")
        print(f"   gates={g}")
        print(f"   L1H1[:5]={l1h1}  L1H0[:5]={l1h0}")
        print(f"   EMBED nib lo={nib(r,EL)} hi={nib(r,EH)} | "
              f"CLEAN_EMBED nib lo={nib(r,CEL)} hi={nib(r,CEH)}")


if __name__ == "__main__":
    main()
