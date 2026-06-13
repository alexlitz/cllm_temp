#!/usr/bin/env python3
"""Runtime verification of the AX byte-1 DUMP carry head (when enabled).

For each add program, traces the H1 band at the LM-head read point (block 39)
on BOTH the fresh-AX step (must stay byte-identical: one-hot present) and the
carried (PSH) step (the carry must RE-SUPPLY the one-hot). Also dumps the
decoded AX bytes per step so the byte-1 truncation fix is visible end to end.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_h1dump_carry_runtime.py [BLOCK]
  BLOCK = stop_after_block for the H1 trace (default 39 = LM-head input).
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


H1 = pos("H1")
AXC_LO, AXC_HI = pos("AX_CARRY_LO"), pos("AX_CARRY_HI")

PROGRAMS = {
    "654+114(b1=2)": "int main() { return 654 + 114; }",
    "913+558(b1=3)": "int main() { return 913 + 558; }",
    "300+0(b1=1)":   "int main() { return 300 + 0; }",
    "1024+0(b1=4)":  "int main() { return 1024 + 0; }",
}


@torch.no_grad()
def fwd(probe, ctx, block):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block)[0]
    return x.to_dense() if x.is_sparse else x


def h1_band(r):
    return [round(float(r[H1 + k]), 1) for k in range(7)]


def main():
    stop = int(sys.argv[1]) if len(sys.argv) > 1 else 39
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)
    print(f"=== H1 band at block {stop} (byte-1 predictor row = marker+1) ===\n")
    for name, src in PROGRAMS.items():
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=4)
        ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        ctx = probe._final_context(bc, max_steps=4)
        X = fwd(probe, ctx, stop)
        rf, rc = X[ms[0] + 1], X[ms[1] + 1]
        print(f"{name}:")
        print(f"   FRESH(step0) H1={h1_band(rf)}   <- must stay one-hot")
        print(f"   CARRIED(step1) H1={h1_band(rc)} <- carry must re-supply")


if __name__ == "__main__":
    main()
