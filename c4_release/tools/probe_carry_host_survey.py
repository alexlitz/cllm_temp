#!/usr/bin/env python3
"""Survey late blocks for the AX byte-1 carry head host.

For each physical block, measure at the block's INPUT residual (= state
after block N-1) the two gating signals the carry head needs, across
multiple add programs, on BOTH the fresh (IMM, step 0) and carried (PSH,
step 1) byte-1 predictor rows:

  * ``L1H1+2`` row signature (must be ~1.0 on BOTH steps & all programs:
    it fires the Q on the current carried row AND matches the K on the
    prev fresh row).
  * ``Sum(AX_CARRY)`` carried-vs-fresh gate (must be separable: large on
    one class, ~0 on the other, stable across programs).
  * The ``H1`` band at the PREV step's byte-1 predictor row (the K
    target) — confirms the one-hot is present there to be V-copied.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_carry_host_survey.py
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


L1H1 = pos("L1H1")
H1 = pos("H1")
AXC_LO, AXC_HI = pos("AX_CARRY_LO"), pos("AX_CARRY_HI")
IS_BYTE = pos("IS_BYTE")

PROGRAMS = {
    "654+114(b1=2)": "int main() { return 654 + 114; }",
    "754+104(b1=2)": "int main() { return 754 + 104; }",
    "913+558(b1=3)": "int main() { return 913 + 558; }",
    "300+0(b1=1)":   "int main() { return 300 + 0; }",
}


@torch.no_grad()
def fwd(probe, ctx, block):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block)[0]
    return x.to_dense() if x.is_sparse else x


def axc_sum(r):
    s = 0.0
    for k in range(16):
        s += float(r[AXC_LO + k]) + float(r[AXC_HI + k])
    return s


def h1_band(r):
    return [round(float(r[H1 + k]), 1) for k in range(7)]


def main():
    probe = GroundTruthProbe.build()
    RAX = int(Token.REG_AX)
    nblocks = len(probe.model.blocks)

    # Precompute per-program: marker positions + ctx, plus residual cache
    # per block (one forward per (program, block)).
    progs = {}
    for name, src in PROGRAMS.items():
        bc, _ = compile_c(src)
        trace = probe.probe(bc, max_steps=4)
        ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        ctx = probe._final_context(bc, max_steps=4)
        progs[name] = (ctx, ms)

    print("Surveying blocks for carry-head gate cleanliness.")
    print("Per block: L1H1+2 sig (fresh/carried predictor rows), "
          "Sum(AX_CARRY) (fresh/carried), prevH1 (K target one-hot).\n")
    print("Legend: fresh=IMM step0 predictor (ms[0]+1); "
          "carried=PSH step1 predictor (ms[1]+1); "
          "prev=ms[0]+1 (K target for carried step).\n")

    # Survey blocks: input to block B = state after block B-1, so to read
    # 'block B read point' we stop_after_block=B-1.
    for B in range(8, nblocks):
        stop = B - 1
        rows = []  # per program: (sig_fresh, sig_carried, axc_fresh, axc_carried, sep_ok)
        prevh1 = None
        allsig_ok = True
        sep_dirs = []
        for name, (ctx, ms) in progs.items():
            X = fwd(probe, ctx, stop)
            r_fresh = X[ms[0] + 1]
            r_carried = X[ms[1] + 1]
            sig_f = round(float(r_fresh[L1H1 + 2]), 2)
            sig_c = round(float(r_carried[L1H1 + 2]), 2)
            axc_f = round(axc_sum(r_fresh), 1)
            axc_c = round(axc_sum(r_carried), 1)
            if name == "654+114(b1=2)":
                prevh1 = h1_band(X[ms[0] + 1])  # prev fresh predictor = K target
            if not (sig_f > 0.7 and sig_c > 0.7):
                allsig_ok = False
            sep_dirs.append((axc_f, axc_c))
            rows.append((name, sig_f, sig_c, axc_f, axc_c))
        # Separability: is fresh AX_CARRY consistently distinct from carried?
        fresh_vals = [a for a, c in sep_dirs]
        carried_vals = [c for a, c in sep_dirs]
        fmin, fmax = min(fresh_vals), max(fresh_vals)
        cmin, cmax = min(carried_vals), max(carried_vals)
        separable = (fmax < cmin - 50) or (cmax < fmin - 50)
        tag = []
        if allsig_ok:
            tag.append("SIG_OK")
        if separable:
            tag.append("SEP_OK")
        flag = "  <== " + "+".join(tag) if (allsig_ok and separable) else ""
        print(f"block {B:2d} (read after {stop:2d}): "
              f"fresh_axc[{fmin:.0f},{fmax:.0f}] carried_axc[{cmin:.0f},{cmax:.0f}]"
              f" sigOK={allsig_ok} sep={separable}{flag}")
        if allsig_ok and separable:
            for name, sf, sc, af, ac in rows:
                print(f"      {name:16s} sig f/c={sf}/{sc}  axc f/c={af}/{ac}")
            print(f"      prevH1(K target, 654+114)={prevh1}")


if __name__ == "__main__":
    main()
