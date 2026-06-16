#!/usr/bin/env python3
"""Find the byte-1 high-nibble carrier using BUILT layout positions.

The widen repack moves dims, so the static registry mislabels cells. This scan
resolves EVERY 16-wide band from compile_full_vm_dynamic()[1].dim_positions and
finds the band whose (lo,hi) nibble argmax tracks byte-1 at the predictor row,
across 4 literals with distinct high nibbles. Same model for layout + forward.

Run: CUDA_VISIBLE_DEVICES=0 [C4_AX_BYTE1_FULL_WIDTH=1] python tools/probe_byte1_builtscan.py
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
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
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)

# byte1 hi nibble = 0,1,2,3 ; lo nibble = 5 -> byte1 = 0x05,0x15,0x25,0x35
SRCS = {"h0": (1365, 0x05), "h1": (5461, 0x15),
        "h2": (9557, 0x25), "h3": (13653, 0x35)}


def main():
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = getattr(layout, "dim_positions", layout)
    reg = build_default_registry_dynamic()
    # 16-wide band names (from the static registry) but resolved at BUILT pos.
    BANDS = [nm for nm, slot in reg.slots.items()
             if slot.size == 16 and nm in dp]
    probe = GroundTruthProbe.build()
    model = probe.model
    last = len(model.blocks) - 1
    RAX = int(Token.REG_AX)
    rows = {}
    markers = {}
    for key, (val, _b1) in SRCS.items():
        bc, _ = compile_c(f"int main() {{ return {val}; }}")
        ctx = probe._final_context(bc, max_steps=3)
        trace = probe.probe(bc, max_steps=3)
        m = [p for p in sorted(trace) if trace[p]["token"] == RAX][-1]
        padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
        with torch.no_grad():
            x = model.forward(padded, stop_after_block=last)
        rows[key] = x[0, m + 1].float().cpu()  # byte-1 predictor row
    print("Scanning BUILT-position bands for the byte-1 HIGH-nibble carrier "
          "(hi: 0,1,2,3)\n", flush=True)
    for band in BANDS:
        base = dp[band]
        am = {}
        strong = True
        for key in SRCS:
            v = rows[key][base:base + 16]
            if float(v.max()) < 0.25:
                strong = False
                break
            am[key] = int(v.argmax())
        if not strong:
            continue
        seq = tuple(am[k] for k in ("h0", "h1", "h2", "h3"))
        # The HIGH nibbles are 0,1,2,3. A carrier's argmax == hi + const.
        diffs = {seq[i] - i for i in range(4)}
        if len(diffs) == 1:  # constant offset -> tracks the high nibble!
            off = diffs.pop()
            print(f"  HI-CARRIER {band:24s} argmax(h0..h3)={seq} "
                  f"(== hi + {off})", flush=True)
        elif len(set(seq)) > 1:
            print(f"  distinguishes {band:24s} argmax={seq}", flush=True)


if __name__ == "__main__":
    main()
