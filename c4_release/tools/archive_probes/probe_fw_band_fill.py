#!/usr/bin/env python3
"""Self-consistent: does AX_BYTE1_FULL_WIDE fill + win the LM logit?

Builds the model the GroundTruthProbe uses, then resolves the band from a fresh
matching layout, reads the wide band + LM logits at the byte-1 predictor row.

Run: CUDA_VISIBLE_DEVICES=0 C4_AX_BYTE1_FULL_WIDTH=1 python tools/probe_fw_band_fill.py
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
from neural_vm.unified_compiler.full_vm_compiler_dynamic import (  # noqa: E402
    compile_full_vm_dynamic)


def main():
    # Resolve the band from a build matching the runner's flag state.
    _m, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = getattr(layout, "dim_positions", layout)
    band = dp.get("AX_BYTE1_FULL_WIDE")
    ALU_LO, ALU_HI, ISB = dp["ALU_LO"], dp["ALU_HI"], dp["IS_BYTE"]
    print(f"AX_BYTE1_FULL_WIDE band = {band}", flush=True)

    probe = GroundTruthProbe.build()
    model = probe.model
    last = len(model.blocks) - 1
    RAX = int(Token.REG_AX)
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    bias = model.head.bias

    for val in (9257, 5561, 50007):
        bc, _ = compile_c(f"int main() {{ return {val}; }}")
        ctx = probe._final_context(bc, max_steps=3)
        trace = probe.probe(bc, max_steps=3)
        m = [p for p in sorted(trace) if trace[p]["token"] == RAX][-1]
        padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
        with torch.no_grad():
            x = model.forward(padded, stop_after_block=last)
        row = x[0, m + 1].float().cpu()
        exp_b1 = (val >> 8) & 0xFF
        print(f"\n=== return {val} (byte1=0x{exp_b1:02x}) ===")
        print(f"  ALU_LO am={int(row[ALU_LO:ALU_LO+16].argmax())} "
              f"ALU_HI am={int(row[ALU_HI:ALU_HI+16].argmax())} "
              f"IS_BYTE={float(row[ISB]):.2f}")
        if band is not None:
            nz = [(j, round(float(row[band + j]), 2)) for j in range(256)
                  if abs(float(row[band + j])) > 1e-2]
            print(f"  wide-band nonzero: {nz[:10]}")
        res = row.to(W.device)
        logits = ((W @ res) + bias).float().cpu()
        top = torch.argsort(logits, descending=True)[:5].tolist()
        print("  top: " + ", ".join(
            f"0x{t:02x}:{float(logits[t]):.1f}" for t in top))
        print(f"  logit[exp 0x{exp_b1:02x}]={float(logits[exp_b1]):.1f}  "
              f"logit[alias 0x{exp_b1 & 0xF:02x}]={float(logits[exp_b1 & 0xF]):.1f}")


if __name__ == "__main__":
    main()
