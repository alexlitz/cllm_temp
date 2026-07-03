#!/usr/bin/env python3
"""Probe C4_B1_TO_OUTPUT: does OUTPUT_LO now carry the byte-1 on the carried row?

Builds the model with C4_B1_TO_OUTPUT set to whatever env holds (set it on the
command line), runs add_0 (IMM 654; PSH; IMM 114; ADD -> AX=768, byte-1=3), and
reads OUTPUT_LO/HI + H1_DUMP_OUT at the AX byte-1 predictor row of BOTH the fresh
(IMM) step and the carried (PSH) step, at the LM-head read point (last block).

ALL bands resolve via the BUILT layout dim_positions (NOT the static registry) —
the widen-repack moves dims.

Run:
  CUDA_VISIBLE_DEVICES=0 C4_B1_TO_OUTPUT=1 python tools/probe_b1_to_output.py
  CUDA_VISIBLE_DEVICES=0 C4_B1_TO_OUTPUT=0 python tools/probe_b1_to_output.py
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


@torch.no_grad()
def residual_full(probe, bc, block_idx, position, max_steps):
    ctx = probe._final_context(bc, max_steps=max_steps)
    if position < 0:
        position = len(ctx) + position
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return r.to_dense().cpu() if r.is_sparse else r.cpu()


def decode_byte(res, olo, ohi):
    lo = max(range(16), key=lambda k: float(res[olo + k]))
    hi = max(range(16), key=lambda k: float(res[ohi + k]))
    return (hi << 4) | lo, lo, hi


def main():
    flag = os.environ.get("C4_B1_TO_OUTPUT", "0")
    probe = GroundTruthProbe.build()
    model = probe.model
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    _, layout = compile_full_vm_dynamic(disk_cache=True, alu_mode="efficient")
    dp = layout.dim_positions
    assert layout.d_model == model.head.weight.shape[1], (
        f"layout d_model {layout.d_model} != model d_model "
        f"{model.head.weight.shape[1]}"
    )
    OLO = dp["OUTPUT_LO"]
    OHI = dp["OUTPUT_HI"]
    H1D = dp.get("H1_DUMP_OUT")
    ADDR_B1_HI = dp["ADDR_B1_HI"]
    ADDR_B0_LO = dp["ADDR_B0_LO"]
    nblocks = len(model.blocks)
    print(f"C4_B1_TO_OUTPUT={flag} d_model={layout.d_model} n_blocks={nblocks}")
    print(f"OUTPUT_LO={OLO} OUTPUT_HI={OHI} H1_DUMP_OUT={H1D}")

    # AX = 654 + 114 = 768 = 0x0300 -> byte-0 = 0x00, byte-1 = 0x03.
    bc, _ = compile_c("int main() { return 654 + 114; }")
    trace = probe.probe(bc, max_steps=4)
    RAX = int(Token.REG_AX)
    ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
    print("AX marker positions:", ms, "(expect byte-1 value = 3)")
    last_blk = nblocks - 1
    for label, mi in (("IMM(fresh)", 0), ("PSH(carried)", 1)):
        if mi >= len(ms):
            continue
        # byte-1 predictor row = AX marker + 2 (marker, byte-0 token, byte-1 tok).
        # The dump/H1 emission fires on the byte-1 predictor row; probe marker+1
        # and marker+2 to locate it.
        for off in (1, 2):
            pos = ms[mi] + off
            res = residual_full(probe, bc, last_blk, pos, 4)
            b1hi = float(res[ADDR_B1_HI + 8])
            b0lo = float(res[ADDR_B0_LO + 5])
            byte, lo, hi = decode_byte(res, OLO, OHI)
            olo_top = sorted(range(16), key=lambda k: -float(res[OLO + k]))[:3]
            olo_vals = {k: round(float(res[OLO + k]), 2) for k in olo_top}
            h1d = ([round(float(res[H1D + j]), 2) for j in range(7)]
                   if H1D else None)
            tag = ""
            if b1hi > 2.0 and b0lo > 0.5:
                tag = "  <== AX byte-1 predictor row"
            print(f"\n{label} pos {pos} (marker+{off}){tag}")
            print(f"  ADDR_B1_HI+8={b1hi:.2f} ADDR_B0_LO+5={b0lo:.2f}")
            print(f"  OUTPUT decode -> byte=0x{byte:02x} (lo={lo} hi={hi})")
            print(f"  OUTPUT_LO top3 {olo_vals}")
            if h1d is not None:
                print(f"  H1_DUMP_OUT {h1d}")


if __name__ == "__main__":
    main()
