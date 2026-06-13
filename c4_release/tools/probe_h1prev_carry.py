#!/usr/bin/env python3
"""Probe the H1_PREV_STEP carry band + AX_CARRY gate at the tail (spec_k=0).

Confirms (a) the carry head copies the prev step's H1 one-hot into
H1_PREV_STEP at the carried (PSH) step's byte-1 predictor row, and (b) the
AX_CARRY fresh-vs-carried separation at the final block (the dump FFN read
point). H1_PREV_STEP / H1_DUMP_OUT are NEW bands -> resolve via the built
layout's dim_positions, not the legacy registry.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_h1prev_carry.py
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


@torch.no_grad()
def residual_full(probe, bc, block_idx, position, max_steps):
    ctx = probe._final_context(bc, max_steps=max_steps)
    if position < 0:
        position = len(ctx) + position
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0, position].to_dense().cpu() if x[0, position].is_sparse else x[0, position].cpu()


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    # Match the GroundTruthProbe's EFFICIENT-ALU model (trust_neural_alu=True ->
    # alu_mode='efficient'); the lookup-mode layout has different band positions.
    _, layout = compile_full_vm_dynamic(disk_cache=True, alu_mode="efficient")
    dp = layout.dim_positions
    assert layout.d_model == model.head.weight.shape[1], (
        f"layout d_model {layout.d_model} != model d_model "
        f"{model.head.weight.shape[1]} — probe layout/model mismatch"
    )
    H1 = _REG.slots["H1"].start
    H1P = dp["H1_PREV_STEP"]
    H1D = dp.get("H1_DUMP_OUT")
    AXC = _REG.slots["AX_CARRY_LO"].start
    AXCH = _REG.slots["AX_CARRY_HI"].start
    nblocks = len(model.blocks)
    print(f"d_model={layout.d_model} n_blocks={nblocks} H1={H1} H1_PREV_STEP={H1P} H1_DUMP_OUT={H1D} AX_CARRY_LO={AXC}")

    bc, _ = compile_c("int main() { return 654 + 114; }")  # AX byte1=0x02
    trace = probe.probe(bc, max_steps=4)
    RAX = int(Token.REG_AX)
    ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]
    print("AX marker positions:", ms)
    # byte-1 predictor row = AX marker + 1 (the AX byte-0 token row)
    last_blk = nblocks - 1
    for label, mi in (("IMM(fresh)", 0), ("PSH(carried)", 1)):
        if mi >= len(ms):
            continue
        pos = ms[mi] + 1
        res = residual_full(probe, bc, last_blk, pos, 4)
        h1 = [round(float(res[H1 + j]), 2) for j in range(7)]
        h1p = [round(float(res[H1P + j]), 2) for j in range(7)]
        h1d = [round(float(res[H1D + j]), 2) for j in range(7)] if H1D else None
        axc_sum = float(sum(res[AXC + j] for j in range(16)) + sum(res[AXCH + j] for j in range(16)))
        print(f"\n=== {label} byte-1 predictor pos {pos} (last block {last_blk}) ===")
        print(f"  H1          {h1}")
        print(f"  H1_PREV_STEP{h1p}")
        if h1d is not None:
            print(f"  H1_DUMP_OUT {h1d}")
        print(f"  sum(AX_CARRY) = {axc_sum:.2f}")
        ADDR = _REG.slots["ADDR_B0_LO"].start
        print(f"  ADDR_B0_LO+5 = {float(res[ADDR + 5]):.3f}")
        for mk in ("MARK_AX", "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM", "MARK_SE"):
            mp = _REG.slots[mk].start
            print(f"  {mk} = {float(res[mp]):.3f}", end="  ")
        print()


if __name__ == "__main__":
    main()
