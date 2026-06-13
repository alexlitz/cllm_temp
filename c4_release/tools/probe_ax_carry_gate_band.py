#!/usr/bin/env python3
"""Measure Σ AX_CARRY at the dump read point (last block, AX byte-1 predictor
rows) for the carry program vs the SHL / JMP over-fire programs.

Goal: pick the two-sided band-pass thresholds [lo, hi] so the dump gate fires
ONLY in the carry band (~[2,5]) and excludes SHL (~+12.8) / JMP (~+47.9).

Run: CUDA_VISIBLE_DEVICES=0 C4_AX_BYTE1_DUMP=1 python tools/probe_ax_carry_gate_band.py
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
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from neural_vm.embedding import Opcode

_REG = build_default_registry_dynamic()


@torch.no_grad()
def residual_full(probe, bc, block_idx, position, max_steps):
    ctx = probe._final_context(bc, max_steps=max_steps)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    row = x[0, position]
    return (row.to_dense().cpu() if row.is_sparse else row.cpu()), len(ctx)


def make_bc(instrs):
    from tests.test_smoke import _make_bytecode
    return _make_bytecode(instrs)


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    nblocks = len(model.blocks)
    last_blk = nblocks - 1
    AXC = _REG.slots["AX_CARRY_LO"].start
    AXCH = _REG.slots["AX_CARRY_HI"].start
    ADDR = _REG.slots["ADDR_B0_LO"].start
    AXREG = _REG.slots["ADDR_B1_HI"].start
    RAX = int(Token.REG_AX)

    progs = {
        "add_654_114(carry)": (None, 4),  # special: compiled below
        "shl_8bit(1<<8=256)": (make_bc([
            (Opcode.IMM, 1), Opcode.PSH, (Opcode.IMM, 8), Opcode.SHL, Opcode.EXIT,
        ]), 20),
        "shl_small(21<<1=42)": (make_bc([
            (Opcode.IMM, 21), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHL, Opcode.EXIT,
        ]), 20),
        "shr_small(84>>1=42)": (make_bc([
            (Opcode.IMM, 84), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHR, Opcode.EXIT,
        ]), 20),
        "jmp_forward": (make_bc([
            (Opcode.JMP, 2), (Opcode.IMM, 99), (Opcode.IMM, 42), Opcode.EXIT,
        ]), 15),
    }
    from src.compiler import compile_c
    add_bc, _ = compile_c("int main() { return 654 + 114; }")
    progs["add_654_114(carry)"] = (add_bc, 4)

    for label, (bc, ms) in progs.items():
        trace = probe.probe(bc, max_steps=ms)
        ax_pos = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        print(f"\n##### {label}  AX markers={ax_pos}")
        for mi, mp in enumerate(ax_pos):
            pos = mp + 1  # byte-1 predictor row
            res, ctxlen = residual_full(probe, bc, last_blk, pos, ms)
            if pos >= ctxlen:
                continue
            axc = float(sum(res[AXC + j] for j in range(16)) +
                        sum(res[AXCH + j] for j in range(16)))
            sig = float(res[ADDR + 5])
            axr = float(res[AXREG + 8])
            print(f"  AX[{mi}] pos {pos}: Σ AX_CARRY = {axc:8.2f}  "
                  f"ADDR_B0_LO+5={sig:.3f}  ADDR_B1_HI+8={axr:.3f}")


if __name__ == "__main__":
    main()
