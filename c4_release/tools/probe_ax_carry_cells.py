#!/usr/bin/env python3
"""Dump the per-cell AX_CARRY_LO/HI activation at the AX byte-1 predictor row
for carry vs SHL vs JMP — is it a one-hot nibble band (so band_range_check on
the CELL INDEX works) or a distributed value?

Run: CUDA_VISIBLE_DEVICES=0 C4_AX_BYTE1_DUMP=1 python tools/probe_ax_carry_cells.py
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
    last_blk = len(model.blocks) - 1
    AXC = _REG.slots["AX_CARRY_LO"].start
    AXCH = _REG.slots["AX_CARRY_HI"].start
    RAX = int(Token.REG_AX)

    from src.compiler import compile_c
    add_bc, _ = compile_c("int main() { return 654 + 114; }")
    # axi = None means "all AX byte-1 rows"
    progs = {
        "add_carry": (add_bc, 4, None),
        "shl_small": (make_bc([(Opcode.IMM, 21), Opcode.PSH, (Opcode.IMM, 1), Opcode.SHL, Opcode.EXIT]), 20, None),
        "or_basic": (make_bc([(Opcode.IMM, 12), Opcode.PSH, (Opcode.IMM, 10), Opcode.OR, Opcode.EXIT]), 20, None),
        "si_li": (make_bc([(Opcode.IMM, 0x1000), Opcode.PSH, (Opcode.IMM, 42), (Opcode.SI, 0), (Opcode.LI, 0), Opcode.EXIT]), 25, None),
        "jmp_forward": (make_bc([(Opcode.JMP, 2), (Opcode.IMM, 99), (Opcode.IMM, 42), Opcode.EXIT]), 15, None),
    }
    for label, (bc, ms, axi) in progs.items():
        trace = probe.probe(bc, max_steps=ms)
        ax_pos = [p for p in sorted(trace) if trace[p]["token"] == RAX]
        idxs = range(len(ax_pos)) if axi is None else [axi]
        for ai in idxs:
            if ai >= len(ax_pos):
                continue
            pos = ax_pos[ai] + 1
            res, ctxlen = residual_full(probe, bc, last_blk, pos, ms)
            if pos >= ctxlen:
                continue
            c2 = float(res[AXC + 2]) + float(res[AXCH + 2])
            s = float(sum(res[AXC + k] for k in range(16)) + sum(res[AXCH + k] for k in range(16)))
            print(f"{label:14s} AX[{ai}] pos {pos:4d}  Σ={s:9.2f}  cell2(LO+HI)={c2:7.2f}  "
                  f"LO+2={float(res[AXC+2]):.2f} HI+2={float(res[AXCH+2]):.2f}")


if __name__ == "__main__":
    main()
