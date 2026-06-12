#!/usr/bin/env python3
"""Row-by-row dump of the SUB step to map cascade byte_idx -> row.

Dumps rows 150-160 across blocks for sub_16bit: the byte_index flags,
TEMP+8/9 (cascade SUB selector), CARRY band, OUTPUT byte, and
STACK0_BYTE_VAL_1, so we can see exactly which rows the cascade byte-1/2
SUB rules fire on (BYTE_INDEX_1/2 + TEMP+9) and whether the relayed
high byte must land there.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


PROGRAMS = {
    "sub_16bit":  (_mk([(Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFF),
    "sub_borrow": (_mk([(Opcode.IMM, 0), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFFFFFFFF),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    return [float(row[base + i].item()) for i in range(width)] if base is not None else None


def nibble(cells):
    if cells is None:
        return None
    mx = max(range(len(cells)), key=lambda i: cells[i])
    return mx if cells[mx] > 0.3 else None


def db(lo, hi):
    nl, nh = nibble(lo), nibble(hi)
    return None if nl is None and nh is None else ((nh or 0) << 4) | (nl or 0)


def sc(row, dp, name):
    base = dp.get(name)
    return round(float(row[base].item()), 1) if base is not None else None


def main(selected, blks, rows):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()
    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        ax_idx = max(i for i in range(S) if ctx[i] == Token.REG_AX)
        print(f"=== {pname} exp={expected:#x} REG_AX@{ax_idx} ===")
        for blk in blks:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (L{lg}) --")
            for r in rows:
                row = resid[r]
                bi = [sc(row, dp, f"BYTE_INDEX_{i}") for i in range(3)]
                o = db(band(row, dp, "OUTPUT_LO"),
                       band(row, dp, "OUTPUT_HI_THIS_STEP") or band(row, dp, "OUTPUT_HI"))
                sv1 = db(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                         band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                carry = [sc(row, dp, "CARRY"), sc(row, dp, "CARRY+1"),
                         sc(row, dp, "CARRY+2"), sc(row, dp, "CARRY+3")]
                print(f"  r{r} BI={bi} t8={sc(row,dp,'TEMP+8')} "
                      f"t9={sc(row,dp,'TEMP+9')} OUT={o} SV1={sv1} CARRY={carry} "
                      f"OP_SUB={sc(row,dp,'OP_SUB')}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")] or [16, 17, 18]
    rows = [int(a[2:]) for a in args if a.startswith("r=")] or list(range(153, 160))
    sel = [a for a in args if not a[1:2] == "="] or ["sub_16bit"]
    main(sel, blks, rows)
