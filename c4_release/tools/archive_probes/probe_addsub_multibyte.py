#!/usr/bin/env python3
"""Probe the ADD and no-borrow SUB multi-byte cascade rows (spec_k=0).

Builds the actual 1096-style programs (IMM a, PSH, IMM b, OP, EXIT) and
dumps the cascade/predict rows so we can see, at the SUB/ADD byte-1 emit
row, what the relayed STACK0_BYTE_VAL_h band carries, what OUTPUT carries,
and whether the high byte is emitted.

Usage:
    python tools/probe_addsub_multibyte.py [name ...] [b=BLK ...] [r=ROW ...]
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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


def _prog(a, b, op):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), op, Opcode.EXIT])


PROGRAMS = {
    # no-borrow SUB: 827 - 26 = 801 = 0x321. byte0: 0x3B-0x1A=0x21 no borrow.
    "sub_noborrow": (_prog(827, 26, Opcode.SUB), 801),
    # borrow SUB (passes today): 1537 - 87 = 1450 = 0x5AA. byte0: 0x01-0x57 borrow.
    "sub_borrow1096": (_prog(1537, 87, Opcode.SUB), 1450),
    # ADD multi-byte: 654 + 114 = 768 = 0x300. byte0: 0x8E+0x72=0x100 carry.
    "add_carry": (_prog(654, 114, Opcode.ADD), 768),
    # ADD multi-byte no-carry: 281 + 250 = 531 = 0x213. byte0: 0x19+0xFA=0x113 carry.
    "add_2": (_prog(281, 250, Opcode.ADD), 531),
    # ADD: 228 + 142 = 370 = 0x172. byte0: 0xE4+0x8E=0x172 carry.
    "add_3": (_prog(228, 142, Opcode.ADD), 370),
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
    base = dp.get(name.split("+")[0])
    if base is None:
        return None
    off = int(name.split("+")[1]) if "+" in name else 0
    return round(float(row[base + off].item()), 1)


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
        print(f"=== {pname} exp={expected:#x} REG_AX@{ax_idx} S={S} ===")
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
                sv2 = db(band(row, dp, "STACK0_BYTE_VAL_2_LO"),
                         band(row, dp, "STACK0_BYTE_VAL_2_HI"))
                carry = [sc(row, dp, "CARRY"), sc(row, dp, "CARRY+1"),
                         sc(row, dp, "CARRY+2"), sc(row, dp, "CARRY+3")]
                print(f"  r{r} BI={bi} t8={sc(row,dp,'TEMP+8')} "
                      f"t9={sc(row,dp,'TEMP+9')} OUT={o} SV1={sv1} SV2={sv2} "
                      f"C={carry} OP_ADD={sc(row,dp,'OP_ADD')} OP_SUB={sc(row,dp,'OP_SUB')}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")] or [12, 14, 16, 18, 20]
    rows = [int(a[2:]) for a in args if a.startswith("r=")] or list(range(153, 162))
    sel = [a for a in args if "=" not in a] or ["sub_noborrow", "add_carry"]
    main(sel, blks, rows)
