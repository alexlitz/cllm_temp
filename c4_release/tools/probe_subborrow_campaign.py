#!/usr/bin/env python3
"""Campaign sub_borrow_cascade byte-1/2/3 trace.

Run with the campaign env (C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1).
Dumps, for 0-1 (=0xFFFFFFFF) and 0x100-1 (=0xFF) at the SUB byte rows,
the OUTPUT byte / STACK0_BYTE_VAL_1 / CARRY band across every block, so we
can see (a) which row is the byte-1 predictor, (b) where the empty band sits,
(c) which block slams OUTPUT_HI.
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
    "sub_200":    (_mk([(Opcode.IMM, 0x200), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0x1FF),
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


def _resolve(dp, name):
    if "+" in name:
        b, o = name.rsplit("+", 1)
        base = dp.get(b)
        return base + int(o) if base is not None else None
    return dp.get(name)


def sc(row, dp, name):
    base = _resolve(dp, name)
    return round(float(row[base].item()), 1) if base is not None else None


def main(selected, blks, rows):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()
    nblocks = len(blmap)
    print(f"NBLOCKS={nblocks}")
    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        ax_idx = max(i for i in range(S) if ctx[i] == Token.REG_AX)
        print(f"=== {pname} exp={expected:#x} S={S} REG_AX@{ax_idx} ===")
        for blk in blks:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (L{lg}) --")
            for r in rows:
                if r >= S:
                    continue
                row = resid[r]
                bi = [sc(row, dp, f"BYTE_INDEX_{i}") for i in range(4)]
                ohi_lo = band(row, dp, "OUTPUT_HI_THIS_STEP") or band(row, dp, "OUTPUT_HI")
                o = db(band(row, dp, "OUTPUT_LO"), ohi_lo)
                ol = band(row, dp, "OUTPUT_LO")
                oh = ohi_lo
                sv1 = db(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                         band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                sv1lo = band(row, dp, "STACK0_BYTE_VAL_1_LO")
                carry = [sc(row, dp, "CARRY"), sc(row, dp, "CARRY+1"),
                         sc(row, dp, "CARRY+2"), sc(row, dp, "CARRY+3")]
                # raw OUTPUT cell-0 and max for slam detection
                ol0 = round(ol[0], 1) if ol else None
                olmax = round(max(ol), 1) if ol else None
                oh0 = round(oh[0], 1) if oh else None
                ohmax = round(max(oh), 1) if oh else None
                sv1max = round(max(sv1lo), 1) if sv1lo else None
                print(f"  r{r} BI={bi} t8={sc(row,dp,'TEMP+8')} "
                      f"t9={sc(row,dp,'TEMP+9')} OUT={o} OL0={ol0} OLmx={olmax} "
                      f"OH0={oh0} OHmx={ohmax} SV1={sv1} SV1mx={sv1max} "
                      f"CARRY={carry} OP_SUB={sc(row,dp,'OP_SUB')}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks_arg = [a for a in args if a.startswith("b=")]
    if blks_arg:
        blks = []
        for a in blks_arg:
            spec = a[2:]
            if "-" in spec:
                lo, hi = spec.split("-")
                blks.extend(range(int(lo), int(hi) + 1))
            else:
                blks.append(int(spec))
    else:
        blks = list(range(0, 55))
    rows = [int(a[2:]) for a in args if a.startswith("r=")] or list(range(0, 30))
    sel = [a for a in args if "=" not in a] or ["sub_borrow"]
    main(sel, blks, rows)
