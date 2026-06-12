#!/usr/bin/env python3
"""Trace the ADD byte-1 0x88 leak (spec_k=0, hook-free).

For add_basic (IMM 10; PSH; IMM 32; ADD; EXIT) the emitted AX byte 1 = 0x88
instead of 0x00. This walks EVERY physical block at the byte-1 emit row and at
the MARK_AX compute row, reading OUTPUT_LO/HI (the band that decodes to the
emitted byte) so we can see WHICH block first writes the 0x88 (= LO nibble 8,
HI nibble 8).
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
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGRAMS = {
    "add_basic":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 32), Opcode.ADD, Opcode.EXIT]), 42),
    "add_16bit":  (_mk([(Opcode.IMM, 200), Opcode.PSH, (Opcode.IMM, 100), Opcode.ADD, Opcode.EXIT]), 300),
    "add_carry":  (_mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 1), Opcode.ADD, Opcode.EXIT]), 0x100),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, round(v, 1)) for i, v in enumerate(cells) if abs(v) > thr]


def nibble(cells):
    if cells is None:
        return None
    mx = max(range(len(cells)), key=lambda i: cells[i])
    return mx if cells[mx] > 0.3 else None


def main(selected, blk_arg):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        # Find last REG_AX and the byte emit positions after it.
        ax_idx = None
        for i in range(S - 1, -1, -1):
            if ctx[i] == Token.REG_AX:
                ax_idx = i
                break
        b0 = ax_idx + 1  # byte0 emit pos
        b1 = ax_idx + 2  # byte1 emit pos
        print(f"=== {pname} expected={expected:#06x} got={got:#06x} S={S} "
              f"REG_AX@{ax_idx} byte-emit b0@{b0} b1@{b1} ===")
        print(f"  ctx tail: {ctx[max(0,ax_idx-2):ax_idx+6]}")
        print(f"  emitted bytes b0={ctx[b0]:#04x} b1={ctx[b1]:#04x} "
              f"b2={ctx[b1+1]:#04x} b3={ctx[b1+2]:#04x}")
        # MARK_AX rows
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_base = dp["MARK_AX"]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        print(f"  MARK_AX rows={ax_rows}")
        # Walk blocks at the byte-1 emit row (position b1-1 is the row whose
        # logits PREDICT token at b1; but to read the residual that produced
        # the emitted byte at b1 we look at row b1-1 -> next-token).  The
        # emitted byte AT b1 was predicted from the residual at row (b1-1).
        # We trace BOTH the byte-1 prediction row and the MARK_AX rows.
        watch = sorted(set([b0 - 1, b1 - 1, b1] + ax_rows))
        for blk in blk_arg:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (logical L{lg}) --")
            for r in watch:
                row = resid[r]
                olo = band(row, dp, "OUTPUT_LO")
                ohi = band(row, dp, "OUTPUT_HI")
                nlo, nhi = nibble(olo), nibble(ohi)
                byteval = (nhi << 4 | nlo) if (nlo is not None and nhi is not None) else None
                bvs = f"{byteval:#04x}" if byteval is not None else "None"
                bi = [round(float(row[dp[f'BYTE_INDEX_{k}']].item()), 1) for k in range(4)]
                print(f"  row{r:3d} BI={bi} O_byte={bvs}  O_LO={hot(olo)} O_HI={hot(ohi)}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    sel = [a for a in args if not a.startswith("b=")] or ["add_basic"]
    if not blks:
        blks = list(range(37))
    main(sel, blks)
