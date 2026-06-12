#!/usr/bin/env python3
"""Trace the SUB byte-1/2/3 0xFF leak (spec_k=0, hook-free).

For sub_16bit (IMM 0x100; PSH; IMM 1; SUB; EXIT) the emitted AX bytes 1-3 =
0xFF instead of 0x00 -> exit 0xFFFFFFFF instead of 255. Byte 0 (0xFF) is
correct; only bytes 1-3 sign-extend. This walks the physical blocks at the
byte-1/2/3 emit rows reading OUTPUT_LO/HI plus the SUB-relevant dims
(TEMP+8/9 = ADD/SUB relay, CARRY band, ALU_LO/HI) so we can see WHICH tail
corrector should fire (and why it is blocked) for byte = 0x00.

sub_basic (50-8=42, no borrow, PASSES) and sub_borrow_cascade (0-1=0xFFFFFFFF,
genuine all-0xFF, PASSES) are the mutual-exclusion partners: any byte-1=0x00
fix MUST keep sub_borrow_cascade at 0xFF.
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
    # 50 - 8 = 42 = 0x2A, no borrow, bytes 1-3 = 0x00 (PASSES)
    "sub_basic":  (_mk([(Opcode.IMM, 50), Opcode.PSH, (Opcode.IMM, 8), Opcode.SUB, Opcode.EXIT]), 42),
    # 0x100 - 1 = 0xFF, borrow out of byte0 but byte1 = 0x00 net (FAILS -> 0xFFFFFFFF)
    "sub_16bit":  (_mk([(Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFF),
    # 0 - 1 = 0xFFFFFFFF, genuine all-0xFF underflow (PASSES)
    "sub_borrow": (_mk([(Opcode.IMM, 0), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFFFFFFFF),
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


def scalar(row, dp, name):
    base = dp.get(name)
    if base is None:
        return None
    return round(float(row[base].item()), 2)


def carryband(row, dp, width=4):
    base = dp.get("CARRY")
    if base is None:
        return None
    return [round(float(row[base + i].item()), 2) for i in range(width)]


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
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        ax_idx = None
        for i in range(S - 1, -1, -1):
            if ctx[i] == Token.REG_AX:
                ax_idx = i
                break
        b0 = ax_idx + 1
        b1, b2, b3 = ax_idx + 2, ax_idx + 3, ax_idx + 4
        print(f"=== {pname} expected={expected:#010x} got={got:#010x} S={S} "
              f"REG_AX@{ax_idx} b0@{b0} b1@{b1} b2@{b2} b3@{b3} ===")
        print(f"  emitted bytes b0={ctx[b0]:#04x} b1={ctx[b1]:#04x} "
              f"b2={ctx[b2]:#04x} b3={ctx[b3]:#04x}")
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        # Rows whose next-token logits PREDICT bytes 0..3 are b0-1..b3-1.
        watch = sorted(set([b0 - 1, b1 - 1, b2 - 1, b3 - 1]))
        for blk in blk_arg:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (logical L{lg}) --")
            for r in watch:
                row = resid[r]
                olo = band(row, dp, "OUTPUT_LO")
                ohi = band(row, dp, "OUTPUT_HI_THIS_STEP")
                if ohi is None:
                    ohi = band(row, dp, "OUTPUT_HI")
                nlo, nhi = nibble(olo), nibble(ohi)
                byteval = (nhi << 4 | nlo) if (nlo is not None and nhi is not None) else None
                bvs = f"{byteval:#04x}" if byteval is not None else "None"
                which = {b0 - 1: "->b0", b1 - 1: "->b1", b2 - 1: "->b2", b3 - 1: "->b3"}[r]
                alo = band(row, dp, "ALU_LO")
                print(f"  row{r:3d}{which} O_byte={bvs} t8={scalar(row,dp,'TEMP+8')} "
                      f"t9={scalar(row,dp,'TEMP+9')} CARRY={carryband(row,dp)} "
                      f"O_LO={hot(olo)} O_HI={hot(ohi)} ALU_LO={hot(alo)}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    sel = [a for a in args if not a.startswith("b=")] or ["sub_16bit"]
    if not blks:
        blks = [33, 34, 35, 36]
    main(sel, blks)
