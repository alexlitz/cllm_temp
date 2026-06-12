#!/usr/bin/env python3
"""Trace the PSH store of a multi-byte value into the STACK0 byte band.

For sub_16bit (IMM 0x100; PSH; IMM 1; SUB; EXIT) we want to find:
  - which sequence rows carry STACK0_BYTE0/1/2/3 position flags after PSH,
  - whether the pushed value byte-1 (0x01) is present in CLEAN_EMBED /
    OUTPUT / ALU / MEM_VAL at those rows, and at which block it appears,
  - where the byte-1 value is dropped before the SUB borrow loop reads it.

Compare against sub_basic (50-8, byte1=0x00) and sub_borrow (0-1, all 0xFF).
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    "sub_basic":  (_mk([(Opcode.IMM, 50), Opcode.PSH, (Opcode.IMM, 8), Opcode.SUB, Opcode.EXIT]), 42),
    "sub_16bit":  (_mk([(Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFF),
    "sub_borrow": (_mk([(Opcode.IMM, 0), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]), 0xFFFFFFFF),
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def hot(cells, thr=0.4):
    if cells is None:
        return []
    return [(i, round(v, 1)) for i, v in enumerate(cells) if abs(v) > thr]


def scal(row, dp, name):
    base = dp.get(name)
    if base is None:
        return None
    return round(float(row[base].item()), 2)


def main(selected, blks, rows_arg):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        # locate the PSH'd-value rows: scan all rows, print marker scalars at
        # the final (post-decode) residual so we can see which rows carry
        # MARK_STACK0 / STACK0_BYTE flags.
        print(f"=== {pname} expected={expected:#x} S={S} ctx={ctx} ===")
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        for blk in blks:
            lg = blmap[blk]["logical"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            print(f" -- block {blk} (logical L{lg}) --")
            if rows_arg:
                rows = rows_arg
            else:
                # auto: rows where any STACK0_BYTE flag is hot
                rows = []
                for r in range(S):
                    row = resid[r]
                    if (abs(scal(row, dp, "STACK0_BYTE0") or 0) > 0.4 or
                        abs(scal(row, dp, "STACK0_BYTE1") or 0) > 0.4 or
                        abs(scal(row, dp, "STACK0_BYTE2") or 0) > 0.4 or
                        abs(scal(row, dp, "MARK_STACK0") or 0) > 0.4):
                        rows.append(r)
            for r in rows:
                row = resid[r]
                sb = [scal(row, dp, f"STACK0_BYTE{i}") for i in range(4)]
                flags = (f"mST={scal(row,dp,'MARK_STACK0')} mSP={scal(row,dp,'MARK_SP')} "
                         f"mStore={scal(row,dp,'MEM_STORE')} isB={scal(row,dp,'IS_BYTE')} "
                         f"S0B={sb}")
                clo = hot(band(row, dp, "CLEAN_EMBED_LO"))
                chi = hot(band(row, dp, "CLEAN_EMBED_HI"))
                olo = hot(band(row, dp, "OUTPUT_LO"))
                ohi = hot(band(row, dp, "OUTPUT_HI"))
                alo = hot(band(row, dp, "ALU_LO"))
                mv = [scal(row, dp, f"MEM_VAL_B{i}") for i in range(4)]
                print(f"  row{r:3d} {flags}")
                print(f"        CLEAN_LO={clo} CLEAN_HI={chi}")
                print(f"        OUT_LO={olo} OUT_HI={ohi} ALU_LO={alo} MEM_VAL={mv}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")]
    rows = [int(a[2:]) for a in args if a.startswith("r=")]
    sel = [a for a in args if not a.startswith("b=") and not a.startswith("r=")] or ["sub_16bit"]
    if not blks:
        blks = [11]
    main(sel, blks, rows)
