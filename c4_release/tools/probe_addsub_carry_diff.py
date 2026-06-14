#!/usr/bin/env python3
"""Compare the L8 AddSub block-10 CARRY/OUTPUT output: declarative vs imperative.

Set C4_ADDSUB_DECLARATIVE before running (0 = imperative, default = declarative).
Dumps, at the OP MARK_AX compute row, the OUTPUT_LO/HI + CARRY[0..3] + AX_FULL
bands AFTER the AddSub block (block 10), so we can see what the wrap emits for
the downstream multi-byte CarryPropagation + byte-1 relay.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
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
    "sub_basic": (_mk([(Opcode.IMM, 5), Opcode.PSH, (Opcode.IMM, 6),
                       Opcode.SUB, Opcode.EXIT]), -1 & 0xFFFFFFFF),
    "add_16bit": (_mk([(Opcode.IMM, 200), Opcode.PSH, (Opcode.IMM, 100),
                       Opcode.ADD, Opcode.EXIT]), 300),
    "sub_16bit": (_mk([(Opcode.IMM, 300), Opcode.PSH, (Opcode.IMM, 100),
                       Opcode.SUB, Opcode.EXIT]), 200),
}


def hot(row, dp, name, w=16):
    base = dp[name]
    return [(i, round(float(row[base + i].item()), 2))
            for i in range(w) if abs(row[base + i].item()) > 0.3]


def main(sel):
    mode = "DECL" if os.environ.get("C4_ADDSUB_DECLARATIVE", "1") != "0" \
        else "IMPERATIVE"
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()
    addsub_blk = next(i for i, r in enumerate(blmap)
                      if r["is_post_op_expansion"] and r["logical"] == 8)
    print(f"### MODE={mode} addsub_block={addsub_blk} "
          f"ffn={blmap[addsub_blk]['ffn']}")
    for name in sel:
        bc, want = PROGRAMS[name]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        Slen = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(Slen)
                   if emb[r, dp["MARK_AX"]].abs().item() > 0.5]
        op_row = ax_rows[-1]
        print(f"\n== {name} want={want:#x} got={got:#x} op_row={op_row} ==")
        for blk in (addsub_blk - 1, addsub_blk, addsub_blk + 1):
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            row = resid[op_row]
            carry = [round(float(row[dp["CARRY"] + k].item()), 2)
                     for k in range(4)]
            axf = []
            if "AX_FULL_LO" in dp:
                axf = ("AXF_LO=" + str(hot(row, dp, "AX_FULL_LO"))
                       + " AXF_HI=" + str(hot(row, dp, "AX_FULL_HI")))
            print(f"  blk{blk} L{blmap[blk]['logical']} "
                  f"{blmap[blk]['ffn'][:22]}")
            print(f"    OUT_LO={hot(row, dp, 'OUTPUT_LO')} "
                  f"OUT_HI={hot(row, dp, 'OUTPUT_HI')} CARRY={carry}")
            print(f"    {axf}")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if a in PROGRAMS] or ["sub_basic", "add_16bit"]
    main(sel)
