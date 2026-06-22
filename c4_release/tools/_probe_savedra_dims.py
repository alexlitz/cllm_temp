#!/usr/bin/env python3
"""At a given step's marker/store rows, decode EVERY byte-valued dim family so we
can find where JSR_PC+8 (=90) is recoverable in the campaign frame, vs golden.

Usage: python tools/_probe_savedra_dims.py <id> <step> [maxsteps] [block]
  block: stop_after_block (default = L16-1). 'last' = full stack.
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

BYTE_FAMILIES = [
    ("CLEAN_EMBED_LO","CLEAN_EMBED_HI"), ("EMBED_LO","EMBED_HI"),
    ("OPCODE_BYTE_LO","OPCODE_BYTE_HI"), ("FETCH_LO","FETCH_HI"),
    ("ALU_LO","ALU_HI"), ("OUTPUT_LO","OUTPUT_HI"),
    ("OUTPUT_LO","OUTPUT_HI_THIS_STEP"),
    ("ADDR_B0_LO","ADDR_B0_HI"), ("ADDR_B1_LO","ADDR_B1_HI"),
    ("AX_CARRY_LO","AX_CARRY_HI"), ("PC_BYTE_LO","PC_BYTE_HI"),
    ("PC_PREV_LO","PC_PREV_HI"), ("TEMP_LO","TEMP_HI"),
    ("LOOKAHEAD_PC_LO","LOOKAHEAD_PC_HI"), ("AX_FULL_LO","AX_FULL_HI"),
    ("EMBED_LO.*.-1","EMBED_HI.*.-1"),
]

@torch.no_grad()
def main():
    pid = int(sys.argv[1]); step = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    blkarg = sys.argv[4] if len(sys.argv) > 4 else None
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    L16 = [i for i, b in enumerate(model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    if blkarg == "last":
        stop = len(model.blocks) - 1
    elif blkarg is not None:
        stop = int(blkarg)
    else:
        stop = L16 - 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    xin = model.forward(padded, stop_after_block=stop)[0].float()
    def hot(p, base, n=16):
        seg = xin[p, base:base + n]
        return int(torch.argmax(seg).item()), float(seg.max().item())
    rstart = pl + step * int(Token.STEP_TOKENS)
    rend = rstart + int(Token.STEP_TOKENS)
    print(f"id{pid} {desc} step={step} rows {rstart}..{rend} stop_block={stop} STEP_TOKENS={int(Token.STEP_TOKENS)}")
    # markers per row
    MK = ["MARK_PC","MARK_AX","MARK_SP","MARK_BP","MARK_STACK0","MARK_MEM",
          "MEM_STORE","IS_BYTE","HAS_SE","OP_JSR","OP_LEV","OP_ENT","OP_PSH"]
    for p in range(rstart, min(rend, len(ctx))):
        mk = [f"{m}={xin[p,dimp[m]].item():.1f}" for m in MK if m in dimp and abs(xin[p,dimp[m]].item())>0.4]
        bvals = []
        for lo, hi in BYTE_FAMILIES:
            if lo in dimp and hi in dimp:
                li, lv = hot(p, dimp[lo]); hi_i, hv = hot(p, dimp[hi])
                if lv > 0.4 or hv > 0.4:
                    bv = li | (hi_i << 4)
                    bvals.append(f"{lo.replace('_LO','')}=0x{bv:02x}({lv:.1f}/{hv:.1f})")
        print(f" row{p:>4} tok={ctx[p]:>4} | {' '.join(mk)}")
        if bvals:
            print(f"        {' '.join(bvals)}")

if __name__ == "__main__":
    main()
