#!/usr/bin/env python3
"""Confirm SP low-byte one-hot source for the SP_ADDR_LO relay.

Dumps OUTPUT_LO (and OUTPUT_HI) one-hot cell at every MARK_SP row, plus the
MEM store value rows + ADD query row, at the L8-attn-block INPUT (block 10
output / block 11 input) so we know what the relay (which must land BEFORE
block 11) can read. The relay copies OUTPUT_LO from the nearest prior MARK_SP.
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def amax(v):
    m = int(v.argmax()); return m, float(v[m])


def main(idx, upto_block):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    OL = int(dp["OUTPUT_LO"]); OH = int(dp["OUTPUT_HI"])
    bc, exp, desc = _corpus(idx)
    ctx = p._final_context(bc, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    out = p.model.forward(padded, stop_after_block=upto_block)[0]
    sp_rows = [i for i, t in enumerate(ctx) if t == Token.REG_SP]
    ax_rows = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
    print(f"id={idx} {desc} exp={exp}  blockInput={upto_block}", flush=True)
    print(f"  MARK_SP rows: {sp_rows}", flush=True)
    print(f"  MARK_AX rows: {ax_rows}", flush=True)
    for r in sp_rows:
        ol = amax(out[r, OL:OL+16]); oh = amax(out[r, OH:OH+16])
        print(f"   SP row {r:>3} tok={ctx[r]:>3}: OUTPUT_LO cell={ol[0]:>2}({ol[1]:+.1f}) OUTPUT_HI cell={oh[0]:>2}({oh[1]:+.1f})", flush=True)
    # MEM store value rows known from head5 probe: 123 (a), 183 (b)
    for r in (123, 183, 253):
        if r < len(ctx):
            ol = amax(out[r, OL:OL+16]); oh = amax(out[r, OH:OH+16])
            print(f"   ROW {r:>3} tok={ctx[r]:>3}: OUTPUT_LO cell={ol[0]:>2}({ol[1]:+.1f}) OUTPUT_HI cell={oh[0]:>2}({oh[1]:+.1f})", flush=True)


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 816
    blk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(idx, blk)
