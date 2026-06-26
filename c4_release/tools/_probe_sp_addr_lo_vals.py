#!/usr/bin/env python3
"""Dump the relayed SP_ADDR_LO / SP_ADDR_PRESENT band values at the L8-attn-block
INPUT (block 10 output) for the candidate store rows + ADD/binary-op query rows,
on var_simple (noisy) vs expr_add_mul (clean). Used to design the block-10
winner-take-all sharpener (bound PRESENT to [0,1]).
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


def main(ids, upto_block=10):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    LO = int(dp["SP_ADDR_LO"]); PR = int(dp["SP_ADDR_PRESENT"])
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = p._final_context(bc, max_steps=12)
        padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
        out = p.model.forward(padded, stop_after_block=upto_block)[0]
        ax = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
        # candidate store value rows: scan all rows with nonzero SP_ADDR_LO
        print(f"\n== id={idx} {desc} exp={exp} block_in={upto_block+1} ==", flush=True)
        rows = []
        for r in range(len(ctx)):
            band = out[r, LO:LO+16]
            pr = float(out[r, PR])
            if band.abs().sum() > 0.01 or abs(pr) > 0.01:
                rows.append(r)
        for r in rows:
            band = out[r, LO:LO+16]
            pr = float(out[r, PR])
            nz = (band.abs() > 0.01).sum().item()
            amax = int(band.argmax()); amaxv = float(band[amax])
            tag = "AX" if r in ax else ("" )
            print(f"  row {r:>3} tok={ctx[r]:>3} {tag:2}: PRESENT={pr:+.3f} nz_cells={nz} "
                  f"argmax_cell={amax}({amaxv:+.3f}) sum={float(band.sum()):+.3f} "
                  f"min={float(band.min()):+.3f} max={float(band.max()):+.3f}", flush=True)


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [250, 816]
    main(ids)
