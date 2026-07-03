#!/usr/bin/env python3
"""Dump operand-B (AX_CARRY_LO/HI) and the MUL result band (MUL_RESULT_HI,
OUTPUT_LO/HI) at the MUL step's MARK_AX row, comparing add_mul (816) vs
standalone mul (143). Identifies whether operand-B or the result is the bug.

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       python tools/probe_addmul_opb.py 816 143
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


def _ax_rows(ctx):
    return [i for i, t in enumerate(ctx) if t == Token.REG_AX]


def _amax(band):
    v = band.tolist()
    m = max(range(len(v)), key=lambda i: v[i])
    return m, v[m]


def main(ids, mul_step_map):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    BL = int(dp["AX_CARRY_LO"]); BH = int(dp["AX_CARRY_HI"])
    OL = int(dp["OUTPUT_LO"]); OH = int(dp["OUTPUT_HI"])
    MRL = int(dp["MUL_RESULT_HI_LO"]); MRH = int(dp["MUL_RESULT_HI_HI"])
    blocks = [int(x) for x in os.environ.get("BLOCKS", "11,16,17,19").split(",")]
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = p._final_context(bc, max_steps=12)
        ax = _ax_rows(ctx)
        ms = mul_step_map.get(idx, 3)
        pos = ax[ms]
        print(f"\n==== id={idx} {desc} exp={exp} MUL row pos={pos} ====")
        padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
        for blk in blocks:
            r = p.model.forward(padded, stop_after_block=blk)[0, pos]
            (blm, blv) = _amax(r[BL:BL+16]); (bhm, bhv) = _amax(r[BH:BH+16])
            (olm, olv) = _amax(r[OL:OL+16]); (ohm, ohv) = _amax(r[OH:OH+16])
            (mlm, mlv) = _amax(r[MRL:MRL+16]); (mhm, mhv) = _amax(r[MRH:MRH+16])
            print(f"  blk{blk:2d}: opB_LO cell={blm}({blv:+.1f}) opB_HI cell={bhm}({bhv:+.1f}) | "
                  f"OUT_LO cell={olm}({olv:+.1f}) OUT_HI cell={ohm}({ohv:+.1f}) | "
                  f"MRH_LO cell={mlm}({mlv:+.1f}) MRH_HI cell={mhm}({mhv:+.1f})",
                  flush=True)


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [816, 143]
    mul_step = {816: 5, 800: 5, 143: 3, 850: 3, 825: 6}
    main(ids, mul_step)
