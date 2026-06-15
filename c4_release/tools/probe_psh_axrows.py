#!/usr/bin/env python3
"""For id852, dump the rows of the PSH-of-MUL-result frame (the MARK_AX row +
its BYTE_INDEX_0..3 rows + the following STACK0 frame byte rows) showing
MARK_AX, BYTE_INDEX, STACK0_BYTE*, and the candidate high-byte carriers
(AX_FULL, MUL_RESULT_HI, CLEAN_EMBED, OUTPUT, ADDR_B1) at the block where the
broadcast head runs and just after.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/probe_psh_axrows.py
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from tools.probe_groundtruth import build_groundtruth_probe


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return bc, exp, desc


def decode(row, lo, hi):
    los = [i for i in range(16) if float(row[lo + i].item()) > 0.5]
    his = [i for i in range(16) if float(row[hi + i].item()) > 0.5]
    if len(los) == 1 and len(his) == 1:
        return his[0] * 16 + los[0]
    if not los and not his:
        return "."
    return f"n{los}/{his}"


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    flags = {nm: dp[nm] for nm in
             ("MARK_AX", "MARK_STACK0", "STACK0_BYTE0", "STACK0_BYTE1",
              "BYTE_INDEX_0", "BYTE_INDEX_1", "OP_PSH", "OP_MUL", "OP_IMM")
             if nm in dp}
    bands = [(nm, dp[nm + "_LO"], dp[nm + "_HI"]) for nm in
             ("AX_FULL", "MUL_RESULT_HI", "CLEAN_EMBED", "OUTPUT",
              "STACK0_BYTE_VAL_1", "ALU", "AX_CARRY")
             if (nm + "_LO") in dp and (nm + "_HI") in dp]
    addr = {nm: dp[nm] for nm in ("ADDR_B0", "ADDR_B1") if nm in dp}

    bc, exp, desc = _corpus(852)
    ctx = probe._final_context(bc, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    full = model.forward(padded, stop_after_block=nblocks - 1)[0]
    nrows = full.shape[0]

    # PSH-of-MUL row: the MARK_AX + OP_PSH row that comes after the MUL row.
    mul_row = max([r for r in range(nrows)
                   if float(full[r, dp["MARK_AX"]].item()) > 0.5
                   and float(full[r, dp["OP_MUL"]].item()) > 0.5], default=None)
    psh_row = min([r for r in range(nrows)
                   if float(full[r, dp["MARK_AX"]].item()) > 0.5
                   and float(full[r, dp["OP_PSH"]].item()) > 0.5
                   and r > mul_row], default=None)
    print(f"# {desc} MUL_row={mul_row} PSH_of_mul_row={psh_row}")
    lo_r = mul_row
    hi_r = (psh_row + 40) if psh_row else mul_row + 80

    for b in (11, 15):
        resid = model.forward(padded, stop_after_block=b)[0]
        print(f"\n=== block {b} : rows {lo_r}..{hi_r} ===")
        for r in range(lo_r, min(hi_r, nrows)):
            fl = [nm for nm, d in flags.items() if float(resid[r, d].item()) > 0.5]
            if not fl:
                continue
            bvals = "  ".join(f"{nm}={decode(resid[r], lo, hi)}" for nm, lo, hi in bands)
            avals = "  ".join(f"{nm}={int(round(float(resid[r, d].item())))}" for nm, d in addr.items())
            print(f"  row {r:3d} [{','.join(fl)}]: {bvals}  | {avals}")


if __name__ == "__main__":
    main()
