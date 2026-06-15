#!/usr/bin/env python3
"""For id852 (14*56/8, MUL result 784=0x310), at the MUL AX row, the PSH AX
row, and the STACK0_BYTE_VAL rows that follow, dump EVERY 16-wide LO/HI band
present in dim_positions across blocks 8..16, printing any band that decodes
to a non-zero value. Goal: locate where the computed high byte 0x03 lives.

Usage: CUDA_VISIBLE_DEVICES=1 C4_MUL_WIDTH2=1 python tools/probe_mul_psh_allbands.py
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


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)

    # All LO bands that have a sibling HI band.
    pairs = []
    for name in dp:
        if name.endswith("_LO"):
            base = name[:-3]
            hi = base + "_HI"
            if hi in dp:
                pairs.append((base, dp[name], dp[hi]))
    print(f"# {len(pairs)} LO/HI band pairs")

    STACK0_BYTE1 = dp["STACK0_BYTE1"]; MARK_AX = dp["MARK_AX"]
    OP_MUL = dp["OP_MUL"]; OP_PSH = dp["OP_PSH"]; OP_DIV = dp["OP_DIV"]

    bc, exp, desc = _corpus(852)
    ctx = probe._final_context(bc, max_steps=12)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    full = model.forward(padded, stop_after_block=nblocks - 1)[0]
    nrows = full.shape[0]
    ax = {r: [nm for nm, dd in (("MUL", OP_MUL), ("PSH", OP_PSH), ("DIV", OP_DIV))
              if float(full[r, dd].item()) > 0.5]
          for r in range(nrows) if float(full[r, MARK_AX].item()) > 0.5}
    mul_row = max([r for r, o in ax.items() if "MUL" in o], default=None)
    psh_after = min([r for r, o in ax.items() if "PSH" in o and (mul_row is None or r > mul_row)], default=None)
    s1_rows = [r for r in range(nrows) if float(full[r, STACK0_BYTE1].item()) > 0.5]
    s1_after_mul = [r for r in s1_rows if mul_row is not None and r > mul_row][:2]

    def decode(row, lo, hi):
        los = [(i, float(row[lo + i].item())) for i in range(16) if float(row[lo + i].item()) > 0.5]
        his = [(i, float(row[hi + i].item())) for i in range(16) if float(row[hi + i].item()) > 0.5]
        if len(los) == 1 and len(his) == 1:
            return his[0][0] * 16 + los[0][0]
        if not los and not his:
            return None
        return f"n(lo={[i for i,_ in los]},hi={[i for i,_ in his]})"

    targets = {"MUL_AX": mul_row, "PSH_after_mul": psh_after}
    for i, r in enumerate(s1_after_mul):
        targets[f"s1_after_mul_{i}"] = r
    print(f"# {desc} MUL_row={mul_row} PSH_after_mul={psh_after} s1_after_mul={s1_after_mul}")
    for label, r in targets.items():
        if r is None:
            continue
        print(f"\n### {label} row {r} ###")
        for b in range(8, min(17, nblocks)):
            resid = model.forward(padded, stop_after_block=b)[0]
            hits = []
            for base, lo, hi in pairs:
                v = decode(resid[r], lo, hi)
                if v is not None and v != 0:
                    hits.append(f"{base}={v}")
            if hits:
                print(f"  block {b}: " + "  ".join(hits))


if __name__ == "__main__":
    main()
