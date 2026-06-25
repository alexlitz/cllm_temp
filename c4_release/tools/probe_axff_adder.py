#!/usr/bin/env python3
"""Probe the L10 ADD high-byte adder gate dims at id325 step-8 (b1=0xff OK) vs
step-14 (b1 nuked to 0x00) byte-1 rows, at block 44 (just before block-45 adder).
Determines whether TEMP+12 leaks at step-14 (the adder false-fires) or whether
the nuke is some other 128-unit op.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 50; x = x + 7; return x; }"
STEPS = [8, 14]


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEPT = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)
    bc, _ = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    rows = {}
    for s in STEPS:
        base = prefix_len + s * STEPT
        seg = ctx[base:base + STEPT]
        ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
        rows[s] = base + ax_i + 1  # byte-1 predictor row

    scalars = ["IS_BYTE", "HAS_SE", "BYTE_INDEX_0", "BYTE_INDEX_1", "CONST",
               "OP_LEA", "OP_ADD", "OP_SI"]
    bands = ["TEMP", "STACK0_BYTE_VAL_1_LO", "ADDR_B1_LO", "CARRY", "H1",
             "AX_CARRY_LO", "AX_CARRY_HI"]
    for b in (44, 45):
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        print(f"=== block {b} byte-1 rows ===")
        for s in STEPS:
            row = resid[rows[s]]
            sc = " ".join(f"{nm}={float(row[dp[nm]].item()):.2f}"
                          for nm in scalars if nm in dp)
            print(f"  step{s}: {sc}")
            for nm in bands:
                base = dp.get(nm)
                if base is None:
                    continue
                vals = [float(row[base + i].item()) for i in range(16)]
                lit = [(i, v) for i, v in enumerate(vals) if abs(v) > 0.4]
                print(f"      {nm}: " + " ".join(f"{i}={v:.1f}" for i, v in lit))


if __name__ == "__main__":
    main()
