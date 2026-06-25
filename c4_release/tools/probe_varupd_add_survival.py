#!/usr/bin/env python3
"""var_update ADD operand survival probe (campaign config, spec_k=0) — ORACLE TAPE.

Reads operand A (ALU_LO/HI) at the ADD step's MARK_AX row across a block sweep,
to see whether the loaded x value (operand A) carries an address-nibble
contaminant (the +0xF0 leak seen in the GPU baseline: got = sum + 240).

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_varupd_add_survival.py [blocks...]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

PROGS = {
    "x50_p7":  ("int main() { int x; x = 50; x = x + 7; return x; }", 50, 7, 57),
    "x36_p16": ("int main() { int x; x = 36; x = x + 16; return x; }", 36, 16, 52),
    "x8_p30":  ("int main() { int x; x = 8; x = x + 30; return x; }", 8, 30, 38),
}

ADD_STEP = int(os.environ.get("PROBE_ADD_STEP", "12"))


def fmt(row, base, width=16, thr=0.5):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vals) if abs(v) > thr) + "]"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    has_se = "SE_ALU_LO" in dp

    for pname, (src, xval, kval, exp) in PROGS.items():
        bc, data = compile_c(src)
        _, oracle_tokens = runner._oracle_pc_ax_steps(
            bc, data or b"", "", expected_steps=None, with_tokens=True)
        prompt = runner._build_element(bc, data or b"", [], "", spec_k=1,
                                       adaptive_start_k=0, expected_steps=None)
        prefix = list(prompt.context)
        tape = list(prefix)
        for stp in oracle_tokens:
            tape.extend(stp)
        padded = torch.tensor([tape], dtype=torch.long, device=dev)
        ax_base = dp["MARK_AX"]
        with torch.no_grad():
            emb = model.embed(padded)[0]
        step_start = len(prefix) + ADD_STEP * STEP
        ax_off = None
        for off in range(STEP):
            r = step_start + off
            if r < len(tape) and emb[r, ax_base].abs().item() > 0.5:
                ax_off = off
                break
        print(f"=== {pname} x={xval} +{kval} exp={exp} nsteps={len(oracle_tokens)} "
              f"ADD_STEP={ADD_STEP} ax_off={ax_off} ===", flush=True)
        if ax_off is None:
            print("   NO MARK_AX row", flush=True)
            continue
        ax_row = step_start + ax_off
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=b)[0]
            row = resid[ax_row]
            line = (f"   blk{b:2d} "
                    f"ALU_LO={fmt(row, dp['ALU_LO'])} "
                    f"ALU_HI={fmt(row, dp['ALU_HI'])} "
                    f"CARRY_LO={fmt(row, dp['AX_CARRY_LO'])} "
                    f"CARRY_HI={fmt(row, dp['AX_CARRY_HI'])} "
                    f"OUT_LO={fmt(row, dp['OUTPUT_LO'])} "
                    f"OUT_HI={fmt(row, dp['OUTPUT_HI'])}")
            if has_se:
                line += (f" SE_LO={fmt(row, dp['SE_ALU_LO'])} "
                         f"SE_HI={fmt(row, dp['SE_ALU_HI'])}")
            print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [11, 13, 14, 17, 19, 22, 26]
    main(blks)
