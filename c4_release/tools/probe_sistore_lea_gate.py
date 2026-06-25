#!/usr/bin/env python3
"""var_update id325 LEA materializer gate probe (campaign, spec_k=0).

Reads the exact gate dims of the l16_lea_local_ax_byte0_hi_e materializer
(MARK_AX, HAS_SE, OP_LEA, CMP+7, FETCH_LO+8, FETCH_HI+15, OP_ADD, OP_SUB,
OP_IMM) at the step-6 (WORKS) vs step-14 (FAILS) AX-marker rows, across blocks
spanning the lev_routing materializer, to CONFIRM the hypothesis: the OP_ADD
NOT-blocker (-1e9, the #309 arith guard) over-fires at step 14 because the prior
ADD's OP_ADD opcode broadcast PERSISTS into the LEA-after-ADD step.

  CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_sistore \
    python tools/probe_sistore_lea_gate.py [blocks...]
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
STEPS = [int(s) for s in os.environ.get("PROBE_STEPS", "6,14").split(",")]


def cell(row, name, dp, off=0):
    b = dp.get(name)
    if b is None:
        return f"{name}={'n/a'}"
    return f"{name}{('+'+str(off)) if off else ''}={float(row[b+off].item()):+.2f}"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)

    bc, data = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    rows = {}
    for s in STEPS:
        base = prefix_len + s * STEP
        seg = ctx[base:base + STEP]
        ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
        rows[s] = base + ax_i if ax_i is not None else None

    print(f"=== id325 LEA materializer gate (l16_lea_local_ax_byte0_hi_e) "
          f"STEP={STEP} ===", flush=True)
    print(f"   gate: MARK_AX+HAS_SE+OP_LEA+CMP+7 (thr 8); blockers OP_ADD/OP_SUB -1e9",
          flush=True)
    for b in blocks:
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        print(f" --- block {b} ---", flush=True)
        for s in STEPS:
            r = rows[s]
            if r is None:
                continue
            row = resid[r]
            terms = [
                cell(row, "MARK_AX", dp), cell(row, "HAS_SE", dp),
                cell(row, "OP_LEA", dp), cell(row, "CMP", dp, 7),
                cell(row, "OP_ADD", dp), cell(row, "OP_SUB", dp),
                cell(row, "OP_IMM", dp),
                cell(row, "FETCH_LO", dp, 8), cell(row, "FETCH_HI", dp, 15),
            ]
            print(f"   step{s:2d}  " + "  ".join(terms), flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [30, 32, 33, 34]
    main(blks)
