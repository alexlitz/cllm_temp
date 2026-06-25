#!/usr/bin/env python3
"""var_update id325 step-14 LEA AR-decode opcode/operand probe (campaign, spec_k=0).

Builds the PRODUCTION AR context (_final_context — the real spec_k=0 decode that
emits the wrong AX=57 at step 14), then at the step-14 register-frame rows reads
the live opcode flags (OP_LEA vs OP_SI/OP_ADD), the FETCH/ADDR operand bands, and
the AX-output bands. Compares against the WORKING LEA at step 6. This tells us
whether at step 14 (a) OP_LEA is firing, (b) the BP+imm address operand is
present, (c) the AX OUTPUT write is being suppressed by stale SI/ADD residue.

  CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_sistore \
    python tools/probe_sistore_lea_ar.py [blocks...]
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


def fmt(row, base, width=16, thr=0.5):
    if base is None:
        return "n/a"
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.0f}@{i}" for i, v in enumerate(vals) if abs(v) > thr) + "]"


def scalar(row, name, dp):
    b = dp.get(name)
    if b is None:
        return f"{name}=n/a"
    return f"{name}={float(row[b].item()):.1f}"


def main(blocks):
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)
    pc_marker = int(Token.REG_PC)
    ax_marker = int(Token.REG_AX)

    bc, data = compile_c(SRC)
    ctx = p._final_context(bc, max_steps=30)
    prefix_len = len(p._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    # for each step, locate the REG_AX marker row + the AX-byte0 row (the marker+1
    # position; the model FORWARD at marker row predicts the AX byte0 token).
    rows = {}
    for s in STEPS:
        base = prefix_len + s * STEP
        seg = ctx[base:base + STEP]
        ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
        opc_i = 0  # PC marker row carries the step opcode flags after fetch
        rows[s] = dict(ax_marker_row=base + ax_i if ax_i is not None else None,
                       step_start=base, seg=seg)
        # decode the emitted AX value
        axv = None
        if ax_i is not None:
            axv = 0
            for j in range(4):
                axv |= (int(seg[ax_i + 1 + j]) & 0xFF) << (j * 8)
        rows[s]["ax_val"] = axv

    print(f"=== var_update id325 step-14 LEA AR probe  STEP={STEP} prefix={prefix_len} ===",
          flush=True)
    for s in STEPS:
        print(f"   step{s:2d} emitted_ax={rows[s]['ax_val']}", flush=True)

    opflags = ["OP_LEA", "OP_SI", "OP_ADD", "OP_LI", "OP_IMM", "OP_PSH"]
    opflags = [o for o in opflags if dp.get(o) is not None]
    outbands = ["OUTPUT_LO", "OUTPUT_HI", "FETCH_LO", "FETCH_HI",
                "ALU_LO", "ALU_HI", "ADDR_B0_LO", "ADDR_B0_HI"]
    outbands = [o for o in outbands if dp.get(o) is not None]

    for b in blocks:
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        print(f" --- block {b} ---", flush=True)
        for s in STEPS:
            r = rows[s]["ax_marker_row"]
            if r is None:
                continue
            row = resid[r]
            flags = " ".join(scalar(row, o, dp) for o in opflags)
            bands = " ".join(f"{o}={fmt(row, dp.get(o))}" for o in outbands)
            print(f"   step{s:2d}@AXmark  {flags}", flush=True)
            print(f"             {bands}", flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [11, 14, 16, 20, 26, 30, 36]
    main(blks)
