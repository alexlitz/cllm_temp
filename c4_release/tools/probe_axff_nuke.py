#!/usr/bin/env python3
"""Compare WHY id325 step-8 (byte-1=0xff OK) vs step-14 (byte-1=0x00 WRONG)
diverge at the byte-1 nuke. Both are NEG b0=0xe8 with the same b1-row signature.
Trace OUTPUT_LO/HI byte-1 argmax per block 33..46 for both steps + dump the
final-block LM-head decode for byte-1.
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


def bam(row, name, dp, w=16):
    b = dp.get(name)
    if b is None:
        return (None, None)
    vals = [float(row[b + i].item()) for i in range(w)]
    pos_am = max(range(w), key=lambda i: vals[i])
    return pos_am, vals[pos_am]


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

    for b in range(33, 47):
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=b)[0]
        parts = []
        for s in STEPS:
            row = resid[rows[s]]
            lo = bam(row, "OUTPUT_LO", dp)
            hi = bam(row, "OUTPUT_HI", dp)
            parts.append(f"s{s}:LO{lo[0]:2d}({lo[1]:11.1f})HI{hi[0]:2d}({hi[1]:11.1f})")
        print(f"blk{b:2d}  " + "   ".join(parts))

    # Final-block per-step extra signals to find the nuke discriminator.
    print("\n--- extra signals at byte-1 row, block 44 (pre-nuke) ---")
    with torch.no_grad():
        resid = model.forward(padded, stop_after_block=44)[0]
    cands = ["OP_LEA", "OP_SI", "OP_ADD", "OP_LI", "OP_IMM", "HAS_SE", "IS_BYTE",
             "MARK_AX", "OPCODE_BYTE_LO", "OPCODE_BYTE_HI", "AX_CARRY_OVERFLOW",
             "BYTE_INDEX_0", "BYTE_INDEX_1"]
    for s in STEPS:
        row = resid[rows[s]]
        parts = []
        for nm in cands:
            b = dp.get(nm)
            if b is None:
                continue
            if nm.startswith("OPCODE_BYTE"):
                vals = [float(row[b + i].item()) for i in range(16)]
                am = max(range(16), key=lambda i: vals[i])
                parts.append(f"{nm}[am{am}={vals[am]:.2f}]")
            else:
                parts.append(f"{nm}={float(row[b].item()):.3f}")
        print(f"  step{s}: " + " ".join(parts))


if __name__ == "__main__":
    main()

# Extra: TEMP band + adder gate dims at step 8 vs 14 byte-1 row, block 44.
