#!/usr/bin/env python3
"""Validate the AX byte1-3 sign-extension DISCRIMINATOR across all steps of id325.

For each step's AX register dump, report (at block 41, just before the byte-1
nuke) the byte-0 emitted value and the candidate discriminator signals at the
byte-1/2/3 predictor rows:
  - AX_CARRY_LO+15, AX_CARRY_HI+15 (the sign-extension carry nibble)
  - OP_LEA residual
  - byte-0 OUTPUT_HI nibble (>=8 => negative)
We need a signal that is PRESENT only when byte-0 >= 0x80 (negative addr) and
ABSENT when the dumped value is a small positive int.
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

SRCS = {
    "var_update_325": "int main() { int x; x = 50; x = x + 7; return x; }",
    "add_basic": "int main() { return 654 + 114; }",
    "var_simple": "int main() { int x; x = 42; return x; }",
}
PROBE_BLOCK = 41


def sval(row, name, dp, off=0):
    b = dp.get(name)
    return float(row[b + off].item()) if b is not None else float("nan")


def band(row, name, dp, w=16):
    b = dp.get(name)
    if b is None:
        return [float("nan")]
    return [float(row[b + i].item()) for i in range(w)]


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEPT = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)

    for label, src in SRCS.items():
        bc, _ = compile_c(src)
        ctx = p._final_context(bc, max_steps=30)
        prefix_len = len(p._build_context(bc))
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=PROBE_BLOCK)[0]
        nsteps = (len(ctx) - prefix_len) // STEPT
        print(f"\n===== {label}  (steps={nsteps}, block {PROBE_BLOCK}) =====")
        for s in range(min(nsteps, 20)):
            base = prefix_len + s * STEPT
            seg = ctx[base:base + STEPT]
            ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
            if ax_i is None:
                continue
            amr = base + ax_i
            b0 = int(seg[ax_i + 1]) & 0xFF
            b1 = int(seg[ax_i + 2]) & 0xFF
            b2 = int(seg[ax_i + 3]) & 0xFF
            b3 = int(seg[ax_i + 4]) & 0xFF
            # byte-1 predictor row signals
            r1 = resid[amr + 1]
            axcl15 = sval(r1, "AX_CARRY_LO", dp, 15)
            axch15 = sval(r1, "AX_CARRY_HI", dp, 15)
            oplea = sval(r1, "OP_LEA", dp)
            hassse = sval(r1, "HAS_SE", dp)
            # byte-0 row OUTPUT_HI argmax (the dumped high nibble => sign)
            r0 = resid[amr]
            ohi = band(r0, "OUTPUT_HI", dp)
            ohi_am = max(range(16), key=lambda i: ohi[i])
            neg = "NEG" if b0 >= 0x80 else "pos"
            print(f"  step{s:2d} {neg} b0=0x{b0:02x} b1-3=[{b1:02x},{b2:02x},{b3:02x}]"
                  f"  b0_HInib={ohi_am}  | b1row: AXC_LO+15={axcl15:.2f} AXC_HI+15={axch15:.2f}"
                  f" OP_LEA={oplea:.2f} HAS_SE={hassse:.2f}")


if __name__ == "__main__":
    main()
