#!/usr/bin/env python3
"""func_add ALU_HI contaminant-cell sweep at blk13 across many operand pairs.

Confirms WHICH ALU_HI cell carries the address-nibble leak (and that it is never
the true operand-A high nibble cell), so we can pick the cells to clear.

  python tools/probe_funcadd_leak.py
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402

from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

PAIRS = [(57, 11), (36, 16), (8, 30), (99, 100), (13, 13), (50, 50),
         (100, 1), (1, 100), (45, 45), (77, 23)]
BLK = int(os.environ.get("PROBE_BLK", "13"))


def cells(row, base, width=16, thr=0.5):
    return {i: round(float(row[base + i].item()), 2)
            for i in range(width) if abs(float(row[base + i].item())) > thr}


def main():
    p = build_groundtruth_probe()
    model = p.model
    runner = p.runner
    dp = dict(model.dim_positions)
    dev = p._device
    STEP = int(Token.STEP_TOKENS)

    print(f"blk={BLK}  (operand A hi-nibble should be value//16; leak cell flagged)",
          flush=True)
    for (a, b) in PAIRS:
        src = (f"int add(int a, int b) {{ return a + b; }} "
               f"int main() {{ return add({a}, {b}); }}")
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
        # ADD step = step 13 for func_add
        start = len(prefix) + 13 * STEP
        toks = tape[start:start+STEP]
        ax_off = next((i for i, t in enumerate(toks) if t == int(Token.REG_AX)), None)
        ax_row = start + ax_off
        with torch.no_grad():
            r = model.forward(padded, stop_after_block=BLK)[0][ax_row]
        a_hi = cells(r, dp['ALU_HI'])
        a_lo = cells(r, dp['ALU_LO'])
        true_hi = a // 16
        # leak cells = nonzero ALU_HI cells that are NOT the true operand hi nibble
        leak = {k: v for k, v in a_hi.items() if k != true_hi and v > 0.5}
        print(f"  add({a:3d},{b:3d}) exp=0x{a+b:02X} | A=0x{a:02X} true_A_hi_cell={true_hi} "
              f"| ALU_HI={a_hi} LEAK={leak}", flush=True)


if __name__ == "__main__":
    main()
