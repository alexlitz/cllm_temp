#!/usr/bin/env python3
"""Safety check: does a GENUINE multi-byte ADD byte-1 row carry AX_CARRY_LO/HI+15
(the sign-ext signature my adder NOT-blocker would key on)? If AX_CARRY+15 is ~0
on every genuine ADD byte-1 row, the NOT-blocker is safe (only the negative-LEA
sign-ext rows have it).

Probes several multi-byte ADD programs (byte-1 != 0) at the ADD AX byte-1 row
(block 44, pre-adder) and reports AX_CARRY_LO+15, AX_CARRY_HI+15, OP_LEA, TEMP+12.
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

# Multi-byte ADD programs: result byte-1 != 0 so the adder MUST fire correctly.
SRCS = {
    "654+114=768(b1=0x03)": "int main() { return 654 + 114; }",
    "300+300=600(b1=0x02)": "int main() { return 300 + 300; }",
    "1000+1000=2000(b1=0x07)": "int main() { return 1000 + 1000; }",
    "255+1=256(b1=0x01)": "int main() { return 255 + 1; }",
    "200+100=300(b1=0x01)": "int main() { return 200 + 100; }",
    # operand B with high nibble 0xF (b1 byte) to stress AX_CARRY=operandB risk
    "240+15=255": "int main() { return 240 + 15; }",
    "4095+1=4096": "int main() { return 4095 + 1; }",
}
BLOCK = 44


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    dev = p._device
    STEPT = int(Token.STEP_TOKENS)
    ax_marker = int(Token.REG_AX)
    for label, src in SRCS.items():
        bc, _ = compile_c(src)
        ctx = p._final_context(bc, max_steps=12)
        prefix_len = len(p._build_context(bc))
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            resid = model.forward(padded, stop_after_block=BLOCK)[0]
        nsteps = (len(ctx) - prefix_len) // STEPT
        print(f"\n=== {label} ===")
        for s in range(min(nsteps, 12)):
            base = prefix_len + s * STEPT
            seg = ctx[base:base + STEPT]
            ax_i = next((i for i, t in enumerate(seg) if t == ax_marker), None)
            if ax_i is None:
                continue
            b1 = int(seg[ax_i + 2]) & 0xFF
            r1 = resid[base + ax_i + 1]
            def g(nm, off=0):
                b = dp.get(nm)
                return float(r1[b + off].item()) if b is not None else float("nan")
            print(f"  step{s:2d} emit_b1=0x{b1:02x}  AXC_LO+15={g('AX_CARRY_LO',15):.2f}"
                  f" AXC_HI+15={g('AX_CARRY_HI',15):.2f} OP_LEA={g('OP_LEA'):.2f}"
                  f" TEMP+12={g('TEMP',12):.2f}")


if __name__ == "__main__":
    main()
