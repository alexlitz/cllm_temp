"""Direct-operand probe of the LIVE MultiPassDivBlock install (C4_DIV_MULTIPASS=1).

Builds the flag-ON model, constructs a synthetic residual with a div/mod frame
at a single row (MARK_AX + OP_DIV/OP_MOD + dividend in ALU_LO/HI + divisor in
AX_CARRY_LO/HI), runs THAT ONE block (the MultiPassDivBlock at blocks[18].ffn),
and decodes OUTPUT_LO/HI — confirming the install computes a//b (DIV) and a%b
(MOD) end to end in the live layout, including b==0 -> q=0 / r=a.
"""
import os
import sys

os.environ["C4_DIV_MULTIPASS"] = "1"

import torch

from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
    compile_full_vm_dynamic,
)


def main():
    model, layout = compile_full_vm_dynamic(disk_cache=False)
    dp = layout.dim_positions
    d_model = model.blocks[0].attn.dim if hasattr(model.blocks[0], "attn") else None
    # find the multipass block
    blk = None
    for b in model.blocks:
        if getattr(getattr(b, "ffn", None), "_is_multipass_div_block", False):
            blk = b.ffn
            break
    assert blk is not None, "MultiPassDivBlock not installed"
    D = int(blk.output_lo)  # any dim to infer width from a pass
    # infer d_model from a pass W_up
    for p in blk.pipeline:
        if hasattr(p, "W_up") and p.W_up is not None:
            d_model = int(p.W_up.shape[1])
            break

    def frame(a, b, op):  # op in {'div','mod'}
        x = torch.zeros(1, 1, d_model)
        x[0, 0, dp["MARK_AX"]] = 1.0
        x[0, 0, dp["OP_DIV" if op == "div" else "OP_MOD"]] = 1.0
        x[0, 0, dp["ALU_LO"] + (a & 0xF)] = 1.0
        x[0, 0, dp["ALU_HI"] + ((a >> 4) & 0xF)] = 1.0
        x[0, 0, dp["AX_CARRY_LO"] + (b & 0xF)] = 1.0
        x[0, 0, dp["AX_CARRY_HI"] + ((b >> 4) & 0xF)] = 1.0
        return x

    def dec(y, base):
        return int(y[0, 0, dp[base]:dp[base] + 16].argmax())

    cases = [(42, 6), (100, 7), (84, 2), (255, 15), (200, 13), (127, 3),
             (255, 1), (1, 1), (17, 5), (5, 0), (255, 0), (0, 3), (240, 16),
             (13, 4), (250, 9)]
    fails = 0
    with torch.no_grad():
        for op in ("div", "mod"):
            for a, b in cases:
                y = blk(frame(a, b, op))
                lo, hi = dec(y, "OUTPUT_LO"), dec(y, "OUTPUT_HI")
                got = (hi << 4) | lo
                if b == 0:
                    want = 0 if op == "div" else a
                else:
                    want = a // b if op == "div" else a % b
                ok = got == want
                fails += not ok
                if not ok:
                    print(f"FAIL {op} {a}/{b}: got {got} (lo={lo},hi={hi}) "
                          f"want {want}")
    print(f"cases={2 * len(cases)} fails={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
