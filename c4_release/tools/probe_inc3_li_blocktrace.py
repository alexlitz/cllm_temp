#!/usr/bin/env python3
"""Inc-3 cont.: block-by-block trace of OUTPUT_HI at the step-5 AX[0] predictor row.

The AX[0] row (offset 6) PREDICTS the AX[1] (byte-1) token. In golden it carries
OUTPUT_HI argmax nibble = 3 (-> byte1=0x03); in campaign it's 0. This trace dumps,
per physical block, the OUTPUT_HI argmax nibble at off 6 (and the L15 lookup-head
block) so we see WHICH block delivers (golden) / drops (campaign) the byte-1 value.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_li_blocktrace.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_li_blocktrace.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 990; return x; }"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)
    OUT_HI = dp["OUTPUT_HI_THIS_STEP"]
    OUT_LO = dp["OUTPUT_LO"]

    STEP = int(Token.STEP_TOKENS)
    prompt_len = len(p._build_context(bytecode))
    s5 = prompt_len + 5 * STEP
    off6 = s5 + 6   # AX[0] predictor of byte-1
    blk_map = p.block_layer_map()
    print(f"=== {cfg}  STEP_TOKENS={STEP}  off6(AX[0] predictor)={off6} ===")

    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    nblk = len(p.runner.model.blocks)
    prev = -99.0
    for b in range(nblk):
        resid = p.model.forward(padded, stop_after_block=b)
        row = resid[0, off6]
        hi0 = float(row[OUT_HI].item())   # scalar byte-1 magnitude
        lo0 = float(row[OUT_LO].item())
        logical = blk_map[b]["logical"]
        attn = blk_map[b]["attn"][:18]
        # Print blocks where OUTPUT_HI[0] scalar CHANGES by >0.3 (delivery/drop).
        if abs(hi0 - prev) > 0.3:
            print(f" blk{b:2d} L{logical:2d} {attn:18s} OUT_HI[0]={hi0:+.2f}"
                  f" OUT_LO[0]={lo0:+.2f}  <-- changed (was {prev:+.2f})")
            prev = hi0
    print(f" FINAL OUT_HI[0] at off6 = {prev:+.2f}  (golden wants ~+3.0)")


if __name__ == "__main__":
    main()
