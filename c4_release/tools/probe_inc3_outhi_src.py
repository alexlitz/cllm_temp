#!/usr/bin/env python3
"""Inc-3 ROOT A — pin the byte-1 OUTPUT_HI source at blk16 (L11).

GOLDEN: at step5 byte-1 predictor row, OUTPUT_HI nibble decodes to 3 at blk16
(L11) but is 0 in CAMPAIGN. AND AX_FULL_HI decodes to 3 in BOTH configs at
blk18. So the prior AX[1]=0x03 survives in AX_FULL_HI even in campaign -- the
break is the L11 OUTPUT delivery, not the source value.

This probe dumps the FULL 16-wide nibble vectors (not just argmax) for OUTPUT_HI,
OUTPUT_LO, AX_FULL_HI, AX_FULL_LO at the step5 byte-1 predictor row across the
blocks around L10/L11 (blk13..18), in BOTH configs, so we see exactly which
nibble slot lights and at which block OUTPUT_HI gains/loses the 0-index for 3.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_outhi_src.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_outhi_src.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
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


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def topslots(row, base, n=16, k=3):
    vals = [(i, float(row[base + i])) for i in range(n)]
    vals = [(i, v) for i, v in vals if abs(v) > 0.3]
    vals.sort(key=lambda x: -abs(x[1]))
    return " ".join(f"[{i}]={v:.1f}" for i, v in vals[:k]) or "(empty)"


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=9)
    off = pl + 5 * STEP + 6
    padded = torch.tensor([ctx], device=p._device)
    blk_map = p.block_layer_map()
    o_lo, o_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    a_lo, a_hi = dp["AX_FULL_LO"], dp["AX_FULL_HI"]
    print(f"=== {cfg} STEP={STEP} off={off} (step5 byte1 predictor) ===")
    for blk in range(12, 36):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        lg = blk_map[blk]
        lg = lg.get("logical") if isinstance(lg, dict) else lg
        print(f"  blk{blk:2d}(L{lg}): OUT_HI {topslots(row,o_hi)} | "
              f"OUT_LO {topslots(row,o_lo)} | AXF_HI {topslots(row,a_hi)} | "
              f"AXF_LO {topslots(row,a_lo)}")


if __name__ == "__main__":
    main()
