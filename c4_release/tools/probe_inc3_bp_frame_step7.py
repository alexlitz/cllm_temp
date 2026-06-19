#!/usr/bin/env python3
"""Inc-3 cont.: the STEP-7 BP-frame wall (the LARGEST campaign-config var_simple root).

GPU campaign-config baseline (ids 250-449, full_trace, spec_k=0, this session):
36/200 pass, 164 fail. Fail divergence-step histogram:
  step0=25  step3=7  step4=32(if_gt)  step5=20(var_simple x>=256, LI byte-1)
  step6=25(var_mul hi-byte adder)  step7=55(var_simple ALL values, BP-frame).

The step-7 cluster (55, the dominant) is a DISTINCT root from the step-5 LI byte-1:
the var_simple program (return x) collapses INSIDE step 6 at the BP byte-1 row.
x=28 (1-byte) reaches step 6 with AX correct, then BP byte-1 is emitted as the
HALT token (2) instead of 0xFF (255) -> the whole frame floods with token 2 ->
got_pc/got_ax = None at step 7. (x>=256 hits the step-5 byte-1 wall FIRST; fixing
step-5 advances them to THIS step-7 wall -> step-5 fix nets 0 program passes alone.)

ROOT (GPU-residual-probed, this session): at the BP[0] predictor row (off16, which
predicts the BP byte-1 token) the ``l16_bp_frame_byte1_ff`` emitter's gate dims
are PRESENT in golden (H1+3=1.0, BYTE_INDEX_0=0.97, CLEAN_EMBED_HI+15=1.0 ->
OUTPUT_LO+15=+48 -> 0xff) but ABSENT in campaign (H1+3=0, BYTE_INDEX_0=0,
CLEAN_EMBED_HI+15=0) AND the OUTPUT band has EXPLODED to ~4.27e21 (the documented
OUTPUT-band self-reinforcement megaroot). So this is NOT a single-rule re-key like
the Inc-3 step-2 AX byte-1 fix -- the 30-tok BP frame loses its marker-distance
(H1+3) signature AND suffers an upstream OUTPUT-band explosion. Two-part build.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_bp_frame_step7.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_bp_frame_step7.py
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
from neural_vm.batched_pure_neural import Token, _step_offset_field  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = "int main() { int x; x = 28; return x; }"
# The l16_bp_frame_byte1_ff gate dims (see l16_ops.py:1430).
GATE = ["IS_BYTE", "HAS_SE", "H1+3", "BYTE_INDEX_0",
        "CLEAN_EMBED_LO+0", "CLEAN_EMBED_HI+15", "MARK_BP"]
OUT = ["OUTPUT_LO+15", "OUTPUT_HI_THIS_STEP+15", "OUTPUT_LO+0", "OUTPUT_HI_THIS_STEP+0"]


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30-tok)" if nostk else "GOLDEN(35-tok)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)

    def D(n):
        if "+" in n:
            b, o = n.rsplit("+", 1)
            return dp[b] + int(o)
        return dp[n]

    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    last = len(p.runner.model.blocks) - 1
    s6 = pl + 6 * STEP
    bp0 = s6 + 16  # BP[0] row = predictor of the BP byte-1 token
    print(f"=== {cfg} STEP={STEP} BP[0]-predictor pos={bp0} ===")

    ctx = p._final_context(bytecode, max_steps=10)
    padded = torch.tensor([ctx], device=p._device)
    x = p.model.forward(padded, stop_after_block=last)[0]
    x = (x.to_dense() if x.layout != torch.strided else x).cpu().float()
    logits = p.model.forward(torch.tensor([ctx[:bp0 + 1]], device=p._device))[0, bp0]
    print(f" BP[0] pred_next={int(logits.argmax())} (golden wants 255/0xff)")
    print("  GATE: " + " ".join(f"{n}={float(x[bp0, D(n)]):+.2f}" for n in GATE))
    print("  OUT : " + " ".join(f"{n}={float(x[bp0, D(n)]):+.2e}" for n in OUT))


if __name__ == "__main__":
    main()
