#!/usr/bin/env python3
"""Inc-3 if_gt: find the block that writes the OUTPUT-band explosion at the
step3->step4 boundary (predicting the first token of the branch-taken step4).
The argmax probe showed pos abs 196 (step4[0]) emitted byte2 with logit ~3.5e19
=> a giant OUTPUT_LO explosion. Scan per-block residual at the PREDICTING row
(abs 195 = STEP_END of step3) to localize the owning op.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
      PROBE_SRC='int main(){ if (35 > 43) return 1; return 0; }' \
      python tools/probe_inc3_ifgt_explosion.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
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

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")
# Which absolute predicting row (default: STEP_END of step3 = prompt_len+4*STEP-1).
ROW_OVERRIDE = os.environ.get("PROBE_ROW")


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    STEP = int(Token.STEP_TOKENS)
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)
    OL = dp["OUTPUT_LO"]
    OH = dp["OUTPUT_HI"]
    prompt_len = len(p._build_context(bytecode))
    row = int(ROW_OVERRIDE) if ROW_OVERRIDE else prompt_len + 4 * STEP - 1
    ctx = p._final_context(bytecode, max_steps=8)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    bl_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    print(f"=== {cfg} STEP={STEP} predicting-row={row} (predicts pos {row+1}) "
          f"OUTPUT_LO={OL} ===")
    prev_lo = prev_hi = 0.0
    with torch.no_grad():
        for blk in range(nblk):
            resid = p.model.forward(padded, stop_after_block=blk)[0, row]
            lo = float(resid[OL:OL + 16].abs().max().item())
            hi = float(resid[OH:OH + 16].abs().max().item())
            thr = float(os.environ.get("PROBE_THR", "1e6"))
            if abs(lo - prev_lo) > thr or abs(hi - prev_hi) > thr:
                lm = bl_map[blk]
                # which OUTPUT_LO sub-dim is exploding
                lo_vec = resid[OL:OL + 16]
                k = int(lo_vec.abs().argmax().item())
                print(f" blk{blk:2d} (L{lm['logical']:2d} {lm['ffn'][:26]:26s}) "
                      f"|OUT_LO|max {prev_lo:.2e} -> {lo:.2e} (dim+{k}={lo_vec[k].item():.2e})  "
                      f"|OUT_HI|max -> {hi:.2e}")
            prev_lo, prev_hi = lo, hi
    print(f" FINAL row {row}: |OUT_LO|max={prev_lo:.3e} |OUT_HI|max={prev_hi:.3e}")


if __name__ == "__main__":
    main()
