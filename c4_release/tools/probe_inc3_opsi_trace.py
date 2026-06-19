#!/usr/bin/env python3
"""Inc-3 ROOT A — trace OP_SI at the step5 byte-1 predictor row across blocks.

OP_SI=+5 is present at the step5 LI byte-1 predictor row (off=6) in GOLDEN but
ABSENT in CAMPAIGN. The head_1 store_ax_byte1 Q slot 81 gates on OP_SI -> in
golden it fires and selects the AX[1] row (CLEAN_EMBED=3); in campaign it's dead
and the PC_marker (MEM_STORE) row wins -> byte-1=0. Trace OP_SI (+ OP_LI/OP_LEV)
block-by-block to find WHERE golden gains OP_SI and campaign doesn't.
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
    padded = torch.tensor([ctx], device=p._device)
    off = pl + 5 * STEP + 6
    blk_map = p.block_layer_map()
    D = {k: dp[k] for k in ["OP_SI", "OP_SC", "OP_LI", "OP_LI_RELAY", "OP_IMM"]}
    print(f"=== {cfg} STEP={STEP} OP_SI trace at step5 byte1 predictor off={off} ===")
    prev = None
    for blk in range(len(p.model.blocks)):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        vals = {k: float(row[v]) for k, v in D.items()}
        key = tuple(round(v, 1) for v in vals.values())
        if key != prev:
            lg = blk_map[blk]
            lg = lg.get("logical") if isinstance(lg, dict) else lg
            s = " ".join(f"{k}={v:+.2f}" for k, v in vals.items() if abs(v) > 0.1)
            print(f"  blk{blk:2d}(L{lg}): {s or '(all~0)'}")
            prev = key


if __name__ == "__main__":
    main()
