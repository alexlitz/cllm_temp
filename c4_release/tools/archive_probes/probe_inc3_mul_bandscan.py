#!/usr/bin/env python3
"""Inc-3 MAP — full band scan at the MUL result row (got_ax=0 vs correct).

expr_mul_div id850 (10*9/1) step3=MUL. In CAMPAIGN got_ax=0; GOLDEN emits 90.
Find which band carries the MUL RESULT (90=0x5A) and the operand-A (10) at the
step3 byte-0 + byte-1 AX predictor rows. Scan ALL value bands, both configs.

Run TWICE (clear cache between):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=0  python tools/probe_inc3_mul_bandscan.py
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_mul_bandscan.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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

SRC = "int main() { return 10 * 9 / 1; }"
DIV_STEP = 3
RESULT = 90  # 10*9


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item()); lv = float(row[lo + li])
    hi_i = int(torch.argmax(row[hi:hi + 16]).item()); hv = float(row[hi + hi_i])
    return hi_i * 16 + li, min(lv, hv)


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
    ctx = p._final_context(bytecode, max_steps=DIV_STEP + 2)
    s = pl + DIV_STEP * STEP
    print(f"=== {cfg} STEP={STEP} === step{DIV_STEP} AX bytes = {ctx[s+6:s+10]} "
          f"(MUL result want={RESULT})")
    padded = torch.tensor([ctx], device=p._device)
    blk_map = p.block_layer_map()
    # candidate nibble-pair bands
    cand = []
    for nm in ["ALU", "AX_FULL", "OUTPUT", "TEMP", "MUL_RESULT",
               "STACK0_BYTE_VAL_0", "STACK0_BYTE_VAL_1", "CLEAN_EMBED",
               "AX_CARRY", "FETCH"]:
        lo, hi = nm + "_LO", nm + "_HI"
        if lo in dp and hi in dp:
            cand.append((nm, dp[lo], dp[hi]))
    # probe both the AX byte0 predictor (off6) and byte1 predictor (off7=AX[0]),
    # plus AX marker (off5)
    for off_lbl, off in [("AXmark+5", s + 5), ("AXb0+6", s + 6), ("AXb1+7", s + 7)]:
        print(f"  --- row {off_lbl} (off={off}) ---")
        prev = None
        for blk in range(len(p.model.blocks)):
            x = td(p.model.forward(padded, stop_after_block=blk)[0])
            row = x[off]
            vals = []
            for nm, lo, hi in cand:
                v, c = nib(row, lo, hi)
                if c > 0.3:
                    vals.append(f"{nm}={v}(c{c:.1f})")
            key = tuple(vals)
            if key != prev and vals:
                lg = blk_map[blk]
                lg = lg.get("logical") if isinstance(lg, dict) else lg
                print(f"    blk{blk:2d}(L{lg}): " + " ".join(vals))
                prev = key


if __name__ == "__main__":
    main()
