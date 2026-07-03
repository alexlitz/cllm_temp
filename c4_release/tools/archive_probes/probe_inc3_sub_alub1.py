#!/usr/bin/env python3
"""Inc-3 CLAW-BACK — fine block-by-block trace of the SUB byte-1 ALU nibble.

At the SUB-result byte-1 predictor row, golden delivers ALU_LO nibble 3 (value
0x03 for 801) but campaign delivers ALU_LO nibble 0 (value 0). This walks EVERY
physical block 8..18 printing the full ALU_LO / ALU_HI nibble vectors at that
row so we see WHICH block sets the correct nibble in golden and what campaign
does instead. The byte-1 row in-step offset differs by frame: golden in=9,
campaign in=6 (AX marker at off 8 vs 5).

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_sub_alub1.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_sub_alub1.py
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

SRC = os.environ.get("PROBE_SRC", "int main() { return 827 - 26; }")
RESULT_STEP = int(os.environ.get("PROBE_STEP", "3"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def topslots(row, base, n=16, k=4):
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
    ctx = p._final_context(bytecode, max_steps=RESULT_STEP + 2)
    padded = torch.tensor([ctx], device=p._device)
    base = pl + RESULT_STEP * STEP
    # find AX marker offset
    axmark = None
    for j in range(STEP):
        if ctx[base + j] == int(Token.REG_AX):
            axmark = j
            break
    b1row = base + axmark + 1  # byte-1 predictor row
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    blk_map = p.block_layer_map()
    nblk = len(p.model.blocks)
    print(f"=== {cfg} STEP={STEP} axmark_in={axmark} b1row={b1row} src={SRC!r} ===")
    for blk in range(8, 19):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[b1row]
        lg = blk_map[blk]
        lg = lg.get("logical") if isinstance(lg, dict) else lg
        print(f"  blk{blk:2d}(L{lg}): ALU_LO {topslots(row,alu_lo)} | ALU_HI {topslots(row,alu_hi)}")


if __name__ == "__main__":
    main()
