#!/usr/bin/env python3
"""Inc-3 ROOT A: the AX byte-1 dump REPOPULATE-FFN gate signals at the step5
byte-1 predictor row for x=990, GOLDEN vs CAMPAIGN. Reads BEFORE the repopulate
FFN block:
  - ADDR_B0_LO+5  (byte-1 predictor signature)
  - ADDR_B1_HI+8  (AX-register-present prerequisite; >=2.70 legit, <=2.0 kill)
  - Σ AX_CARRY (LO+HI)  (carry band ~2.65 genuine; LEV kill at >=3.0)
  - AX_CARRY_HI+2  (SHL/JMP upper-cut at >=3.0)
  - AX_CARRY_OVERFLOW  (the kill flag, computed by the precursor FFN)
  - H1_PREV_STEP one-hot index (the carried byte-1 source)
  - H1_DUMP_OUT one-hot index (the emitted byte-1; -1 = dump dark)
Determines whether the campaign byte-1=0 is a dump KILL (gate threshold, in
scope) or a wrong H1_PREV source (carry head H1-onehot wall, out of scope).
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


def onehot(vec, base, n=16, thr=0.4):
    best, bv = -1, thr
    for i in range(n):
        v = float(vec[base + i])
        if v > bv:
            bv, best = v, i
    return best, bv


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off = pl + 5 * STEP + 6
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    addr5 = dp["ADDR_B0_LO"] + 5
    b1hi8 = dp["ADDR_B1_HI"] + 8
    axclo, axchi = dp["AX_CARRY_LO"], dp["AX_CARRY_HI"]
    axc_hi2 = dp["AX_CARRY_HI"] + 2
    ovf = dp.get("AX_CARRY_OVERFLOW")
    h1p, h1d = dp.get("H1_PREV_STEP"), dp.get("H1_DUMP_OUT")
    nblk = len(p.model.blocks)
    print(f"=== {cfg} STEP={STEP} off(byte1 pred)={off} ===")
    prev = None
    for blk in range(14, nblk):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        a5 = float(row[addr5])
        b1 = float(row[b1hi8])
        cs = sum(float(row[axclo + i]) for i in range(16)) + \
            sum(float(row[axchi + i]) for i in range(16))
        hi2 = float(row[axc_hi2])
        ov = float(row[ovf]) if ovf is not None else float("nan")
        pi = onehot(row, h1p)[0] if h1p is not None else -1
        di = onehot(row, h1d)[0] if h1d is not None else -1
        key = (round(cs, 1), round(ov, 1), pi, di)
        if key != prev:
            print(f"  blk{blk:2d}: ADDR_B0_LO+5={a5:+.2f} ADDR_B1_HI+8={b1:+.2f} "
                  f"ΣAX_CARRY={cs:+.2f} AX_CARRY_HI+2={hi2:+.2f} "
                  f"OVERFLOW={ov:+.2f} H1_PREV={pi} H1_DUMP_OUT={di}")
            prev = key


if __name__ == "__main__":
    main()
