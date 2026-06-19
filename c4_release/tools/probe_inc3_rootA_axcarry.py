#!/usr/bin/env python3
"""Inc-3 ROOT A: dump the AX_CARRY_LO/HI band sum on the AX-marker / AX[0] rows
(at the carry head's INPUT, block 17 output) GOLDEN vs CAMPAIGN, to confirm the
30-tok stride changes the AX_CARRY magnitude that drives the slot-2 K loop.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_rootA_axcarry.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_rootA_axcarry.py
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

SRC = "int main() { int x; x = 990; return x; }"
BLK = 18


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN" if nostk else "GOLDEN"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    off5 = pl + 5 * STEP + 6
    ctx = p._final_context(bytecode, max_steps=9)
    padded = torch.tensor([ctx], device=p._device)
    x = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    axc_lo = dp["AX_CARRY_LO"]; axc_hi = dp["AX_CARRY_HI"]
    print(f"=== {cfg} STEP={STEP} AX_CARRY_LO={axc_lo} HI={axc_hi} ===")
    print("  per AX-marker/AX[0]/AX[1] row: sum(AX_CARRY_LO[0:16]+HI[0:16])")
    for st in range(0, 6):
        base = pl + st * STEP
        for o in (5, 6, 7):
            r = base + o
            if r > off5:
                break
            lo = float(x[r][axc_lo:axc_lo + 16].sum())
            hi = float(x[r][axc_hi:axc_hi + 16].sum())
            print(f"    s{st}.{o:02d} row={r:4d} {_step_offset_field(o):11s} "
                  f"AXC_LO_sum={lo:+9.2f} AXC_HI_sum={hi:+9.2f} tot={lo+hi:+9.2f} "
                  f"=> slot2_K={-0.2*(lo+hi):+8.1f}")


if __name__ == "__main__":
    main()
