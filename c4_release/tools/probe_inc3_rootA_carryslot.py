#!/usr/bin/env python3
"""Inc-3 ROOT A: per-HD-slot K-contribution decomposition for the carry head,
to find WHICH K signature makes step1 AX_marker peak (raw 1384) in golden but
collapse (raw 815) in campaign. Shows q[s]*K[row,s]/sqrt(HD) for the golden
winner (step1 AX_marker), campaign winner (step4 AX[0]), and the step5 markers.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_rootA_carryslot.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_rootA_carryslot.py
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
import math
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
HEAD = 7


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
    attn = p.model.blocks[BLK].attn
    nh = attn.num_heads
    HD = attn.W_q.shape[0] // nh
    Wq = td(attn.W_q)[HEAD * HD:(HEAD + 1) * HD]
    Wk = td(attn.W_k)[HEAD * HD:(HEAD + 1) * HD]
    d_model = Wq.shape[1]
    xr = x[:, :d_model]
    q = xr[off5] @ Wq.T
    K = xr @ Wk.T
    # rows of interest
    rows = {
        "s1.AXmark": pl + 1 * STEP + 5,
        "s4.AX[0]": pl + 4 * STEP + 6,
        "s5.AXmark": pl + 5 * STEP + 5,
        "s0.AXmark": pl + 0 * STEP + 5,
    }
    print(f"=== {cfg} STEP={STEP} HD={HD} ===")
    for label, r in rows.items():
        contrib = (q * K[r]) / math.sqrt(HD)
        order = torch.argsort(contrib.abs(), descending=True)
        tot = float(contrib.sum())
        print(f"  {label} row={r} total={tot:+.1f}")
        for s in order[:6].tolist():
            print(f"      HDslot{s:2d}: q={float(q[s]):+8.2f} K={float(K[r,s]):+8.2f} "
                  f"-> {float(contrib[s]):+8.1f}")


if __name__ == "__main__":
    main()
