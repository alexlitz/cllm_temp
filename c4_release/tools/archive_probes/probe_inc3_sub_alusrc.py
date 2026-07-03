#!/usr/bin/env python3
"""Inc-3 CLAW-BACK — find WHERE the SUB byte-1 nibble lives before L11.

At blk15 (pre-L11) the byte-1 result row's ALU is nibble-0 (empty); at blk16
(L11) golden gains ALU_LO nibble 3 (the byte-1 value 0x03). L11 attention must
COPY it from some cross-step row. This scans, at blk15 AND blk16, EVERY row of
the result step (and the prior step) for ALU_LO nibble presence, so we see the
source row golden reads and how campaign's row signature shifts.

Run TWICE (clear cache between):
  C4_NO_STACK0_EMIT=0  python tools/probe_inc3_sub_alusrc.py
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_sub_alusrc.py
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
    return " ".join(f"[{i}]={v:.1f}" for i, v in vals[:k]) or "."


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
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    print(f"=== {cfg} STEP={STEP} src={SRC!r} ===")
    for blk in (15, 16):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        print(f"--- blk{blk} ALU_LO presence across rows (prev+result step) ---")
        lo = pl + (RESULT_STEP - 1) * STEP
        hi = min(pl + (RESULT_STEP + 1) * STEP, len(ctx))
        for pos in range(lo, hi):
            row = x[pos]
            s_lo = topslots(row, alu_lo)
            s_hi = topslots(row, alu_hi)
            if s_lo != "." or s_hi != ".":
                instep = (pos - pl) % STEP
                stp = (pos - pl) // STEP
                tok = ctx[pos] if pos < len(ctx) else -1
                print(f"  pos{pos} step{stp} in={instep:2d} tok={tok:3d}: "
                      f"ALU_LO {s_lo} | ALU_HI {s_hi}")


if __name__ == "__main__":
    main()
