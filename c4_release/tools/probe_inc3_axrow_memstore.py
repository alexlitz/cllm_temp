#!/usr/bin/env python3
"""Inc-3 ROOT A — does the CORRECT source row (step4 AX[1]) carry MEM_STORE in CAMPAIGN?

GOLDEN: LI head_1 attends step4 off7 (AX[1], CLEAN_EMBED=3), MEM_STORE=0 there.
CAMPAIGN: it shifts to step4 off0 (PC_marker, CLEAN_EMBED=0, MEM_STORE=+1.0) because
the head's Q slots 44-48 read MEM_STORE*500 and the 30-tok frame BROADCASTS MEM_STORE
onto the PC marker. The prior agent claimed a MARK_AX/MEM_STORE_AT_VAL re-route is
zero-sum because the AX[1] source row ITSELF carries the MEM_STORE leak.

This probe dumps, in CAMPAIGN, the MEM_STORE + MARK_AX + BYTE_INDEX_1 + CLEAN_EMBED of
the candidate source rows for step4 (off0..off9), at the INPUT to blk16, so we see:
  (1) which row carries CLEAN_EMBED=3 (the real byte-1 token row),
  (2) whether THAT row also has MEM_STORE>0 (the prior agent's claimed blocker),
  (3) whether MARK_AX + BYTE_INDEX_1 cleanly discriminate the AX[1] row from the
      PC marker row -> a clean K-veto exists.

Run in CAMPAIGN only (clear cache first):
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_axrow_memstore.py
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
BLK = 16


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item())
    hi_ = int(torch.argmax(row[hi:hi + 16]).item())
    return hi_ * 16 + li, float(row[lo + li]), float(row[hi + hi_])


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
    x_in = td(p.model.forward(padded, stop_after_block=BLK - 1)[0])
    ce_lo, ce_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    D = {k: dp[k] for k in [
        "MEM_STORE", "MARK_AX", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
        "MEM_ADDR_SRC", "MEM_VAL_B1", "IS_BYTE", "H1"]}
    print(f"=== {cfg} STEP={STEP} blk{BLK} input. step4 candidate rows ===")
    for off in range(0, 10):
        r = pl + 4 * STEP + off
        row = x_in[r]
        ceval, celo, cehi = nib(row, ce_lo, ce_hi)
        tag = _step_offset_field(off)
        flags = " ".join(
            f"{k}={float(row[v]):+.2f}" for k, v in D.items()
            if abs(float(row[v])) > 0.2)
        mark = " <<CE=3" if ceval == 3 else ""
        print(f"  row{r:4d} off{off}({tag:9s}) CLEAN_EMBED={ceval:3d}"
              f"(lo{celo:.1f}) | {flags}{mark}")


if __name__ == "__main__":
    main()
