#!/usr/bin/env python3
"""Inc-3 CLAW-BACK — find which row in the campaign frame carries operand-A's
byte-1 CLEAN_EMBED + MEM_VAL_B1 marker, so we know what the L10 stack0_byte_relay
(head 5) should K-match instead of the dropped STACK0_BYTE1 row.

Scans ALL rows of the result step + prior steps at blk13 (the relay read point)
printing, per row: token, in-step offset, MARK_*, MEM_STORE, MEM_VAL_B0..3,
STACK0_BYTE1..3, BYTE_INDEX_0..3, and CLEAN_EMBED_LO/HI nibble. Run in CAMPAIGN.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_sub_memrows.py
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
BLK = int(os.environ.get("PROBE_BLK", "13"))


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def cln(row, base):
    vals = [(i, float(row[base + i])) for i in range(16)]
    vals = [(i, v) for i, v in vals if abs(v) > 0.3]
    vals.sort(key=lambda x: -abs(x[1]))
    return ",".join(f"{i}={v:.1f}" for i, v in vals[:2]) or "."


def main():
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
    x = td(p.model.forward(padded, stop_after_block=BLK)[0])
    cle_lo, cle_hi = dp["CLEAN_EMBED_LO"], dp["CLEAN_EMBED_HI"]
    print(f"=== STEP={STEP} blk{BLK} src={SRC!r} ===")
    lo = pl + (RESULT_STEP - 1) * STEP
    hi = min(pl + (RESULT_STEP + 1) * STEP, len(ctx))
    flag_names = ["MEM_STORE", "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2",
                  "MEM_VAL_B3", "STACK0_BYTE1", "STACK0_BYTE2", "STACK0_BYTE3",
                  "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3"]
    for pos in range(lo, hi):
        row = x[pos]
        flags = [n for n in flag_names if abs(float(row[dp[n]])) > 0.4]
        cl = cln(row, cle_lo)
        ch = cln(row, cle_hi)
        if not flags and cl == "." and ch == ".":
            continue
        instep = (pos - pl) % STEP
        stp = (pos - pl) // STEP
        tok = ctx[pos] if pos < len(ctx) else -1
        print(f"  pos{pos} s{stp} in={instep:2d} tok={tok:3d}: CLE_LO[{cl}] "
              f"CLE_HI[{ch}] flags={flags}")


if __name__ == "__main__":
    main()
