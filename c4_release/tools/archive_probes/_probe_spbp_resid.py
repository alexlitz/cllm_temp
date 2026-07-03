#!/usr/bin/env python3
"""AUTOREGRESSIVE residual probe at the SP-marker row.

Replays the spec_k=0 GPU autoregressive tape, then reads the residual at the
SP-marker row of a chosen step at several blocks. Goal: find WHICH block flips
the SP byte0 and which opcode flag (PSH_AT_SP / CMP+3 (pop) / CMP+4 (jsr) /
OP_* ) is firing at that row.

Run:
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_spbp python tools/_probe_spbp_resid.py 262 4 5
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"; os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARK = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", int(Token.STEP_END): "STEP_END"}


def find_sp_marker(ctx, pl, step, STEP):
    """Return the SP marker position within the given step window."""
    base = pl + step * STEP
    for i in range(base, min(base + STEP, len(ctx))):
        if MARK.get(ctx[i]) == "SP":
            return i
    return None


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 262
    steps = [int(x) for x in sys.argv[2:]] or [4, 5]
    p = build_groundtruth_probe()
    m = p.model
    dp = m.dim_positions
    STEP = int(Token.STEP_TOKENS)
    tests = generate_test_programs()
    src, exp, name = tests[idx]
    bc, _ = compile_c(src)
    ctx = p._final_context(bc, max_steps=20)
    pl = len(p._build_context(bc))
    nblocks = len(m.blocks)
    print(f"=== id{idx} {name} STEP={STEP} nblocks={nblocks} ===")

    # Dims of interest.
    flagdims = {}
    for nm in ("PSH_AT_SP", "CMP+3", "CMP+4", "OP_PSH", "OP_ADJ", "OP_ENT",
               "OP_LEV", "OP_IMM", "OP_LI", "OP_SI", "OP_LC", "OP_SC",
               "OP_JSR", "OP_LEA", "MARK_SP", "HAS_SE"):
        base = nm.split("+")[0]
        off = int(nm.split("+")[1]) if "+" in nm else 0
        if base in dp:
            flagdims[nm] = int(dp[base]) + off

    def band_byte(row, lobase, hibase):
        lo = int(torch.argmax(row[dp[lobase]:dp[lobase] + 16]))
        hi = int(torch.argmax(row[dp[hibase]:dp[hibase] + 16]))
        return hi * 16 + lo

    blocks_to_read = [5, 6, 7, 8, 12, 15, 20, 25, 30, 35, nblocks - 1]
    blocks_to_read = sorted(set(b for b in blocks_to_read if 0 <= b < nblocks))

    padded = torch.tensor([ctx], device=p._device)
    for st in steps:
        sp_pos = find_sp_marker(ctx, pl, st, STEP)
        if sp_pos is None:
            print(f"  step{st}: no SP marker"); continue
        print(f"\n  ### step {st} SP marker @pos {sp_pos} ###")
        # flag readout at L6-ish (block 7 input ~ after L5/6) and OUTPUT trace.
        with torch.no_grad():
            for b in blocks_to_read:
                resid = m.forward(padded, stop_after_block=b)
                if resid.is_sparse:
                    resid = resid.to_dense()
                row = resid[0, sp_pos]
                outb = None
                if "OUTPUT_LO" in dp and "OUTPUT_HI_THIS_STEP" in dp:
                    outb = band_byte(row, "OUTPUT_LO", "OUTPUT_HI_THIS_STEP")
                emb = None
                if "EMBED_LO" in dp and "EMBED_HI" in dp:
                    emb = band_byte(row, "EMBED_LO", "EMBED_HI")
                flags = {nm: round(float(row[d]), 2) for nm, d in flagdims.items()
                         if abs(float(row[d])) > 0.3}
                print(f"   blk{b:2d}: OUT_byte0={outb} EMBED_byte0={emb} flags={flags}")


if __name__ == "__main__":
    main()
