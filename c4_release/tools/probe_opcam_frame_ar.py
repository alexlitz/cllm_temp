#!/usr/bin/env python3
"""AUTOREGRESSIVE frame-depth discriminator probe for the operand-CAM LEA byte-0
root (C4_OPCAM_FRAME).

The multi-param func targets (func_add/mul/max/min) have TWO frame-local
addresses in the body: the 1st param ``&a`` (BP-8, effective byte-0 wants 0xE8)
and the 2nd param ``&b`` (BP-16, wants 0xE0). The single L25-tail discriminator
(FETCH imm=16) is IDENTICAL across the two frames, so the hardcoded 0xE8 writer
slams 0xE8 onto BOTH -> the 2nd-param LEA emits 0xFFE8 instead of 0xFFE0.

This probe REPLAYS the model's OWN autoregressive decode (NOT teacher-forced --
that diverges, per 4 prior agents) and dumps, at every LEA AX-marker row on the
DECODED tape, the candidate frame-depth discriminators:
  * ALU_HI+15 / ALU_HI+13 MAGNITUDE (the BP-frame residue)
  * OUTPUT_HI_THIS_STEP complete-frame nibble winner (0xE=14 / 0xD=13)
  * OUTPUT_LO nibble winner
at BOTH the L8-ALU output block and the L25-tail input block, plus the final
emitted byte. The 1st-param vs 2nd-param separation identifies the discriminator.

Run:
  CUDA_VISIBLE_DEVICES="" C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_opcam python tools/probe_opcam_frame_ar.py
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
import sys
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.verification.faithful_autoregressive import (  # noqa: E402
    build_cpu_model, FaithfulAutoregressiveRunner,
)
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

STEP = int(Token.STEP_TOKENS)
AX_MARK = int(Token.MARK_AX) if hasattr(Token, "MARK_AX") else 258

import argparse

CASES = [
    ("func_add_0 add(57,11)", 575),
    ("func_max_0 max(36,99)", 650),
    ("func_max_1 max(54,44)", 651),
    ("func_min_0 min(13,57)", 675),
    ("func_mul_0 mul(49,36)", 600),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="650",
                    help="comma-sep program ids (default func_max_0)")
    ap.add_argument("--max-steps", type=int, default=14)
    args = ap.parse_args()
    want_ids = [int(x) for x in args.ids.split(",") if x.strip()]

    model, layout = build_cpu_model(disk_cache=True)
    dp = dict(layout.dim_positions)
    n_blocks = len(model.blocks)
    P = generate_test_programs()
    id2label = {pid: lbl for lbl, pid in CASES}

    op_lea_d = dp["OP_LEA"]
    mark_ax_d = dp["MARK_AX"]
    out_lo_d = dp["OUTPUT_LO"]
    out_hi_d = dp["OUTPUT_HI_THIS_STEP"]
    alu_lo_d = dp["ALU_LO"]
    alu_hi_d = dp["ALU_HI"]
    fetch_lo_d = dp["FETCH_LO"]
    fetch_hi_d = dp["FETCH_HI"]

    def nib_win(vec, base):
        seg = vec[base:base + 16]
        i = int(torch.argmax(seg))
        return i, float(seg[i])

    def byteval(vec, lo_base, hi_base):
        l, lv = nib_win(vec, lo_base)
        h, hv = nib_win(vec, hi_base)
        return (h << 4) | l, l, h, lv, hv

    runner = FaithfulAutoregressiveRunner(model=model, layout=layout)
    inner = runner._inner

    # Probe the L8-ALU output block (physical 8..9 region) and the L25 tail
    # (n_blocks-2 = block 41 input, and the final block).
    probe_blocks = sorted(set(
        b for b in (8, 13, 40, n_blocks - 2, n_blocks - 1)
        if 0 <= b < n_blocks
    ))
    print(f"n_blocks={n_blocks} STEP={STEP} probe_blocks={probe_blocks}")

    for pid in want_ids:
        src, exp, desc = P[pid]
        label = id2label.get(pid, desc)
        bc, data = compile_c(src)
        prompt = inner._serial._build_context(list(bc), b"", [], "")
        pl = len(prompt)

        # AR-decode enough steps to cover both LEA references (params live in
        # steps ~ up to 20 for these bodies).
        ctx = list(prompt)
        emitted = []
        max_emit = args.max_steps * STEP
        while len(emitted) < max_emit:
            padded = torch.tensor([ctx + emitted], dtype=torch.long)
            with torch.no_grad():
                logits = model.forward(padded)
                if logits.is_sparse:
                    logits = logits.to_dense()
            nxt = int(torch.argmax(logits[0, -1]))
            emitted.append(nxt)
            if nxt == int(Token.HALT) if hasattr(Token, "HALT") else False:
                break
        full_ctx = ctx + emitted
        padded = torch.tensor([full_ctx], dtype=torch.long)

        # Cache per-block residual for the whole tape once per probe_block.
        block_res = {}
        for blk in probe_blocks:
            with torch.no_grad():
                full = model.forward(padded, stop_after_block=blk)
                if full.is_sparse:
                    full = full.to_dense()
            block_res[blk] = full[0]
        tail_res = block_res[n_blocks - 1]

        # Locate LEA AX rows on the DECODED tape: walk each step window, find the
        # MARK_AX row, check OP_LEA at the tail residual.
        nsteps = (len(full_ctx) - pl) // STEP
        lea_rows = []
        for step in range(nsteps):
            lo = pl + step * STEP
            hi = lo + STEP
            if hi > len(full_ctx):
                break
            seg = tail_res[lo:hi, mark_ax_d]
            ax_off = int(torch.argmax(seg))
            ax_pos = lo + ax_off
            # OP_LEA read at the L8 block (clean before tail overwrite).
            oplea = float(block_res[8][ax_pos, op_lea_d])
            if oplea > 0.5:
                lea_rows.append((step, ax_pos))

        print(f"\n===== {label}  exp={exp}  LEA rows: "
              f"{[s for s, _ in lea_rows]} =====")
        for idx, (step, ax_pos) in enumerate(lea_rows):
            tag = f"LEA#{idx} step{step}"
            print(f"  {tag} (ax_pos={ax_pos}):")
            for blk in probe_blocks:
                r = block_res[blk][ax_pos]
                alu_b, alu_l, alu_h, _, _ = byteval(r, alu_lo_d, alu_hi_d)
                out_b, out_l, out_h, _, _ = byteval(r, out_lo_d, out_hi_d)
                fe_b, _, _, _, _ = byteval(r, fetch_lo_d, fetch_hi_d)
                ah15 = float(r[alu_hi_d + 15])
                ah13 = float(r[alu_hi_d + 13])
                al0 = float(r[alu_lo_d + 0])
                oh14 = float(r[out_hi_d + 14])
                oh13 = float(r[out_hi_d + 13])
                oh0 = float(r[out_hi_d + 0])
                print(f"    b{blk:2d}: ALU=0x{alu_b:02x} OUT=0x{out_b:02x} "
                      f"FE=0x{fe_b:02x} | ALU_HI[15]={ah15:+7.2f} "
                      f"ALU_HI[13]={ah13:+7.2f} ALU_LO[0]={al0:+6.2f} "
                      f"| OUT_HI[14]={oh14:+6.2f} [13]={oh13:+6.2f} [0]={oh0:+6.2f}")
            final_b, _, final_h, _, _ = byteval(tail_res[ax_pos], out_lo_d, out_hi_d)
            print(f"    FINAL emitted OUTPUT byte0 = 0x{final_b:02x}")


if __name__ == "__main__":
    main()
