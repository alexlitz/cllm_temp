#!/usr/bin/env python3
"""CPU-only probe of the AX byte-1 predictor row for edge_literal (byte1>=16).

Dumps the residual bands relevant to the byte-1 HIGH-nibble emission at the
faithful predictor row (prefix + s*35 + 6, which predicts the AX byte-1 token
at slice offset 7). NO GPU: reuses the interp_oracle_gate FaithfulForwardCache
residual (byte-for-byte == neural argmax). Resolves all dims via the BUILT
layout.dim_positions (the widen moves dims).

Run:
  CUDA_VISIBLE_DEVICES="" [C4_AX_BYTE1_FULL_WIDTH=1] python tools/probe_ax_b1_hinib_cpu.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
import warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps, STEP_TOKENS,
)


def band(resid, dp, name, width):
    base = dp[name]
    v = resid[base:base + width]
    am = int(v.argmax())
    return am, float(v[am]), [round(float(x), 2) for x in v[:width]]


def main():
    flag = os.environ.get("C4_AX_BYTE1_FULL_WIDTH", "0")
    fw_flag = os.environ.get("C4_AX_BYTE1_HINIB", "0")
    print(f"=== FULL_WIDTH={flag} HINIB={fw_flag} ===", flush=True)
    ctx = build_gate_context(verbose=True)
    dp = ctx.dim_positions
    fwd = ctx.fwd
    print(f"d_model={ctx.model.d_model} n_heads={ctx.model.blocks[0].attn.num_heads}",
          flush=True)
    have_wide = "AX_BYTE1_FULL_WIDE" in dp
    have_hinib = "AX_BYTE1_HINIB" in dp
    print(f"AX_BYTE1_FULL_WIDE={have_wide} AX_BYTE1_HINIB={have_hinib}", flush=True)

    VALUES = [5561, 9647, 5182, 9257]  # byte1 = 0x15,0x25,0x14,0x24
    for val in VALUES:
        bc, data = compile_c(f"int main() {{ return {val}; }}")
        ot = oracle_tape_and_steps(bc, data, max_steps=4)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        full = prompt + ot.draft_tokens
        resid = fwd._residual_pre_head(full)  # [S, d_model]
        logits = resid @ fwd.head_w.t() + fwd.head_b
        exp_b1 = (val >> 8) & 0xFF
        lo, hi = exp_b1 & 0xF, (exp_b1 >> 4) & 0xF
        print(f"\n--- return {val} byte1=0x{exp_b1:02x} (lo={lo} hi={hi}) ---",
              flush=True)
        for s in range(min(2, len(ot.steps))):
            row = prefix + s * STEP_TOKENS + 6  # predictor for AX byte-1 token
            r = resid[row]
            pred = int(logits[row].argmax())
            opimm = float(r[dp["OP_IMM"]])
            isbyte = float(r[dp["IS_BYTE"]])
            markax = float(r[dp["MARK_AX"]])
            acl = band(r, dp, "AX_CARRY_LO", 16)
            ach = band(r, dp, "AX_CARRY_HI", 16)
            h1 = band(r, dp, "H1", 7)
            h3 = band(r, dp, "H3", 7)
            print(f"  step{s} pred=0x{pred:02x} | OP_IMM={opimm:.2f} "
                  f"IS_BYTE={isbyte:.2f} MARK_AX={markax:.2f}", flush=True)
            print(f"     AC_LO am={acl[0]} v={acl[1]:.2f} | "
                  f"AC_HI am={ach[0]} v={ach[1]:.2f} | "
                  f"H1 am={h1[0]} v={h1[1]:.2f} | H3 am={h3[0]} v={h3[1]:.2f}",
                  flush=True)
            for dname in ("H1_DUMP_OUT", "H2_DUMP_OUT", "H3_DUMP_OUT",
                          "AX_CARRY_OVERFLOW"):
                if dname in dp:
                    w = 7 if dname != "AX_CARRY_OVERFLOW" else 1
                    bb = band(r, dp, dname, w)
                    print(f"     {dname} am={bb[0]} v={bb[1]:.2f} all={bb[2]}",
                          flush=True)
            if have_wide:
                w = r[dp["AX_BYTE1_FULL_WIDE"]:dp["AX_BYTE1_FULL_WIDE"] + 256]
                wam = int(w.argmax())
                print(f"     FULL_WIDE am={wam} v={float(w[wam]):.2f} "
                      f"cell[{exp_b1}]={float(w[exp_b1]):.2f}", flush=True)
            if have_hinib:
                hb = r[dp["AX_BYTE1_HINIB"]:dp["AX_BYTE1_HINIB"] + 16]
                hbam = int(hb.argmax())
                print(f"     HINIB am={hbam} v={float(hb[hbam]):.2f} "
                      f"cell[{hi}]={float(hb[hi]):.2f}", flush=True)
            top = torch.topk(logits[row], 5)
            pairs = [(int(i), round(float(v), 1))
                     for i, v in zip(top.indices, top.values)]
            print(f"     top5: {pairs}", flush=True)


if __name__ == "__main__":
    main()
