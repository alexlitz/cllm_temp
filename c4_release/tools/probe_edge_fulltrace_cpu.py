#!/usr/bin/env python3
"""CPU full_trace decode of edge_literal vs DraftVM oracle (faithful, NO GPU).

Reproduces the production per-step (PC, AX) decode from the FaithfulForwardCache
argmax over the teacher-forced tape (byte-identical to neural), at the fixed
re-anchored offsets, and compares to the DraftVM oracle -- the same full_trace
criterion the gate uses, but printing the full per-step (PC, AX) so a CROSS-STEP
deferral can be inspected directly. Honors C4_AX_BYTE1_HINIB.

Run: CUDA_VISIBLE_DEVICES="" [C4_AX_BYTE1_HINIB=1] python tools/probe_edge_fulltrace_cpu.py
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

from src.compiler import compile_c  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps, STEP_TOKENS,
    _REG_OFFSETS, _decode_reg_fixed,
)

VALUES = [5561, 9647, 5182, 9257, 5831, 6521, 9849, 5172, 6271, 5481,
          2563, 1435, 1488]  # edge_literal cluster incl byte1<16 controls


def main():
    flag = os.environ.get("C4_AX_BYTE1_HINIB", "0")
    print(f"=== C4_AX_BYTE1_HINIB={flag} ===", flush=True)
    ctx = build_gate_context(verbose=True)
    fwd = ctx.fwd
    npass = 0
    for val in VALUES:
        bc, data = compile_c(f"int main() {{ return {val}; }}")
        ot = oracle_tape_and_steps(bc, data, max_steps=6)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        full = prompt + ot.draft_tokens
        logits = fwd.forward(full)
        fa = logits.argmax(dim=-1).tolist()

        def pred_tok(t):
            return int(fa[prefix + t - 1])

        ok = True
        detail = []
        for s in range(len(ot.steps)):
            base = s * STEP_TOKENS
            slc = [pred_tok(base + k) for k in range(STEP_TOKENS)]
            o_pc, o_ax = ot.steps[s]
            g_pc = _decode_reg_fixed(slc, _REG_OFFSETS["PC"])
            g_ax = _decode_reg_fixed(slc, _REG_OFFSETS["AX"])
            step_ok = (g_pc == o_pc and g_ax == o_ax)
            ok = ok and step_ok
            detail.append(
                f"s{s}:{'ok' if step_ok else 'X'} PC {g_pc}/{o_pc} AX {g_ax}/{o_ax}")
        npass += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'} return {val:5d} (b1=0x{(val>>8)&0xff:02x}): "
              + " | ".join(detail), flush=True)
    print(f"\n{npass}/{len(VALUES)} full_trace PASS", flush=True)


if __name__ == "__main__":
    main()
