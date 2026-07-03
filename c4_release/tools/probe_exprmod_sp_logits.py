#!/usr/bin/env python3
"""Dump the SP byte-0 LM-head logit race at the binary-pop CARRY step of a
failing expr_mod, in the TRUE autoregressive context (#319).

Runs the FaithfulAutoregressiveRunner to build the real AR context up to the
divergence, then re-runs ONE faithful forward and dumps the top-K token logits
at the SP byte-0 row of the requested step. Reveals which token (0x00 carry
value vs 0xF8 stale / 0xF0 decrement) wins and by how much.

Usage:
    C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_spcarry \
    python tools/probe_exprmod_sp_logits.py --id 876 --steps 3,4,6
"""
from __future__ import annotations
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")

from neural_vm.verification.faithful_autoregressive import (  # noqa: E402
    FaithfulAutoregressiveRunner,
)
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402

MARKERS = {
    int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
    int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
    int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT",
}


def lbl(t):
    t = int(t)
    if t in MARKERS:
        return MARKERS[t]
    if t < 256:
        return f"0x{t:02X}"
    return f"tok{t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--steps", default="3")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()
    want_steps = [int(x) for x in args.steps.split(",")]

    progs = generate_test_programs()
    src, exp, desc = progs[args.id]
    bc = compile_c(src)[0]

    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    inner = runner._inner
    STEP = int(Token.STEP_TOKENS)

    final_ctx = {"context": None, "prefix_len": None}
    orig_check = inner._ff_check_new_steps

    def patched_check(s, *, criterion="full_trace"):
        final_ctx["context"] = list(s.context)
        final_ctx["prefix_len"] = s.prefix_len
        return orig_check(s, criterion=criterion)

    inner._ff_check_new_steps = patched_check

    verdicts = inner.run_batch_fail_fast(
        [bc], expected_steps_list=[None], max_steps=args.max_steps,
        spec_k=0, criterion="full_trace",
    )
    v = verdicts[0]
    print(f"id {args.id}: {desc.strip()} exp={exp}")
    print(f"  verdict={v.get('status')} div_step={v.get('divergence_step')} "
          f"exp_pc={v.get('expected_pc')} got_pc={v.get('got_pc')}")

    ctx = final_ctx["context"]
    pfx = final_ctx["prefix_len"]
    print(f"  prefix_len={pfx} total_ctx={len(ctx)} STEP={STEP}")

    faithful = inner._faithful
    for step_idx in want_steps:
        start = pfx + step_idx * STEP
        if start + STEP > len(ctx):
            print(f"  step {step_idx}: beyond emitted context")
            continue
        slice_toks = ctx[start:start + STEP]
        # find SP marker within slice
        sp_off = None
        for i, tk in enumerate(slice_toks):
            if tk == int(Token.REG_SP):
                sp_off = i
                break
        if sp_off is None:
            print(f"  step {step_idx}: no SP marker in slice")
            continue
        print(f"\n  === step {step_idx} slice (SP marker at intra-step off "
              f"{sp_off}) ===")
        print("    raw slice:", " ".join(lbl(t) for t in slice_toks))
        # SP byte-0 row is the position right AFTER the SP marker; its logits
        # are computed from the prefix ENDING at the SP marker position.
        for b in range(4):
            byte_pos = start + sp_off + 1 + b  # absolute pos of SP byte b
            # logits predicting token at byte_pos come from forward over
            # ctx[:byte_pos]
            prefix = ctx[:byte_pos]
            logits = faithful.forward(list(prefix), return_logits=True)
            row = logits[len(prefix) - 1]
            top = sorted(
                [(float(row[t]), int(t)) for t in range(row.shape[-1])],
                reverse=True,
            )[: args.topk]
            emitted = ctx[byte_pos]
            print(f"    SP byte{b} @pos {byte_pos} emitted={lbl(emitted)}:")
            for val, t in top:
                tag = ""
                if t == emitted:
                    tag = " <-- emitted"
                if t == 0x00:
                    tag += " [carry 0x00]"
                if t == 0xF8:
                    tag += " [0xF8]"
                if t == 0xF0:
                    tag += " [0xF0]"
                print(f"        {lbl(t):<10} {val:+.3f}{tag}")


if __name__ == "__main__":
    main()
