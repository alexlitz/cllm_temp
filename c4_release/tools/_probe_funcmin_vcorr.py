#!/usr/bin/env python3
"""Pinpoint the func_min/max STEP-0 value correction (the true root the gate's
vcorr=0 flags). Dumps, at every step up to the vcorr step, the model's argmax
draft token vs the DraftVM oracle for EACH register value byte offset, so we see
exactly which register/byte first diverges.

  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/funcvcorr_$$ \
    python tools/_probe_funcmin_vcorr.py --ids 675,650
"""
from __future__ import annotations
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_gate_context, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()

# Register value-byte offsets in a 30-token step slice.
REG_OFF = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675,650")
    ap.add_argument("--maxstep", type=int, default=6)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    ctx = build_gate_context(verbose=True)
    fwd = ctx.fwd

    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        prefix = len(prompt)
        ot = oracle_tape_and_steps(bc, b"", max_steps=48)
        full_ctx = list(prompt) + list(ot.draft_tokens)
        logits = fwd.forward(full_ctx)
        fa = logits.argmax(dim=-1).tolist()

        def pred(t):
            return int(fa[prefix + t - 1])

        print(f"\n===== id{pid} {desc!r} exp={exp} =====")
        print(f"  opcodes: {[hex(o) for o in ot.opcodes]}")
        nsteps = min(len(ot.steps), args.maxstep)
        for s in range(nsteps):
            base = s * STEP
            op = ot.opcodes[s] if s < len(ot.opcodes) else -1
            o_pc, o_ax = ot.steps[s]
            print(f"  -- step{s} op=0x{op:02x} oracle(pc={o_pc},ax={o_ax}) --")
            # For each register, compare the 4 value bytes token-by-token.
            for reg, moff in REG_OFF.items():
                diffs = []
                for k in range(4):
                    t = base + moff + 1 + k
                    p = pred(t)
                    o = int(ot.draft_tokens[t]) if t < len(ot.draft_tokens) else -1
                    mark = "" if p == o else " <-DIFF"
                    diffs.append(f"b{k}:got={p} want={o}{mark}")
                # Also the marker token at moff.
                tm = base + moff
                pm = pred(tm)
                om = int(ot.draft_tokens[tm]) if tm < len(ot.draft_tokens) else -1
                mk = "" if pm == om else " <-MARKDIFF"
                print(f"      {reg}: mark got={pm} want={om}{mk} | "
                      + "  ".join(diffs))
            # Also dump MEM addr/val region + STEP_END for completeness.
            mem = []
            for k in range(STEP):
                t = base + k
                p = pred(t)
                o = int(ot.draft_tokens[t]) if t < len(ot.draft_tokens) else -1
                if p != o and k not in (0, 5, 10, 15) and (k - 1) % 5 > 3:
                    mem.append(f"off{k}:got={p} want={o}")
            if mem:
                print(f"      OTHER-DIFFS: " + "  ".join(mem))


if __name__ == "__main__":
    main()
