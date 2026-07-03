#!/usr/bin/env python3
"""Read the discriminating residual dims at the SP byte-0 row of binary-pop
steps, in the TRUE autoregressive context, for the expr_mod SP carry case (#319).

Compares a PUSH step (byte0 must stay 0xF8) vs the CARRY pop step (byte0 must
become 0x00) so we can find a dim that cleanly separates them and gate the
strengthened carry write WITHOUT breaking the non-carry SP rows.

Dumps the final-residual value of a battery of marker/CMP/OUTPUT dims at the SP
byte-0 row (markers persist to the final residual). Resolves dims via BUILT
layout.dim_positions.

Usage:
    C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
    C4_VM_CACHE_DIR=/tmp/c4cache_spcarry \
    python tools/probe_exprmod_sp_dims.py --id 876 --steps 1,3,4,6
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

PROBE_DIMS = [
    "IS_BYTE", "HAS_SE", "H1+2", "CMP+0", "CMP+2", "CMP+3", "CMP+4",
    "MARK_SP", "PSH_AT_SP", "MEM_STORE", "OP_MOD", "OP_LEV", "OP_ADJ",
    "EMBED_LO+0", "EMBED_HI+0", "EMBED_LO+8", "EMBED_HI+15", "EMBED_HI+13",
    "EMBED_HI+14",
    "OUTPUT_LO+0", "OUTPUT_LO+8", "OUTPUT_HI_THIS_STEP+0",
    "OUTPUT_HI_THIS_STEP+15", "OUTPUT_HI_THIS_STEP+14",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--steps", default="1,3")
    ap.add_argument("--max-steps", type=int, default=14)
    args = ap.parse_args()
    want_steps = [int(x) for x in args.steps.split(",")]

    progs = generate_test_programs()
    src, exp, desc = progs[args.id]
    bc = compile_c(src)[0]

    runner = FaithfulAutoregressiveRunner(disk_cache=True)
    inner = runner._inner
    STEP = int(Token.STEP_TOKENS)
    dimpos = runner.dim_positions

    def resolve(name):
        if "+" in name:
            base, off = name.rsplit("+", 1)
            if base in dimpos:
                return dimpos[base] + int(off)
            return None
        return dimpos.get(name)

    cols = {}
    miss = []
    for d in PROBE_DIMS:
        c = resolve(d)
        if c is None:
            miss.append(d)
        else:
            cols[d] = c
    if miss:
        print("WARN dims not in built layout:", miss)

    final_ctx = {"context": None, "prefix_len": None}
    orig_check = inner._ff_check_new_steps

    def patched_check(s, *, criterion="full_trace"):
        final_ctx["context"] = list(s.context)
        final_ctx["prefix_len"] = s.prefix_len
        return orig_check(s, criterion=criterion)

    inner._ff_check_new_steps = patched_check
    inner.run_batch_fail_fast(
        [bc], expected_steps_list=[None], max_steps=args.max_steps,
        spec_k=0, criterion="full_trace",
    )

    ctx = final_ctx["context"]
    pfx = final_ctx["prefix_len"]
    faithful = inner._faithful

    print(f"id {args.id}: {desc.strip()} exp={exp}  prefix_len={pfx}")
    rows = {}  # step -> {dim: val}
    for step_idx in want_steps:
        start = pfx + step_idx * STEP
        if start + STEP > len(ctx):
            continue
        slice_toks = ctx[start:start + STEP]
        sp_off = next((i for i, tk in enumerate(slice_toks)
                       if tk == int(Token.REG_SP)), None)
        if sp_off is None:
            continue
        byte0_pos = start + sp_off + 1
        emitted = ctx[byte0_pos]
        # The SP byte0 TOKEN is PREDICTED by the residual at position
        # byte0_pos-1 (the SP marker row) -- that is the row whose FFN OUTPUT
        # band drives the LM-head argmax for the byte0 token (confirmed by the
        # logit probe: forward(ctx[:byte0_pos])[byte0_pos-1] == emitted). Read
        # the FFN-rule input dims at THAT predicting row.
        prefix = ctx[:byte0_pos]
        resid = faithful.forward(list(prefix), return_logits=False)
        row = resid[byte0_pos - 1]  # the SP marker / predicting row
        rows[step_idx] = (emitted, {d: float(row[c]) for d, c in cols.items()})

    # tabular print
    hdr = "dim".ljust(24) + "".join(f"s{st}".rjust(12) for st in rows)
    print(hdr)
    print("emitted".ljust(24) + "".join(
        f"0x{rows[st][0]:02X}".rjust(12) for st in rows))
    for d in PROBE_DIMS:
        if d not in cols:
            continue
        line = d.ljust(24)
        for st in rows:
            line += f"{rows[st][1][d]:+.3f}".rjust(12)
        print(line)


if __name__ == "__main__":
    main()
