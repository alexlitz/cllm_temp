#!/usr/bin/env python3
"""HOOK-INJECT test: force the func step-0 JSR BP byte-3 predictor row's
OUTPUT_LO to nibble-0 (clear byte-3 to 0x00), then re-decode the teacher-forced
tape and report the NEW first value-correction step (vcorr). If vcorr advances
past step 0, BP-byte3 IS the first poisoning byte (root confirmed) and shows
which step becomes the next wall.

Injection is applied to the FINAL block output (post all blocks, pre-LM-head)
at the BP byte rows of the JSR step: set OUTPUT_LO+0 large-positive, OUTPUT_LO+j
large-negative for j!=0, on BYTE_INDEX_2 (byte-3 predictor) — i.e., clear byte-3.
Optionally also clear byte-1/byte-2 predictors if --allbytes.

  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/inj_$$ \
    python tools/_probe_funcmin_inject.py --ids 675,650,550,575
"""
from __future__ import annotations
import os, sys, contextlib, io, argparse

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
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()
REG_OFF = {"PC": 0, "AX": 5, "SP": 10, "BP": 15}
_VALUE_OFFS = set(range(1, 5)) | set(range(6, 10)) | set(range(11, 15)) | set(range(16, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675,650,550,575")
    ap.add_argument("--extra", type=str, default="",
                    help="extra rows to clear as step:regoff:byteidx, "
                         "comma-sep (e.g. 2:10:1 = step2 SP byte-2 pred)")
    args = ap.parse_args()
    extra_spec = []
    for tok3 in args.extra.split(","):
        if not tok3.strip():
            continue
        st, ro, bi = (int(x) for x in tok3.split(":"))
        extra_spec.append((st, ro, bi))
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    nblocks = len(model.blocks)
    out_lo = dp["OUTPUT_LO"]

    # We'll register a forward hook on the LAST block that rewrites the BP byte-3
    # predictor row of the JSR step (step 0) to nibble-0. Positions are set per
    # program before each forward.
    inj_positions = {"rows": []}

    def last_hook(m, i, o):
        o = o.clone()
        for pos in inj_positions["rows"]:
            o[0, pos, out_lo:out_lo + 16] = -50.0
            o[0, pos, out_lo + 0] = 50.0
        return o
    h = model.blocks[nblocks - 1].register_forward_hook(last_hook)

    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        prefix = len(prompt)
        ot = oracle_tape_and_steps(bc, b"", max_steps=48)
        tape = list(prompt) + list(ot.draft_tokens)
        # JSR step is step 0. BP byte-3 predictor row = prefix + 0*STEP + 15+1+2
        # (BYTE_INDEX_2 row predicts byte-3). Position of that token is 18.
        bp_b3_pred = prefix + 0 * STEP + REG_OFF["BP"] + 1 + 2

        # baseline (no injection)
        inj_positions["rows"] = []
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                base_logits = model.forward(torch.tensor([tape]))
        base_fa = base_logits.argmax(dim=-1).reshape(-1).tolist()

        # injected
        rows = [bp_b3_pred]
        for st, ro, bi in extra_spec:
            rows.append(prefix + st * STEP + ro + 1 + bi)
        inj_positions["rows"] = rows
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                inj_logits = model.forward(torch.tensor([tape]))
        inj_fa = inj_logits.argmax(dim=-1).reshape(-1).tolist()

        def vcorr(fa):
            for t in range(len(ot.draft_tokens)):
                if (t % STEP) not in _VALUE_OFFS:
                    continue
                if int(fa[prefix + t - 1]) != int(ot.draft_tokens[t]):
                    return t // STEP, t % STEP
            return None, None

        b_step, b_off = vcorr(base_fa)
        i_step, i_off = vcorr(inj_fa)
        print(f"\n== id{pid} {desc!r} exp={exp} ==")
        print(f"   BP_b3_pred pos(rel)={bp_b3_pred-prefix}  "
              f"base emit @that pos: got={base_fa[bp_b3_pred-1]} "
              f"inj emit: got={inj_fa[bp_b3_pred-1]}")
        print(f"   baseline vcorr: step={b_step} off={b_off}")
        print(f"   injected vcorr: step={i_step} off={i_off}"
              + ("   <<< ADVANCED" if (i_step or -1) > (b_step or -1) else
                 "   (no advance)"))

    h.remove()


if __name__ == "__main__":
    main()
