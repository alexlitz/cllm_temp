#!/usr/bin/env python3
"""LM-head logit attribution for the func step-0 BP byte-3 leak (token 1 vs 0).

At the BP byte-3 value position (slice off 19), compute which FINAL-residual
dims drive logit[1] over logit[0] via (W_head[1,d]-W_head[0,d])*resid[d].
Names the dims via reverse dim_positions.

  C4_CAMPAIGN=1 C4_VM_CACHE_DIR=/tmp/bpb3l_$$ \
    python tools/_probe_bp_b3_logit_attr.py --id 675 --step 0 --byte 3
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=675)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--byte", type=int, default=3)
    ap.add_argument("--reg", type=str, default="BP")
    ap.add_argument("--want", type=int, default=0)
    ap.add_argument("--got", type=int, default=1)
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    inv = {}
    for name, off in dp.items():
        inv.setdefault(off, name)
    nblocks = len(model.blocks)
    W = model.head.weight.detach()   # (276, d)
    b = model.head.bias.detach()

    fin = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: fin.__setitem__("o", o.detach().clone()))

    pid = args.id
    src, exp, desc = PROGS[pid]
    bc = compile_c(src)[0]
    prompt = build_code_prompt(bc, b"")
    ot = oracle_tape_and_steps(bc, b"", max_steps=40)
    tape = list(prompt) + list(ot.draft_tokens)
    tok = torch.tensor([tape], dtype=torch.long)
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            model.forward(tok)
    plen = len(prompt)
    # value byte position: the token that PREDICTS byte j is at pos-1.
    moff = REG_OFF[args.reg]
    valpos = plen + args.step * STEP + moff + 1 + args.byte
    predpos = valpos - 1   # logits at predpos predict token at valpos
    r = fin["o"][0, predpos]   # (d,)

    logit_want = float((W[args.want] @ r + b[args.want]).item())
    logit_got = float((W[args.got] @ r + b[args.got]).item())
    print(f"id{pid} {desc!r} {args.reg} byte{args.byte} predpos={predpos} "
          f"(valpos {valpos})")
    print(f"  logit[want={args.want}]={logit_want:.4g} "
          f"logit[got={args.got}]={logit_got:.4g} "
          f"(margin got-want={logit_got-logit_want:.4g})")
    # per-dim contribution to (logit_got - logit_want)
    diff_w = (W[args.got] - W[args.want])   # (d,)
    contrib = diff_w * r                    # (d,)
    order = torch.argsort(contrib, descending=True)
    print(f"  Top dims driving got>want (logit[{args.got}]-logit[{args.want}]):")
    for i in range(25):
        d = int(order[i].item())
        c = float(contrib[d].item())
        if abs(c) < 1e-6:
            break
        nm = inv.get(d, f"dim{d}")
        # find base name + offset
        base = None; boff = None
        for name, o0 in dp.items():
            # match category bands of width up to 16/32
            if o0 <= d < o0 + 32 and (base is None or o0 > (dp.get(base, -1))):
                base = name; boff = d - o0
        print(f"    dim{d:4d} {nm:>22s} base={base}+{boff} "
              f"contrib={c:+.4g} resid={float(r[d].item()):+.4g} "
              f"dW={float(diff_w[d].item()):+.4g}")


if __name__ == "__main__":
    main()
