#!/usr/bin/env python3
"""Pin the EXACT step-1 value-byte correction for the if/bool cluster.

The ``interp_oracle_gate`` flags every if/bool literal + if_var program as
CROSS-STEP with ``value-corruption@step1`` — i.e. some register-VALUE byte the
model emits at VM **step 1** diverges from the DraftVM oracle tape, poisoning
the autoregressive frame from there on. This probe reuses the gate's faithful
forward + oracle tape to print, for each requested corpus id, the FIRST
diverging value offset at step 1: the step-relative offset, the register it
belongs to (PC/AX/SP/BP), the byte index within that register, and the
expected-vs-got token. That names the value root so a fix can be built on the
owning block/rule.

Usage::

    CUDA_VISIBLE_DEVICES="" C4_CAMPAIGN=1 python tools/probe_ifbool_step1_vcorr.py \
        --ids 350,375,400,425,426,430
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")

from c4_release.tools.interp_oracle_gate import (  # noqa: E402
    STEP_TOKENS,
    _REG_OFFSETS,
    _VALUE_OFFSETS,
    build_code_prompt,
    build_gate_context,
    oracle_tape_and_steps,
)


# Reverse map: step-relative value offset -> (register, byte-index).
_OFF_TO_REGBYTE = {}
for _reg, _moff in _REG_OFFSETS.items():
    for _b in range(4):
        _OFF_TO_REGBYTE[_moff + 1 + _b] = (_reg, _b)


def _parse_ids(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-steps", type=int, default=48)
    args = ap.parse_args(argv)

    ids = _parse_ids(args.ids)

    from c4_release.tests.test_suite_1000 import generate_test_programs
    from c4_release.src.compiler import compile_c

    tests = generate_test_programs()
    ctx = build_gate_context(verbose=True)

    for i in ids:
        src, expected, desc = tests[i]
        bc, data = compile_c(src)
        ot = oracle_tape_and_steps(bc, data, max_steps=args.max_steps)
        prompt = build_code_prompt(bc, data)
        prefix = len(prompt)
        full_ctx = prompt + ot.draft_tokens
        logits = ctx.fwd.forward(full_ctx)
        fa = logits.argmax(dim=-1).tolist()

        def pred_tok(t):
            return int(fa[prefix + t - 1])

        print(f"\n=== id={i} {desc}  expect={expected}  n_steps={len(ot.steps)} ===")
        # opcode per step
        opcodes = ot.opcodes
        # Walk every draft token; report EVERY value-offset divergence, tagged by step.
        n_div = 0
        for t in range(len(ot.draft_tokens)):
            off = t % STEP_TOKENS
            if off not in _VALUE_OFFSETS:
                continue
            got = int(pred_tok(t))
            exp = int(ot.draft_tokens[t])
            if got == exp:
                continue
            step = t // STEP_TOKENS
            reg, bidx = _OFF_TO_REGBYTE.get(off, ("?", -1))
            opc = opcodes[step] if step < len(opcodes) else -1
            print(f"  step={step:2d} off={off:2d} {reg}[{bidx}] "
                  f"exp_tok=0x{exp:02x} got_tok=0x{got:02x}  (opcode={opc})")
            n_div += 1
            if n_div >= 12:
                print("  ... (truncated)")
                break
        if n_div == 0:
            print("  (no value-offset divergence — single-forward faithful)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
