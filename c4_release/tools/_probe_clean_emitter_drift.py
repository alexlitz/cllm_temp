#!/usr/bin/env python3
"""FAST AR-oracle framing-drift tape dump (step-capped) for the CLEAN_EMITTER scope.

A step-capped sibling of ``tools/probe_framing_drift_tape.py``. The full probe
runs the fail-fast decode to HALT (``max_steps=None``) which, on CPU with no KV
cache, is O(steps^2) and times out under GPU contention for the ~8-10 step
framing-drift clusters. The ``!= STEP_TOKENS`` drift, however, appears at the
FIRST comparison/store/post-ENT step (step 2-4), so a small ``--max-steps`` cap
reaches it in a few forwards.

It runs the SAME production ``FaithfulAutoregressiveRunner._run_fail_fast`` (the
byte-exact CPU autoregressive path that reproduces the 30/31/34-token miscount),
then re-segments the emitted ``s.context`` tape by REG_PC markers to expose the
step whose emitted length != STEP_TOKENS and the spurious marker/byte row that
won where the NEXT_* marker should have.

Usage (CPU only):
    python tools/_probe_clean_emitter_drift.py --ids 350 --campaign --max-steps 6
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

_MARKERS = {257: "REG_PC", 258: "REG_AX", 259: "REG_SP", 260: "REG_BP",
            261: "MEM", 262: "STEP_END", 263: "HALT", 268: "STACK0",
            264: "TOOL_CALL"}


def _annotate(tok):
    from neural_vm.batched_pure_neural import Token
    if tok == Token.STEP_END:
        return "STEP_END"
    if tok == Token.HALT:
        return "HALT"
    if tok in _MARKERS:
        return f"<{_MARKERS[tok]}>"
    return str(tok)


def _dump_tape(idx, desc, tape, prompt_len, step_tokens):
    from neural_vm.batched_pure_neural import _step_offset_field, Token
    gen = tape[prompt_len:]
    print(f"\n{'='*70}\nid={idx} {desc}")
    print(f"prompt_len={prompt_len} generated={len(gen)} "
          f"(={len(gen)/step_tokens:.2f} x STEP_TOKENS={step_tokens})")

    print("\n-- RE-SEGMENTED BY REG_PC MARKERS (true emitted step lengths) --")
    seg_starts = [i for i, t in enumerate(gen) if t == Token.REG_PC]
    seg_starts.append(len(gen))
    for k in range(len(seg_starts) - 1):
        a, b = seg_starts[k], seg_starts[k + 1]
        seg = gen[a:b]
        n = len(seg)
        flag = "" if n == step_tokens else f"  <<< emits {n} != {step_tokens}"
        spurious = []
        for j, t in enumerate(seg):
            if j == 0:
                continue
            if t in (Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP,
                     261, 262, 268):
                fld = _step_offset_field(j)
                if fld not in ("AX_marker", "SP_marker", "BP_marker",
                               "STACK0_marker", "MEM_marker", "STEP_END/HALT"):
                    spurious.append((j, _annotate(t), fld))
        toks = " ".join(_annotate(t) for t in seg)
        print(f"  seg{k} (len={n}){flag}\n     {toks}")
        if spurious:
            for j, a2, fld in spurious:
                print(f"       *** marker at off{j} where a VALUE byte was due "
                      f"(field={fld}): {a2}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--campaign", action="store_true")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-ctx", type=int, default=256)
    args = ap.parse_args(argv)

    if args.campaign:
        os.environ["C4_CAMPAIGN"] = "1"
        os.environ["C4_NO_STACK0_EMIT"] = "1"
        os.environ["C4_OPERAND_FROM_MEMSP"] = "1"

    from tools.run_1096_fast import _parse_ids, _compile_and_oracle
    from tests.test_suite_1000 import generate_test_programs
    from neural_vm.verification.faithful_autoregressive import (
        FaithfulAutoregressiveRunner,
    )
    from neural_vm.batched_pure_neural import Token

    step_tokens = int(Token.STEP_TOKENS)
    print(f"[probe] STEP_TOKENS={step_tokens} campaign={args.campaign} "
          f"max_steps={args.max_steps}", flush=True)

    all_tests = generate_test_programs()
    wanted = list(dict.fromkeys(_parse_ids(args.ids)))
    selected = [(i, all_tests[i][0], all_tests[i][1], all_tests[i][2])
                for i in wanted if i < len(all_tests)]
    prepared, _errs = _compile_and_oracle(selected)

    print("[probe] building CPU faithful runner...", file=sys.stderr, flush=True)
    runner = FaithfulAutoregressiveRunner()
    inner = runner._inner
    for entry in prepared:
        idx, _exp, desc, _decl_exit, decl_steps, bytecode, data = entry
        s = inner._build_element(
            bytecode, data, [], "",
            spec_k=1, adaptive_start_k=0, expected_steps=decl_steps)
        s.ff_oracle_steps = inner._oracle_pc_ax_steps(
            bytecode, data, "", expected_steps=decl_steps)
        prompt_len = len(s.context)
        inner._run_fail_fast(
            [s], max_steps=args.max_steps, max_context_window=args.max_ctx,
            spec_k=1, criterion="full_trace")
        _dump_tape(idx, desc, s.context, prompt_len, step_tokens)
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
