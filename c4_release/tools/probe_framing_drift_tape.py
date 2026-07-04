#!/usr/bin/env python3
"""Dump the RAW autoregressive token tape (CPU, bit-exact) for a framing-drift
program and locate the step that emits != STEP_TOKENS tokens.

This is the diagnostic for task #388 (the != STEP_TOKENS framing-drift root).
It runs the PRODUCTION fail-fast decode via ``FaithfulAutoregressiveRunner``
(the byte-exact CPU autoregressive path that reproduces the 34/37-token
miscount) but captures the full emitted ``s.context`` tape, then:

  1. splits the tape into STEP_TOKENS-strided frames (the fixed-stride slicer's
     view) and prints each frame annotated with ``_step_offset_field``;
  2. RE-SEGMENTS the tape by REG_PC markers (offset-0 markers), so a frame with
     != STEP_TOKENS real tokens is exposed: the step whose emitted length is
     not STEP_TOKENS is the framing-drift step, and the token at the drifted
     position (a spurious register MARKER where a value byte should be) is the
     culprit.

Because the runner teacher-forces the UNSAFE MEM offsets (21..28 in the
30-token layout) from the DraftVM oracle, a spurious marker can only appear at
a SAFE offset (a PC/AX/SP/BP/STACK0 value byte). That is exactly the
framing-drift mechanism: a value byte fails -> the model emits a MARKER token
-> the next real step's PC marker lands at the wrong absolute offset.

Usage (CPU only, ~85s/program):
    python tools/probe_framing_drift_tape.py --ids 427,436,443 --campaign
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

    # ---- View 1: fixed-stride frames (the slicer's view) ----------------
    print(f"\n-- FIXED-STRIDE FRAMES (stride={step_tokens}) --")
    n_frames = (len(gen) + step_tokens - 1) // step_tokens
    for f in range(n_frames):
        frame = gen[f * step_tokens:(f + 1) * step_tokens]
        # A well-framed step has REG_PC at offset 0.
        head = frame[0] if frame else None
        pc_ok = (head == Token.REG_PC)
        flag = "" if pc_ok else "  <<< off0 is NOT REG_PC (DRIFT)"
        toks = " ".join(_annotate(t) for t in frame)
        print(f"  step{f}{flag}\n     {toks}")

    # ---- View 2: re-segment by REG_PC markers (true step boundaries) ----
    print(f"\n-- RE-SEGMENTED BY REG_PC MARKERS (true emitted step lengths) --")
    seg_starts = [i for i, t in enumerate(gen) if t == Token.REG_PC]
    seg_starts.append(len(gen))
    for k in range(len(seg_starts) - 1):
        a, b = seg_starts[k], seg_starts[k + 1]
        seg = gen[a:b]
        n = len(seg)
        flag = "" if n == step_tokens else f"  <<< emits {n} != {step_tokens}"
        # Identify offset of any spurious marker inside the segment (not off0).
        spurious = []
        for j, t in enumerate(seg):
            if j == 0:
                continue
            if t in (Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP,
                     268):
                fld = _step_offset_field(j)
                if fld not in ("AX_marker", "SP_marker", "BP_marker",
                               "STACK0_marker", "MEM_marker"):
                    spurious.append((j, _annotate(t), fld))
        toks = " ".join(_annotate(t) for t in seg)
        print(f"  seg{k} (len={n}){flag}\n     {toks}")
        if flag and spurious:
            for j, a2, fld in spurious:
                print(f"       spurious marker at off{j} "
                      f"(expected field={fld}): {a2}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True,
                    help="comma/range ids (run_1096 syntax)")
    ap.add_argument("--campaign", action="store_true",
                    help="set the 30-token campaign config env")
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
    print(f"[probe] STEP_TOKENS={step_tokens} campaign={args.campaign}")

    all_tests = generate_test_programs()
    wanted = list(dict.fromkeys(_parse_ids(args.ids)))
    selected = [(i, all_tests[i][0], all_tests[i][1], all_tests[i][2])
                for i in wanted if i < len(all_tests)]
    prepared, _errs = _compile_and_oracle(selected)

    print("[probe] building CPU faithful runner...", file=sys.stderr, flush=True)
    runner = FaithfulAutoregressiveRunner()

    # Build one element per program so we can grab its context tape after decode.
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
            [s], max_steps=None, max_context_window=512, spec_k=1,
            criterion="full_trace")
        _dump_tape(idx, desc, s.context, prompt_len, step_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
