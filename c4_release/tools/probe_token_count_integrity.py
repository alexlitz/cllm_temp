#!/usr/bin/env python3
"""GPU token-count integrity check for the != STEP_TOKENS framing-drift claim.

For each requested program, replay the raw spec_k=0 argmax loop (the model's OWN
argmax at EVERY position, incl. MEM bytes -- no teacher-forcing) and re-segment
the emitted tape by REG_PC markers. A step whose segment length != STEP_TOKENS
is a genuine framing-drift step (a value byte degenerated to a spurious marker).
Reports, per program: #steps, whether ALL segments == STEP_TOKENS, and lists any
drifted segment (length + the spurious marker offset).

This directly tests the task #388 premise. Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_token_count_integrity.py --ids 425-449,250-274,300-324,0-9,50-59
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import warnings  # noqa: E402

warnings.filterwarnings("ignore")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args(argv)

    from tools.run_1096_fast import _parse_ids
    from tests.test_suite_1000 import generate_test_programs
    from tools.probe_groundtruth import build_groundtruth_probe
    from src.compiler import compile_c
    from neural_vm.batched_pure_neural import Token, _step_offset_field

    step_tokens = int(Token.STEP_TOKENS)
    print(f"[integrity] STEP_TOKENS={step_tokens}")

    probe = build_groundtruth_probe()
    all_tests = generate_test_programs()
    wanted = list(dict.fromkeys(_parse_ids(args.ids)))

    n_prog = 0
    n_clean = 0
    n_drift = 0
    drift_ids = []
    for idx in wanted:
        if idx >= len(all_tests):
            continue
        src, exp, desc = all_tests[idx]
        try:
            bc = compile_c(src)[0]
        except Exception as e:  # noqa: BLE001
            print(f"  id={idx} COMPILE-ERR {e}")
            continue
        ctx = probe._final_context(bc, max_steps=args.max_steps)
        prompt_len = len(probe._build_context(bc))
        gen = ctx[prompt_len:]
        # Re-segment by REG_PC markers (true step boundaries).
        seg_starts = [i for i, t in enumerate(gen) if t == Token.REG_PC]
        seg_starts.append(len(gen))
        seg_lens = [seg_starts[k + 1] - seg_starts[k]
                    for k in range(len(seg_starts) - 1)]
        # Ignore a trailing partial segment (decode may stop mid-step at halt).
        bad = [(k, L) for k, L in enumerate(seg_lens)
               if L != step_tokens and k < len(seg_lens) - 1]
        n_prog += 1
        if not bad:
            n_clean += 1
            continue
        n_drift += 1
        drift_ids.append(idx)
        print(f"  id={idx} DRIFT {desc}")
        print(f"     seg_lens={seg_lens}")
        for k, L in bad:
            a = seg_starts[k]
            seg = gen[a:a + max(L, step_tokens)]
            marks = [(j, int(seg[j])) for j in range(1, min(len(seg), step_tokens))
                     if int(seg[j]) in (257, 258, 259, 260, 268)]
            print(f"     seg{k} len={L}: spurious markers at "
                  f"{[(j, _step_offset_field(j)) for j, _ in marks]}")

    print(f"\n[integrity] {n_prog} programs: {n_clean} clean (all segs=="
          f"{step_tokens}), {n_drift} with framing drift.")
    if drift_ids:
        print(f"[integrity] DRIFT ids: {drift_ids}")
    else:
        print(f"[integrity] NO framing drift found -- every step emits exactly "
              f"{step_tokens} tokens across all {n_prog} programs.")


if __name__ == "__main__":
    raise SystemExit(main())
