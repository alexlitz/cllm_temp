#!/usr/bin/env python3
"""Dump the raw emitted token stream for var_simple_0 around the step1->step2
boundary, annotated with the per-step 35-token frame structure, to find the
spurious 0xFF token that shifts step 2's PC marker."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token, _step_offset_field

SRC = "int main() { int x; x = 990; return x; }"


def main():
    p = build_groundtruth_probe()
    bc = compile_c(SRC)[0]
    plen = len(p._build_context(bc))
    ctx = p._final_context(bc, max_steps=12)

    print(f"prompt_len={plen} total={len(ctx)}")
    # Print tokens from start of step0 through step3, with offset field labels.
    # We don't know exact step boundaries (model may emit extra), so print
    # absolute index + token + a guess of the 35-frame offset.
    for i in range(plen, min(len(ctx), plen + 35 * 4)):
        rel = i - plen
        off = rel % 35
        step = rel // 35
        field = _step_offset_field(off)
        tok = ctx[i]
        mark = ""
        if tok in (257, 258, 259, 260, 261, 262, 268):
            mark = " <-MARKER"
        if tok == 255:
            mark = " <-0xFF"
        print(f"  idx{i} (step{step} off{off:2d} {field:13s}) tok={tok}{mark}")


if __name__ == "__main__":
    main()
