#!/usr/bin/env python3
"""Emit the autoregressive spec_k=0 token stream for id262 and dump the first N
emitted register-byte tokens with their positions, so we can diff HEAD vs the
SP-d8 fix and find the FIRST token that changed. Run twice (with/without fix).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

probe = build_groundtruth_probe()
src, exp, _ = generate_test_programs()[262]
bc = compile_c(src)[0]
ctx = probe._final_context(bc, max_steps=9)
pl = len(probe._build_context(bc))
print(f"id262 exp={exp} prompt_len={pl} total={len(ctx)}")
# dump the emitted suffix (post-prompt) as token ids
print("EMITTED_SUFFIX=" + ",".join(str(t) for t in ctx[pl:pl+90]))
