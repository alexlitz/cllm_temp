#!/usr/bin/env python3
"""Dump raw token stream for var_simple_12 (id 262) steps 2-5, and the
per-position LM-head argmax + top tokens, to understand the step-3/4 structure
(SP byte3 leak + doubled step-4 register emission)."""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

MARKERS = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 262: "STEP_END",
           263: "HALT", 268: "?268"}

probe = build_groundtruth_probe()
m = probe.model
dev = next(m.parameters()).device
tests = generate_test_programs()
src, exp, desc = tests[262]
bc, data = compile_c(src)
ctx = probe._final_context(bc)
prompt_len = len(probe._build_context(bc))

padded = torch.tensor([ctx], dtype=torch.long, device=dev)
with torch.no_grad():
    logits = m.forward(padded)[0].float()

# Print positions 198..290 (steps 2 end through step 5)
print(f"prompt_len={prompt_len} total={len(ctx)}")
print(f"{'pos':>4} {'tok':>4} {'name':>8}  {'argmax':>6} {'top3'}")
for p in range(198, min(290, len(ctx))):
    tok = ctx[p]
    nm = MARKERS.get(tok, f"b{tok}" if tok < 256 else str(tok))
    # row p-1 predicts token at p
    if p - 1 >= 0:
        row = logits[p - 1]
        am = int(row.argmax())
        tk = torch.topk(row, 3)
        top3 = [(int(t), round(float(v), 1)) for v, t in zip(tk.values, tk.indices)]
    else:
        am = -1; top3 = []
    print(f"{p:>4} {tok:>4} {nm:>8}  {am:>6} {top3}")
