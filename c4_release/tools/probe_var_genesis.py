#!/usr/bin/env python3
"""Find the GENESIS block+position of the dim-85/79 explosion for var_simple_12.

For each physical block, report the global max|val| over (pos, watched-dims)
and where it occurs. Pins the first block where magnitude crosses ~1e3 -> 1e6,
and the position/token that seeds it.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

probe = build_groundtruth_probe()
tests = generate_test_programs()
src, exp, _ = tests[262]
bc, _ = compile_c(src)
ctx = probe._final_context(bc)
pl = len(probe._build_context(bc))
dev = next(probe.model.parameters()).device
padded = torch.tensor([ctx], dtype=torch.long, device=dev)
bl = probe.block_layer_map()

# Decode step boundaries so we can label positions by step.
def step_at(pos):
    # count STEP_END before pos (after prompt)
    s = 0
    for j in range(pl, pos):
        if ctx[j] == int(Token.STEP_END): s += 1
    return s

WATCH = [69, 70, 71, 72, 76, 77, 78, 79, 85]
print(f"pl={pl} S={len(ctx)}", flush=True)
print(f"\n{'phys':>4} {'log':>3} {'global_max':>14} {'pos':>5} {'dim':>4} "
      f"{'tok':>5} {'step':>4}", flush=True)
with torch.no_grad():
    for phys in range(len(probe.model.blocks)):
        r = probe.model.forward(padded, stop_after_block=phys)[0].float()  # [S,D]
        sub = r[:, WATCH].abs()  # [S, len(WATCH)]
        flat = sub.flatten()
        amax = int(flat.argmax().item())
        pos = amax // len(WATCH)
        di = WATCH[amax % len(WATCH)]
        val = float(r[pos, di])
        tok = ctx[pos] if pos < len(ctx) else -1
        print(f"{phys:>4} {bl[phys]['logical']:>3} {abs(val):>14.1f} "
              f"{pos:>5} {di:>4} {tok:>5} {step_at(pos):>4}", flush=True)
