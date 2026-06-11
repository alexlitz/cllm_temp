#!/usr/bin/env python3
"""Trace CMP band (396-399) + dims 79/85 + OUTPUT_BYTE_HI_PIN+8 (678) across
blocks at pos 92 (REG_PC marker) for var_simple_12. Determine whether CMP+1's
66.98 value is the upstream cause of the dim-79 runaway, and where CMP+1 is set.

Also compare to a PASSING flat program (int main(){return 28;}) at its first
emitted marker row, to see what CMP+1 should be.
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
dev = next(probe.model.parameters()).device
bl = probe.block_layer_map()

WATCH = {"CMP+0":396, "CMP+1":397, "CMP+2":398, "CMP+3":399,
         "H2+5(79)":79, "H3+4(85)":85, "OBHP+8(678)":678,
         "OP_JSR(265)":265, "OP_EXIT(292)":292}

def trace(idx, label):
    src, exp, _ = tests[idx]
    bc, _ = compile_c(src)
    ctx = probe._final_context(bc)
    pl = len(probe._build_context(bc))
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    # first emitted marker row = pl (the first REG_PC)
    POS = pl
    print(f"\n=== id={idx} {label} pl={pl} tok@pl={ctx[pl]} (REG_PC=257) ===", flush=True)
    hdr = f"{'phys':>4} {'log':>3}  " + "  ".join(f"{k:>11}" for k in WATCH)
    print(hdr, flush=True)
    with torch.no_grad():
        for phys in range(len(probe.model.blocks)):
            r = probe.model.forward(padded, stop_after_block=phys)[0].float()
            vals = "  ".join(f"{float(r[POS, d]):>11.3f}" for d in WATCH.values())
            print(f"{phys:>4} {bl[phys]['logical']:>3}  {vals}", flush=True)

# var_simple (failing) id 262
trace(262, "var_simple_12 (FAIL)")
