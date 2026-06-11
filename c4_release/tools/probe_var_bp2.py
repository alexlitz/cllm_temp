#!/usr/bin/env python3
"""Fast: logits at step-0 REG_BP byte rows + STACK0-marker row for var_simple_12.
No per-block residual sweep (that's too slow). Caches one final context.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

TOKNAME = {257:"REG_PC",258:"REG_AX",259:"REG_SP",260:"REG_BP",261:"MEM",
           262:"STEP_END",263:"HALT",264:"CODE_START",265:"CODE_END",
           266:"DATA_START",267:"DATA_END",268:"STACK0",271:"TOOL_CALL"}
def nm(t): return TOKNAME.get(t, f"byte:0x{t:02x}" if t < 256 else f"tok{t}")

probe = build_groundtruth_probe()
tests = generate_test_programs()

def run(idx, label):
    src, exp, _ = tests[idx]
    bc, _ = compile_c(src)
    ctx = probe._final_context(bc)
    pl = len(probe._build_context(bc))
    dev = next(probe.model.parameters()).device
    print(f"\n##### id={idx} {label} expected={exp} pl={pl} #####", flush=True)
    # step-0 markers
    i = pl
    bp_idx = None
    while i < len(ctx):
        if ctx[i] == int(Token.REG_BP):
            bp_idx = i; break
        i += 1
    print(f"step0 REG_BP marker idx={bp_idx} ctx[{bp_idx}:{bp_idx+10}]="
          f"{[nm(t) for t in ctx[bp_idx:bp_idx+10]]}", flush=True)
    # logits at: BP byte0..3 rows, and the row right AFTER BP bytes (should
    # predict STACK0=268).
    @torch.no_grad()
    def lg(prefix_len):
        padded = torch.tensor([ctx[:prefix_len]], dtype=torch.long, device=dev)
        v = probe.model.forward(padded)[0, prefix_len-1]
        top = torch.topk(v, 6)
        return [(nm(int(t)), round(float(val),2)) for val,t in
                zip(top.values.tolist(), top.indices.tolist())]
    exp_bp = [0,0,1,0]
    for b in range(4):
        pos = bp_idx+1+b
        print(f"  BP byte{b}: emitted={nm(ctx[pos])} exp=0x{exp_bp[b]:02x} "
              f"top={lg(pos)}", flush=True)
    # row predicting token after BP's 4 bytes (should be STACK0)
    pos = bp_idx+5
    print(f"  after-BP (expect STACK0): emitted={nm(ctx[pos])} top={lg(pos)}",
          flush=True)

# var_simple_12 (fails) and the control IMM-only (passes, no JSR/ENT)
run(262, "var_simple_12 x=28 (FAIL)")
