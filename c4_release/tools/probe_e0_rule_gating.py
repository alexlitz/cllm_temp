#!/usr/bin/env python3
"""Read every gating dim of l16_jsr_mem_addr0_e0_from_l14_evidence at the
STEP_END row (pos 126) vs the MEM-store row of step 0, to see WHY it mis-fires
(h=1e6) at the STEP_END row. Residual is read at the block-29 INPUT (after
block 28). spec_k=0, hook-free.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_CSR_INFERENCE"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

# Conditions of l16_jsr_mem_addr0_e0_from_l14_evidence (name, weight).
COND = [
    ("OP_JSR", 1000.0), ("OP_ENT", -10.0), ("PSH_AT_SP", -100000.0),
    ("MARK_MEM", 1.0), ("MEM_STORE", 1.0), ("HAS_SE", 1.0),
    ("IS_BYTE", -1e12), ("MARK_PC", -1e6), ("MARK_AX", -1e6),
    ("MARK_SP", -1e6), ("MARK_BP", -1e6), ("MARK_STACK0", -1e6),
    ("OUTPUT_LO+0", 1.0), ("OUTPUT_LO+8", -1.0),
    ("OUTPUT_HI_THIS_STEP+14", 1.0), ("OUTPUT_HI_THIS_STEP+15", -1.0),
]
THRESHOLD = 500.0


def getdim(dp, name):
    base, _, off = name.partition("+")
    b = int(dp[base])
    return b + (int(off) if off else 0)


def main():
    probe = build_groundtruth_probe()
    m = probe.model; dp = m.dim_positions; dev = next(m.parameters()).device
    src, exp, _ = generate_test_programs()[262]
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=9)
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    with torch.no_grad():
        r = m.forward(padded, stop_after_block=28)[0].float()

    # Find the step-0 MEM marker row (JSR store) for comparison.
    from neural_vm.batched_pure_neural import Token
    mem_tok = int(Token.REG_MEM) if hasattr(Token, "REG_MEM") else None
    rows = {126: "STEP_END (mis-fires 0xE0)"}
    # The JSR store MEM row: scan for the MEM marker in step 0 region (pos 92-126).
    for p in range(92, 127):
        # MARK_MEM dim high => MEM marker row.
        if float(r[p][int(dp["MARK_MEM"])]) > 0.5:
            rows[p] = "step0 MEM-store marker row"
    for pos, label in sorted(rows.items()):
        score = 0.0
        print(f"\n=== ctx[{pos}]={ctx[pos]} : {label} ===")
        for name, w in COND:
            try:
                v = float(r[pos][getdim(dp, name)])
            except KeyError:
                v = float("nan")
            term = w * v
            score += term
            if abs(term) > 0.01 or name in ("OP_JSR", "MARK_MEM", "MEM_STORE", "HAS_SE", "IS_BYTE"):
                print(f"   {name:<22} v={v:+.3f}  w={w:+.0e}  term={term:+.3e}")
        print(f"   --> score={score:+.4e}  threshold={THRESHOLD}  "
              f"{'FIRE' if score >= THRESHOLD else 'no fire'}")


if __name__ == "__main__":
    main()
