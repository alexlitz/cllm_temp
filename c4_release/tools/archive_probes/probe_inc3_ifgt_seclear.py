#!/usr/bin/env python3
"""Inc-3 if_gt: at the GT/branch step3 STEP_END row (abs 195), check whether the
no_stack0_se_output_clear op fires (MARK_SE_ONLY present?) and whether OUTPUT is
actually sunk after the final block. Compares a branch-TAKEN step3 (GT=0) row to
a CLEAN passing step's STEP_END row. One forward, read final residual.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_seclear.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ["C4_SKIP_DIM_INTEGRITY"] = "1"
os.environ["C4_SKIP_GATE_CHECK"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402

SRC = os.environ.get("PROBE_SRC", "int main() { if (35 > 43) return 1; return 0; }")


def main():
    STEP = int(Token.STEP_TOKENS)
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _layout = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_layout.dim_positions)
    OL, OH = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    SE = dp.get("MARK_SE_ONLY")
    prompt_len = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=8)
    padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
    nblk = len(p.model.blocks)
    with torch.no_grad():
        resid_full = p.model.forward(padded, stop_after_block=nblk - 1)[0]  # [T, d]
    logits_full = None
    print(f"=== STEP={STEP} src={SRC!r} OUTPUT_LO={OL} MARK_SE_ONLY={SE} ===")
    print(" step | SE_row | tok | MARK_SE_ONLY | |OUT_LO|max | |OUT_HI|max | "
          "next_pred(argmax) next_emitted")
    head_w = p.model.head.weight  # [vocab, d]
    tn = {257: "PC", 258: "AX", 259: "SP", 260: "BP", 261: "MEM",
          262: "SE", 263: "HALT", 268: "ST0"}
    for step in range(7):
        se_row = prompt_len + step * STEP + (STEP - 1)
        if se_row + 1 >= len(ctx):
            break
        resid = resid_full[se_row]
        se_v = float(resid[SE].item()) if SE is not None else float("nan")
        lo = float(resid[OL:OL + 16].abs().max().item())
        hi = float(resid[OH:OH + 16].abs().max().item())
        nxt = int(ctx[se_row + 1])
        tok = int(ctx[se_row])
        print(f"  {step}   | {se_row:5d}  | {tok:3d} | {se_v:+.3f}      "
              f"| {lo:.2e} | {hi:.2e} | emitted_next={tn.get(nxt,'b'+str(nxt))}")


if __name__ == "__main__":
    main()
