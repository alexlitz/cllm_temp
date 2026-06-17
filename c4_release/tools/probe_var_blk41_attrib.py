#!/usr/bin/env python3
"""Attribute the block-41 (L25 tail) FFN units that flip OUTPUT_LO 0->8 on the
step-6 LEA AX row (var_mul / var_three). silu(up)*gate, BUILT dim layout.

Prints the top units by |contribution to OUTPUT_LO[8]| (and OUTPUT_LO[0]) at the
step-6 AX-marker row of a FULLY teacher-forced context, so we can pin the exact
rule that writes the 0xE8 stack-top byte over the correct 0xE0 LEA result.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, contextlib, io
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from neural_vm.batched_pure_neural import Token
from neural_vm.speculative import DraftVM

STEP = int(Token.STEP_TOKENS)


def dense(w):
    return w.to_dense() if w.layout != torch.strided else w


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 275
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)

    from tools.probe_groundtruth import build_groundtruth_probe
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_groundtruth_probe()
        _m, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = layout.dim_positions
    OUT_LO = dp["OUTPUT_LO"]
    dev = p._device

    ctx = p._build_context(bc)
    plen = len(ctx)
    dv = DraftVM(list(bc))
    for _ in range(10):
        dv.step()
        ctx.extend(int(t) for t in dv.draft_tokens())
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    blm = p.block_layer_map()
    # block 41 in the tail bank.
    BLK = int(sys.argv[3]) if len(sys.argv) > 3 else 41
    axrow = plen + step * STEP + 5

    xin = p.model.forward(padded, stop_after_block=BLK - 1)[0]
    x = xin[axrow].float()
    ffn = p.model.blocks[BLK].ffn
    if not hasattr(ffn, "W_up"):
        print(f"block {BLK} ffn type {type(ffn).__name__} has no W_up")
        return
    Wup = dense(ffn.W_up).float()
    Wgate = dense(ffn.W_gate).float()
    Wdown = dense(ffn.W_down).float()
    bup = ffn.b_up.float()
    bgate = ffn.b_gate.float()
    up = Wup @ x + bup
    gate = Wgate @ x + bgate
    hidden = F.silu(up) * gate
    for target_lo in (8, 0):
        d = OUT_LO + target_lo
        contrib = Wdown[d] * hidden  # [H]
        order = torch.argsort(contrib.abs(), descending=True)
        print(f"\n=== block {BLK} top units by |contribution to OUTPUT_LO[{target_lo}]| "
              f"(id{idx} step{step} axrow={axrow}) ===")
        tot = float(contrib.sum())
        print(f"  total Σ contribution to OUTPUT_LO[{target_lo}] = {tot:+.3f}")
        for h in order[:12].tolist():
            c = float(contrib[h])
            if abs(c) < 0.5:
                break
            print(f"  unit{h:5d}: contrib={c:+9.3f} hidden={float(hidden[h]):+8.3f} "
                  f"up={float(up[h]):+8.2f} gate={float(gate[h]):+8.2f} "
                  f"Wdown={float(Wdown[d,h]):+7.3f}")


if __name__ == "__main__":
    main()
