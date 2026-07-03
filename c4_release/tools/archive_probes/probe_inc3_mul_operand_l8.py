#!/usr/bin/env python3
"""Inc-3 MAP — expr_mul_div MUL operand-A from mem[SP] at L8 (got_ax=0).

expr_mul_div id850 (10*9/1) diverges at step3=MUL with got_ax=0: operand-A (10,
PSH'd at step1) is not delivered into ALU. This is the make_layer8_mem_to_alu_op
(C4_OPERAND_FROM_MEMSP) binary-op operand path. Probe the step3 AX byte-0 predictor
row block-by-block; decode ALU_LO/HI (the operand-A target) in GOLDEN vs CAMPAIGN.

Run TWICE (clear cache between):
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=0  python tools/probe_inc3_mul_operand_l8.py
  CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_mul_operand_l8.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
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

SRC = "int main() { return 10 * 9 / 1; }"
DIV_STEP = 3
WANT = 10  # operand-A popped from stack


def td(w):
    return (w.to_dense() if w.layout != torch.strided else w).detach().cpu().float()


def nib(row, lo, hi):
    li = int(torch.argmax(row[lo:lo + 16]).item()); lv = float(row[lo + li])
    hi_i = int(torch.argmax(row[hi:hi + 16]).item()); hv = float(row[hi + hi_i])
    return hi_i * 16 + li, min(lv, hv)


def main():
    nostk = os.environ.get("C4_NO_STACK0_EMIT", "0") != "0"
    cfg = "CAMPAIGN(30)" if nostk else "GOLDEN(35)"
    bytecode, _ = compile_c(SRC)
    p = build_groundtruth_probe()
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic)
    _m, _l = compile_full_vm_dynamic(disk_cache=True)
    dp = dict(_l.dim_positions)
    STEP = int(Token.STEP_TOKENS)
    pl = len(p._build_context(bytecode))
    ctx = p._final_context(bytecode, max_steps=DIV_STEP + 2)
    s = pl + DIV_STEP * STEP
    print(f"=== {cfg} STEP={STEP} === step{DIV_STEP} AX bytes = {ctx[s+6:s+10]} "
          f"(MUL; operand-A want={WANT})")
    off = s + 5  # AX MARKER row (where L8 mem-to-alu writes ALU)
    padded = torch.tensor([ctx], device=p._device)
    nblk = len(p.model.blocks)
    blk_map = p.block_layer_map()
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    out_lo, out_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    print(f"  probing off={off}")
    prev = None
    for blk in range(nblk):
        x = td(p.model.forward(padded, stop_after_block=blk)[0])
        row = x[off]
        av, ac = nib(row, alu_lo, alu_hi)
        ov, oc = nib(row, out_lo, out_hi)
        key = (av, round(ac, 1), ov, round(oc, 1))
        if key != prev:
            lg = blk_map[blk]
            lg = lg.get("logical") if isinstance(lg, dict) else lg
            hit = f" <<ALU={WANT}!" if av == WANT and ac > 0.3 else ""
            print(f"  blk{blk:2d}(L{lg}): ALU={av}(c{ac:.1f}) OUT={ov}(c{oc:.1f}){hit}")
            prev = key


if __name__ == "__main__":
    main()
