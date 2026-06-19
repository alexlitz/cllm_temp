#!/usr/bin/env python3
"""Inc-3 if_gt: at the branch-step AX-marker predictor row (off=5), dump the
INPUT residual (= output of blk33) for the dims that drive unit 254's
(l16_stack0_e0_marker_from_alu_lo_0) up (conditions) and gate (ALU_LO+0 +
MARK_MEM/IS_BYTE gate_terms). Explains WHY the e0 materializer fires + why its
gate is negative -> the OUTPUT crush.

  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 python tools/probe_inc3_ifgt_e0gate.py
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
BLK = 34
TARGET_STEP = int(os.environ.get("PROBE_STEP", "4"))
OFF = int(os.environ.get("PROBE_OFF", "5"))

DIMS = ["MARK_STACK0", "HAS_SE", "ADDR_B0_LO+0", "ADDR_B0_LO+8",
        "ADDR_B0_HI+14", "ADDR_B0_HI+15", "IS_BYTE", "OP_JSR", "OP_ENT",
        "OP_LEV", "MEM_STORE", "MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP",
        "MARK_MEM", "ALU_LO+0", "ALU_HI+0"]


def main():
    bytecode, data = compile_c(SRC)
    p = build_groundtruth_probe()
    runner = p.runner
    dp = dict(p.model.dim_positions)

    def D(n):
        if "+" in n:
            b, o = n.rsplit("+", 1)
            return dp[b] + int(o)
        return dp[n]

    STEP = int(Token.STEP_TOKENS)
    _, oracle_tokens = runner._oracle_pc_ax_steps(
        bytecode, data or b"", "", expected_steps=None, with_tokens=True)
    prompt = runner._build_element(bytecode, data or b"", [], "", spec_k=1,
                                   adaptive_start_k=0, expected_steps=None)
    prefix = list(prompt.context)
    tape = list(prefix)
    for stp in oracle_tokens:
        tape.extend(stp)
    row = len(prefix) + TARGET_STEP * STEP + OFF
    padded = torch.tensor([tape], dtype=torch.long, device=p._device)
    with torch.no_grad():
        xin = p.model.forward(padded, stop_after_block=BLK - 1)[0, row]
    print(f"=== blk{BLK} INPUT resid at step{TARGET_STEP} off={OFF} row={row} src={SRC!r} ===")
    for nm in DIMS:
        try:
            print(f"  {nm:16s} = {float(xin[D(nm)]):+.4f}")
        except KeyError:
            print(f"  {nm:16s} = <missing dim>")


if __name__ == "__main__":
    main()
