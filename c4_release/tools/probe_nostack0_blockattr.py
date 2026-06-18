#!/usr/bin/env python3
"""Block-by-block attribution of OUTPUT_LO/HI at a target (step, offset) row.

Runs the model with stop_after_block=b for every b, reporting OUTPUT_LO+0,
OUTPUT_LO+1, OUTPUT_HI+0 at the target row after each block. The block where
the 0x01 pattern (LO+1 jumps positive / LO+0 negative) first appears is the
producer.

Usage:
    CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
        python tools/probe_nostack0_blockattr.py 1 0 18   # id step offset
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import torch  # noqa: E402

from neural_vm.run_vm import AutoregressiveVMRunner  # noqa: E402
from neural_vm.batched_pure_neural import BatchedPureNeuralRunner  # noqa: E402
from neural_vm.vm_step import Token  # noqa: E402


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src)
    return list(bc), exp, desc


@torch.no_grad()
def main(idx, target_step, target_off, max_steps):
    ST = Token.STEP_TOKENS
    print(f"### STEP_TOKENS={ST}")
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions
    OLO = dp["OUTPUT_LO"]
    OHI = dp["OUTPUT_HI"]

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    pos = len(prefix) + target_step * ST + target_off
    padded = torch.tensor([context], dtype=torch.long, device=dev)
    nb = len(model.blocks)
    print(f"\n== id={idx} {desc[:40]} step{target_step} off{target_off} pos{pos} ==")
    print("  block | LO+0      LO+1      HI+0      | block name")
    prev = (0.0, 0.0, 0.0)
    for b in range(nb):
        resid = model.forward(padded, stop_after_block=b)[0]
        lo0 = float(resid[pos][OLO + 0])
        lo1 = float(resid[pos][OLO + 1])
        hi0 = float(resid[pos][OHI + 0])
        cur = (lo0, lo1, hi0)
        changed = any(abs(c - p) > 0.5 for c, p in zip(cur, prev))
        bn = type(model.blocks[b]).__name__
        mark = " <== JUMP" if changed else ""
        if changed or b == nb - 1:
            print(f"  {b:5d} | {lo0:9.1f} {lo1:9.1f} {hi0:9.1f} | {bn}{mark}")
        prev = cur


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.lstrip('-').isdigit()]
    idx = args[0] if args else 1
    step = args[1] if len(args) > 1 else 0
    off = args[2] if len(args) > 2 else 18
    ms = args[3] if len(args) > 3 else 4
    main(idx, step, off, ms)
