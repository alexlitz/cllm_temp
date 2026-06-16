#!/usr/bin/env python3
"""Probe the L0 NEXT_* marker-transition dims + threshold heads at every
position of a step, under the no-STACK0 layout. Reveals WHICH transition
fires at which offset and where the MEM/SE chain breaks.

Teacher-forces the model's OWN emitted context up to a chosen step, then reads
the residual at the L0 FFN block (NEXT_* dims) and at the L0 attn block (H*
threshold heads) for the positions in that step.

Usage:
    CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 python tools/probe_nostack0_next.py 250 5
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


def _tok(t):
    names = {256: "SEP", 257: "rPC", 258: "rAX", 259: "rSP", 260: "rBP",
             261: "MEM", 262: "SE", 263: "HALT", 268: "STK0"}
    return names.get(t, f"{t:02x}" if t < 256 else f"t{t}")


@torch.no_grad()
def main(idx, target_step):
    ST = Token.STEP_TOKENS
    print(f"### STEP_TOKENS={ST}")
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions

    NEXTS = {nm: dp[nm] for nm in (
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_MEM", "NEXT_SE")}
    if "NEXT_STACK0" in dp:
        NEXTS["NEXT_STACK0"] = dp["NEXT_STACK0"]
    # threshold heads H0..H4, marker slot positions PC=0..SE=5
    HBASE = {f"H{k}": dp[f"H{k}"] for k in range(8) if f"H{k}" in dp}

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    # generate up to (target_step+1) full steps
    for _ in range(ST * (target_step + 2)):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    # find the L0 FFN block index (block 0). Probe residual after block 0.
    padded = torch.tensor([context], dtype=torch.long, device=dev)
    resid0 = model.forward(padded, stop_after_block=0)[0]  # [S, D]

    emitted = context[len(prefix):]
    start = len(prefix) + target_step * ST
    print(f"\n===== id={idx} {desc[:46]} step {target_step} =====")
    print("pos: tok | NEXT_* fired (>0.3) | H* threshold(marker slot fired)")
    for off in range(ST):
        pos = start + off
        if pos >= len(context):
            break
        tok = context[pos]
        row = resid0[pos]
        nexts = [nm for nm, d in NEXTS.items() if float(row[d].item()) > 0.3]
        # threshold heads: for each head, which marker slots fire
        hfired = []
        for hn, hb in HBASE.items():
            slots = [i for i in range(6) if float(row[hb + i].item()) > 0.5]
            if slots:
                hfired.append(f"{hn}[{','.join('PC AX SP BP MEM SE'.split()[s] for s in slots)}]")
        em = _tok(tok)
        # what token does the model PREDICT next from this pos?
        nxt_pred = _tok(int(model.forward(
            torch.tensor([context[:pos + 1]], dtype=torch.long, device=dev)
        )[0][-1].argmax().item()))
        print(f" off {off:2d} [{em:>4}] -> pred_next={nxt_pred:>4} | "
              f"NEXT={nexts} | {' '.join(hfired)}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    idx = args[0] if args else 250
    step = args[1] if len(args) > 1 else 5
    main(idx, step)
