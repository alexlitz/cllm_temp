#!/usr/bin/env python3
"""Dump LM-head top-k logits at the SE / step-boundary positions under the
no-STACK0 layout, plus the OUTPUT_LO/HI + NEXT_* residual at those positions.
Reveals which byte logit beats REG_PC at the SE slot (the +1 drift root).

Usage:
    CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 python tools/probe_nostack0_logits.py 0 0
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
    return names.get(t, f"byte0x{t:02x}" if t < 256 else f"t{t}")


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

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * (target_step + 2)):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    padded = torch.tensor([context], dtype=torch.long, device=dev)
    logits = model.forward(padded)[0]   # [S, V]
    resid_last = model.forward(padded, stop_after_block=len(model.blocks) - 1)[0]

    start = len(prefix) + target_step * ST
    NEXTS = {nm: dp[nm] for nm in (
        "NEXT_PC", "NEXT_AX", "NEXT_SP", "NEXT_BP", "NEXT_MEM", "NEXT_SE")}
    OLO = dp.get("OUTPUT_LO")
    OHI = dp.get("OUTPUT_HI")

    print(f"\n===== id={idx} {desc[:46]} step {target_step} (offsets 18-29) =====")
    for off in range(18, ST):
        pos = start + off
        if pos >= len(context) - 1:
            break
        tok = context[pos]
        lg = logits[pos]   # predicts token at pos+1
        topk = torch.topk(lg, 6)
        tk = [(int(i), float(v)) for i, v in zip(topk.indices, topk.values)]
        nexts = [nm for nm, d in NEXTS.items()
                 if float(resid_last[pos][d].item()) > 0.3]
        olo = ""
        if OLO is not None:
            vals = [round(float(resid_last[pos][OLO + k].item()), 2)
                    for k in range(16)]
            nz = {k: v for k, v in enumerate(vals) if abs(v) > 0.05}
            olo = f"OUT_LO_nz={nz}"
        print(f" off {off:2d} tok={_tok(tok):>8} | "
              f"top: " + ", ".join(f"{_tok(i)}={v:.1f}" for i, v in tk))
        print(f"          NEXT={nexts} {olo}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    idx = args[0] if args else 0
    step = args[1] if len(args) > 1 else 0
    main(idx, step)
