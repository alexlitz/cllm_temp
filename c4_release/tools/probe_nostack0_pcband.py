#!/usr/bin/env python3
"""Attribute the 0x01 high-byte leak under the no-STACK0 (30-token) layout.

Autoregressively emits a program, then at EVERY offset of a target step dumps:
  - the emitted token (and what the LM head predicted)
  - the OUTPUT_LO / OUTPUT_HI residual nibble vector AFTER the final block
  - the top-3 byte logits

so we can see exactly which OUTPUT cell carries the stray 0x01 into the
register high bytes, and at which row it first appears.

Usage:
    CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
        python tools/probe_nostack0_pcband.py 1 0   # id=1, step 0
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
    if t in names:
        return names[t]
    if t < 256:
        return f"{t:02x}"
    return f"t{t}"


def _off_name(off, ST):
    if ST == 30:
        starts = [(0, "PC"), (5, "AX"), (10, "SP"), (15, "BP"), (20, "MEM")]
        if off == 29:
            return "SE"
        if off >= 21:
            return f"MADDR[{off-21}]" if off <= 24 else f"MVAL[{off-25}]"
    else:
        starts = [(0, "PC"), (5, "AX"), (10, "SP"), (15, "BP"),
                  (20, "STK0"), (25, "MEM")]
        if off == 34:
            return "SE"
        if off >= 26:
            return f"MADDR[{off-26}]" if off <= 29 else f"MVAL[{off-30}]"
    for s, nm in reversed(starts):
        if off >= s:
            r = off - s
            return f"{nm}m" if r == 0 else f"{nm}[{r-1}]"
    return f"o{off}"


@torch.no_grad()
def main(idx, target_step, max_steps):
    ST = Token.STEP_TOKENS
    print(f"### STEP_TOKENS={ST}")
    mr = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers = {}
    mr._syscall_handlers = {}
    runner = BatchedPureNeuralRunner(model_runner=mr)
    model = runner.model
    dev = next(model.parameters()).device
    dp = model.dim_positions
    OLO = dp.get("OUTPUT_LO")
    OHI = dp.get("OUTPUT_HI")
    FLAGS = [f for f in ("BYTE_INDEX_1", "BYTE_INDEX_2", "BYTE_INDEX_3",
                         "PSH_AT_SP", "H1+0", "H1+1", "H1+2", "H1+3",
                         "H1+10", "H4+3", "CMP+7", "OP_ENT")
             if f.split("+")[0] in dp]

    bc, exp, desc = _corpus(idx)
    prefix = list(mr._build_context(bc, b"", [], ""))
    context = list(prefix)
    for _ in range(ST * max_steps):
        padded = torch.tensor([context], dtype=torch.long, device=dev)
        nxt = int(model.forward(padded)[0][-1].argmax().item())
        context.append(nxt)
        if nxt == Token.HALT:
            break

    padded = torch.tensor([context], dtype=torch.long, device=dev)
    logits = model.forward(padded)[0]
    resid = model.forward(padded, stop_after_block=len(model.blocks) - 1)[0]

    start = len(prefix) + target_step * ST
    print(f"\n===== id={idx} {desc[:46]} step {target_step} =====")
    print("  off  name     tok  | top byte logits        | OUT_LO nz / OUT_HI nz")
    for off in range(ST):
        pos = start + off
        if pos >= len(context):
            break
        tok = context[pos]
        lg = logits[pos - 1] if pos > 0 else logits[pos]  # logits predicting tok at pos
        topk = torch.topk(lg, 3)
        tk = ", ".join(f"{_tok(int(i))}={float(v):.0f}"
                       for i, v in zip(topk.indices, topk.values))
        olo = ohi = ""
        if OLO is not None:
            lo = [round(float(resid[pos][OLO + k]), 1) for k in range(16)]
            olo = {k: v for k, v in enumerate(lo) if abs(v) > 0.3}
        if OHI is not None:
            hi = [round(float(resid[pos][OHI + k]), 1) for k in range(16)]
            ohi = {k: v for k, v in enumerate(hi) if abs(v) > 0.3}
        def _rd(f):
            base, _, o = f.partition("+")
            return float(resid[pos][dp[base] + (int(o) if o else 0)])
        flg = {f: round(_rd(f), 1) for f in FLAGS if abs(_rd(f)) > 0.3}
        print(f"  {off:3d}  {_off_name(off, ST):8} {_tok(tok):>4} | {tk:28} | "
              f"LO={olo} HI={ohi}")
        if flg:
            print(f"       flags={flg}")


if __name__ == "__main__":
    args = [int(x) for x in sys.argv[1:] if x.isdigit()]
    idx = args[0] if args else 1
    step = args[1] if len(args) > 1 else 0
    ms = args[2] if len(args) > 2 else 4
    main(idx, step, ms)
