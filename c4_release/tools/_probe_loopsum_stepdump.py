#!/usr/bin/env python3
"""Dump loop_sum id450 per-step PC/AX bytes (autoregressive replay) so we can see
whether the LEA-byte0 fix corrupts the PC row at step 2.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loops2b python tools/_probe_loopsum_stepdump.py [pid]
"""
from __future__ import annotations
import os, sys
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


class _St:
    def __init__(self, m): self.model = m


def bic(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_St(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, ms):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(ms * STEP):
        p = torch.tensor([ctx], dtype=torch.long, device=dev)
        l = model.forward(p)[0]
        ctx.append(int(l[len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= ms: break
    return ctx


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 450
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    plc = bic(model, bc); pl = len(plc)
    ctx = replay(model, plc, 20)
    sm = smk(ctx, pl)
    print(f"id{pid} {desc} steps={len(sm)}")
    for st in range(min(len(sm), 6)):
        d = sm[st]
        pcr = d.get("PC"); axr = d.get("AX")
        pcb = [hex(ctx[pcr + 1 + j]) for j in range(4)] if pcr else None
        axb = [hex(ctx[axr + 1 + j]) for j in range(4)] if axr else None
        print(f"  step{st}: PCrow={pcr} PCbytes={pcb}  AXrow={axr} AXbytes={axb}")


if __name__ == "__main__":
    main()
