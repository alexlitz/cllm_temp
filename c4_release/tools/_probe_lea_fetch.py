#!/usr/bin/env python3
"""At the LEA AX-marker row, report ALU_LO (frame base, from L7 head-1) and
FETCH_LO (the immediate operand) entering the L8 ALU (block-input = stop_after
block 10), for the FIRST LEA (step 8, imm=24) vs the RE-READ LEA (step 11,
imm=16). Confirms whether FETCH_LO is STALE on the re-read (=8 from imm=24) vs
the correct 0 (imm=16).

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_funcreread python tools/_probe_lea_fetch.py
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


class _CtxStub:
    def __init__(self, model): self.model = model


def build_initial_context(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_CtxStub(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, max_steps):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(max_steps * STEP):
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        ctx.append(int(model.forward(padded)[0][len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= max_steps: break
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
    pid = 575
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    pl_ctx = build_initial_context(model, bc); pl = len(pl_ctx)
    ctx = replay(model, pl_ctx, 18)
    sm = smk(ctx, pl)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    # block-input to L8 ALU: head-1 writes ALU at blk11; ALU lea rules at blk later.
    # Probe at stop_after_block=11 (after L7 head-1 + L8 attn) so ALU_LO is fresh.
    fetch_lo = dp["FETCH_LO"]; alu_lo = dp["ALU_LO"]
    for b in [10, 11, 12, 13]:
        resid = model.forward(toks, stop_after_block=b)[0]
        print(f"=== block-output {b} ===")
        for s, want_imm in [(8, 24), (11, 16)]:
            ax = sm[s]["AX"]
            al = [(round(float(resid[ax, alu_lo + k]), 1), k) for k in range(16)
                  if abs(float(resid[ax, alu_lo + k])) > 1]
            fl = [(round(float(resid[ax, fetch_lo + k]), 1), k) for k in range(16)
                  if abs(float(resid[ax, fetch_lo + k])) > 1]
            print(f"  step{s} LEA imm={want_imm} AXrow={ax}: ALU_LO={al} FETCH_LO={fl}")


if __name__ == "__main__":
    main()
