#!/usr/bin/env python3
"""Trace loop_sum id450 step-2 LEA &i AX-marker byte-0 (OUTPUT_LO) per block.

Finds (1) why OUTPUT_LO byte0 is dead through block 41 and (2) what slams it to
cell 0 at block 43. Mirrors tools/_probe_lea_addr_trace.py but for the loop_sum
in-loop LEA row, and dumps the full 16-cell OUTPUT_LO band at the suspect blocks
so we can see the argmax migrate.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loops2b python tools/_probe_loopsum_lea_b0.py [pid] [step]
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
        logits = model.forward(padded)[0]
        ctx.append(int(logits[len(ctx) - 1].argmax().item()))
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
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 450
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    pl_ctx = build_initial_context(model, bc); pl = len(pl_ctx)
    ctx = replay(model, pl_ctx, max(step + 4, 18))
    sm = smk(ctx, pl)
    if step >= len(sm) or "AX" not in sm[step]:
        print(f"id{pid}: step {step} not reached ({len(sm)} steps)"); return
    axrow = sm[step]["AX"]; bprow = sm[step].get("BP")
    out_lo = dp["OUTPUT_LO"]; out_hi = dp.get("OUTPUT_HI", out_lo)
    nblk = len(model.blocks)
    print(f"id{pid} {desc} step{step} AXrow={axrow} BProw={bprow} blocks={nblk}")
    print(f"  emitted AX bytes (ctx) = ", [hex(ctx[axrow+1+j]) for j in range(4) if axrow+1+j < len(ctx)])
    S = len(ctx); toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    # per-block OUTPUT_LO byte0 band at the AX row
    prev_amx = None; prev_b0 = None
    for b in range(nblk):
        r = model.forward(toks, stop_after_block=b)[0]
        band = r[axrow, out_lo:out_lo + 16]
        amx = int(band.argmax().item()); amv = float(band.max().item())
        b0 = float(r[axrow, out_lo + 0]); b8 = float(r[axrow, out_lo + 8])
        chg = (prev_amx is not None and amx != prev_amx) or (prev_b0 is not None and abs(b0 - prev_b0) > 50)
        tag = "  <== CHANGED" if chg else ""
        if b >= 38 or tag:
            print(f"  blk{b:2d} argmax_cell={amx:2d}@{amv:9.1f}  LO[0]={b0:11.1f} LO[8]={b8:11.1f}{tag}")
        prev_amx = amx; prev_b0 = b0
    # dump full 16-cell band at blocks 40,41,42,43,44
    for b in [40, 41, 42, 43, 44, nblk - 1]:
        if b >= nblk: continue
        r = model.forward(toks, stop_after_block=b)[0]
        band = [round(float(r[axrow, out_lo + k]), 1) for k in range(16)]
        print(f"  [FULL blk{b:2d}] OUTPUT_LO 16 cells: {band}")
    logits = model.forward(toks)[0]
    tok_b0 = int(logits[axrow].argmax().item())
    print(f"  FINAL: emitted AX byte0 token (argmax @row{axrow}) = {tok_b0} (0x{tok_b0 & 0xff:02x})")


if __name__ == "__main__":
    main()
