#!/usr/bin/env python3
"""Trace the re-read LEA AX-marker byte-0 through the late blocks + the final
LM-head argmax, to localize corruptor #2 (BP-byte0 / address byte-0 drift) now
that the L7 head-1 re-sharpen delivers the BP frame into ALU at full strength.

Builds the real baked model via ``build_cpu_model`` (same weights as the
verdict), autoregressive-replays func_add, and at the re-read LEA AX row reports
OUTPUT_LO[0] per block + the BP marker's OUTPUT_LO[0] + the final argmax token.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_funcreread python tools/_probe_lea_addr_trace.py [pid] [step]
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
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 575
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 11
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
    S = len(ctx); toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    out_lo = dp["OUTPUT_LO"]; out_hi = dp.get("OUTPUT_HI", out_lo)
    nblk = len(model.blocks)
    print(f"id{pid} {desc} step{step} AXrow={axrow} BProw={bprow} blocks={nblk}")
    # OUTPUT_LO byte0 per block at the AX row + BP row
    prev = None
    for b in range(nblk):
        r = model.forward(toks, stop_after_block=b)[0]
        axb0 = float(r[axrow, out_lo + 0])
        # argmax over OUTPUT_LO 16 cells at AX row (the byte value as one-hot)
        band = r[axrow, out_lo:out_lo + 16]
        amx = int(band.argmax().item()); amv = float(band.max().item())
        bpb0 = float(r[bprow, out_lo + 0]) if bprow is not None else 0.0
        tag = "" if prev is None or abs(axb0 - prev) < 2 else "  <== CHANGED"
        if b >= 28 or tag:
            print(f"  blk{b:2d} AX OUT_LO[0]={axb0:8.2f} band_argmax_cell={amx:2d}@{amv:6.1f} "
                  f"BP OUT_LO[0]={bpb0:7.2f}{tag}")
        prev = axb0
    # final LM-head argmax at the position that emits AX byte0 (axrow+1)
    logits = model.forward(toks)[0]
    emit_pos = axrow  # the token AT axrow+1 is byte0; predicted from logits[axrow]
    tok_b0 = int(logits[axrow].argmax().item())
    print(f"  FINAL: emitted AX byte0 token (argmax @row{axrow}) = {tok_b0} "
          f"(0x{tok_b0 & 0xff:02x}); actual next ctx tok = {ctx[axrow+1] if axrow+1 < len(ctx) else '?'}")


if __name__ == "__main__":
    main()
