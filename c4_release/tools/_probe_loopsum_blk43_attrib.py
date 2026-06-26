#!/usr/bin/env python3
"""Attribute the loop_sum id450 step-2 block-43 OUTPUT_LO byte-0 slam to its
FFN unit(s) + the dims those units key on (so we can name the corruptor rule).

Runs the model up to stop_after_block=42 (the residual that feeds block 43),
then manually runs block 43's attn+ffn and, for the LEA AX row, ranks FFN hidden
units by their |contribution| to OUTPUT_LO+0 (= silu(up)*gate * W_down[OUTPUT_LO+0,h]).
For the top units, dumps the dims their W_up/W_gate read (the rule signature).

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loops2b python tools/_probe_loopsum_blk43_attrib.py [pid] [step] [block]
"""
from __future__ import annotations
import os, sys
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
import torch.nn.functional as F  # noqa
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
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 43
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    inv_dp = {}
    for name, pos in dp.items():
        inv_dp.setdefault(pos, name)
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    pl_ctx = build_initial_context(model, bc); pl = len(pl_ctx)
    ctx = replay(model, pl_ctx, max(step + 4, 18))
    sm = smk(ctx, pl)
    axrow = sm[step]["AX"]
    out_lo = dp["OUTPUT_LO"]
    D = E_DIM = model.blocks[0].ffn.dim if hasattr(model.blocks[0].ffn, "dim") else None
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    N = len(ctx)
    # residual feeding block `blk` = output of block blk-1
    x = model.forward(toks, stop_after_block=blk - 1)  # [1, N, D]
    Dm = x.shape[-1]
    block = model.blocks[blk]
    # run attn part of block `blk`
    xa = block.attn(x)
    ffn = block.ffn
    # FlattenedPureFFN: flatten to [1,1,N*D]
    is_flat = hasattr(ffn, "flat_dim")
    print(f"id{pid} step{step} AXrow={axrow} blk={blk} D={Dm} flattened_ffn={is_flat} "
          f"ffn_hidden={getattr(ffn, 'hidden_dim', '?')}")
    if is_flat:
        xf = xa.reshape(1, 1, N * Dm)
        inner = ffn.ffn
        up = F.linear(xf, inner.W_up) + inner.b_up      # [1,1,H]
        gate = F.linear(xf, inner.W_gate) + inner.b_gate
        hidden = (F.silu(up) * gate)[0, 0]              # [H]
        # target flat dim = OUTPUT_LO+0 at the AX row
        tgt = axrow * Dm + (out_lo + 0)
        wdown_col = inner.W_down[tgt]                   # [H] contributions
        contrib = hidden * wdown_col                    # [H]
        order = contrib.abs().argsort(descending=True)[:8]
        print(f"  total OUTPUT_LO+0 contribution (sum) = {float(contrib.sum()):.2f}")
        for h in order.tolist():
            c = float(contrib[h]); hv = float(hidden[h])
            if abs(c) < 1.0: continue
            # which input flat-dims this unit reads (W_up row h)
            wrow = inner.W_up[h]
            nz = wrow.nonzero(as_tuple=True)[0]
            reads = []
            for fd in nz.tolist():
                pos = fd // Dm; d = fd % Dm
                nm = inv_dp.get(d, f"d{d}")
                reads.append(f"{nm}@p{pos}({float(wrow[fd]):.0f})")
                if len(reads) >= 8: break
            # which OUTPUT_LO cells this unit writes (W_down col scan for axrow)
            wcells = []
            for k in range(16):
                w = float(inner.W_down[axrow * Dm + out_lo + k, h])
                if abs(w) > 1e-3: wcells.append(f"LO+{k}={w:.1f}")
            print(f"  unit {h}: contrib={c:11.2f} hidden={hv:.3f}")
            print(f"      reads: {reads}")
            print(f"      writes OUTPUT_LO: {wcells[:8]}")
    else:
        # regular PureFFN: per-position units. Attribute at the AX row.
        inner = ffn
        xrow = xa[0, axrow]                              # [D]
        up = F.linear(xrow, inner.W_up) + inner.b_up    # [H]
        gate = F.linear(xrow, inner.W_gate) + inner.b_gate
        hidden = F.silu(up) * gate                      # [H]
        wdown_col = inner.W_down[out_lo + 0]            # [H] -> OUTPUT_LO+0
        contrib = hidden * wdown_col                    # [H]
        order = contrib.abs().argsort(descending=True)[:12]
        print(f"  total OUTPUT_LO+0 contribution (sum) = {float(contrib.sum()):.2f}")
        for h in order.tolist():
            c = float(contrib[h]); hv = float(hidden[h])
            wrow = inner.W_up[h]
            nz = wrow.nonzero(as_tuple=True)[0]
            reads = []
            for d in nz.tolist():
                nm = inv_dp.get(int(d), f"d{d}")
                reads.append(f"{nm}({float(wrow[d]):.1f})")
                if len(reads) >= 12: break
            wcells = []
            for k in range(16):
                w = float(inner.W_down[out_lo + k, h])
                if abs(w) > 1e-3: wcells.append(f"LO+{k}={w:.1f}")
            print(f"  unit {h}: contrib={c:11.2f} hidden={hv:.3f} up={float(up[h]):.2f} gate={float(gate[h]):.2f}")
            print(f"      reads: {reads}")
            print(f"      writes OUTPUT_LO: {wcells[:10]}")


if __name__ == "__main__":
    main()
