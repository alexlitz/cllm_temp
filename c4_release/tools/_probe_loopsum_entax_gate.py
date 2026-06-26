#!/usr/bin/env python3
"""Probe the loop_sum id450 step-2 LEA AX row's _l10_ent_axcarry discriminator +
gate dims at the residual feeding block 43 (output of block 42), so we know which
AX_CARRY cell the corruptor's winner-take-all gates on, and confirm the
discriminator (MARK_AX/OP_LEA/MEM_ADDR_SRC/owning-opcodes) fires.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loops2b python tools/_probe_loopsum_entax_gate.py [pid] [step]
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
    axrow = sm[step]["AX"]
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    # residual feeding block 43 = output of block 42
    r = model.forward(toks, stop_after_block=42)[0]

    def g(name):
        return float(r[axrow, dp[name]]) if name in dp else None

    print(f"id{pid} step{step} AXrow={axrow}  (residual after blk42, feeds blk43 ent_axcarry)")
    disc = ["MARK_AX", "OP_LEA", "MEM_ADDR_SRC", "OP_JSR", "OP_ENT", "HAS_SE",
            "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
            "IS_BYTE", "CMP+7", "FETCH_HI+1", "FETCH_LO+8", "FETCH_HI+15",
            "FETCH_LO+0", "FETCH_HI+14"]
    for nm in disc:
        base = nm.split("+")[0]
        if base in dp:
            off = int(nm.split("+")[1]) if "+" in nm else 0
            print(f"  {nm:16s} = {float(r[axrow, dp[base]+off]):.3f}")
    # owning ops block list
    print("  --- owning-op NOT-blocks (any >0 vetoes ent_axcarry) ---")
    for op in ["OP_IMM","OP_ADD","OP_SUB","OP_AND","OP_OR","OP_XOR","OP_EQ","OP_NE",
               "OP_LT","OP_GT","OP_LE","OP_GE","OP_MUL","OP_DIV","OP_MOD","OP_SHL",
               "OP_SHR","OP_JMP","OP_BZ","OP_BNZ"]:
        if op in dp:
            v = float(r[axrow, dp[op]])
            if abs(v) > 0.05: print(f"  {op:10s} = {v:.3f}")
    # AX_CARRY bands (the gate the winner-take-all reads)
    for band in ["AX_CARRY_LO", "AX_CARRY_HI", "FETCH_LO", "FETCH_HI", "CMP",
                 "OUTPUT_HI_THIS_STEP", "OUTPUT_HI", "OUTPUT_LO"]:
        if band in dp:
            cells = [round(float(r[axrow, dp[band]+k]), 2) for k in range(16)]
            amx = max(range(16), key=lambda k: cells[k])
            print(f"  {band}: argmax_cell={amx} cells={cells}")


if __name__ == "__main__":
    main()
