#!/usr/bin/env python3
"""var_mul id278: is the SI store-target effective address (&a=0xE8 / &b=0xE0)
present ANYWHERE in the frame at the SI store STEP (steps 9 = SI b? / earlier SI
a), so a fresh band could snapshot it onto the value row (BLUEPRINT B)? Dumps the
SI-step AX marker + the surrounding rows across ADDR_KEY / ADDR_B0 / SP_ADDR /
EFF_ADDR-ish dims at several block depths.

Run: CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_vmcam \
     python tools/_probe_varmul_si_addr.py
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import torch  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from neural_vm.unified_compiler.faithful_autoregressive import build_cpu_model  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


class _CtxStub:
    def __init__(self, model): self.model = model


def bic(model, bc):
    from neural_vm.run_vm import AutoregressiveVMRunner
    return list(AutoregressiveVMRunner._build_context(_CtxStub(model), list(bc), b"", [], ""))


@torch.no_grad()
def replay(model, ctx, max_steps):
    dev = next(model.parameters()).device
    ctx = list(ctx)
    for _ in range(max_steps * STEP):
        t = torch.tensor([ctx], dtype=torch.long, device=dev)
        ctx.append(int(model.forward(t)[0][len(ctx) - 1].argmax().item()))
        if ctx.count(SE) >= max_steps:
            break
    return ctx


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur:
        out.append(cur)
    return out


def amax(v, base, n=16):
    b = v[base:base + n]
    return int(b.argmax().item()), round(float(b.max().item()), 2)


@torch.no_grad()
def main():
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = ("cuda" if (torch.cuda.is_available()
                      and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu")
    model = model.to(dev).eval()
    dp = layout.dim_positions
    progs = generate_test_programs()
    bc = compile_c(progs[278][0])[0]
    ctx = replay(model, bic(model, bc), 15)
    pl = len(bic(model, bc))
    sm = smk(ctx, pl)
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)

    # candidate address-carrying dims
    cand = {}
    for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_KEY_LO", "ADDR_KEY_HI",
               "SP_ADDR_LO", "SP_ADDR_HI", "EFF_ADDR_LO", "EFF_ADDR_HI",
               "OUTPUT_LO", "OUTPUT_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
               "ALU_LO", "ALU_HI"):
        if nm in dp:
            cand[nm] = dp[nm]

    print("=== id278 var_mul: per-step AX marker address-band argmax ===")
    print("    (looking for a step whose AX marker carries &a=0xE8 lo8/hi14 or")
    print("     &b=0xE0 lo0/hi14 -- the SI store step's effective address) ===")
    for blk in (8, 13, 14):
        x = model.forward(toks, stop_after_block=blk)[0]
        print(f"\n  --- block {blk} ---")
        for s in range(min(13, len(sm))):
            ax = sm[s].get("AX")
            if ax is None:
                continue
            parts = []
            for nm, base in cand.items():
                a = amax(x[ax], base)
                if a[1] > 0.3:
                    parts.append(f"{nm}={a[0]}({a[1]})")
            print(f"    step{s:2d} AX@{ax}: " + " ".join(parts))


if __name__ == "__main__":
    main()
