#!/usr/bin/env python3
"""var_mul id278: where does the store-address provenance live? Dump the full
MEM frame (all d=0..9 from each MEM marker) of CLEAN_EMBED + ADDR_B0 + every
address-ish dim, so we can find a dim that DISTINGUISHES a's store frame
(&a=0xE8) from b's store frame (&b=0xE0) and route the #313 CAM by it.

Run: CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_vmcam \
     python tools/_probe_varmul_memaddr_src.py
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
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    clo = dp["CLEAN_EMBED_LO"]; chi = dp["CLEAN_EMBED_HI"]
    alo = dp["ADDR_B0_LO"]; ahi = dp["ADDR_B0_HI"]

    # Probe at several block depths to see where addr provenance lives
    for blk in (13, 14, 34):
        x = model.forward(toks, stop_after_block=blk)[0]
        print(f"\n===== block {blk} =====")
        # a's store MEM frame is at 326, b's at 446 (from prior probe)
        for label, mempos in (("a-store &a=0xE8", 326), ("b-store &b=0xE0", 446)):
            print(f"  {label} MEM@{mempos}:")
            for d in range(0, 10):
                j = mempos + d
                if j >= x.shape[0]:
                    break
                cl = amax(x[j], clo); ch = amax(x[j], chi)
                al = amax(x[j], alo); ah = amax(x[j], ahi)
                # also raw embed token
                print(f"    d{d} pos{j} tok={ctx[j]:3d} "
                      f"CLEAN(lo{cl}/hi{ch}) ADDR_B0(lo{al}/hi{ah})")


if __name__ == "__main__":
    main()
