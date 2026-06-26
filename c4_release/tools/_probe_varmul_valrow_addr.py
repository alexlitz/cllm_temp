#!/usr/bin/env python3
"""var_mul id278 step-11 LI-a: dump the committed value rows' available address
signals + the surrounding MEM addr structure so the (A)+(B) build can key on a
REAL discriminator.

For each committed store value row (LI_ZEROADDR_COMMITTED > 1) prints the
ADDR_B0_LO/HI argmax, the raw MEM addr-byte-0 nibble cells, MEM_VAL_B0..B3, the
MEM_STORE / MEM_STORE_AT_VAL indicators, and the BP-relative byte rows around it.
Also dumps the q_a query row's ADDR_B0 nibble amplitudes at L15-in (post-slam) so
we know which Q cells survive for the #313 CAM to key on.

Run: CUDA_VISIBLE_DEVICES=0 C4_VM_CACHE_DIR=/tmp/c4cache_vmcam \
     python tools/_probe_varmul_valrow_addr.py
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
    q_a = sm[11]["AX"]; q_b = sm[12]["AX"]
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    alo = dp["ADDR_B0_LO"]; ahi = dp["ADDR_B0_HI"]
    zc = dp["LI_ZEROADDR_COMMITTED"]
    clo = dp["CLEAN_EMBED_LO"]; chi = dp["CLEAN_EMBED_HI"]
    vb0 = dp["MEM_VAL_B0"]; vb1 = dp["MEM_VAL_B1"]
    vb2 = dp["MEM_VAL_B2"]; vb3 = dp["MEM_VAL_B3"]
    ms = dp.get("MEM_STORE"); msav = dp.get("MEM_STORE_AT_VAL")
    avalid = dp.get("ADDR_B0_VALID")

    print(f"=== id278 var_mul a=8 b=25; q_a={q_a} q_b={q_b} ===")
    x34 = model.forward(toks, stop_after_block=34)[0]

    def cell(j, base):
        return round(float(x34[j, base]), 2)

    print(f"  q_a ADDR_B0: lo{amax(x34[q_a], alo)} hi{amax(x34[q_a], ahi)} "
          f"LO[8]={cell(q_a, alo+8)} HI[14]={cell(q_a, ahi+14)} "
          f"LO[0]={cell(q_a, alo)} HI[0]={cell(q_a, ahi)}")
    print(f"  q_b ADDR_B0: lo{amax(x34[q_b], alo)} hi{amax(x34[q_b], ahi)} "
          f"LO[9]={cell(q_b, alo+9)} HI[1]={cell(q_b, ahi+1)}")

    print("\n  committed value rows + their address signals:")
    for j in range(x34.shape[0]):
        if float(x34[j, zc]) > 1.0:
            cl = amax(x34[j], clo); ch = amax(x34[j], chi)
            val = ch[0] * 16 + cl[0]
            print(f"    row{j} val={val:3d} "
                  f"ADDR_B0=lo{amax(x34[j], alo)}/hi{amax(x34[j], ahi)} "
                  f"VB0={cell(j,vb0)} VB1={cell(j,vb1)} VB2={cell(j,vb2)} VB3={cell(j,vb3)} "
                  f"MS={cell(j,ms) if ms else '-'} MSAV={cell(j,msav) if msav else '-'} "
                  f"AVALID={cell(j,avalid) if avalid else '-'} ZC={cell(j,zc)}")

    # Dump the raw MEM frame around a's and b's stores (find MEM markers in ctx)
    print("\n  MEM markers (token==261) and their addr/val byte structure (ctx pos):")
    mem_tok = 261
    for i in range(pl, len(ctx)):
        if ctx[i] == mem_tok:
            # addr byte 0 is at d=1, value bytes at d=5..8 (per L13 gather comment)
            lo_at = lambda d, b: amax(x34[i + d], b) if i + d < x34.shape[0] else (-1, 0)
            print(f"    MEM@{i}: d1 ADDR_B0(lo{lo_at(1,alo)}/hi{lo_at(1,ahi)}) "
                  f"d1 CLEAN(lo{lo_at(1,clo)}/hi{lo_at(1,chi)}) "
                  f"d5 CLEAN(lo{lo_at(5,clo)}/hi{lo_at(5,chi)})")

    # which #313/#318 head-0 slots would a's value row 331 collect on the query?
    print("\n  KEY: for the #313 CAM to route q_a->a's value row, a's value row")
    print("       must carry ADDR_B0_LO+8 / ADDR_B0_HI+14 (a=&0xE8). Currently it")
    print("       carries lo0/hi0 -> address-blind -> tie with b -> ALiBi picks b.")


if __name__ == "__main__":
    main()
