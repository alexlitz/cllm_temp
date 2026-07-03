#!/usr/bin/env python3
"""Does the loop_lea_b0_e8 op (block 45) fire at the step-2 PC value-byte rows?

If it does, that's the source of the PC high-byte 0xE8 leak. Mirrors the working
_probe_loopsum_entax_gate.py structure.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loops2b python tools/_probe_loopsum_pcfire.py [pid] [step] [block]
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
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    plc = bic(model, bc); pl = len(plc)
    ctx = replay(model, plc, max(step + 4, 18))
    sm = smk(ctx, pl)
    pcrow = sm[step]["PC"]; axrow = sm[step]["AX"]
    ol = dp["OUTPUT_LO"]
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    x = model.forward(toks, stop_after_block=blk - 1)
    block = model.blocks[blk]
    xa = block.attn(x)
    ffn = block.ffn
    rows = [("PCmark", pcrow), ("PCb1", pcrow + 1), ("PCb2", pcrow + 2),
            ("PCb3", pcrow + 3), ("PCb4", pcrow + 4), ("AXrow", axrow)]
    print(f"id{pid} step{step} blk{blk} PCrow={pcrow} AXrow={axrow} ffn_hidden={ffn.hidden_dim}")
    for label, row in rows:
        xr = xa[0, row]
        up = F.linear(xr, ffn.W_up) + ffn.b_up
        gate = F.linear(xr, ffn.W_gate) + ffn.b_gate
        hidden = F.silu(up) * gate
        c8 = float((hidden * ffn.W_down[ol + 8]).sum())
        c0 = float((hidden * ffn.W_down[ol + 0]).sum())
        mh = float(hidden.abs().max())
        print(f"  {label:7s} row{row}: max|hidden|={mh:12.1f}  ->OUTPUT_LO+8={c8:11.1f}  +0={c0:11.1f}")


if __name__ == "__main__":
    main()
