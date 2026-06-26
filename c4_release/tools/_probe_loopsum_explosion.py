#!/usr/bin/env python3
"""Per-block residual trace at the loop_sum id450 step-9 SP-value-byte1 drift row.

The drift row's LM-head byte logits are ~ -5e11 (exploded) -> a marker token
wins by default -> 11-token over-run -> step-10 PC desync. This probe finds WHICH
block injects the explosion and WHICH residual dim carries it, at that abs row.

Usage: CUDA_VISIBLE_DEVICES=1 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_loopdesync python tools/_probe_loopsum_explosion.py
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

SE = int(Token.STEP_END); STEP = int(Token.STEP_TOKENS); SP = int(Token.REG_SP)


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


@torch.no_grad()
def main():
    with contextlib.redirect_stderr(io.StringIO()):
        model, layout = build_cpu_model(disk_cache=True)
    dev = "cuda" if (torch.cuda.is_available()
                     and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    model = model.to(dev); model.eval()
    dp = layout.dim_positions
    inv = {}
    for nm, pos in dp.items():
        inv.setdefault(pos, nm)
    src, exp, desc = generate_test_programs()[450]
    bc = compile_c(src)[0]
    plc = bic(model, bc); pl = len(plc)
    ctx = replay(model, plc, 14)
    se_pos = [i for i, t in enumerate(ctx) if t == SE and i >= pl]
    bounds = [pl] + [p + 1 for p in se_pos]
    a9 = bounds[9]; b9 = bounds[10]
    sp_marker = next(i for i in range(a9, b9) if ctx[i] == SP)
    drift_row = sp_marker + 1  # row predicting SP byte1 (input tok=0xE8)
    print(f"step9 abs[{a9}:{b9}] SP@{sp_marker} drift_row={drift_row} "
          f"input_tok={ctx[drift_row]} (0x{ctx[drift_row]&0xff:02x})")
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    nblk = len(model.blocks)
    prev = None
    for b in range(nblk):
        r = model.forward(toks, stop_after_block=b)[0]
        rr = r[drift_row]
        mx = float(rr.abs().max()); amx = int(rr.abs().argmax())
        nm = inv.get(amx, f"d{amx}")
        norm = float(rr.norm())
        jump = ""
        if prev is not None and mx > prev * 20 and mx > 1e4:
            jump = "  <<< EXPLOSION"
        if b >= nblk - 6 or jump or (prev is not None and mx > 1e4 and prev <= 1e4):
            print(f"  blk{b:2d}: |max|={mx:14.1f} @dim{amx}({nm}) norm={norm:12.1f}{jump}")
        prev = mx
    # final residual: top exploded dims
    r = model.forward(toks, stop_after_block=nblk - 1)[0][drift_row]
    order = r.abs().argsort(descending=True)[:12]
    print("  FINAL residual top-12 dims by |value|:")
    for d in order.tolist():
        print(f"    dim{d:4d}({inv.get(d,'?'):24s}) = {float(r[d]):14.1f}")


if __name__ == "__main__":
    main()
