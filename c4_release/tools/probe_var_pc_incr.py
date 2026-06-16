#!/usr/bin/env python3
"""Probe the var step-4 IMM PC byte0 increment: what EMBED (prev-PC) does L3
read, and which L3 rule produces OUTPUT 0x42 instead of 0x3a?

READ-ONLY. spec_k=0, hook-free. Usage: python tools/probe_var_pc_incr.py [id]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402
from neural_vm.batched_pure_neural import Token  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402

MARKERS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX",
           int(Token.REG_SP): "SP", int(Token.REG_BP): "BP",
           int(Token.STEP_END): "STEP_END", int(Token.HALT): "HALT"}


def find_pc_pred_rows(ctx, prompt_len):
    out = []
    i = prompt_len
    n = len(ctx)
    step = 0
    while i < n:
        t = ctx[i]
        name = MARKERS.get(t)
        if name == "PC":
            out.append((step, i, i + 1))
            i += 5
        elif name in ("AX", "SP", "BP"):
            i += 5
        elif name == "STEP_END":
            step += 1
            i += 1
        elif name == "HALT":
            break
        else:
            i += 1
    return out


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    target_step = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)
    print(f"=== id={idx} {desc} ===")

    probe = build_groundtruth_probe()
    m = probe.model
    dp = m.dim_positions
    ctx = probe._final_context(bc)
    prompt_len = len(probe._build_context(bc))
    pred_rows = find_pc_pred_rows(ctx, prompt_len)

    tr = [r for r in pred_rows if r[0] == target_step]
    if not tr:
        print("no PC pred row for step", target_step); return
    _, pred_row, byte_pos = tr[0]
    print(f"step {target_step}: pred_row={pred_row} byte_pos={byte_pos} "
          f"emitted_token={ctx[byte_pos]} (0x{ctx[byte_pos]:02x})")
    print(f"ctx around marker: {ctx[pred_row-1:pred_row+6]}")

    resids = []
    hooks = []
    def mk(i):
        def h(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            resids.append((i, o.detach()[0, pred_row].cpu()))
        return h
    for bi, blk in enumerate(m.blocks):
        hooks.append(blk.register_forward_hook(mk(bi)))
    toks = torch.tensor([ctx], device=next(m.parameters()).device)
    with torch.no_grad():
        m(toks)
    for hk in hooks:
        hk.remove()

    EMBED_LO = dp.get("EMBED_LO"); EMBED_HI = dp.get("EMBED_HI")
    OUT_LO = dp.get("OUTPUT_LO"); OUT_HI = dp.get("OUTPUT_HI")
    print(f"dims: EMBED_LO={EMBED_LO} EMBED_HI={EMBED_HI} "
          f"OUTPUT_LO={OUT_LO} OUTPUT_HI={OUT_HI}")

    def nib(vec, base):
        if base is None:
            return None
        sl = vec[base:base+16]
        am = int(sl.argmax().item())
        return am, float(sl[am].item())

    print(f"\n{'blk':>3} {'EMBED_LO':>11} {'EMBED_HI':>11} "
          f"{'OUT_LO':>11} {'OUT_HI':>11}  OUTdec")
    for bi, vec in resids:
        el = nib(vec, EMBED_LO); eh = nib(vec, EMBED_HI)
        ol = nib(vec, OUT_LO); oh = nib(vec, OUT_HI)
        outdec = (ol[0] | (oh[0] << 4)) if (ol and oh) else 0
        embdec = (el[0] | (eh[0] << 4)) if (el and eh) else 0
        def f(x):
            return f"{x[0]}({x[1]:.2f})" if x else "-"
        print(f"{bi:>3} {f(el):>11} {f(eh):>11} {f(ol):>11} {f(oh):>11}  "
              f"OUT=0x{outdec:02x} EMB=0x{embdec:02x}")


if __name__ == "__main__":
    main()
