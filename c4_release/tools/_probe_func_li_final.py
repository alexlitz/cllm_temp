#!/usr/bin/env python3
"""Decode the FINAL (post-all-blocks) AX byte-0 at the func LI AX-marker row,
and trace the OUTPUT_LO/HI band argmax block-by-block from L15 to the end, to
see whether the L15-delivered LI value survives to the emission or is clobbered
by a downstream (blk35-41 tail) writer.

Usage: python tools/_probe_func_li_final.py <id> <li_step> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT); RAX = int(Token.REG_AX)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); li_step = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t in (STEP_END, HALT):
            steps.append((s, i)); s = i + 1
    st, en = steps[li_step]
    axm = next(i for i in range(st, en+1) if ctx[i] == RAX)
    # the emit token row = axm+1 (byte 0 token)
    emit = axm + 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    nblocks = len(model.blocks)
    ol = dimp.get("OUTPUT_LO"); oh = dimp.get("OUTPUT_HI")
    print(f"id{pid} {desc} exp={exp}  LI step={li_step} AX_marker={axm} emit_row={emit} emit_tok={ctx[emit]} nblocks={nblocks}")
    print("  (emit token already decoded =", ctx[emit], ")")
    print("\n  block-by-block OUTPUT argmax at AX-marker row", axm, "and emit row", emit)
    for stop in range(10, nblocks):
        resid = model.forward(padded, stop_after_block=stop)[0].float()
        if resid.is_sparse: resid = resid.to_dense()
        def dec(p):
            lo = int(torch.argmax(resid[p, ol:ol+16]).item())
            hi = int(torch.argmax(resid[p, oh:oh+16]).item())
            lov = float(resid[p, ol+lo]); hiv = float(resid[p, oh+hi])
            return (hi<<4)|lo, lov, hiv
        v_ax, loa, hia = dec(axm)
        v_em, loe, hie = dec(emit)
        print(f"   blk{stop:2d}: AX-row val=0x{v_ax:02x}({v_ax}) [lo{loa:.0f}/hi{hia:.0f}]   emit-row val=0x{v_em:02x}({v_em}) [lo{loe:.0f}/hi{hie:.0f}]")


if __name__ == "__main__":
    main()
