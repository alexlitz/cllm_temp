#!/usr/bin/env python3
"""Trace the PC value at the step-9 (post-LEV ADJ) PC marker across blocks, and
the durable cross-step PC the step reads. Determines whether the LEV's restored
PC=90 propagates as the durable PC into step 9 (so ADJ can advance 90->98) or
whether step 9 reads a stale default (10) and so cannot advance.

Decodes OUTPUT_LO/HI (emitted PC) and any durable PC band (PC_PREV / REG_PC
section bytes) at each block, at the step-(lev) and step-(lev+1) PC markers.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_func_step9_pc.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


def hot(r, dp, lo, hi):
    if lo not in dp or hi not in dp:
        return None
    a = r[dp[lo]:dp[lo]+16]; b = r[dp[hi]:dp[hi]+16]
    return (int(b.argmax()) << 4) | int(a.argmax())


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    lev = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    nblk = len(probe.model.blocks)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    # candidate PC-carrying bands
    pcbands = [("OUTPUT", "OUTPUT_LO", "OUTPUT_HI")]
    for cand in ("REG_PC_LO", "PC_LO", "PC_PREV_LO", "PC_BYTE_LO", "POST_PRTF_PC_LO"):
        if cand in dp:
            pcbands.append((cand.replace("_LO", ""), cand, cand.replace("_LO", "_HI")))
    print(f"id{pid} {desc} exp={exp} : PC bands at step{lev} & step{lev+1} PC markers")
    print("  bands:", [b[0] for b in pcbands])
    for st in (lev, lev + 1):
        row = sm[st].get("PC")
        print(f"\n=== step {st} PC-marker row={row} ===")
        for blk in [0, 4, 8, 12, 16, 20, 24, 28, 32, nblk - 1]:
            if blk >= nblk:
                continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][row]
            cells = []
            for nm, lo, hi in pcbands:
                v = hot(r, dp, lo, hi)
                if v is not None:
                    cells.append(f"{nm}=0x{v:02x}({v})")
            print(f"  blk{blk:2d}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
