#!/usr/bin/env python3
"""Dump the raw ADDR_B0/B1/B2 LO+HI one-hot vectors (full 16-wide) at the LEV
query PC-marker row and at a set of candidate store rows, so we can see the
*confidence* (magnitude + sharpness) of each store's 24-bit address key.

The CAM match score is bilinear in the bit-encoded address one-hots; a store
whose address one-hot is SOFTER (lower peak) loses the binary-address slots even
when its address VALUE matches the query. This probe quantifies that gap.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/_probe_lev_addr_conf.py <id> <lev_step> <pos...> [--ms N]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_markers(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


def find_l15_block(probe):
    for phys, blk in enumerate(probe.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None: continue
        nh = getattr(attn, "num_heads", None)
        if nh is not None and nh >= 15: return phys
    return None


def dump_addr(r, dp, label):
    parts = []
    for nm in ("ADDR_B0_LO", "ADDR_B0_HI", "ADDR_B1_LO", "ADDR_B1_HI", "ADDR_B2_LO", "ADDR_B2_HI"):
        v = r[dp[nm]:dp[nm]+16]
        i = int(v.argmax()); peak = float(v[i]); l2 = float(v.norm())
        parts.append(f"{nm}=arg{i:2d}(pk{peak:5.2f},l2{l2:5.2f})")
    print(f"  {label}: " + "  ".join(parts))


@torch.no_grad()
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ms = 20
    if "--ms" in sys.argv:
        ms = int(sys.argv[sys.argv.index("--ms")+1])
    pid = int(args[0]); lev = int(args[1]); poss = [int(x) for x in args[2:]]
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    pc_marker = sm[lev].get("PC")
    l15_blk = find_l15_block(probe)
    in_blk = l15_blk - 1
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    resid_in = probe.model.forward(padded, stop_after_block=in_blk)[0]
    print(f"id{pid} {desc} lev_step={lev} q@PC={pc_marker} L15blk={l15_blk}")
    print("\n== address one-hot confidence (resid into L15) ==")
    dump_addr(resid_in[pc_marker], dp, f"QUERY @{pc_marker}")
    for p in poss:
        dump_addr(resid_in[p], dp, f"store @{p}")


if __name__ == "__main__":
    main()
