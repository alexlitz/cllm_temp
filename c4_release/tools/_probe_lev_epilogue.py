#!/usr/bin/env python3
"""LEV-epilogue ground-truth probe (spec_k=0, BUILT dims).

For a func_* program, at the LEV step's PC-marker row it dumps:
  * BP value (the frame base) decoded from the BP register section
  * the L9-relayed query address (ADDR_B0/B1 LO/HI one-hots -> the BP+8 query)
    read after the L9 relay physical block
  * every MEM_STORE token in context with its ADDR_KEY-decoded address and
    MEM_VAL_B0 value (the store-disambiguation surface)
  * whether L15 delivers a return-addr value to OUTPUT/TEMP at the PC row

Usage: python tools/_probe_lev_epilogue.py <id> <lev_step> [maxsteps]
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


def onehot_byte(resid, dp, lo_name, hi_name):
    """Decode an 8-bit value from a LO/HI nibble one-hot pair."""
    lo = resid[dp[lo_name]:dp[lo_name]+16]
    hi = resid[dp[hi_name]:dp[hi_name]+16]
    lo_i = int(lo.argmax()); hi_i = int(hi.argmax())
    lo_max = float(lo[lo_i]); hi_max = float(hi[hi_i])
    return (hi_i << 4) | lo_i, lo_max, hi_max


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    if lev >= len(sm):
        print(f"lev_step {lev} out of range (nsteps={len(sm)})"); return
    pc_marker = sm[lev].get("PC")
    print(f"id{pid} {desc} exp={exp} lev_step={lev} PC_marker_idx={pc_marker}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    # L9 relay lands at logical L9 -> physical block. Probe a sweep of blocks at
    # the PC-marker row for the BP+8 query (ADDR_B0/B1) + the OUTPUT/TEMP value.
    print("\n== query address @ PC-marker row across blocks ==")
    for blk in [8, 9, 10, 11, 12, 13, 26, 27, nblk-1]:
        if blk >= nblk: continue
        r = probe.model.forward(padded, stop_after_block=blk)[0][pc_marker]
        b0, b0lo, b0hi = onehot_byte(r, dp, "ADDR_B0_LO", "ADDR_B0_HI")
        b1, b1lo, b1hi = onehot_byte(r, dp, "ADDR_B1_LO", "ADDR_B1_HI")
        out0, _, _ = onehot_byte(r, dp, "OUTPUT_LO", "OUTPUT_HI")
        addr = b0 | (b1 << 8)
        print(f"  blk{blk:2d}: ADDR_B0=0x{b0:02x}(max {b0lo:.1f}/{b0hi:.1f}) "
              f"ADDR_B1=0x{b1:02x}(max {b1lo:.1f}/{b1hi:.1f}) -> addr=0x{addr:04x}({addr})  "
              f"OUTPUT_b0=0x{out0:02x}")

    # Scan all MEM markers (261) in context: decode their stored addr + value.
    print("\n== MEM_STORE tokens in context (addr via ADDR_KEY, val via MEM_VAL/CLEAN_EMBED) ==")
    r_last = probe.model.forward(padded, stop_after_block=nblk-1)[0]
    r_l14 = probe.model.forward(padded, stop_after_block=24)[0]  # post-L14 mem-gen region
    i = pl
    while i < len(ctx):
        if ctx[i] == 261:  # MEM marker
            # MEM section layout: marker, addr(4), val(4). addr byte tokens at i+1..i+4
            addr_bytes = [ctx[i+1+j] & 0xFF for j in range(4)]
            val_bytes = [ctx[i+5+j] & 0xFF for j in range(4)]
            addr = sum(b << (8*j) for j, b in enumerate(addr_bytes))
            val = sum(b << (8*j) for j, b in enumerate(val_bytes))
            # which step?
            stp = sum(1 for m in sm if m.get("MEM", 1<<30) <= i)
            print(f"  pos={i} step~{stp-1}: tokens addr={addr_bytes}=0x{addr:04x}  "
                  f"val={val_bytes}=0x{val:04x}({val})")
            i += 9; continue
        i += 1


if __name__ == "__main__":
    main()
