#!/usr/bin/env python3
"""Track the AX byte-1 value at the LEV step across blocks (find the leak producer).
spec_k=0, BUILT dims. Usage: python tools/_probe_lev_ax_byte1.py <id> <lev_step> [maxsteps]

AX bytes are emitted at AX_marker (byte0) .. AX_marker+3 (byte3). The byte-k value
is the model's OUTPUT at the AX_marker+(k-1) row (each row predicts the NEXT byte).
We read OUTPUT_LO/HI at the byte-1 prediction row (= AX_marker+0 predicts byte1?).
Actually the register section is [marker, b0, b1, b2, b3]; the token AT marker+k IS
byte-k's emitted value. We read CLEAN_EMBED/OUTPUT at AX_marker+1 (byte1 token row)
across blocks to see which block writes the 0x06 leak.
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


def decode_pair(r, dp, lo, hi):
    L = r[dp[lo]:dp[lo]+16]; H = r[dp[hi]:dp[hi]+16]
    return (int(H.argmax()) << 4) | int(L.argmax()), float(L.max()), float(H.max())


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
    ax = sm[lev].get("AX")
    print(f"id{pid} {desc} exp={exp} lev_step={lev} AX_marker_idx={ax}")
    print(f"  AX section tokens: {[ctx[ax+j] & 0xFF for j in range(5)]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    # byte1 is the token at ax+2 (marker, b0@+1, b1@+2). The PREDICTOR row for
    # the b1 token is ax+1 (the b0 token row predicts b1). Probe both rows.
    for label, rowoff in (("b0_pred(@ax)", 0), ("b1_pred(@ax+1)", 1), ("b1_tok(@ax+2)", 2)):
        pos = ax + rowoff
        print(f"\n-- {label} pos={pos} tok={ctx[pos]&0xFF} --")
        for blk in [10, 11, 12, 13, 18, 24, 26, 27, 30, 32, nblk-1]:
            if blk >= nblk: continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            out, lo, hi = decode_pair(r, dp, "OUTPUT_LO", "OUTPUT_HI")
            print(f"   blk{blk:2d}: OUTPUT=0x{out:02x} (lo_max {lo:.2f} hi_max {hi:.2f})")


if __name__ == "__main__":
    main()
