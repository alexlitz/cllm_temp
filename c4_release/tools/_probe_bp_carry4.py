#!/usr/bin/env python3
"""Probe 4: signatures for the carry-head Q (ENT MEM val rows) and K (prev BP
byte rows), plus the gate discriminator (ENT-store rows vs SI/SC store rows).
spec_k=0, hook-free.

Usage: python tools/_probe_bp_carry4.py <id> <maxsteps> <ent_step>
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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


def step_rows(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], []).append(i); i += 1; continue
        i += 1
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    ent_step = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    rows = step_rows(ctx, pl)
    nblk = len(probe.model.blocks)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    print(f"id{pid} {desc} ENT step={ent_step}")

    sig_dims = {}
    for k in ("MARK_MEM", "MARK_BP", "MARK_STACK0", "MEM_VAL_B0", "MEM_VAL_B1",
              "MEM_VAL_B2", "MEM_VAL_B3", "MEM_STORE", "OP_ENT", "OP_JSR",
              "OP_SI", "OP_SC", "OP_PSH", "BYTE_INDEX_0", "BYTE_INDEX_1",
              "BYTE_INDEX_2", "BYTE_INDEX_3", "IS_BYTE", "CONST",
              "ADDR_B1_HI", "ADDR_B0_LO"):
        if k in dp:
            sig_dims[k] = dp[k]

    # Read the residual at a chosen block at given positions, dump active sig dims.
    def dump(pos, blk):
        r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
        out = {}
        for k, base in sig_dims.items():
            v = float(r[base])
            if abs(v) > 0.5:
                out[k] = round(v, 1)
        return out

    # Read at the L13 host block (16) — where a carry head would sit — and the
    # late tail block (32) where the dump FFN would run.
    BLK = 16
    print(f"\n=== ENT step {ent_step} MEM rows @ block {BLK} (carry-head Q point) ===")
    mem_idx = rows[ent_step]["MEM"][0]
    for off in range(0, 9):
        pos = mem_idx + off
        label = ("marker" if off == 0 else f"addr{off-1}" if off <= 4 else f"val{off-5}")
        print(f"  off{off} ({label:6s}) pos={pos} tok={ctx[pos]:3d}: {dump(pos, BLK)}")

    prev = ent_step - 1
    bp_idx = rows[prev]["BP"][0]
    print(f"\n=== prev step {prev} BP rows @ block {BLK} (carry-head K point) ===")
    for off in range(0, 5):
        pos = bp_idx + off
        label = "marker" if off == 0 else f"byte{off-1}"
        print(f"  off{off} ({label:6s}) pos={pos} tok={ctx[pos]:3d}: {dump(pos, BLK)}")

    # Gate discriminator: compare ENT-store MEM val rows vs an SI/SC store and a
    # PSH store elsewhere in the program (to ensure dump fires only on ENT).
    print(f"\n=== Other MEM-store steps @ block {BLK} (gate must NOT fire) ===")
    for s, r in enumerate(rows):
        if "MEM" not in r or s == ent_step:
            continue
        mi = r["MEM"][0]
        v0 = dump(mi + 5, BLK)
        print(f"  step{s} MEM val0 pos={mi+5} tok={ctx[mi+5]:3d}: {v0}")

    # Also at the dump block (32) — gate reads there.
    BLK2 = 32
    print(f"\n=== ENT MEM val0 @ block {BLK2} (dump gate read point) ===")
    print(f"  val0 pos={mem_idx+5}: {dump(mem_idx+5, BLK2)}")
    for s, r in enumerate(rows):
        if "MEM" not in r or s == ent_step:
            continue
        mi = r["MEM"][0]
        print(f"  step{s} val0 pos={mi+5} tok={ctx[mi+5]:3d}: {dump(mi+5, BLK2)}")


if __name__ == "__main__":
    main()
