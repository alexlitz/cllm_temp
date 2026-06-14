#!/usr/bin/env python3
"""Probe 3: is old_BP available in CLEAN_EMBED (token embedding) at the prev
step's BP byte rows? CLEAN_EMBED holds the EMITTED token's value, which is the
clean old_BP even when OUTPUT is corrupted. spec_k=0, hook-free.

Usage: python tools/_probe_bp_carry3.py <id> <maxsteps> <ent_step>
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
    print(f"id{pid} {desc} exp={exp} ENT step={ent_step} (prev={ent_step-1})")

    # The carry head V-reads at the EMBEDDING (block 0 input) or an early block.
    # CLEAN_EMBED holds the emitted-token value. Check which bands hold old_BP
    # cleanly at the prev step's BP byte rows, AT THE EMBEDDING (pre-block-0) and
    # at a few early blocks.
    cands = {}
    for k in ("CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "OUTPUT_LO", "OUTPUT_HI",
              "H1", "H3", "AX_FULL_LO", "AX_FULL_HI"):
        if k in dp:
            cands[k] = dp[k]

    prev = ent_step - 1
    bp_idx = rows[prev]["BP"][0]
    print(f"\nprev-step BP marker idx={bp_idx}; bytes at +1..+4")
    print("Emitted BP byte tokens (the clean old_BP):",
          [ctx[bp_idx + o] for o in range(1, 5)])

    def nib(resid, base):
        idx = int(torch.argmax(resid[base:base+16]))
        return idx, round(float(resid[base+idx]), 1)

    # At each prev BP byte row, read candidate band nibble decodes at the
    # EMBEDDING (block input, stop_after_block=-1 not supported; use raw embed).
    emb = probe.model.embed(padded)[0]  # token embeddings (pre-block-0)
    print("\n=== prev BP byte rows: EMBEDDING band nibble decodes ===")
    for off in range(1, 5):
        pos = bp_idx + off
        e = emb[pos]
        parts = [f"  byte{off-1} pos={pos} tok={ctx[pos]:3d}:"]
        for k, base in cands.items():
            if base + 16 > e.shape[0]:
                continue
            i, m = nib(e, base)
            if abs(m) > 0.3:
                parts.append(f"{k}={i}@{m}")
        print(" ".join(parts))

    # Also: at the MARKER row (which predicts byte0) and across early blocks,
    # to confirm there's a row where old_BP nibbles are clean in OUTPUT.
    print("\n=== prev BP byte rows across early blocks: OUTPUT_LO/HI nibble ===")
    OL = dp["OUTPUT_LO"]; OH = dp["OUTPUT_HI"]
    for off in range(1, 5):
        pos = bp_idx + off
        parts = [f"  byte{off-1} pos={pos} tok={ctx[pos]:3d}:"]
        for blk in [0, 1, 2, 3, 5, 8]:
            if blk >= nblk:
                continue
            r = probe.model.forward(padded, stop_after_block=blk)[0][pos]
            li, _ = nib(r, OL); hi, _ = nib(r, OH)
            parts.append(f"b{blk}=0x{(hi<<4)|li:02x}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
