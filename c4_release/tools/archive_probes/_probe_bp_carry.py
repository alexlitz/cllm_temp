#!/usr/bin/env python3
"""Probe for the BP_SAVE_PREV carry-band strategy (func ENT saved-BP store).

Goal: find where old_BP=[0,0,1,0]=65536 lives as a CLEAN value at the step
BEFORE the ENT saved-BP store, and inspect the ENT-step MEM val byte rows
(which currently emit 0xFF garbage). spec_k=0, hook-free.

Usage: python tools/_probe_bp_carry.py <id> <maxsteps>
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
    """Return list of dicts: per step, {marker_name: [token indices...]}."""
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], []).append(i)
            i += 1; continue
        i += 1
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    rows = step_rows(ctx, pl)
    print(f"id{pid} {desc} exp={exp} n_steps={len(rows)} n_blocks={len(probe.model.blocks)}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    # bands of interest for VALUE carry: OUTPUT_LO/HI (the LM head byte source),
    # H1 (high-nibble emission band), BP register bytes, MEM val bands.
    bands = {}
    for k in ("OUTPUT_LO", "OUTPUT_HI", "OUTPUT_HI_THIS_STEP", "H1", "H3",
              "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3",
              "MEM_STORE", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
              "MARK_BP", "MARK_MEM", "OP_ENT", "OP_JSR"):
        if k in dp:
            bands[k] = dp[k]

    def decode_byte(resid):
        """Decode emitted byte from OUTPUT_LO/HI argmax nibbles."""
        lo = dp["OUTPUT_LO"]; hi = dp["OUTPUT_HI"]
        lo_n = int(torch.argmax(resid[lo:lo+16]))
        hi_n = int(torch.argmax(resid[hi:hi+16]))
        return (hi_n << 4) | lo_n, lo_n, hi_n

    # Identify the ENT step: first step whose OP_ENT broadcast is high AND it has
    # a MEM section. For id550 that's step 1 (main ENT) and step 5 (callee ENT).
    print("\n=== Per-step MEM marker presence + opcode ===")
    for s, r in enumerate(rows):
        has_mem = "MEM" in r
        print(f"  step{s}: markers={ {k: len(v) for k,v in r.items()} } has_mem={has_mem}")

    # Focus: the callee ENT step (step 5 for id550) — its saved-BP store.
    # Inspect every MEM-section row at that step across blocks.
    ent_steps = [s for s, r in enumerate(rows) if "MEM" in r]
    print(f"\n=== MEM-bearing steps: {ent_steps} ===")

    for s in ent_steps:
        r = rows[s]
        mem_idxs = r["MEM"]
        print(f"\n##### step {s}: MEM token indices {mem_idxs} #####")
        # The MEM section is [MEM_marker, addr0,addr1,addr2,addr3, val0,val1,val2,val3]
        # The probe groups consecutive MEM-marker tokens; only the first is the
        # marker, the rest are byte rows. Show the emitted byte at each row by
        # decoding the FINAL-block residual.
        mem_marker = mem_idxs[0]
        for off in range(0, 9):
            pos = mem_marker + off
            if pos >= len(ctx):
                break
            rfin = probe.model.forward(padded, stop_after_block=nblk-1)[0][pos]
            bval, lo_n, hi_n = decode_byte(rfin)
            label = ("marker" if off == 0 else
                     f"addr{off-1}" if off <= 4 else f"val{off-5}")
            print(f"   off{off:1d} ({label:6s}) pos={pos} tok={ctx[pos]:3d} "
                  f"emit_byte=0x{bval:02x} (lo={lo_n} hi={hi_n})")


if __name__ == "__main__":
    main()
