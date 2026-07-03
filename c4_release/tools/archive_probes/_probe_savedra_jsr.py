#!/usr/bin/env python3
"""Probe the JSR step in the campaign frame: find what compute signals are
available at the JSR step to recover the return PC (90 = JSR_PC + 8), and where
the JSR MEM-store row is. Dump per-step opcode markers + the residual dims that
could carry PC/return-addr at the JSR step.

Usage: python tools/_probe_savedra_jsr.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa


@torch.no_grad()
def main():
    pid = int(sys.argv[1])
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    # stop after the LAST block to see fully-computed residual
    L_last = len(model.blocks) - 1
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    # We want the residual AT each step's marker rows. Use L16-1 (before lev routing) so we
    # see what's available to the materializer's INPUT. Also dump opcode markers.
    L16 = [i for i, b in enumerate(model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    xin = model.forward(padded, stop_after_block=L16 - 1)[0].float()
    def hot(p, base, n=16): return int(torch.argmax(xin[p, base:base + n]).item())
    def g(p, name): return float(xin[p, dimp[name]].item()) if name in dimp else 0.0
    # candidate value-carrying dims that might hold JSR_PC or return PC
    cand = [k for k in dimp if any(s in k for s in (
        "OPCODE_BYTE", "FETCH", "PC_PREV", "REG_PC", "PC_BYTE", "ALU_LO", "ALU_HI",
        "CLEAN_EMBED", "TEMP", "ADDR_B0", "ADDR_B1"))]
    SE = int(Token.STEP_END)
    print(f"STEP_TOKENS={int(Token.STEP_TOKENS)} ctxlen={len(ctx)} prelude={pl} desc={desc}")
    # Walk steps; for each STEP_END row + each opcode, dump OP_* markers and the
    # decoded value of OPCODE_BYTE / FETCH / ALU at the AX/MEM/PC marker rows.
    step = 0; i = pl
    while i < len(ctx) and step < ms:
        # find the STEP_END at the end of this step
        j = i
        while j < len(ctx) and ctx[j] != SE:
            j += 1
        # opcode markers at the STEP_END row (in-step opcode broadcast)
        ops = {n: g(j, n) for n in dimp if n.startswith("OP_") and g(j, n) > 2.0}
        # Decode candidate compute values at the first PC marker row of the step (row i)
        # find PC marker (REG_PC token) -> the marker row
        # decode OPCODE_BYTE_LO/HI and FETCH at the STEP_END row
        ob = hot(j, dimp["OPCODE_BYTE_LO"]) | (hot(j, dimp["OPCODE_BYTE_HI"]) << 4) if "OPCODE_BYTE_LO" in dimp else -1
        fl = (hot(j, dimp["FETCH_LO"]) | (hot(j, dimp["FETCH_HI"]) << 4)) if "FETCH_LO" in dimp else -1
        al = (hot(j, dimp["ALU_LO"]) | (hot(j, dimp["ALU_HI"]) << 4)) if "ALU_LO" in dimp else -1
        print(f"step{step:>2} [rows {i}..{j}] OPCODE_BYTE=0x{ob:02x} FETCH=0x{fl:02x} ALU=0x{al:02x} ops={ {k:round(v,1) for k,v in sorted(ops.items(),key=lambda x:-x[1])[:6]} }")
        i = j + 1; step += 1


if __name__ == "__main__":
    main()
