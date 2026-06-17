#!/usr/bin/env python3
"""Per-step active opcode (OP_* dim) at the AX-marker row for a func_* program,
plus the emitted PC/AX. spec_k=0, hook-free. Tells us exactly which opcode runs
at each step so we can map the AX clobber to LEA vs LI vs LEV.

Usage: python tools/_probe_func_opcode_per_step.py <id> [maxsteps] [blk]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT)
RAX = int(Token.REG_AX); RPC = int(Token.REG_PC)
OP_NAMES = ["OP_LEA","OP_IMM","OP_JMP","OP_JSR","OP_BZ","OP_BNZ","OP_ENT","OP_ADJ",
            "OP_LEV","OP_LI","OP_LC","OP_SI","OP_SC","OP_PSH"]


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); ms = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; device = probe._device
    dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t == STEP_END or t == HALT:
            steps.append((s, i)); s = i + 1
    padded = torch.tensor([ctx], dtype=torch.long, device=device)
    resid = model.forward(padded, stop_after_block=blk)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    print(f"id{pid} {desc} exp={exp}  (opcode read at block {blk})")
    for si, (st, en) in enumerate(steps):
        # AX marker row
        axm = next((i for i in range(st, en + 1) if ctx[i] == RAX), None)
        pcm = next((i for i in range(st, en + 1) if ctx[i] == RPC), None)
        if axm is None:
            continue
        # active opcode: scan OP_* dims at the PC marker row (opcode decode row)
        row = pcm if pcm is not None else axm
        active = []
        for nm in OP_NAMES:
            d = dimp.get(nm)
            if d is None:
                continue
            v = float(resid[row, d].item())
            if v > 0.5:
                active.append(f"{nm}={v:.1f}")
        ax_b0 = ctx[axm + 1] if axm + 1 <= en else None
        ax_b1 = ctx[axm + 2] if axm + 2 <= en else None
        pc_b0 = ctx[pcm + 1] if (pcm is not None and pcm + 1 <= en) else None
        print(f"  step {si:2d}: PC_b0={pc_b0} AX=0x{(ax_b1 or 0):02x}{(ax_b0 or 0):02x}"
              f"  active_op@PCrow={active}")


if __name__ == "__main__":
    main()
