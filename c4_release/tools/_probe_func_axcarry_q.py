#!/usr/bin/env python3
"""At the func LI query (AX marker) row, dump every plausible address source so
we can see WHERE the correct LEA byte-0 (0xE8 for &a) lives and where the stale
0x0B comes from: AX_CARRY_LO/HI, CLEAN_EMBED_LO/HI, ALU_LO/HI, OUTPUT_LO/HI,
ADDR_B0_LO/HI, FETCH_LO, and the OP_* flags. Trace the LEA step AND the LI step.

Usage: python tools/_probe_func_axcarry_q.py <id> <lea_step> <li_step> [maxsteps]
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


def _find_l15(model, dimp):
    opli = dimp.get("OP_LI_RELAY"); cand = []
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        wq = wq.to_dense() if (wq.is_sparse or wq.is_sparse_csr) else wq
        if opli is not None and wq.shape[1] > opli and abs(float(wq[:, opli].abs().max())) > 1000:
            cand.append((bi, blk.attn.num_heads))
    cand.sort(key=lambda t: -t[1]); return cand[0][0] if cand else None


def nib(resid, p, base):
    sub = resid[p, base:base+16]
    return [(k, round(float(sub[k]), 2)) for k in range(16) if abs(float(sub[k])) > 0.25]


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); want = [int(x) for x in sys.argv[2:4]]
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 14
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
    L15 = _find_l15(model, dimp)
    # Probe the L15-input residual (after all of L0..L14) AND the L8-output too
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    resid = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    bands = ["AX_CARRY_LO", "AX_CARRY_HI", "CLEAN_EMBED_LO", "CLEAN_EMBED_HI",
             "ALU_LO", "ALU_HI", "OUTPUT_LO", "OUTPUT_HI",
             "ADDR_B0_LO", "ADDR_B0_HI", "FETCH_LO"]
    flags = ["OP_LEA", "OP_LI", "OP_LI_RELAY", "MARK_AX", "MEM_ADDR_SRC"]
    print(f"id{pid} {desc} exp={exp}  L15={L15}")
    for stp in want:
        st, en = steps[stp]
        axm = next((i for i in range(st, en+1) if ctx[i] == RAX), None)
        print(f"\n=== step {stp} (rows {st}..{en}) AX_marker={axm} ===")
        if axm is None: continue
        for b in bands:
            d = dimp.get(b)
            if d is None: continue
            print(f"   {b:16s} {nib(resid, axm, d)}")
        fl = []
        for f in flags:
            d = dimp.get(f)
            if d is not None:
                fl.append(f"{f}={float(resid[axm, d]):.2f}")
        print("   flags:", "  ".join(fl))


if __name__ == "__main__":
    main()
