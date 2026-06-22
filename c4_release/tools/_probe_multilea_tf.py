#!/usr/bin/env python3
"""TEACHER-FORCED multi-param/multi-local LEA byte-0 discriminator probe.

Extends _probe_learelay_tf.py to the multi-param func targets
(func_square 625, func_max 650/651, func_min 675, nested 950) plus the
single-param keystone references (func_identity 550). Runs the model forward
over the DraftVM byte-exact CORRECT tape so LEA AX rows are drift-free, then
dumps the candidate discriminators at every LEA AX row + the max over non-LEA
AX rows (the false-fire trap).

Run inside campaign env:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
  C4_VM_CACHE_DIR=/tmp/c4cache_multilea python tools/_probe_multilea_tf.py
"""
from __future__ import annotations
import os, sys, contextlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()

# Candidate discriminators to dump at every LEA AX row.
DIMS = [
    "OP_LEA", "ALU_HI+15", "ALU_HI+14", "ALU_HI+13", "ALU_HI+0",
    "ALU_LO+0", "ALU_LO+8", "ALU_LO+15",
    "CMP+7", "HAS_SE", "MARK_AX", "IS_BYTE",
    "FETCH_LO+8", "FETCH_LO+0", "FETCH_HI+15", "FETCH_HI+14", "FETCH_HI+13",
    "MEM_ADDR_SRC", "OP_IMM",
]


def dump_row(arr, ax_pos, dimpos, out_lo_d, hi_d, out_arr):
    vals = {}
    for d in DIMS:
        base = d.split("+")[0]
        off = int(d.split("+")[1]) if "+" in d else 0
        vals[d] = (float(arr[0, ax_pos, dimpos[base] + off].item())
                   if base in dimpos else float("nan"))
    lo_nib = int(torch.argmax(arr[0, ax_pos, out_lo_d:out_lo_d + 16]).item())
    hi_nib = int(torch.argmax(arr[0, ax_pos, hi_d:hi_d + 16]).item())
    cur_byte = (hi_nib << 4) | lo_nib
    out_byte = -1
    if out_arr is not None:
        lo2 = int(torch.argmax(out_arr[0, ax_pos, out_lo_d:out_lo_d + 16]).item())
        hi2 = int(torch.argmax(out_arr[0, ax_pos, hi_d:hi_d + 16]).item())
        out_byte = (hi2 << 4) | lo2
    return vals, cur_byte, out_byte


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    nblocks = len(model.blocks)
    tail_block = nblocks - 1
    mark_ax_d = dimpos["MARK_AX"]
    out_lo_d = dimpos["OUTPUT_LO"]
    hi_d = dimpos["OUTPUT_HI_THIS_STEP"]
    op_lea_d = dimpos["OP_LEA"]
    alu_hi15_d = dimpos["ALU_HI"] + 15
    print(f"nblocks={nblocks} tail_block={tail_block} STEP={STEP} d_model={model.d_model}")

    cap = {}
    h1 = model.blocks[tail_block].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("in", i[0].detach().clone()))
    h2 = model.blocks[tail_block].register_forward_hook(
        lambda m, i, o: cap.__setitem__("out", o.detach().clone()))

    cases = [
        ("func_identity_550 (&x BP-8 -> E8)", 550),
        ("func_square_625 (&x re-read step9? -> E8)", 625),
        ("func_max_650 (&a BP-8 E8, &b BP-16 E0; step11?)", 650),
        ("func_min_675", 675),
        ("nested_quad_950 (deeper frame -> D0?)", 950),
    ]
    for label, pid in cases:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        ot = oracle_tape_and_steps(bc, b"", max_steps=40)
        tape = list(prompt) + list(ot.draft_tokens)
        tok = torch.tensor([tape], dtype=torch.long)
        cap.clear()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                model.forward(tok)
        arr = cap.get("in")
        out_arr = cap.get("out")
        if arr is None:
            print(f"  {label}: NO CAPTURE"); continue
        plen = len(prompt)
        print(f"\n===== {label}  exp={exp} =====")
        print(f"  opcodes: {[hex(o) for o in ot.opcodes]}")
        nsteps = len(ot.steps)
        nonlea_max = -1e9; nonlea_step = -1
        for step in range(nsteps):
            lo = plen + step * STEP
            hi = lo + STEP
            if hi > arr.shape[1]:
                break
            seg = arr[0, lo:hi, mark_ax_d]
            ax_off = int(torch.argmax(seg).item())
            ax_pos = lo + ax_off
            op_lea = float(arr[0, ax_pos, op_lea_d].item())
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            if op_lea < 0.5:
                v = float(arr[0, ax_pos, alu_hi15_d].item())
                if v > nonlea_max:
                    nonlea_max = v; nonlea_step = step
                continue
            vals, cur_byte, out_byte = dump_row(arr, ax_pos, dimpos, out_lo_d, hi_d, out_arr)
            print(f" step{step:2d} op=0x{op:02x} axoff={ax_off:2d} "
                  f"ALU_HI[15/14/13/0]={vals['ALU_HI+15']:6.1f}/{vals['ALU_HI+14']:6.1f}/"
                  f"{vals['ALU_HI+13']:6.1f}/{vals['ALU_HI+0']:6.1f} "
                  f"ALU_LO[0/8/15]={vals['ALU_LO+0']:6.1f}/{vals['ALU_LO+8']:6.1f}/{vals['ALU_LO+15']:6.1f}")
            print(f"          FETCH_LO[8/0]={vals['FETCH_LO+8']:6.1f}/{vals['FETCH_LO+0']:6.1f} "
                  f"FETCH_HI[15/14/13]={vals['FETCH_HI+15']:6.1f}/{vals['FETCH_HI+14']:6.1f}/{vals['FETCH_HI+13']:6.1f} "
                  f"MEM_ADDR_SRC={vals['MEM_ADDR_SRC']:5.1f} CMP+7={vals['CMP+7']:5.1f} "
                  f"IN=0x{cur_byte:02x} OUT=0x{out_byte:02x}")
        print(f"  >>> non-LEA AX max ALU_HI+15 = {nonlea_max:.1f} (step {nonlea_step})")
    h1.remove(); h2.remove()


if __name__ == "__main__":
    main()
