#!/usr/bin/env python3
"""TEACHER-FORCED residual probe for the LEA byte-0 relay (ROOT 1, 30-tok frame).

Unlike _probe_learelay_band.py (autoregressive, drifts after first divergence),
this runs the model forward over the DraftVM's byte-exact CORRECT tape, so the
LEA AX-marker rows are CLEAN (no drift). Reads the residual entering the L25
tail block at each LEA AX row to characterize the ALU_HI+15 magnitude
discriminator for the 3 frame offsets 0xE8/0xE0/0xD8.

Run inside campaign env:
  C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
  C4_VM_CACHE_DIR=/tmp/c4cache_learelay_wt python tools/_probe_learelay_tf.py
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

DIMS = [
    "OP_LEA", "ALU_HI+15", "ALU_HI+14", "ALU_HI+13", "ALU_LO+0",
    "CMP+7", "HAS_SE", "MARK_AX", "IS_BYTE",
    "FETCH_LO+8", "FETCH_LO+0", "FETCH_HI+15", "FETCH_HI+14", "FETCH_HI+13",
    "MEM_ADDR_SRC", "OP_IMM", "OP_ADD", "OP_SUB", "OP_DIV", "OP_MOD",
    "MARK_PC", "MARK_SP", "MARK_BP", "MARK_STACK0", "MARK_MEM",
]
SHOW_NONLEA = os.environ.get("SHOW_NONLEA") == "1"


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    nblocks = len(model.blocks)
    tail_block = nblocks - 1
    mark_ax_d = dimpos["MARK_AX"]
    out_lo_d = dimpos["OUTPUT_LO"]
    hi_d = dimpos["OUTPUT_HI_THIS_STEP"]
    print(f"nblocks={nblocks} tail_block={tail_block} STEP={STEP} d_model={model.d_model}")

    cap = {}

    def pre_hook(mod, inp):
        cap["tail_in"] = inp[0].detach().clone()

    def post_hook(mod, inp, out):
        cap["tail_out"] = out.detach().clone()

    handle = model.blocks[tail_block].register_forward_pre_hook(pre_hook)
    handle2 = model.blocks[tail_block].register_forward_hook(post_hook)

    cases = [
        ("func_identity_550 (only &x BP-8 want 0xE8)", 550),
        ("var_mul_275 (&a BP-8=E8, &b BP-16=E0)", 275),
        ("var_three_300 (&a=E8 &b=E0 &c BP-24=D8)", 300),
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
        arr = cap.get("tail_in")
        if arr is None:
            print(f"  {label}: NO CAPTURE"); continue
        plen = len(prompt)
        print(f"\n===== {label}  exp={exp} =====")
        print(f"  opcodes per step: {[hex(o) for o in ot.opcodes]}")
        nsteps = len(ot.steps)
        # scan ALL AX rows: track max ALU_HI+15 / ALU_LO+0 over NON-LEA AX rows
        nonlea_max_aluhi15 = -1e9
        nonlea_max_aluhi15_step = -1
        alu_hi15_d = dimpos["ALU_HI"] + 15
        for step in range(nsteps):
            lo = plen + step * STEP
            hi = lo + STEP
            if hi > arr.shape[1]:
                break
            seg = arr[0, lo:hi, mark_ax_d]
            ax_off = int(torch.argmax(seg).item())
            ax_pos = lo + ax_off
            op_lea = float(arr[0, ax_pos, dimpos["OP_LEA"]].item())
            if op_lea < 0.5:
                v = float(arr[0, ax_pos, alu_hi15_d].item())
                if v > nonlea_max_aluhi15:
                    nonlea_max_aluhi15 = v; nonlea_max_aluhi15_step = step
                if SHOW_NONLEA and float(arr[0, ax_pos, mark_ax_d].item()) > 0.5:
                    print(f"   nonLEA step{step:2d} op=0x{ot.opcodes[step] if step<len(ot.opcodes) else -1:02x} "
                          f"ALU_HI+15={v:7.2f} OP_LEA={op_lea:5.2f}")
                continue
            vals = {}
            for d in DIMS:
                base = d.split("+")[0]; off = int(d.split("+")[1]) if "+" in d else 0
                vals[d] = float(arr[0, ax_pos, dimpos[base] + off].item()) if base in dimpos else float("nan")
            lo_nib = int(torch.argmax(arr[0, ax_pos, out_lo_d:out_lo_d + 16]).item())
            hi_nib = int(torch.argmax(arr[0, ax_pos, hi_d:hi_d + 16]).item())
            cur_byte = (hi_nib << 4) | lo_nib
            arr_out = cap.get("tail_out")
            out_byte = -1
            if arr_out is not None:
                lo_n2 = int(torch.argmax(arr_out[0, ax_pos, out_lo_d:out_lo_d + 16]).item())
                hi_n2 = int(torch.argmax(arr_out[0, ax_pos, hi_d:hi_d + 16]).item())
                out_byte = (hi_n2 << 4) | lo_n2
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            print(f" step{step:2d} op=0x{op:02x} axrow={ax_off:2d} OP_LEA={op_lea:5.2f} "
                  f"ALU_HI+15={vals['ALU_HI+15']:7.2f} ALU_HI+14={vals['ALU_HI+14']:7.2f} "
                  f"ALU_HI+13={vals['ALU_HI+13']:7.2f} ALU_LO+0={vals['ALU_LO+0']:6.2f} "
                  f"CMP+7={vals['CMP+7']:5.2f} HAS_SE={vals['HAS_SE']:5.2f} IS_BYTE={vals['IS_BYTE']:6.2f} "
                  f"FETCH_LO+8={vals['FETCH_LO+8']:5.2f} "
                  f"FETCH_HI+15={vals['FETCH_HI+15']:5.2f} FETCH_HI+14={vals['FETCH_HI+14']:5.2f} "
                  f"OP_IMM={vals['OP_IMM']:5.2f} IN=0x{cur_byte:02x} TAILOUT=0x{out_byte:02x}")
        print(f"  >>> NON-LEA AX rows max ALU_HI+15 = {nonlea_max_aluhi15:.2f} (step {nonlea_max_aluhi15_step})")
    handle.remove()
    handle2.remove()


if __name__ == "__main__":
    main()
