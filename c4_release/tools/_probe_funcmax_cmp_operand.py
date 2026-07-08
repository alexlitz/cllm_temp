#!/usr/bin/env python3
"""func_max/func_min CMP loaded-operand two-hot probe (task #428).

TEACHER-FORCED forward over the DraftVM byte-exact CORRECT tape. Focuses on the
COMPARE step (op 0x13 LT / 0x14 GT). At that step's MARK_SE_ONLY row (where the
L9 CMP factory reads its operands) it dumps:
  * operand-A band: SE_ALU_LO / SE_ALU_HI (16 cells each) + raw ALU_LO/HI at the
    MARK_AX row (the pre-relay source the L8 wrap cleans),
  * operand-B band: SE_AX_CARRY_LO / SE_AX_CARRY_HI + raw AX_CARRY_LO/HI,
  * the L9 CMP cascade (CMP+0..3) at the SE row,
  * the decoded branch result.

The func_add root is a TWO-HOT ALU_HI on operand A (true a//16 one-hot PLUS a
spurious leak cell). This probe checks whether the SAME leak contaminates the
CMP-path operand read (SE_ALU_HI two-hot) at the compare step.

Run:
  C4_CAMPAIGN=1 C4_JSR_BP_BYTE3_CLEAR=1 C4_VM_CACHE_DIR=/tmp/fmcmp_$$ \
    python tools/_probe_funcmax_cmp_operand.py --ids 650,675
"""
from __future__ import annotations
import os, sys, contextlib, io, argparse

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
AX_MARK_OFF = 5

# LT=0x13, GT=0x14 (the compare opcodes for func_min / func_max).
CMP_OPS = {0x13: "LT", 0x14: "GT", 0x0f: "EQ", 0x10: "NE",
           0x11: "LE", 0x12: "GE"}


def _cells(arr, pos, base):
    return [round(float(arr[0, pos, base + i].item()), 2) for i in range(16)]


def _amax(arr, pos, base):
    sl = arr[0, pos, base:base + 16]
    return int(torch.argmax(sl).item()), float(sl.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="650,675")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    nblocks = len(model.blocks)
    mark_ax_d = dp["MARK_AX"]
    mark_se_d = dp["MARK_SE_ONLY"]
    cmp_d = dp["CMP"]
    print(f"nblocks={nblocks} STEP={STEP} d_model={model.d_model}")
    print(f"OPERAND_CAM_FIX flag effect -> "
          f"C4_OPERAND_CAM_FIX={os.environ.get('C4_OPERAND_CAM_FIX','<unset>')}")

    bands = {}
    for nm in ["SE_ALU_LO", "SE_ALU_HI", "SE_AX_CARRY_LO", "SE_AX_CARRY_HI",
               "ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI"]:
        if nm in dp:
            bands[nm] = dp[nm]

    final_cap = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: final_cap.__setitem__("out", o.detach().clone()))

    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        ot = oracle_tape_and_steps(bc, b"", max_steps=40)
        tape = list(prompt) + list(ot.draft_tokens)
        tok = torch.tensor([tape], dtype=torch.long)
        final_cap.clear()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                model.forward(tok)
        fin = final_cap["out"]
        plen = len(prompt)
        nsteps = len(ot.steps)
        print(f"\n===== id{pid} {desc!r} exp={exp} nsteps={nsteps} =====")
        # Find the compare step(s).
        for step in range(nsteps):
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            if op not in CMP_OPS:
                continue
            lo = plen + step * STEP
            hi = lo + STEP
            if hi > fin.shape[1]:
                break
            # SE row: argmax MARK_SE_ONLY in the slice.
            se_seg = fin[0, lo:hi, mark_se_d]
            se_off = int(torch.argmax(se_seg).item())
            se_pos = lo + se_off
            ax_seg = fin[0, lo:hi, mark_ax_d]
            ax_off = int(torch.argmax(ax_seg).item())
            ax_pos = lo + ax_off
            exp_pc, exp_ax = ot.steps[step]
            print(f"  -- step{step} op=0x{op:02x}({CMP_OPS[op]}) "
                  f"exp(pc={exp_pc},ax={exp_ax}) "
                  f"SE@off{se_off} AX@off{ax_off} --")
            cmp_casc = [round(float(fin[0, se_pos, cmp_d + i].item()), 2)
                        for i in range(4)]
            print(f"     CMP cascade @SE = {cmp_casc}  "
                  f"[hi_lt,hi_eq,lo_eq,lo_lt]")
            # Operand A (SE_ALU) at SE row; raw ALU at AX row.
            for aband, rband, lbl in (("SE_ALU_HI", "ALU_HI", "opA_HI"),
                                      ("SE_ALU_LO", "ALU_LO", "opA_LO")):
                if aband in bands:
                    a_arg, a_max = _amax(fin, se_pos, bands[aband])
                    # count "hot" cells (> 0.5) -> two-hot detection
                    cells = fin[0, se_pos, bands[aband]:bands[aband] + 16]
                    hot = [(i, round(float(cells[i].item()), 2))
                           for i in range(16) if float(cells[i].item()) > 0.5]
                    print(f"     {lbl} SE argmax={a_arg}(={a_max:.2f}) "
                          f"HOT(>0.5)={hot}")
                if rband in bands:
                    r_arg, r_max = _amax(fin, ax_pos, bands[rband])
                    cells = fin[0, ax_pos, bands[rband]:bands[rband] + 16]
                    hot = [(i, round(float(cells[i].item()), 2))
                           for i in range(16) if float(cells[i].item()) > 0.5]
                    print(f"     {lbl} rawALU@AX argmax={r_arg}(={r_max:.2f}) "
                          f"HOT(>0.5)={hot}")
            # Operand B (SE_AX_CARRY) at SE row; raw at AX row.
            for aband, rband, lbl in (("SE_AX_CARRY_HI", "AX_CARRY_HI", "opB_HI"),
                                      ("SE_AX_CARRY_LO", "AX_CARRY_LO", "opB_LO")):
                if aband in bands:
                    a_arg, a_max = _amax(fin, se_pos, bands[aband])
                    cells = fin[0, se_pos, bands[aband]:bands[aband] + 16]
                    hot = [(i, round(float(cells[i].item()), 2))
                           for i in range(16) if float(cells[i].item()) > 0.5]
                    print(f"     {lbl} SE argmax={a_arg}(={a_max:.2f}) "
                          f"HOT(>0.5)={hot}")
                if rband in bands:
                    r_arg, r_max = _amax(fin, ax_pos, bands[rband])
                    cells = fin[0, ax_pos, bands[rband]:bands[rband] + 16]
                    hot = [(i, round(float(cells[i].item()), 2))
                           for i in range(16) if float(cells[i].item()) > 0.5]
                    print(f"     {lbl} rawALU@AX argmax={r_arg}(={r_max:.2f}) "
                          f"HOT(>0.5)={hot}")


if __name__ == "__main__":
    main()
