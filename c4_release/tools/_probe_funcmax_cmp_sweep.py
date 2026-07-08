#!/usr/bin/env python3
"""func_max/func_min CMP operand-A leak SWEEP (task #428).

Teacher-forced; for each id, at the COMPARE step's MARK_AX row dump the RAW
ALU_HI/ALU_LO operand-A bands (the pre-SE-relay source) and report:
  a, b (from desc), a_hi/a_lo/b_hi/b_lo, the ALU_HI argmax (read a_hi),
  and whether a spurious cell==b_hi one-hot > the true a_hi cell.

Goal: characterise the leak magnitude vs a_hi (is it always ~6.0 at cell==b_hi
when a_hi==0, weaker when a_hi!=0?), so the discriminator can be constructed
correctly.

Run:
  C4_CAMPAIGN=1 C4_JSR_BP_BYTE3_CLEAR=1 C4_VM_CACHE_DIR=/tmp/fmsw_$$ \
    python tools/_probe_funcmax_cmp_sweep.py --ids 650,653,666,667,675,677,680,690
"""
from __future__ import annotations
import os, sys, contextlib, io, argparse, re

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
CMP_OPS = {0x13: "LT", 0x14: "GT"}


def _parse_ab(desc):
    m = re.search(r"\((\d+),\s*(\d+)\)", desc)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str,
                    default="650,653,666,667,672,675,677,680,690,691,694")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    nblocks = len(model.blocks)
    mark_ax_d = dp["MARK_AX"]
    alu_hi_d = dp["ALU_HI"]
    alu_lo_d = dp["ALU_LO"]

    final_cap = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: final_cap.__setitem__("out", o.detach().clone()))

    print(f"{'id':>4} {'a':>4} {'b':>4} {'a_hi':>4} {'b_hi':>4} "
          f"{'read_ahi':>8} {'ahicell':>8} {'bhicell':>8}  verdict")
    for pid in ids:
        src, exp, desc = PROGS[pid]
        a, b = _parse_ab(desc)
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
        for step in range(nsteps):
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            if op not in CMP_OPS:
                continue
            lo = plen + step * STEP
            hi = lo + STEP
            if hi > fin.shape[1]:
                break
            ax_seg = fin[0, lo:hi, mark_ax_d]
            ax_pos = lo + int(torch.argmax(ax_seg).item())
            a_hi = (a >> 4) & 0xF if a is not None else -1
            b_hi = (b >> 4) & 0xF if b is not None else -1
            cells = fin[0, ax_pos, alu_hi_d:alu_hi_d + 16]
            read_ahi = int(torch.argmax(cells).item())
            ahi_cell = float(cells[a_hi].item()) if a_hi >= 0 else 0.0
            bhi_cell = float(cells[b_hi].item()) if b_hi >= 0 else 0.0
            ok = "OK" if read_ahi == a_hi else "WRONG(reads b_hi)" \
                if read_ahi == b_hi else f"WRONG(reads {read_ahi})"
            print(f"{pid:>4} {a:>4} {b:>4} {a_hi:>4} {b_hi:>4} "
                  f"{read_ahi:>8} {ahi_cell:>8.2f} {bhi_cell:>8.2f}  {ok}")
            break


if __name__ == "__main__":
    main()
