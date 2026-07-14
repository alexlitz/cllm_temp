#!/usr/bin/env python3
"""func_min id675 LT-result leak trace (C4_ALU_OPERAND_SURVIVE regression).

Teacher-forced forward over the DraftVM correct tape (same reliable slice math
as _probe_funcmax_cmp_operand). At the LT compare step it decodes the AX result
byte (OUTPUT_LO/HI at the AX row) at EVERY block INPUT + FINAL, and dumps the
surviving raw ALU_LO/HI@AX band, so we see exactly which block turns the
operand-survive'd ALU band into the wrong AX byte (0xE8 vs 0x01).

  C4_ALU_OPERAND_SURVIVE=1 python tools/_probe_funcmin_leak.py --ids 675
  C4_ALU_OPERAND_SURVIVE=0 python tools/_probe_funcmin_leak.py --ids 675
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
from neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()
CMP_OPS = {0x13: "LT", 0x14: "GT", 0x0f: "EQ", 0x10: "NE", 0x11: "LE", 0x12: "GE"}


def _amax(t):
    return int(torch.argmax(t).item()), float(t.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="675")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    flag = os.environ.get("C4_ALU_OPERAND_SURVIVE", "1")
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dp = layout.dim_positions
    nblocks = len(model.blocks)
    out_lo, out_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI_THIS_STEP"]
    alu_lo, alu_hi = dp["ALU_LO"], dp["ALU_HI"]
    mark_ax_d = dp["MARK_AX"]
    print(f"=== C4_ALU_OPERAND_SURVIVE={flag} nblocks={nblocks} STEP={STEP} ===")

    caps = {}

    def mk(bi):
        def _h(m, inp):
            caps[bi] = inp[0].detach()
        return _h

    for bi in range(nblocks):
        model.blocks[bi].register_forward_pre_hook(mk(bi))
    fin = {}
    model.blocks[nblocks - 1].register_forward_hook(
        lambda m, i, o: fin.__setitem__("o", o.detach()))

    for pid in ids:
        src, exp, desc = PROGS[pid]
        bc = compile_c(src)[0]
        prompt = build_code_prompt(bc, b"")
        ot = oracle_tape_and_steps(bc, b"", max_steps=40)
        tape = list(prompt) + list(ot.draft_tokens)
        tok = torch.tensor([tape], dtype=torch.long)
        caps.clear(); fin.clear()
        with torch.no_grad():
            with contextlib.redirect_stdout(io.StringIO()):
                model.forward(tok)
        plen = len(prompt)
        print(f"\n===== id{pid} {desc!r} exp={exp} =====")
        for step in range(len(ot.steps)):
            op = ot.opcodes[step] if step < len(ot.opcodes) else -1
            if op not in CMP_OPS:
                continue
            lo = plen + step * STEP
            hi = lo + STEP
            ax_off = int(torch.argmax(fin["o"][0, lo:hi, mark_ax_d]).item())
            ax_pos = lo + ax_off
            exp_pc, exp_ax = ot.steps[step]
            print(f"-- step{step} {CMP_OPS[op]} AX@off{ax_off} exp_ax={exp_ax} --")
            # Trace AX result byte + surviving raw ALU band per block.
            print("  blk | OUT_LO argmax | OUT_HI argmax => byte | ALU_LO amax | ALU_HI amax")
            for bi in list(range(nblocks)) + ["FINAL"]:
                t = fin["o"] if bi == "FINAL" else caps.get(bi)
                if t is None:
                    continue
                lo_a, lo_v = _amax(t[0, ax_pos, out_lo:out_lo + 16])
                hi_a, hi_v = _amax(t[0, ax_pos, out_hi:out_hi + 16])
                byte = (hi_a << 4) | lo_a
                al_a, al_v = _amax(t[0, ax_pos, alu_lo:alu_lo + 16])
                ah_a, ah_v = _amax(t[0, ax_pos, alu_hi:alu_hi + 16])
                tag = ""
                # flag the block that first sets the WRONG byte (0xe8) vs right (0x01)
                print(f"  {str(bi):>5} | {lo_a:2d}(={lo_v:6.2f}) | {hi_a:2d}"
                      f"(={hi_v:6.2f}) => 0x{byte:02x} | "
                      f"ALU_LO {al_a:2d}(={al_v:6.2f}) | ALU_HI {ah_a:2d}(={ah_v:6.2f})")


if __name__ == "__main__":
    main()
