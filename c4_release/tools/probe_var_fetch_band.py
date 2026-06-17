#!/usr/bin/env python3
"""Dump the full FETCH_LO / FETCH_HI band values at the LEA AX-marker row for
each LEA step of var_mul / var_three, at the block where tail_lea reads them
(block 40 = input to block 41). Confirms which nibbles carry the immediate so a
robust NOT-blocker for tail_lea_local_ax_marker_byte0_e8 can be designed.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"
os.environ["C4_TEST_SPEC_K"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys, contextlib, io
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.compiler import compile_c
from tests.test_suite_1000 import generate_test_programs
from neural_vm.batched_pure_neural import Token
from neural_vm.speculative import DraftVM
from neural_vm.unified_compiler.symbolic_program import (
    idx_to_pc, _decode_static_instruction,
)

STEP = int(Token.STEP_TOKENS)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 275
    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, data = compile_c(src)

    from tools.probe_groundtruth import build_groundtruth_probe
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import (
        compile_full_vm_dynamic,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = build_groundtruth_probe()
        _m, layout = compile_full_vm_dynamic(disk_cache=True)
    dp = layout.dim_positions
    FL, FH = dp["FETCH_LO"], dp["FETCH_HI"]
    CMP = dp.get("CMP")
    MEMADDR = dp.get("MEM_ADDR_SRC")
    dev = p._device

    # which oracle steps are LEA? map pc->imm
    pc_imm = {idx_to_pc(i): _decode_static_instruction(instr)[1]
              for i, instr in enumerate(bc)
              if _decode_static_instruction(instr)[0] == 0}
    from neural_vm.unified_compiler.symbolic_program import (
        SymbolicDeclarativeProgramRunner)
    tr = SymbolicDeclarativeProgramRunner().run(list(bc), data, max_steps=40).trace
    lea_steps = [t.step for t in tr if t.name == "LEA"]
    print(f"id{idx} LEA steps: {lea_steps}  pc->imm {pc_imm}")

    ctx = p._build_context(bc)
    plen = len(ctx)
    dv = DraftVM(list(bc))
    for _ in range(max(lea_steps) + 2):
        dv.step()
        ctx.extend(int(t) for t in dv.draft_tokens())
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)

    BLK = 40  # input to block 41
    resid = p.model.forward(padded, stop_after_block=BLK)[0]
    for step in lea_steps:
        # pc_before of this oracle step:
        t = tr[step]
        imm = pc_imm.get(t.pc_before, "?")
        axrow = plen + step * STEP + 5
        r = resid[axrow].float()
        fl = [round(float(r[FL + k]), 1) for k in range(16)]
        fh = [round(float(r[FH + k]), 1) for k in range(16)]
        fl_mx = max(range(16), key=lambda k: r[FL + k])
        fh_mx = max(range(16), key=lambda k: r[FH + k])
        print(f"\nstep{step} (LEA imm={imm}) axrow={axrow}")
        print(f"  FETCH_LO argmax nib={fl_mx} ({fl[fl_mx]})  nib8={fl[8]} nib0={fl[0]}")
        print(f"  FETCH_HI argmax nib={fh_mx} ({fh[fh_mx]})  nib15={fh[15]} nib14={fh[14]}")


if __name__ == "__main__":
    main()
