#!/usr/bin/env python3
"""_agent_io_neural_probe.py — run the strict/burst I/O programs through the REAL
neural pure-forward (the same weights the all-C binary runs) and print the per-step
AX trace, so we can see which opcode (if any) the model mis-computes.  Builds the
compact full-ISA model ONCE, then runs each program.  Tooling only."""
from __future__ import annotations
import sys

import torch

from c4_min import isa
from c4_min import compact_alloc as CA
import c4_min.nibble_pure_forward_complete as PFC
from c4_min.selfhost import _agent_io_progs as IOP


def build_model():
    model, L, _ = CA.build_compact_pure_forward_model(code_size=48)
    model.eval()
    return model, L


def trace_program(model, L, prog: IOP.IOProg, max_steps=200):
    """Run one program, return per-step (pc, opname, ax) + the PRTF byte stream."""
    code = isa.assemble(prog.code)
    out = []
    # Patch run to also record per-step pc/op/ax: easiest is to reuse the driver's
    # trace return (list of AX per step) + reconstruct pc via a python shadow.  But
    # the driver already returns the AX trace; we align it to the executed pcs by a
    # shadow reference interpret to know the pc sequence.
    trace = PFC.run_pure_forward_complete(
        model, L, code, max_steps=max_steps, out=out, seed_mem=prog.seed_mem)
    return trace, bytes(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "echo"
    text = sys.argv[2] if len(sys.argv) > 2 else "hi\n"
    print("building model ...", flush=True)
    model, L = build_model()
    prog = IOP.build(which, "strict", text=text)
    print(f"program: {prog.note}  code={len(prog.code)} ops  "
          f"seed_mem={len(prog.seed_mem)}  expected={prog.expected!r}")
    for i, (name, imm) in enumerate(prog.code):
        print(f"  [{i:2d}] {name} {imm}")
    trace, out = trace_program(model, L, prog)
    print(f"AX trace ({len(trace)} steps): {[hex(a) for a in trace]}")
    print(f"PRTF output: {out!r}   BYTE-EXACT={out == prog.expected}")
