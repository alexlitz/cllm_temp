#!/usr/bin/env python3
"""LM-head logit attribution for the 16-bit bitwise byte-0 result token,
at its PREDICTOR row, in the CAMPAIGN config.

For or_16bit (0x0F00 | 0x00FF -> byte0 should be 0xFF) campaign emits 0x00.
This shows which residual band drives logit[0xFF] vs logit[0x00] at the
byte-0 predictor row (= the REG_AX marker row, off=0). Comparing golden
vs campaign tells us which band carries the byte-0 result in golden and
is missing in campaign.

Set C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 for the campaign build.
spec_k=0, tooling-only (model byte-identical).

PORTED to ``tools/probe_lib`` — bytecode via ``probe_lib.bc``, the byte-0
predictor row via ``probe_lib.register_marker_rows``, and the per-dim
attribution + dim->name via ``probe_lib.logit_attrib`` / ``DimNamer``. The old
``name_for`` read the STALE static registry (wrong cell post-widen); ``DimNamer``
uses the BUILT ``model.dim_positions``.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

from neural_vm.embedding import Opcode  # noqa: E402
from tools import probe_lib as P  # noqa: E402


# (program, expected byte0, wrong byte0 we observe)
PROGRAMS = {
    "or_16bit":  (P.bc([(Opcode.IMM, 0x0F00), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.OR,  Opcode.EXIT]), 0xFF, 0x00),
    "xor_16bit": (P.bc([(Opcode.IMM, 0x0F0F), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.XOR, Opcode.EXIT]), 0xF0, 0x00),
    "and_16bit": (P.bc([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0xFF, 0x00),
}


def main(selected):
    probe = P.build_probe()
    namer = P.DimNamer(probe.model)
    print(f"campaign NO_STACK0_EMIT={os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')}")

    for pname in selected:
        bc, want, got_w = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        # last REG_AX marker = byte0 predictor row (off=0).
        trace = probe.probe(bc, max_steps=20)
        ax_rows = P.register_marker_rows(trace, "REG_AX")
        pos = ax_rows[-1]  # byte-0 token sits at ax_marker+1, predicted here
        attr = P.logit_attrib(probe, bc, pos, want=want, got=got_w,
                              max_steps=20)
        print(f"\n=== {pname} got_exit={hex(got)} pos(AXmarker)={pos} ===")
        print(f"   logit[want={hex(want)}]={attr.logit_want:.2f}  "
              f"logit[wrong={hex(got_w)}]={attr.logit_got:.2f}  "
              f"diff(want-wrong)={attr.diff:.2f}")
        print("   top dims driving (logit_want - logit_wrong):")
        for di in attr.top(16):
            print(f"      dim {di:4d} {namer.name_for(di):24s} "
                  f"res={float(attr.residual[di]):8.3f} "
                  f"contrib={float(attr.contrib[di]):8.3f}")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or list(PROGRAMS.keys())
    main(sel)
