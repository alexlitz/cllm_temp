#!/usr/bin/env python3
"""Operand-A survival probe for MUL rows at the L9 ALU-clear (block 15).

Traces operand-A (ALU_LO/HI) at the MUL AX marker row across blocks 14/15/16
for the value-dependent MUL cases the C4_ALU_CLEAR_SPARE_OPERAND fix targets
(idx104 = mul 23*65, plus 7*x). spec_k=0, campaign config.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
import torch  # noqa: E402
from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            oc, imm = op
            bc.append(oc | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGS = {
    "mul_23x65": _mk([(Opcode.IMM, 23), Opcode.PSH, (Opcode.IMM, 65),
                      Opcode.MUL, Opcode.EXIT]),
    "mul_7x9":   _mk([(Opcode.IMM, 7),  Opcode.PSH, (Opcode.IMM, 9),
                      Opcode.MUL, Opcode.EXIT]),
}


def fmt(row, base, width=16, thr=0.5):
    vals = [float(row[base + i].item()) for i in range(width)]
    return "[" + ", ".join(f"{v:.1f}@{i}" for i, v in enumerate(vals)
                           if abs(v) > thr) + "]"


def main(blocks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    for pname, bc in PROGS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        ax_row = ax_rows[-1]
        print(f"=== {pname} got={got} ax_row={ax_row} ===", flush=True)
        for b in blocks:
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=b)[0]
            row = resid[ax_row]
            print(f"  blk{b:2d} ALU_LO={fmt(row, dp['ALU_LO'])} "
                  f"ALU_HI={fmt(row, dp['ALU_HI'])}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    blks = [int(x) for x in sys.argv[1:]] or [14, 15, 16]
    main(blks)
