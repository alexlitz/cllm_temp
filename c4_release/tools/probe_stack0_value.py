#!/usr/bin/env python3
"""Check what byte value the STACK0_BYTE0 carrier holds at the AND step,
for a sweep of pushed operand-A values. Determines whether the
context-builder is placing the CORRECT operand at STACK0 or a stale value.

Usage:
    CUDA_VISIBLE_DEVICES=1 python tools/probe_stack0_value.py
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


VALUES = [0xFF, 0xF0, 0x0F, 0x2A, 0x08, 0x80, 0x88, 0x7E, 0x12, 0xAB]


def main():
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    se_base = dp["MARK_SE_ONLY"]
    stack0_marker = dp.get("MARK_STACK0")
    s0byte0 = dp.get("STACK0_BYTE0")

    for a in VALUES:
        bc = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT])
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        se_row = se_rows[-1] if se_rows else None
        ax_row = max((r for r in ax_rows if (se_row is None or r < se_row)),
                     default=(ax_rows[-1] if ax_rows else None))
        # the AND step's STACK0 group: MARK_STACK0 just before ax_row
        with torch.no_grad():
            r7 = model.forward(toks, stop_after_block=7)[0]
        stack0_rows = [r for r in range(S) if stack0_marker is not None and r7[r, stack0_marker].abs().item() > 0.5 and r < ax_row]
        last_stack0 = stack0_rows[-1] if stack0_rows else None
        # the STACK0_BYTE0 carrier is typically MARK_STACK0+1
        carrier = None
        if last_stack0 is not None:
            for off in (1, 2):
                if last_stack0 + off < S and s0byte0 is not None and abs(r7[last_stack0 + off, s0byte0].item()) > 0.3:
                    carrier = last_stack0 + off
                    break
        carrier_tok = ctx[carrier] if carrier is not None else None
        print(f"A={hex(a):>5} ax_row={ax_row} last_MARK_STACK0={last_stack0} "
              f"carrier_row={carrier} carrier_token={carrier_tok}"
              f"{'='+hex(carrier_tok) if carrier_tok is not None else ''} "
              f"(operand should be {hex(a)})")


if __name__ == "__main__":
    main()
