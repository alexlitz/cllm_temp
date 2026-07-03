#!/usr/bin/env python3
"""EQ decode-margin probe (cached/trustworthy build_groundtruth_probe path).

EFFICIENT: replays the spec_k=0 emission loop ONCE per program, then does a
single truncated forward per block (NO per-block replay). Traces, for
eq_true / eq_false / lt_true / ne_true:
  - CMP[0..3] at BOTH the binop AX row and the SE row, all blocks
  - OUTPUT_LO[0..15] at BOTH rows, all blocks (argmax = decoded nibble)
  - the final emitted exit code (got)

spec_k=0, hook-free, CACHED build (matches pytest test_smoke.py).
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


PROGRAMS = {
    "eq_true":  (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 5),  Opcode.EQ, Opcode.EXIT]), 1),
    "eq_false": (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 7),  Opcode.EQ, Opcode.EXIT]), 0),
    "lt_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LT, Opcode.EXIT]), 1),
    "ne_true":  (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 7),  Opcode.NE, Opcode.EXIT]), 1),
}


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
    ax_base = dp["MARK_AX"]
    nblocks = len(model.blocks)

    def band(base_name, width, resid, pos):
        base = dp.get(base_name)
        if base is None:
            return None
        return [float(resid[0, pos, base + i].item()) for i in range(width)]

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)   # replay ONCE
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(padded)[0]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        se_row = se_rows[-1]
        ax_row = max((r for r in ax_rows if r < se_row), default=ax_rows[-1])
        print(f"\n=== {pname} expected={expected} got={got} "
              f"{'PASS' if got==expected else 'FAIL'} S={S} "
              f"ax_row={ax_row} se_row={se_row} nblocks={nblocks} ===")
        prev = None
        for blk in range(8, nblocks):
            with torch.no_grad():
                resid = model.forward(padded, stop_after_block=blk)  # one fwd/block
            ax_cmp = band("CMP", 4, resid, ax_row)
            ax_olo = band("OUTPUT_LO", 16, resid, ax_row)
            se_cmp = band("CMP", 4, resid, se_row)
            se_olo = band("OUTPUT_LO", 16, resid, se_row)
            ax_am = max(range(16), key=lambda i: ax_olo[i]) if ax_olo else None
            se_am = max(range(16), key=lambda i: se_olo[i]) if se_olo else None
            ax_m = round(ax_olo[1] - ax_olo[0], 2) if ax_olo else None
            se_m = round(se_olo[1] - se_olo[0], 2) if se_olo else None
            sig = (
                tuple(round(x, 1) for x in (ax_cmp or [])),
                tuple(round(x, 1) for x in (se_cmp or [])),
                tuple(round(x, 1) for x in (ax_olo or [])),
                tuple(round(x, 1) for x in (se_olo or [])),
            )
            if sig != prev:
                print(f"  blk{blk:2d} "
                      f"AX[cmp={hot(ax_cmp)} am={ax_am} m10={ax_m} olo={hot(ax_olo)}] "
                      f"SE[cmp={hot(se_cmp)} am={se_am} m10={se_m} olo={hot(se_olo)}]")
                prev = sig


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
