#!/usr/bin/env python3
"""SE-row CMP + SE_ALU/SE_AX_CARRY operand mirror probe, spec_k=0.

For comparison programs, reads at the binop SE (MARK_SE_ONLY) row, after a
chosen block, the per-cell SE_ALU_LO/HI + SE_AX_CARRY_LO/HI (the operands
the L9 CMP cascade reads) and the resulting CMP+0..3 (hi_lt/hi_eq/lo_eq/
lo_lt). Shows whether clean operands break the lt/le cascade.
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from neural_vm.embedding import Opcode  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


PROGS = {
    "lt_true_10_20": (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                           Opcode.LT, Opcode.EXIT]), 1),
    "le_true_10_20": (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                           Opcode.LE, Opcode.EXIT]), 1),
    "gt_true_20_10": (_mk([(Opcode.IMM, 20), Opcode.PSH, (Opcode.IMM, 10),
                           Opcode.GT, Opcode.EXIT]), 1),
    "eq_false_10_20": (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20),
                            Opcode.EQ, Opcode.EXIT]), 0),
    "eq_true_42": (_mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42),
                        Opcode.EQ, Opcode.EXIT]), 1),
}

BANDS16 = ["SE_ALU_LO", "SE_ALU_HI", "SE_AX_CARRY_LO", "SE_AX_CARRY_HI"]


def cells(probe, bc, block, pos, dp, name, width):
    base = dp[name]
    dn = {f"{name}+{i}": base + i for i in range(width)}
    vals = probe.residual_at(bc, block_idx=block, position=pos, dim_names=dn)
    return [vals[f"{name}+{i}"] for i in range(width)]


def fmt(vec, thr=0.5):
    return "[" + ", ".join(f"{v:.2f}@{i}" for i, v in enumerate(vec)
                           if abs(v) > thr) + "]"


def main(block):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
    cmp_base = dp["CMP"]

    for pname, (bc, exp) in PROGS.items():
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        if not se_rows:
            print(f"=== {pname}: no SE rows got={got} ===")
            continue
        se_row = se_rows[-1]
        print(f"=== {pname} exp={exp} got={got} se_row={se_row} block={block} ===")
        for nm in BANDS16:
            vec = cells(probe, bc, block, se_row, dp, nm, 16)
            print(f"  {nm:16s} = {fmt(vec)}")
        cmp = cells(probe, bc, block, se_row, dp, "CMP", 4)
        print(f"  CMP[hi_lt,hi_eq,lo_eq,lo_lt] = "
              f"[{cmp[0]:.2f}, {cmp[1]:.2f}, {cmp[2]:.2f}, {cmp[3]:.2f}]")
        print()


if __name__ == "__main__":
    # default block 11 = where the SE relay + L9 CMP land (per probe_se_relay)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
