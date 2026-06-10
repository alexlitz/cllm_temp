#!/usr/bin/env python3
"""Spec_k=0 SE_* relay measurement (uses tools.probe_groundtruth, NO hooks).

Measures whether the L9 Wave A v2 ``layer9_step_end_operand_relay`` populates
SE_ALU_LO/HI, SE_AX_CARRY_LO/HI, SE_OP_<cmp>, SE_CMP at the MARK_SE_ONLY
row of the binop step. Reads the model's own post-block residual via
``probe_groundtruth.residual_at`` (the hook-free ``stop_after_block`` kwarg).

FINDINGS (2026-06-10, spec_k=0 ground truth):
  * The relay's two heads physically land on the L10 attention (physical
    block 11 == logical L10; the relay binds to ``layer9_marker_suppress``
    so it fires one physical attn BEFORE the L10 FFN that reads SE_*). It is
    NOT at physical block 10. Set L9_BLOCK=11 (the default here).
  * On that attn the relay resolves to head slots 3/4 (via ``_l9_head_idx``),
    which COLLIDE with the L10 byte-passthrough heads. ``alu_ops.py``'s
    ``make_layer10_residual_alibi_slopes_op`` (phase 999+) overwrites
    ``alibi_slopes[3]=0.5`` / ``[4]=1.0`` AFTER the relay's phase-9.3
    ``[3]=[4]=0.2`` bake, so the relay's long-range SE->AX hop is
    distance-penalised and transmission collapses to ~20%:
        SE_ALU_LO=1.41 (vs ALU_LO=6.0), SE_OP_EQ=0.01 (vs OP_EQ=5.0),
        SE_CMP_GROUP=0.002 (vs CMP_GROUP=1.0)  [measured at baseline].
  * Pinning the relay to the FREE slots 6/7 (untouched by the L10 slope op)
    with L=14 / slope 0.15 restores ~1.0 fidelity:
        SE_ALU_LO=6.76, SE_AX_CARRY=1.0, SE_OP_EQ=4.79, SE_CMP_GROUP=0.96.
  * BUT the only consumers of these SE_* dims are the L9 CMP cascade rules
    (``_layer9_cmp_rules`` hi_eq/lo_eq/hi_lt/lo_lt), and activating them
    REGRESSES lt_true/le_true and does NOT fix eq_true: the cascade
    double-writes the still-working raw-AX LT path, and the nibble-pair
    (2-cell) operand encoding makes hi_eq/lo_eq fire on UNEQUAL operands
    (e.g. lt_true 10<20 produces CMP_eq=[_,20.3,69.25,_] at the SE row).
    The SE comparison cascade is architecturally incompatible with the
    relayed 2-cell encoding -> downstream-blocked, separate from the relay.

Usage:
    CUDA_VISIBLE_DEVICES="" L9_BLOCK=11 python tools/probe_se_relay.py [prog ...]
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from neural_vm.embedding import Opcode
from neural_vm.token_layout import TOKENS_PER_STEP
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
    "eq_true": (_mk([(Opcode.IMM, 42), Opcode.PSH, (Opcode.IMM, 42), Opcode.EQ, Opcode.EXIT]), 1, "EQ"),
    "eq_false": (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.EQ, Opcode.EXIT]), 0, "EQ"),
    "and_basic": (_mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 0x2A), Opcode.AND, Opcode.EXIT]), 0x2A, "AND"),
    "mul_basic": (_mk([(Opcode.IMM, 6), Opcode.PSH, (Opcode.IMM, 7), Opcode.MUL, Opcode.EXIT]), 42, "MUL"),
    "add_basic": (_mk([(Opcode.IMM, 40), Opcode.PSH, (Opcode.IMM, 2), Opcode.ADD, Opcode.EXIT]), 42, "ADD"),
}

# dim band widths (for max-abs over a band)
WIDTHS = {
    "SE_ALU_LO": 16, "SE_ALU_HI": 16, "SE_AX_CARRY_LO": 16, "SE_AX_CARRY_HI": 16,
    "SE_CMP": 4, "SE_OP_EQ": 1, "SE_CMP_GROUP": 1,
    "ALU_LO": 16, "ALU_HI": 16, "AX_CARRY_LO": 16, "AX_CARRY_HI": 16,
    "CMP": 4, "OP_EQ": 1, "OP_AND": 1, "OP_MUL": 1, "OP_ADD": 1, "CMP_GROUP": 1,
    "MARK_AX": 1, "MARK_SE_ONLY": 1,
}

SE_REPORT = ["SE_ALU_LO", "SE_ALU_HI", "SE_AX_CARRY_LO", "SE_AX_CARRY_HI",
             "SE_CMP", "SE_OP_EQ", "SE_CMP_GROUP"]
AX_REPORT = ["ALU_LO", "ALU_HI", "AX_CARRY_LO", "AX_CARRY_HI", "CMP",
             "OP_EQ", "OP_AND", "OP_MUL", "OP_ADD", "CMP_GROUP"]

L9_BLOCK = int(os.environ.get("L9_BLOCK","11"))


def band_max(probe, prog_bc, block, pos, dp, names):
    """Read max-abs over each named dim band at (block, pos)."""
    # Build a flat dim_names dict covering every cell of every band.
    dim_names = {}
    for nm in names:
        base = dp.get(nm)
        if base is None:
            continue
        for i in range(WIDTHS.get(nm, 1)):
            dim_names[f"{nm}+{i}"] = base + i
    vals = probe.residual_at(prog_bc, block_idx=block, position=pos,
                             dim_names=dim_names)
    out = {}
    for nm in names:
        cells = [abs(vals[f"{nm}+{i}"]) for i in range(WIDTHS.get(nm, 1))
                 if f"{nm}+{i}" in vals]
        out[nm] = max(cells) if cells else None
    return out


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    se_base = dp["MARK_SE_ONLY"]
    ax_base = dp["MARK_AX"]

    for pname in selected:
        bc, expected, binop = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]  # [S, d_model]
        se_rows = [r for r in range(S) if emb[r, se_base].abs().item() > 0.5]
        ax_rows = [r for r in range(S) if emb[r, ax_base].abs().item() > 0.5]
        # The binop step is the 4th of 5 (IMM,PSH,IMM,<binop>,EXIT). With
        # the runner halting on EXIT, the LAST completed step is the binop.
        # Its SE row is the last MARK_SE_ONLY row; its AX row is the last
        # MARK_AX row before it.
        if not se_rows:
            print(f"=== {pname}: NO MARK_SE_ONLY rows; got={got} ===")
            continue
        se_row = se_rows[-1]
        ax_row = max((r for r in ax_rows if r < se_row), default=ax_rows[-1])

        print(f"=== {pname} binop={binop} expected={expected} got={got} "
              f"S={S} se_row={se_row} ax_row={ax_row} ===")
        # Measure AX-row raw operand state and SE-row SE_* state after L9.
        ax_vals = band_max(probe, bc, L9_BLOCK, ax_row, dp, AX_REPORT)
        se_vals = band_max(probe, bc, L9_BLOCK, se_row, dp, SE_REPORT)
        # Also raw operand state AT the SE row (should be ~0 pre-relay).
        se_raw = band_max(probe, bc, L9_BLOCK, se_row, dp, AX_REPORT)
        mark = band_max(probe, bc, L9_BLOCK, se_row, dp, ["MARK_SE_ONLY", "MARK_AX"])

        print(f"  [AX row {ax_row}] raw operands (block {L9_BLOCK}):")
        print("    " + " ".join(f"{k}={v:.2f}" for k, v in ax_vals.items()
                                 if v is not None))
        print(f"  [SE row {se_row}] markers: MARK_SE_ONLY={mark['MARK_SE_ONLY']:.2f} "
              f"MARK_AX={mark['MARK_AX']:.2f}")
        print(f"  [SE row {se_row}] SE_* relay outputs (block {L9_BLOCK}):")
        print("    " + " ".join(f"{k}={v:.3f}" for k, v in se_vals.items()
                                 if v is not None))
        print(f"  [SE row {se_row}] raw operands (should be ~0 pre-relay):")
        print("    " + " ".join(f"{k}={v:.3f}" for k, v in se_raw.items()
                                 if v is not None and v > 0.01))
        print()


if __name__ == "__main__":
    sel = sys.argv[1:] or ["eq_false", "and_basic", "eq_true"]
    main(sel)
