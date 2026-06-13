#!/usr/bin/env python3
"""Validate the ARITHMETIC-opcode NOT-blocker discriminator on the EXACT smoke
bytecode programs the over-fire broke (add_16bit, add_carry_cascade,
mul_overflow, cmp_and_branch, jmp_forward, bz_branch) — using raw bytecode, not
C source, so the residual matches the smoke gate.

For each program, dump Σarith on EVERY STACK0-marker row at the dump block input
(41). The gate is sound iff every arith/jmp row has Σarith ~ 0.11 (darkened) and
every comparison-result row has Σarith ~ 0 (fires).

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_smoke_gate.py
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import torch
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
from neural_vm.embedding import Opcode

STACK0_MARK = 268
ARITH = ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MOD",
         "OP_AND", "OP_OR", "OP_XOR", "OP_SHL", "OP_SHR"]
CTRL = ["OP_JMP", "OP_BZ", "OP_BNZ", "OP_JSR", "OP_LEV", "OP_ENT"]
CMP = ["OP_LT", "OP_GT", "OP_LE", "OP_GE", "OP_EQ", "OP_NE"]
DUMP_BLK = 41


def mk(ops):
    out = []
    for op in ops:
        if isinstance(op, tuple):
            opc, imm = op
            out.append(opc | (imm << 8))
        else:
            out.append(op)
    return out


PROGS = {
    # ARITH (over-fire victims) -- Σarith must be HIGH (darken dump)
    "add_16bit":     mk([(Opcode.IMM, 200), Opcode.PSH, (Opcode.IMM, 100), Opcode.ADD, Opcode.EXIT]),
    "add_cascade":   mk([(Opcode.IMM, 0xFF), Opcode.PSH, (Opcode.IMM, 1), Opcode.ADD, Opcode.EXIT]),
    "mul_overflow":  mk([(Opcode.IMM, 100), Opcode.PSH, (Opcode.IMM, 5), Opcode.MUL, Opcode.EXIT]),
    "sub_16bit":     mk([(Opcode.IMM, 0x100), Opcode.PSH, (Opcode.IMM, 1), Opcode.SUB, Opcode.EXIT]),
    # CONTROL -- Σarith must be ~0 (these have no arith STACK0 result row)
    "jmp_forward":   mk([(Opcode.JMP, 2), (Opcode.IMM, 99), (Opcode.IMM, 42), Opcode.EXIT]),
    "bz_branch":     mk([(Opcode.IMM, 0), (Opcode.BZ, 3), (Opcode.IMM, 99), (Opcode.IMM, 42), Opcode.EXIT]),
    # COMPARISON (target -- dump SHOULD fire) -- Σarith must be ~0
    "cmp_and_branch": mk([(Opcode.IMM, 5), Opcode.PSH, (Opcode.IMM, 5), Opcode.EQ,
                          (Opcode.BZ, 6), (Opcode.IMM, 42), Opcode.EXIT,
                          (Opcode.IMM, 0), Opcode.EXIT]),
}
MAX_STEPS = {"add_16bit": 20, "add_cascade": 20, "mul_overflow": 20, "sub_16bit": 20,
             "jmp_forward": 15, "bz_branch": 15, "cmp_and_branch": 30}


@torch.no_grad()
def residual_at_block(probe, ctx, block_idx, position):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return (r.to_dense() if r.is_sparse else r).float().cpu()


def find_markers(ctx, prompt_len):
    rows, i, step = [], prompt_len, 0
    while i < len(ctx):
        if ctx[i] == STACK0_MARK:
            rows.append((step, i))
        if ctx[i] == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    P = {n: int(dp[n]) for n in (ARITH + CTRL + CMP) if n in dp}
    b_carr = int(dp["STACK0_B0_CARRIED"]); b_sharp = int(dp["STACK0_B0_SHARP"])
    probe = GroundTruthProbe.build()

    for tag, bc in PROGS.items():
        ctx = probe._final_context(bc, max_steps=MAX_STEPS[tag])
        pl = len(probe._build_context(bc))
        rows = find_markers(ctx, pl)
        print(f"\n### {tag}  STACK0 rows={[(s,p) for s,p in rows]}")
        for step, p in rows:
            res = residual_at_block(probe, ctx, DUMP_BLK, p)
            v = {n: round(float(res[P[n]]), 3) for n in P}
            sa = sum(v[n] for n in ARITH if n in v)
            sc = sum(v[n] for n in CTRL if n in v)
            scmp = sum(v[n] for n in CMP if n in v)
            # The GATE discriminator: arith opcodes + OP_JMP (the "non-comparison
            # value/transfer" mass). Dump fires iff this is ~0.
            block_mass = sa + v.get("OP_JMP", 0.0) + v.get("OP_JSR", 0.0) + v.get("OP_LEV", 0.0)
            carr = round(float(res[b_carr]), 1); sharp = round(float(res[b_sharp]), 1)
            # Existing gate fires iff CARRIED~1 AND SHARP~1. The new discriminator
            # only matters on those rows.
            gated_on = (carr > 50) and (sharp > 0.5)
            top = sorted(v.items(), key=lambda kv: -abs(kv[1]))[:2]
            top = {k: x for k, x in top if abs(x) > 0.005}
            tag2 = ""
            if gated_on:
                tag2 = "  <== GATED-ON (fires today)" + (
                    "  *** OVERFIRE if block_mass<thr" if block_mass < 0.05 else
                    "  -> block_mass darkens it")
            print(f"   step{step} pos{p}: CARRIED={carr} SHARP={sharp} "
                  f"block_mass={block_mass:.3f}(arith={sa:.3f},jmp={v.get('OP_JMP',0):.3f}) "
                  f"Σcmp={scmp:.3f}{tag2} {top}")


if __name__ == "__main__":
    main()
