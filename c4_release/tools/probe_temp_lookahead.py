#!/usr/bin/env python3
"""Probe: does TEMP (=PC+1) point at op2, and is op2's opcode fetchable from it?

The C4 ISA is single-slot: each instruction is one word ``op + (imm<<8)`` at
instruction index = slot. So PC+1 = the NEXT instruction's slot = op2's address.
The L4 PC-relay already computes PC+1 into the ``TEMP`` register (used by the L5
immediate-fetch head). If TEMP holds op2's address, a lookahead opcode-fetch
head (mirror L5 head 1, but Q from TEMP instead of EMBED) reads op2's OPCODE
BYTE from program memory -- a causally-available consumer-opcode signal.

This probe, for the canonical expr/if programs, dumps at an early block the
TEMP nibble value at the AX-marker row of each step and compares it to the
expected (current instruction index + 1). It also dumps OPCODE_BYTE_LO/HI and
the ADDR_KEY of code rows so we can confirm the address representation.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_temp_lookahead.py
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
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic

AX_MARK = 258
PC_MARK = 257


@torch.no_grad()
def full_residual_block(probe, ctx, block_idx):
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    return x[0].float().cpu()


def step_of(ctx, position):
    step = 0
    for i in range(position):
        if ctx[i] == int(Token.STEP_END):
            step += 1
    return step


def nibbles_to_int(res, base, n=16):
    """A band of n one-hot nibble slots -> integer (argmax per nibble group).
    The address is stored as nibbles; here we just read the argmax of each of
    the LO/HI/TOP 16-slot groups as a single nibble each."""
    vals = [float(res[base + k]) for k in range(n)]
    if max(vals) < 0.3:
        return None
    return int(max(range(n), key=lambda k: vals[k]))


def main():
    _m, layout = compile_full_vm_dynamic(alu_mode="efficient", strict=False)
    dp = layout.dim_positions
    have = lambda n: n in dp
    TEMP = int(dp["TEMP"]) if have("TEMP") else None
    EMBED_LO = int(dp["EMBED_LO"]) if have("EMBED_LO") else None
    EMBED_HI = int(dp["EMBED_HI"]) if have("EMBED_HI") else None
    print("dims: TEMP", TEMP, "EMBED_LO", EMBED_LO, "EMBED_HI", EMBED_HI)
    probe = GroundTruthProbe.build()

    progs = {
        "expr_mul_div(14*56/8)": "int main() { return 14 * 56 / 8; }",
        "if_gt(28>9)":           "int main() { if (28 > 9) return 1; return 0; }",
    }
    # TEMP = PC+1 is built by L4 (block ~4); read AFTER it, before fetch consumes.
    for BLK in (5, 6, 8):
        print(f"\n###### BLOCK {BLK} ######")
        for tag, src in progs.items():
            bc = compile_c(src)[0]
            ctx = probe._final_context(bc, max_steps=20)
            full = full_residual_block(probe, ctx, BLK)
            print(f"\n== {tag} ==  (bytecode opcodes:",
                  [w & 0xFF for w in bc], ")")
            # per-step AX + PC marker rows
            for i in range(len(ctx)):
                if ctx[i] in (AX_MARK, PC_MARK):
                    s = step_of(ctx, i)
                    res = full[i]
                    temp_lo = nibbles_to_int(res, TEMP) if TEMP else None
                    temp_hi = nibbles_to_int(res, TEMP + 16) if TEMP else None
                    emb_lo = nibbles_to_int(res, EMBED_LO) if EMBED_LO else None
                    tname = "AX" if ctx[i] == AX_MARK else "PC"
                    print(f"  step={s} [{tname}] pos={i}: "
                          f"TEMP_lo={temp_lo} TEMP_hi={temp_hi} "
                          f"EMBED_lo={emb_lo}")


if __name__ == "__main__":
    main()
