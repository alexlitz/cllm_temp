#!/usr/bin/env python3
"""HYPOTHESIS TEST (hook, no model edit): does restoring a clean positive
ALU_LO/HI one-hot at the divmod AX row (recovered from SE_ALU) prevent the
downstream L18/L20/L25 OUTPUT corruption?

We register a forward-hook on the divmod block (the one whose post-op / FFN
runs the FlattenedDivMod) that, on the divmod AX row, overwrites ALU_LO/HI
with the SE_ALU one-hot. Then we decode the exit code and compare.
"""
import os, sys
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            o, i = op; bc.append(o | (i << 8))
        else:
            bc.append(op)
    return bc


def main():
    args = sys.argv[1:] or ["1162/37", "462/13", "2009/43", "106/4", "1132/33", "843/31", "176/4"]
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    nblocks = len(model.blocks)
    ALU_LO = dp["ALU_LO"]; ALU_HI = dp["ALU_HI"]
    SE_ALU_LO = dp["SE_ALU_LO"]; SE_ALU_HI = dp["SE_ALU_HI"]
    AXC_LO = dp["AX_CARRY_LO"]; AXC_HI = dp["AX_CARRY_HI"]
    OP_DIV = dp["OP_DIV"]; OP_MOD = dp["OP_MOD"]; MARK_AX = dp["MARK_AX"]

    MODE = os.environ.get("MODE", "alu")  # alu | clearaxc | clearboth
    # Hook EVERY block from 28 on, applying the mutation at the divmod AX row
    # so the mutated state persists to all downstream corruptors regardless of
    # row shift (we recompute the mask each block).
    DIVMOD_BLOCK = 28

    def clean_onehot(band):
        # band: [..., 16]; keep argmax as 1.0, rest 0 (only if max > 0.5)
        m = band.max(dim=-1, keepdim=True).values
        oh = (band >= m) & (band > 0.5)
        return oh.to(band.dtype)

    state = {"on": False}

    def hook(module, inp, out):
        if not state["on"]:
            return out
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out
        # x: [B, S, D]
        divmod_ax = (
            ((x[:, :, OP_DIV] > 0.5) | (x[:, :, OP_MOD] > 0.5))
            & (x[:, :, MARK_AX] > 0.5)
        )  # [B,S]
        se_lo = clean_onehot(x[:, :, SE_ALU_LO:SE_ALU_LO + 16])
        se_hi = clean_onehot(x[:, :, SE_ALU_HI:SE_ALU_HI + 16])
        mask = divmod_ax[:, :, None].to(x.dtype)
        if MODE in ("alu", "clearboth"):
            x[:, :, ALU_LO:ALU_LO + 16] = (
                x[:, :, ALU_LO:ALU_LO + 16] * (1 - mask) + se_lo * mask
            )
            x[:, :, ALU_HI:ALU_HI + 16] = (
                x[:, :, ALU_HI:ALU_HI + 16] * (1 - mask) + se_hi * mask
            )
        if MODE in ("clearaxc", "clearboth"):
            x[:, :, AXC_LO:AXC_LO + 16] = x[:, :, AXC_LO:AXC_LO + 16] * (1 - mask)
            x[:, :, AXC_HI:AXC_HI + 16] = x[:, :, AXC_HI:AXC_HI + 16] * (1 - mask)
        if isinstance(out, tuple):
            return (x,) + out[1:]
        return x

    # Hook ALL blocks from DIVMOD_BLOCK on so the mutation persists downstream.
    hs = [model.blocks[b].register_forward_hook(hook) for b in range(DIVMOD_BLOCK, nblocks)]

    print(f"# MODE={MODE}: hook blocks {DIVMOD_BLOCK}..{nblocks-1} at divmod AX row")
    for arg in args:
        a, b = arg.split("/"); a = int(a); b = int(b)
        PROG = _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.DIV, Opcode.EXIT])
        state["on"] = False
        _, base = probe.emitted_result(PROG, max_steps=6)
        state["on"] = True
        _, fixed = probe.emitted_result(PROG, max_steps=6)
        ok_base = "OK" if base == a // b else "  "
        ok_fix = "OK" if fixed == a // b else "  "
        flip = "  <== FIXED" if fixed == a // b and base != a // b else (
            "  <== BROKE" if base == a // b and fixed != a // b else "")
        print(f"  {a}/{b} q={a//b}: base={base}({ok_base}) fixed={fixed}({ok_fix}){flip}")
    for h in hs:
        h.remove()


if __name__ == "__main__":
    main()
