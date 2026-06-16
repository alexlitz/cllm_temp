#!/usr/bin/env python3
"""Read the AX byte-1 dump-repopulate GATE dims at REGISTRY positions on the AX
byte-1 predictor row for a given step. Confirms whether OP_LEV / OP_LI separate
the LEV step from genuine carries.
Usage: python tools/_probe_dump_gate.py <id> <step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

_REG = build_default_registry_dynamic()
SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_markers(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


def reg(r, name, n=1):
    s = _REG.slots[name]
    v = r[s.start:s.start+s.size]
    return [round(float(x), 2) for x in v]


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); stp = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    ax = sm[stp].get("AX")
    pos = ax + 1  # byte-1 predictor row (= byte0 token row)
    print(f"id{pid} {desc} step={stp} AX_marker={ax} byte1_pred_row={pos}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    last = nblk - 1
    r = probe.model.forward(padded, stop_after_block=last)[0][pos]
    axc_lo = reg(r, "AX_CARRY_LO"); axc_hi = reg(r, "AX_CARRY_HI")
    axc_sum = sum(axc_lo) + sum(axc_hi)
    print(f"  @last block: OP_LEV={reg(r,'OP_LEV')} OP_LI={reg(r,'OP_LI')} "
          f"OP_LC={reg(r,'OP_LC')} OP_PSH={reg(r,'OP_PSH')} OP_ADJ={reg(r,'OP_ADJ')}")
    print(f"  AX_CARRY_sum={axc_sum:.2f}  ADDR_B0_LO+5={reg(r,'ADDR_B0_LO')[5]} "
          f"ADDR_B1_HI+8={reg(r,'ADDR_B1_HI')[8]}")
    # AX_CARRY_OVERFLOW lives in the LAYOUT, not registry
    dp = probe.model.embed._dim_positions
    if "AX_CARRY_OVERFLOW" in dp:
        ofl = float(r[dp["AX_CARRY_OVERFLOW"]])
        print(f"  AX_CARRY_OVERFLOW={ofl:.2f}")


if __name__ == "__main__":
    main()
