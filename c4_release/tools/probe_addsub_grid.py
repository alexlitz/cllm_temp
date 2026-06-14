#!/usr/bin/env python3
"""ADD/SUB value-grid pass count + operand-magnitude probe (spec_k=0).

ADD grid:  IMM a; PSH; IMM b; ADD; EXIT  for indices 0..49 (a,b derived).
SUB grid:  IMM a; PSH; IMM b; SUB; EXIT  for indices 50..99.

Counts how many emit the correct exit code (the 43/50 + 29/50 gate). Also
dumps operand band magnitudes at the L8 ADD/SUB MARK_AX compute row and the
OUTPUT_LO band after each block for one representative program, so the
declarative wrap can be magnitude-matched (cf. the bitwise/MUL wraps).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/probe_addsub_grid.py            # grid counts
    CUDA_VISIBLE_DEVICES=0 python tools/probe_addsub_grid.py mag        # + magnitude dump
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch

from neural_vm.embedding import Opcode
from neural_vm.batched_pure_neural import Token
from tools.probe_groundtruth import build_groundtruth_probe
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

_PROGS = generate_test_programs()


def _add_prog(a, b):
    from src.compiler import compile_c as _cc
    return _cc(f"int main() {{ return {a} + {b}; }}")[0]


def _sub_prog(a, b):
    from src.compiler import compile_c as _cc
    return _cc(f"int main() {{ return {a} - {b}; }}")[0]


def run_grid(probe, verbose=False):
    add_pass = 0
    add_fail = []
    for i in range(0, 50):
        src, want, *_ = _PROGS[i]
        bc = compile_c(src)[0]
        _, got = probe.emitted_result(bc, max_steps=20)
        if got == want:
            add_pass += 1
        else:
            add_fail.append((i, want, got))
    sub_pass = 0
    sub_fail = []
    for i in range(50, 100):
        src, want, *_ = _PROGS[i]
        bc = compile_c(src)[0]
        _, got = probe.emitted_result(bc, max_steps=20)
        if got == want:
            sub_pass += 1
        else:
            sub_fail.append((i, want, got))
    print(f"ADD grid (ids 0..49):   {add_pass}/50 correct")
    print(f"SUB grid (ids 50..99):  {sub_pass}/50 correct")
    if verbose:
        print("  ADD fails:", [(i, hex(w), hex(g)) for i, w, g in add_fail])
        print("  SUB fails:", [(i, hex(w), hex(g)) for i, w, g in sub_fail])
    return add_pass, sub_pass


def band(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    return [float(row[base + i].item()) for i in range(width)]


def hot(cells, thr=0.3):
    if cells is None:
        return []
    return [(i, round(v, 2)) for i, v in enumerate(cells) if abs(v) > thr]


def dump_magnitudes(probe, a=10, b=32, blocks=(9, 10, 11, 12, 13, 14), tp=None):
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()
    cases = (
        ("ADD", _add_prog(a, b), a + b),
        ("SUB", _sub_prog(a + 50, b), a + 50 - b),
    )
    for opname, prog, want in cases:
        ctx = probe._final_context(prog, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            emb = model.embed(toks)[0]
        ax_rows = [r for r in range(S) if emb[r, dp["MARK_AX"]].abs().item() > 0.5]
        # The OP compute row is the LAST MARK_AX row (the opcode token).
        op_row = ax_rows[-1]
        print(f"\n=== {opname} a/b -> want={want} got={got} "
              f"op_compute_row={op_row} CARRY base={dp['CARRY']} ===")
        for blk in blocks:
            lg = blmap[blk]["logical"]
            ffn = blmap[blk]["ffn"]
            with torch.no_grad():
                resid = model.forward(toks, stop_after_block=blk)[0]
            row = resid[op_row]
            alo = band(row, dp, "ALU_LO")
            ahi = band(row, dp, "ALU_HI")
            bxlo = band(row, dp, "AX_CARRY_LO")
            bxhi = band(row, dp, "AX_CARRY_HI")
            olo = band(row, dp, "OUTPUT_LO")
            ohi = band(row, dp, "OUTPUT_HI")
            carry = [round(float(row[dp["CARRY"] + k].item()), 2) for k in range(4)]
            print(f" blk{blk:2d} L{lg:<2} {ffn[:24]:<24} "
                  f"A_LO={hot(alo)} A_HI={hot(ahi)}")
            print(f"        B_LO={hot(bxlo)} B_HI={hot(bxhi)}")
            print(f"        OUT_LO={hot(olo)} OUT_HI={hot(ohi)} CARRY={carry}")


def main():
    probe = build_groundtruth_probe()
    if "mag" in sys.argv[1:]:
        dump_magnitudes(probe)
    else:
        run_grid(probe, verbose="-v" in sys.argv[1:])


if __name__ == "__main__":
    main()
