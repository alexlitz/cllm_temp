#!/usr/bin/env python3
"""Autoregressive (spec_k=0) probe of the ADD byte-1 carry discriminator.

THE crux the prior ADD agent missed: the byte-1 add result is predicted at
the byte-0 *token*'s position in the GROWING autoregressive context, NOT a
fixed teacher-forced row. This probe runs the REAL spec_k=0 loop (mirror of
BatchedPureNeuralRunner), rebuilds the final context, locates the rows that
PREDICT each REG_AX result byte by argmaxing the LM head at every position,
and dumps -- at the byte-1 predictor row -- the live carry band, OUTPUT, the
relayed operand-A high byte (STACK0_BYTE_VAL_1), operand-B high byte
(ADDR_B1), and TEMP/BYTE_INDEX selectors.

Finding (spec_k=0): at the byte-1 predictor row (the BYTE_INDEX_0 row,
TEMP+8=1) CARRY+1 = 2.0 when byte 0 carries and 0.0 when it does not -- a
CLEAN discriminator on the REAL autoregressive runner, unlike the
teacher-forced residual (CARRY+1 there matched only at a fixed row). a1 is
relayed into STACK0_BYTE_VAL_1 (L13 head 5), b1 sits in ADDR_B1_LO.

Usage:
    python tools/probe_add_autoregressive_carry.py [name ...] [b=BLK ...]
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


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            bc.append(op[0] | (op[1] << 8))
        else:
            bc.append(op)
    return bc


def _prog(a, b, op):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), op, Opcode.EXIT])


PROGRAMS = {
    "add_0": (_prog(654, 114, Opcode.ADD), 768),       # a1=2 b1=0 c=1 -> 3
    "add_basic": (_prog(10, 32, Opcode.ADD), 42),      # a1=0 b1=0 c=0 -> 0
    "add_16bit": (_prog(200, 100, Opcode.ADD), 300),   # a1=0 b1=0 c=1 -> 1
    "add_carry": (_prog(0xFF, 1, Opcode.ADD), 0x100),  # a1=0 b1=0 c=1 -> 1
    "add_8": (_prog(432, 32, Opcode.ADD), 464),        # a1=1 b1=0 c=0 -> 1
    "add_2": (_prog(281, 250, Opcode.ADD), 531),       # a1=1 b1=0 c=1 -> 2
    "add_18": (_prog(828, 890, Opcode.ADD), 1718),     # a1=3 b1=3 c=0 -> 6
}


def band(row, dp, name, width=16):
    base = dp.get(name)
    return [float(row[base + i].item()) for i in range(width)] if base is not None else None


def nibble(cells):
    if cells is None:
        return None
    mx = max(range(len(cells)), key=lambda i: cells[i])
    return mx if cells[mx] > 0.3 else None


def db(lo, hi):
    nl, nh = nibble(lo), nibble(hi)
    return None if nl is None and nh is None else ((nh or 0) << 4) | (nl or 0)


def sc(row, dp, name):
    base = dp.get(name.split("+")[0])
    if base is None:
        return None
    off = int(name.split("+")[1]) if "+" in name else 0
    return round(float(row[base + off].item()), 2)


def main(selected, blks):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    blmap = probe.block_layer_map()

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits = model.forward(toks)[0]
        ax_positions = [i for i in range(S) if ctx[i] == Token.REG_AX]
        ax_idx = max(ax_positions) if ax_positions else None
        print(f"=== {pname} exp={expected:#x} S={S} REG_AX@{ax_idx} ===")
        if ax_idx is not None:
            for k in range(4):
                pred_row = ax_idx + k
                emitted = ctx[ax_idx + 1 + k] if ax_idx + 1 + k < S else None
                argmax_tok = int(logits[pred_row].argmax().item())
                print(f"  result byte{k}: predicted@row{pred_row} "
                      f"emitted_token={emitted} argmax={argmax_tok}")
        for k in (0, 1):
            pred_row = ax_idx + k
            print(f" --- result byte{k} predictor row {pred_row} ---")
            for blk in blks:
                lg = blmap[blk]["logical"]
                with torch.no_grad():
                    resid = model.forward(toks, stop_after_block=blk)[0]
                row = resid[pred_row]
                bi = [sc(row, dp, f"BYTE_INDEX_{i}") for i in range(4)]
                out = db(band(row, dp, "OUTPUT_LO"),
                         band(row, dp, "OUTPUT_HI_THIS_STEP")
                         or band(row, dp, "OUTPUT_HI"))
                sv1 = db(band(row, dp, "STACK0_BYTE_VAL_1_LO"),
                         band(row, dp, "STACK0_BYTE_VAL_1_HI"))
                b1 = db(band(row, dp, "ADDR_B1_LO"),
                        band(row, dp, "ADDR_B1_HI"))
                carry = [sc(row, dp, f"CARRY+{i}") for i in range(4)]
                print(f"   b{blk}(L{lg}) BI={bi} t8={sc(row,dp,'TEMP+8')} "
                      f"t9={sc(row,dp,'TEMP+9')} OUT={out} a1(SV1)={sv1} "
                      f"b1(ADDR_B1)={b1} C={carry} "
                      f"OP_ADD={sc(row,dp,'OP_ADD')} IS_BYTE={sc(row,dp,'IS_BYTE')} "
                      f"H1+1={sc(row,dp,'H1+1')}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    blks = [int(a[2:]) for a in args if a.startswith("b=")] or [12, 14, 15, 16, 17]
    sel = [a for a in args if "=" not in a] or ["add_0", "add_basic", "add_16bit"]
    main(sel, blks)
