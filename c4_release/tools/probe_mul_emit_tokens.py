#!/usr/bin/env python3
"""Decode the emitted REG_AX byte tokens for mul programs (spec_k=0).

Shows the 4 emitted AX bytes vs expected, then for the byte-1 emit token
position, dumps OUTPUT_LO/HI + AX_FULL + the top LM-head logits so we can
see WHY byte 1 is wrong (the byte-1 emit corruption).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_emit_tokens.py [A B ...]
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
from neural_vm.batched_pure_neural import Token
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


def prog(a, b):
    return _mk([(Opcode.IMM, a), Opcode.PSH, (Opcode.IMM, b), Opcode.MUL, Opcode.EXIT])


def argmax_nib(row, dp, name, width=16):
    base = dp.get(name)
    if base is None:
        return None
    cells = [float(row[base + i].item()) for i in range(width)]
    mx = max(range(width), key=lambda i: cells[i])
    return mx, round(cells[mx], 3)


def main():
    pairs = []
    args = sys.argv[1:]
    for i in range(0, len(args) - 1, 2):
        pairs.append((int(args[i]), int(args[i + 1])))
    if not pairs:
        pairs = [(6, 7), (97, 94), (100, 5), (52, 86), (89, 26)]

    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device

    for a, b in pairs:
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        exp = a * b
        # Find the LAST REG_AX token and the 4 byte tokens after it.
        ax_pos = None
        for i in range(len(ctx) - 1, -1, -1):
            if ctx[i] == int(Token.REG_AX):
                ax_pos = i
                break
        emitted_bytes = [ctx[ax_pos + 1 + j] & 0xFF for j in range(4)] if ax_pos is not None else []
        emitted = 0
        for j, bb in enumerate(emitted_bytes):
            emitted |= bb << (8 * j)
        ok = "OK" if emitted == exp else "**FAIL**"
        exp_bytes = [(exp >> (8 * j)) & 0xFF for j in range(4)]
        print(f"\n=== A={a} B={b} expect={exp}=0x{exp:08x} emitted=0x{emitted:08x} {ok} ===")
        print(f"   AX_pos={ax_pos}  emitted bytes={[hex(x) for x in emitted_bytes]} "
              f"expected={[hex(x) for x in exp_bytes]}")

        # Residual at byte-1 emit token row = ax_pos+2 (byte0=ax_pos+1, byte1=ax_pos+2).
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        with torch.no_grad():
            r_final = model.forward(toks)[0]
        for bidx in range(2):
            row = ax_pos + 1 + bidx
            ol = argmax_nib(r_final[row], dp, 'OUTPUT_LO')
            oh = argmax_nib(r_final[row], dp, 'OUTPUT_HI')
            byte_from_band = (oh[0] << 4) | ol[0]
            print(f"   [byte{bidx} emit row={row}] OUTPUT_LO={ol} OUTPUT_HI={oh} "
                  f"band->0x{byte_from_band:02x}  emitted=0x{emitted_bytes[bidx]:02x}")


if __name__ == "__main__":
    main()
