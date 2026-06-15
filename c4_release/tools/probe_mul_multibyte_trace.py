#!/usr/bin/env python3
"""Trace a multi-byte MUL through the byte-1 emit chain at BUILT dims.

For a chosen mul program (e.g. 97*94=9118=0x239E) decode the AX register
bytes the model actually emits (spec_k=0, hook-free), and dump the key
residual bands at the MUL row across the chain:
  - L11 out:  OUTPUT_LO/HI (byte0), MUL_RESULT_HI_LO/HI (byte1)
  - after L13: AX_FULL_LO/HI (staged byte1)
  - the final OUTPUT_LO/HI at the AX byte-0 / byte-1 emit token rows.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/probe_mul_multibyte_trace.py [A B ...]
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


def band(row, dp, name, width=16, thr=0.05):
    base = dp.get(name)
    if base is None:
        return None
    cells = [round(float(row[base + i].item()), 3) for i in range(width)]
    return [(i, v) for i, v in enumerate(cells) if abs(v) > thr]


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
        pairs = [(6, 7), (97, 94), (100, 5), (31, 8), (11, 11)]

    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    ax_base = dp["MARK_AX"]
    opmul_base = dp["OP_MUL"]
    isbyte = dp["IS_BYTE"]
    bi0 = dp["BYTE_INDEX_0"]
    bi1 = dp["BYTE_INDEX_1"]
    h1 = dp["H1"]

    # block map
    def phys_for_logical(lg):
        for phys, blk in enumerate(model.blocks):
            if getattr(blk, "_logical_layer", phys) == lg:
                return phys
        return None
    l11_phys = phys_for_logical(11)
    l13_phys = phys_for_logical(13)
    nblocks = len(model.blocks)
    print(f"L11_phys={l11_phys} L13_phys={l13_phys} nblocks={nblocks} d_model={model.blocks[0].attn.W_q.shape[1]}")
    has_w2 = "MUL_RESULT_HI_LO" in dp
    print(f"MUL_RESULT_HI present={has_w2}")

    for a, b in pairs:
        bc = prog(a, b)
        ctx = probe._final_context(bc, max_steps=20)
        S = len(ctx)
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        exp = a * b
        with torch.no_grad():
            r_in = model.forward(toks, stop_after_block=l11_phys - 1)[0]
        cand = [r for r in range(S)
                if r_in[r, ax_base].abs().item() > 0.5
                and r_in[r, opmul_base].item() > 0.5]
        ax_row = cand[-1] if cand else None
        # operands at L11 input
        a_lo = argmax_nib(r_in[ax_row], dp, 'ALU_LO')
        a_hi = argmax_nib(r_in[ax_row], dp, 'ALU_HI')
        b_lo = argmax_nib(r_in[ax_row], dp, 'AX_CARRY_LO')
        b_hi = argmax_nib(r_in[ax_row], dp, 'AX_CARRY_HI')
        a_got = (a_hi[0] << 4) | a_lo[0]
        b_got = (b_hi[0] << 4) | b_lo[0]
        with torch.no_grad():
            r_l11 = model.forward(toks, stop_after_block=l11_phys)[0]
            r_l13 = model.forward(toks, stop_after_block=l13_phys)[0]
            r_final = model.forward(toks)[0]

        # decode emitted exit code
        exit_code = probe._decode_exit_code(ctx)
        ok = "OK" if exit_code == exp else "**FAIL**"
        print(f"\n=== A={a} B={b} expect={exp}=0x{exp:04x} (b0=0x{exp&0xff:02x} b1=0x{(exp>>8)&0xff:02x}) "
              f"emitted={exit_code}=0x{exit_code:04x} {ok}  ax_row={ax_row} ===")
        print(f"  [L11 IN operands] A_got=0x{a_got:02x}(want 0x{a:02x}) B_got=0x{b_got:02x}(want 0x{b:02x})  "
              f"ALU_LO={a_lo} ALU_HI={a_hi} AXC_LO={b_lo} AXC_HI={b_hi}")
        print(f"  [L11 out @ax_row] OUTPUT_LO argmax={argmax_nib(r_l11[ax_row],dp,'OUTPUT_LO')} "
              f"OUTPUT_HI argmax={argmax_nib(r_l11[ax_row],dp,'OUTPUT_HI')}")
        if has_w2:
            print(f"                    MUL_RESULT_HI_LO argmax={argmax_nib(r_l11[ax_row],dp,'MUL_RESULT_HI_LO')} "
                  f"MUL_RESULT_HI_HI argmax={argmax_nib(r_l11[ax_row],dp,'MUL_RESULT_HI_HI')}")
        print(f"  [L13 out @ax_row] AX_FULL_LO argmax={argmax_nib(r_l13[ax_row],dp,'AX_FULL_LO')} "
              f"AX_FULL_HI argmax={argmax_nib(r_l13[ax_row],dp,'AX_FULL_HI')}")

        # Find the AX register byte-emit token rows. The AX register section
        # emits 4 byte tokens; row with IS_BYTE + H1[AX] + BYTE_INDEX_k.
        AX_I = 1
        for bidx, bidim in [(0, bi0), (1, bi1)]:
            rows = [r for r in range(S)
                    if r_final[r, isbyte].item() > 0.5
                    and r_final[r, h1 + AX_I].item() > 0.5
                    and r_final[r, bidim].item() > 0.5]
            for r in rows[-1:]:
                ol = argmax_nib(r_final[r], dp, 'OUTPUT_LO')
                oh = argmax_nib(r_final[r], dp, 'OUTPUT_HI')
                print(f"  [final @AX byte{bidx} row={r}] OUTPUT_LO={ol} OUTPUT_HI={oh} "
                      f"-> byte=0x{(oh[0]<<4 | ol[0]):02x}")


if __name__ == "__main__":
    main()
