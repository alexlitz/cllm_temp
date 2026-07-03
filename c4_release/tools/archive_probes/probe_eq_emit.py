#!/usr/bin/env python3
"""EQ emit-row probe: dump the final context tail (REG_AX + value bytes) and
trace OUTPUT_LO / argmax-logit at the value-emit rows for eq_true/eq_false.

This pinpoints WHICH row's OUTPUT_LO band drives the wrong emitted byte and
how thin the [winner]-vs-[runner-up] margin is.

spec_k=0, hook-free, CACHED build (matches pytest test_smoke.py).
"""
import os
import sys

os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from neural_vm.embedding import Opcode
from tools.probe_groundtruth import build_groundtruth_probe, Token


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
    "eq_true":  (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 5),  Opcode.EQ, Opcode.EXIT]), 1),
    "eq_false": (_mk([(Opcode.IMM, 5),  Opcode.PSH, (Opcode.IMM, 7),  Opcode.EQ, Opcode.EXIT]), 0),
    "lt_true":  (_mk([(Opcode.IMM, 10), Opcode.PSH, (Opcode.IMM, 20), Opcode.LT, Opcode.EXIT]), 1),
}


def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dp = model.dim_positions
    dev = next(model.parameters()).device
    olo = dp.get("OUTPUT_LO")
    REG_AX = int(Token.REG_AX)

    for pname in selected:
        bc, expected = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        # locate the REG_AX that the exit decode reads
        ax_pos = max(i for i in range(S) if ctx[i] == REG_AX and i + 4 < S)
        val_bytes = [ctx[ax_pos + 1 + j] & 0xFF for j in range(4)]
        print(f"\n=== {pname} exp={expected} got={got} "
              f"{'PASS' if got==expected else 'FAIL'} S={S} reg_ax@{ax_pos} "
              f"val_bytes={val_bytes} (byte0={val_bytes[0]}) ===")
        # For each value-emit row: the token at ax_pos+1..+4 was emitted from
        # logits at position ax_pos..ax_pos+3. Trace the final-block residual
        # OUTPUT_LO + the head logits' top-2 at those emit positions.
        padded = torch.tensor([ctx], dtype=torch.long, device=dev)
        nblk = len(model.blocks)
        with torch.no_grad():
            logits = model.forward(padded)                              # [1,S,V]
            resid_full = model.forward(padded, stop_after_block=nblk - 1)  # [1,S,D]
        for j in range(4):
            emit_pos = ax_pos + j          # logits[emit_pos] predicts token at emit_pos+1
            if olo is not None:
                band = [round(float(resid_full[0, emit_pos, olo + k].item()), 2) for k in range(16)]
                am = max(range(16), key=lambda k: resid_full[0, emit_pos, olo + k].item())
                m_win = sorted(band, reverse=True)[:3]
            else:
                band, am, m_win = None, None, None
            lg = logits[0, emit_pos]
            top = torch.topk(lg, 3)
            top_tok = [(int(t), round(float(v), 2)) for t, v in zip(top.indices, top.values)]
            emitted = ctx[emit_pos + 1] if emit_pos + 1 < S else None
            print(f"  byte{j} emit_pos={emit_pos} emitted_tok={emitted} "
                  f"OLO_am={am} top3band={m_win} top3logit={top_tok}")
            if band is not None and j == 0:
                print(f"        OLO[0..15]={band}")


if __name__ == "__main__":
    sel = sys.argv[1:] or list(PROGRAMS.keys())
    main(sel)
