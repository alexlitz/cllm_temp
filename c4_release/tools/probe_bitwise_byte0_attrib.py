#!/usr/bin/env python3
"""LM-head logit attribution for the 16-bit bitwise byte-0 result token,
at its PREDICTOR row, in the CAMPAIGN config.

For or_16bit (0x0F00 | 0x00FF -> byte0 should be 0xFF) campaign emits 0x00.
This shows which residual band drives logit[0xFF] vs logit[0x00] at the
byte-0 predictor row (= the REG_AX marker row, off=0). Comparing golden
vs campaign tells us which band carries the byte-0 result in golden and
is missing in campaign.

Set C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 for the campaign build.
spec_k=0, tooling-only (model byte-identical).
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
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic
from tools.probe_groundtruth import build_groundtruth_probe

_REG = build_default_registry_dynamic()


def name_for(pos):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= pos < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{pos - slot.start}", slot.size)
    return best[0] if best else f"dim{pos}"


def _mk(ops):
    bc = []
    for op in ops:
        if isinstance(op, tuple):
            opcode, imm = op
            bc.append(opcode | (imm << 8))
        else:
            bc.append(op)
    return bc


# (program, expected byte0, wrong byte0 we observe)
PROGRAMS = {
    "or_16bit":  (_mk([(Opcode.IMM, 0x0F00), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.OR,  Opcode.EXIT]), 0xFF, 0x00),
    "xor_16bit": (_mk([(Opcode.IMM, 0x0F0F), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.XOR, Opcode.EXIT]), 0xF0, 0x00),
    "and_16bit": (_mk([(Opcode.IMM, 0x0FFF), Opcode.PSH, (Opcode.IMM, 0x00FF), Opcode.AND, Opcode.EXIT]), 0xFF, 0x00),
}

RAX = 258  # REG_AX marker token


@torch.no_grad()
def main(selected):
    probe = build_groundtruth_probe()
    model = probe.model
    dev = next(model.parameters()).device
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    W = W.float()
    b = model.head.bias
    last_block = len(model.blocks) - 1
    print(f"campaign NO_STACK0_EMIT={os.environ.get('C4_NO_STACK0_EMIT')} "
          f"OPERAND_FROM_MEMSP={os.environ.get('C4_OPERAND_FROM_MEMSP')}")

    for pname in selected:
        bc, want, got_w = PROGRAMS[pname]
        ctx = probe._final_context(bc, max_steps=20)
        got = probe._decode_exit_code(ctx)
        S = len(ctx)
        # last REG_AX marker = byte0 predictor row (off=0).
        ax_marker = max(i for i in range(S) if ctx[i] == RAX)
        pos = ax_marker  # byte-0 token sits at ax_marker+1, predicted here
        toks = torch.tensor([ctx], dtype=torch.long, device=dev)
        x = model.forward(toks, stop_after_block=last_block)[0, pos]
        if x.is_sparse:
            x = x.to_dense()
        x = x.to(W.device).float()
        logit_want = float((W[want] * x).sum() + b[want])
        logit_got = float((W[got_w] * x).sum() + b[got_w])
        argmax_tok = int((W @ x + b).argmax())
        print(f"\n=== {pname} got_exit={hex(got)} pos(AXmarker)={pos} "
              f"byte0_tok@{pos+1}={ctx[pos+1]} argmax_decode={argmax_tok} ===")
        print(f"   logit[want={hex(want)}]={logit_want:.2f}  "
              f"logit[wrong={hex(got_w)}]={logit_got:.2f}  "
              f"diff(want-wrong)={logit_want - logit_got:.2f}")
        dw = (W[want] - W[got_w]) * x
        if dw.is_sparse:
            dw = dw.to_dense()
        order = torch.argsort(dw.abs(), descending=True)
        print("   top dims driving (logit_want - logit_wrong):")
        for di in order[:16].tolist():
            print(f"      dim {di:4d} {name_for(di):24s} "
                  f"res={float(x[di]):8.3f} "
                  f"dW={float(W[want, di] - W[got_w, di]):7.3f} "
                  f"contrib={float(dw[di]):8.3f}")


if __name__ == "__main__":
    sel = [a for a in sys.argv[1:] if not a.startswith("-")] or list(PROGRAMS.keys())
    main(sel)
