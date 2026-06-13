#!/usr/bin/env python3
"""Attribute the AX byte-1 emission logit to residual dims (no final norm).

logit[t] = head.weight[t] . residual_block39 + head.bias[t].  At the byte-1
predictor row, compute per-dim contribution to (logit[want] - logit[got]) on
the IMM step (want=got=correct) and PSH step (want!=got, the bug), to find the
exact dims the LM head reads to emit AX byte 1.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_ax_logit_attrib.py
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
from neural_vm.dim_registry_dynamic import build_default_registry_dynamic

_REG = build_default_registry_dynamic()


def name_for(pos):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= pos < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{pos - slot.start}", slot.size)
    return best[0] if best else f"dim{pos}"


@torch.no_grad()
def residual_full(probe, bc, block_idx, position, max_steps):
    ctx = probe._final_context(bc, max_steps=max_steps)
    if position < 0:
        position = len(ctx) + position
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)  # [1,S,D]
    return x[0, position]  # [D]


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    D = W.shape[1]

    bc, _ = compile_c("int main() { return 654 + 114; }")  # AX byte1 = 0x02
    trace = probe.probe(bc, max_steps=4)
    RAX = int(Token.REG_AX)
    ms = [p for p in sorted(trace) if trace[p]["token"] == RAX]

    last_block = len(model.blocks) - 1
    for step, label in ((0, "IMM(correct=0x02)"), (1, "PSH(wrong->0x00)")):
        m = ms[step]
        pos = m + 1  # byte-1 predictor row
        res = residual_full(probe, bc, last_block, pos, 4)  # [D]
        if res.is_sparse:
            res = res.to_dense()
        res = res.to(W.device).float()
        want, got = 0x02, 0x00
        # contribution to (logit_want - logit_got) per dim
        dw = (W[want] - W[got]) * res  # [D]
        dw = dw.to_dense() if dw.is_sparse else dw
        logit_want = float((W[want] * res).sum() + b[want])
        logit_got = float((W[got] * res).sum() + b[got])
        print(f"\n=== {label} step{step} byte-1 predictor pos {pos} ===")
        print(f"   logit[0x02]={logit_want:.3f}  logit[0x00]={logit_got:.3f}  "
              f"diff(02-00)={logit_want - logit_got:.3f}")
        order = torch.argsort(dw.abs(), descending=True)
        print("   top dims driving (logit02 - logit00):")
        for di in order[:18].tolist():
            print(f"      dim {di:4d} {name_for(di):26s} "
                  f"res={float(res[di]):8.3f} "
                  f"dW={float(W[want, di] - W[got, di]):7.3f} "
                  f"contrib={float(dw[di]):8.3f}")


if __name__ == "__main__":
    main()
