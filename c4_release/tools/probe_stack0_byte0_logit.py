#!/usr/bin/env python3
"""Attribute the STACK0 byte-0 emission logit at the carried CMP step.

At the STACK0 byte-0 predictor row (the row right AFTER the [STACK0] marker on
the carried CMP step), the model emits a next-register marker ([PC]) instead of
the byte value for both-nibbles-nonzero operands. logit[t] = head.weight[t] .
residual_lastblock + bias (no final norm). We compute per-dim contribution to
(logit[want_byte] - logit[got_marker]) on the PSH step (correct) and the CMP
step (the bug), to find the dim the STACK0 byte-0 emission reads.

Run: CUDA_VISIBLE_DEVICES=0 python tools/probe_stack0_byte0_logit.py
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
STACK0_MARK = 268
PC_MARK = int(Token.REG_PC)


def name_for(pos):
    best = None
    for nm, slot in _REG.slots.items():
        if slot.start <= pos < slot.start + slot.size:
            if best is None or slot.size < best[1]:
                best = (f"{nm}+{pos - slot.start}", slot.size)
    return best[0] if best else f"dim{pos}"


@torch.no_grad()
def residual_full(probe, ctx, block_idx, position):
    if position < 0:
        position = len(ctx) + position
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x = probe.model.forward(padded, stop_after_block=block_idx)
    r = x[0, position]
    return r.to_dense() if r.is_sparse else r


def find_stack0_byte0_rows(ctx, prompt_len):
    """Return list of (step_idx, stack0_marker_pos) for each emitted step that
    has a STACK0 marker (the byte-0 predictor row is marker_pos+1)."""
    rows = []
    i = prompt_len
    step = 0
    while i < len(ctx):
        t = ctx[i]
        if t == STACK0_MARK:
            rows.append((step, i))
        if t == int(Token.STEP_END):
            step += 1
        i += 1
    return rows


def main():
    probe = GroundTruthProbe.build()
    model = probe.model
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    b = model.head.bias
    last_block = len(model.blocks) - 1

    src = "int main() { if (17 > 35) return 1; return 0; }"  # DRIFT 0x11
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=12)
    pl = len(probe._build_context(bc))
    rows = find_stack0_byte0_rows(ctx, pl)
    print(f"src={src!r}")
    print(f"STACK0 markers at steps: {[(s, p) for s, p in rows]}")

    # PSH step (step 1, correct: emits 0x11) vs CMP step (step 2, bug: emits [PC])
    # The PREDICTOR row of the STACK0 byte-0 token is the STACK0 MARKER row
    # itself (its next-token logits == the byte-0 token). pos = marker.
    for want_step, want_byte in ((1, 0x11), (2, 0x11)):
        marker = next((p for s, p in rows if s == want_step), None)
        if marker is None:
            print(f"\n(no STACK0 marker at step {want_step})")
            continue
        pos = marker  # the STACK0 marker row PREDICTS the byte-0 token
        emitted = ctx[pos + 1] if pos + 1 < len(ctx) else None
        res = residual_full(probe, ctx, last_block, pos).to(W.device).float()
        got = PC_MARK if emitted == PC_MARK else (emitted if emitted is not None else PC_MARK)
        logit_want = float((W[want_byte] * res).sum() + b[want_byte])
        logit_got = float((W[got] * res).sum() + b[got])
        print(f"\n=== step {want_step} STACK0 byte-0 predictor pos {pos} "
              f"(emitted token={emitted}) ===")
        print(f"   logit[0x{want_byte:02x}]={logit_want:.3f}  "
              f"logit[got={got}]={logit_got:.3f}  "
              f"diff={logit_want - logit_got:.3f}")
        dw = (W[want_byte] - W[got]) * res
        dw = dw.to_dense() if dw.is_sparse else dw
        order = torch.argsort(dw.abs(), descending=True)
        print(f"   top dims driving (logit[byte] - logit[got]):")
        for di in order[:18].tolist():
            print(f"      dim {di:4d} {name_for(di):28s} "
                  f"res={float(res[di]):8.3f} "
                  f"dW={float(W[want_byte, di] - W[got, di]):7.3f} "
                  f"contrib={float(dw[di]):8.3f}")


if __name__ == "__main__":
    main()
