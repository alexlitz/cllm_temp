#!/usr/bin/env python3
"""Attribute the loop_pow2 STEP-4 AX byte-2 0xFF leak (got 0x00FF0001).

cpu_full_trace: loop_pow2_2 (id527) diverges step=4, oracle ax=1, got
ax=16711681=0x00FF0001 (byte-2 = 0xFF). Neither C4_AX_HIBYTE_CLEAR (byte-2/3
tail clamp) NOR C4_LOOP_AX_BYTE3_CAP fixes it -> the byte-2 0xFF is NOT emitted
through the OUTPUT-canonical tail path. This probe localizes the byte-2 leak
row (want 0x00, got 0xFF) via exact LM-head logit attribution over the step-4
residual, and dumps candidate discriminator/source dims so the true emitter can
be found. Shallow (max_steps=6) so it is fast.

Run: CUDA_VISIBLE_DEVICES=0 C4_CAMPAIGN=1 python tools/probe_loop_step4_byte2.py [id] [step]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)
import warnings; warnings.filterwarnings("ignore")
import torch
from src.compiler import compile_c
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 527
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    probe = GroundTruthProbe.build()
    model = probe.model
    dp = dict(model.dim_positions)
    W = model.head.weight
    if W.is_sparse:
        W = W.to_dense()
    W = W.float()
    b = model.head.bias

    def name_for(pos):
        best = None
        for nm, base in dp.items():
            base = int(base)
            if base <= pos < base + 16 and (best is None or pos - base < best[1]):
                best = (f"{nm}+{pos-base}", pos - base)
        return best[0] if best else f"d{pos}"

    RAX = int(Token.REG_AX)
    src = generate_test_programs()[pid][0]
    bc, _ = compile_c(src)
    MAXS = step + 3
    ctx = probe._final_context(bc, max_steps=MAXS)
    pad = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    tr = probe.probe(bc, max_steps=MAXS)
    ms = [q for q in sorted(tr) if tr[q]["token"] == RAX]
    print(f"id{pid} step{step}: {len(ms)} AX markers in {MAXS}-step trace")
    if step >= len(ms):
        print("!! step unavailable"); return
    mk = ms[step]
    last_block = len(model.blocks) - 1
    res_full = model.forward(pad, stop_after_block=last_block)[0]
    for byte_idx, byte_off in ((0, 1), (1, 2), (2, 3), (3, 4)):
        pred_pos = mk + byte_off - 1
        row = res_full[pred_pos]
        if row.is_sparse:
            row = row.to_dense()
        row = row.float()
        logits = W @ row + b
        got = int(torch.argmax(logits))
        emit = ctx[mk + byte_off] if mk + byte_off < len(ctx) else None
        print(f"\n=== byte{byte_idx} predictor pos {pred_pos} -> token pos "
              f"{mk+byte_off}: argmax=0x{got:02x} "
              f"emitted={'0x%02x'%emit if emit is not None else None} "
              f"logit[got]={float(logits[got]):.1f} logit[0x00]={float(logits[0]):.1f} ===")
        if byte_idx != 2 and got == 0:
            continue
        for nm in ("IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
                   "BYTE_INDEX_3", "MARK_AX", "AX_CARRY_OVERFLOW",
                   "AX_FULL_LO", "AX_FULL_HI", "AX_CARRY_LO", "AX_CARRY_HI"):
            d = dp.get(nm)
            if d is None:
                continue
            sz = 16 if nm in ("AX_FULL_LO", "AX_FULL_HI", "AX_CARRY_LO", "AX_CARRY_HI") else 1
            vals = [round(float(row[int(d)+k]), 2) for k in range(sz)
                    if abs(float(row[int(d)+k])) > 0.5]
            print(f"    {nm:20s} dim{int(d):4d} {vals if sz>1 else round(float(row[int(d)]),3)}")
        for nm in ("OUTPUT_LO", "OUTPUT_HI"):
            d = dp.get(nm)
            if d is None:
                continue
            vals = [f"{k}:{float(row[int(d)+k]):.1f}" for k in range(16)
                    if abs(float(row[int(d)+k])) > 0.5]
            print(f"    {nm:20s} {vals}")
        if got != 0:
            dw = (W[0] - W[got]) * row
            if dw.is_sparse:
                dw = dw.to_dense()
            order = torch.argsort(dw.abs(), descending=True)
            print(f"    >>> LEAK want=0x00 got=0x{got:02x}; top dims (00 - got):")
            for di in order[:14].tolist():
                print(f"        dim {di:4d} {name_for(di):24s} res={float(row[di]):8.2f} "
                      f"dW={float(W[0,di]-W[got,di]):7.2f} contrib={float(dw[di]):8.2f}")


if __name__ == "__main__":
    main()
