#!/usr/bin/env python3
"""Localize the FINAL-dump AX byte-3 leak on the loop_* / edge_loop clusters.

Task: loop_mul/loop_countdown/loop_pow2 leak AX byte-3 (0x82) at the final
LEA;LOAD;HALT dump (PC correct throughout — value leak, not framing). This
probe builds the FULL neural context ONCE (spec_k=0 replay), does ONE forward,
reads the LAST AX marker's four byte-predictor rows and attributes the leak
(want=0x00) via exact LM-head logit contribution so a value-invariant cap can
be built. Uses SHORT programs (edge_loop_never) to keep the replay cheap.

Run: CUDA_VISIBLE_DEVICES=1 C4_CAMPAIGN=1 python tools/probe_loop_axbyte3_final.py [id...]
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
    ids = [int(a) for a in sys.argv[1:]] or [1029, 1030]
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
    last_block = len(model.blocks) - 1
    for pid in ids:
        src = generate_test_programs()[pid][0]
        bc, _ = compile_c(src)
        # oracle-budgeted replay (no manual max_steps -> uses expected steps)
        ctx = probe._final_context(bc)
        pad = torch.tensor([ctx], dtype=torch.long, device=probe._device)
        # find the LAST REG_AX marker directly in the emitted context
        ax_positions = [i for i, t in enumerate(ctx) if t == RAX]
        if not ax_positions:
            print(f"id{pid}: NO AX markers in {len(ctx)}-token ctx"); continue
        mk = ax_positions[-1]
        res_full = model.forward(pad, stop_after_block=last_block)[0]
        emitted = [ctx[mk + k] if mk + k < len(ctx) else None for k in range(1, 5)]
        print(f"\n############ id{pid} FINAL AX dump marker pos {mk} "
              f"(ctx {len(ctx)} tok); emitted bytes 0..3 = "
              f"{[hex(e) if e is not None else None for e in emitted]} ############")
        for byte_idx, byte_off in ((0, 1), (1, 2), (2, 3), (3, 4)):
            pred_pos = mk + byte_off - 1  # residual at pred_pos predicts token mk+byte_off
            if pred_pos < 0 or pred_pos >= res_full.shape[0]:
                continue
            row = res_full[pred_pos]
            if row.is_sparse:
                row = row.to_dense()
            row = row.float()
            logits = W @ row + b
            got = int(torch.argmax(logits))
            want = 0x00
            emit = ctx[mk + byte_off] if mk + byte_off < len(ctx) else None
            print(f"  byte{byte_idx} predictor pos {pred_pos} -> emits token pos "
                  f"{mk+byte_off}: argmax=0x{got:02x} emitted=0x{emit:02x} "
                  f"logit[got]={float(logits[got]):.1f} logit[0x00]={float(logits[0]):.1f}"
                  if emit is not None else
                  f"  byte{byte_idx} predictor pos {pred_pos}: argmax=0x{got:02x}")
            if byte_idx < 2:  # only deep-attribute the high bytes
                continue
            print("    discriminators:")
            for nm in ("IS_BYTE", "BYTE_INDEX_0", "BYTE_INDEX_1", "BYTE_INDEX_2",
                       "BYTE_INDEX_3", "MARK_AX", "AX_CARRY_OVERFLOW"):
                d = dp.get(nm)
                if d is None:
                    print(f"      {nm:20s} (no dim)"); continue
                print(f"      {nm:20s} dim{int(d):4d} = {float(row[int(d)]):.3f}")
            for nm in ("OUTPUT_LO", "OUTPUT_HI"):
                d = dp.get(nm)
                if d is None:
                    continue
                vals = [f"{k}:{float(row[int(d)+k]):.1f}" for k in range(16)
                        if abs(float(row[int(d)+k])) > 0.5]
                print(f"      {nm:20s} {vals}")
            if got != want:
                dw = (W[want] - W[got]) * row
                order = torch.argsort(dw.abs(), descending=True)
                print(f"    >>> LEAK want=0x00 got=0x{got:02x}; top dims (00 - got):")
                for di in order[:10].tolist():
                    print(f"        dim {di:4d} {name_for(di):24s} res={float(row[di]):8.2f} "
                          f"dW={float(W[want,di]-W[got,di]):7.2f} contrib={float(dw[di]):8.2f}")


if __name__ == "__main__":
    main()
