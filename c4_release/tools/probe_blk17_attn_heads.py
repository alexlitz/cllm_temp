#!/usr/bin/env python3
"""Dump block-17 (L11) attention heads that WRITE OUTPUT_LO/HI, + their Q/K/V.

block16->17 changes ONLY OUTPUT_LO+0 / OUTPUT_HI+0 (0 -> 2.0) on the SHR row,
and the L11 FFN writes only TEMP. So an L11 attention head must copy OUTPUT+0
from another position. This inspects the baked W_o of block 17 to find every head
whose output projection writes into the OUTPUT_LO/HI band, and prints the head's
Q/K attend bands (from W_q/W_k column energy) to identify the source row.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_NO_STACK0_EMIT", "1")
os.environ.setdefault("C4_OPERAND_FROM_MEMSP", "1")
os.environ.setdefault("C4_SKIP_DIM_INTEGRITY", "1")
os.environ.setdefault("C4_SKIP_GATE_CHECK", "1")
os.environ.setdefault("C4_SHIFT_OUTPUT_B0_CLEAR", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from tools.probe_groundtruth import build_groundtruth_probe  # noqa: E402


def band_label(dp, d):
    best = None
    for name, base in dp.items():
        b = int(base)
        if b <= d < b + 16 and (best is None or b > best[1]):
            best = (name, b)
    return f"{best[0]}+{d-best[1]}" if best else f"dim{d}"


def main():
    p = build_groundtruth_probe()
    model = p.model
    dp = dict(model.dim_positions)
    blk = model.blocks[17]
    attn = blk.attn
    Wq, Wk, Wv, Wo = attn.W_q, attn.W_k, attn.W_v, attn.W_o
    nh = attn.num_heads
    HD = Wq.shape[0] // nh
    d_model = Wq.shape[1] if Wq.dim() == 2 else Wq.shape[-1]
    print(f"block17 attn: num_heads={nh} HD={HD} "
          f"W_q{tuple(Wq.shape)} W_o{tuple(Wo.shape)}")
    out_lo, out_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]

    # W_o maps [nh*HD] -> [d_model].  Find heads whose slice writes OUTPUT band.
    # Shapes vary; handle W_o as [d_model, nh*HD] or [nh*HD, d_model].
    if Wo.shape[0] == d_model:
        Wo_dm_h = Wo  # [d_model, nh*HD]
    else:
        Wo_dm_h = Wo.t()  # -> [d_model, nh*HD]
    for h in range(nh):
        sl = slice(h * HD, (h + 1) * HD)
        # energy this head projects into OUTPUT_LO/HI band
        out_energy = 0.0
        writes = []
        for tgt in list(range(out_lo, out_lo + 16)) + list(range(out_hi, out_hi + 16)):
            e = float(Wo_dm_h[tgt, sl].abs().sum().item())
            if e > 0.1:
                writes.append((band_label(dp, tgt), e))
                out_energy += e
        if out_energy > 0.1:
            # Q/K read bands for this head
            if Wq.shape[0] == nh * HD:
                Wq_h = Wq[sl]        # [HD, d_model]
                Wk_h = Wk[sl]
                Wv_h = Wv[sl]
            else:
                Wq_h = Wq[:, sl].t()
                Wk_h = Wk[:, sl].t()
                Wv_h = Wv[:, sl].t()
            def top_read(W, n=6):
                col_e = W.abs().sum(dim=0)  # [d_model]
                idx = torch.argsort(col_e, descending=True)[:n]
                return [(band_label(dp, int(i)), float(col_e[int(i)])) for i in idx if col_e[int(i)] > 0.1]
            print(f"\n=== HEAD {h}: writes OUTPUT band (energy {out_energy:.2f}) ===")
            print(f"  O writes: {writes}")
            print(f"  Q reads : {top_read(Wq_h)}")
            print(f"  K reads : {top_read(Wk_h)}")
            print(f"  V reads : {top_read(Wv_h)}")


if __name__ == "__main__":
    main()
