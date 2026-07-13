#!/usr/bin/env python3
"""For each block-17 OUTPUT-writing head, print its Q weight on OP_SHR / OP_SHL
and the full Q read energy, to see how OP_SHR routes the emission-head softmax
(the sole block-16 diff between the leaking SHR row and the clean SHL row)."""
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
    attn = model.blocks[17].attn
    Wq = attn.W_q
    nh = attn.num_heads
    HD = Wq.shape[0] // nh
    Wo = model.blocks[17].attn.W_o
    d_model = Wo.shape[0]
    out_lo, out_hi = dp["OUTPUT_LO"], dp["OUTPUT_HI"]
    Wo_dm_h = Wo if Wo.shape[0] == d_model else Wo.t()

    for h in range(nh):
        sl = slice(h * HD, (h + 1) * HD)
        out_e = sum(float(Wo_dm_h[t, sl].abs().sum()) for t in
                    list(range(out_lo, out_lo + 16)) + list(range(out_hi, out_hi + 16)))
        if out_e < 0.1:
            continue
        Wq_h = Wq[sl] if Wq.shape[0] == nh * HD else Wq[:, sl].t()  # [HD, d_model]
        def qcol(nm, off=0):
            if nm not in dp:
                return None
            return float(Wq_h[:, dp[nm] + off].abs().sum())
        print(f"HEAD {h} (OUTPUT energy {out_e:.1f}):")
        for nm in ["OP_SHR", "OP_SHL", "OP_PSH", "OP_LI", "OP_SI", "OP_MUL",
                   "CONST", "MARK_AX", "IS_BYTE", "PSH_AT_SP", "MARK_SP"]:
            v = qcol(nm)
            if v is not None and v > 0.1:
                print(f"    Q[{nm}] = {v:.1f}")


if __name__ == "__main__":
    main()
