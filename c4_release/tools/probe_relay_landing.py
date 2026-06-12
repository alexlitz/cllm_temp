"""Locate exactly where the SE-tagged operand relay landed: for physical
blocks 9..13, print each head's Q@MARK_SE / K@MARK_AX anchor, its alibi
slope, and whether V reads ALU_LO / O writes SE_ALU_LO (the head-A relay
signature) or V reads OP_LT / O writes SE_OP_LT.

Run::
    CUDA_VISIBLE_DEVICES="" python c4_release/tools/probe_relay_landing.py
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PROJ_ROOT = os.path.dirname(REPO_ROOT)
for _p in (PROJ_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from c4_release.tools.capture_residual_trace import ResidualTracer  # noqa: E402


def dense(w):
    return w.to_dense() if hasattr(w, "is_sparse_csr") and w.is_sparse_csr else w


def main() -> int:
    tracer = ResidualTracer()
    d = tracer.dim_positions
    se = d.get("MARK_SE_ONLY", d.get("MARK_SE"))
    ax = d["MARK_AX"]
    for li in range(8, 14):
        if li >= len(tracer.model.blocks):
            break
        attn = tracer.model.blocks[li].attn
        wq, wk, wv, wo = (dense(attn.W_q), dense(attn.W_k), dense(attn.W_v), dense(attn.W_o))
        HD = wq.shape[0] // attn.num_heads
        slopes = getattr(attn, "alibi_slopes", None)
        lg = getattr(tracer.model.blocks[li], "_logical_layer", "?")
        print(f"-- physical block {li} (logical L{lg}) num_heads={attn.num_heads} HD={HD} --")
        for h in range(attn.num_heads):
            r = h * HD
            qse = wq[r, se].item() if r < wq.shape[0] else 0.0
            kax = wk[r, ax].item() if r < wk.shape[0] else 0.0
            sl = None if slopes is None else round(float(slopes[h].item()), 4)
            v_alu = float(wv[r:r+HD, d["ALU_LO"]:d["ALU_LO"]+16].abs().max().item())
            o_sealu = float(wo[d["SE_ALU_LO"]:d["SE_ALU_LO"]+16, r:r+HD].abs().max().item())
            o_seop = float(wo[d["SE_OP_LT"], r:r+HD].abs().max().item()) if "SE_OP_LT" in d else 0.0
            flag = ""
            if abs(qse) >= 5 and o_sealu > 0.1:
                flag = "  <== SE_ALU relay (head A)"
            elif abs(qse) >= 5 and o_seop > 0.1:
                flag = "  <== SE_OP relay"
            if abs(qse) >= 1 or v_alu > 0.1 or o_sealu > 0.1:
                print(f"   head{h}: Q[SE]={qse:.1f} K[AX]={kax:.1f} slope={sl} "
                      f"V.read(ALU_LO)={v_alu:.2f} O.write(SE_ALU_LO)={o_sealu:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
