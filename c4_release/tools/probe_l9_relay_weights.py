"""Why does the L9 step_end_operand_relay not transmit? Inspect the baked
weights on the L9 attention block: head 3/4 Q/K/V/O nonzero rows + the
alibi_slopes vector. Confirms whether the imperative slope write
(attn.alibi_slopes[3]=0.2) survived and whether Q@MARK_SE / K@MARK_AX are
present.

Run::

    CUDA_VISIBLE_DEVICES="" python c4_release/tools/probe_l9_relay_weights.py
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PROJ_ROOT = os.path.dirname(REPO_ROOT)
for _p in (PROJ_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import torch  # noqa: E402
from c4_release.tools.capture_residual_trace import ResidualTracer  # noqa: E402


def dense(w):
    return w.to_dense() if hasattr(w, "is_sparse_csr") and w.is_sparse_csr else w


def main() -> int:
    tracer = ResidualTracer()
    dimp = tracer.dim_positions
    se = dimp.get("MARK_SE_ONLY", dimp.get("MARK_SE"))
    ax = dimp["MARK_AX"]
    se_alu_lo = dimp["SE_ALU_LO"]
    alu_lo = dimp["ALU_LO"]
    # Find the block whose attn W_q has a strong MARK_SE_ONLY anchor on
    # some head row -> the L9 relay's Q signature.
    for li, block in enumerate(tracer.model.blocks):
        attn = block.attn
        wq = dense(attn.W_q)
        HD = wq.shape[0] // attn.num_heads
        slopes = getattr(attn, "alibi_slopes", None)
        for h in range(attn.num_heads):
            row = h * HD
            if row < wq.shape[0] and abs(wq[row, se].item()) >= 5.0:
                wk = dense(attn.W_k)
                wv = dense(attn.W_v)
                wo = dense(attn.W_o)
                sl = None if slopes is None else float(slopes[h].item())
                # Does V at this head read ALU_LO and O write SE_ALU_LO?
                # W_v shape [n_heads*HD, d_model]; W_o shape [d_model, n_heads*HD]
                v_reads_alu = float(wv[row:row + HD, alu_lo:alu_lo + 16].abs().max().item())
                o_writes_se = float(wo[se_alu_lo:se_alu_lo + 16, row:row + HD].abs().max().item())
                print(f"blk{li} head{h}: Wq[row,SE]={wq[row,se].item():.2f} "
                      f"Wk[row,AX]={dense(attn.W_k)[row,ax].item():.2f} "
                      f"slope={sl} | V.maxread(ALU_LO)={v_reads_alu:.2f} "
                      f"O.maxwrite(SE_ALU_LO)={o_writes_se:.2f}")
    print("\nFull alibi_slopes per block with a SE-anchored head:")
    for li, block in enumerate(tracer.model.blocks):
        attn = block.attn
        wq = dense(attn.W_q)
        HD = wq.shape[0] // attn.num_heads
        slopes = getattr(attn, "alibi_slopes", None)
        anchored = any(
            abs(wq[h * HD, se].item()) >= 5.0
            for h in range(attn.num_heads) if h * HD < wq.shape[0]
        )
        if anchored and slopes is not None:
            print(f"  blk{li}: slopes={[round(float(x),3) for x in slopes.tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
