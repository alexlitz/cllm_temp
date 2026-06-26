#!/usr/bin/env python3
"""Like probe_sp_penalty_exact but AUTO-DETECTS the head-5 block (which shifts
when the block-10 sharpener post_op is present) and reports the per-step penalty
magnitude + winner flips on the SHARPENED bands. A bounded penMaxAbs (~G) on
var_simple == the sharpener fixed the blow-up; a winner flip on id816 step6 ==
the discriminator still works.
"""
import os, sys, math
os.environ.setdefault("C4_TEST_SPEC_K", "0"); os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
import torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token


def _corpus(idx):
    from src.compiler import compile_c
    from tests.test_suite_1000 import generate_test_programs
    src, exp, desc = generate_test_programs()[idx]
    bc, _ = compile_c(src); return bc, exp, desc


def main(ids):
    p = build_groundtruth_probe(); dp = p.model.dim_positions
    LOS = int(dp["SP_ADDR_LO_SHARP"])
    # auto-detect head-5 block (attn W_q reads SP_ADDR_LO_SHARP)
    bi = None
    for j, blk in enumerate(p.model.blocks):
        a = getattr(blk, "attn", None)
        if a is not None and hasattr(a, "W_q"):
            Wq = a.W_q.to_dense() if (a.W_q.is_sparse or getattr(a.W_q, "is_sparse_csr", False)) else a.W_q
            if Wq[:, LOS:LOS+16].abs().sum() > 1e-6:
                bi = j; break
    assert bi is not None, "head-5 SHARP-reading block not found"
    attn = p.model.blocks[bi].attn
    HD = attn.W_q.shape[0]//attn.num_heads; base = 5*HD
    def _d(W): return W.to_dense() if (W.is_sparse or getattr(W, "is_sparse_csr", False)) else W
    Wq = _d(attn.W_q); Wk = _d(attn.W_k)
    slope = float(attn.alibi_slopes[5]) if getattr(attn, "alibi_slopes", None) is not None else 0.0
    pen = list(range(30, 47))
    print(f"head-5 block = {bi}", flush=True)
    for idx in ids:
        bc, exp, desc = _corpus(idx); ctx = p._final_context(bc, max_steps=14)
        ax = [i for i, t in enumerate(ctx) if t == Token.REG_AX]; seq = len(ctx)
        padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
        resid = p.model.forward(padded, stop_after_block=bi-1)[0]
        print(f"\n== id={idx} {desc} exp={exp} ax_rows={ax} ==", flush=True)
        for st, qpos in enumerate(ax):
            q = resid[qpos]@Wq[base:base+HD].T; K = resid@Wk[base:base+HD].T
            sc = (K@q)/math.sqrt(HD)
            qn = q.clone(); qn[pen] = 0.0; scn = (K@qn)/math.sqrt(HD)
            dist = (qpos-torch.arange(seq, device=resid.device)).clamp(min=0).float()
            al = -slope*dist; causal = torch.arange(seq, device=resid.device) <= qpos
            tot = (sc+al).masked_fill(~causal, float("-inf")); totn = (scn+al).masked_fill(~causal, float("-inf"))
            w = int(tot.argmax()); wn = int(totn.argmax())
            penw = float((sc-scn)[w]); penmax = float((sc-scn).abs().max())
            chg = "" if w == wn else "  <-- WINNER CHANGED"
            if penmax > 0.01 or w != wn:
                print(f"  step{st} q={qpos}: winner={w}(nopen={wn}) pen@win={penw:+.2f} penMaxAbs={penmax:.2f}{chg}", flush=True)


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [250, 251, 816]
    main(ids)
