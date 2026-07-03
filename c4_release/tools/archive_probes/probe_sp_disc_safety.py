#!/usr/bin/env python3
"""Single-store safety probe for the L8 head-5 SP-frame discriminator.

For each (idx) at its LAST binary-op AX step, recompute head-5's per-candidate
score WITH and WITHOUT the SP-penalty dims (slots 30-46), and report whether the
WINNER row changes. For a single-store op the winner MUST be identical (penalty
== 0 on the live store == query SP-frame). Also reports the delivered ALU_LO
cell. Run on GPU1 (cached build).
"""
import os, sys, math
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
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
    bc, _ = compile_c(src)
    return bc, exp, desc


def main(ids, step_override):
    p = build_groundtruth_probe()
    dp = p.model.dim_positions
    SP = int(dp["SP_ADDR_LO"]) if "SP_ADDR_LO" in dp else None
    bi = 11
    attn = p.model.blocks[bi].attn
    HD = attn.W_q.shape[0] // attn.num_heads
    base = 5 * HD
    SP_PEN_BASE = 30
    def _dense(W):
        if W.is_sparse or getattr(W, "is_sparse_csr", False):
            return W.to_dense()
        return W
    Wq = _dense(attn.W_q)
    Wk = _dense(attn.W_k)
    slope = float(attn.alibi_slopes[5]) if getattr(attn, "alibi_slopes", None) is not None else 0.0
    for idx in ids:
        bc, exp, desc = _corpus(idx)
        ctx = p._final_context(bc, max_steps=14)
        ax = [i for i, t in enumerate(ctx) if t == Token.REG_AX]
        st = step_override.get(idx, len(ax) - 1)  # default: LAST binary-op AX step
        qpos = ax[st]
        seq = len(ctx)
        padded = torch.tensor([ctx], dtype=torch.long, device=p._device)
        resid = p.model.forward(padded, stop_after_block=bi - 1)[0]
        q = resid[qpos] @ Wq[base:base + HD].T
        K = resid @ Wk[base:base + HD].T
        scores_full = (K @ q) / math.sqrt(HD)
        # penalty-only contribution = scores using ONLY dims 30..46
        pen_slots = list(range(SP_PEN_BASE, SP_PEN_BASE + 17))
        qp = q.clone(); Kp = K.clone()
        mask = torch.ones(HD, dtype=torch.bool); mask[pen_slots] = False
        q_nopen = q.clone(); q_nopen[pen_slots] = 0.0
        scores_nopen = (K @ q_nopen) / math.sqrt(HD)
        dist = (qpos - torch.arange(seq, device=resid.device)).clamp(min=0).float()
        alibi = -slope * dist
        causal = torch.arange(seq, device=resid.device) <= qpos
        tot_full = (scores_full + alibi).masked_fill(~causal, float("-inf"))
        tot_nopen = (scores_nopen + alibi).masked_fill(~causal, float("-inf"))
        w_full = int(tot_full.argmax()); w_nopen = int(tot_nopen.argmax())
        out = p.model.forward(padded, stop_after_block=bi)[0]
        alu = out[qpos, dp["ALU_LO"]:dp["ALU_LO"] + 16]
        # SP_ADDR_LO at query + winner
        def spcell(r):
            if SP is None: return None
            v = resid[r, SP:SP + 16]
            return (int(v.argmax()), float(v.max()))
        pen_at_win = float((scores_full - scores_nopen)[w_full])
        flag = "OK" if w_full == w_nopen else "*** WINNER CHANGED ***"
        print(f"id={idx} {desc} exp={exp} step={st}/{len(ax)-1} q={qpos}: "
              f"winner_full={w_full} winner_nopen={w_nopen} {flag} | "
              f"pen@winner={pen_at_win:+.1f} | SPq={spcell(qpos)} SPwin={spcell(w_full)} | "
              f"ALU_LO cell={int(alu.argmax())}({float(alu.max()):+.1f})", flush=True)


if __name__ == "__main__":
    ids = [int(x) for x in sys.argv[1:]] or [816, 0, 877, 850, 825]
    step_override = {816: 6}  # add_mul ADD step (others: last AX)
    main(ids, step_override)
