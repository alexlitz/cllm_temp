#!/usr/bin/env python3
"""Dump head-5's penalty contribution (slots 30-46) per candidate row at EVERY
binary-op AX step (incl SI/SC), to verify the exact-cancel holds (penalty ~0 on
the attended/winning store row for single-store ops). Reports, per AX step:
the winner row WITH and WITHOUT the penalty dims, and the penalty value at the
winner. A nonzero penalty at the winner of a single-store op == the bug.
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
    bi = 11; attn = p.model.blocks[bi].attn
    HD = attn.W_q.shape[0]//attn.num_heads; base = 5*HD
    def _d(W): return W.to_dense() if (W.is_sparse or getattr(W,"is_sparse_csr",False)) else W
    Wq=_d(attn.W_q); Wk=_d(attn.W_k)
    slope=float(attn.alibi_slopes[5]) if getattr(attn,"alibi_slopes",None) is not None else 0.0
    pen=list(range(30,47))
    for idx in ids:
        bc,exp,desc=_corpus(idx); ctx=p._final_context(bc,max_steps=14)
        ax=[i for i,t in enumerate(ctx) if t==Token.REG_AX]; seq=len(ctx)
        padded=torch.tensor([ctx],dtype=torch.long,device=p._device)
        resid=p.model.forward(padded,stop_after_block=bi-1)[0]
        print(f"\n== id={idx} {desc} exp={exp} ax_rows={ax} ==",flush=True)
        for st,qpos in enumerate(ax):
            q=resid[qpos]@Wq[base:base+HD].T; K=resid@Wk[base:base+HD].T
            sc=(K@q)/math.sqrt(HD)
            qn=q.clone(); qn[pen]=0.0; scn=(K@qn)/math.sqrt(HD)
            dist=(qpos-torch.arange(seq,device=resid.device)).clamp(min=0).float()
            al=-slope*dist; causal=torch.arange(seq,device=resid.device)<=qpos
            tot=(sc+al).masked_fill(~causal,float("-inf")); totn=(scn+al).masked_fill(~causal,float("-inf"))
            w=int(tot.argmax()); wn=int(totn.argmax())
            penw=float((sc-scn)[w])
            chg="" if w==wn else "  <-- WINNER CHANGED"
            # only print steps where head-5 plausibly fires (winner score not -inf and penalty nonzero somewhere)
            penmax=float((sc-scn).abs().max())
            if penmax>0.01 or w!=wn:
                print(f"  step{st} q={qpos}: winner={w}(nopen={wn}) pen@win={penw:+.2f} penMaxAbs={penmax:.2f}{chg}",flush=True)

if __name__=="__main__":
    ids=[int(x) for x in sys.argv[1:]] or [250,816]
    main(ids)
