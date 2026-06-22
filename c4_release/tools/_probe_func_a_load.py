#!/usr/bin/env python3
"""Why does the FIRST-param LI (load `a`) attend the WRONG row?

func_add step9 (LI a, want 0x39): head-0 winner is rows 292-295 (no marker,
no value) instead of store@266 (a's value row, CLEAN_LO=9). This dumps the
full per-row head-0 score around the a-store and the winner cluster + the
nonzero residual dims of both, to localize what content the winner carries
that store@266's value row lacks (or what makes store@266 lose).
"""
from __future__ import annotations
import os, sys, contextlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0")
os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from tests.test_suite_1000 import generate_test_programs  # noqa: E402
from src.compiler import compile_c  # noqa: E402
from c4_release.neural_vm.vm_step import Token  # noqa: E402
from tools.interp_oracle_gate import (  # noqa: E402
    build_production_model, build_code_prompt, oracle_tape_and_steps,
)

STEP = int(Token.STEP_TOKENS)
PROGS = generate_test_programs()


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    rev = {v: k for k, v in dimpos.items()}
    op_li = dimpos["OP_LI_RELAY"]
    L15 = None
    for bi, blk in enumerate(model.blocks):
        wq = blk.attn.W_q
        if wq.is_sparse_csr or wq.is_sparse:
            wq = wq.to_dense()
        if abs(float(wq[0, op_li].item())) > 1000.0:
            L15 = bi

    pid = 575  # func_add(57,11): a=0x39 at step9
    src, exp, desc = PROGS[pid]
    bc = compile_c(src)[0]
    prompt = build_code_prompt(bc, b"")
    ot = oracle_tape_and_steps(bc, b"", max_steps=40)
    tape = list(prompt) + list(ot.draft_tokens)
    tok = torch.tensor([tape], dtype=torch.long)
    plen = len(prompt)

    cap = {}
    h = model.blocks[L15].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("pre", i[0].detach().clone()))
    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            model.forward(tok)
    h.remove()
    pre = cap["pre"][0]
    seq = pre.shape[0]
    attn = model.blocks[L15].attn
    nH, hd = attn.num_heads, attn.head_dim
    for w in ("W_q", "W_k"):
        t = getattr(attn, w)
        if t.is_sparse_csr or t.is_sparse:
            setattr(attn, "_d_" + w, t.to_dense())
    Wq = getattr(attn, "_d_W_q", attn.W_q)
    Wk = getattr(attn, "_d_W_k", attn.W_k)
    x = pre
    Q = (x @ Wq.T).view(seq, nH, hd)
    K = (x @ Wk.T).view(seq, nH, hd)

    def mk(pos):
        return "|".join(nm.replace("MARK_", "") for nm in
                        ("MARK_PC", "MARK_AX", "MARK_SP", "MARK_BP", "MARK_MEM",
                         "MARK_STACK0") if pre[pos, dimpos[nm]].item() > 0.5) or "-"

    step = 9  # LI a
    base = plen + step * STEP
    b0row = base + 5
    qv = Q[b0row, 0]
    scores = (K[:, 0, :] @ qv) / (hd ** 0.5)
    scores[b0row + 1:] = float("-inf")
    print(f"=== func_add LI step9 b0row={b0row} (want a=0x39) ===")
    # store@266 is a's value frame; vrow 271
    print("--- scores around a-store (260-300) ---")
    order = torch.argsort(scores[:b0row+1], descending=True)
    rank = {int(r): i for i, r in enumerate(order.tolist())}
    for pos in range(260, 300):
        s = float(scores[pos].item())
        if s < -1e8:
            continue
        clo = pre[pos, dimpos["CLEAN_EMBED_LO"]:dimpos["CLEAN_EMBED_LO"]+16]
        cli = int(clo.argmax().item()) if float(clo.max()) > 0.3 else -1
        alo = pre[pos, dimpos["ADDR_B0_LO"]:dimpos["ADDR_B0_LO"]+16]
        ali = int(alo.argmax().item()) if float(alo.max()) > 0.3 else -1
        mvb0 = float(pre[pos, dimpos["MEM_VAL_B0"]].item())
        ms = float(pre[pos, dimpos["MEM_STORE"]].item())
        tag = " <<a-VALrow" if pos == 271 else (" <store" if ms > 0.5 else "")
        print(f"  pos{pos} score={s:8.1f} rank={rank.get(pos,'?'):>3} mk={mk(pos):6} "
              f"CLEAN_LO={cli:>2} ADDR_B0_LO={ali:>2} MEM_VAL_B0={mvb0:5.1f} "
              f"MEM_STORE={ms:.1f}{tag}")
    # top-8 overall
    print("--- top-8 overall ---")
    for i in range(8):
        pos = int(order[i].item())
        s = float(scores[pos].item())
        nz = torch.nonzero(pre[pos].abs() > 0.5).flatten().tolist()
        sig = ",".join(f"{rev.get(d,d)}={float(pre[pos,d]):.1f}" for d in nz[:14])
        print(f"  #{i} pos{pos} score={s:8.1f} mk={mk(pos)} :: {sig}")
    # the a-value row 271 detail
    print("--- a-value row 271 full nonzero ---")
    nz = torch.nonzero(pre[271].abs() > 0.4).flatten().tolist()
    print("  " + ", ".join(f"{rev.get(d,d)}={float(pre[271,d]):.2f}" for d in nz[:40]))


if __name__ == "__main__":
    main()
