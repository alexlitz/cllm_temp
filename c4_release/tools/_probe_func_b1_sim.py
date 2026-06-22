#!/usr/bin/env python3
"""Simulate adding a MEM_VAL_B1 positive K to L15 head-0 slot-3 (the LI value-row
selector) and check whether the a-value row (271) then OUT-scores the spurious
ADDR_B0_LO=8 cluster (292-295) at func_add step9. We add B1_W * Q_b1 * K_b1 /
sqrt(hd) where Q is gated MARK_AX (1.0 at the lookup row) and K reads MEM_VAL_B1.
This is the exact bilinear the real fix would add. Sweeps B1_W to find the
strength that flips a WITHOUT disturbing b (step12)."""
from __future__ import annotations
import os, sys, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PROJ = os.path.dirname(REPO)
for p in (PROJ, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("C4_SMOKE_SPEC_K", "0"); os.environ.setdefault("C4_TEST_SPEC_K", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa
from c4_release.neural_vm.vm_step import Token  # noqa
from tools.interp_oracle_gate import (build_production_model, build_code_prompt,
                                       oracle_tape_and_steps)  # noqa
STEP = int(Token.STEP_TOKENS); PROGS = generate_test_programs()


def run(pid, li_steps):
    with contextlib.redirect_stdout(io.StringIO()):
        model, layout = build_production_model("cpu")
    dimpos = layout.dim_positions
    op_li = dimpos["OP_LI_RELAY"]
    L15 = next(bi for bi, blk in enumerate(model.blocks)
               if abs(float((blk.attn.W_q.to_dense() if blk.attn.W_q.is_sparse
                             else blk.attn.W_q)[0, op_li].item())) > 1000.0)
    src, exp, _ = PROGS[pid]
    bc = compile_c(src)[0]
    prompt = build_code_prompt(bc, b"")
    ot = oracle_tape_and_steps(bc, b"", max_steps=40)
    plen = len(prompt)
    tok = torch.tensor([list(prompt) + list(ot.draft_tokens)], dtype=torch.long)
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
    Wq = attn.W_q.to_dense() if attn.W_q.is_sparse else attn.W_q
    Wk = attn.W_k.to_dense() if attn.W_k.is_sparse else attn.W_k
    Q = (pre @ Wq.T).view(seq, nH, hd)
    K = (pre @ Wk.T).view(seq, nH, hd)
    mark_ax = dimpos["MARK_AX"]; vb1 = dimpos["MEM_VAL_B1"]
    clean_lo = dimpos["CLEAN_EMBED_LO"]
    for step in li_steps:
        base = plen + step * STEP
        b0row = base + 5
        pc, ax = ot.steps[step]
        scores0 = (K[:, 0, :] @ Q[b0row, 0]) / (hd ** 0.5)
        scores0[b0row + 1:] = float("-inf")
        for B1_W in (0.0, 5e4, 1e5, 2e5, 5e5):
            # extra = B1_W * Q_markax(b0row) * K_memvalb1(cand)
            qax = float(pre[b0row, mark_ax].item())
            extra = B1_W * qax * pre[:, vb1]
            sc = scores0 + extra
            sc[b0row + 1:] = float("-inf")
            win = int(sc.argmax().item())
            wlo = pre[win, clean_lo:clean_lo+16]
            wli = int(wlo.argmax().item()) if float(wlo.max()) > 0.3 else -1
            print(f"  pid{pid} step{step} want=0x{ax:04x} B1_W={B1_W:>7.0f} "
                  f"winner={win} CLEAN_LO_nib={wli}")
        print()


if __name__ == "__main__":
    run(575, [9, 12])   # func_add: a@9 (broken), b@12 (ok)
    run(650, [9, 12])   # func_max
