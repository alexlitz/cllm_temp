#!/usr/bin/env python3
"""Reconstruct L7 operand_gather HEAD 1 attention weights at the LEA AX-marker
query row, for the FIRST LEA (works, full BP gather) vs the RE-READ LEA (fails,
attenuated 5.5 residue). Shows WHICH BP/SP marker row head 1 attends to and what
OUTPUT_LO that row carries — to localize the staleness (dilution across many
stale BP rows vs attending the wrong/older row).

Hook-free: reads the block-input residual (stop_after_block=blk-1), applies the
block's W_q/W_k + alibi + softmax1 directly, reports top attended rows + their
OUTPUT_LO/OUTPUT_HI (the V source for head 1) and marker-band membership.

Usage: CUDA_VISIBLE_DEVICES=0 C4_NO_STACK0_EMIT=1 C4_OPERAND_FROM_MEMSP=1 \
       C4_VM_CACHE_DIR=/tmp/c4cache_funcreread python tools/_probe_lea_head1_attn.py 575 11
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch  # noqa
import torch.nn.functional as F  # noqa
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


@torch.no_grad()
def analyze(probe, pid, step, blk, head):
    dp = probe.model.embed._dim_positions
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    ctx = probe._final_context(bc, max_steps=max(step + 3, 16))
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    row = sm[step]["AX"]
    dev = probe._device
    toks = torch.tensor([ctx], dtype=torch.long, device=dev)
    S = len(ctx)
    resid = probe.model.forward(toks, stop_after_block=blk - 1)[0]  # [S,D]
    x = resid.unsqueeze(0)
    attn = probe.model.blocks[blk].attn
    H = attn.num_heads; HD = attn.head_dim
    Q = F.linear(x, attn.W_q).view(1, S, H, HD).transpose(1, 2)
    K = F.linear(x, attn.W_k).view(1, S, H, HD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
    slopes = getattr(attn, "alibi_slopes", None)
    pos = torch.arange(S, device=dev).float()
    if slopes is not None:
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
        scores = scores - slopes.view(1, H, 1, 1) * dist.view(1, 1, S, S)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
    scores = scores + causal.view(1, 1, S, S)
    if getattr(attn, "use_softmax1", False):
        sink = torch.zeros(1, H, S, 1, device=dev)
        ext = torch.cat([scores, sink], dim=-1)
        w_all = F.softmax(ext, dim=-1)[..., :S]
    else:
        w_all = F.softmax(scores, dim=-1)
    w = w_all[0, head, row]
    raw = scores[0, head, row]
    bp_base = dp["MARK_BP"]; sp_base = dp["MARK_SP"]; out_lo = dp["OUTPUT_LO"]
    out_hi = dp["OUTPUT_HI"]
    slope = float(slopes[head]) if slopes is not None else None
    print(f"id{pid} {desc} step{step} blk{blk} head{head} qrow={row} S={S} "
          f"slope={slope} sum_w={float(w.sum()):.4f}")
    top = torch.topk(w, k=min(12, S))
    for w_i, r in zip(top.values.tolist(), top.indices.tolist()):
        is_bp = abs(float(resid[r, bp_base])) > 0.5
        is_sp = abs(float(resid[r, sp_base])) > 0.5
        marker = "BP" if is_bp else ("SP" if is_sp else "")
        lo = [(round(float(resid[r, out_lo + k]), 1), k) for k in range(16)
              if abs(float(resid[r, out_lo + k])) > 1]
        hi = [(round(float(resid[r, out_hi + k]), 1), k) for k in range(16)
              if abs(float(resid[r, out_hi + k])) > 1]
        print(f"  w={w_i:.4f} raw={float(raw[r]):6.1f} row={r:3d} tok={ctx[r]:3d} "
              f"{marker:2s} OUT_LO={lo} OUT_HI={hi}")


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 575
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    head = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    analyze(probe, pid, step, blk, head)


if __name__ == "__main__":
    main()
