#!/usr/bin/env python3
"""L15 (block 29) per-head attention at the AX-marker (and byte) rows of a
func step, to see WHERE the LI-from-frame lookup attends and what value it
copies. Prints, per head, the top-3 attended K positions + the emitted token
at those positions (so we can tell if it pulls the frame VALUE byte vs the
address byte / STACK0 byte).

Usage: python tools/_probe_l15_li_attn.py <id> <step> [blk=29] [maxsteps]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
import torch.nn.functional as F
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

STEP_END = int(Token.STEP_END); HALT = int(Token.HALT)
NAMES = {int(v): k for k, v in vars(Token).items() if isinstance(v, int)}


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); want_step = int(sys.argv[2])
    blk = int(sys.argv[3]) if len(sys.argv) > 3 else 29
    ms = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; device = probe._device
    ctx = probe._final_context(bc, max_steps=ms)
    RAX = int(Token.REG_AX)
    steps = []; s = 0
    for i, t in enumerate(ctx):
        if t == STEP_END or t == HALT:
            steps.append((s, i)); s = i + 1
    st, en = steps[want_step]
    axm = next(i for i in range(st, en + 1) if ctx[i] == RAX)
    padded = torch.tensor([ctx], dtype=torch.long, device=device)
    resid = model.forward(padded, stop_after_block=blk - 1)[0].float()
    if resid.is_sparse: resid = resid.to_dense()
    x = resid.unsqueeze(0)
    attn = model.blocks[blk].attn
    H = attn.num_heads; HD = attn.head_dim
    Wq = attn.W_q.to_dense() if attn.W_q.is_sparse else attn.W_q
    Wk = attn.W_k.to_dense() if attn.W_k.is_sparse else attn.W_k
    Q = (x @ Wq.float().t()).view(1, -1, H, HD).transpose(1, 2)
    K = (x @ Wk.float().t()).view(1, -1, H, HD).transpose(1, 2)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale
    S = x.shape[1]
    scores = scores + attn.mask[:S, :S]
    probs = F.softmax(scores, dim=-1)  # [1,H,S,S]
    print(f"id{pid} {desc} step{want_step} blk{blk} (L15) AX_marker={axm} "
          f"nheads={H} emit_ax_b0={ctx[axm+1]}")
    # query rows of interest: AX marker (b0 predictor) + the 3 byte rows
    for qoff, qlabel in [(0, "b0/marker"), (1, "b1row"), (2, "b2row")]:
        qpos = axm + qoff
        print(f" -- query row {qpos} ({qlabel}) --")
        for h in range(H):
            p = probs[0, h, qpos]
            top = torch.topk(p, 3)
            parts = []
            for w, kk in zip(top.values.tolist(), top.indices.tolist()):
                tok = ctx[kk]
                nm = NAMES.get(tok, str(tok))
                parts.append(f"K{kk}(tok{tok}/{nm[:6]} w{w:.2f})")
            if top.values[0] > 0.15:
                print(f"   h{h:2d}: {'  '.join(parts)}")


if __name__ == "__main__":
    main()
