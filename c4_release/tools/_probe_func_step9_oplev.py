#!/usr/bin/env python3
"""Check whether OP_LEV / MARK_PC leak into step 9 (the post-LEV step) at the PC
marker, which would let head 14 (gated on OP_LEV AND MARK_PC) bleed there and
pin PC=90 instead of advancing to 98.

Dumps, per step, at the PC-marker row: OP_LEV / OP_ADJ / OP_POP residual at the
head-14 INPUT block (pre-L15), the decoded opcode, and head-14 firing weight.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_func_step9_oplev.py <id> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch, torch.nn.functional as F
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


def _dn(W):
    return (W.to_dense() if (W.is_sparse or W.layout != torch.strided) else W).float()


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    print(f"id{pid} {desc} exp={exp} nsteps={len(sm)}")
    nblk = len(probe.model.blocks)
    l15 = [ph for ph, b in enumerate(probe.model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    pre15 = probe.model.forward(padded, stop_after_block=l15 - 1)[0]

    # head-14 firing weight at each PC marker
    blk = probe.model.blocks[l15]; attn = blk.attn
    x = blk.attn_norm(pre15.unsqueeze(0)) if (getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None) else pre15.unsqueeze(0)
    Wq = _dn(attn.W_q); Wk = _dn(attn.W_k)
    H = attn.num_heads; hd = Wq.shape[0] // H; S = x.shape[1]
    Q = (x[0] @ Wq.t()).view(S, H, hd).transpose(0, 1)
    K = (x[0] @ Wk.t()).view(S, H, hd).transpose(0, 1)
    scale = hd ** -0.5
    sc14 = (Q[14] @ K[14].t()) * scale
    mask = attn.mask[:S, :S].float() if getattr(attn, "mask", None) is not None else torch.zeros(S, S, device=x.device)

    names = [n for n in ("OP_LEV", "OP_ADJ", "OP_POP", "OP_JMP", "OP_ENT", "OP_JSR", "MARK_PC") if n in dp]
    print("\nstep | PC-marker-row | " + " ".join(f"{n}" for n in names) + " | head14 self-weight(diag) | head14 top")
    for st in range(len(sm)):
        row = sm[st].get("PC")
        if row is None:
            continue
        r = pre15[row]
        vals = " ".join(f"{n}={r[dp[n]].item():+.1f}" for n in names)
        attw = F.softmax(sc14[row] + mask[row], dim=-1)
        top = attw.argsort(descending=True)[:2].tolist()
        topstr = ",".join(f"p{p}(w{float(attw[p]):.2f})" for p in top)
        print(f"  {st:2d} | row{row} | {vals} | {topstr}")


if __name__ == "__main__":
    main()
