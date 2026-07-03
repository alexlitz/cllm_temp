#!/usr/bin/env python3
"""Diagnose the step-9 AX-corruption (70->72) for func_identity_0 when the LEV
PC-restore flag stack is on.

For the LEV step and the NEXT step (step 9), at EVERY register marker row
(PC/AX/SP/BP) it dumps:
  * the decoded byte-0 value of that register at the FINAL block
  * head-14's softmax-1 attention top (does it self-fire / bleed here?)
  * head-14's OUTPUT byte-0 delivery at that row (post-L15 vs final)

This isolates whether head 14 bleeds onto the step-9 AX row (the documented
"BP/AX restored as 90" bleed) or whether the AX corruption is an independent
epilogue rule.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_func_step9_axbleed.py <id> <lev_step> [maxsteps]
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


def hot(r, dp, lo, hi):
    a = r[dp[lo]:dp[lo]+16]; b = r[dp[hi]:dp[hi]+16]
    return (int(b.argmax()) << 4) | int(a.argmax())


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    lev = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 14
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
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15]
    l15 = l15[0] if l15 else None
    print(f"L15(>=15 heads) block = {l15}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)

    fin = probe.model.forward(padded, stop_after_block=nblk - 1)[0]
    pre15 = probe.model.forward(padded, stop_after_block=l15 - 1)[0] if l15 is not None else None
    post15 = probe.model.forward(padded, stop_after_block=l15)[0] if l15 is not None else None

    # head-14 Q/K
    if l15 is not None:
        blk = probe.model.blocks[l15]; attn = blk.attn
        x_in = pre15.unsqueeze(0)
        x = blk.attn_norm(x_in) if (getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None) else x_in
        Wq = _dn(attn.W_q); Wk = _dn(attn.W_k)
        H = attn.num_heads; hd = Wq.shape[0] // H; S = x.shape[1]
        Q = (x[0] @ Wq.t()).view(S, H, hd).transpose(0, 1)
        K = (x[0] @ Wk.t()).view(S, H, hd).transpose(0, 1)
        scale = hd ** -0.5
        sc14 = (Q[14] @ K[14].t()) * scale
        mask = attn.mask[:S, :S].float() if getattr(attn, "mask", None) is not None else torch.zeros(S, S, device=x.device)

    for st in (lev, lev + 1):
        if st >= len(sm):
            continue
        print(f"\n=== step {st} ===")
        for reg in ("PC", "AX", "SP", "BP"):
            row = sm[st].get(reg)
            if row is None:
                continue
            v_fin = hot(fin[row], dp, "OUTPUT_LO", "OUTPUT_HI")
            line = f"  [{reg} @row{row}] OUTPUT_final=0x{v_fin:02x}({v_fin})"
            if l15 is not None:
                o15 = hot(post15[row], dp, "OUTPUT_LO", "OUTPUT_HI")
                attw = F.softmax(sc14[row] + mask[row], dim=-1)
                top = attw.argsort(descending=True)[:3].tolist()
                topstr = ", ".join(f"p{p}(w{float(attw[p]):.2f},v0x{hot(pre15[p],dp,'CLEAN_EMBED_LO','CLEAN_EMBED_HI'):02x})" for p in top)
                line += f"  OUTPUT_postL15=0x{o15:02x}  head14:[{topstr}]"
            print(line)


if __name__ == "__main__":
    main()
