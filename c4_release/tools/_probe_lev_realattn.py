#!/usr/bin/env python3
"""Extract head-14's REAL attention weights (with head_dim^-0.5 scale + the
baked ALiBi/causal mask) at the LEV PC-marker query, plus the OUTPUT byte-0 the
L15 block actually delivers. The CAM re-score probes ignore the scale, mask and
self/marker rows; THIS probe is the ground truth for "does head 14 attend the
genuine return store and deliver its value to OUTPUT".

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_lev_realattn.py <id> <lev_step> [maxsteps]
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
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    if lev >= len(sm):
        print(f"lev {lev} OOR (nsteps={len(sm)})"); return
    pcm = sm[lev].get("PC")
    l15 = [ph for ph, b in enumerate(probe.model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    blk = probe.model.blocks[l15]; attn = blk.attn
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    x_in = probe.model.forward(padded, stop_after_block=l15 - 1)
    x = x_in
    if getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None:
        x = blk.attn_norm(x_in)
    Wq = _dn(attn.W_q); Wk = _dn(attn.W_k)
    H = attn.num_heads; hd = Wq.shape[0] // H; S = x.shape[1]
    Q = (x[0] @ Wq.t()).view(S, H, hd).transpose(0, 1)
    K = (x[0] @ Wk.t()).view(S, H, hd).transpose(0, 1)
    scale = hd ** -0.5
    sc = (Q[14] @ K[14].t()) * scale
    mask = attn.mask[:S, :S].float() if getattr(attn, "mask", None) is not None else torch.zeros(S, S, device=x.device)
    row = sc[pcm] + mask[pcm]
    attw = F.softmax(row, dim=-1)
    top = attw.argsort(descending=True)[:8].tolist()
    print(f"id{pid} {desc} lev={lev} q@{pcm} L15blk={l15} hd={hd} scale={scale:.4f}")
    print(f"  head14 attn top: " + ", ".join(
        f"p{p}(addr0x{(hot(x_in[0][p],dp,'ADDR_B0_LO','ADDR_B0_HI')|(hot(x_in[0][p],dp,'ADDR_B1_LO','ADDR_B1_HI')<<8)):04x},"
        f"v0x{hot(x_in[0][p],dp,'CLEAN_EMBED_LO','CLEAN_EMBED_HI'):02x},w{float(attw[p]):.3f})"
        for p in top))
    # OUTPUT delivered after L15 and at final block.
    o_l15 = hot(probe.model.forward(padded, stop_after_block=l15)[0][pcm], dp, "OUTPUT_LO", "OUTPUT_HI")
    o_fin = hot(probe.model.forward(padded, stop_after_block=len(probe.model.blocks)-1)[0][pcm], dp, "OUTPUT_LO", "OUTPUT_HI")
    print(f"  OUTPUT byte0 after L15(blk{l15})=0x{o_l15:02x}({o_l15})  final=0x{o_fin:02x}({o_fin})")


if __name__ == "__main__":
    main()
