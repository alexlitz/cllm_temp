#!/usr/bin/env python3
"""Per-key score breakdown for L15 head 14 at the LEV PC marker: compare the
genuine return store byte-0 row vs the wrong same-address rows. Shows the raw
score (pre-softmax) for a list of candidate key positions + the dim flags that
drive them, so we can see why the wrong store wins.

Usage: python tools/_probe_lev_head14_keys.py <id> <lev_step> [maxsteps] [keys csv]
"""
from __future__ import annotations
import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END); HALT = int(Token.HALT)
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
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    keys = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    model = probe.model; dev = probe._device; dimp = model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    pcm = sm[lev].get("PC")
    L15 = [i for i, b in enumerate(model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    blk = model.blocks[L15]; attn = blk.attn
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    xin = model.forward(padded, stop_after_block=L15 - 1)[0].float()
    x = xin
    if getattr(blk, "use_pre_norm", False) and getattr(blk, "attn_norm", None) is not None:
        x = blk.attn_norm(xin.unsqueeze(0))[0]
    Wq = attn.W_q.to_dense() if (attn.W_q.is_sparse or attn.W_q.layout != torch.strided) else attn.W_q
    Wk = attn.W_k.to_dense() if (attn.W_k.is_sparse or attn.W_k.layout != torch.strided) else attn.W_k
    H = attn.num_heads; HD = Wq.shape[0] // H
    Q = (x @ Wq.float().t()).view(-1, H, HD).transpose(0, 1)
    K = (x @ Wk.float().t()).view(-1, H, HD).transpose(0, 1)
    scale = HD ** -0.5
    h = 14
    qv = Q[h, pcm]
    if keys is None:
        # auto: all MEM_STORE>0.3 rows up to pcm
        keys = [p for p in range(pcm) if "MEM_STORE" in dimp and xin[p, dimp["MEM_STORE"]].item() > 0.3]
    print(f"id{pid} {desc} lev={lev} q@{pcm} L15={L15} hd={HD} scale={scale:.4f}")
    print(f"{'key':>4} {'tok':>4} {'val':>4} {'addr':>6} {'MSTORE':>6} {'STK0':>5} "
          f"{'BIDX':>4} {'score':>10}")
    cl_lo = dimp["CLEAN_EMBED_LO"]; cl_hi = dimp["CLEAN_EMBED_HI"]
    def hot(p, b): return int(torch.argmax(xin[p, b:b + 16]).item())
    for p in keys:
        sc = float((qv * K[h, p]).sum().item()) * scale
        clv = hot(p, cl_lo) | (hot(p, cl_hi) << 4)
        a = (hot(p, dimp["ADDR_B0_LO"]) | (hot(p, dimp["ADDR_B0_HI"]) << 4)) | \
            ((hot(p, dimp["ADDR_B1_LO"]) | (hot(p, dimp["ADDR_B1_HI"]) << 4)) << 8)
        mstore = float(xin[p, dimp["MEM_STORE"]].item())
        stk0 = float(xin[p, dimp["STACK0_BYTE0"]].item()) if "STACK0_BYTE0" in dimp else 0
        bidx = "".join(str(i) for i in range(4) if dimp.get(f"BYTE_INDEX_{i}") and xin[p, dimp[f"BYTE_INDEX_{i}"]].item() > 0.5)
        opjsr = float(xin[p, dimp["OP_JSR"]].item()) if "OP_JSR" in dimp else 0
        opent = float(xin[p, dimp["OP_ENT"]].item()) if "OP_ENT" in dimp else 0
        print(f"{p:>4} {ctx[p]:>4} {clv:>4} 0x{a:04x} {mstore:>6.2f} {stk0:>5.1f} "
              f"{bidx:>4} {sc:>10.1f}  OP_JSR={opjsr:.1f} OP_ENT={opent:.1f}")


if __name__ == "__main__":
    main()
