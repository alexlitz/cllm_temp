#!/usr/bin/env python3
"""Per-slot head-14 score decomposition between two candidate store rows.

For a func_* LEV step, recompute the head-14 Q@K score slot-by-slot
(per-head-dim) for the QUERY at the LEV PC marker against each of two K rows,
and print the per-slot contribution + the running deficit. This pinpoints WHICH
head-dim slots make the wrong store win (the CAM-aliasing discriminator hunt).

Usage: CUDA_VISIBLE_DEVICES=1 python tools/_probe_lev_slot_decomp.py <id> <lev_step> <winner_pos> <target_pos> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_markers(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            out.append(cur); cur = {}; i += 1; continue
        if t in REGS:
            cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


def find_l15_block(probe):
    for phys, blk in enumerate(probe.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        nh = getattr(attn, "num_heads", None)
        if nh is not None and nh >= 15:
            return phys
    return None


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    wpos = int(sys.argv[3]); tpos = int(sys.argv[4])
    ms = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    pc_marker = sm[lev].get("PC")
    print(f"id{pid} {desc} lev_step={lev} q@PC={pc_marker} winner={wpos} target={tpos}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)
    l15_blk = find_l15_block(probe)
    in_blk = l15_blk - 1
    resid_in = probe.model.forward(padded, stop_after_block=in_blk)[0]

    blk = probe.model.blocks[l15_blk]
    attn = blk.attn
    Wq = attn.W_q; Wk = attn.W_k
    H = attn.num_heads
    D = resid_in.shape[-1]
    HEAD = 14
    if Wq.shape[1] == D:
        hd = Wq.shape[0] // H
        q_all = resid_in[pc_marker] @ Wq.t()
        kw = lambda pos: resid_in[pos] @ Wk.t()
    else:
        hd = Wq.shape[1] // H
        q_all = resid_in[pc_marker] @ Wq
        kw = lambda pos: resid_in[pos] @ Wk
    qh = q_all[HEAD*hd:(HEAD+1)*hd]
    kw_win = kw(wpos)[HEAD*hd:(HEAD+1)*hd]
    kw_tgt = kw(tpos)[HEAD*hd:(HEAD+1)*hd]
    print(f"head_dim={hd}")
    print(f"  TOTAL: winner={float((qh*kw_win).sum()):.1f}  target={float((qh*kw_tgt).sum()):.1f}  "
          f"deficit(tgt-win)={float((qh*kw_tgt).sum()-(qh*kw_win).sum()):.1f}")
    print("\n  slot  q_h       k_win    k_tgt    win_prod   tgt_prod   tgt-win")
    cum = 0.0
    rows = []
    for s in range(hd):
        wp = float(qh[s]*kw_win[s]); tp = float(qh[s]*kw_tgt[s])
        d = tp - wp; cum += d
        rows.append((abs(d), s, float(qh[s]), float(kw_win[s]), float(kw_tgt[s]), wp, tp, d, cum))
    # print the slots that matter most (largest |deficit contribution|)
    for (ad, s, q, kw_, kt, wp, tp, d, cum) in sorted(rows, reverse=True)[:30]:
        print(f"  {s:>4}  {q:>8.2f}  {kw_:>7.2f}  {kt:>7.2f}  {wp:>9.1f}  {tp:>9.1f}  {d:>+8.1f}")


if __name__ == "__main__":
    main()
