#!/usr/bin/env python3
"""LEV CAM-aliasing probe (spec_k=0, BUILT dims).

For a func_* program at the LEV step's PC-marker row, this dumps the
content-addressable-memory disambiguation surface that L15 head 14 sees:

  * the LEV query key = the ADDR_B0/B1/B2 (LO+HI nibble one-hots) read at the
    PC-marker row after the L9 BP+8 relay (the address the lookup matches on)
  * EVERY store-ish token in context (MARK_STACK0 byte rows + MARK_MEM rows),
    decoding each token's K-side ADDR_B0/B1/B2 key, its CLEAN_EMBED value, its
    MEM_STORE tag, and the per-byte-0 byte-selection signature
  * the head-14 attention SCORE each candidate store receives (recomputed from
    the built W_q/W_k at the L15 block) and the resulting softmax weight, so we
    can see WHICH store head 14 picks and whether it aliases
  * the OUTPUT byte-0 the L15 block delivers at the PC row (the restored PC)

Usage: python tools/_probe_lev_cam.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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


def onehot_byte(resid, dp, lo_name, hi_name):
    lo = resid[dp[lo_name]:dp[lo_name]+16]
    hi = resid[dp[hi_name]:dp[hi_name]+16]
    lo_i = int(lo.argmax()); hi_i = int(hi.argmax())
    return (hi_i << 4) | lo_i, float(lo[lo_i]), float(hi[hi_i])


def find_l15_block(probe):
    """Return the physical block index whose attn has the L15 lookup head 14."""
    for phys, blk in enumerate(probe.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None:
            continue
        nh = getattr(attn, "num_heads", None)
        # L15 memory_lookup block is the one with the widest head count (>=15
        # when the flag is on) AND a W_q/W_k. The resize grows it to 15.
        if nh is not None and nh >= 15:
            return phys
    return None


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
    sm = step_markers(ctx, pl)
    if lev >= len(sm):
        print(f"lev_step {lev} out of range (nsteps={len(sm)})"); return
    pc_marker = sm[lev].get("PC")
    print(f"id{pid} {desc} exp={exp} lev_step={lev} PC_marker_idx={pc_marker} nblk={len(probe.model.blocks)}")
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    l15_blk = find_l15_block(probe)
    print(f"L15 lookup block (num_heads>=15) = physical {l15_blk}")
    # Residual fed INTO the L15 block = output of block l15_blk-1.
    in_blk = l15_blk - 1 if l15_blk else nblk - 2
    resid_in = probe.model.forward(padded, stop_after_block=in_blk)[0]  # [S, D]

    # --- 1. The LEV query key at the PC-marker row (post-L9 relay). ---
    print("\n== query address @ PC-marker row across blocks (the CAM key) ==")
    for blk in [9, 10, 11, 12, 13, in_blk]:
        if blk >= nblk: continue
        r = probe.model.forward(padded, stop_after_block=blk)[0][pc_marker]
        b0, b0lo, b0hi = onehot_byte(r, dp, "ADDR_B0_LO", "ADDR_B0_HI")
        b1, b1lo, b1hi = onehot_byte(r, dp, "ADDR_B1_LO", "ADDR_B1_HI")
        b2, _, _ = onehot_byte(r, dp, "ADDR_B2_LO", "ADDR_B2_HI")
        addr = b0 | (b1 << 8) | (b2 << 16)
        print(f"  blk{blk:2d}: ADDR=0x{addr:06x}({addr})  "
              f"b0=0x{b0:02x}(m{b0lo:.1f}/{b0hi:.1f}) b1=0x{b1:02x}(m{b1lo:.1f}/{b1hi:.1f}) b2=0x{b2:02x}")

    # --- 2. Enumerate store tokens + their K-side keys / values. ---
    # Store-ish K rows: STACK0 byte-0 rows (MARK_STACK0) and MEM rows (MARK_MEM).
    print("\n== candidate store rows (K keys @ resid into L15) ==")
    cand = []
    i = pl
    while i < len(ctx):
        t = ctx[i]
        nm = REGS.get(t)
        if nm in ("STACK0", "MEM"):
            # byte-0 row is the marker+1 row (the materialized byte-0 token).
            for boff in range(1, 5):
                pos = i + boff
                if pos >= len(ctx):
                    break
                r = resid_in[pos]
                b0, b0lo, b0hi = onehot_byte(r, dp, "ADDR_B0_LO", "ADDR_B0_HI")
                b1, _, _ = onehot_byte(r, dp, "ADDR_B1_LO", "ADDR_B1_HI")
                b2, _, _ = onehot_byte(r, dp, "ADDR_B2_LO", "ADDR_B2_HI")
                cev, _, _ = onehot_byte(r, dp, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
                mem_store = float(r[dp["MEM_STORE"]]) if "MEM_STORE" in dp else float("nan")
                addr = b0 | (b1 << 8) | (b2 << 16)
                stp = sum(1 for m in sm if m.get(nm, 1 << 30) <= i) - 1
                if boff == 1:  # only byte-0 row is the CAM key carrier
                    cand.append((pos, nm, stp, addr, cev, mem_store))
                    print(f"  pos={pos} {nm} step~{stp} byte{boff-1}: "
                          f"K_addr=0x{addr:06x}  CLEAN_EMBED=0x{cev:02x}({cev})  "
                          f"MEM_STORE={mem_store:.2f}")
            i += 5; continue
        i += 1

    # --- 3. Head-14 attention scores: recompute Q@K for the PC-marker query. ---
    blk = probe.model.blocks[l15_blk] if l15_blk is not None else None
    attn = getattr(blk, "attn", None) if blk is not None else None
    if attn is None or not hasattr(attn, "W_q"):
        print("\n(no W_q on L15 attn; cannot score head 14)"); return
    Wq = attn.W_q; Wk = attn.W_k  # [H*hd, D] or [D, H*hd]?
    H = attn.num_heads
    D = resid_in.shape[-1]
    # infer head_dim + orientation
    if Wq.shape[1] == D:
        hd = Wq.shape[0] // H
        q_all = resid_in[pc_marker] @ Wq.t()  # [H*hd]
    else:
        hd = Wq.shape[1] // H
        q_all = resid_in[pc_marker] @ Wq      # [H*hd]
    HEAD = 14
    qh = q_all[HEAD*hd:(HEAD+1)*hd]
    print(f"\n== head 14 scores (hd={hd}, H={H}) ==")
    scores = []
    for (pos, nm, stp, addr, cev, ms_tag) in cand:
        if Wk.shape[1] == D:
            kh = (resid_in[pos] @ Wk.t())[HEAD*hd:(HEAD+1)*hd]
        else:
            kh = (resid_in[pos] @ Wk)[HEAD*hd:(HEAD+1)*hd]
        s = float((qh * kh).sum())
        scores.append((pos, nm, stp, addr, cev, s))
    # softmax over the candidate scores (approximate; real attn also has sink)
    import math
    if scores:
        mx = max(s for *_, s in scores)
        exps = [math.exp(s - mx) for *_, s in scores]
        Z = sum(exps)
        print("  pos   tok        step addr      val  score      softmax")
        for (pos, nm, stp, addr, cev, s), e in zip(scores, exps):
            print(f"  {pos:<5} {nm:<10} {stp:>3}  0x{addr:06x} 0x{cev:02x}  {s:>9.1f}  {e/Z:.4f}")
        best = max(range(len(scores)), key=lambda j: scores[j][-1])
        bp = scores[best]
        print(f"  -> head14 picks pos={bp[0]} ({bp[1]} step~{bp[2]}) val=0x{bp[4]:02x}={bp[4]}")

    # --- 4. OUTPUT byte-0 at PC row after L15 (the restored PC). ---
    print("\n== OUTPUT @ PC-marker row across late blocks (restored PC byte-0) ==")
    for blk_i in [l15_blk, l15_blk+1 if l15_blk else None, 32, 37, nblk-1]:
        if blk_i is None or blk_i >= nblk:
            continue
        r = probe.model.forward(padded, stop_after_block=blk_i)[0][pc_marker]
        out0, m0l, m0h = onehot_byte(r, dp, "OUTPUT_LO", "OUTPUT_HI")
        print(f"  blk{blk_i:2d}: OUTPUT_b0=0x{out0:02x}({out0}) max {m0l:.1f}/{m0h:.1f}")

    # The actual emitted PC byte-0 token at this LEV step (next step's PC).
    nxt = sm[lev+1] if lev+1 < len(sm) else None
    if nxt and "PC" in nxt:
        pcm = nxt["PC"]; pcbytes = [ctx[pcm+1+j] & 0xFF for j in range(4)]
        print(f"\nNEXT step PC tokens = {pcbytes} = "
              f"0x{sum(b<<(8*j) for j,b in enumerate(pcbytes)):x}")


if __name__ == "__main__":
    main()
