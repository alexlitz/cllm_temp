#!/usr/bin/env python3
"""Re-score the head-14 LEV CAM gather under candidate address-encoding levers,
WITHOUT rebuilding the model. Reads the L15-input residual for func_identity_0
(or any func/nested id), recomputes the head-14 Q@K score for every candidate
store row under several hypothetical Q/K modifications, and reports which store
wins under each. Lets us validate an address-widening lever in seconds before
baking it.

Levers tested:
  base        : the head as-built.
  qlo_sharp K : set the query's ADDR_B0_LO one-hot to a hard one-hot at its
                argmax with peak K (sharpen the weak BP+8 byte-0 lo-nibble).
  b0_scale M  : multiply the head's ADDR_B0 binary-address slots (4..11) by M
                (sharper byte-0 address discrimination).
  memstore W  : add W * MEM_STORE to the K side (anchor real stores over stale).
  combos.

Usage: CUDA_VISIBLE_DEVICES=1 python tools/_probe_lev_rescore.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys, math
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
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


def find_l15_block(probe):
    for phys, blk in enumerate(probe.model.blocks):
        attn = getattr(blk, "attn", None)
        if attn is None: continue
        if getattr(attn, "num_heads", 0) >= 15: return phys
    return None


def onehot_byte(r, dp, lo, hi):
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
    sm = step_markers(ctx, pl)
    pcm = sm[lev].get("PC")
    l15 = find_l15_block(probe)
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    resid = probe.model.forward(padded, stop_after_block=l15 - 1)[0].float()
    attn = probe.model.blocks[l15].attn
    D = resid.shape[-1]; H = attn.num_heads
    HEAD = 14
    def _dense(W):
        if W.is_sparse or W.layout != torch.strided:
            W = W.to_dense()
        return W.float()
    Wq = _dense(attn.W_q); Wk = _dense(attn.W_k)
    qhd = (Wq.shape[0] // H) if Wq.shape[1] == D else (Wq.shape[1] // H)

    # candidate store rows (byte-0 rows of STACK0/MEM markers)
    cand = []
    i = pl
    while i < len(ctx):
        t = ctx[i]; nm = REGS.get(t)
        if nm in ("STACK0", "MEM"):
            pos = i + 1
            if pos < len(ctx):
                b0 = onehot_byte(resid[pos], dp, "ADDR_B0_LO", "ADDR_B0_HI")
                b1 = onehot_byte(resid[pos], dp, "ADDR_B1_LO", "ADDR_B1_HI")
                b2 = onehot_byte(resid[pos], dp, "ADDR_B2_LO", "ADDR_B2_HI")
                addr = b0 | (b1 << 8) | (b2 << 16)
                cev = onehot_byte(resid[pos], dp, "CLEAN_EMBED_LO", "CLEAN_EMBED_HI")
                msr = float(resid[pos][dp["MEM_STORE"]])
                cand.append((pos, nm, addr, cev, msr))
            i += 5; continue
        i += 1

    # build the head-14 Q/K weight rows from the BUILT attn (slice the head).
    def head_slice(W):
        if W.shape[1] == D:  # [H*hd, D]
            return W[HEAD*qhd:(HEAD+1)*qhd, :]  # [hd, D]
        else:                # [D, H*hd]
            return W[:, HEAD*qhd:(HEAD+1)*qhd].t()  # [hd, D]
    Wq_h = head_slice(Wq).clone()  # [hd, D]
    Wk_h = head_slice(Wk).clone()  # [hd, D]

    # dim indices for ADDR_B0 binary slots are head-dim slots 4..11; but those
    # are ROWS of the head weight that read ADDR_B0_LO/HI. We approximate the
    # lever by scaling the head-dim rows 4..11 (the ADDR_B0 bit slots).
    def score(qrow, Wq_use, Wk_use):
        qv = Wq_use @ qrow  # [hd]
        out = []
        for (pos, nm, addr, cev, msr) in cand:
            kv = Wk_use @ resid[pos]  # [hd]
            out.append((pos, nm, addr, cev, float((qv * kv).sum())))
        return out

    TARGET = {550: 268, 575: None, 950: None}.get(pid)

    def report(label, sc):
        sc2 = sorted(sc, key=lambda x: -x[-1])
        win = sc2[0]
        tgt = [s for s in sc if s[0] == TARGET]
        tnote = ""
        if tgt:
            t = tgt[0]; rank = [s[0] for s in sc2].index(TARGET)
            tnote = f"  TARGET@{TARGET} val=0x{t[3]:02x} score={t[-1]:.0f} rank={rank}"
        print(f"  [{label}] winner=pos{win[0]}({win[1]}) addr=0x{win[2]:06x} val=0x{win[3]:02x} score={win[-1]:.0f}{tnote}")

    q = resid[pcm]
    print(f"id{pid} {desc} lev={lev} q@{pcm} L15={l15} TARGET={TARGET}")
    report("base", score(q, Wq_h, Wk_h))

    if "--full" in sys.argv:
        Wq8 = Wq_h.clone(); Wk8 = Wk_h.clone(); Wq8[4:12, :] *= 8.0; Wk8[4:12, :] *= 8.0
        sc = sorted(score(q, Wq8, Wk8), key=lambda x: -x[-1])
        print("  full table @ b0_scale=8:")
        for (pos, nm, addr, cev, s) in sc:
            mark = " <== TARGET" if pos == TARGET else ""
            print(f"    pos{pos:<4} {nm:<7} 0x{addr:06x} val0x{cev:02x} score={s:.0f}{mark}")
        return

    # Lever: sharpen query ADDR_B0_LO to a hard one-hot at its argmax, peak K.
    for K in (2.0, 4.0, 8.0):
        qmod = q.clone()
        lo = dp["ADDR_B0_LO"]; seg = qmod[lo:lo+16]
        arg = int(seg.argmax()); qmod[lo:lo+16] = 0.0; qmod[lo+arg] = K
        report(f"qlo_sharp K={K}", score(qmod, Wq_h, Wk_h))

    # Lever: scale head ADDR_B0 bit slots (head-dim rows 4..11) by M.
    for M in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        Wq2 = Wq_h.clone(); Wk2 = Wk_h.clone()
        Wq2[4:12, :] *= M; Wk2[4:12, :] *= M
        sc = score(q, Wq2, Wk2)
        report(f"b0_scale M={M}", sc)
        if TARGET is not None:
            sc2 = sorted(sc, key=lambda x: -x[-1])
            if sc2[0][0] == TARGET:
                margin = sc2[0][-1] - sc2[1][-1]
                print(f"        -> TARGET wins; margin over 2nd(pos{sc2[1][0]} addr0x{sc2[1][2]:06x})={margin:.0f}")

    # Lever: combo sharpen query + scale slots.
    for K, M in ((4.0, 4.0), (8.0, 8.0)):
        qmod = q.clone(); lo = dp["ADDR_B0_LO"]; seg = qmod[lo:lo+16]
        arg = int(seg.argmax()); qmod[lo:lo+16] = 0.0; qmod[lo+arg] = K
        Wq2 = Wq_h.clone(); Wk2 = Wk_h.clone(); Wq2[4:12, :] *= M; Wk2[4:12, :] *= M
        report(f"combo K={K} M={M}", score(qmod, Wq2, Wk2))


if __name__ == "__main__":
    main()
