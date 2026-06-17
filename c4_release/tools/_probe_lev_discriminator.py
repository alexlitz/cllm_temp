#!/usr/bin/env python3
"""Hunt for a VALUE-INDEPENDENT discriminator between the genuine LEV return
store and a stale same-address re-emission. Dumps, for two store rows (and
optionally a control set), every named dim category where the two rows differ
materially -- so we can find a store-recency / per-step / byte-offset signal
that uniquely marks the genuine return store and that the LEV query could match.

Usage: CUDA_VISIBLE_DEVICES=0 python tools/_probe_lev_discriminator.py <id> <lev_step> <genuine_pos> <stale_pos> [maxsteps]
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


def smk(ctx, pl):
    out = []; i = pl; cur = {}
    while i < len(ctx):
        t = ctx[i]
        if t == SE: out.append(cur); cur = {}; i += 1; continue
        if t in REGS: cur.setdefault(REGS[t], i); i += 5; continue
        i += 1
    if cur: out.append(cur)
    return out


# value-carrying dims to EXCLUDE from the discriminator hunt (circular).
VALUE_DIMS = {"CLEAN_EMBED_LO", "CLEAN_EMBED_HI", "EMBED_LO", "EMBED_HI",
              "OUTPUT_LO", "OUTPUT_HI", "OUTPUT_BYTE_LO", "OUTPUT_BYTE_HI",
              "MEM_VAL_B0", "MEM_VAL_B1", "MEM_VAL_B2", "MEM_VAL_B3"}


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    gpos = int(sys.argv[3]); spos = int(sys.argv[4])
    ms = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = smk(ctx, pl)
    pcm = sm[lev].get("PC")
    l15 = [ph for ph, b in enumerate(probe.model.blocks)
           if getattr(getattr(b, "attn", None), "num_heads", 0) >= 15][0]
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    resid = probe.model.forward(padded, stop_after_block=l15 - 1)[0].float()
    rg = resid[gpos]; rs = resid[spos]; rq = resid[pcm]
    # Build a per-dim name map from dp (category -> base). Categories are usually
    # 16-wide nibble one-hots or single flags; treat each named base + offset.
    names = sorted(dp.items(), key=lambda kv: kv[1])
    print(f"id{pid} {desc} lev={lev} genuine@{gpos} stale@{spos} query@{pcm}")
    print(f"  Dims where genuine and stale differ by >=0.3 (|g-s|), excluding value dims:")
    # walk every dim, attribute to the nearest named base.
    base_for = {}
    cats = list(dp.items())
    for nm, base in cats:
        for off in range(16):
            base_for.setdefault(base + off, (nm, off))
    diffs = []
    D = resid.shape[-1]
    for d in range(D):
        g = float(rg[d]); s = float(rs[d]); q = float(rq[d])
        if abs(g - s) < 0.3:
            continue
        nm, off = base_for.get(d, ("?", 0))
        if nm in VALUE_DIMS:
            continue
        diffs.append((abs(g - s), d, nm, off, g, s, q))
    for (ad, d, nm, off, g, s, q) in sorted(diffs, reverse=True)[:40]:
        flag = ""
        # a USABLE discriminator is high at genuine, low at stale, AND non-zero
        # at the query (so the query can match it).
        if abs(g) > abs(s) and abs(q) > 0.2:
            flag = "  <== query-matchable, genuine>stale"
        elif abs(g) > abs(s):
            flag = "  (genuine>stale, query=0)"
        print(f"    dim{d:4d} {nm}+{off:<2} g={g:+.2f} s={s:+.2f} q={q:+.2f}{flag}")


if __name__ == "__main__":
    main()
