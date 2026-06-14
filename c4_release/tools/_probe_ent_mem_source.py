#!/usr/bin/env python3
"""Trace the ENT-step MEM-value generation source for func_identity_0.

Strategy: the L14 mem_generation value heads (4-7) copy CLEAN_EMBED from the
position they attend. On ENT, slot 44 (l14_ops._layer14_mem_generation_head_specs_with_overrides)
attends the JSR-prologue old-BP source via K[OP_JSR]/K[H1+bp]/K[byte_idx].
This probe finds:
  (a) which context positions carry OP_JSR / OP_ENT (the candidate sources),
  (b) the CLEAN_EMBED bytes at those positions across blocks (is old_BP=[0,0,1,0] there?),
  (c) the MEM-value rows of the ENT step across blocks (where does garbage appear?).

spec_k=0, hook-free. Usage: python tools/_probe_ent_mem_source.py [id]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io
import torch
from tools.probe_groundtruth import build_groundtruth_probe  # noqa
from neural_vm.batched_pure_neural import Token  # noqa
from tests.test_suite_1000 import generate_test_programs  # noqa
from src.compiler import compile_c  # noqa

SE = int(Token.STEP_END)
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def step_rows(ctx, pl):
    """Return list-of-dict: per step, {marker_name: position} plus 'start','end'."""
    out = []
    i = pl; cur = {"start": i}
    while i < len(ctx):
        t = ctx[i]
        if t == SE:
            cur["end"] = i
            out.append(cur); cur = {"start": i + 1}; i += 1; continue
        if t in REGS:
            cur[REGS[t]] = i; i += 5; continue
        i += 1
    return out


def read_bytes(resid, base_pos, lo_dim, hi_dim):
    """Decode 4 bytes from CLEAN_EMBED/OUTPUT style LO/HI nibble bands at a row."""
    # Actually CLEAN_EMBED_LO[k] is one-hot over nibble value; decode argmax per byte? No.
    # For these probes we just read the raw 16 LO + 16 HI dims.
    lo = [round(float(resid[lo_dim + k]), 2) for k in range(16)]
    hi = [round(float(resid[hi_dim + k]), 2) for k in range(16)]
    return lo, hi


def nibble_decode(vals16):
    """argmax of a 16-dim one-hot nibble band -> nibble value, with confidence."""
    m = max(range(16), key=lambda k: vals16[k])
    return m, round(vals16[m], 2)


@torch.no_grad()
def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    dp = probe.model.embed._dim_positions
    ctx = probe._final_context(bc, max_steps=16)
    pl = len(probe._build_context(bc))
    steps = step_rows(ctx, pl)
    print(f"id{pid} {desc} exp={exp} ctxlen={len(ctx)} prompt_len={pl} nsteps={len(steps)}")

    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    nblk = len(probe.model.blocks)

    CE_LO = dp["CLEAN_EMBED_LO"]; CE_HI = dp["CLEAN_EMBED_HI"]
    OUT_LO = dp["OUTPUT_LO"]; OUT_HI = dp["OUTPUT_HI"]
    OP_JSR = dp["OP_JSR"]; OP_ENT = dp["OP_ENT"]
    MEM_STORE = dp["MEM_STORE"]
    MARK_MEM = dp["MARK_MEM"]
    MEM_VAL = [dp["MEM_VAL_B0"], dp["MEM_VAL_B1"], dp["MEM_VAL_B2"], dp["MEM_VAL_B3"]]

    # Identify the ENT step: identity body ENT is step 1 (main) and step 5 (callee).
    # We focus on step 1 (the main-frame ENT, where old_BP=65536 should be stored).
    ent_steps = []
    # Find steps where the MEM section is the saved-BP store. Use the marker-aware
    # decode: print MEM bytes per step (final residual / token level).
    print("\n== per-step MEM tokens (raw) ==")
    for s, st in enumerate(steps):
        mp = st.get("MEM")
        if mp is None:
            print(f"  step {s}: no MEM marker"); continue
        addr = sum((ctx[mp+1+j] & 0xFF) << (8*j) for j in range(4))
        val = sum((ctx[mp+5+j] & 0xFF) << (8*j) for j in range(4))
        valb = [ctx[mp+5+j] & 0xFF for j in range(4)]
        print(f"  step {s}: MEM@{mp} addr={addr} val={val} valbytes={valb}")

    # Print which positions carry OP_JSR / OP_ENT high at the final block (post-decode).
    resid_full = probe.model.forward(padded, stop_after_block=nblk - 1)[0]
    # opcode dims are durable; check at a mid block too (post L5 decode ~ block 8)
    print("\n== positions with OP_JSR / OP_ENT high (post-block 8 = L5 decode) ==")
    resid8 = probe.model.forward(padded, stop_after_block=min(8, nblk-1))[0]
    for i in range(pl, len(ctx)):
        j = float(resid8[i][OP_JSR]); e = float(resid8[i][OP_ENT])
        if abs(j) > 1.0 or abs(e) > 1.0:
            tok = ctx[i]; nm = REGS.get(tok, str(tok))
            print(f"  pos {i} tok={nm}: OP_JSR={j:.2f} OP_ENT={e:.2f}")

    # The ENT step we care about: choose the step whose oracle stores old_BP.
    # For identity, step 1 stores old_BP=65536 at addr 65520.
    target_steps = [s for s in range(len(steps)) if steps[s].get("MEM") is not None]
    # Focus on step 1 specifically.
    focus = 1
    print(f"\n== focus ENT step {focus}: MEM-value rows across blocks ==")
    st = steps[focus]
    mp = st.get("MEM")
    if mp is None:
        print("  no MEM marker at focus step"); return
    # The MEM value bytes are emitted at positions mp+5..mp+8 (after addr).
    # Their PREDICTOR rows (where L14 writes OUTPUT) are mp+4..mp+7 (the token
    # before each value byte). We inspect OUTPUT_LO/HI at the value-byte predictor rows.
    val_pred_rows = [mp + 4 + k for k in range(4)]  # predicts val byte k
    blocks = [13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28]
    blocks = [b for b in blocks if b < nblk]
    for k, vr in enumerate(val_pred_rows):
        print(f"\n  -- MEM val byte {k} predictor row pos={vr} (tok={REGS.get(ctx[vr], ctx[vr])}) --")
        print(f"     {'blk':>3} {'OUT_LO nib':>10} {'OUT_HI nib':>10}  (decoded byte)")
        for blk in blocks:
            resid = probe.model.forward(padded, stop_after_block=blk)[0]
            row = resid[vr]
            lo = [float(row[OUT_LO + j]) for j in range(16)]
            hi = [float(row[OUT_HI + j]) for j in range(16)]
            ln, lc = nibble_decode(lo)
            hn, hc = nibble_decode(hi)
            byteval = (hn << 4) | ln
            print(f"     {blk:>3} {ln:>2}({lc:>5}) {hn:>2}({hc:>5})  => 0x{byteval:02x}={byteval}")


if __name__ == "__main__":
    main()
