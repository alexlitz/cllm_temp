#!/usr/bin/env python3
"""Find the L10 (phys block 11) FFN unit that materializes the wrong step-1
ENT AX byte0 leak for the function-call clusters (absdiff/func/nested).

Scans every L10 FFN unit's contribution to OUTPUT_LO/OUTPUT_HI at the step-1
AX-marker row and ranks by activation. Reports the top writers + their gate
input dims so the misfiring OP_ENT-gated rule can be identified.

Run: CUDA_VISIBLE_DEVICES=0 C4_SMOKE_SPEC_K=0 python tools/probe_l10_ent_ax_unit.py 1046
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("C4_SMOKE_SPEC_K", "0"); os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import torch
from tools.probe_groundtruth import GroundTruthProbe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c

PHYS = 11

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1046
    probe = GroundTruthProbe.build()
    m = probe.model
    dp = m.dim_positions
    inv = {}
    for k, v in dp.items():
        if int(v) not in inv and ".*." not in str(k):
            inv[int(v)] = k
    def lbl(d):
        for bd in range(d, max(-1, d - 40), -1):
            if bd in inv:
                return f"{inv[bd]}+{d-bd}" if d != bd else inv[bd]
        return str(d)

    tests = generate_test_programs()
    src, exp, desc = tests[idx]
    bc, _ = compile_c(src)
    print(f"id {idx} {desc} exp={exp}")

    ctx = probe._build_context(bc)
    emitted = []
    for _ in range(45):
        logits = probe._forward_logits(ctx)
        nxt = int(logits[len(ctx)-1].argmax().item())
        emitted.append(nxt); ctx.append(nxt)
    PC = int(Token.REG_PC); AX = int(Token.REG_AX)
    pl = len(ctx) - len(emitted)
    ax_emit = [i for i, t in enumerate(emitted) if t == AX]
    if len(ax_emit) < 2:
        print("not enough AX markers", ax_emit); return
    ax_row = pl + ax_emit[1]  # step-1 AX marker row
    print(f"step-1 AX-marker row = {ax_row}, next tok = {ctx[ax_row+1]} (expect 0)")

    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    # residual entering block 11
    x_in = m.forward(padded, stop_after_block=PHYS-1)[0, ax_row]
    if x_in.is_sparse: x_in = x_in.to_dense()
    x_in = x_in.float()

    ffn = m.blocks[PHYS].ffn
    Wg = ffn.W_gate.to_dense().float() if hasattr(ffn.W_gate, "to_dense") else ffn.W_gate.float()
    Wu = ffn.W_up.to_dense().float() if hasattr(ffn.W_up, "to_dense") else ffn.W_up.float()
    Wd = ffn.W_down.to_dense().float() if hasattr(ffn.W_down, "to_dense") else ffn.W_down.float()
    bg = ffn.b_gate.float() if getattr(ffn, "b_gate", None) is not None else torch.zeros(Wg.shape[0])
    bu = ffn.b_up.float() if getattr(ffn, "b_up", None) is not None else torch.zeros(Wu.shape[0])
    n_units = Wg.shape[0]
    print(f"L10 FFN units={n_units}")

    # SwiGLU: hidden = silu(Wu@x + bu) * (Wg@x + bg); out = Wd @ hidden
    g = (Wg @ x_in + bg)
    u = (Wu @ x_in + bu)
    act = torch.nn.functional.silu(u) * g  # per-unit hidden activation

    OLO = dp.get("OUTPUT_LO"); OHI = dp.get("OUTPUT_HI"); OHTS = dp.get("OUTPUT_HI_THIS_STEP")
    OLO = int(OLO); OHI = int(OHI)
    OHTS = int(OHTS) if OHTS is not None else None
    out_dims = list(range(OLO, OLO+16)) + list(range(OHI, OHI+16))
    if OHTS is not None: out_dims += list(range(OHTS, OHTS+16))

    # per-unit contribution to the OUTPUT band = act[u] * Wd[out_dim, u]
    rows = []
    for U in range(n_units):
        a = float(act[U])
        if abs(a) < 1e-3: continue
        contrib = {d: a * float(Wd[d, U]) for d in out_dims if abs(a*float(Wd[d,U])) > 0.5}
        if contrib:
            rows.append((U, a, contrib))
    rows.sort(key=lambda r: -max(abs(v) for v in r[2].values()))
    print(f"\n=== L10 units writing OUTPUT at step-1 AX row (top 15) ===")
    for U, a, contrib in rows[:15]:
        cstr = ", ".join(f"{lbl(d)}:{v:+.1f}" for d, v in sorted(contrib.items(), key=lambda kv:-abs(kv[1]))[:5])
        # which input dims drive the AND detector (W_up = conditions)
        uw = Wu[U]
        active_up = [(lbl(d), float(x_in[d]), float(uw[d])) for d in range(len(x_in))
                     if abs(float(x_in[d])) > 0.3 and abs(float(uw[d])) > 0.3]
        active_up.sort(key=lambda t: -abs(t[1]*t[2]))
        ustr = ", ".join(f"{n}(x={xv:.1f},w={wv:.1f})" for n, xv, wv in active_up[:8])
        gw = Wg[U]
        active_gate = [(lbl(d), float(x_in[d]), float(gw[d])) for d in range(len(x_in))
                       if abs(float(x_in[d])) > 0.3 and abs(float(gw[d])) > 0.3]
        active_gate.sort(key=lambda t: -abs(t[1]*t[2]))
        gstr = ", ".join(f"{n}(x={xv:.1f},w={wv:.1f})" for n, xv, wv in active_gate[:4])
        print(f"  U{U:4d} silu*g act={a:8.2f} up={float(u[U]):8.1f} g={float(g[U]):8.1f} | OUT[{cstr}]")
        print(f"          AND(up)<- {ustr}")
        print(f"          gate<- {gstr}")

if __name__ == "__main__":
    main()
