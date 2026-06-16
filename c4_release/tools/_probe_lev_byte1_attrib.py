#!/usr/bin/env python3
"""Attribute the LEV-step AX byte-1 emission logit to residual dims.

At the LEV step's AX byte-1 predictor row, compute per-dim contribution to
(logit[got] - logit[0]) where got is the leaked byte-1 token (e.g. 6). Finds the
exact residual dims the LM head reads to emit the spurious byte-1.

Usage: python tools/_probe_lev_byte1_attrib.py <id> <lev_step> [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from src.compiler import compile_c
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
SE = int(Token.STEP_END)
_BUILT_ITEMS = None  # set in main() from the BUILT dim_positions
REGS = {int(Token.REG_PC): "PC", int(Token.REG_AX): "AX", int(Token.REG_SP): "SP",
        int(Token.REG_BP): "BP", 268: "STACK0", 261: "MEM"}


def name_for(pos):
    if _BUILT_ITEMS is None:
        return f"dim{pos}"
    for i, (name, start) in enumerate(_BUILT_ITEMS):
        nxt = _BUILT_ITEMS[i+1][1] if i+1 < len(_BUILT_ITEMS) else start + 16
        if start <= pos < nxt:
            return f"{name}+{pos - start}"
    return f"dim{pos}"


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


@torch.no_grad()
def main():
    pid = int(sys.argv[1]); lev = int(sys.argv[2])
    ms = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
    global _BUILT_ITEMS
    _BUILT_ITEMS = sorted(probe.model.embed._dim_positions.items(),
                          key=lambda kv: kv[1])
    model = probe.model
    W = model.head.weight
    if W.is_sparse: W = W.to_dense()
    b = model.head.bias
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    sm = step_markers(ctx, pl)
    ax = sm[lev].get("AX")
    sect = [ctx[ax+j] & 0xFF for j in range(5)]
    print(f"id{pid} {desc} exp={exp} lev_step={lev} AX_section={sect}")
    got = sect[2]  # byte1 emitted token
    pos = ax + 1   # byte0 token row predicts byte1
    last_block = len(model.blocks) - 1
    padded = torch.tensor([ctx], dtype=torch.long, device=probe._device)
    res = model.forward(padded, stop_after_block=last_block)[0, pos]
    try:
        res = res.to_dense()
    except Exception:
        pass
    res = res.to(W.device).float().reshape(-1)
    want = 0  # byte1 should be 0
    Wg = W[got].to_dense() if W[got].layout != torch.strided else W[got]
    Ww = W[want].to_dense() if W[want].layout != torch.strided else W[want]
    dw = (Wg - Ww) * res
    logit_got = float((Wg * res).sum() + b[got])
    logit_want = float((Ww * res).sum() + b[want])
    print(f"byte1 predictor pos={pos} tok={ctx[pos]&0xFF} got_byte1={got}")
    print(f"  logit[{got}]={logit_got:.3f}  logit[0]={logit_want:.3f}  diff={logit_got-logit_want:.3f}")
    order = torch.argsort(dw.abs(), descending=True)
    print(f"  top dims driving (logit[{got}] - logit[0]):")
    for di in order[:22].tolist():
        if abs(float(dw[di])) < 0.05: break
        print(f"    dim {di:4d} {name_for(di):28s} res={float(res[di]):9.3f} "
              f"dW={float(Wg[di]-Ww[di]):7.3f} contrib={float(dw[di]):8.3f}")


if __name__ == "__main__":
    main()
