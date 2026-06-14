#!/usr/bin/env python3
"""Localize ALL cascade emitters of the post-ENT spurious leading token.

Builds the efficient-mode model (== smoke/probe path) ONCE, replays
func_identity_0 (id 550), and for each spurious post-SE leading row finds, per
physical block, the logit delta for the EMITTED spurious token. Reports every
block with a >threshold positive jump (the cascade members) and, for each, the
FFN units that drive the value-decode output dims, with their gate reads
labelled by BUILT dim_positions.

Usage: python tools/_probe_cascade_emitters.py [id] [maxsteps]
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["C4_SMOKE_SPEC_K"] = "0"; os.environ["C4_TEST_SPEC_K"] = "0"
_HERE = os.path.dirname(os.path.abspath(__file__)); _PKG = os.path.dirname(_HERE)
if _PKG not in sys.path: sys.path.insert(0, _PKG)
import contextlib, io, torch
from tools.probe_groundtruth import build_groundtruth_probe
from neural_vm.batched_pure_neural import Token
from tests.test_suite_1000 import generate_test_programs
from src.compiler import compile_c
from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic


def _d(t):
    try:
        if t.layout != torch.strided:
            return t.to_dense()
    except Exception:
        pass
    return t.detach()


def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 550
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    src, exp, desc = generate_test_programs()[pid]
    bc = compile_c(src)[0]
    with contextlib.redirect_stderr(io.StringIO()):
        probe = build_groundtruth_probe()
        _, layout = compile_full_vm_dynamic(disk_cache=True, alu_mode='efficient')
    inv = {}
    for k, v in layout.dim_positions.items():
        inv.setdefault(int(v), []).append(k)
    m = probe.model; dev = probe._device; SE = int(Token.STEP_END)
    ctx = probe._final_context(bc, max_steps=ms)
    pl = len(probe._build_context(bc))
    lead = [i - 1 for i in range(pl, len(ctx)) if ctx[i - 1] == SE and ctx[i] != 257]
    print(f"id{pid} {desc} blocks={len(m.blocks)} d_model={m.blocks[0].ffn.W_down.shape[0]}")
    print(f"post-SE leading spurious rows (row, emitted_tok): {[(r, ctx[r+1]) for r in lead]}")
    padded = torch.tensor([ctx], dtype=torch.long, device=dev)
    W = _d(m.head.weight)
    for r in lead:
        tok = ctx[r + 1]
        wT = W[tok]
        print(f"\n=== row {r} emits tok {tok} (0x{tok:02x}); per-block logit[{tok}] jumps ===")
        prev = 0.0
        jumps = []
        for blk in range(len(m.blocks)):
            rr = _d(m.forward(padded, stop_after_block=blk))[0, r]
            l = float((rr * wT).sum())
            d = l - prev
            if d > 500:
                print(f"  block {blk}: logit={l:.3e} delta={d:+.3e}")
                jumps.append(blk)
            prev = l
        # which output dims drive token `tok`? (head.weight[tok] nonzero)
        tdims = (wT.abs() > 1.0).nonzero().flatten().tolist()
        tdims = [d for d in tdims if d not in range(21, 32)]  # drop -80 baseline
        print(f"  token {tok} value-decode dims: {[(d, round(float(wT[d]),1), (inv.get(d,[''])[0])) for d in tdims]}")


if __name__ == "__main__":
    main()
