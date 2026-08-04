#!/usr/bin/env python3
"""_agent_megablock_bf16.py — lever 3: bf16/fp16 resident residual.

Measures the fp32 (byte-exact) vs bf16/fp16 (packed) resident residual for the
dead-FFN mega-chain: the exact per-op AX/PC/SP/BP/STK decode divergence it introduces
(so the user can decide the trade) AND the timing speedup from the halved residual
traffic.  Runs the SAME real-frame decode battery as _agent_megablock_byteexact.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time
import torch

from . import isa
from .compact_alloc import build_compact_sparse_streaming
from .step_block_skip import StepBlockSkipRunner, build_live_index
from .block_sparse_ffn import install_block_sparse_ffn
from .live_head_attention import install_live_head_attention, install_dead_block_fusion
from .fused_megablock import MegaBlockChain
from . import nibble_pure_forward_complete as pfc
from .nibble_pure_forward_complete import (make_overlay_complete, _build_frame,
                                           _seed_frames, SP_INIT,
                                           _decode_reg_from_nibbles)


def build():
    dev = "cuda:0"; torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


def make_stream(L, S_tokens):
    seed_frames, store_log = _seed_frames({})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    sp = SP_INIT; pc = 1
    while len(stream) < S_tokens:
        sp -= 4
        stream += _build_frame(pc, pc * 3 + 7, sp, SP_INIT, pc, mem_addr=sp,
                               mem_val=(pc & 0xFF))
        store_log[len(store_log) + 1] = (sp, pc & 0xFF)
        pc += 1
    return stream, store_log


def run_op_mega(model, L, dev, op, live_index, x0, S, dtype):
    """Run op's DIV-free live blocks with dead-FFN segments fused (given resid dtype)."""
    op_live = set(live_index[op])
    qp = torch.arange(S, device=dev, dtype=torch.long)
    h = x0.clone()
    if 0 in op_live:
        h, _ = model.blocks[0](h, past_kv=None, q_positions=qp, use_cache=True)
    seg = []
    def flush(hh):
        nonlocal seg
        if seg:
            ch = MegaBlockChain(model, dev, seg, 128, resid_dtype=dtype)
            hh = ch.run(hh); seg = []
        return hh
    for b in sorted(x for x in op_live if x >= 1):
        if getattr(model.blocks[b].attn, "_dead_block_fused", False):
            seg.append(b)
        else:
            h = flush(h)
            h, _ = model.blocks[b](h, past_kv=None, q_positions=qp, use_cache=True)
    return flush(h)


def main():
    model, L, dev = build()
    runner = StepBlockSkipRunner(model, L)
    live_index = build_live_index(model, L)
    ops = [op for op in live_index if op is not None and op not in (isa.DIV, isa.MOD)]
    bases = {"AX": L.AX, "PC": L.PC, "SP": L.SP, "BP": L.BP, "STK": L.STACK0}

    S_tokens = 91
    stream, store_log = make_stream(L, S_tokens)
    code = isa.assemble([("IMM", 0), ("HALT", 0)])
    overlay = make_overlay_complete(code, L, store_log=store_log)
    toks = torch.tensor([stream], device=dev); S = len(stream)

    for dtype, nm in [(torch.float32, "fp32"), (torch.bfloat16, "bf16"),
                      (torch.float16, "fp16")]:
        n_ok = n_tot = 0; max_linf = 0.0; mism = []
        for op in sorted(ops):
            with torch.no_grad():
                x0 = model.embed[toks].clone(); overlay(x0)
                ref = runner.forward(x0.clone(), op)
                got = run_op_mega(model, L, dev, op, live_index, x0, S, dtype)
            er = ref[0, -1]; mr = got[0, -1].float()
            max_linf = max(max_linf, (er - mr).abs().max().item())
            ev = {k: _decode_reg_from_nibbles(er, L, b) for k, b in bases.items()}
            mv = {k: _decode_reg_from_nibbles(mr, L, b) for k, b in bases.items()}
            n_tot += 1
            if ev == mv:
                n_ok += 1
            else:
                mism.append((isa.NAMES.get(op, str(op)), ev, mv))
        print(f"[{nm}] DECODE match {n_ok}/{n_tot}  max residual-row Linf={max_linf:.3e}",
              flush=True)
        for onm, ev, mv in mism[:6]:
            diffs = {k: (ev[k], mv[k]) for k in ev if ev[k] != mv[k]}
            print(f"    MISMATCH {onm}: {diffs}", flush=True)

    # timing: fp32 vs bf16 on the pure dead-FFN chain
    dead = sorted(b for b in set().union(*[live_index[o] for o in ops]) if b >= 1
                  and getattr(model.blocks[b].attn, "_dead_block_fused", False))
    print("\ntiming (pure dead-FFN chain, ONE graph):", flush=True)
    for dtype, nm in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        ch = MegaBlockChain(model, dev, dead, 128, resid_dtype=dtype)
        for K in (2048, 8192):
            hq = torch.randn(1, K, model.dim, device=dev) * 0.1
            ch.run_graphed(hq)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50):
                ch.run_graphed(hq)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 50 * 1e3
            print(f"    [{nm}] K={K}: {ms:.4f} ms  ({ms/K*1e3:.3f} us/step)", flush=True)


if __name__ == "__main__":
    main()
