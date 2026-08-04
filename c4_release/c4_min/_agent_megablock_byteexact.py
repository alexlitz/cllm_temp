#!/usr/bin/env python3
"""_agent_megablock_byteexact.py — DECODE-level byte-exactness gate for the fused
megablock, on REAL frame streams (the task's "per-step AX trace vs reference").

For each DIV-free opcode we build a real per-step re-embed stream (BOS + register
frames), run the FULL block-skip live set (the driver's actual per-step forward)
in two ways:
  (A) EAGER: the unmodified StepBlockSkipRunner.forward (the golden path).
  (B) MEGA:  block 0 eager + the DIV-free [cut=1,N) dead-FFN chain fused into the
             megakernel + the 3 live CAM blocks eager.
and decode every register (AX, PC, SP, BP, STK) nibble band of the query row from
BOTH.  Byte-exact iff every decoded register integer matches (the nibble-argmax
re-quantiser is residue-immune, so fp-accum-order residue below the decode margin
is byte-identical).  This is the authoritative gate the golden decode uses.
"""
from __future__ import annotations
import os
os.environ.setdefault("C4_PF_CFM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
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
    dev = "cuda:0"
    torch.cuda.set_device(0)
    model, L, _ = build_compact_sparse_streaming(code_size=48,
                                                 compute_mode="dense_kernel")
    model.to(dev); model.materialize_dense(dev)
    install_block_sparse_ffn(model, mode="coo", verbose=False)
    model.to(dev)
    install_live_head_attention(model, verbose=False)
    install_dead_block_fusion(model, verbose=False)
    return model, L, dev


def reg_bases(L):
    """Resolve the register nibble-band base dims for AX/PC/SP/BP/STK."""
    dp = L.dim_positions if hasattr(L, "dim_positions") else None
    out = {}
    for nm, key in [("AX", "AX"), ("PC", "REG_PC"), ("SP", "REG_SP"),
                    ("BP", "REG_BP"), ("STK", "STACK0")]:
        for cand in (key, nm):
            b = getattr(L, f"{cand.lower()}_base", None)
            if b is not None:
                out[nm] = b; break
    return out


def make_stream(L, S_tokens, seed_mem=None):
    seed_frames, store_log = _seed_frames(seed_mem or {})
    stream = [pfc.V.BOS] + seed_frames + _build_frame(0, 0, SP_INIT, SP_INIT, 0)
    sp = SP_INIT; pc = 1
    while len(stream) < S_tokens:
        sp -= 4
        stream += _build_frame(pc, pc * 3 + 7, sp, SP_INIT, pc, mem_addr=sp,
                               mem_val=(pc & 0xFF))
        store_log[len(store_log) + 1] = (sp, pc & 0xFF)
        pc += 1
    return stream, store_log


def divfree_ops(live_index):
    return [op for op in live_index
            if op is not None and op not in (isa.DIV, isa.MOD)]


def build_mega_items(model, dev, region, live, block_k):
    live_set = set(live); items = []; seg = []
    for b in region:
        if b in live_set:
            if seg:
                items.append(("mega", MegaBlockChain(model, dev, seg, block_k))); seg = []
            items.append(("live", b))
        else:
            seg.append(b)
    if seg:
        items.append(("mega", MegaBlockChain(model, dev, seg, block_k)))
    return items


def main():
    model, L, dev = build()
    runner = StepBlockSkipRunner(model, L)
    live_index = build_live_index(model, L)
    ops = divfree_ops(live_index)

    # decode bands (authoritative: L.AX is the emitted register the driver decodes)
    bases = {"AX": L.AX, "PC": L.PC, "SP": L.SP, "BP": L.BP, "STK": L.STACK0}
    print(f"decode bands resolved: {bases}", flush=True)

    S_tokens = 91
    stream, store_log = make_stream(L, S_tokens)
    code = isa.assemble([("IMM", 0), ("HALT", 0)])
    overlay = make_overlay_complete(code, L, store_log=store_log)
    toks = torch.tensor([stream], device=dev)
    S = len(stream)

    # DIV-free full-region (for the mega composition): union over ops, [cut=1,N)
    union = set()
    for op in ops:
        union.update(live_index[op])
    region = sorted(b for b in union if b >= 1)
    live = [b for b in region
            if not getattr(model.blocks[b].attn, "_dead_block_fused", False)]
    items = build_mega_items(model, dev, region, live, block_k=64)

    print(f"\n{'op':6s} {'eager AX/PC/SP/BP/STK':>34s}  {'mega AX/PC/SP/BP/STK':>34s}  match",
          flush=True)
    n_ok = n_tot = 0
    max_band_linf = 0.0
    for op in sorted(ops):
        nm = isa.NAMES.get(op, str(op))
        with torch.no_grad():
            x0 = model.embed[toks].clone(); overlay(x0)
            # EAGER: the golden per-step block-skip forward for this op
            eager_out = runner.forward(x0.clone(), op)
            # MEGA: run this op's live blocks, but the dead-FFN [cut,N) ones fused.
            # We reproduce the per-op forward: block 0 (if live) + op-live blocks,
            # with the dead ones routed through the mega chain when contiguous.
            op_live = set(live_index[op])
            # block 0 (below cut): eager
            h = x0.clone()
            qp = torch.arange(S, device=dev, dtype=torch.long)
            if 0 in op_live:
                h, _ = model.blocks[0](h, past_kv=None, q_positions=qp, use_cache=True)
            # blocks [1,N) that are live for THIS op, dead ones via mega chain
            cur_seg = []
            def flush_seg(hh):
                nonlocal cur_seg
                if cur_seg:
                    ch = MegaBlockChain(model, dev, cur_seg, 64)
                    hh = ch.run(hh)
                    cur_seg = []
                return hh
            for b in sorted(x for x in op_live if x >= 1):
                if getattr(model.blocks[b].attn, "_dead_block_fused", False):
                    cur_seg.append(b)
                else:
                    h = flush_seg(h)
                    h, _ = model.blocks[b](h, past_kv=None, q_positions=qp,
                                           use_cache=True)
            h = flush_seg(h)
            mega_out = h
        # decode the query row (last row)
        er = eager_out[0, -1]; mr = mega_out[0, -1]
        # band linf over the whole residual query row
        band_linf = (er - mr).abs().max().item()
        max_band_linf = max(max_band_linf, band_linf)
        evals = {}; mvals = {}
        for rn, rb in bases.items():
            evals[rn] = _decode_reg_from_nibbles(er, L, rb)
            mvals[rn] = _decode_reg_from_nibbles(mr, L, rb)
        match = (evals == mvals)
        n_tot += 1; n_ok += int(match)
        es = "/".join(str(evals.get(k, "-")) for k in ("AX", "PC", "SP", "BP", "STK"))
        ms = "/".join(str(mvals.get(k, "-")) for k in ("AX", "PC", "SP", "BP", "STK"))
        print(f"{nm:6s} {es:>34s}  {ms:>34s}  {'OK' if match else 'MISMATCH'}",
              flush=True)
    print(f"\nDECODE byte-exact: {n_ok}/{n_tot} ops match  "
          f"(max residual-row Linf={max_band_linf:.3e})", flush=True)


if __name__ == "__main__":
    main()
