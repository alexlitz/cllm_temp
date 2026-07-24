#!/usr/bin/env python3
"""Enumerate the FULL lib model's matmul-portion dims + nonzero structure.

Builds ``build_lib_model_streaming(code_size=32, recurrent_divmod=True,
addr32=True)`` and walks every block's attention projection (W_q/W_k/W_v/W_o),
FFN (W_up/W_gate/W_down), and the LM head — each a ``[out, in]`` linear = a matmul
``x[S,in] @ W.T[in,out]`` with ``dense MACs = S*in*out`` and ``COO inner iters =
S*nnz`` (S output rows each contracting the ``nnz`` nonzero weights).

Per the task, we take the WITHOUT-DIVMOD subset: DROP the ``alu-div-*`` block span
(the emulated matmul needs MUL, not the recurrent DIV megablocks).  Returns the
per-matmul records so ``ground_full_model_steps`` can run the paged COO kernel
for each and sum ACTUAL draft-VM steps.

``S`` = the per-step forward's query-row count.  The tiny model used its real
frame width (M=4 rows).  The full model's per-step forward is over the 30-token
emit frame, so ``S = 30`` by default (``--seq`` overrides).
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Dict


def enumerate_matmuls(seq: int = 30, code_size: int = 32,
                      drop_divmod: bool = True, verbose: bool = True):
    from c4_min.lib_neural import build_lib_model_streaming
    t0 = time.time()
    model, L, _stats = build_lib_model_streaming(
        code_size=code_size, recurrent_divmod=True, addr32=True)
    build_s = time.time() - t0
    names = L._block_names
    n_blocks = len(model.blocks)

    recs: List[Dict] = []
    dropped_blocks = 0
    for bi in range(n_blocks):
        name = names[bi] if bi < len(names) else f"blk{bi}"
        if drop_divmod and name.startswith("alu-div"):
            dropped_blocks += 1
            continue
        blk = model.blocks[bi]
        at = getattr(blk, "attn", None)
        ff = getattr(blk, "ffn", None)
        if at is not None:
            for wn in ("W_q", "W_k", "W_v", "W_o"):
                w = getattr(at, wn, None)
                if w is None:
                    continue
                recs.append(dict(block=bi, name=name, kind=f"attn.{wn}",
                                 K=w.in_dim, N=w.out_dim, S=seq, nnz=int(w.nnz),
                                 dense=seq * w.in_dim * w.out_dim,
                                 coo_iters=seq * int(w.nnz)))
        if ff is not None:
            for wn in ("W_up", "W_gate", "W_down"):
                w = getattr(ff, wn, None)
                if w is None:
                    continue
                recs.append(dict(block=bi, name=name, kind=f"ffn.{wn}",
                                 K=w.in_dim, N=w.out_dim, S=seq, nnz=int(w.nnz),
                                 dense=seq * w.in_dim * w.out_dim,
                                 coo_iters=seq * int(w.nnz)))
    # LM head (model.head): [vocab, d_model] linear
    head = getattr(model, "head", None)
    if head is not None:
        hw = getattr(head, "weight", None)
        if hw is not None:
            import torch
            nnz = int((hw != 0).sum().item())
            out_dim, in_dim = hw.shape
            recs.append(dict(block=n_blocks, name="lm-head", kind="head",
                             K=in_dim, N=out_dim, S=seq, nnz=nnz,
                             dense=seq * in_dim * out_dim, coo_iters=seq * nnz))

    dense_macs = sum(r["dense"] for r in recs)
    coo_iters = sum(r["coo_iters"] for r in recs)
    total_nnz = sum(r["nnz"] for r in recs)

    if verbose:
        print(f"FULL lib model: {n_blocks} blocks (built {build_s:.1f}s), "
              f"dropped {dropped_blocks} alu-div blocks")
        print(f"  {len(recs)} matmuls (attn proj + FFN + LM head), S(seq)={seq}")
        print(f"  DENSE total MACs = {dense_macs:,}")
        print(f"  total nnz (nonzero weights) = {total_nnz:,}")
        print(f"  COO inner iters (S*nnz summed) = {coo_iters:,}")
        # sparsity
        dense_w = sum(r["K"] * r["N"] for r in recs)
        print(f"  weight entries = {dense_w:,}, nonzero = {total_nnz:,} "
              f"({100 * (1 - total_nnz / dense_w):.4f}% weight-sparse)")
        # nnz-length histogram (how many distinct COO dot lengths we must run)
        lengths = sorted({r["nnz"] for r in recs if r["nnz"] > 0})
        print(f"  distinct nnz lengths to run: {len(lengths)} "
              f"(min={lengths[0] if lengths else 0}, max={lengths[-1] if lengths else 0})")

    return dict(recs=recs, n_blocks=n_blocks, dropped=dropped_blocks,
                dense_macs=dense_macs, coo_iters=coo_iters, total_nnz=total_nnz,
                build_s=build_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--keep-divmod", action="store_true")
    args = ap.parse_args()
    enumerate_matmuls(seq=args.seq, drop_divmod=not args.keep_divmod, verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
