"""#751 — BAND-STRUCTURE analysis of the live-block FFN GEMMs (the band-grouped
GEMM feasibility probe).

The band-grouped GEMM plan claims each live block's FFN decomposes into a fixed set
of (input-band -> hidden-slice -> output-band) DENSE sub-blocks that can be grouped
across (bands x live-blocks x K rows) into a few big tensor-core GEMMs.  The STEP-1
verdict in ``block_sparse_ffn.py`` says the opposite: ~99.93% sparse, ~1 nonzero per
row, scattered -> dense sub-blocks are ~97-99% padding.

This probe MEASURES the reality on the ACTUAL built compact model, over the LIVE
blocks of a representative arith op (ADD) and the DIV/MOD megablock:

  1. per-live-block FFN nnz / density / rows-per-unit histogram (the STEP-1 claim).
  2. band-block density: partition W_up/W_gate/W_down rows+cols by the semantic
     residual bands (from L._names) and the hidden units, measure the mean fill of
     the ACTIVE (input-band x hidden-slice) tiles — is a band sub-block dense?
  3. tile-density curve at tile in {8,16,32} (the tensor-core MMA tile).
  4. the grouped-GEMM ceiling: how many distinct (in_dim, out_dim, Dff) FFN shapes
     are there across live blocks (can they batch into one bmm)? and the padding
     waste if we densify each band sub-block.

Read-only; builds the model but touches no golden weights.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_band_structure --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch

from . import isa
from .pf_kbatch import KBatchBoundedRunner
from .pf_speculative import draft_pf_program


def _dense(w):
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


def _tile_fill(mask: torch.Tensor, tile: int):
    """(active_tiles, total_tiles, mean_fill_of_active, nnz) for [out,in] mask."""
    m = (mask != 0)
    out, inn = m.shape
    pr = math.ceil(out / tile)
    pc = math.ceil(inn / tile)
    mp = torch.zeros(pr * tile, pc * tile, dtype=torch.bool, device=m.device)
    mp[:out, :inn] = m
    blocks = mp.view(pr, tile, pc, tile).permute(0, 2, 1, 3).reshape(pr * pc, tile * tile)
    per = blocks.sum(dim=1)
    active = int((per > 0).sum())
    nnz = int(m.sum())
    mean_fill = (nnz / active) if active else 0.0
    return active, pr * pc, mean_fill, nnz


def _live_ops(runner, code, max_steps=200):
    draft = draft_pf_program(code, max_steps=max_steps, mask=0xFFFFFFFF)
    cur_pc, ops = 0, []
    for f in draft.frames:
        op = code[cur_pc].op if 0 <= cur_pc < len(code) else None
        ops.append(op)
        cur_pc = f["pc"]
    return runner.live_union(ops)


def analyze_block_ffn(kb, L, names, bi, tiles=(8, 16, 32)):
    b = kb.b
    if b.is_passthrough and False:
        pass
    if b.routed or getattr(b.ffn, "W_up", None) is None:
        return None
    Wu = _dense(b.ffn.W_up).cpu()
    Wg = _dense(b.ffn.W_gate).cpu()
    Wd = _dense(b.ffn.W_down).cpu()
    D = Wu.shape[1]
    Dff = Wu.shape[0]
    rep = {"bi": bi, "name": names[bi] if bi < len(names) else str(bi),
           "D": D, "Dff": Dff, "passthrough": b.is_passthrough}
    # STEP-1: nnz + rows-per-unit histogram (how many residual dims each hidden
    # unit reads).
    for tag, W in (("up", Wu), ("gate", Wg), ("down", Wd)):
        m = (W != 0)
        nnz = int(m.sum())
        numel = W.numel()
        reads_per_row = m.sum(dim=1)          # nonzeros per output row
        active_rows = int((reads_per_row > 0).sum())
        med = int(reads_per_row[reads_per_row > 0].median()) if active_rows else 0
        mx = int(reads_per_row.max()) if nnz else 0
        rep[tag] = {
            "nnz": nnz, "density_pct": 100.0 * nnz / max(numel, 1),
            "active_rows": active_rows, "total_rows": W.shape[0],
            "median_reads_per_active_row": med, "max_reads_per_row": mx,
        }
        # tile fill at each tensor-core tile.
        rep[tag]["tiles"] = {}
        for t in tiles:
            a, tot, fill, _ = _tile_fill(W, t)
            rep[tag]["tiles"][t] = {
                "active": a, "total": tot,
                "mean_fill_of_active": round(fill, 2),
                "tile_fill_frac_pct": round(100.0 * fill / (t * t), 2),
                "block_flops_ratio_pct": round(100.0 * a * t * t / max(numel, 1), 2),
            }
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--code-size", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=18.0)
    ap.add_argument("--stable-s", type=float, default=60.0)
    args = ap.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    idx = int(device.split(":")[1]) if (device.startswith("cuda") and ":" in device) else 0
    if device.startswith("cuda") and not args.no_wait:
        from .bench_composed_fast_path import wait_for_gpu
        wait_for_gpu(idx, min_free_gb=args.min_free_gb, stable_s=args.stable_s)

    os.environ["C4_POS_SPARSE"] = "1"
    from .compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    if device != "cpu":
        model.to(device)
        model.materialize_dense(device)
    runner = KBatchBoundedRunner(model, L, window=args.window, selective_fp64=True)
    names = list(getattr(L, "_block_names", []))
    D = model.embed.shape[1]
    print(f"[built] blocks={len(model.blocks)} dim={D}", flush=True)

    add = isa.assemble([("IMM", 12), ("PSH", 0), ("IMM", 30), ("ADD", 0), ("HALT", 0)])
    div = isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 7), ("DIV", 0), ("HALT", 0)])
    live_add = sorted(_live_ops(runner, add))
    live_div = sorted(_live_ops(runner, div))
    print(f"[live] ADD -> {len(live_add)} live blocks; DIV -> {len(live_div)} live blocks",
          flush=True)

    for tag, live in (("ADD", live_add), ("DIV", live_div)):
        print(f"\n{'='*94}\n[{tag}] per-live-block FFN sparsity + tile fill "
              f"({len(live)} live blocks)\n{'='*94}", flush=True)
        # aggregate the whole live set to a summary + per-block for the first ~14.
        agg = {"nnz": 0, "numel": 0}
        shape_counter = Counter()
        tile16_active = 0
        tile16_total = 0
        tile16_nnz = 0
        n_pass = 0
        header = (f"  {'blk':>4} {'name':22s} {'Dff':>5} "
                  f"{'up_nnz':>7} {'up_dens%':>8} {'up_med':>6} {'up_max':>6} "
                  f"{'t16_fill%':>9} {'t16_flops%':>10}")
        print(header, flush=True)
        for bi in live:
            kb = runner.kblocks[bi]
            if kb.b.is_passthrough:
                n_pass += 1
            rep = analyze_block_ffn(kb, L, names, bi)
            if rep is None:
                continue
            shape_counter[(rep["D"], rep["Dff"])] += 1
            up = rep["up"]
            agg["nnz"] += up["nnz"]
            agg["numel"] += rep["D"] * rep["Dff"]
            t16 = up["tiles"][16]
            tile16_active += t16["active"]; tile16_total += t16["total"]
            tile16_nnz += up["nnz"]
            print(f"  {bi:>4} {rep['name'][:22]:22s} {rep['Dff']:>5} "
                  f"{up['nnz']:>7} {up['density_pct']:>8.4f} "
                  f"{up['median_reads_per_active_row']:>6} {up['max_reads_per_row']:>6} "
                  f"{t16['tile_fill_frac_pct']:>9.2f} {t16['block_flops_ratio_pct']:>10.2f}",
                  flush=True)
        dens = 100.0 * agg["nnz"] / max(agg["numel"], 1)
        t16_fill = (tile16_nnz / tile16_active) if tile16_active else 0.0
        print(f"\n  [{tag} summary] {len(live)} live ({n_pass} passthrough-attn); "
              f"W_up aggregate density={dens:.4f}%  "
              f"tile16: {tile16_active}/{tile16_total} active tiles, "
              f"mean_fill={t16_fill:.2f}/256 ({100.0*t16_fill/256:.2f}% useful)", flush=True)
        print(f"  [{tag} shapes] distinct (D,Dff) FFN shapes: {dict(shape_counter)}",
              flush=True)
        # grouped-GEMM ceiling: if we densify each block's active-row x active-col
        # sub-block, what is the padding waste vs the useful nnz?
        print(f"  [{tag} grouped-GEMM ceiling] a batched bmm over these blocks would "
              f"run active_tiles x tile^2 MACs; useful fraction ~= tile_fill above.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
