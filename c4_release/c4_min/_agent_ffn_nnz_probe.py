"""Probe the EXACT per-block FFN sparse structure of the compact model so we can
design a compact-tile sparse-GEMM kernel.

For each block's W_up / W_gate / W_down (the padded [Dff, D] weights):
  * nnz, active rows (hidden units that read >=1 dim), active cols (input dims read)
  * rows-per-unit histogram (does each unit read 1 dim as #804 claims?)
  * the UNION of input columns actually read across all three matrices (this is the
    compact input width if we gather only the read dims)
  * the UNION of output rows written (compact hidden width)
  * whether the (active_rows x active_cols) sub-block is dense enough to tile

Aggregate: total nnz across the whole model, and the compact-tile ceiling — if we
gather each block's active input cols + active hidden rows into a small dense tile,
how much useful vs padded FLOP do we run?

Read-only; touches no golden weight.  Run:
  OMP_NUM_THREADS=4 python -m c4_min._agent_ffn_nnz_probe --device cuda:0
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import torch


def _dense(w):
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--code-size", type=int, default=44)
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--dump", type=int, default=30, help="per-block rows to print")
    args = ap.parse_args()

    from .compact_alloc import build_compact_sparse_streaming
    model, L, _ = build_compact_sparse_streaming(
        code_size=max(args.code_size, 20), compute_mode="sparse_mm",
        recurrent_divmod=args.recurrent)
    print(f"[built] blocks={len(model.blocks)} dim={model.dim}", flush=True)

    names = list(getattr(L, "_block_names", []))
    tot_nnz = 0
    tot_dense_flop = 0
    tot_compact_flop = 0   # active_rows * active_cols per matrix (compact dense tile)
    tot_useful_flop = 0    # 2*nnz
    n_ffn = 0
    max_active_cols = 0
    max_active_rows = 0
    shape_counter = {}
    per_block = []
    seen = set()
    for bi, b in enumerate(model.blocks):
        if id(b) in seen:
            continue
        seen.add(id(b))
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        n_ffn += 1
        row = {"bi": bi, "name": names[bi] if bi < len(names) else str(bi)}
        block_active_cols = set()
        block_active_rows = 0
        block_nnz = 0
        block_compact = 0
        block_dense = 0
        D = None
        Dff = None
        for tag in ("W_up", "W_gate", "W_down"):
            W = _dense(getattr(b.ffn, tag)).cpu()
            out_d, in_d = W.shape
            m = (W != 0)
            nnz = int(m.sum())
            cols = torch.nonzero(m.any(dim=0)).flatten()
            rows = torch.nonzero(m.any(dim=1)).flatten()
            ac, ar = cols.numel(), rows.numel()
            block_active_cols |= set(cols.tolist())
            block_active_rows += ar
            block_nnz += nnz
            block_compact += ar * ac
            block_dense += out_d * in_d
            row[tag] = {"shape": (out_d, in_d), "nnz": nnz,
                        "active_rows": ar, "active_cols": ac}
            if tag == "W_up":
                D, Dff = in_d, out_d
        tot_nnz += block_nnz
        tot_dense_flop += 2 * block_dense
        tot_compact_flop += 2 * block_compact
        tot_useful_flop += 2 * block_nnz
        max_active_cols = max(max_active_cols, len(block_active_cols))
        max_active_rows = max(max_active_rows, block_active_rows)
        shape_counter[(D, Dff)] = shape_counter.get((D, Dff), 0) + 1
        row["nnz"] = block_nnz
        row["D"] = D
        row["Dff"] = Dff
        row["union_active_cols"] = len(block_active_cols)
        row["sum_active_rows"] = block_active_rows
        per_block.append(row)

    print(f"\n[FFN blocks] {n_ffn} distinct FFN blocks")
    print(f"[shapes] distinct (D,Dff): {len(shape_counter)} shapes")
    for (D, Dff), c in sorted(shape_counter.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    (D={D}, Dff={Dff}) x {c}")
    print(f"\n[nnz] total FFN nnz = {tot_nnz:,}")
    print(f"[flop] dense-equiv per-token = {tot_dense_flop/1e9:.4f} GFLOP")
    print(f"[flop] COMPACT-tile per-token = {tot_compact_flop/1e6:.4f} MFLOP "
          f"({tot_compact_flop/tot_dense_flop*100:.4f}% of dense)")
    print(f"[flop] USEFUL (2*nnz) per-token = {tot_useful_flop/1e6:.4f} MFLOP "
          f"({tot_useful_flop/tot_dense_flop*100:.6f}% of dense)")
    print(f"[compact-tile useful frac] {tot_useful_flop/max(tot_compact_flop,1)*100:.2f}% "
          f"(useful nnz / compact active_rows*active_cols)")
    print(f"[max per-block] union_active_cols={max_active_cols}  "
          f"sum_active_rows={max_active_rows}")

    print(f"\n[per-block detail, first {args.dump}]")
    hdr = (f"  {'blk':>4} {'name':24s} {'D':>5} {'Dff':>6} {'nnz':>7} "
           f"{'un_cols':>7} {'sum_rows':>8} {'up_ar':>6} {'up_ac':>6} "
           f"{'up_nnz':>7} {'compact_useful%':>15}")
    print(hdr)
    for row in per_block[:args.dump]:
        up = row["W_up"]
        compact = 0
        for tag in ("W_up", "W_gate", "W_down"):
            t = row[tag]
            compact += t["active_rows"] * t["active_cols"]
        cu = 100.0 * row["nnz"] / max(compact, 1)
        print(f"  {row['bi']:>4} {row['name'][:24]:24s} {row['D']:>5} {row['Dff']:>6} "
              f"{row['nnz']:>7} {row['union_active_cols']:>7} {row['sum_active_rows']:>8} "
              f"{up['active_rows']:>6} {up['active_cols']:>6} {up['nnz']:>7} {cu:>14.2f}%")

    # nnz-per-row histogram over ALL W_up (the "each unit reads ~1 dim" claim)
    print(f"\n[nnz-per-unit histogram over ALL W_up rows]")
    hist = {}
    seen2 = set()
    for b in model.blocks:
        if id(b) in seen2:
            continue
        seen2.add(id(b))
        if getattr(b, "_routed", False) or getattr(b.ffn, "W_up", None) is None:
            continue
        W = _dense(b.ffn.W_up).cpu()
        rp = (W != 0).sum(dim=1)
        for v in rp[rp > 0].tolist():
            hist[v] = hist.get(v, 0) + 1
    for k in sorted(hist)[:15]:
        print(f"    reads={k:>3}: {hist[k]:>7} units")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
