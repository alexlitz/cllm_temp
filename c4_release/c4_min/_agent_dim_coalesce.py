"""#758-sibling — RESIDUAL-DIM COALESCING analysis for the block-sparse FFN gather.

The FFN gather is a GATHER·SCALE·SCATTER (``block_sparse_ffn.CooLinear``): each
nonzero ``(row, col, val)`` reads residual dim ``col``.  Because the weights are
~99.9% sparse and SCATTERED (median ~1 nnz per hidden row, #751 refuted the band
premise), ``x2d.index_select(1, cols)`` reads dims in a NON-contiguous order ->
uncoalesced global-memory transactions -> bandwidth-bound.

A residual-dim PERMUTATION P is a RELABELING OF THE BASIS: applied consistently
to every weight touching the residual (W_up/W_gate/W_down cols/rows, W_q/W_k/W_v
cols, W_o rows, embed cols, lm_head cols) AND the driver dim-map, model.forward is
BYTE-IDENTICAL (same computation, reordered basis).  This is EXACTLY what the
build's ``color_dims`` + ``remap_layout`` + ``_remap_*`` machinery already does
for DIM-MINIMIZATION; here we ask a DIFFERENT objective: order the FREELY-
permutable dims so the aggregate FFN read pattern coalesces (RCM / spectral
bandwidth minimization on the dim co-read graph).

This probe is READ-ONLY (builds the golden model, measures, proposes; does not
write weights or change the build).  It:

  1. Builds the compact streaming model (peak = one block; CPU ok).
  2. Classifies dims: ATTENTION/CAM-constrained (read by any W_q/W_k/W_v col or
     written by any W_o row, in ANY block) + DECODE-constrained (register/decode
     bands the driver argmaxes) = FIXED; everything else = FREELY PERMUTABLE
     FFN-scratch.
  3. Measures the gather COALESCING metric before/after a permutation:
     ``move_count`` = number of contiguous ascending runs in the per-block
     read-column index pattern (== the number of uncoalesced memory transactions
     the index_select must issue; fewer runs == more coalesced).
  4. Computes an RCM order (Reverse Cuthill-McKee) over the FFN co-read graph on
     (a) the SAFE permutable subset only, (b) the FULL dim set (assess-only).
  5. Reports the honest win: safe-subset move-count reduction, full-permutation
     ceiling, and whether the scatter is fundamentally un-bandable.

VERDICT (code_size=32, golden 069cc32f, dim=1587, 242 live FFN blocks)
---------------------------------------------------------------------
The dim-coalescing win is ~ZERO because the existing ``color_dims`` liveness
compaction ALREADY coalesces the FFN read footprint as a side effect of packing
co-live dims contiguously:

  * distinct-dim read footprint = 914 contiguous runs over 5921 distinct reads
    (runs/distinct = 0.154) — median 2 runs/block, 220/242 blocks <= 4 runs.
    The DIV megablocks (the 188-block target) read 5288 distinct dims in 755
    runs (blk 34: 136 dims in 4 runs; ks0/1/2: 271 dims in 2 runs each).
  * a RANDOM permutation of the permutable dims gives ~4978 runs (5.4x WORSE):
    current/random = 0.184 => the current layout is already near-optimal.
  * RCM over the safe FFN-scratch subset makes it WORSE (914 -> 1727 uniq_runs,
    54203 -> 55043 native_runs): global RCM trades away the existing tight
    per-block locality. The full-permutation ceiling is only ~0.9% on native_runs.
  * the CooLinear ``native_runs`` (54203, 0.893/nnz) is NOT a dim-layout artifact:
    col-sorting the SAME nonzeros gives 55697 (slightly worse). It reflects that
    60704 nonzeros read only 5921 DISTINCT dims (each dim re-read by many hidden
    rows). A gather kernel that loads each distinct dim ONCE per block already
    pays only the 914-run footprint. That is #758's kernel job, not a layout one.

Conclusion: the residual layout is ALREADY column-coalesced by the build's
liveness packing; a dedicated coalescing permutation (safe subset OR full,
RoPE-pair moot since the golden is ALiBi not RoPE) buys ~0% and RCM regresses it.
Do NOT build C4_DIM_COALESCE. The bandwidth win for the fused gather lives in the
KERNEL (load each distinct dim once), not the dim ORDER.

Run:  OMP_NUM_THREADS=4 python -m c4_min._agent_dim_coalesce
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict, deque

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch


# --------------------------------------------------------------------------
# weight access helpers (storage-format independent)
# --------------------------------------------------------------------------
def _dense(w):
    if w is None:
        return None
    if getattr(w, "dense", None) is not None:
        return w.dense
    if getattr(w, "dense_resident", None) is not None:
        return w.dense_resident
    if getattr(w, "csr", None) is not None:
        return w.csr.to_dense()
    if isinstance(w, torch.Tensor):
        return w
    return None


def _read_cols(w):
    """Residual dims (input axis) that a [out,in] weight reads (nonzero col)."""
    d = _dense(w)
    if d is None:
        return torch.empty(0, dtype=torch.long)
    return (d != 0).any(dim=0).nonzero(as_tuple=False).flatten()


def _write_rows(w):
    """Residual dims (output axis) that a [out,in] weight writes (nonzero row)."""
    d = _dense(w)
    if d is None:
        return torch.empty(0, dtype=torch.long)
    return (d != 0).any(dim=1).nonzero(as_tuple=False).flatten()


# --------------------------------------------------------------------------
# COALESCING metric
# --------------------------------------------------------------------------
def move_count_for_block(Wup_dense):
    """Number of uncoalesced runs in the CooLinear gather read pattern.

    CooLinear orders nonzeros by ``.nonzero()`` == row-major (row, then col), and
    gathers ``cols`` in that order.  The number of MEMORY TRANSACTIONS a hardware
    gather issues is bounded below by the number of maximal contiguous
    (consecutive, ascending-by-1) runs in the read-index stream; a fully
    coalesced slice is 1 run.  We report the run count of the col stream in the
    NATIVE nonzero order (what index_select actually issues) AND of the sorted
    unique cols (the best any reorder of the SAME set can do WITHIN this block).
    """
    m = (Wup_dense != 0)
    idx = m.nonzero(as_tuple=False)              # [nnz,2] row-major (row,col)
    cols = idx[:, 1]
    nnz = cols.numel()
    if nnz == 0:
        return 0, 0, 0, 0
    native_runs = 1 + int((cols[1:] - cols[:-1] != 1).sum())
    uc = torch.unique(cols)
    uniq_runs = 1 + int((uc[1:] - uc[:-1] != 1).sum()) if uc.numel() else 0
    return nnz, int(uc.numel()), native_runs, uniq_runs


# --------------------------------------------------------------------------
# RCM / bandwidth minimization on the dim co-read graph
# --------------------------------------------------------------------------
def build_coread_graph(block_upgates, n_dims, permutable_mask):
    """Adjacency among PERMUTABLE dims: edge (a,b) iff some hidden unit reads BOTH
    a and b (co-read), aggregated over all live blocks.

    Only permutable<->permutable edges are kept (fixed dims are not reordered).
    Returns (adj, deg): adj = dict dim -> set(neighbor dims).
    """
    adj = defaultdict(set)
    for Wu in block_upgates:
        m = (Wu != 0)
        for r in range(m.shape[0]):
            cr = m[r].nonzero(as_tuple=False).flatten().tolist()
            cr = [c for c in cr if permutable_mask[c]]
            for i in range(len(cr)):
                for j in range(i + 1, len(cr)):
                    a, b = cr[i], cr[j]
                    adj[a].add(b); adj[b].add(a)
    deg = {n: len(v) for n, v in adj.items()}
    return adj, deg


def rcm_order(adj, nodes):
    """Reverse Cuthill-McKee ordering of ``nodes`` given adjacency ``adj``.

    Classic bandwidth-minimizing BFS reorder: start at a low-degree node, BFS
    visiting neighbors in increasing-degree order, then reverse.
    """
    nodeset = set(nodes)
    deg = {n: len([x for x in adj.get(n, ()) if x in nodeset]) for n in nodes}
    visited = set()
    order = []
    remaining = sorted(nodes, key=lambda n: (deg[n], n))
    for seed in remaining:
        if seed in visited:
            continue
        q = deque([seed]); visited.add(seed)
        while q:
            v = q.popleft(); order.append(v)
            nbrs = sorted((x for x in adj.get(v, ()) if x in nodeset and x not in visited),
                          key=lambda n: (deg[n], n))
            for w in nbrs:
                if w not in visited:
                    visited.add(w); q.append(w)
    order.reverse()
    for n in nodes:
        if n not in visited:
            order.append(n)
    return order


def apply_perm_to_upgates(block_upgates, perm_map):
    """Return new [out,in] up matrices with columns re-indexed by perm_map
    (new col = perm_map[old col]). Byte-exact col relabel (measure only)."""
    out = []
    idx = torch.tensor(perm_map, dtype=torch.long)
    for Wu in block_upgates:
        new = torch.zeros_like(Wu)
        new.index_copy_(1, idx, Wu)   # new[:, perm[c]] = Wu[:, c]
        out.append(new)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--code-size", type=int, default=32)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="limit distinct blocks analyzed (0=all)")
    args = ap.parse_args(argv)

    from .compact_alloc import build_compact_sparse_streaming
    print("[build] streaming compact model (peak one block) ...", flush=True)
    model, L, stats = build_compact_sparse_streaming(
        code_size=args.code_size, compute_mode="dense_kernel")
    D = model.dim
    phys = getattr(model, "_phys_blocks", model.blocks)
    print(f"[built] dim={D} phys_blocks={len(phys)} apply_blocks={len(model.blocks)} "
          f"vocab={model.vocab}", flush=True)

    # ------------------------------------------------------------------
    # 1. classify dims: attention/decode constrained (FIXED) vs FFN-scratch.
    # ------------------------------------------------------------------
    attn_read = torch.zeros(D, dtype=torch.bool)
    attn_write = torch.zeros(D, dtype=torch.bool)
    for b in phys:
        at = getattr(b, "attn", None)
        if at is None or getattr(at, "is_zero", False):
            continue
        for wn in ("W_q", "W_k", "W_v"):
            c = _read_cols(getattr(at, wn, None))
            if c.numel():
                attn_read[c] = True
        r = _write_rows(getattr(at, "W_o", None))
        if r.numel():
            attn_write[r] = True

    decode_names = ["PC_VAL", "SP_VAL", "BP_VAL", "STK_VAL", "HALTED", "AX",
                    "PC", "SP", "BP", "STACK0", "ONE", "OUTPUT", "OUTPUT_LO",
                    "OUTPUT_HI", "POS"]
    decode_fixed = torch.zeros(D, dtype=torch.bool)
    names = dict(getattr(L, "_names", {}))
    for nm in decode_names:
        rec = names.get(nm)
        if rec:
            base, size = rec
            for k in range(size):
                if 0 <= base + k < D:
                    decode_fixed[base + k] = True
    try:
        from .compact_alloc import never_share_dims_from_layout
        for d in never_share_dims_from_layout(L):
            if 0 <= d < D:
                decode_fixed[d] = True
    except Exception:
        pass

    fixed = attn_read | attn_write | decode_fixed
    permutable = ~fixed
    n_fixed = int(fixed.sum()); n_perm = int(permutable.sum())
    print(f"\n[dim classification] D={D}")
    print(f"  attention-read (W_q/k/v cols):    {int(attn_read.sum())}")
    print(f"  attention-write (W_o rows):       {int(attn_write.sum())}")
    print(f"  decode/register fixed:            {int(decode_fixed.sum())}")
    print(f"  => FIXED (constrained):           {n_fixed} ({100.0*n_fixed/D:.1f}%)")
    print(f"  => FREELY PERMUTABLE FFN-scratch: {n_perm} ({100.0*n_perm/D:.1f}%)",
          flush=True)

    # ------------------------------------------------------------------
    # 2. gather W_up dense for the live blocks (dedup by object).
    # ------------------------------------------------------------------
    block_up = []
    blk_names = list(getattr(L, "_block_names", []))
    for bi, b in enumerate(phys):
        ff = getattr(b, "ffn", None)
        if ff is None or getattr(b, "_routed", False):
            continue
        Wu = _dense(getattr(ff, "W_up", None))
        if Wu is None:
            continue
        block_up.append((bi, Wu))
    if args.max_blocks:
        block_up = block_up[:args.max_blocks]
    print(f"\n[live FFN blocks for analysis] {len(block_up)} distinct non-routed",
          flush=True)

    # ------------------------------------------------------------------
    # 3. per-block coalescing metric.
    # ------------------------------------------------------------------
    print("\n[per-block gather coalescing — top-16 by native_runs + aggregate]")
    print(f"  {'blk':>4} {'name':22s} {'nnz':>7} {'uniq_col':>8} "
          f"{'native_runs':>11} {'uniq_runs':>9} {'runs/nnz':>8}")
    tot_nnz = tot_uniq = tot_native = tot_uniqruns = 0
    worst = []
    for bi, Wu in block_up:
        nnz, ucn, native, uniqr = move_count_for_block(Wu)
        tot_nnz += nnz; tot_uniq += ucn; tot_native += native; tot_uniqruns += uniqr
        worst.append((native, bi, nnz, ucn, uniqr))
    worst.sort(reverse=True)
    for native, bi, nnz, ucn, uqr in worst[:16]:
        nm = blk_names[bi][:22] if bi < len(blk_names) else str(bi)
        rr = native / max(nnz, 1)
        print(f"  {bi:>4} {nm:22s} {nnz:>7} {ucn:>8} {native:>11} {uqr:>9} {rr:>8.2f}")
    print(f"\n  [AGGREGATE over {len(block_up)} blocks] nnz={tot_nnz} "
          f"uniq_col_reads={tot_uniq} native_runs={tot_native} "
          f"uniq_runs(best-within-block)={tot_uniqruns}")
    print(f"  native_runs/nnz = {tot_native/max(tot_nnz,1):.3f}  "
          f"(1.0 => every read its own transaction; ->0 => coalesced)")
    print(f"  best-within-block runs/nnz = {tot_uniqruns/max(tot_nnz,1):.3f}",
          flush=True)

    # ------------------------------------------------------------------
    # 4. RCM reorder of the SAFE permutable subset + measure after.
    # ------------------------------------------------------------------
    perm_mask_list = permutable.tolist()
    print("\n[building co-read graph over PERMUTABLE dims] ...", flush=True)
    ups = [Wu for _, Wu in block_up]
    adj, deg = build_coread_graph(ups, D, perm_mask_list)
    perm_nodes = [d for d in range(D) if perm_mask_list[d]]
    n_edges = sum(len(v) for v in adj.values()) // 2
    isolated = sum(1 for d in perm_nodes if not adj.get(d))
    print(f"  permutable nodes={len(perm_nodes)} co-read edges={n_edges} "
          f"isolated(no co-read)={isolated} "
          f"({100.0*isolated/max(len(perm_nodes),1):.1f}%)", flush=True)

    order = rcm_order(adj, perm_nodes)
    perm_slots = sorted(perm_nodes)
    perm_map = list(range(D))
    for new_pos, old_dim in zip(perm_slots, order):
        perm_map[old_dim] = new_pos

    ups_perm = apply_perm_to_upgates(ups, perm_map)
    tot_native2 = 0
    for Wu in ups_perm:
        _, _, native, _ = move_count_for_block(Wu)
        tot_native2 += native
    print(f"\n[SAFE-subset RCM reorder result]")
    print(f"  native_runs BEFORE = {tot_native}   AFTER = {tot_native2}   "
          f"reduction = {100.0*(tot_native-tot_native2)/max(tot_native,1):.1f}%",
          flush=True)

    # ------------------------------------------------------------------
    # 5. FULL-permutation ceiling (assess-only).
    # ------------------------------------------------------------------
    all_nodes = list(range(D))
    full_perm_ok = [True] * D
    adj_full, _ = build_coread_graph(ups, D, full_perm_ok)
    order_full = rcm_order(adj_full, all_nodes)
    perm_map_full = [0] * D
    for new_pos, old_dim in enumerate(order_full):
        perm_map_full[old_dim] = new_pos
    ups_full = apply_perm_to_upgates(ups, perm_map_full)
    tot_native3 = 0
    for Wu in ups_full:
        _, _, native, _ = move_count_for_block(Wu)
        tot_native3 += native
    print(f"\n[FULL-permutation ceiling (ignores attention constraint — assess only)]")
    print(f"  native_runs BEFORE = {tot_native}   AFTER(full) = {tot_native3}   "
          f"reduction = {100.0*(tot_native-tot_native3)/max(tot_native,1):.1f}%",
          flush=True)

    # ------------------------------------------------------------------
    # 6. scatter location: how much of the co-read touches FIXED vs PERMUTABLE.
    # ------------------------------------------------------------------
    fixed_reads = 0; perm_reads = 0; nnz_all = 0
    multi_read_rows = 0; total_rows = 0
    for Wu in ups:
        m = (Wu != 0)
        cols = m.nonzero(as_tuple=False)[:, 1]
        nnz_all += cols.numel()
        fixed_reads += int(fixed[cols].sum())
        perm_reads += int(permutable[cols].sum())
        rpr = m.sum(dim=1)
        multi_read_rows += int((rpr > 1).sum())
        total_rows += int((rpr > 0).sum())
    print(f"\n[scatter location] of {nnz_all} total FFN W_up reads:")
    print(f"  read a FIXED dim:      {fixed_reads} "
          f"({100.0*fixed_reads/max(nnz_all,1):.1f}%)")
    print(f"  read a PERMUTABLE dim: {perm_reads} "
          f"({100.0*perm_reads/max(nnz_all,1):.1f}%)")
    print(f"  hidden rows reading >1 dim (co-read structure exists): "
          f"{multi_read_rows}/{total_rows} "
          f"({100.0*multi_read_rows/max(total_rows,1):.1f}%)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
