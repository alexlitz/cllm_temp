"""Read-only candidate enumeration for cross-op structural dedup.

Companion to ``crossop_dedup`` (which APPLIES the permutation tie).  This tool
only ENUMERATES + classifies sharing candidates so the win can be audited and the
ALMOST-shareable (never-tied) candidates are on the record.  It builds the compact
sparse model, applies the EXISTING byte-identical tie, and reports, over the
DISTINCT nonzero weight storages that survive:

  1. PERMUTATION classes — tensors related by an exact row/col permutation to a
     shared representative (the tie-able cross-op families).  Each is annotated
     with the block-name prefixes so the cross-FUNCTIONAL sharing is visible
     (e.g. ``alu-add-b* + alu-sub-b*`` share one adder core).
  2. SCALE / SIGN classes — tensors whose *normalised* value multiset matches but
     whose raw multiset differs (a genuine scale/sign relation, NOT a permutation).
     Reported, NEVER tied by ``crossop_dedup`` (would need a per-use scale op).
  3. IDENTITY / permutation-matrix lanes — square all-diagonal (``c*I``) weights
     (does-no-compute passthrough).  The within-block counterpart to the
     block-level identity consolidation.

Run:  python -m c4_min.crossop_analyze [--bitwise] [--divmod]
Memory-safe: one build; divmod peaks ~5 GB RSS.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict

import torch

from c4_min.compact_alloc import build_compact_sparse_streaming
from c4_min.weight_dedup import dedup_sparse_transformer, _dense_of
from c4_min.crossop_dedup import find_crossop_groups


def _distinct(sparse, L):
    names = getattr(L, "_block_names", None)
    seen = set()
    out = []
    for bi, b in enumerate(sparse.blocks):
        for kind, w in (("W_q", b.attn.W_q), ("W_k", b.attn.W_k),
                        ("W_v", b.attn.W_v), ("W_o", b.attn.W_o),
                        ("W_up", b.ffn.W_up), ("W_gate", b.ffn.W_gate),
                        ("W_down", b.ffn.W_down)):
            store = w.csr if w.is_sparse else w.dense
            if id(store) in seen or int(w.nnz) == 0:
                continue
            seen.add(id(store))
            nm = names[bi] if names and bi < len(names) else f"blk{bi}"
            out.append((nm, kind, _dense_of(w)))
    return out


def _multiset_key(t, signed=True):
    v = t[t != 0].reshape(-1)
    if not signed:
        m = v.abs().max()
        v = (t / m).abs()
        v = v[v != 0].reshape(-1)
    sv = torch.sort(v)[0]
    rd = torch.sort((t != 0).sum(1))[0]
    cd = torch.sort((t != 0).sum(0))[0]
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(sv.numpy().tobytes())
    h.update(rd.to(torch.int32).numpy().tobytes())
    h.update(cd.to(torch.int32).numpy().tobytes())
    return h.hexdigest()


def _prefix(nm):
    return re.sub(r'[-_]?\d+.*$', '', nm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitwise", action="store_true")
    ap.add_argument("--divmod", action="store_true")
    args = ap.parse_args()
    sparse, L, _ = build_compact_sparse_streaming(
        code_size=48, include_bitwise=args.bitwise or args.divmod,
        include_divmod=args.divmod, compute_mode="dense_kernel")
    dedup_sparse_transformer(sparse, L)
    dists = _distinct(sparse, L)
    print(f"distinct nonzero weight storages (post byte-identical tie): "
          f"{len(dists)}")

    # 1) permutation classes (exact) via the tie planner
    groups = find_crossop_groups(sparse, L)
    tie_tensors = sum(len(g.members) for g in groups)
    almost = sum(len(g.almost) for g in groups)
    print(f"\n[1] PERMUTATION classes (exact, TIE-ABLE): "
          f"{sum(1 for g in groups if g.members)} classes, "
          f"{tie_tensors} sibling tensors collapsible")
    for g in sorted(groups, key=lambda gg: -sum(m['nnz'] for m in gg.members)):
        if not g.members:
            continue
        rep_nm = None
        names = getattr(L, "_block_names", None)
        rb, rk = g.rep
        rep_nm = names[rb] if names and rb < len(names) else f"blk{rb}"
        prefs = sorted({_prefix(rep_nm)} |
                       {_prefix(m['name']) for m in g.members})
        rels = sorted({m['rel'] for m in g.members})
        print(f"  x{len(g.members)+1:3d} {rk:6s} {g.shape} nnz={g.rep_nnz} "
              f"rel={rels}  prefixes={prefs}")

    # 2) scale / sign classes (normalised multiset matches, raw differs)
    ss = defaultdict(list)
    for nm, kind, t in dists:
        ss[_multiset_key(t, signed=False)].append((nm, kind, t))
    print(f"\n[2] SCALE/SIGN classes (normalised match, raw differs — "
          f"REPORTED, NOT tied):")
    n_ss = 0
    for k, v in ss.items():
        if len(v) < 2:
            continue
        raw = {_multiset_key(t, signed=True) for _, _, t in v}
        if len(raw) == 1:
            continue                      # already a pure permutation class
        n_ss += 1
        print(f"  x{len(v)} {tuple(v[0][2].shape)} "
              f"{[f'{n}:{kk}' for n, kk, _ in v][:4]}")
    if n_ss == 0:
        print("  none (no true scale/sign-only cross-op family; the integer "
              "quantization gives a single canonical scale)")

    # 3) identity / permutation-matrix / passthrough lanes
    print(f"\n[3] IDENTITY / scaled-I passthrough lanes:")
    n_id = 0
    for nm, kind, t in dists:
        if t.shape[0] != t.shape[1]:
            continue
        diag = torch.diagonal(t)
        if (t - torch.diag(diag)).abs().max() == 0 and (diag != 0).any():
            n_id += 1
            uv = torch.unique(diag[diag != 0]).tolist()
            print(f"  {nm}:{kind} {tuple(t.shape)} diag_nz="
                  f"{int((diag != 0).sum())} vals={uv[:4]}")
    if n_id == 0:
        print("  none among distinct storages (within-block identity lanes "
              "absent; block-level identity is the block-MoE agent's scope)")

    print(f"\nsummary: {tie_tensors} tensors tie-able by exact permutation; "
          f"{almost} almost-shareable (multiset match, no exact perm) reported "
          f"not tied; {n_ss} scale/sign families reported not tied; "
          f"{n_id} identity lanes.")


if __name__ == "__main__":
    main()
