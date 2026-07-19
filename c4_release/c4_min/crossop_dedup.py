"""Cross-functional STRUCTURAL weight dedup for the c4_min sparse VM.

This is the *next* layer of sharing on top of ``weight_dedup.py``.  The existing
pass ties only BYTE-IDENTICAL weight tensors (same sha256).  What it leaves behind
is a large family of tensors that are the SAME numbers in a DIFFERENT LAYOUT — the
identical value multiset addressed through a row- or column-PERMUTATION.  These
turn out to be *cross-functional*: e.g. ADD's carry-propagation byte lanes
(``alu-add-b*``) and SUB's borrow cascade (``alu-sub-b*``) are one adder core up
to a column/row permutation; the base-16 long-division ``kb``-precompute and
``qb``-carry correctors (``alu-div-kb*`` / ``alu-div-qbc*``) are one core; etc.

A row/column permutation of a weight matrix has the IDENTICAL nonzero-value
multiset, so every one of its *scalar weight values* is a duplicate of the
representative's.  This pass stores the representative ONCE and replaces every
permutation-related sibling's private storage with

    (base_ref, row_perm_index, col_perm_index)

where the index vectors are INTEGERS (structural addressing, not learned weights).
``F.linear`` is left byte-EXACT: at call time the sibling reconstructs the
*identical* dense tensor ``base[row_perm][:, col_perm]`` and calls the SAME
``F.linear`` the dense/sparse model called — L-inf = 0, greedy decode unchanged,
no architecture change (still vanilla SwiGLU / softmax1-ALiBi attention).

Purity / vanilla guarantee
--------------------------
* No new op, layer, router or block is introduced; the block dispatch is untouched.
* The only change is that a shared sibling's ``linear`` method is swapped (per
  INSTANCE, via ``types.MethodType``) for a closure that reconstructs the exact
  same dense weight from the shared base + an integer permutation, then calls the
  unchanged ``F.linear``.  The reconstructed tensor is byte-identical to what was
  stored, so every ``model.forward`` produces the same logits (proven by the
  L-inf=0 gate below), i.e. this is still the vanilla forward.
* No approximation, no quantization change, no low-rank truncation: the integer
  requant table is reproduced EXACTLY (torch.equal), which is the acceptance gate.

Accounting
----------
The tie is priced honestly two ways:

* **unique nonzero SCALAR weights** — a permutation reuses the base's scalars, so
  every sibling's nnz scalars become duplicates: unique-scalar count drops by the
  full summed sibling nnz.  This is the primary "unique weight count" metric.
* **stored BYTES** — a sibling stores only a SPARSE integer permutation instead of
  its CSR value+col arrays.  The full axis perm is ~99% identity (it mostly
  permutes all-zero base rows/cols among themselves), so it is stored as identity
  + a handful of ``(pos -> tgt)`` int16 overrides (:class:`SparsePerm`), which is
  tiny (e.g. the 16x divmod ``kb`` core needs 270 overrides, not 16*1633 index
  entries).  The pass is ECONOMICALLY GATED to only tie a group when it reduces
  both the unique-scalar count AND the stored bytes (``econ=True``, the default),
  or — with ``econ=False`` — whenever it reduces unique scalars.  With the sparse
  perm the two modes essentially coincide (the index is near-free).

ALMOST-shareable but NOT tied (reported, never tied): tensors that match a
representative's value multiset + row/col degree sequences but for which NO exact
row/col permutation reconstructs them (a coincidental multiset match).  Those are
reported by :func:`find_crossop_groups` with ``kind='none'`` and left alone — no
silent lossy sharing.
"""
from __future__ import annotations

import hashlib
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .weight_dedup import _dense_of


# ---------------------------------------------------------------------------
# Permutation detection (exact).
# ---------------------------------------------------------------------------
def _value_multiset_key(t: torch.Tensor) -> str:
    """Fingerprint invariant under row+col permutation (necessary condition).

    Sorted nonzero-value multiset + sorted per-row and per-col nnz-degree
    sequences + shape.  Two tensors that differ get different keys; two related
    by a permutation get the same key (so it is a cheap CANDIDATE filter before
    the exact O(n) permutation search)."""
    vals = t[t != 0].detach().cpu()
    sv = torch.sort(vals.reshape(-1))[0]
    row_deg = torch.sort((t != 0).sum(1))[0]
    col_deg = torch.sort((t != 0).sum(0))[0]
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(sv.numpy().tobytes())
    h.update(row_deg.to(torch.int32).numpy().tobytes())
    h.update(col_deg.to(torch.int32).numpy().tobytes())
    return h.hexdigest()


def _row_hashes(M: torch.Tensor) -> List[str]:
    return [hashlib.sha256(M[i].contiguous().numpy().tobytes()).hexdigest()
            for i in range(M.shape[0])]


def _col_hashes(M: torch.Tensor) -> List[str]:
    return [hashlib.sha256(M[:, j].contiguous().numpy().tobytes()).hexdigest()
            for j in range(M.shape[1])]


def _find_axis_perm(A: torch.Tensor, B: torch.Tensor, axis: str
                    ) -> Optional[torch.Tensor]:
    """Return an index ``idx`` s.t. ``A == B[idx]`` (row) / ``A == B[:, idx]``
    (col), or ``None`` if no such permutation exactly reconstructs A."""
    hf = _row_hashes if axis == "row" else _col_hashes
    hA, hB = hf(A), hf(B)
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, h in enumerate(hB):
        buckets[h].append(i)
    idx: List[int] = []
    used: Dict[str, int] = defaultdict(int)
    for h in hA:
        lst = buckets.get(h)
        if not lst or used[h] >= len(lst):
            return None
        idx.append(lst[used[h]])
        used[h] += 1
    idx_t = torch.tensor(idx, dtype=torch.long)
    recon = B[idx_t] if axis == "row" else B[:, idx_t]
    return idx_t if torch.equal(A, recon) else None


def _relate(A: torch.Tensor, B: torch.Tensor
            ) -> Tuple[str, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Classify how A relates to base B.  Returns ``(kind, row_idx, col_idx)``.

    ``kind`` in {"identical","row","col","rowcol","none"}.  For "row"/"col"/"rowcol"
    the returned indices reconstruct ``A`` EXACTLY from ``B``
    (``A == B[row_idx][:, col_idx]``, with an identity index where absent).
    """
    if A.shape != B.shape:
        return "none", None, None
    if torch.equal(A, B):
        return "identical", None, None
    q = _find_axis_perm(A, B, "col")
    if q is not None:
        return "col", None, q
    p = _find_axis_perm(A, B, "row")
    if p is not None:
        return "row", p, None
    # combined: try to align rows first (by full-row hash after a col sort is
    # ambiguous), so attempt row-perm on a col-canonicalised pair.
    #   canonicalise columns of both by sorting column hashes, find the row perm
    #   on the canonical form, then compose.  Only accept on exact torch.equal.
    def canon_cols(M):
        ch = _col_hashes(M)
        order = sorted(range(M.shape[1]), key=lambda j: ch[j])
        return M[:, order], torch.tensor(order, dtype=torch.long)
    Ac, Aord = canon_cols(A)
    Bc, Bord = canon_cols(B)
    p2 = _find_axis_perm(Ac, Bc, "row")
    if p2 is not None:
        # A == Ac[:, inv(Aord)]; Ac == Bc[p2]; Bc == B[:, Bord]
        # => A[:, Aord] == B[p2][:, Bord]  => A == B[p2][:, Bord][:, inv(Aord)]
        inv_Aord = torch.empty_like(Aord)
        inv_Aord[Aord] = torch.arange(Aord.numel())
        col_idx = Bord[inv_Aord]
        recon = B[p2][:, col_idx]
        if torch.equal(A, recon):
            return "rowcol", p2, col_idx
    return "none", None, None


# ---------------------------------------------------------------------------
# Distinct-storage enumeration (post byte-identical tie).
# ---------------------------------------------------------------------------
def _iter_distinct(sparse):
    """Yield ``(bi, kind, sparse_weight, dense_tensor)`` for each DISTINCT
    nonzero 2-D weight storage (by ``id()``) in the built model."""
    seen = set()
    for bi, b in enumerate(sparse.blocks):
        for kind, w in (("W_q", b.attn.W_q), ("W_k", b.attn.W_k),
                        ("W_v", b.attn.W_v), ("W_o", b.attn.W_o),
                        ("W_up", b.ffn.W_up), ("W_gate", b.ffn.W_gate),
                        ("W_down", b.ffn.W_down)):
            store = w.csr if w.is_sparse else w.dense
            if id(store) in seen or int(w.nnz) == 0:
                continue
            seen.add(id(store))
            yield bi, kind, w, _dense_of(w)


# ---------------------------------------------------------------------------
# Perm-index storage sizing.
# ---------------------------------------------------------------------------
def _perm_dtype(n: int) -> torch.dtype:
    """Smallest int dtype that indexes ``n`` positions (structural addressing)."""
    if n <= (1 << 15):
        return torch.int16          # covers every c4_min dim (< 32768)
    return torch.int32


def _sparse_weight_storage_bytes(w) -> int:
    return int(w.storage_bytes())


# ---------------------------------------------------------------------------
# Group finder (candidate classes -> exact relation to a representative).
# ---------------------------------------------------------------------------
@dataclass
class CrossOpGroup:
    rep: Tuple[int, str]                     # (block_idx, kind) of representative
    shape: Tuple[int, int]
    rep_nnz: int
    members: List[dict] = field(default_factory=list)   # each: bi,kind,kind_rel,row,col,nnz
    almost: List[dict] = field(default_factory=list)     # non-permutation coincidences


def find_crossop_groups(sparse, L=None) -> List[CrossOpGroup]:
    """Enumerate cross-op permutation-sharing groups (exact) + almost-groups.

    Groups every DISTINCT nonzero weight by value-multiset key, then within each
    candidate class finds the EXACT row/col permutation of every member relative
    to the class representative.  Members that permute exactly become tie targets;
    members whose multiset matches but which NO permutation reconstructs are
    recorded in ``almost`` (reported, never tied)."""
    names = getattr(L, "_block_names", None) if L is not None else None
    cand: Dict[str, List] = defaultdict(list)
    for bi, kind, w, t in _iter_distinct(sparse):
        cand[_value_multiset_key(t)].append((bi, kind, w, t))
    groups: List[CrossOpGroup] = []
    for key, members in cand.items():
        if len(members) < 2:
            continue
        rep_bi, rep_kind, rep_w, rep_t = members[0]
        g = CrossOpGroup(rep=(rep_bi, rep_kind), shape=tuple(rep_t.shape),
                         rep_nnz=int((rep_t != 0).sum()))
        for bi, kind, w, t in members[1:]:
            rel, row_idx, col_idx = _relate(t, rep_t)
            entry = dict(bi=bi, kind=kind, rel=rel, row=row_idx, col=col_idx,
                         w=w, nnz=int((t != 0).sum()),
                         name=(names[bi] if names and bi < len(names)
                               else f"blk{bi}"))
            if rel == "none":
                g.almost.append(entry)
            else:
                g.members.append(entry)
        if g.members or g.almost:
            groups.append(g)
    groups.sort(key=lambda gg: -(sum(m["nnz"] for m in gg.members)))
    return groups


# ---------------------------------------------------------------------------
# The tie (in place).
# ---------------------------------------------------------------------------
@dataclass
class SparsePerm:
    """A permutation index stored SPARSELY as identity + explicit overrides.

    A permutation ``idx`` of length ``n`` reconstructs a tensor axis as
    ``base[idx]``.  The c4_min weights are ~99% zero rows/cols, so ``idx`` is
    almost entirely permuting all-zero base rows/cols among themselves — those
    positions can be CANONICALISED to identity (a zero-source landing on a
    zero-identity-target produces the same zero line either way), leaving only a
    handful of positions that truly differ.  We store just those override pairs
    ``(pos -> tgt)`` (int16) + the length; the full ``idx`` is materialised as
    ``arange(n)`` with the overrides written in.  The reconstruction is verified
    ``torch.equal`` to the original weight at tie time, so it is byte-EXACT.
    """
    n: int
    pos: torch.Tensor        # int16/int32 override positions
    tgt: torch.Tensor        # int16/int32 override targets

    def full(self) -> torch.Tensor:
        idx = torch.arange(self.n, dtype=torch.long)
        if self.pos.numel():
            idx[self.pos.to(torch.long)] = self.tgt.to(torch.long)
        return idx

    def bytes(self) -> int:
        w = 2 if self.n <= (1 << 15) else 4
        return int(self.pos.numel()) * w + int(self.tgt.numel()) * w + 4


def _compact_perm(idx: torch.Tensor, base_axis_nonzero: torch.Tensor
                  ) -> SparsePerm:
    """Compact a full axis permutation into a :class:`SparsePerm`.

    ``base_axis_nonzero[k]`` is True iff base row/col ``k`` has any nonzero.  A
    position ``j`` whose source ``idx[j]`` is an all-zero base line AND whose
    identity target ``j`` is also all-zero can be reset to identity without
    changing the reconstructed weight (both produce a zero line).  Every other
    position that differs from identity is stored as an override."""
    n = idx.numel()
    it = idx.to(torch.long)
    ar = torch.arange(n)
    src_zero = ~base_axis_nonzero[it]
    tgt_zero = ~base_axis_nonzero[ar]
    canon_id = src_zero & tgt_zero          # safe to force identity
    keep = (it != ar) & (~canon_id)         # must store these
    pos = ar[keep]
    tgt = it[keep]
    dt = _perm_dtype(n)
    return SparsePerm(n=n, pos=pos.to(dt), tgt=tgt.to(dt))


def _make_permuted_linear(base_w, row_perm: "SparsePerm | None",
                          col_perm: "SparsePerm | None"):
    """Return a ``linear`` closure that reconstructs the exact dense weight from
    ``base_w`` + sparse perms and calls the unchanged ``F.linear``.

    Byte-identical: the reconstructed tensor equals the sibling's original weight
    (proven at tie time), so the GEMM and accumulation order are the same the
    stored-dense model used in ``dense_kernel`` mode -> L-inf = 0."""
    ri = None if row_perm is None else row_perm.full()
    ci = None if col_perm is None else col_perm.full()

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        # base dense (resident if materialized, else CSR->dense / dense).
        if base_w.dense_resident is not None:
            base = base_w.dense_resident
        elif base_w.is_sparse:
            base = base_w.csr.to_dense()
        else:
            base = base_w.dense
        W = base
        if ri is not None:
            W = W[ri]
        if ci is not None:
            W = W[:, ci]
        return F.linear(x, W)

    return linear


@dataclass
class CrossOpStats:
    groups_tied: int = 0
    tensors_tied: int = 0
    scalars_before: int = 0            # summed nnz over ALL distinct storages
    scalars_after: int = 0             # summed nnz over UNIQUE-after-crossop bases
    bytes_before: int = 0
    bytes_after: int = 0
    perm_index_bytes: int = 0
    almost_count: int = 0
    almost_detail: List[str] = field(default_factory=list)
    group_detail: List[str] = field(default_factory=list)

    @property
    def scalars_saved(self) -> int:
        return self.scalars_before - self.scalars_after

    def summary(self) -> str:
        bb, ba = self.bytes_before, self.bytes_after
        lines = [
            "=== cross-op structural dedup (permutation sharing) ===",
            f"  groups tied        : {self.groups_tied}",
            f"  sibling tensors tied: {self.tensors_tied} "
            f"(each -> base ref + SPARSE int perm)",
            f"  unique NONZERO scalars: {self.scalars_before} -> {self.scalars_after} "
            f"(saved {self.scalars_saved}, "
            f"{100 * self.scalars_saved / max(1, self.scalars_before):.1f}%)",
            f"  weight stored bytes  : {bb/1e6:.3f} MB -> {ba/1e6:.3f} MB "
            f"(saved {(bb-ba)/1e6:.3f} MB; incl {self.perm_index_bytes/1e6:.3f} MB "
            f"of sparse int perm indices)",
            f"  ALMOST-shareable (multiset match, NO exact perm) NOT tied: "
            f"{self.almost_count}",
        ]
        if self.group_detail:
            lines.append("  tied groups:")
            lines.extend("    " + d for d in self.group_detail[:20])
        if self.almost_detail:
            lines.append("  almost (reported, left alone):")
            lines.extend("    " + d for d in self.almost_detail[:20])
        return "\n".join(lines)


def crossop_dedup(sparse, L=None, econ: bool = True) -> CrossOpStats:
    """Tie cross-op permutation-related weights IN PLACE on a built model.

    For each group whose siblings are exact row/col permutations of a
    representative, drop the siblings' private storage and install a
    reconstruct-then-``F.linear`` closure keyed on the base + integer perm.

    ``econ=True`` (default): only tie a group if it reduces BOTH the unique-scalar
    count AND the stored bytes (the perm index must be smaller than the freed
    value+index arrays).  ``econ=False``: tie whenever it reduces unique scalars
    (bytes may be neutral); use this to maximise the unique-weight-count metric.

    Byte-identity: every sibling's reconstructed weight equals its original
    (``torch.equal`` checked here), so the forward is L-inf=0 unchanged.
    """
    stats = CrossOpStats()
    # price the pre-crossop distinct storages
    for _bi, _kind, w, t in _iter_distinct(sparse):
        stats.scalars_before += int((t != 0).sum())
        stats.bytes_before += _sparse_weight_storage_bytes(w)

    groups = find_crossop_groups(sparse, L)

    # decide + apply per group
    for g in groups:
        rep_bi, rep_kind = g.rep
        rep_w = _resolve_slot(sparse, rep_bi, rep_kind)
        rep_nnz = int(rep_w.nnz)
        rep_dense = _dense_of(rep_w)
        base_col_nz = (rep_dense != 0).any(0)
        base_row_nz = (rep_dense != 0).any(1)

        # build the COMPACT (sparse) perms for every member + price the tie
        sib_scalars = sum(m["nnz"] for m in g.members)
        sib_bytes = sum(_sparse_weight_storage_bytes(m["w"]) for m in g.members)
        idx_bytes = 0
        for m in g.members:
            rp = (None if m["row"] is None
                  else _compact_perm(m["row"], base_row_nz))
            cp = (None if m["col"] is None
                  else _compact_perm(m["col"], base_col_nz))
            m["rowp"], m["colp"] = rp, cp
            idx_bytes += (0 if rp is None else rp.bytes()) \
                + (0 if cp is None else cp.bytes())
        scalar_win = sib_scalars > 0
        byte_win = sib_bytes > idx_bytes
        do_tie = g.members and scalar_win and (byte_win or not econ)

        # record almost-shareable for the group (reported, never tied)
        for a in g.almost:
            stats.almost_count += 1
            stats.almost_detail.append(
                f"x1 {a['name']}:{a['kind']} shape={g.shape} "
                f"(multiset==rep {g.rep} but no exact perm)")

        if not do_tie:
            continue

        stats.groups_tied += 1
        for m in g.members:
            sib = m["w"]
            rp, cp = m["rowp"], m["colp"]
            # verify EXACT reconstruction from the compact perms (acceptance gate)
            W = rep_dense
            if rp is not None:
                W = W[rp.full()]
            if cp is not None:
                W = W[:, cp.full()]
            assert torch.equal(W, _dense_of(sib)), \
                f"perm reconstruction mismatch {m['name']}:{m['kind']}"
            # drop private storage, install reconstruction closure
            sib.csr = None
            sib.dense = None
            sib.dense_resident = None
            sib._crossop_base = rep_w
            sib._crossop_row = rp
            sib._crossop_col = cp
            sib.linear = types.MethodType(
                _make_permuted_linear(rep_w, rp, cp), sib)
            stats.tensors_tied += 1
            stats.perm_index_bytes += (0 if rp is None else rp.bytes()) \
                + (0 if cp is None else cp.bytes())
        stats.group_detail.append(
            f"x{len(g.members)+1} {rep_kind} shape={g.shape} nnz={rep_nnz} "
            f"base={_resolve_name(L, rep_bi)} "
            f"siblings={[m['name'] for m in g.members][:4]}")

    # recompute unique-scalar + bytes AFTER (distinct bases by id)
    stats.scalars_after, stats.bytes_after = _post_accounting(sparse)
    return stats


def _resolve_slot(sparse, bi, kind):
    b = sparse.blocks[bi]
    if kind in ("W_q", "W_k", "W_v", "W_o"):
        return getattr(b.attn, kind)
    return getattr(b.ffn, kind)


def _resolve_name(L, bi):
    names = getattr(L, "_block_names", None) if L is not None else None
    return names[bi] if names and bi < len(names) else f"blk{bi}"


def _post_accounting(sparse) -> Tuple[int, int]:
    """Unique nonzero scalars + stored bytes AFTER the cross-op tie.

    Mirrors the baseline ``_distinct_scalars_bytes`` accounting EXACTLY so the
    before/after numbers are comparable: every DISTINCT storage (by ``id()``,
    zeros folded and counted once per shared object) is charged once, a
    crossop-tied sibling contributes ONLY its integer perm index (its value
    scalars are the base's, counted with the base)."""
    seen_store = set()
    scalars = 0
    total_bytes = 0
    for bi, b in enumerate(sparse.blocks):
        for kind, w in (("W_q", b.attn.W_q), ("W_k", b.attn.W_k),
                        ("W_v", b.attn.W_v), ("W_o", b.attn.W_o),
                        ("W_up", b.ffn.W_up), ("W_gate", b.ffn.W_gate),
                        ("W_down", b.ffn.W_down)):
            if getattr(w, "_crossop_base", None) is not None:
                # tied sibling: only its sparse perm index is new storage.
                rp = getattr(w, "_crossop_row", None)
                cp = getattr(w, "_crossop_col", None)
                total_bytes += (0 if rp is None else rp.bytes()) \
                    + (0 if cp is None else cp.bytes())
                continue
            store = w.csr if w.is_sparse else w.dense
            if store is None or id(store) in seen_store or int(w.nnz) == 0:
                continue
            seen_store.add(id(store))
            scalars += int(w.nnz)
            total_bytes += _sparse_weight_storage_bytes(w)
    return scalars, total_bytes


# ---------------------------------------------------------------------------
# Verification: forward L-inf vs a fresh un-tied twin.
# ---------------------------------------------------------------------------
def verify_crossop_identical(tied, ref, streams) -> Tuple[float, bool]:
    """Return ``(max_linf, argmax_ok)`` comparing ``tied`` vs an un-tied ``ref``
    over ``streams`` (full block stack + LM head)."""
    worst = 0.0
    argmax_ok = True
    with torch.no_grad():
        for stream in streams:
            toks = torch.tensor([stream], dtype=torch.long)
            xt = tied.embed[toks].clone()
            xr = ref.embed[toks].clone()
            for bt, br in zip(tied.blocks, ref.blocks):
                xt = bt(xt); xr = br(xr)
            worst = max(worst, float((xt - xr).abs().max()))
            lt = F.linear(xt, tied.lm_head, tied.lm_bias)
            lr = F.linear(xr, ref.lm_head, ref.lm_bias)
            if not torch.equal(lt.argmax(-1), lr.argmax(-1)):
                argmax_ok = False
    return worst, argmax_ok
