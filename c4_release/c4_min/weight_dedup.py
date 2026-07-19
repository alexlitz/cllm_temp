"""Weight-TYING (deduplication) for the c4_min compact / sparse pure-forward VM.

The compact model (``compact_alloc.build_compact_sparse_streaming``) already
removes the dead-zero padding (Fix #1 dim-sharing + Fix #2 ragged FFN + Fix #3
COO storage).  What it does NOT remove is *replication of identical weight
tensors*: the model is a repetitive VM whose blocks re-emit the SAME sub-program
over and over.

  * The base-16 long-division (DIV/MOD) stack is an 8-iteration loop
    (``compile_divmod_blocks``): every iteration re-emits ``shift / gteq / qdigit
    / qcopy / qb / 9x qb-carry / RN x sub-nibble / r2r``.  Because each of those
    per-nibble corrector blocks writes into the SAME residual dims regardless of
    the iteration, the compaction pass leaves them BYTE-IDENTICAL — e.g. the QB
    carry-round block ``_carry_round_block(QB, QB, RN)`` is emitted 8 iters x 9
    rounds = 48 identical copies of its W_up / W_gate / W_down.
  * The ``kb-precompute`` (KB[k]=k*b) blocks repeat one per-nibble tile 6x.
  * The mul-carry blocks repeat their tile 3-4x.

None of those copies carry unique information: they are the SAME numbers.  This
pass finds every set of byte-identical weight tensors and *ties* the set to a
single shared tensor object (``N`` block references to ``1`` stored copy) — pure
weight-tying, so ``model.forward`` reads exactly the same values it read before
and the greedy (argmax) decode is byte-identical (L-inf=0 in ``dense_kernel``
mode).

This is the whole-tensor analogue of the per-nibble sub-tensor tie prototyped on
the ``nibble-expert-dedup`` branch (``nibble_bitwise_dedup``): there the 16
per-nibble lanes of ONE bitwise table were tied; here the compaction pass has
already collapsed the intra-tensor lanes onto shared dims, so the surviving
replication is at the WHOLE-TENSOR (per-block) granularity, which is what this
pass ties.

The tie is applied IN PLACE on a built :class:`sparse_forward.SparseTransformer`:
for every group of byte-identical ``SparseWeight`` objects we point every member's
storage (``.csr`` / ``.dense``) at the group representative's storage, and for
byte-identical dense bias vectors we alias the tensor.  ``id()``-level sharing
means the unique *stored* tensor count drops by the replication factor while the
logical model is unchanged.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Tensor fingerprinting.
# ---------------------------------------------------------------------------
def _dense_of(w) -> torch.Tensor:
    """Materialise a :class:`sparse_forward.SparseWeight`'s dense tensor."""
    if not w.is_sparse:
        return w.dense
    return w.csr.to_dense()


def _fingerprint(t: torch.Tensor) -> Tuple:
    """A byte-exact fingerprint ``(shape, dtype, sha256)`` of a tensor's bytes.

    Two tensors with the same fingerprint are byte-identical (same dtype/shape
    and same raw value bytes), so tying them is provably value-preserving.
    """
    t = t.detach().contiguous().cpu()
    return (tuple(t.shape), str(t.dtype),
            hashlib.sha256(t.numpy().tobytes()).hexdigest())


def _sparse_weight_storage_bytes(w) -> int:
    """Stored bytes for one :class:`SparseWeight` (its OWN storage, not shared).

    Mirrors :meth:`SparseWeight.storage_bytes` (CSR = crow+col int64 + val fp32;
    dense = numel*4).  Used to price the tie savings honestly.
    """
    return int(w.storage_bytes())


# ---------------------------------------------------------------------------
# Dedup stats.
# ---------------------------------------------------------------------------
@dataclass
class DedupStats:
    # weight-tensor accounting (2-D attn/ffn matrices only)
    weight_tensors_before: int = 0
    unique_weight_tensors_after: int = 0
    weight_tensors_zero: int = 0            # all-zero tensors (folded to one shared 0)
    # bias-vector accounting (1-D FFN biases + alibi)
    bias_tensors_before: int = 0
    unique_bias_tensors_after: int = 0
    # nonzero-weight accounting
    nonzero_before: int = 0                 # summed over every stored tensor
    nonzero_after: int = 0                  # summed over UNIQUE stored tensors
    # storage accounting (bytes of the WEIGHT tensors' own storage)
    weight_storage_bytes_before: int = 0
    weight_storage_bytes_after: int = 0
    # per-group detail (rep_factor, kind, nnz, example block names)
    groups: List[Tuple[int, str, int, List[str]]] = field(default_factory=list)

    @property
    def nonzero_saved(self) -> int:
        return self.nonzero_before - self.nonzero_after

    @property
    def weight_storage_mb_before(self) -> float:
        return self.weight_storage_bytes_before / 1e6

    @property
    def weight_storage_mb_after(self) -> float:
        return self.weight_storage_bytes_after / 1e6

    def summary(self) -> str:
        rep = self.weight_tensors_before / max(1, self.unique_weight_tensors_after)
        wsb, wsa = self.weight_storage_bytes_before, self.weight_storage_bytes_after
        lines = [
            "=== weight-dedup (weight-tying) ===",
            f"  2-D weight tensors : {self.weight_tensors_before:5d} "
            f"-> {self.unique_weight_tensors_after:5d} unique ({rep:.2f}x)"
            f"   [{self.weight_tensors_zero} all-zero folded to 1 shared/shape]",
            f"  1-D bias vectors   : {self.bias_tensors_before:5d} "
            f"-> {self.unique_bias_tensors_after:5d} unique",
            f"  nonzero weights    : {self.nonzero_before:8d} "
            f"-> {self.nonzero_after:8d} unique "
            f"(saved {self.nonzero_saved}, "
            f"{100 * self.nonzero_saved / max(1, self.nonzero_before):.1f}%)",
            f"  weight storage     : {self.weight_storage_mb_before:8.2f} MB "
            f"-> {self.weight_storage_mb_after:8.2f} MB "
            f"(saved {(wsb - wsa) / 1e6:.2f} MB, "
            f"{100 * (wsb - wsa) / max(1, wsb):.1f}%)",
        ]
        if self.groups:
            lines.append("  top tied groups (rep x kind, nnz each, example blocks):")
            for rep_n, kind, nnz, ex in self.groups[:15]:
                lines.append(f"    x{rep_n:4d}  {kind:<7} nnz={nnz:6d}  "
                             f"saves={nnz * (rep_n - 1):7d}  e.g. {ex[:2]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The tie pass.
# ---------------------------------------------------------------------------
def _iter_weight_slots(sparse):
    """Yield ``(block_idx, kind, sparse_weight)`` for every 2-D weight tensor."""
    for bi, b in enumerate(sparse.blocks):
        yield bi, "W_q", b.attn.W_q
        yield bi, "W_k", b.attn.W_k
        yield bi, "W_v", b.attn.W_v
        yield bi, "W_o", b.attn.W_o
        yield bi, "W_up", b.ffn.W_up
        yield bi, "W_gate", b.ffn.W_gate
        yield bi, "W_down", b.ffn.W_down


def _iter_bias_slots(sparse):
    """Yield ``(block_idx, kind, tensor, setter)`` for every 1-D FFN/alibi vector."""
    for bi, b in enumerate(sparse.blocks):
        ff = b.ffn
        yield bi, "b_up", ff.b_up, (lambda t, ff=ff: setattr(ff, "b_up", t))
        yield bi, "b_gate", ff.b_gate, (lambda t, ff=ff: setattr(ff, "b_gate", t))
        yield bi, "b_down", ff.b_down, (lambda t, ff=ff: setattr(ff, "b_down", t))
        yield bi, "alibi", b.attn.alibi_slopes, \
            (lambda t, at=b.attn: setattr(at, "alibi_slopes", t))


def dedup_sparse_transformer(sparse, L=None, tie_zeros: bool = True) -> DedupStats:
    """Tie byte-identical weight tensors in a built ``SparseTransformer`` IN PLACE.

    Every group of byte-identical 2-D weight tensors (attn Q/K/V/O + ffn
    up/gate/down) is pointed at ONE shared storage tensor (the group
    representative), and every group of byte-identical 1-D bias / alibi vectors is
    aliased likewise.  The forward is unchanged (each block still reads the same
    numbers), so greedy decode is byte-identical.  Returns :class:`DedupStats`.

    ``tie_zeros``: also fold every all-zero weight tensor of a given SHAPE onto a
    single shared zero tensor (the c4_min compact divmod model has 301 of 304
    all-zero attention blocks, each storing an empty CSR of the same shape;
    sharing them costs one empty CSR per shape instead of one per block).

    ``L`` (optional) supplies ``L._block_names`` for readable group examples.
    """
    names = getattr(L, "_block_names", None) if L is not None else None
    stats = DedupStats()

    # ---- 2-D weights: group by fingerprint ----
    wgroups: Dict[Tuple, List[Tuple[int, str, object]]] = defaultdict(list)
    for bi, kind, w in _iter_weight_slots(sparse):
        stats.weight_tensors_before += 1
        stats.nonzero_before += int(w.nnz)
        stats.weight_storage_bytes_before += _sparse_weight_storage_bytes(w)
        if int(w.nnz) == 0 and tie_zeros:
            key = ("ZERO", w.is_sparse, w.out_dim, w.in_dim)
        else:
            key = _fingerprint(_dense_of(w))
        wgroups[key].append((bi, kind, w))

    # Tie each group: point every member's storage at the representative's.
    n_zero_tensors = 0
    for key, members in wgroups.items():
        rep_bi, rep_kind, rep_w = members[0]
        is_zero = isinstance(key, tuple) and len(key) and key[0] == "ZERO"
        if is_zero:
            n_zero_tensors += len(members)
        for bi, kind, w in members[1:]:
            _tie_sparse_weight(w, rep_w)
        stats.unique_weight_tensors_after += 1
        stats.nonzero_after += int(rep_w.nnz)
        stats.weight_storage_bytes_after += _sparse_weight_storage_bytes(rep_w)
        if len(members) > 1 and int(rep_w.nnz) > 0:
            stats.groups.append((len(members), rep_kind, int(rep_w.nnz),
                                 _example_block_names(members, names)))
    stats.weight_tensors_zero = n_zero_tensors
    stats.groups.sort(key=lambda g: -(g[0] - 1) * g[2])   # by nnz saved

    # ---- 1-D biases + alibi ----
    bgroups: Dict[Tuple, List] = defaultdict(list)
    for bi, kind, t, setter in _iter_bias_slots(sparse):
        stats.bias_tensors_before += 1
        bgroups[_fingerprint(t)].append((bi, kind, t, setter))
    for key, members in bgroups.items():
        rep = members[0][2]
        for bi, kind, t, setter in members[1:]:
            setter(rep)                       # alias every duplicate onto the rep
        stats.unique_bias_tensors_after += 1

    return stats


def _tie_sparse_weight(w, rep) -> None:
    """Point ``w``'s storage at ``rep``'s storage (byte-identical -> value-safe).

    ``w`` and ``rep`` are byte-identical :class:`SparseWeight` objects; after this
    ``w`` holds NO private large tensor — its ``.csr`` / ``.dense`` IS ``rep``'s.
    ``w`` keeps its own tiny scalar metadata (out_dim/in_dim/nnz/mode) so the
    forward path is untouched.
    """
    assert w.is_sparse == rep.is_sparse
    assert w.out_dim == rep.out_dim and w.in_dim == rep.in_dim
    if w.is_sparse:
        w.csr = rep.csr
        w.dense = None
    else:
        w.dense = rep.dense
        w.csr = None


def _example_block_names(members, names) -> List[str]:
    seen, uniq = set(), []
    for bi, kind, _w in members:
        nm = names[bi] if (names and bi < len(names)) else f"blk{bi}"
        if nm not in seen:
            seen.add(nm)
            uniq.append(nm)
    return uniq


# ---------------------------------------------------------------------------
# Unique-storage accounting by tensor identity (post-tie truth check).
# ---------------------------------------------------------------------------
def count_unique_stored_tensors(sparse) -> Tuple[int, int, int]:
    """Count DISTINCT stored tensors by ``id()`` after a tie (truth check).

    Returns ``(n_weight_tensor_refs, n_unique_weight_storages, unique_nnz)``: the
    number of 2-D weight REFERENCES vs the number of DISTINCT underlying storage
    tensors (by ``id()``), plus the summed nnz over the distinct storages.  A tie
    is real iff ``n_unique_weight_storages < n_weight_tensor_refs``.
    """
    refs = 0
    seen: Dict[int, int] = {}
    unique_nnz = 0
    for _bi, _kind, w in _iter_weight_slots(sparse):
        refs += 1
        store = w.csr if w.is_sparse else w.dense
        sid = id(store)
        if sid not in seen:
            seen[sid] = int(w.nnz)
            unique_nnz += int(w.nnz)
    return refs, len(seen), unique_nnz


# ---------------------------------------------------------------------------
# Verification helpers (argmax + L-inf on a battery / corpus sample).
# ---------------------------------------------------------------------------
def verify_forward_identical(sparse_a, sparse_b, token_streams,
                             atol: float = 0.0) -> Tuple[float, bool]:
    """Compare two SparseTransformers' ``forward`` over ``token_streams``.

    Returns ``(max_linf, argmax_identical)``.  ``dense_kernel`` mode makes the tie
    L-inf=0 (same stored bytes, same GEMM); ``sparse_mm`` mode may differ only in
    fp accumulation order, in which case ``atol`` allows a tolerance while argmax
    identity is still asserted (the greedy-decode correctness criterion).
    """
    max_linf = 0.0
    argmax_ok = True
    with torch.no_grad():
        for stream in token_streams:
            toks = torch.tensor([stream], dtype=torch.long)
            la = sparse_a.forward(toks)
            lb = sparse_b.forward(toks)
            linf = float((la - lb).abs().max())
            max_linf = max(max_linf, linf)
            if not torch.equal(la.argmax(-1), lb.argmax(-1)):
                argmax_ok = False
    return max_linf, argmax_ok
