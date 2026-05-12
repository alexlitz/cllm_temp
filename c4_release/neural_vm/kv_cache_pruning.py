"""KV cache pruning per ``docs/KV_CACHE_PRUNING_SPEC.md``.

Two complementary eviction rules running together on a ``LayerKVCache``:

1. **Key-similarity** (§2.1): for each token-type bucket, find pairs of
   positions ``i < j`` with cosine similarity of layer-0 averaged keys
   ``> tau`` (default ``0.99``). Evict ``i`` — the older entry.
2. **Zero-V** (§2.2): for each position, if the L2 norm of the layer-0
   averaged V row is ``< eps`` (default ``1e-6``), evict it. Under
   softmax1 a zero V row contributes zero to the attention output, so
   it is dead weight.

The pruner does **not** mutate ``ALiBi`` distance semantics: per spec §9
"surviving positions retain their original indices". The current
``TransformerKVCache`` (``kv_cache.py``) does not store per-position
absolute indices yet — it relies on the tensor's sequence dimension for
ALiBi distance via ``vm_step.AutoregressiveAttention.forward``. So the
pruner is **only safe to use** when the caller is comfortable with the
surviving positions being re-numbered ``0..S'-1`` (a stronger recency
bias, not a weaker one — old surviving entries appear closer to the
query, which is conservative for correctness but a regression vs the
blog spec). Wiring the pruner into the runner is deferred until
``TransformerKVCache`` adds ``pos_ids`` per the open question in
``docs/V5_V6_ATTENTION_AS_MEMORY_PLAN.md`` §5.

This module is **standalone** and exhaustively unit-tested. It is not
called from the runtime hot path yet (the runner still uses the
existing context-token-list eviction in ``key_similarity_eviction.py``
and the LRU MEM history in ``run_vm.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .vm_step import Token


# Token-type bucket names per ``KV_CACHE_PRUNING_SPEC.md`` §4.2.
# Buckets are searched independently for duplicate K; cross-bucket
# similarity is never computed (kept out by the very fact that the
# token types differ in their embedding).
_BUCKET_REG_PC = "REG_PC"
_BUCKET_REG_AX = "REG_AX"
_BUCKET_REG_SP = "REG_SP"
_BUCKET_REG_BP = "REG_BP"
_BUCKET_STACK0 = "STACK0"
_BUCKET_MEM_MARKER = "MEM_marker"
_BUCKET_STEP_END = "STEP_END"
_BUCKET_OTHER = "other"

# Token id → bucket name mapping for the marker positions.
# Prefix tokens (CODE_START/DATA_START/...) never reach the bucketing loop
# because ``bucket_positions`` only enumerates ``[prefix_len, cache_size)``.
_MARKER_TO_BUCKET: Dict[int, str] = {
    Token.REG_PC: _BUCKET_REG_PC,
    Token.REG_AX: _BUCKET_REG_AX,
    Token.REG_SP: _BUCKET_REG_SP,
    Token.REG_BP: _BUCKET_REG_BP,
    Token.STACK0: _BUCKET_STACK0,
    Token.MEM: _BUCKET_MEM_MARKER,
    Token.STEP_END: _BUCKET_STEP_END,
}


@dataclass
class PruneStats:
    """Per-call pruning statistics."""

    initial_size: int = 0
    final_size: int = 0
    evicted_by_similarity: int = 0
    evicted_by_zero_v: int = 0
    per_bucket: Dict[str, int] = field(default_factory=dict)

    @property
    def total_evicted(self) -> int:
        return self.initial_size - self.final_size


def bucket_positions(
    context: Sequence[int],
    prefix_len: int,
    cache_size: int,
) -> Dict[str, List[int]]:
    """Map each cached position to a token-type bucket.

    Positions ``[0, prefix_len)`` are the protected prefix and are not
    bucketed (they will never be evicted; see
    ``KV_CACHE_PRUNING_SPEC.md`` §6).

    Positions ``[prefix_len, cache_size)`` are bucketed by the source
    token id at insertion time. For marker tokens (REG_PC, REG_AX,
    REG_SP, REG_BP, STACK0, MEM, STEP_END) the bucket is the marker
    name. Every other position lands in ``other`` (this includes
    register value bytes, MEM addr/val bytes, etc. — their keys vary
    with the byte value, so cross-bucket near-duplicates are unlikely
    enough that we skip them and let zero-V eviction handle the rest).

    Args:
        context: Token id sequence representing the cache content
            (same order/length as the K/V sequence dimension).
        prefix_len: Number of leading positions that are protected.
        cache_size: Length of the K/V cache; bucketing is restricted to
            positions ``[prefix_len, cache_size)``.

    Returns:
        Dict ``bucket_name -> list of cache positions (ints)``.
    """
    buckets: Dict[str, List[int]] = {}
    upper = min(cache_size, len(context))
    for pos in range(prefix_len, upper):
        tok = context[pos]
        bucket = _MARKER_TO_BUCKET.get(tok, _BUCKET_OTHER)
        buckets.setdefault(bucket, []).append(pos)
    return buckets


def _layer0_key_fingerprint(layer_cache) -> Optional[torch.Tensor]:
    """Compute the per-position layer-0 K fingerprint (averaged over heads).

    Per spec §4.3 we use the layer-0 K because its similarity structure
    most-directly reflects the source token kind/value, and we average
    across heads for a stable per-token fingerprint.

    Returns ``None`` if the layer cache is empty.
    """
    return _layer_key_fingerprint(layer_cache)


def _layer_key_fingerprint(layer_cache) -> Optional[torch.Tensor]:
    """Compute per-position head-averaged K fingerprint for *this* layer.

    Same shape contract as ``_layer0_key_fingerprint`` (``[S, HD]``) — the
    only difference is the caller chooses which layer's K to read. Used by
    per-layer pruning (Phase A) where each layer's similarity decisions
    are based on that layer's own K, not layer 0's.
    """
    K = layer_cache.cached_k
    if K is None:
        return None
    return K[0].mean(dim=0)  # [S, HD]


def _layer_per_head_key(layer_cache) -> Optional[torch.Tensor]:
    """Per-head K (no head-averaging) for Phase B per-head fingerprints.

    Returns ``[H, S, HD]`` for batch 0, or ``None``.
    """
    K = layer_cache.cached_k
    if K is None:
        return None
    return K[0]  # [H, S, HD]


def _layer0_value_norms(layer_cache) -> Optional[torch.Tensor]:
    """Compute per-position L2 norms of the layer-0 V row (head-averaged).

    Returns ``None`` if the layer cache is empty.
    """
    return _layer_value_norms(layer_cache)


def _layer_value_norms(layer_cache) -> Optional[torch.Tensor]:
    """Per-position head-averaged V L2 norms for *this* layer (``[S]``)."""
    V = layer_cache.cached_v
    if V is None:
        return None
    avg = V[0].mean(dim=0)  # [S, HD]
    return avg.norm(dim=-1)  # [S]


def _layer_per_head_value_norms(layer_cache) -> Optional[torch.Tensor]:
    """Per-head V L2 norms for Phase B (``[H, S]``)."""
    V = layer_cache.cached_v
    if V is None:
        return None
    return V[0].norm(dim=-1)  # [H, S]


def _find_similarity_victims(
    K_fp: torch.Tensor,
    bucket_positions: List[int],
    tau: float,
) -> List[int]:
    """Find positions to evict within a single bucket via cosine sim.

    For each pair ``(i, j)`` with ``i < j`` and
    ``cos(K_fp[i], K_fp[j]) > tau``, mark position ``i`` for eviction.
    A position only needs to appear in *one* qualifying pair to be
    marked — the rule is "older of any pair survives".

    Args:
        K_fp: ``[S, HD]`` per-position key fingerprint for the *full*
            cache (the bucket positions index into this).
        bucket_positions: list of integer positions to consider.
        tau: similarity threshold.

    Returns:
        List of victim positions (subset of ``bucket_positions``).
    """
    if len(bucket_positions) < 2:
        return []

    pos_idx = torch.tensor(bucket_positions, dtype=torch.long, device=K_fp.device)
    K_b = K_fp.index_select(0, pos_idx)  # [B, HD]
    K_norm = F.normalize(K_b, p=2, dim=-1)
    # Cosine similarity matrix for the bucket.
    sim = K_norm @ K_norm.T  # [B, B]
    # Only consider strict upper triangle (i < j).
    B = sim.shape[0]
    mask = torch.triu(sim > tau, diagonal=1)  # [B, B] bool
    if not mask.any():
        return []
    # Any column j with mask[i, j]=True means position i (older) gets evicted.
    # Take the row indices where mask is True; those are the bucket-local
    # indices of victims.
    victim_local = mask.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
    return [bucket_positions[k] for k in victim_local]


def _victims_for_layer(
    K_fp: torch.Tensor,
    V_norms: Optional[torch.Tensor],
    buckets: Dict[str, List[int]],
    prefix_len: int,
    tau: float,
    eps: float,
    stats: PruneStats,
) -> set:
    """Compute victim positions for one (K_fp, V_norms) pair.

    Mutates ``stats`` in place for similarity / zero-V counts and per-bucket
    breakdown. Returns the set of victim positions.
    """
    victims: set = set()
    for name, positions in buckets.items():
        if len(positions) < 2:
            continue
        bucket_victims = _find_similarity_victims(K_fp, positions, tau)
        if bucket_victims:
            stats.per_bucket[name] = stats.per_bucket.get(name, 0) + len(bucket_victims)
            stats.evicted_by_similarity += len(bucket_victims)
            victims.update(bucket_victims)
    if V_norms is not None:
        zero_mask = V_norms < eps
        zero_positions = zero_mask.nonzero(as_tuple=False).flatten().tolist()
        zero_unprotected = [
            p for p in zero_positions if p >= prefix_len and p not in victims
        ]
        if zero_unprotected:
            stats.evicted_by_zero_v += len(zero_unprotected)
            victims.update(zero_unprotected)
    return victims


def prune_kv_cache(
    kv_cache,
    context: Sequence[int],
    prefix_len: int,
    tau: float = 0.99,
    eps: float = 1e-6,
    min_cache_size: int = 0,
    per_layer: bool = False,
    per_head: bool = False,
) -> Union[PruneStats, List[PruneStats]]:
    """Run two-rule pruning on a ``LayerKVCache`` (in-place).

    Implements ``docs/KV_CACHE_PRUNING_SPEC.md`` §2.1 + §2.2 + §4.2 plus
    the Phase A (per-layer) and Phase B (per-head) extensions.

    Three modes:

    - **Uniform** (default, ``per_layer=False``, ``per_head=False``):
      Single layer-0 fingerprint drives a keep-mask shared across all
      layers. Returns a single ``PruneStats``.
    - **Per-layer** (``per_layer=True``): Each layer computes its own
      fingerprint from its own K (not layer 0's) and its own keep-mask.
      Different layers may shrink to different sizes ``S_i'``. Returns a
      ``List[PruneStats]``, one per layer.
    - **Per-head** (``per_head=True``): Each (layer, head) pair gets its
      own keep decision. The K/V tensors stay full-size (one row per
      original cached position) but ``per_head_keep_mask`` flips entries
      to ``False`` so the attention layer masks those scores to ``-inf``.
      Hard global eviction only fires if *all* heads in *all* layers
      agree to drop a row. Returns a ``List[PruneStats]`` per layer.

    ``per_layer`` and ``per_head`` are independent. If both are False, the
    legacy uniform path runs. If both are True, per-head fingerprints
    drive each layer's per-head decisions and a global keep-mask still
    rides on AND-across-heads-and-layers.

    Algorithm (uniform):

    1. Bucket each cached position by source token type using
       ``context`` (the runner's live token list, which the caller
       must have kept in sync with ``cache_size``).
    2. For each bucket, find positions whose layer-0 K fingerprint is
       cosine-similar (``> tau``) to a newer position; mark the older
       one for eviction.
    3. Find positions whose layer-0 V row has L2 norm below ``eps``;
       mark them for eviction.
    4. Build a ``keep_idx`` LongTensor and call ``index_select(dim=2,
       index=keep_idx)`` on every layer's ``cached_k``/``cached_v``.

    The prefix region ``[0, prefix_len)`` is never evicted.

    Args:
        kv_cache: ``LayerKVCache`` instance (``kv_cache.caches[i]`` is
            the per-layer ``TransformerKVCache``).
        context: token-id sequence with length ``>= cache_size``; index
            ``i`` in ``context`` is the source token id of cached
            position ``i``.
        prefix_len: number of leading protected positions.
        tau: cosine-similarity threshold for §2.1.
        eps: L2-norm threshold for §2.2.
        min_cache_size: skip pruning if cache is below this size.
            Spec §3.2 recommends ``256`` for runtime callers; defaulting
            to ``0`` here keeps the function policy-free so the runner
            integration (and unit tests) can choose.
        per_layer: when True, compute per-layer fingerprints and return a
            ``List[PruneStats]``.
        per_head: when True, compute per-head fingerprints and apply via
            ``per_head_keep_mask`` (soft eviction). Returns a
            ``List[PruneStats]``.

    Returns:
        ``PruneStats`` (uniform) or ``List[PruneStats]`` (per-layer /
        per-head). In per-head mode the stats record "soft" evictions
        — the cache rows are not removed unless all heads agree.
    """
    if not kv_cache.caches:
        empty = PruneStats()
        return [empty] if (per_layer or per_head) else empty

    layer0 = kv_cache.caches[0]
    if layer0.cached_k is None:
        empty = PruneStats()
        return (
            [PruneStats() for _ in kv_cache.caches]
            if (per_layer or per_head)
            else empty
        )

    cache_size = layer0.cache_size

    if cache_size <= min_cache_size or cache_size <= prefix_len:
        if per_layer or per_head:
            return [
                PruneStats(initial_size=cache_size, final_size=cache_size)
                for _ in kv_cache.caches
            ]
        return PruneStats(initial_size=cache_size, final_size=cache_size)

    # Bucket positions once — the bucket assignment is purely about
    # source-token type, not the K vectors, so it's shared across modes.
    buckets = bucket_positions(context, prefix_len, cache_size)

    if per_head:
        return _prune_per_head(
            kv_cache, buckets, prefix_len, tau, eps
        )

    if per_layer:
        return _prune_per_layer(
            kv_cache, buckets, prefix_len, tau, eps
        )

    # Uniform path: drive everything off layer 0.
    stats = PruneStats(initial_size=cache_size, final_size=cache_size)
    K_fp = _layer_key_fingerprint(layer0)
    if K_fp is None:
        return stats

    V_norms = _layer_value_norms(layer0)
    victims = _victims_for_layer(
        K_fp, V_norms, buckets, prefix_len, tau, eps, stats
    )
    if not victims:
        return stats

    # Spec §3.3: keep-mask must be identical across layers. The
    # ``LayerKVCache.prune`` API is the single chokepoint that enforces
    # that invariant and keeps ``cached_pos_ids`` in lock-step with K/V.
    keep = [p for p in range(cache_size) if p not in victims]
    device = layer0.cached_k.device
    keep_idx = torch.tensor(keep, dtype=torch.long, device=device)
    new_size = kv_cache.prune(keep_idx)

    stats.final_size = new_size
    return stats


def _prune_per_layer(
    kv_cache,
    buckets: Dict[str, List[int]],
    prefix_len: int,
    tau: float,
    eps: float,
) -> List[PruneStats]:
    """Phase A: each layer computes its own fingerprint + keep-mask.

    Each layer can shrink to its own ``S_i'``. ``LayerKVCache.prune``
    handles the per-layer index_select including ``cached_pos_ids``, so
    ``AutoregressiveAttention.forward`` keeps reading the right pos_ids
    per layer.
    """
    per_layer_stats: List[PruneStats] = []
    per_layer_keep: List[torch.Tensor] = []
    any_eviction = False

    for layer_cache in kv_cache.caches:
        stats = PruneStats()
        if layer_cache.cached_k is None:
            per_layer_stats.append(stats)
            # Pass an empty index tensor; LayerKVCache.prune treats this
            # layer as already-empty and leaves it alone.
            per_layer_keep.append(
                torch.zeros(0, dtype=torch.long)
            )
            continue

        layer_size = layer_cache.cache_size
        stats.initial_size = layer_size
        stats.final_size = layer_size

        K_fp = _layer_key_fingerprint(layer_cache)
        V_norms = _layer_value_norms(layer_cache)
        if K_fp is None:
            per_layer_stats.append(stats)
            # Keep everything (identity).
            keep_idx = torch.arange(
                layer_size, dtype=torch.long, device=layer_cache.cached_k.device
            )
            per_layer_keep.append(keep_idx)
            continue

        victims = _victims_for_layer(
            K_fp, V_norms, buckets, prefix_len, tau, eps, stats
        )
        device = layer_cache.cached_k.device
        if victims:
            keep = [p for p in range(layer_size) if p not in victims]
            keep_idx = torch.tensor(keep, dtype=torch.long, device=device)
            any_eviction = True
        else:
            keep_idx = torch.arange(layer_size, dtype=torch.long, device=device)
        per_layer_keep.append(keep_idx)
        per_layer_stats.append(stats)

    if any_eviction:
        new_sizes = kv_cache.prune(per_layer_keep)
        for s, ns in zip(per_layer_stats, new_sizes):
            s.final_size = ns

    return per_layer_stats


def _prune_per_head(
    kv_cache,
    buckets: Dict[str, List[int]],
    prefix_len: int,
    tau: float,
    eps: float,
) -> List[PruneStats]:
    """Phase B: per-(layer, head) decisions via ``per_head_keep_mask``.

    For each head h in each layer L:
    - Compute per-head K fingerprint (no head-average).
    - Run the §2.1 / §2.2 rules at head granularity.
    - Soft-evict via ``per_head_keep_mask[B, h, S] = False`` so the
      forward pass masks those scores to ``-inf``.

    A row is hard-evicted (deleted from K/V) only when every head in
    every layer agrees to drop it — i.e. for position ``s``,
    ``all_layers ∧_h per_head_keep_mask[:, h, s] == False``. This keeps
    memory bounded while still allowing different heads to disagree on
    cheap rows.
    """
    per_layer_stats: List[PruneStats] = []

    # AND across all (layer, head) of the keep mask. Initialized to "keep"
    # and flipped to False at any (layer, head) that votes evict.
    layer0 = kv_cache.caches[0]
    cache_size = layer0.cache_size
    device = layer0.cached_k.device

    global_evict_vote = torch.ones(
        cache_size, dtype=torch.bool, device=device
    )  # True = at least one head voted to keep.

    for layer_cache in kv_cache.caches:
        stats = PruneStats()
        if layer_cache.cached_k is None:
            per_layer_stats.append(stats)
            continue

        layer_size = layer_cache.cache_size
        stats.initial_size = layer_size
        stats.final_size = layer_size  # per-head doesn't shrink rows.

        K_per_head = _layer_per_head_key(layer_cache)  # [H, S, HD]
        V_norms_per_head = _layer_per_head_value_norms(layer_cache)  # [H, S]

        if K_per_head is None:
            per_layer_stats.append(stats)
            continue

        H = K_per_head.shape[0]
        S = K_per_head.shape[1]
        dev = K_per_head.device

        # Build or fetch per-head keep mask: [B, H, S]; initialize to all True.
        if (
            layer_cache.per_head_keep_mask is None
            or layer_cache.per_head_keep_mask.shape[-1] != S
        ):
            B = layer_cache.cached_k.shape[0]
            layer_cache.per_head_keep_mask = torch.ones(
                B, H, S, dtype=torch.bool, device=dev
            )

        # Decisions per head.
        for h in range(H):
            K_fp_h = K_per_head[h]  # [S, HD]
            V_norms_h = V_norms_per_head[h] if V_norms_per_head is not None else None
            head_victims = _victims_for_layer(
                K_fp_h, V_norms_h, buckets, prefix_len, tau, eps, stats
            )
            if head_victims:
                idx = torch.tensor(
                    sorted(head_victims), dtype=torch.long, device=dev
                )
                layer_cache.per_head_keep_mask[:, h, idx] = False

        # Update global evict vote: a position is hard-evictable only when
        # *every* head in this layer voted False. We compute the per-layer
        # AND of "did all heads vote evict?" = ``~any_head_keeps``.
        # Across layers we AND those together (a position survives globally
        # if any (layer, head) wants to keep it).
        any_head_keeps_layer = layer_cache.per_head_keep_mask[0].any(dim=0)  # [S]
        global_evict_vote = global_evict_vote & any_head_keeps_layer

        per_layer_stats.append(stats)

    # global_evict_vote[s] = True means at least one (layer, head) wants
    # to keep position s. We hard-evict positions where the vote is False
    # (i.e. all heads in all layers said evict), and only outside the
    # prefix region.
    keep_bool = global_evict_vote.clone()
    if prefix_len > 0:
        keep_bool[:prefix_len] = True  # protected prefix.

    # Only hard-evict if at least one position is fully out-voted.
    if (~keep_bool).any():
        keep_idx = keep_bool.nonzero(as_tuple=False).flatten()
        new_size = kv_cache.prune(keep_idx)
        for s in per_layer_stats:
            if s.initial_size > 0:
                s.final_size = new_size

    return per_layer_stats
