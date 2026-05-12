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
from typing import Dict, List, Optional, Sequence

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
    K = layer_cache.cached_k
    if K is None:
        return None
    # K: [B, H, S, HD]. Take batch 0 and mean across heads.
    return K[0].mean(dim=0)  # [S, HD]


def _layer0_value_norms(layer_cache) -> Optional[torch.Tensor]:
    """Compute per-position L2 norms of the layer-0 V row (head-averaged).

    Returns ``None`` if the layer cache is empty.
    """
    V = layer_cache.cached_v
    if V is None:
        return None
    # V: [B, H, S, HD]. Take batch 0, mean across heads, then L2 norm.
    avg = V[0].mean(dim=0)  # [S, HD]
    return avg.norm(dim=-1)  # [S]


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


def prune_kv_cache(
    kv_cache,
    context: Sequence[int],
    prefix_len: int,
    tau: float = 0.99,
    eps: float = 1e-6,
    min_cache_size: int = 0,
) -> PruneStats:
    """Run two-rule pruning on a ``LayerKVCache`` (in-place).

    Implements ``docs/KV_CACHE_PRUNING_SPEC.md`` §2.1 + §2.2 + §4.2.

    Algorithm:

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

    The keep-mask is shared across all layers (spec §3.3).
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

    Returns:
        ``PruneStats`` with eviction counts and per-bucket breakdown.
    """
    stats = PruneStats()
    if not kv_cache.caches:
        return stats

    layer0 = kv_cache.caches[0]
    if layer0.cached_k is None:
        return stats

    cache_size = layer0.cache_size
    stats.initial_size = cache_size
    stats.final_size = cache_size

    if cache_size <= min_cache_size or cache_size <= prefix_len:
        return stats

    # 1. Bucket positions by source token.
    buckets = bucket_positions(context, prefix_len, cache_size)

    # 2. Per-bucket key-similarity eviction.
    K_fp = _layer0_key_fingerprint(layer0)
    if K_fp is None:
        return stats

    victims: set = set()
    for name, positions in buckets.items():
        if len(positions) < 2:
            continue
        bucket_victims = _find_similarity_victims(K_fp, positions, tau)
        if bucket_victims:
            stats.per_bucket[name] = len(bucket_victims)
            stats.evicted_by_similarity += len(bucket_victims)
            victims.update(bucket_victims)

    # 3. Zero-V eviction.
    V_norms = _layer0_value_norms(layer0)
    if V_norms is not None:
        # Skip the protected prefix range.
        zero_mask = V_norms < eps
        zero_positions = zero_mask.nonzero(as_tuple=False).flatten().tolist()
        zero_unprotected = [p for p in zero_positions if p >= prefix_len and p not in victims]
        if zero_unprotected:
            stats.evicted_by_zero_v = len(zero_unprotected)
            victims.update(zero_unprotected)

    if not victims:
        return stats

    # 4. Build keep-mask + delegate the layer-uniform apply to LayerKVCache.
    # Spec §3.3: keep-mask must be identical across layers. The
    # ``LayerKVCache.prune`` API is the single chokepoint that enforces
    # that invariant and keeps ``cached_pos_ids`` in lock-step with K/V.
    keep = [p for p in range(cache_size) if p not in victims]
    device = layer0.cached_k.device
    keep_idx = torch.tensor(keep, dtype=torch.long, device=device)
    new_size = kv_cache.prune(keep_idx)

    stats.final_size = new_size
    return stats
