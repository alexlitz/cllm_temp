"""Phase 7.F.2: Runtime KV eviction pass.

This module turns the static :class:`LivenessReport` produced by
:mod:`neural_vm.kv_liveness_analyzer` into a runtime artifact attached
to each :class:`PureAttention` (or :class:`AutoregressiveAttention`)
module. At step boundaries the attention forward pass calls
:func:`apply_eviction`, which zeros (or marks-unused) the K/V cache rows
that the analyzer has guaranteed dead at that step.

Design constraints
------------------

* **Deterministic.** Eviction decisions come exclusively from the
  precomputed :class:`KVEvictionState`. They are *not* a function of any
  runtime tensor, so the spec-decode and main-decode paths reach the
  same decisions when given the same step index.
* **Byte-identity by default.** :data:`KVEvictionPolicy.OFF` is the
  default; no caller observes any change unless the policy is opted in
  via the ``compile_full_vm`` flag.
* **Safe-by-construction.** ``apply_eviction`` only zeros rows that the
  analyzer flagged as guaranteed dead. If the analyzer is conservative
  the worst outcome is "no rows zeroed". The byte-identity gate in
  :mod:`tests.test_kv_eviction` catches any analyzer bug that marks a
  still-live row dead.
* **Read-only API for the cache.** The eviction pass touches
  ``K_cache`` / ``V_cache`` only when these attributes exist on the
  attention module. PureAttention (which recomputes K/V from the
  residual every forward) treats the call as a no-op.

The :class:`KVEvictionPolicy` enum is intentionally small so future
policies (e.g. ``DYNAMIC_LIVENESS``, ``SCORE_BASED``) can be added
without breaking existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .kv_liveness_analyzer import KVEntry, LivenessReport


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------


class KVEvictionPolicy(str, Enum):
    """Runtime KV-eviction policy.

    Members:
        OFF: No eviction. Default; preserves byte-identity with all
            historical baselines.
        STATIC_LIVENESS: Drive eviction from a precomputed
            :class:`KVEvictionState` built at compile time from the
            static liveness analyzer (Phase 7.F.1).
    """

    OFF = "off"
    STATIC_LIVENESS = "static_liveness"

    @classmethod
    def from_str(cls, name: Optional[str]) -> "KVEvictionPolicy":
        """Parse a CLI / config string, accepting hyphenated aliases."""

        if name is None:
            return cls.OFF
        norm = name.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == norm:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown KV eviction policy {name!r}; valid options: {valid}"
        )


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------


@dataclass
class KVEvictionState:
    """Per-attention compile-time eviction artifact.

    Attributes:
        policy: The active policy. ``OFF`` makes :func:`apply_eviction`
            a no-op even if the state object is attached.
        layer_idx: The layer this state was built for. ``None`` matches
            every layer (used by tests that build a single state).
        evictable_positions_at_step: ``step -> set of positions`` (per
            ``(layer_idx, head)``) that the analyzer marked guaranteed
            dead at the end of step ``step``. Positions are absolute
            cache row indices.
        evicted_positions: positions whose K/V rows were last zeroed by
            :func:`apply_eviction` — used by tests to inspect history
            and by determinism gates to compare spec/main decode runs.
        total_evictions: monotonic counter of (step, position) zero
            events. Diagnostic only; never affects behaviour.
    """

    policy: KVEvictionPolicy = KVEvictionPolicy.OFF
    layer_idx: Optional[int] = None
    evictable_positions_at_step: Dict[int, Set[int]] = field(default_factory=dict)
    evicted_positions: Set[int] = field(default_factory=set)
    total_evictions: int = 0

    def positions_at_step(self, step_idx: int) -> Set[int]:
        """Return the (possibly empty) set of positions evictable after step."""

        return self.evictable_positions_at_step.get(int(step_idx), set())

    def is_active(self) -> bool:
        """``True`` when the policy is not OFF and there's anything to evict."""

        return (
            self.policy is not KVEvictionPolicy.OFF
            and bool(self.evictable_positions_at_step)
        )


# ---------------------------------------------------------------------------
# Builder from LivenessReport
# ---------------------------------------------------------------------------


def build_state_from_report(
    report: "LivenessReport",
    *,
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
    policy: KVEvictionPolicy = KVEvictionPolicy.STATIC_LIVENESS,
) -> KVEvictionState:
    """Project a :class:`LivenessReport` onto one (layer, head) slot.

    The analyzer reports per-(layer, head, dim_name) entries; the runtime
    cache however indexes by absolute position only (each K/V row covers
    *all* dims for one token). A row is therefore safe to evict only if
    *every* dim slot at that position has been declared evictable. This
    builder applies that conservative AND.

    Parameters
    ----------
    report:
        The :class:`LivenessReport` from
        :func:`neural_vm.kv_liveness_analyzer.analyze_kv_liveness`.
    layer_idx:
        Restrict to this layer. ``None`` ignores the filter.
    head_idx:
        Restrict to this head. ``None`` ignores the filter.
    policy:
        Policy stamped onto the resulting state.

    Returns
    -------
    KVEvictionState
        Per-step evictable position sets ready to be attached to a
        ``PureAttention`` (or ``AutoregressiveAttention``) module.
    """

    # Index entries by (step, position) so we can reason about whether
    # *every* dim slot at that position was declared evictable.
    per_step_dim_count: Dict[int, Dict[int, Set[str]]] = {}
    all_dims: Set[str] = set()
    for step, entries in report.evictable_at_step.items():
        per_step_dim_count.setdefault(step, {})
        for entry in entries:
            if layer_idx is not None and entry.layer != layer_idx:
                continue
            if head_idx is not None and entry.head != head_idx:
                continue
            per_step_dim_count[step].setdefault(entry.position, set()).add(
                entry.dim_name
            )
            all_dims.add(entry.dim_name)

    # The denominator for "every dim slot is dead" comes from the
    # universe of dim names the analyzer considered. We approximate it
    # by the set of dim names that appear in *any* step's evictable
    # bucket plus the conservative-kept bucket; this matches the
    # analyzer's ``all_dim_names`` universe.
    for entry in report.cycle_conservative:
        if layer_idx is not None and entry.layer != layer_idx:
            continue
        if head_idx is not None and entry.head != head_idx:
            continue
        all_dims.add(entry.dim_name)

    # Phase 7.F.6: project the AND universe onto dims that any attention
    # head's K/V projection actually samples. Dims that no attention
    # head reads contribute zero columns to W_K / W_V; their residual
    # value can be anything (including the cached stale value) without
    # affecting the K/V dot product. So those dims don't need to be in
    # the per-row AND.
    #
    # Prefer the per-layer attention universe when the caller supplied a
    # specific ``layer_idx`` — the K/V projection at layer L only reads
    # the dims declared by ops pinned to L (other layers' attn-read dims
    # don't affect L's row). When the layer-specific universe is empty
    # (e.g. attn ops are dynamically routed and not annotated with
    # ``layer_idx``), fall back to the global universe; that's still
    # tighter than the analyzer-derived ``all_dims`` (which also folds
    # in FFN-only writes).
    attn_universe_by_layer = getattr(
        report, "attention_read_dim_names_by_layer", None
    )
    attn_universe_global = getattr(report, "attention_read_dim_names", None)
    attn_universe = None
    if attn_universe_by_layer and layer_idx is not None:
        attn_universe = attn_universe_by_layer.get(layer_idx)
    if not attn_universe and attn_universe_global:
        attn_universe = attn_universe_global
    if attn_universe:
        all_dims = set(attn_universe)

    evictable_positions_at_step: Dict[int, Set[int]] = {}
    if all_dims:
        for step, by_pos in per_step_dim_count.items():
            positions: Set[int] = set()
            for position, dims_dead in by_pos.items():
                # Conservative AND: every attention-K/V-read dim at this
                # position must have been declared evictable. Other dims
                # (FFN-only) contribute no K/V projection and are safe
                # to leave in the (possibly zeroed) row.
                if dims_dead >= all_dims:
                    positions.add(position)
            if positions:
                evictable_positions_at_step[step] = positions

    return KVEvictionState(
        policy=policy,
        layer_idx=layer_idx,
        evictable_positions_at_step=evictable_positions_at_step,
    )


# ---------------------------------------------------------------------------
# Runtime application
# ---------------------------------------------------------------------------


def apply_eviction(attn, state: Optional[KVEvictionState], step_idx: int) -> int:
    """Zero out the K/V cache rows that are dead at the end of ``step_idx``.

    The eviction targets ``attn.K_cache`` and ``attn.V_cache`` when those
    attributes exist (the runtime attention paths populate them on each
    forward via the :class:`TransformerKVCache` adapter). Modules that
    do not retain a KV cache between calls (e.g. plain
    :class:`PureAttention` in train-style forward) treat the call as a
    no-op — the eviction state is still consulted so determinism /
    decision sets stay observable for tests.

    Parameters
    ----------
    attn:
        The attention module (PureAttention or AutoregressiveAttention).
        Attributes consulted (if present): ``K_cache``, ``V_cache``,
        ``kv_cache`` (a :class:`TransformerKVCache`).
    state:
        Precomputed :class:`KVEvictionState`. ``None`` or a state with
        ``KVEvictionPolicy.OFF`` short-circuits to zero work.
    step_idx:
        VM step index whose boundary we're crossing. Determines which
        set of positions we evict.

    Returns
    -------
    int
        Count of rows actually zeroed. Always ``0`` for ``OFF`` /
        ``None`` / no-cache modules.
    """

    if state is None or state.policy is KVEvictionPolicy.OFF:
        return 0
    positions = state.positions_at_step(step_idx)
    if not positions:
        return 0

    # Collect concrete (K_tensor, V_tensor) handles to operate on. We
    # accept both the "explicit K_cache / V_cache attribute" shape (used
    # by some bench/runtime adapters) and the canonical
    # ``attn.kv_cache.cached_k / cached_v`` shape (TransformerKVCache).
    targets = []
    K = getattr(attn, "K_cache", None)
    V = getattr(attn, "V_cache", None)
    if K is not None and V is not None:
        targets.append((K, V))
    kv = getattr(attn, "kv_cache", None)
    if kv is not None:
        cached_k = getattr(kv, "cached_k", None)
        cached_v = getattr(kv, "cached_v", None)
        if cached_k is not None and cached_v is not None:
            targets.append((cached_k, cached_v))

    zeroed = 0
    for K_t, V_t in targets:
        # Cache tensors are [B, H, S_kv, HD]. The eviction state stores
        # positions in absolute cache-row terms; clamp to current size.
        if K_t is None or V_t is None:
            continue
        if K_t.dim() < 3 or V_t.dim() < 3:
            continue
        S_kv = K_t.shape[-2]
        for pos in positions:
            if 0 <= pos < S_kv:
                # In-place zero. ``index_fill_`` would be ideal but the
                # row dim varies per tensor shape; a direct slice is
                # equally fast for the small per-step batches involved.
                K_t[..., pos, :].zero_()
                V_t[..., pos, :].zero_()
                zeroed += 1

    state.evicted_positions.update(positions)
    state.total_evictions += zeroed
    return zeroed


__all__ = [
    "KVEvictionPolicy",
    "KVEvictionState",
    "apply_eviction",
    "build_state_from_report",
]
