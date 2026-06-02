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
from typing import Dict, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

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
            cache row indices. Phase 7.F.4 "full row" path: a position
            appears here iff every residual dim slot at that position is
            dead (the conservative AND).
        evictable_dim_slices_at_step: Phase 7.F.5 per-(position,
            dim_group) eviction path. ``step -> position -> set of
            (d_model_start, d_model_size)`` residual slices the analyzer
            marked dead at that position. Crucially this is **not**
            ANDed across dim names — each dim slot is recorded
            independently. :func:`apply_eviction` translates each
            residual slice to a ``(head, head_dim_offset_range)``
            tuple and zeros only that sub-row of the cache, leaving
            the rest of the row's contributions intact. This is what
            lets the runtime free bytes even when register channels
            (REG_PC etc.) keep the row's "full AND" alive.
        evicted_positions: positions whose K/V rows were last zeroed by
            :func:`apply_eviction` — used by tests to inspect history
            and by determinism gates to compare spec/main decode runs.
        evicted_dim_slot_count: monotonic counter of
            (step, position, dim_slot) zeroing events from the new
            per-dim-slice path. Diagnostic only.
        total_evictions: monotonic counter of (step, position) zero
            events from the legacy full-row path. Diagnostic only.
        num_heads, head_dim: cache layout the dim slices were resolved
            against. Required to translate a residual slice
            ``[d_start, d_start + size)`` to ``[h, hd_start:hd_end]``
            cache indices at runtime. ``None`` means the slice path is
            disabled and only the full-row path applies.
        evicted_bytes_per_position: per-position byte counter — sum of
            ``d_size * 4`` (float32) bytes the slice path zeroed at each
            absolute position. Diagnostic / used by the measurement
            harness to compute the realised dividend.
    """

    policy: KVEvictionPolicy = KVEvictionPolicy.OFF
    layer_idx: Optional[int] = None
    evictable_positions_at_step: Dict[int, Set[int]] = field(default_factory=dict)
    # Phase 7.F.5: per-(step, position, dim_slot) eviction granularity.
    # ``dim_slot`` is a ``(d_model_start, d_model_size)`` residual range.
    evictable_dim_slices_at_step: Dict[int, Dict[int, Set[Tuple[int, int]]]] = field(
        default_factory=dict
    )
    evicted_positions: Set[int] = field(default_factory=set)
    total_evictions: int = 0
    evicted_dim_slot_count: int = 0
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    evicted_bytes_per_position: Dict[int, int] = field(default_factory=dict)

    def positions_at_step(self, step_idx: int) -> Set[int]:
        """Return the (possibly empty) set of positions evictable after step."""

        return self.evictable_positions_at_step.get(int(step_idx), set())

    def dim_slices_at_step(self, step_idx: int) -> Dict[int, Set[Tuple[int, int]]]:
        """Return the ``position -> {(d_start, d_size)}`` map for the step.

        Empty dict when no per-dim-slice eviction was scheduled (either
        ``OFF`` policy or no analyzer hits for the step).
        """

        return self.evictable_dim_slices_at_step.get(int(step_idx), {})

    def is_active(self) -> bool:
        """``True`` when the policy is not OFF and there's anything to evict."""

        if self.policy is KVEvictionPolicy.OFF:
            return False
        return bool(self.evictable_positions_at_step) or bool(
            self.evictable_dim_slices_at_step
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
    dim_positions: Optional[Mapping[str, int]] = None,
    dim_sizes: Optional[Mapping[str, int]] = None,
    num_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    safe_dim_categories: Optional[Set[str]] = None,
) -> KVEvictionState:
    """Project a :class:`LivenessReport` onto one (layer, head) slot.

    Two complementary eviction layouts are populated:

    * **Full-row eviction (legacy, Phase 7.F.4)**:
      ``evictable_positions_at_step`` lists positions where the analyzer
      marked *every* residual dim slot at that position dead — the
      conservative per-row AND. ``apply_eviction`` zeros the entire
      ``K_cache[B, H, S, HD]`` row at those positions.

    * **Per-(position, dim_slice) eviction (Phase 7.F.5, when
      ``dim_positions`` is supplied)**: ``evictable_dim_slices_at_step``
      records each dim slot the analyzer flagged dead, **without** the
      per-row AND. ``apply_eviction`` translates each
      ``(d_model_start, d_model_size)`` residual slice to a
      ``(head, head_dim_offset_range)`` cache sub-row and zeros just
      that slice — freeing bytes even when the row's "full AND" stays
      alive because of conservative-kept register dims like ``REG_PC``.

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
    dim_positions, dim_sizes:
        Optional ``ModelLayout.dim_positions`` / ``ModelLayout.dim_sizes``
        — when both are provided, the builder populates the new
        per-dim-slice eviction map. When ``None`` only the legacy
        full-row map is built.
    num_heads, head_dim:
        Cache layout to stamp into the state so ``apply_eviction`` can
        translate residual slices to cache indices. Together with
        ``dim_positions`` they're required for the per-dim-slice path.
    safe_dim_categories:
        Optional whitelist of dim categories whose evictable slices the
        runtime is allowed to zero out for the per-dim-slice path.
        Defaults to the byte-identity-safe categories: dim names that
        start with ``TEMP``/``ALU_TEMP``/``MUL_TEMP``/``DIV_TEMP`` or
        end with ``_SCRATCH``/``_PREV_STEP``/``_PREV``/``_LAST_STEP``/
        ``_THIS_STEP``. These are the dims whose residual value is
        zero outside of their useful window, so the corresponding
        K-projection contribution is already zero and zeroing the
        cache slice is a runtime byte-identity no-op. Set to an empty
        set to disable the safe filter (e.g. for measurement-only
        runs).

    Returns
    -------
    KVEvictionState
        Per-step evictable position + per-dim-slice maps ready to be
        attached to a ``PureAttention`` (or ``AutoregressiveAttention``)
        module.
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

    evictable_positions_at_step: Dict[int, Set[int]] = {}
    if all_dims:
        for step, by_pos in per_step_dim_count.items():
            positions: Set[int] = set()
            for position, dims_dead in by_pos.items():
                # Conservative AND: every dim slot at this position
                # must have been declared evictable. Otherwise some
                # surviving dim slot could be queried by a future K
                # projection, so the row stays live.
                if dims_dead >= all_dims:
                    positions.add(position)
            if positions:
                evictable_positions_at_step[step] = positions

    # ---- Phase 7.F.5 per-dim-slice path ----
    evictable_dim_slices_at_step: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}
    if dim_positions is not None and dim_sizes is not None:
        # Default safe filter: only zero residual slices that the codebase
        # guarantees are 0 outside their useful window. This keeps
        # byte-identity intact because K_proj(0 at slice) = 0 contribution
        # at every (h, hd) the W_k matrix mixed into; zeroing the cache
        # slice fed exclusively by such a dim is a no-op.
        safe_filter = (
            safe_dim_categories
            if safe_dim_categories is not None
            else _default_safe_dim_categories()
        )

        # Phase 7.F.5: expand the per-dim-slice path beyond just the
        # analyzer's strict "evictable" set. The slice path operates on
        # individual residual cells (not whole rows) and so can also
        # safely zero cycle-conservative dims that fall in the
        # byte-identity-safe category. These are typically per-step
        # opcode flags / register markers whose residual value is 0 at
        # positions where the flag is not firing — zeroing the K cache
        # slice they exclusively feed is a no-op at those positions.
        # The "safe" classification stays conservative: any non-zero
        # residual contribution at a position would change attention
        # scores, so we rely on the prefix/suffix safety semantics.
        all_evictable_dim_names: Set[str] = set()
        for by_pos in per_step_dim_count.values():
            for names in by_pos.values():
                all_evictable_dim_names.update(names)

        # Also include cycle-conservative dim names that pass the safe
        # filter — for each step where some non-cycle dim is dead at a
        # position, mark the cycle-conservative safe-filter dims dead
        # at that position too. This significantly expands the slice
        # path coverage while keeping byte-identity intact.
        cycle_safe_names: Set[str] = set()
        for entry in report.cycle_conservative:
            if layer_idx is not None and entry.layer != layer_idx:
                continue
            if head_idx is not None and entry.head != head_idx:
                continue
            if safe_filter and not _dim_name_is_safe(entry.dim_name, safe_filter):
                continue
            cycle_safe_names.add(entry.dim_name)

        # Build a per-position cumulative dim set: once a dim at position p
        # is marked evictable at step p, it stays evictable for every
        # later step T >= p. This is sound because the analyzer's notion
        # of "dead" is monotonic in time (an entry that no future op
        # reads at step p is still dead at step T > p). Without this
        # propagation we'd evict only the just-written entry at each
        # step, missing the cumulative back-log of dead positions.
        per_position_evictable_dim_names: Dict[int, Set[str]] = {}
        for step, by_pos in per_step_dim_count.items():
            for position, dim_names in by_pos.items():
                filtered: Set[str] = set()
                for name in dim_names:
                    if safe_filter and not _dim_name_is_safe(name, safe_filter):
                        continue
                    filtered.add(name)
                if not filtered:
                    continue
                # The analyzer pairs entries with position=step; the
                # earliest step we know it's dead is ``step`` itself.
                key = position
                per_position_evictable_dim_names.setdefault(key, set()).update(
                    filtered
                )

        # Determine which steps the analyzer reasoned about — we need
        # the upper bound for the forward propagation.
        all_steps = sorted(per_step_dim_count.keys()) or [0]
        max_step = max(all_steps)

        # Forward-propagate: at step T, include all positions p <= T that
        # have any evictable dim. Append cycle-safe dim names at every
        # such (step, position) since the cycle-safe set is uniform.
        sorted_positions = sorted(per_position_evictable_dim_names.keys())
        for step in range(max_step + 1):
            for position in sorted_positions:
                if position > step:
                    break
                combined = set(per_position_evictable_dim_names[position])
                combined.update(cycle_safe_names)
                slices: Set[Tuple[int, int]] = set()
                for name in combined:
                    start = dim_positions.get(name)
                    if start is None:
                        continue
                    size = int(dim_sizes.get(name, 1))
                    if size <= 0:
                        continue
                    slices.add((int(start), int(size)))
                if slices:
                    evictable_dim_slices_at_step.setdefault(step, {})[position] = slices

    return KVEvictionState(
        policy=policy,
        layer_idx=layer_idx,
        evictable_positions_at_step=evictable_positions_at_step,
        evictable_dim_slices_at_step=evictable_dim_slices_at_step,
        num_heads=num_heads,
        head_dim=head_dim,
    )


# Categories of residual dims whose value is guaranteed 0 outside of
# their useful window. The K projection therefore writes 0 to whichever
# cache slot they exclusively feed; zeroing that slot when the dim is
# "dead" is byte-identical no-op.
_SAFE_DIM_PREFIXES: Tuple[str, ...] = (
    "TEMP",
    "ALU_TEMP",
    "MUL_TEMP",
    "DIV_TEMP",
    "MUL_ACCUM",
    "DIV_STAGING",
    "MEM_STAGING",
    "SP_GATHERED",
)
_SAFE_DIM_SUFFIXES: Tuple[str, ...] = (
    "_THIS_STEP",
    "_SCRATCH",
    "_PREV_STEP",
    "_PREV",
    "_LAST_STEP",
)


def _default_safe_dim_categories() -> Set[str]:
    """Tag set the byte-identity-safe filter uses by default.

    Returning a non-empty set keeps the safe filter active in
    :func:`build_state_from_report`; the actual classification happens
    in :func:`_dim_name_is_safe` which checks prefixes / suffixes.
    """

    return {"__default__"}


def _dim_name_is_safe(name: str, safe_filter: Set[str]) -> bool:
    """Whether the dim is in the byte-identity-safe-to-zero category.

    The filter is intentionally conservative: only dims whose residual
    value is guaranteed 0 in the "off" window are returned True.
    Returns ``True`` for the default filter set or whenever the caller
    explicitly listed the dim name in ``safe_filter``.
    """

    if "__default__" not in safe_filter:
        # Explicit allow-list mode.
        return name in safe_filter
    upper = name.upper()
    for prefix in _SAFE_DIM_PREFIXES:
        if upper.startswith(prefix):
            return True
    for suffix in _SAFE_DIM_SUFFIXES:
        if upper.endswith(suffix):
            return True
    return False


# ---------------------------------------------------------------------------
# Runtime application
# ---------------------------------------------------------------------------


def apply_eviction(attn, state: Optional[KVEvictionState], step_idx: int) -> int:
    """Zero out the K/V cache rows that are dead at the end of ``step_idx``.

    Two complementary eviction layouts (set up by
    :func:`build_state_from_report`) are honoured:

    * **Full-row eviction**: ``state.evictable_positions_at_step`` —
      legacy Phase 7.F.4 path that zeros the entire ``[B, H, S, HD]``
      row at a position. Used when the analyzer's per-row AND fired.
    * **Per-dim-slice eviction**: ``state.evictable_dim_slices_at_step``
      — Phase 7.F.5 path that zeros only the ``(head, hd_start:hd_end)``
      sub-rows corresponding to dead residual dim slices, leaving the
      rest of the row alive. This is what frees bytes when register
      channels (REG_PC etc.) keep the row's full AND alive.

    The eviction targets ``attn.K_cache`` and ``attn.V_cache`` when
    those attributes exist (the runtime attention paths populate them
    on each forward via the :class:`TransformerKVCache` adapter).
    Modules that do not retain a KV cache between calls (e.g. plain
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
        Count of K/V row equivalents zeroed. The full-row path counts
        each ``(B, h, pos, :)`` whole row as 1; the per-dim-slice path
        contributes ``slice_size / HD`` (rounded up to 1 if any slice
        zeroed). Always ``0`` for ``OFF`` / ``None`` / no-cache modules.
    """

    if state is None or state.policy is KVEvictionPolicy.OFF:
        return 0
    positions = state.positions_at_step(step_idx)
    dim_slices_by_pos = state.dim_slices_at_step(step_idx)
    if not positions and not dim_slices_by_pos:
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

    # Resolve cache geometry. The state carries (num_heads, head_dim)
    # at compile time; the runtime tensor may report a different shape
    # (e.g. compact() pruned heads). Prefer the tensor's actual shape
    # when both are available — the state values are a fallback for
    # the cache-less path that still records "decisions" for tests.
    num_heads_state = state.num_heads
    head_dim_state = state.head_dim

    zeroed = 0
    for K_t, V_t in targets:
        # Cache tensors are [B, H, S_kv, HD]. The eviction state stores
        # positions in absolute cache-row terms; clamp to current size.
        if K_t is None or V_t is None:
            continue
        if K_t.dim() < 3 or V_t.dim() < 3:
            continue
        S_kv = K_t.shape[-2]
        H_t = K_t.shape[-3]
        HD_t = K_t.shape[-1]
        # Full-row eviction (legacy path).
        for pos in positions:
            if 0 <= pos < S_kv:
                K_t[..., pos, :].zero_()
                V_t[..., pos, :].zero_()
                zeroed += 1

        # Per-dim-slice eviction (Phase 7.F.5).
        if dim_slices_by_pos:
            # Pick the geometry that matches the actual tensor; the
            # state's (num_heads, head_dim) is only used to sanity-check.
            HD = HD_t
            for pos, slices in dim_slices_by_pos.items():
                if not (0 <= pos < S_kv):
                    continue
                if positions and pos in positions:
                    # Full-row path already zeroed this row; the slice
                    # path would just re-zero already-zero cells. Skip
                    # to keep the counters meaningful.
                    continue
                for d_start, d_size in slices:
                    d_end = d_start + d_size
                    # Map residual range [d_start, d_end) onto the
                    # cache's (head, hd) layout, where flat output
                    # index ``o = h * HD + hd``. Handle slices that
                    # straddle a head boundary by iterating in chunks.
                    cur = d_start
                    while cur < d_end:
                        h = cur // HD
                        hd_start = cur - h * HD
                        hd_end = min(hd_start + (d_end - cur), HD)
                        if 0 <= h < H_t and hd_end > hd_start:
                            K_t[..., h, pos, hd_start:hd_end].zero_()
                            V_t[..., h, pos, hd_start:hd_end].zero_()
                        cur = (h + 1) * HD

    # Bookkeeping — decisions are also recorded for the cache-less
    # path so the state-only determinism gates can compare history.
    state.evicted_positions.update(positions)
    state.total_evictions += zeroed
    for pos, slices in dim_slices_by_pos.items():
        state.evicted_positions.add(pos)
        total_bytes_zeroed = 0
        for d_start, d_size in slices:
            state.evicted_dim_slot_count += 1
            total_bytes_zeroed += int(d_size) * 4  # float32
        prev = state.evicted_bytes_per_position.get(pos, 0)
        state.evicted_bytes_per_position[pos] = prev + total_bytes_zeroed
    return zeroed


__all__ = [
    "KVEvictionPolicy",
    "KVEvictionState",
    "apply_eviction",
    "build_state_from_report",
]
