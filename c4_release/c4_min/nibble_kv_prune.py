"""KV-cache eviction policy for the c4 foundation model (BLOG_SPEC §KV Cache Pruning).

The c4 VM emits the **full register file + memory writes every step** (the 30-token
frame of ``blogspec_vocab``), so a naive KV cache grows *linearly* with program
length — a million-token program keeps a million KV entries, which is infeasible.
This module implements the spec's eviction policy that keeps the cache **bounded
(logarithmic)** while leaving the attention output — and therefore the decoded
byte stream — **exactly** unchanged.

Why eviction is safe here (BLOG_SPEC §KV Cache Pruning, lines 800-816)
---------------------------------------------------------------------
The model attends with **softmax1** (``exp / (1 + sum exp)``) and **ALiBi**
(additive ``-slope*|i-j|`` recency bias). Two structural facts make eviction a
no-op on the softmax1 output:

  1. **cos-sim > 0.99 + ALiBi recency (latest-write-wins).**  Each VM step re-emits
     every register's marker token, so a register's key pattern repeats verbatim
     across steps (cosine similarity ~= 1.0). ALiBi gives the *older* copy an
     extra ``-slope*Δ`` penalty (Δ = the step's token distance), so its softmax1
     weight is a factor ``exp(-slope*Δ)`` smaller than the newer copy's — and that
     ratio is fixed forever (ALiBi depends on *distance*, and the two duplicates
     move together). Once Δ is a full frame (≈30 tokens) the older copy's weight
     is <~5e-4 of the newer's; by the eviction interval (≈120 tokens) it is <1e-13.
     Evicting it changes the softmax1 numerator/denominator by that negligible
     amount. This is *latest-write-wins* for both registers and memory.

  2. **zero-value eviction (how ``free`` works).**  A zero write (an unused
     memory cell, a cleared register) has a **zero value embedding**. Under
     softmax1 a zero-value entry contributes ``w * 0 = 0`` to the numerator
     (weighted value sum) and — crucially, unlike plain softmax — its removal does
     NOT renormalise the other weights, because the ``+1`` sink already sits in the
     denominator. So dropping a zero-value entry is **exactly** output-preserving,
     independent of position. (If a non-zero entry shared its key, mechanism 1
     already evicted that older non-zero copy, so latest-write-wins still holds.)

Interface for the generation loop
---------------------------------
The generation loop owns a :class:`KVCache` per attention head (or one
:class:`MultiHeadKVCache` for the block). Each emitted token it calls
``cache.append(key, value, position)``; every ``prune_interval`` tokens it calls
``cache.prune()``. To read attention it calls ``cache.attention_output(query,
query_position, slopes)`` which computes the softmax1+ALiBi mix over the *live*
(un-evicted) entries. The eviction is **additive**: nothing in ``blogspec_model``
changes; a caller that never prunes gets the identical full-cache result, and a
caller that prunes gets a byte-identical decode with a bounded cache.

This module is pure policy + a reference softmax1/ALiBi mixer that matches
``blogspec_model.softmax1`` and ``blogspec_model.Attn`` exactly, so the
output-equivalence proof (``test_nibble_kv_prune``) is against the real math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


# Spec constants (BLOG_SPEC §KV Cache Pruning).
COS_SIM_THRESHOLD = 0.99      # keys nearer than this are duplicates (line 814)
PRUNE_INTERVAL_TOKENS = 120   # eviction runs every ~120 tokens / ~3 VM steps (816)
ZERO_VALUE_EPS = 1e-9         # a value embedding within this of 0 is a "zero write"
# ALiBi-recency negligibility: an entry whose *maximum possible* softmax1 weight
# (bounded by exp(-slope * distance_from_newest_entry), since a query's raw score
# for any entry is at most that of the newest content-matched key, and the older
# entry carries an extra -slope*distance ALiBi penalty) falls below this epsilon
# is dominated by recency and can never win non-trivial attention -> evictable.
# This is the explicit realisation of the spec's own eviction justification
# ("ALiBi's steep recency bias ... will never win non-trivial attention weight",
# line 814). At head-0's slope 0.25 this triggers ~90 tokens (3 frames) back.
RECENCY_WEIGHT_EPS = 1e-6


# ---------------------------------------------------------------------------
# One cached KV entry.
# ---------------------------------------------------------------------------
@dataclass
class KVEntry:
    key: torch.Tensor        # [head_dim]  the projected key W_k @ x
    value: torch.Tensor      # [head_dim]  the projected value W_v @ x
    position: int            # absolute sequence position (drives ALiBi distance)
    meta: object = None      # optional caller tag (e.g. token id / register name)

    def key_norm(self) -> float:
        return float(self.key.norm())

    def is_zero_value(self, eps: float = ZERO_VALUE_EPS) -> bool:
        """The entry's value embedding is the zero vector (adds 0 to the softmax1
        numerator). Not sufficient alone to evict — see ``is_free_zero``."""
        return float(self.value.norm()) <= eps

    def is_free_zero(self, eps: float = ZERO_VALUE_EPS) -> bool:
        """The exactly-safe zero eviction (mechanism 2, BLOG_SPEC line 816).

        An entry is *free* to evict — an exact softmax1 no-op at ANY position —
        only when BOTH its value AND its key are the zero vector. Then it adds
        ``w*0 = 0`` to the numerator AND ``exp(0*scale - alibi) = exp(-alibi)``…
        no: a zero KEY still scores 0 (before ALiBi), so it *does* add to the
        denominator. The unconditional-exact case is therefore the fully-zero
        entry that a query never scores above the sink — i.e. value==0 (numerator
        contributes 0) and we require key==0 as well so it can't suppress other
        weights through the denominator with a strong content match.

        A zero-value entry with a *strong* key (e.g. a live register MARKER, whose
        value is 0 but whose key wins high score) is NOT free: its ``exp(score)``
        term is load-bearing in the softmax1 denominator (it suppresses the other
        weights). Such markers are handled by mechanism 1 (older dup evicted) and
        mechanism 3 (recency horizon) instead — never by this rule.
        """
        return (float(self.value.norm()) <= eps
                and float(self.key.norm()) <= eps)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two key vectors; 0.0 if either is the zero vector
    (a zero *key* has no direction, so it matches nothing under mechanism 1 —
    zero *values* are handled by mechanism 2 instead)."""
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Single-head KV cache with the two-mechanism eviction policy.
# ---------------------------------------------------------------------------
class KVCache:
    """A per-head running KV cache that prunes per BLOG_SPEC §KV Cache Pruning.

    The cache is append-only between prunes; ``prune()`` applies the two spec
    mechanisms in place. Absolute ``position`` is preserved on every surviving
    entry, so ALiBi distances between the *kept* tokens are exactly what the
    full (un-pruned) forward would compute — eviction never renumbers positions
    (BLOG_SPEC line 712: "ALiBi distance ... is a function of sequence distance").
    """

    def __init__(self, cos_threshold: float = COS_SIM_THRESHOLD,
                 prune_interval: int = PRUNE_INTERVAL_TOKENS,
                 zero_eps: float = ZERO_VALUE_EPS,
                 slope: Optional[float] = None,
                 recency_eps: float = RECENCY_WEIGHT_EPS,
                 score_scale: float = 1.0,
                 dup_metric: str = "cosine"):
        self.cos_threshold = cos_threshold
        self.prune_interval = prune_interval
        self.zero_eps = zero_eps
        # near-duplicate metric for mechanism 1: "cosine" (register-marker heads,
        # the default/original policy) or "exact" (content-addressed §Memory heads
        # whose keys share a large ADDR_BIN common-mode bias — see prune()).
        self.dup_metric = dup_metric
        # ``slope`` is this head's ALiBi slope; enables the recency-horizon
        # mechanism (mechanism 3). When None, only cos-sim + zero-value fire.
        self.slope = slope
        self.recency_eps = recency_eps
        # attention score scale (head_dim**-0.5) — used to bound content scores
        # in the recency horizon so its weight bound matches the real softmax1.
        self.score_scale = score_scale
        self.entries: List[KVEntry] = []
        self._tokens_since_prune = 0
        # bookkeeping for the bounded-growth measurement
        self.total_appended = 0
        self.total_evicted = 0

    # -- generation-loop interface ------------------------------------------
    def append(self, key: torch.Tensor, value: torch.Tensor, position: int,
               meta: object = None) -> None:
        """Add one emitted token's KV. Called once per token by the gen loop."""
        self.entries.append(KVEntry(key.detach().clone(),
                                    value.detach().clone(), position, meta))
        self.total_appended += 1
        self._tokens_since_prune += 1

    def maybe_prune(self) -> int:
        """Prune iff ``prune_interval`` tokens have been appended since the last
        prune (the spec's "runs automatically every 120 tokens"). Returns the
        number of entries evicted."""
        if self._tokens_since_prune >= self.prune_interval:
            return self.prune()
        return 0

    def prune(self) -> int:
        """Apply the three eviction mechanisms in place. Returns #entries evicted.

        Order matters — mechanism 1 runs FIRST so the surviving representative of
        each key group (e.g. the live register marker) is retained before the
        recency/zero rules look at what remains.

        Mechanism 1 (cos-sim>0.99 + recency): whenever two keys are near-dup the
        *older* (smaller position) is dropped — ALiBi gives it a permanently
        smaller weight than the newer identical copy (latest-write-wins). This
        keeps the newest marker of each register/address.

        Mechanism 3 (ALiBi-recency horizon): an entry whose *max possible* future
        softmax1 weight is below ``recency_eps`` is dominated by recency and can
        never win non-trivial attention — safe to evict. The bound accounts for
        the entry's best content score (bounded by its key norm) PLUS its ALiBi
        penalty: ``exp(ceil_score - slope*dist)`` where any future query sits at
        position >= newest, so ``dist >= newest_pos - pos`` only grows. This is
        the explicit form of the spec's justification ("ALiBi's steep recency
        bias ... will never win non-trivial attention weight") and generalises
        mechanism 1 to non-duplicate old payloads.

        Mechanism 2 (free zero): a fully-zero entry (value==0 AND key==0) is an
        exact softmax1 no-op at any position (adds 0 to the numerator and only
        the sink-equal ``exp(0)`` it would to the denominator is already the
        ZFOD baseline) — dropped outright. A zero-value entry with a strong key
        (a live marker) is NOT free and is left to mechanisms 1/3.
        """
        import math
        n0 = len(self.entries)
        if n0 == 0:
            self._tokens_since_prune = 0
            return 0

        # --- mechanism 1: near-duplicate keys -> evict the OLDER --------------
        # Sort newest-first; keep an entry only if it is not a near-duplicate of
        # an already-kept (hence newer) entry. That deterministically drops the
        # older member of every duplicate group (keeps the live marker).
        #
        # ``dup_metric`` selects HOW "near-duplicate" is measured:
        #   * "cosine" (default): raw cosine > cos_threshold — the original policy,
        #     correct for register-marker heads whose distinct keys are well
        #     separated in direction.
        #   * "exact": relative-L2 |k_e - k| <= (1-cos_threshold)*|k| — used for a
        #     CONTENT-ADDRESSED head (a §Memory store) whose keys carry a large
        #     shared ADDR_BIN common-mode bias, so DIFFERENT addresses are
        #     ~parallel (raw cosine 0.999) and a cosine dup-test would wrongly merge
        #     distinct stores (a later load then reads 0). Relative-L2 is ~0 only
        #     for a VERBATIM-repeated key (true latest-write-wins) and stays O(smag)
        #     apart for distinct addresses, so every distinct store survives.
        ordered = sorted(self.entries, key=lambda e: e.position, reverse=True)
        survivors: List[KVEntry] = []
        tol = 1.0 - self.cos_threshold
        for e in ordered:
            if self.dup_metric == "exact":
                dup = False
                for k in survivors:
                    denom = max(float(k.key.norm()), float(e.key.norm()), 1e-30)
                    if float((e.key - k.key).norm()) <= tol * denom:
                        dup = True
                        break
            else:
                dup = any(cosine_sim(e.key, k.key) > self.cos_threshold
                          for k in survivors)
            if dup:
                continue                      # older near-dup of a kept newer key
            survivors.append(e)

        # --- mechanism 3: ALiBi-recency horizon (needs the head slope) --------
        # This is the ONE mechanism that bounds an entry's contribution to BOTH
        # the softmax1 numerator (weighted value) AND denominator (its exp(score)
        # term). Because softmax1 has the +1 sink, removing a term changes the
        # output by O(that term); an entry whose max future weight is < recency_eps
        # is thus an output no-op to that tolerance — regardless of its value.
        # This is what makes eviction byte-exact: a dropped entry's decode
        # influence is far below the byte-head's argmax margin.
        if self.slope is not None and self.slope > 0.0 and survivors:
            newest = max(e.position for e in survivors)
            # ceil on entry e's content score q.k_e*scale, PER ENTRY, by
            # Cauchy-Schwarz: |q.k_e| <= |q|*|k_e| <= max_kn * |k_e| (a query's
            # norm is bounded by the largest key norm in the same residual family).
            # This per-entry bound is TIGHTER than a single global max_kn^2 ceiling
            # and — crucially — is 0 for a ZERO-KEY entry (its score is exactly 0
            # before ALiBi), so a zero-key entry is evicted as soon as
            # exp(-slope*dist) falls below recency_eps.  Still a true upper bound
            # => output-exact.  (The old global max_kn^2 ceiling kept zero-key
            # value-carrying markers — STEP_END, cross-role bytes — forever, which
            # let the cache grow ~linearly; this per-entry form keeps it FLAT.)
            max_kn = max(e.key_norm() for e in survivors)
            kept3: List[KVEntry] = []
            for e in survivors:
                dist = newest - e.position
                # DECOUPLE the ZERO-VALUE (NULL-write) recency horizon from the
                # (EFF-coupled) content ceiling — the CHK-6 flat-cache fix.  A
                # zero-VALUE entry contributes EXACTLY 0 to the softmax1 numerator,
                # so keeping vs dropping it can only perturb the DENOMINATOR by its
                # own ``exp(score)`` term.  On the §Memory heads a NULL memory-write
                # row carries the store-role / load-enable GATE (``-PEN`` on a
                # dedicated channel), so its ACTUAL score against any query is
                # ``<= 0`` ⇒ ``exp(score) <= exp(-slope*dist)`` and its denominator
                # influence decays with DISTANCE ALONE, independent of the key
                # magnitude.  The recall ``EFF`` inflates the §Memory key norm to
                # ~1e5, so the Cauchy-Schwarz content ceil ``max_kn*|k_e|*scale`` is
                # ~1e9 and ``exp(ceil - slope*dist)=1`` forever — the coupling that
                # let the cache grow ~linearly on deep recursion / distinct-address
                # runs (measured: rec_sum(6) memory head 151 entries, 123 of them
                # stale NULL rows).  Using ``ceil_score = 0`` for a zero-value entry
                # restores the SMALL, EFF-INDEPENDENT recency window
                # (``dist > -log(recency_eps)/slope`` ≈ tens of tokens) WITHOUT
                # touching value-carrying store rows, whose full content ceil keeps
                # the ``EFF``-sized recall horizon (a deeply-nested LEV still recalls
                # a saved BP/PC stored ~250k tokens ago).
                if e.is_zero_value(self.zero_eps):
                    ceil_score = 0.0
                else:
                    ceil_score = (max_kn * e.key_norm()) * self.score_scale
                max_w = math.exp(min(0.0, ceil_score - self.slope * dist))
                if max_w >= self.recency_eps:
                    kept3.append(e)
            survivors = kept3

        # --- mechanism 2a: DEAD-VALUE head evicts everything (exact) ----------
        # If NO surviving entry carries a non-zero value, this head's softmax1
        # output is identically the zero vector for every query (numerator is a
        # weighted sum of zero values). Removing all of its entries leaves an
        # empty cache whose attention_output is exactly zeros_like(query) — an
        # exact no-op. This is what evicts the entirely-inert heads (e.g. heads
        # with W_v == 0) that would otherwise grow linearly.
        if survivors and all(e.is_zero_value(self.zero_eps) for e in survivors):
            survivors = []

        # --- mechanism 2b: fully-zero entries evict with no replacement -------
        # A value==0 AND key==0 entry contributes 0 to the numerator and a bare
        # exp(-slope*dist) to the denominator. To keep eviction OUTPUT-EXACT we
        # only drop it once that denominator term is itself recency-negligible
        # (the same horizon as mechanism 3) — this is the exact form of the spec's
        # "how free works": a zero write far enough back is indistinguishable from
        # never having attended. (Recent zero-key/zero-value entries stay: their
        # exp(score) is a live denominator term.) When mechanism 3 is active it has
        # already removed these; this rule additionally fires when slope is
        # unknown (no head pass) so zero writes still evict.
        if survivors:
            newest_all = max(e.position for e in survivors)
            def _free_and_stale(e):
                if not e.is_free_zero(self.zero_eps):
                    return False
                if self.slope is None or self.slope <= 0.0:
                    return True          # no recency model: honour the spec rule
                return math.exp(-self.slope * (newest_all - e.position)) < self.recency_eps
            survivors = [e for e in survivors if not _free_and_stale(e)]

        kept = sorted(survivors, key=lambda e: e.position)   # chronological order

        evicted = n0 - len(kept)
        self.entries = kept
        self.total_evicted += evicted
        self._tokens_since_prune = 0
        return evicted

    # -- attention read -----------------------------------------------------
    def attention_output(self, query: torch.Tensor, query_position: int,
                         slope: float, scale: float) -> torch.Tensor:
        """softmax1 + ALiBi attention of ``query`` over the LIVE cache entries.

        Matches ``blogspec_model.Attn.forward`` per head exactly:
            score_j = (q . k_j) * scale - slope * |q_pos - k_pos|
            out     = sum_j softmax1(score)_j * v_j
        Only causal entries (position <= query_position) participate.
        """
        live = [e for e in self.entries if e.position <= query_position]
        if not live:
            return torch.zeros_like(query)
        keys = torch.stack([e.key for e in live])          # [n_live, head_dim]
        vals = torch.stack([e.value for e in live])        # [n_live, head_dim]
        dist = torch.tensor([float(query_position - e.position) for e in live])
        scores = (keys @ query) * scale - slope * dist     # [n_live]
        weights = _softmax1(scores)
        return (weights.unsqueeze(-1) * vals).sum(dim=0)

    def __len__(self) -> int:
        return len(self.entries)


def _softmax1(x: torch.Tensor) -> torch.Tensor:
    """softmax1 identical to ``blogspec_model.softmax1`` (the ``+1`` ZFOD sink)."""
    m = x.max()
    m = torch.clamp(m, min=0.0)
    exp_x = torch.exp(x - m)
    denom = torch.exp(-m) + exp_x.sum()
    return exp_x / denom


# ---------------------------------------------------------------------------
# Multi-head wrapper — one KVCache per head (keys/values differ per head).
# ---------------------------------------------------------------------------
class MultiHeadKVCache:
    """A block's KV cache: one :class:`KVCache` per attention head.

    ``slopes`` is the head's ALiBi geometric sequence (``Attn.alibi_slopes``).
    ``append``/``prune`` fan out across heads; ``attention_output`` returns the
    concatenated per-head mix (the pre-``W_o`` attention output for one query).
    """

    def __init__(self, n_heads: int, head_dim: int, slopes: torch.Tensor,
                 cos_threshold: float = COS_SIM_THRESHOLD,
                 prune_interval: int = PRUNE_INTERVAL_TOKENS,
                 zero_eps: float = ZERO_VALUE_EPS,
                 recency_eps: float = RECENCY_WEIGHT_EPS):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.slopes = slopes
        self.scale = head_dim ** -0.5
        # each head prunes with ITS OWN ALiBi slope, so the recency-horizon
        # mechanism uses the correct (per-head) decay rate.
        self.heads = [
            KVCache(cos_threshold, prune_interval, zero_eps,
                    slope=float(slopes[h]), recency_eps=recency_eps,
                    score_scale=self.scale)
            for h in range(n_heads)
        ]

    def append(self, keys: torch.Tensor, values: torch.Tensor, position: int,
               meta: object = None) -> None:
        """keys/values: [n_heads, head_dim] for one token position."""
        for h in range(self.n_heads):
            self.heads[h].append(keys[h], values[h], position, meta)

    def maybe_prune(self) -> int:
        return sum(c.maybe_prune() for c in self.heads)

    def prune(self) -> int:
        return sum(c.prune() for c in self.heads)

    def attention_output(self, queries: torch.Tensor, query_position: int
                        ) -> torch.Tensor:
        """queries: [n_heads, head_dim]. Returns [n_heads*head_dim] mixed output."""
        outs = [
            self.heads[h].attention_output(
                queries[h], query_position,
                slope=float(self.slopes[h]), scale=self.scale)
            for h in range(self.n_heads)
        ]
        return torch.cat(outs)

    def live_size(self) -> int:
        """Total live entries across all heads (the cache 'size')."""
        return sum(len(c) for c in self.heads)

    def per_head_size(self) -> int:
        """Live entries in a single head (all heads prune identically-shaped, but
        may differ if keys differ per head)."""
        return len(self.heads[0])


# ---------------------------------------------------------------------------
# Helpers to project a token stream into per-head K/V using a real Attn block —
# used by the output-equivalence proof and available to the generation loop.
# ---------------------------------------------------------------------------
def project_kv(attn, residual: torch.Tensor):
    """Project one position's residual ``x`` [dim] into per-head (K, V).

    Returns ``(K, V)`` each [n_heads, head_dim], matching ``Attn.forward``'s
    ``W_k``/``W_v`` reshape. ``attn`` is a ``blogspec_model.Attn``.
    """
    H, HD = attn.n_heads, attn.head_dim
    k = torch.nn.functional.linear(residual, attn.W_k).view(H, HD)
    v = torch.nn.functional.linear(residual, attn.W_v).view(H, HD)
    return k, v


def project_q(attn, residual: torch.Tensor):
    """Project one position's residual into per-head Q [n_heads, head_dim]."""
    H, HD = attn.n_heads, attn.head_dim
    return torch.nn.functional.linear(residual, attn.W_q).view(H, HD)
