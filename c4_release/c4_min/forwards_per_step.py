"""DEPTH-VIA-FORWARDS-PER-STEP — split an effective digit-extraction depth ``D``
into ``L`` physical layers x ``F`` forwards-per-step (``L * F >= D``).

This is a SELF-CONTAINED, ADDITIVE module implementing the §5 lever of
``docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md`` ("Getting depth from forwards-per-step —
the vanilla resolution").  It supplies the (L, F) split MATH; it does not build,
bake, or import the model, and it does NOT edit the solver — it is designed to be
wired in by the concurrent solver work through the one-line ``INTEGRATION`` hook
below.

The idea, in one paragraph
==========================
The clever-fp64 whole-value construction realizes a required effective depth
``D`` (summed per-op digit-extraction places — ``D ~= 51`` for the full ISA,
``10`` for DIV, ``20`` for MUL) as a chain of ``D`` layer-applications.  §4 of the
note leaves a tension: a *distinct-layer* stack ``D`` deep does NOT fit a stock
24-layer 0.5B, and *weight-tying* it is only honest for a Universal Transformer.
The THIRD place to put the depth is the **autoregressive token loop**: extract one
(or a few) digit(s) per EMITTED TOKEN — per forward pass — instead of per layer,
threading the running remainder through the KV cache / token stream.  A network of
``L`` physical layers, re-invoked ``F = ceil(D / L)`` times per VM step, realizes
the same ``D`` layer-applications:

    L * F  >=  D          (the split constraint)
    L * F  ~=  D          (COMPUTE IS CONSERVED — the same ~D layer-applications
                           per VM step; forwards-per-step re-books depth into
                           SEQUENCE, it does NOT cut FLOPs)

Two corners
===========
  * **distinct-layers (deep)** — ``L = D, F = 1``.  Small seq_len (few tokens/step)
    -> small KV, few launches -> low latency.  But ``L = D ~= 51`` DISTINCT stored
    layers EXCEED a stock 24-layer 0.5B: fits width, NOT depth.
  * **forwards-per-step (shallow)** — ``L`` small (1-4, comfortably inside stock
    24), ``F = ceil(D / L)`` forwards.  Fits a STOCK VANILLA feed-forward 0.5B (the
    win) — but each extra digit-token lengthens the sequence, so KV grows ~ ``F``,
    and there are ``F`` launch-chains + ``F`` KV re-reads per VM step (loses on KV /
    launch / latency).

So the effective depth budget can be satisfied by LAYERS (deep net, small KV) OR
by FORWARDS/tokens (shallow net, KV proportional to ``F``), and each split is costed
so the parent solver can trade a depth-bound *layer* budget for a longer *sequence*
budget and vice-versa.

Effective-depth anchors (from opconfig / the note)
==================================================
  * FULL clever-fp64 ISA:  D ~= 51   (summed per-op digit-depth)
  * DIV / MOD:             D ~= 10
  * MUL:                   D ~= 20
  * ADD / SUB:             D ~= 11
These are the ``qwen_fit_solver.summed_unrolled_depth`` totals / the per-op
``_CLEVER_DEPTH_DIGIT`` entries; this module takes ``D`` as a plain int so it stays
decoupled from that (possibly-concurrently-edited) accounting.

KV model (matches ``qwen_fit_solver.kv_cache_bytes``)
=====================================================
    KV_bytes = 2 (K+V) * n_layers * n_heads * head_dim * seq_len * batch
               * precision_bytes

Forwards-per-step adds ``F`` tokens per VM step, so over a run of ``steps`` VM steps
the sequence grows by ``~ F * steps`` -> KV grows with the forwards factor ``F``.
The DEEP split keeps ``seq_len`` small but ``n_layers = D`` large; the SHALLOW split
keeps ``n_layers`` small but ``seq_len`` (KV) large.  Both are modelled below.

KV COUPLING (the honest first-order result).  Both splits multiply the SAME
``n_layers * seq_len`` KV core: deep = ``D * base_seq``, shallow = ``L * F *
base_seq`` = ``(L*F) * base_seq``.  At an EXACT-DIVISOR split (``L*F == D``, e.g.
51x1 / 17x3 / 3x17 / 1x51) the two are IDENTICAL — so the pure DEEP and SHALLOW
corners TIE on KV bytes.  Forwards-per-step therefore does NOT trade KV for the
vanilla-fit at these corners; the strict differences are: deep wins ``launch_count``
(min-latency: 1 launch-chain / KV re-read vs F of them), shallow wins ``n_layers``
(min stored params -> the vanilla-fit).  KV only RISES above the tie when the split
is non-exact (``L*F > D``, the ceiling rounding inflates the sequence a little).
This is exactly why the solver must cost each split explicitly rather than assume
"deep = small KV, shallow = big KV".

=============================================================================
INTEGRATION  (how the parent solver wires this in — c4_min/qwen_fit_solver.py)
=============================================================================
The concurrent solver owns ``FitConstraints`` (fields: ``precision, max_layers,
max_hidden, max_intermediate, kv_budget_bytes, seq_len, batch, n_heads,
head_dim``), the ``kv_cache_bytes(...)`` formula, ``min_kv_precision``, and
``account_opconfig`` (whose ``summed_unrolled_depth`` yields the effective ``D``).
This module is the ADDITIVE supplier of the layers<->forwards split math.

``FitConstraints`` will gain ONE field::

    forwards_per_step: int = 1     # 1 == distinct-layers (deep); >1 == shallow

and the solver applies this module to the accounted geometry with the EXACT
one-line call::

    geom = apply_forwards_per_step(geom, constraints.forwards_per_step)

where ``geom`` is the ``account_opconfig`` geometry as a dict
(``{"n_layers", "hidden", "n_heads", "head_dim", "seq_len", ...}``).  The returned
dict carries ``n_layers_physical`` (= ``ceil(D / F)``), ``forwards_per_step``,
``effective_seq_len`` (= ``seq_len * F``), and ``kv_bytes`` — which the solver then
checks against ``max_layers`` (now met by the SHALLOW ``n_layers_physical``) and
``kv_budget_bytes`` (now paid on the F-inflated ``effective_seq_len``).  To ENUMERATE
the split choices for a given ``D`` and pick a corner, call
``depth_realizations(D, max_layers, seq_len, batch, precision, n_heads,
head_dim)`` and read ``.pareto`` / ``.deepest`` / ``.shallowest`` / ``.fits_layers``.

Nothing here touches the build path -> the golden fingerprint (``174ece66``) is
UNCHANGED by this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import opconfig as _OC


# ===========================================================================
# Effective-depth anchors (from the note / opconfig accounting).  Provided as
# named constants for the tests + the worked table; the API takes a plain ``D``.
# ===========================================================================
D_FULL_ISA: int = 51        # summed per-op clever-fp64 digit-depth (full ISA)
D_DIV: int = 10             # DIV / MOD long-division digit-extract places
D_MUL: int = 20             # MUL 64-bit product digit-extract places
D_ADDSUB: int = 11          # ADD / SUB decimal digit-extract places

# The stock-vanilla depth budget (a stock Qwen2.5-0.5B has 24 distinct layers).
STOCK_VANILLA_MAX_LAYERS: int = 24

# Default KV-cache sizing context (matches qwen_fit_solver.FitConstraints
# defaults so a bare call lines up with the solver's numbers).
DEFAULT_SEQ_LEN: int = 2048
DEFAULT_BATCH: int = 1
# Stock Qwen2.5-0.5B GQA KV geometry (the honest KV head count is the KEY-VALUE
# heads, not the 14 query heads).
DEFAULT_KV_HEADS: int = 2
DEFAULT_HEAD_DIM: int = 64
DEFAULT_PRECISION: str = "fp64"      # the clever whole-value datapath


# ===========================================================================
# KV bytes — the SAME formula as qwen_fit_solver.kv_cache_bytes, reproduced here
# so this module is self-contained (no import of the concurrently-edited solver).
# ===========================================================================
def kv_bytes(n_layers: int, n_heads: int, head_dim: int, seq_len: int,
             batch: int, precision: str) -> int:
    """KV-cache size in bytes:

        KV = 2 (K+V) * n_layers * n_heads * head_dim * seq_len * batch
             * precision_bytes

    ``n_layers`` is the APPLIED depth (layer-applications the cache holds).  Uses
    ``opconfig.precision_bytes`` — byte-identical to
    ``qwen_fit_solver.kv_cache_bytes`` so the two modules agree."""
    pb = _OC.precision_bytes(precision)
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * pb


# ===========================================================================
# The (L, F) split.
# ===========================================================================
@dataclass(frozen=True)
class DepthSplit:
    """ONE ``(L, F)`` realization of an effective depth ``D``.

    ``n_layers`` (L) physical layers x ``forwards_per_step`` (F) forwards, with
    ``L * F >= D``.  ``layer_applications`` (= L * F) is the CONSERVED compute per
    VM step (~ D in every split).  ``tokens_added_per_step`` == F (one extra token
    per forward).  ``kv_bytes`` is the cache footprint at the F-inflated sequence.
    """
    D: int                          # the effective depth being realized
    n_layers: int                   # L — physical stored layers
    forwards_per_step: int          # F — forwards (emitted tokens) per VM step
    tokens_added_per_step: int      # == F
    layer_applications: int         # == L * F (the conserved per-step compute)
    effective_seq_len: int          # base seq_len * F (the F-inflated sequence)
    kv_bytes: int
    launch_count: int               # ~ F (F kernel-launch chains + F KV re-reads)
    fits_layers: bool               # L <= max_layers (fits a stock feed-forward)

    @property
    def is_distinct_layers(self) -> bool:
        """True for the DEEP corner (F == 1: all depth lives in the layers)."""
        return self.forwards_per_step == 1

    @property
    def slack(self) -> int:
        """Wasted layer-applications above ``D`` (0 when ``D`` divides ``L*F``)."""
        return self.layer_applications - self.D

    def label(self) -> str:
        kind = "distinct-layers/DEEP" if self.is_distinct_layers else "forwards/SHALLOW"
        return (f"L={self.n_layers} x F={self.forwards_per_step} "
                f"(L*F={self.layer_applications}>=D={self.D}) [{kind}]")


def valid_splits(D: int, max_layers: int) -> List[Tuple[int, int]]:
    """Every valid ``(L, F)`` for depth ``D`` with ``1 <= L <= max_layers`` and
    ``F = ceil(D / L)`` (the MINIMAL forwards that cover the depth at that ``L``).

    L ranges from 1 (maximally shallow, F=D) up to ``min(D, max_layers)`` (as deep
    as the layer budget / the depth allows; F=1 at L>=D).  Each ``L`` gets its
    minimal ``F`` so ``L*F`` is the smallest layer-application count >= D at that
    ``L`` (no wasted compute beyond the ceil rounding).  De-duplicated on
    ``(L, F)``."""
    if D < 1:
        raise ValueError(f"D must be >= 1, got {D}")
    if max_layers < 1:
        raise ValueError(f"max_layers must be >= 1, got {max_layers}")
    l_hi = min(D, max_layers)
    seen = set()
    out: List[Tuple[int, int]] = []
    for L in range(1, l_hi + 1):
        F = math.ceil(D / L)
        if (L, F) in seen:
            continue
        seen.add((L, F))
        out.append((L, F))
    return out


def cost_split(D: int, L: int, F: int, *, seq_len: int, batch: int,
               precision: str, n_heads: int, head_dim: int,
               max_layers: int) -> DepthSplit:
    """Cost one ``(L, F)`` split into a fully-populated ``DepthSplit``.

    KV is sized on the APPLIED depth (L physical layers) at the F-INFLATED
    sequence ``seq_len * F`` — forwards-per-step lengthens the sequence, so KV
    grows with F even though the physical layer count shrinks.  ``launch_count``
    ~ F (F kernel-launch chains + F KV re-reads per VM step)."""
    eff_seq = seq_len * F
    kvb = kv_bytes(L, n_heads, head_dim, eff_seq, batch, precision)
    return DepthSplit(
        D=D, n_layers=L, forwards_per_step=F, tokens_added_per_step=F,
        layer_applications=L * F, effective_seq_len=eff_seq, kv_bytes=kvb,
        launch_count=F, fits_layers=(L <= max_layers))


# ===========================================================================
# depth_realizations — enumerate + Pareto the split choices.
# ===========================================================================
@dataclass
class DepthRealizations:
    """The enumerated + costed ``(L, F)`` splits for one effective depth ``D``."""
    D: int
    max_layers: int
    seq_len: int
    batch: int
    precision: str
    n_heads: int
    head_dim: int
    splits: List[DepthSplit]        # all valid splits, ordered DEEP -> SHALLOW
    pareto: List[DepthSplit]        # the Pareto-optimal frontier (see pareto_set)

    @property
    def deepest(self) -> DepthSplit:
        """The DISTINCT-LAYERS corner — max L (min F).  Min-KV / min-latency."""
        return max(self.splits, key=lambda s: s.n_layers)

    @property
    def shallowest(self) -> DepthSplit:
        """The FORWARDS-PER-STEP corner — min L (max F).  Min stored-params /
        best vanilla-fit."""
        return min(self.splits, key=lambda s: s.n_layers)

    @property
    def fits_layers(self) -> List[DepthSplit]:
        """The splits whose L fits ``max_layers`` (a stock feed-forward budget)."""
        return [s for s in self.splits if s.fits_layers]

    def min_kv(self) -> DepthSplit:
        """The min-KV split (the DEEP corner: fewest tokens/step)."""
        return min(self.splits, key=lambda s: (s.kv_bytes, s.forwards_per_step))

    def vanilla_fit(self, max_layers: Optional[int] = None) -> Optional[DepthSplit]:
        """The min-STORED-params / best-vanilla-fit split: the SHALLOWEST split
        whose L fits ``max_layers`` (default the enumeration's ``max_layers``).
        ``None`` if even L=1 does not fit (impossible for max_layers>=1)."""
        cap = self.max_layers if max_layers is None else max_layers
        fitting = [s for s in self.splits if s.n_layers <= cap]
        if not fitting:
            return None
        # shallowest fitting == fewest stored params; ties -> fewer forwards.
        return min(fitting, key=lambda s: (s.n_layers, s.forwards_per_step))

    def table(self) -> str:
        hdr = (f"{'split':40s} {'L':>4s} {'F':>4s} {'L*F':>5s} {'tok/step':>8s} "
               f"{'eff_seq':>8s} {'kv_bytes':>12s} {'launch':>6s} {'fitL':>5s}")
        lines = [f"D={self.D}  max_layers={self.max_layers}  seq_len={self.seq_len} "
                 f"batch={self.batch}  prec={self.precision}  "
                 f"n_heads={self.n_heads} head_dim={self.head_dim}", hdr,
                 "-" * len(hdr)]
        for s in self.splits:
            lines.append(
                f"{s.label():40s} {s.n_layers:4d} {s.forwards_per_step:4d} "
                f"{s.layer_applications:5d} {s.tokens_added_per_step:8d} "
                f"{s.effective_seq_len:8d} {s.kv_bytes:12d} {s.launch_count:6d} "
                f"{'yes' if s.fits_layers else 'no':>5s}")
        return "\n".join(lines)


def pareto_set(splits: List[DepthSplit]) -> List[DepthSplit]:
    """The Pareto-optimal frontier over (kv_bytes, launch_count, n_layers).

    A split is dominated if another split is <= on ALL THREE cost axes and < on at
    least one.  The three axes capture the note's three corners:
      * min-KV        (favours least ceiling-waste L*F; the pure corners TIE at
                       exact-divisor splits, so KV alone does NOT separate deep
                       from shallow)
      * min-latency   (deep — fewest launches / KV re-reads, ~ F)
      * min-STORED    (shallow — fewest physical layers -> fewest stored params,
                       the vanilla-fit corner)
    launches favour DEEP; stored-layers favours SHALLOW; KV ties them at the
    exact-divisor corners — so the deep and shallow corners are BOTH non-dominated
    (deep trades stored-params for launches, shallow vice-versa), and the frontier
    spans them.  Ordered DEEP -> SHALLOW."""
    def dominates(a: DepthSplit, b: DepthSplit) -> bool:
        le = (a.kv_bytes <= b.kv_bytes and a.launch_count <= b.launch_count
              and a.n_layers <= b.n_layers)
        lt = (a.kv_bytes < b.kv_bytes or a.launch_count < b.launch_count
              or a.n_layers < b.n_layers)
        return le and lt
    front = [s for s in splits
             if not any(o is not s and dominates(o, s) for o in splits)]
    front.sort(key=lambda s: -s.n_layers)      # DEEP -> SHALLOW
    return front


def depth_realizations(D: int, max_layers: int = STOCK_VANILLA_MAX_LAYERS,
                       seq_len: int = DEFAULT_SEQ_LEN, batch: int = DEFAULT_BATCH,
                       precision: str = DEFAULT_PRECISION,
                       n_heads: int = DEFAULT_KV_HEADS,
                       head_dim: int = DEFAULT_HEAD_DIM) -> DepthRealizations:
    """Enumerate + cost every valid ``(L, F)`` split of effective depth ``D`` and
    return the ``DepthRealizations`` (all splits + the Pareto frontier).

    ``max_layers`` bounds the DEEPEST split (and flags ``fits_layers`` — a stock
    feed-forward budget is 24).  Every split costs ``(n_layers, forwards_per_step,
    tokens_added_per_step, kv_bytes, launch_count ~ F)``.  Corners:
      * min-KV / min-latency = the DEEP split (``.deepest`` / ``.min_kv()``)
      * min-STORED-params / vanilla-fit = the SHALLOW split (``.shallowest`` /
        ``.vanilla_fit()``)

    NB the enumeration caps L at ``max_layers``; to see the TRUE distinct-layers
    corner (L=D) when D exceeds the budget, pass ``max_layers >= D`` (see
    ``fits_stock_vanilla`` which does exactly this to show D=51 does NOT fit 24)."""
    pairs = valid_splits(D, max_layers)
    splits = [cost_split(D, L, F, seq_len=seq_len, batch=batch, precision=precision,
                         n_heads=n_heads, head_dim=head_dim, max_layers=max_layers)
              for (L, F) in pairs]
    splits.sort(key=lambda s: -s.n_layers)     # DEEP -> SHALLOW
    return DepthRealizations(
        D=D, max_layers=max_layers, seq_len=seq_len, batch=batch,
        precision=precision, n_heads=n_heads, head_dim=head_dim,
        splits=splits, pareto=pareto_set(splits))


# ===========================================================================
# fits_stock_vanilla — the §5 headline: shallow fits stock-24, deep does not.
# ===========================================================================
@dataclass
class StockVanillaVerdict:
    """Whether an effective depth ``D`` fits a stock feed-forward budget, and how."""
    D: int
    max_layers: int
    distinct_layers: DepthSplit     # L=D, F=1 (the deep corner — evaluated even if
                                    #   L>max_layers, to show it does NOT fit)
    shallow_fit: Optional[DepthSplit]   # the shallowest split that DOES fit (or None)
    distinct_fits: bool             # does the L=D distinct-layers stack fit?
    shallow_fits: bool              # does a shallow forwards-per-step split fit?

    def summary(self) -> str:
        d = self.distinct_layers
        s = self.shallow_fit
        lines = [
            f"D={self.D} into a stock feed-forward budget of {self.max_layers} layers:",
            f"  distinct-layers (deep):   L={d.n_layers} F={d.forwards_per_step} -> "
            f"{'FITS' if self.distinct_fits else 'does NOT fit'} "
            f"(needs {d.n_layers} distinct layers > {self.max_layers})"
            if not self.distinct_fits else
            f"  distinct-layers (deep):   L={d.n_layers} F={d.forwards_per_step} -> FITS",
        ]
        if s is not None:
            lines.append(
                f"  forwards-per-step (shallow): L={s.n_layers} F={s.forwards_per_step} "
                f"(+{s.tokens_added_per_step} tokens/step) -> "
                f"{'FITS' if self.shallow_fits else 'does NOT fit'} a stock "
                f"{self.max_layers}-layer vanilla 0.5B")
        return "\n".join(lines)


def fits_stock_vanilla(D: int, max_layers: int = STOCK_VANILLA_MAX_LAYERS,
                       *, seq_len: int = DEFAULT_SEQ_LEN, batch: int = DEFAULT_BATCH,
                       precision: str = DEFAULT_PRECISION,
                       n_heads: int = DEFAULT_KV_HEADS,
                       head_dim: int = DEFAULT_HEAD_DIM) -> StockVanillaVerdict:
    """Show that the SHALLOW forwards-per-step split (L <= max_layers) fits a stock
    feed-forward 0.5B while the DISTINCT-layers split (L = D) does NOT (for D >
    max_layers, e.g. the full ISA's D~=51 vs a 24-layer budget).

    Returns a ``StockVanillaVerdict`` with the deep corner (L=D, F=1 — evaluated
    regardless of the budget so its non-fit is explicit) and the shallowest fitting
    split."""
    # The TRUE distinct-layers corner: L=D, F=1 — costed even if it blows the budget.
    distinct = cost_split(D, D, 1, seq_len=seq_len, batch=batch, precision=precision,
                          n_heads=n_heads, head_dim=head_dim, max_layers=max_layers)
    # The shallowest split that fits the budget (enumerate within the budget).
    real = depth_realizations(D, max_layers=max_layers, seq_len=seq_len, batch=batch,
                              precision=precision, n_heads=n_heads, head_dim=head_dim)
    shallow = real.vanilla_fit(max_layers=max_layers)
    return StockVanillaVerdict(
        D=D, max_layers=max_layers, distinct_layers=distinct, shallow_fit=shallow,
        distinct_fits=(distinct.n_layers <= max_layers),
        shallow_fits=(shallow is not None and shallow.n_layers <= max_layers))


# ===========================================================================
# compute_conserved — the invariant: L * F ~= D layer-applications per VM step.
# ===========================================================================
def compute_conserved(D: int, L: int, F: int, *, tol: int = 0) -> bool:
    """The COMPUTE-CONSERVED invariant: a split does ``L * F`` layer-applications
    per VM step, which must cover ``D`` and not waste more than the ceil-rounding
    slack (``L * F - D < L``, since ``F = ceil(D / L)`` overshoots by < L).

    Returns True iff ``D <= L * F <= D + max(L - 1, tol)`` — i.e. compute equals
    ``D`` up to the unavoidable ceiling rounding (``tol`` widens the allowed slack
    for looser accounting; default 0 uses the exact ceil bound).  Forwards-per-step
    does NOT cut FLOPs: it re-books the SAME ~D layer-applications from
    physical-depth into sequence-length."""
    apps = L * F
    if apps < D:
        return False
    return apps <= D + max(L - 1, tol)


def conserved_report(realizations: DepthRealizations) -> Dict[Tuple[int, int], bool]:
    """``{(L, F): compute_conserved}`` for every split — every entry must be True."""
    return {(s.n_layers, s.forwards_per_step):
            compute_conserved(s.D, s.n_layers, s.forwards_per_step)
            for s in realizations.splits}


# ===========================================================================
# INTEGRATION HOOK — the one-line call the parent solver uses.
# ===========================================================================
def apply_forwards_per_step(geometry: Dict, forwards_per_step: int) -> Dict:
    """Re-book a geometry's depth into forwards-per-step and return the adjusted
    geometry (the ADDITIVE integration hook for ``qwen_fit_solver``).

    ``geometry`` is a dict describing the accounted (deep) geometry; it MUST carry
    ``n_layers`` (the effective depth ``D`` = the distinct-layers stored count) and
    the KV-sizing context.  Recognised keys (missing ones default to this module's
    stock-0.5B defaults, so a partial ``account_opconfig`` dict works):

        n_layers   (REQUIRED)  the effective depth D (distinct-layers count)
        hidden                 pass-through (forwards-per-step does not change width)
        n_heads / head_dim     KV head geometry (default 2 / 64 — GQA KV heads)
        seq_len / batch        KV sizing context (default 2048 / 1)
        precision              KV bytes/elem (default fp64)

    With ``forwards_per_step = F`` the physical layer count becomes
    ``n_layers_physical = ceil(D / F)`` (shallow) and the sequence inflates to
    ``effective_seq_len = seq_len * F``; ``kv_bytes`` is re-sized on those.  The
    returned dict is the input dict COPIED + updated with:

        n_layers_physical    ceil(D / F)     (the SHALLOW stored-layer count)
        forwards_per_step    F
        tokens_added_per_step F
        effective_seq_len    seq_len * F
        layer_applications   n_layers_physical * F   (~ D — compute conserved)
        kv_bytes             KV at (n_layers_physical, effective_seq_len)
        D                    the input effective depth (== input n_layers)

    ``F = 1`` is the identity split (distinct-layers): ``n_layers_physical == D``,
    ``effective_seq_len == seq_len``.

    Solver wiring (one line, per this module's INTEGRATION docstring)::

        geom = apply_forwards_per_step(geom, constraints.forwards_per_step)

    where ``constraints`` is the ``FitConstraints`` box (gaining a
    ``forwards_per_step: int = 1`` field) and ``geom`` is the ``account_opconfig``
    geometry as a dict.  The solver then checks ``n_layers_physical`` against
    ``max_layers`` and ``kv_bytes`` against ``kv_budget_bytes``."""
    if forwards_per_step < 1:
        raise ValueError(f"forwards_per_step must be >= 1, got {forwards_per_step}")
    if "n_layers" not in geometry:
        raise KeyError("geometry must carry 'n_layers' (the effective depth D)")
    D = int(geometry["n_layers"])
    if D < 1:
        raise ValueError(f"geometry['n_layers'] (D) must be >= 1, got {D}")
    F = int(forwards_per_step)
    seq_len = int(geometry.get("seq_len", DEFAULT_SEQ_LEN))
    batch = int(geometry.get("batch", DEFAULT_BATCH))
    precision = geometry.get("precision", DEFAULT_PRECISION)
    n_heads = int(geometry.get("n_heads", DEFAULT_KV_HEADS))
    head_dim = int(geometry.get("head_dim", DEFAULT_HEAD_DIM))

    n_layers_physical = math.ceil(D / F)
    eff_seq = seq_len * F
    kvb = kv_bytes(n_layers_physical, n_heads, head_dim, eff_seq, batch, precision)

    out = dict(geometry)
    out.update(
        D=D,
        n_layers_physical=n_layers_physical,
        forwards_per_step=F,
        tokens_added_per_step=F,
        effective_seq_len=eff_seq,
        layer_applications=n_layers_physical * F,
        kv_bytes=kvb,
    )
    return out
