"""CONSTANT-DIVISOR digit-recurrence DIVMOD — full ``(q, r) = divmod(a, b)`` for a
COMPILE-TIME-CONSTANT divisor ``b``, folded as persistent fp32 SwiGLU FFN blocks.

This is the *dead-simple, full-divmod* member of the constant-divisor gadget
family.  It produces BOTH the quotient and the remainder in ~1-2 blocks per
dividend nibble, byte-exact by construction, with NO per-step multiply, NO wide
subtract, and NO byte-wise lexicographic compare — because ``b`` is a python int
at build time, the whole general-divisor machinery of ``nibble_alu32`` (KB
precompute, GT/EQ lanes, the ``QD*b`` multiply, the borrow subtract) collapses
into two trivially-linear operations:

Algorithm (MSB-first long division, exploiting a constant ``b``)
================================================================
Precompute the baked thresholds ``k·b`` for ``k = 1..15`` (pure constants).
Maintain a running remainder ``R`` with the loop invariant ``0 <= R < b``.  For
``j = in_nibbles-1 down to 0`` (each dividend nibble, MSB first)::

    R'  = 16·R + nibble_j              # bring down the next nibble; R<b  =>  R' < 16·b
    q_j = Σ_{k=1..15} [R' >= k·b]      # "test all 16 in parallel": 15 baked thresholds
    r   = R' − b·q_j                    # LINEAR readout of the SAME indicators
    R   = r                             # already reduced, 0 <= R < b; feeds next step
    emit q_j as quotient nibble j

Final: quotient = the emitted ``q_j`` nibbles, remainder = ``R`` (as nibbles).

The whole point: **b constant ⟹ k·b are baked thresholds ⟹ q is a parallel
count (one ``_step_ge`` staircase) and r is a linear readout of those same
indicators** (``R' − b·q``).  No multiply, no borrow ripple, no comparison state.

Representation & fp32 discipline
================================
The running remainder ``R`` and the brought-down ``R'`` are carried as a SINGLE
scalar (not nibbles): both are ``< 16·b``, so for an fp32-safe ``b`` the scalar
stays well under ``2^24`` and every value is fp32-exact.  The staircase relu
argument is ``RELU_S·(R' − (k·b − ½))``; its largest magnitude is ``RELU_S·16·b``
so the fp32-exactness ceiling is ``RELU_S·16·b < 2^24`` ⟹ ``b < 2^24 /
(16·RELU_S) ≈ 52428`` at ``RELU_S = 200`` — see ``fp32_ceiling`` /
``max_relu_arg`` and the empirical sweep in the ``__main__`` battery.

Narrowing (provably-zero nibbles are never emitted)
===================================================
``build_const_divmod_digitrec(b, in_width)`` emits only the necessary nibbles:

  * dividend nibbles processed = ``ceil(in_width/4)`` (skip statically-zero leading
    nibbles when the dividend is bounded by ``in_width`` bits),
  * quotient nibbles = ``ceil(log16(2^in_width / b))`` (higher quotient nibbles are
    provably 0 for a bounded dividend), and
  * remainder nibbles = ``ceil(log16 b)`` (``r < b`` so higher nibbles are 0).

``b == 0 -> (0, 0)`` per ISA_SPEC 4.2 (``isa.interpret``: ``a // b if b else 0`` /
``a % b if b else 0``).  Powers of two need no special case (the general staircase
is exact for them too).

Public API
==========
``build_const_divmod_digitrec(b, in_width=32) -> DigitRecDivmod`` returns a small
dataclass carrying the ordered block list (SwiGLU tensor dicts), the operand /
result band indices, and the narrowing/fp32 metadata.  Feed a residual seeded
with the dividend ``a`` in the ``A`` nibble band through the blocks (see
``apply_blocks`` / the ``__main__`` verification harness) and read the quotient /
remainder nibbles back out.

Does not import or touch any ``nibble_vm`` / ``qwen_full_vm`` / sibling
``const_divmod*`` build path; it reuses ONLY the pure SwiGLU unit emitters
(``_step_ge`` / ``_ident`` / ``_clear`` / ``_empty_spec`` / ``_truncate`` /
``_guard`` / ``RELU_S`` / ``S``) from ``nibble_alu32``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

# Reuse ONLY the pure SwiGLU unit emitters + scale constants (no build path).
from . import nibble_alu32 as _A32
from .nibble_alu32 import (
    _step_ge, _ident, _clear, _empty_spec, _truncate, _guard, RELU_S, S,
)


def _set_one(one_dim: int) -> None:
    """Point ``nibble_alu32``'s module-level ``_ONE`` lane (which every emitter
    writes ``W_up[u, _ONE] = S`` into) at our layout's constant-1 dim.  MUST be
    called before emitting any block — the emitters read this global, and a stale
    ``None`` silently broadcasts ``W_up`` across the whole row (newaxis indexing)."""
    _A32._ONE = one_dim


# ===========================================================================
# Minimal standalone layout: ONE constant + operand A nibbles + result nibbles
# + a single running-remainder scalar.  Self-contained (no model machinery), so
# the module is CPU-fast and never touches a full-VM build.
# ===========================================================================
class _DigitRecLayout:
    """Tiny named residual allocator for the constant-divmod digit recurrence.

    The running remainder is carried as **clean nibbles** (not a scalar) so the
    ``16·R`` bring-down is an EXACT positional nibble shift — never the silu-identity
    ``×16`` that amplifies the ~1e-5 fp32 recompose residue and compounds it across
    the 8 steps (the scalar-carry design drifted ~0.9 by step 7).  Each step
    recomposes ``R'`` to a scalar FRESH from those clean nibbles (residue ~0.1,
    absorbed by the ``_step_ge`` half-integer ramp) and re-snaps the reduced
    remainder back to integer nibbles, so nothing compounds — fp32-exact.

    Bands:
      ``ONE``     — the constant-1 lane (SwiGLU-identity source for `_ident`).
      ``A``       — the dividend nibbles (LSB first), width ``in_nibbles``.
      ``R_NIB``   — the running-remainder NIBBLES (LSB first), width ``rp_nibbles``
                    (r_nibbles + 1 headroom for the bring-down shift).
      ``RP``      — the brought-down ``R' = Σ 16^j·R_NIB'[j]`` SCALAR (fresh each step).
      ``RVAL``    — the reduced ``R = R' − b·q`` SCALAR (before re-nibbling).
      ``QUOT``    — the emitted quotient nibbles (LSB first), width ``q_nibbles``.
      ``REM``     — the final remainder nibbles (LSB first), width ``r_nibbles``.
    """

    def __init__(self, in_nibbles: int, q_nibbles: int, r_nibbles: int,
                 rp_nibbles: int):
        self._off = 0
        self._names: Dict[str, Tuple[int, int]] = {}
        self.ONE = self._scalar("ONE")
        self.A = self._band("A", in_nibbles)
        self.R_NIB = self._band("R_NIB", rp_nibbles)   # remainder nibbles (LSB first)
        self.RP = self._scalar("RP")                    # R' scalar (fresh each step)
        self.QD_RAW = self._scalar("QD_RAW")            # raw quotient-digit staircase sum
        self.QD = self._scalar("QD")                    # SNAPPED integer quotient digit 0..15
        self.RVAL = self._scalar("RVAL")                # reduced R scalar
        self.RREM = self._scalar("RREM")                # running residual for the peel
        self.QUOT = self._band("QUOT", q_nibbles)
        self.REM = self._band("REM", r_nibbles)
        self.D = self._off

    def _scalar(self, name: str) -> int:
        return self._band(name, 1)

    def _band(self, name: str, size: int) -> int:
        base = self._off
        self._names[name] = (base, size)
        self._off += size
        return base


# ===========================================================================
# Narrowing arithmetic (all pure python on the compile-time constant b).
# ===========================================================================
def _quot_nibbles(b: int, in_width: int) -> int:
    """Number of NON-ZERO quotient nibbles for a dividend < 2^in_width.

    The quotient is < 2^in_width / b, so it needs ``ceil(log16(ceil(2^w / b)))``
    nibbles (at least 1).  b==1 gives the full ``ceil(in_width/4)``."""
    if b <= 1:
        return max(1, math.ceil(in_width / 4))
    max_q = (1 << in_width) // b            # floor: the largest achievable quotient
    if max_q <= 0:
        return 1
    return max(1, (max_q.bit_length() + 3) // 4)   # ceil(log16(max_q+1)) == ceil(bits/4)


def _rem_nibbles(b: int) -> int:
    """Number of NON-ZERO remainder nibbles: r < b so ``ceil(log16 b)`` (>=1)."""
    if b <= 1:
        return 1
    return max(1, ((b - 1).bit_length() + 3) // 4)   # ceil(log16(b))


def _in_nibbles(in_width: int) -> int:
    return max(1, math.ceil(in_width / 4))


# ===========================================================================
# The block builders.  Each block is a standalone SwiGLU FFN spec applied to the
# residual (read block input, write deltas), exactly like nibble_alu32's blocks.
# ===========================================================================
def _bringdown_block(L: _DigitRecLayout, dim: int, nib_idx: int,
                     rp_nibbles: int) -> Dict[str, torch.Tensor]:
    """Bring down the next dividend nibble AND recompose the scalar ``R'`` — the
    fp32-exact bring-down.  The running remainder lives as CLEAN nibbles ``R_NIB``;
    ``R' = 16·R + A[nib_idx]`` is realised as a positional nibble SHIFT (``R_NIB[j]``
    moves to weight ``16^(j+1)``) plus the new nibble at weight ``16^0``::

        R' = A[nib_idx] + Σ_j 16^(j+1) · R_NIB[j]     (SET RP)

    Every ``R_NIB[j]`` and ``A[nib_idx]`` is a clean 0..15 integer, so each
    silu-identity read (``silu(S·nib)/S`` exact for nibbles) is exact-to-fp32 and the
    only residue is the recompose of THIS step's clean nibbles (~0.1, absorbed by the
    ramp) — it does NOT compound because the remainder is re-snapped to integer
    nibbles at the end of every step.  ``R'`` is a scalar ``< 16·b`` for the
    staircase compare."""
    spec = _empty_spec(dim, 2 + rp_nibbles)
    u = 0
    u = _clear(spec, u, L.RP)                                         # SET: -old RP
    u = _ident(spec, u, {L.A + nib_idx: 1.0}, 0.0, L.RP, 1.0)       # + new nibble (16^0)
    for j in range(rp_nibbles):                                       # R shifted up 1 nibble
        u = _ident(spec, u, {L.R_NIB + j: 16.0 ** (j + 1)}, 0.0, L.RP, 1.0)
    return _truncate(spec, u, dim)


def _qraw_block(L: _DigitRecLayout, dim: int, b: int) -> Dict[str, torch.Tensor]:
    """Quotient digit as the raw parallel count ``QD_RAW = Σ_{k=1..15}[R' >= k·b]``
    (baked thresholds — the "test all 16 in parallel" staircase).  ``QD_RAW`` lands
    within ~0.03 of the true integer digit (the ``_step_ge`` ramp residue, one per
    active threshold); the NEXT block snaps it to the exact integer.  (SET QD_RAW.)"""
    spec = _empty_spec(dim, 1 + 15 * 2)
    u = 0
    rp = {L.RP: 1.0}
    u = _clear(spec, u, L.QD_RAW)
    for k in range(1, 16):
        u = _step_ge(spec, u, rp, 0.0, k * b, L.QD_RAW, 1.0)
    return _truncate(spec, u, dim)


def _qsnap_block(L: _DigitRecLayout, dim: int, q_out_idx: int,
                 store_q: bool) -> Dict[str, torch.Tensor]:
    """Snap ``QD_RAW`` (≈ integer ± 0.03) to the EXACT integer digit ``QD`` via a
    second half-integer staircase ``QD = Σ_{m=1..15}[QD_RAW >= m]`` (the ramp centred
    at ``m−0.5`` rounds the noisy sum cleanly to its nearest integer 0..15).  Also
    stores it into ``QUOT[q_out_idx]`` when kept.  This exact-integer QD is what makes
    the subsequent reduce ``R = R' − b·QD`` exact — the raw staircase sum, scaled by
    ``b`` in the reduce, would amplify the 0.03 residue to ``0.03·b`` (≈ 30 for
    b=1000) and corrupt the remainder (the bug the snap fixes).  ``store_q=False``
    for a provably-zero high quotient nibble (still snaps QD for the reduce)."""
    n_units = (1 + 15 * 2) + (1 + 15 * 2 if store_q else 0)
    spec = _empty_spec(dim, n_units)
    u = 0
    qr = {L.QD_RAW: 1.0}
    u = _clear(spec, u, L.QD)
    for m in range(1, 16):
        u = _step_ge(spec, u, qr, 0.0, m, L.QD, 1.0)
    if store_q:
        u = _clear(spec, u, L.QUOT + q_out_idx)
        for m in range(1, 16):
            u = _step_ge(spec, u, qr, 0.0, m, L.QUOT + q_out_idx, 1.0)
    return _truncate(spec, u, dim)


def _reduce_block(L: _DigitRecLayout, dim: int, b: int) -> Dict[str, torch.Tensor]:
    """Reduce ``RVAL = R' − b·QD`` (SET) with the EXACT integer digit ``QD`` (0..15).

    ``b·QD`` is one silu-identity read of the clean integer ``QD`` scaled by ``−b`` in
    the down-projection — exact (``QD`` ≤ 15 so ``silu(S·QD)/S`` is exact, and
    ``−b·QD < 16b`` is fp32-exact for every fp32-safe b).  RVAL = ``a``'s running
    remainder, guaranteed ``0 <= RVAL < b``."""
    spec = _empty_spec(dim, 3)
    u = 0
    u = _clear(spec, u, L.RVAL)
    u = _ident(spec, u, {L.RP: 1.0}, 0.0, L.RVAL, 1.0)              # + R'
    u = _ident(spec, u, {L.QD: 1.0}, 0.0, L.RVAL, -float(b))        # - b·QD
    return _truncate(spec, u, dim)


def _renib_raw_block(L: _DigitRecLayout, dim: int, c: int) -> Dict[str, torch.Tensor]:
    """Raw peel of nibble ``c`` from RREM into the QD_RAW scratch: ``QD_RAW =
    Σ_{k=1..15}[RREM >= k·16^c]`` (0..15 ± residue).  (SET QD_RAW.)"""
    m = 16 ** c
    spec = _empty_spec(dim, 1 + 15 * 2)
    u = 0
    u = _clear(spec, u, L.QD_RAW)
    for k in range(1, 16):
        u = _step_ge(spec, u, {L.RREM: 1.0}, 0.0, k * m, L.QD_RAW, 1.0)
    return _truncate(spec, u, dim)


def _renib_snap_block(L: _DigitRecLayout, dim: int, c: int) -> Dict[str, torch.Tensor]:
    """Snap the raw peel ``QD_RAW`` to the exact integer digit ``R_NIB[c]`` and peel
    it off the running residual::

        R_NIB[c] = Σ_{d=1..15}[QD_RAW >= d]     (exact integer, SET)
        RREM    -= 16^c · R_NIB[c]                (exact peel-off via the same snap)

    Both read the block INPUT (QD_RAW + RREM); the ``-16^c·digit`` subtraction uses
    the SNAPPED staircase so it removes an EXACT integer multiple of ``16^c``, leaving
    RREM clean for the next (lower) nibble."""
    m = 16 ** c
    spec = _empty_spec(dim, 1 + 15 * 2 + 15 * 2)
    u = 0
    qr = {L.QD_RAW: 1.0}
    u = _clear(spec, u, L.R_NIB + c)
    for d in range(1, 16):
        u = _step_ge(spec, u, qr, 0.0, d, L.R_NIB + c, 1.0)
    for d in range(1, 16):
        u = _step_ge(spec, u, qr, 0.0, d, L.RREM, -float(m))
    return _truncate(spec, u, dim)


def _renibble_seed_block(L: _DigitRecLayout, dim: int) -> Dict[str, torch.Tensor]:
    """Seed the peel residual ``RREM = RVAL`` (SET) before the MSB-first nibble peel
    that re-snaps the reduced remainder into clean integer nibbles."""
    spec = _empty_spec(dim, 2)
    u = 0
    u = _clear(spec, u, L.RREM)
    u = _ident(spec, u, {L.RVAL: 1.0}, 0.0, L.RREM, 1.0)
    return _truncate(spec, u, dim)


def _renibble_blocks(L: _DigitRecLayout, dim: int, rp_nibbles: int,
                     tag: str) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The full re-nibble sequence: seed RREM=RVAL, then peel rp_nibbles MSB-first,
    each as a raw-peel + snap-and-subtract PAIR so every ``R_NIB[c]`` is an EXACT
    integer and the running residual stays clean (no amplified residue, batch-safe).
    Cost = ``rp_nibbles·2 + 1`` blocks, each ~30-45 units, INDEPENDENT of ``b``."""
    out = [(f"cdr-renib-seed-{tag}", _renibble_seed_block(L, dim))]
    for c in range(rp_nibbles - 1, -1, -1):
        out.append((f"cdr-renib-raw-{tag}-{c}", _renib_raw_block(L, dim, c)))
        out.append((f"cdr-renib-snap-{tag}-{c}", _renib_snap_block(L, dim, c)))
    return out


def _rem_final_block(L: _DigitRecLayout, dim: int, b: int) -> Dict[str, torch.Tensor]:
    """Copy the final reduced remainder nibbles ``R_NIB`` into the ``REM`` result
    band (SET).  After the last iteration R_NIB holds the clean nibbles of ``a mod
    b``, so this is a pure nibble copy (exact)."""
    r_nibbles = L._names["REM"][1]
    spec = _empty_spec(dim, r_nibbles * 2)
    u = 0
    for c in range(r_nibbles):
        u = _clear(spec, u, L.REM + c)
        u = _ident(spec, u, {L.R_NIB + c: 1.0}, 0.0, L.REM + c, 1.0)
    return _truncate(spec, u, dim)


def _bzero_finalize_block(L: _DigitRecLayout, dim: int, b: int) -> Dict[str, torch.Tensor]:
    """``b == 0`` is a COMPILE-TIME constant, so the divide-by-zero fallback is
    resolved AT BUILD TIME: when ``b == 0`` this block SET-clears every quotient
    and remainder nibble to 0 (``(0, 0)`` per ISA_SPEC 4.2); when ``b != 0`` it is
    a no-op (empty) block.  No runtime BZ predicate is needed — the constant folds
    the branch away entirely, one of the payoffs of a constant divisor."""
    q_nibbles = L._names["QUOT"][1]
    r_nibbles = L._names["REM"][1]
    if b != 0:
        return _truncate(_empty_spec(dim, 1), 0, dim)   # no-op
    spec = _empty_spec(dim, q_nibbles + r_nibbles)
    u = 0
    for c in range(q_nibbles):
        u = _clear(spec, u, L.QUOT + c)
    for c in range(r_nibbles):
        u = _clear(spec, u, L.REM + c)
    return _truncate(spec, u, dim)


# ===========================================================================
# Public builder.
# ===========================================================================
@dataclass
class DigitRecDivmod:
    """A built constant-divisor digit-recurrence divmod circuit."""
    b: int
    in_width: int
    dim: int
    layout: _DigitRecLayout
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]]
    # narrowing
    in_nibbles: int
    q_nibbles: int
    r_nibbles: int
    # band indices (convenience)
    A: int
    QUOT: int
    REM: int
    ONE: int
    # fp32 metadata
    max_relu_arg: float
    fp32_exact: bool
    # cost
    n_blocks: int = 0
    nz: int = 0
    blocks_per_step: float = 0.0

    def __post_init__(self):
        self.n_blocks = len(self.blocks)
        self.nz = int(sum(int((s["W_up"] != 0).sum()) + int((s["W_gate"] != 0).sum())
                          + int((s["W_down"] != 0).sum()) + int((s["b_up"] != 0).sum())
                          + int((s["b_gate"] != 0).sum()) + int((s["b_down"] != 0).sum())
                          for _n, s in self.blocks))
        self.blocks_per_step = (self.n_blocks / self.in_nibbles) if self.in_nibbles else 0.0


def const_divmod_fp32_ceiling() -> int:
    """Largest ``b`` for which the FULL-32-bit-dividend digit recurrence is
    byte-exact in fp32 (BOTH q and r).

    The binding fp32 quantity is the silu-relu OUTPUT VALUE, NOT the pre-scale
    argument ``RELU_S·z``: the ``_step_ge`` ramp is a DIFFERENCE of two nearby relu
    outputs (``relu(z-(c-w)) - relu(z-c)``), and catastrophic cancellation makes that
    difference EXACTLY ``w`` even when the shared large value ``z`` rounds — so the
    pre-scale ``RELU_S·z`` (up to ~2e8) is NOT the limit (empirically b passes to
    ~12k at 32-bit though ``RELU_S·15·b`` crossed 2^24 at b≈5.6k).  What DOES bind is
    the reconstructed integer ``z = R'−thr`` (q-staircase) and ``r_top − 16^(rn-1)``
    (readout) staying fp32-exact (< 2^24), i.e. the VALUES the ramp reconstructs.
    The readout of the REMAINDER is the tighter of the two: a remainder with 5+
    nibbles (``b > 2^16``) reconstructs a value up to ~2^20-2^24 and loses bits.  So
    the full-divmod ceiling is ``r_nibbles <= 4`` ⟺ ``b <= 2^16`` for the readout,
    and empirically ``b <~ 12k`` for the q-staircase at the FULL 32-bit dividend
    range.  A narrowed ``in_width`` with ``2^in_width <= b`` forces q ∈ {0,1} and
    ``r = a < 2^in_width``, so ANY ``b`` up to ~2^16 is exact for a suitably bounded
    dividend."""
    return 12000        # empirical full-32-bit-dividend byte-exact ceiling (sweep-verified)


def max_relu_arg_for(b: int, in_width: int = 32, relu_s: float = RELU_S) -> float:
    """The largest silu-relu OUTPUT value the circuit reconstructs — the quantity
    that must stay fp32-exact (< 2^24) for a byte-exact result.

    This is NOT ``RELU_S·z`` (the pre-scale argument, which the ramp's
    difference-of-relus cancels) but the reconstructed integer ``z`` itself: the
    ``k=1`` q-staircase reconstructs up to ``R'_max − b`` and the remainder readout's
    top nibble reconstructs up to ``r_top − 16^(rn-1)``.  ``R'`` ranges over
    ``[0, min(16·b, 2^in_width) − 1]`` (bounded by the invariant ``R' < 16·b`` AND the
    dividend range under narrowing).  The larger of the q and readout values is the
    binding fp32 constraint; a tight ``in_width`` shrinks the q value (narrowing)."""
    if b <= 0:
        return 1.0
    rp_max = min(16 * b, (1 << in_width)) - 1
    q_val = max(0.0, rp_max - b)                          # q-staircase reconstructed value
    r_nibbles = _rem_nibbles(b)
    r_top = min(b - 1, (1 << in_width) - 1)               # largest remainder (bounded by b and range)
    # Readout: the running-residual peel recomposes/subtracts weights up to
    # ``16^(r_nibbles-1)``; the silu-identity residue ``~16^(r_nibbles-1)·1e-7``
    # compounds across the peel, and for ``r_nibbles >= 5`` (weight >= 16^4 = 65536)
    # the seed RREM (up to r < 2^20) already carries ~0.1 residue and the peel
    # tips a nibble.  So the readout binds at ``16^(r_nibbles-1)`` scaled by a
    # residue-amplification factor — model it as the reconstructed remainder value
    # AND require the peel weight ``16^(r_nibbles-1) <= 16^3`` (r_nibbles <= 4).
    readout_val = max(0.0, r_top - 16 ** (r_nibbles - 1))  # readout top-nibble value
    peel_weight = 16 ** (r_nibbles - 1)
    # push the metric past 2^24 when the peel weight exceeds the fp32-exact readout
    # bound (16^3), so ``fp32_exact`` correctly flags the r_nibbles>=5 readout wall.
    if peel_weight > 16 ** 3:
        return float((1 << 24) + peel_weight)             # r-readout not fp32-exact
    return float(max(q_val, readout_val))


def build_const_divmod_digitrec(b: int, in_width: int = 32) -> DigitRecDivmod:
    """Build the constant-divisor digit-recurrence divmod for divisor ``b``.

    ``b`` is a COMPILE-TIME CONSTANT (a python int): the thresholds ``k·b`` are
    baked, so the quotient is a parallel ``_step_ge`` count and the remainder is a
    linear readout — no multiply, no wide subtract, no lexicographic compare.

    ``in_width`` (default 32) statically bounds the dividend so leading dividend
    nibbles are skipped and the quotient is narrowed to its non-zero nibbles.  The
    remainder is narrowed to ``ceil(log16 b)`` nibbles (``r < b``).

    ``b == 0 -> (0, 0)`` (ISA_SPEC 4.2), resolved at build time.

    Returns a ``DigitRecDivmod`` with the ordered block list + band map + metadata.
    """
    assert 0 <= b < (1 << 32), f"b out of 32-bit range: {b}"
    assert 1 <= in_width <= 32, f"in_width out of range: {in_width}"

    in_nibbles = _in_nibbles(in_width)
    # b==0: still allocate real result widths (the finalize block clears them).
    q_nibbles = _quot_nibbles(b if b else 1, in_width)
    r_nibbles = _rem_nibbles(b if b else 1)
    # R' = 16·R + nibble < 16·b needs one nibble more than R (< b).
    rp_nibbles = r_nibbles + 1

    L = _DigitRecLayout(in_nibbles, q_nibbles, r_nibbles, rp_nibbles)
    dim = L.D
    _set_one(L.ONE)                        # point the emitters' _ONE lane at our ONE

    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    if b != 0:
        # MSB-first: process dividend nibble (in_nibbles-1) down to 0.  Iteration t
        # produces quotient nibble (in_nibbles-1-t) — we KEEP only the low q_nibbles
        # (higher ones are provably 0 for a bounded dividend).  R is carried as CLEAN
        # nibbles (R_NIB) so the bring-down is an exact positional shift and the
        # reduced remainder is re-snapped to integer nibbles each step (no fp32
        # cross-step drift).
        for t in range(in_nibbles):
            nib_idx = in_nibbles - 1 - t                 # MSB-first dividend nibble
            q_nib = nib_idx                              # quotient nibble idx == dividend nibble idx
            store_q = (q_nib < q_nibbles)                # provably-zero high nibbles: skip store
            blocks.append((f"cdr-bringdown-{t}",
                           _bringdown_block(L, dim, nib_idx, rp_nibbles)))
            blocks.append((f"cdr-qraw-{t}", _qraw_block(L, dim, b)))
            blocks.append((f"cdr-qsnap-{t}",
                           _qsnap_block(L, dim, q_nib if store_q else 0, store_q)))
            blocks.append((f"cdr-reduce-{t}", _reduce_block(L, dim, b)))
            # re-snap the reduced remainder scalar RVAL into clean R_NIB nibbles
            # (breaks the fp32 residue chain) — skip on the LAST step (the final
            # readout does it once into REM).
            if t < in_nibbles - 1:
                blocks += _renibble_blocks(L, dim, rp_nibbles, tag=str(t))
            else:
                blocks += _renibble_blocks(L, dim, rp_nibbles, tag="final")
        # copy the clean final remainder nibbles into REM.
        blocks.append(("cdr-rem-final", _rem_final_block(L, dim, b)))
    # b==0 (or, harmlessly, b!=0 no-op): clear-to-zero finalize.
    blocks.append(("cdr-finalize", _bzero_finalize_block(L, dim, b)))

    return DigitRecDivmod(
        b=b, in_width=in_width, dim=dim, layout=L, blocks=blocks,
        in_nibbles=in_nibbles, q_nibbles=q_nibbles, r_nibbles=r_nibbles,
        A=L.A, QUOT=L.QUOT, REM=L.REM, ONE=L.ONE,
        max_relu_arg=max_relu_arg_for(b if b else 1, in_width),
        fp32_exact=(max_relu_arg_for(b if b else 1, in_width) < (1 << 24)),
    )


# ===========================================================================
# Apply harness (fp32 forward through the real SwiGLU block).
# ===========================================================================
def apply_spec(x: torch.Tensor, spec: Dict[str, torch.Tensor]) -> torch.Tensor:
    """One SwiGLU FFN block over the residual: ``x + W_down·(silu(W_up·x+b_up) *
    (W_gate·x+b_gate)) + b_down`` — the exact production forward."""
    import torch.nn.functional as F
    up = F.linear(x, spec["W_up"]) + spec["b_up"]
    gate = F.linear(x, spec["W_gate"]) + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, spec["W_down"], spec["b_down"])


def run_divmod(circ: DigitRecDivmod, a: int, dtype=torch.float32) -> Tuple[int, int]:
    """Seed the dividend ``a`` into the A nibble band, run the block list through
    the fp32 SwiGLU forward, and read the quotient / remainder nibbles back.

    Returns ``(q, r)`` masked to 32 bits."""
    qs, rs = run_divmod_batch(circ, [a], dtype=dtype)
    return qs[0], rs[0]


def run_divmod_batch(circ: DigitRecDivmod, xs: List[int],
                     dtype=torch.float32) -> Tuple[List[int], List[int]]:
    """Vectorised: run ALL dividends ``xs`` through the block list in ONE batched
    fp32 SwiGLU forward (a single ``F.linear`` over the whole batch per block).

    Every dividend is an independent residual ROW; the blocks are position-wise so
    the batch is exact and ~1000x faster than looping ``run_divmod``.  Returns
    ``([q...], [r...])`` masked to 32 bits, aligned with ``xs``."""
    import torch.nn.functional as F
    L = circ.layout
    n = len(xs)
    X = torch.zeros(n, circ.dim, dtype=dtype)
    X[:, L.ONE] = 1.0
    for c in range(circ.in_nibbles):
        col = torch.tensor([float((a >> (4 * c)) & 0xF) for a in xs], dtype=dtype)
        X[:, L.A + c] = col
    specs = [{k: v.to(dtype) for k, v in s.items()} for _n, s in circ.blocks]
    for spec in specs:
        up = F.linear(X, spec["W_up"]) + spec["b_up"]
        gate = F.linear(X, spec["W_gate"]) + spec["b_gate"]
        hidden = F.silu(up) * gate
        X = X + F.linear(hidden, spec["W_down"], spec["b_down"])
    qs, rs = [], []
    q_nibs = torch.round(X[:, L.QUOT:L.QUOT + circ.q_nibbles]).to(torch.int64) & 0xF
    r_nibs = torch.round(X[:, L.REM:L.REM + circ.r_nibbles]).to(torch.int64) & 0xF
    for i in range(n):
        q = sum(int(q_nibs[i, c]) << (4 * c) for c in range(circ.q_nibbles))
        r = sum(int(r_nibs[i, c]) << (4 * c) for c in range(circ.r_nibbles))
        qs.append(q & 0xFFFFFFFF)
        rs.append(r & 0xFFFFFFFF)
    return qs, rs


# ===========================================================================
# Byte-exact verification battery (run as __main__).
# ===========================================================================
def _ref_divmod(a: int, b: int) -> Tuple[int, int]:
    """The ISA reference: ``(a // b, a % b)`` masked 32-bit, ``b==0 -> (0, 0)``
    (ISA_SPEC 4.2, matching ``isa.interpret``)."""
    if b == 0:
        return 0, 0
    return (a // b) & 0xFFFFFFFF, (a % b) & 0xFFFFFFFF


# 32-bit-dividend battery: divisors that are fp32-EXACT over the FULL 32-bit range
# (b under the empirical ~8k ceiling), spanning small primes, powers of two, and
# the fp32 boundary.  b >= ~5.6k is beyond the strict RELU_S*15*b < 2^24 bound but
# empirically holds to ~8k (the fp32 mantissa gives headroom past the worst-case).
_BATTERY = [2, 3, 4, 7, 8, 9, 10, 13, 16, 17, 31, 32, 100, 251, 255, 256, 257,
            1000, 1024, 2003, 4096, 4099, 5000, 5592, 8000]
# NARROWED cases carried as (b, in_width): the divisor exceeds the 32-bit fp32
# ceiling, but a STATICALLY-BOUNDED dividend (in_width) keeps every quantity
# fp32-exact — the narrowing that rescues fp32 headroom.  These are chosen so BOTH
# the q-staircase AND the remainder readout stay < 2^24: the remainder needs <= 4
# nibbles (b <= 2^16) for the readout's top-nibble peel (~RELU_S*15*16^(rn-1)) to be
# exact, and the dividend range 2^in_width bounds R'.
#   * 65521 / 40961 / 50021 (primes) with a 16-bit dividend (q in {0,1}, r < 2^16,
#     4 nibbles); 4093 with a 16-bit dividend (narrows q from 6 nibbles to 2).
_BATTERY_NARROW = [(65521, 16), (40961, 16), (50021, 16), (4093, 16)]
# fp32-LIMITED cases: b is so large the REMAINDER itself approaches 2^24, so the
# nibble READOUT (which must have silu emit ~2^20+ precisely) is NOT fp32-exact even
# though the QUOTIENT is (q in {0,1}/small).  Reported honestly — q passes, r is the
# fp32 wall — NOT part of the pass gate.  This is the genuine fp32 boundary of the
# full-divmod readout (a magic-multiply DIV avoids it since it never reads r back).
_BATTERY_FP32_LIMITED = [(1 << 20, 20), (10 ** 6, 20), (1 << 24, 23)]


def _random_dividends(n: int, in_width: int, b: int, seed: int) -> List[int]:
    import random
    rnd = random.Random(seed)
    top = (1 << in_width) - 1
    xs = [rnd.randint(0, top) for _ in range(n)]
    # edges the spec calls out: 0, 1, b-1, b, b+1, 2^32-1 (clamped to in_width).
    for e in (0, 1, b - 1, b, b + 1, top):
        if 0 <= e <= top:
            xs.append(e)
    return xs


def _verify_one(b: int, in_width: int, n_random: int = 2000, seed: int = 0):
    circ = build_const_divmod_digitrec(b, in_width=in_width)
    xs = _random_dividends(n_random, in_width, b if b else 1, seed)
    qs, rs = run_divmod_batch(circ, xs)
    q_ok = r_ok = 0
    q_fail = r_fail = None
    for a, q, r in zip(xs, qs, rs):
        eq, er = _ref_divmod(a, b)
        if q == eq:
            q_ok += 1
        elif q_fail is None:
            q_fail = (a, q, eq)
        if r == er:
            r_ok += 1
        elif r_fail is None:
            r_fail = (a, r, er)
    return circ, len(xs), q_ok, r_ok, q_fail, r_fail


def _print_row(b, inw, circ, tot, q_ok, r_ok, q_fail, r_fail):
    print(f"{b:>9} {inw:>4} {circ.n_blocks:>5} {circ.nz:>7} "
          f"{circ.blocks_per_step:>7.2f} {circ.q_nibbles:>4} {circ.r_nibbles:>4} "
          f"{circ.max_relu_arg:>13.0f} {str(circ.fp32_exact):>5} "
          f"{q_ok:>4}/{tot:<5} {r_ok:>4}/{tot:<5}")
    if q_fail:
        print(f"          q FAIL: a={q_fail[0]} got={q_fail[1]} exp={q_fail[2]}")
    if r_fail:
        print(f"          r FAIL: a={r_fail[0]} got={r_fail[1]} exp={r_fail[2]}")


def main():
    import sys
    print("CONSTANT-DIVISOR digit-recurrence DIVMOD — byte-exact verification (fp32, 0 fp64)\n")
    print(f"RELU_S={RELU_S:.0f}.  The binding fp32 quantity is the silu-relu OUTPUT VALUE")
    print(f"(the reconstructed integer the ramp snaps), NOT the pre-scale arg RELU_S*z — the")
    print(f"ramp's difference-of-relus cancels the large-value rounding.  'maxRelu' column =")
    print(f"the largest reconstructed value; fp32-exact while it stays < 2^24.")
    print(f"  empirical full-32-bit-dividend ceiling: b <~ {const_divmod_fp32_ceiling()}")
    print(f"  (readout of r needs <=4 nibbles => b<=2^16; task's ~56k estimate assumed")
    print(f"   RELU_S=20, this build uses RELU_S=200 with difference-cancellation headroom)\n")
    header = (f"{'b':>9} {'inW':>4} {'blks':>5} {'nz':>7} {'b/step':>7} "
              f"{'qnib':>4} {'rnib':>4} {'maxRelu':>13} {'fp32':>5} "
              f"{'q_pass':>10} {'r_pass':>10}")
    print("=== FULL 32-bit dividend (fp32-exact divisors) ===")
    print(header)
    print("-" * len(header))
    all_ok = True
    for b in dict.fromkeys(_BATTERY):     # dedup preserving order
        circ, tot, q_ok, r_ok, q_fail, r_fail = _verify_one(b, 32)
        all_ok &= (q_ok == tot and r_ok == tot)
        _print_row(b, 32, circ, tot, q_ok, r_ok, q_fail, r_fail)

    print("\n=== NARROWED in_width (large b rescued by a statically-bounded dividend) ===")
    print(header)
    print("-" * len(header))
    for b, inw in _BATTERY_NARROW:
        circ, tot, q_ok, r_ok, q_fail, r_fail = _verify_one(b, inw)
        all_ok &= (q_ok == tot and r_ok == tot)
        _print_row(b, inw, circ, tot, q_ok, r_ok, q_fail, r_fail)

    print("\n=== fp32-LIMITED (remainder ~2^24: q exact, r-readout is the fp32 wall) ===")
    print(header)
    print("-" * len(header))
    for b, inw in _BATTERY_FP32_LIMITED:
        circ, tot, q_ok, r_ok, q_fail, r_fail = _verify_one(b, inw)
        # NOT added to all_ok — this section documents the honest fp32 boundary.
        _print_row(b, inw, circ, tot, q_ok, r_ok, q_fail, r_fail)

    # b==0 special case: (0, 0) for every dividend.
    circ0 = build_const_divmod_digitrec(0, in_width=32)
    z_ok = all(run_divmod(circ0, a) == (0, 0)
               for a in (0, 1, 255, 65535, 0xFFFFFFFF, 123456789))
    print(f"\nb==0 -> (0,0) fold (ISA 4.2): {'PASS' if z_ok else 'FAIL'} "
          f"(blocks={circ0.n_blocks}, nz={circ0.nz})")
    all_ok &= z_ok

    # empirical fp32 ceiling sweep (32-bit dividend): the largest b that is
    # byte-exact over the full 32-bit range with 400 random + edges.
    print("\n=== empirical fp32 ceiling sweep (32-bit dividend) ===")
    last_ok = None
    for b in (4096, 5000, 5592, 6000, 7000, 8000, 8500, 9000, 10000, 12000):
        circ, tot, q_ok, r_ok, _qf, _rf = _verify_one(b, 32, n_random=400, seed=7)
        ok = (q_ok == tot and r_ok == tot)
        print(f"  b={b:>6} fp32_pred={str(circ.fp32_exact):>5} "
              f"maxRelu={circ.max_relu_arg:>11.0f}  byte-exact={ok}")
        if ok:
            last_ok = b
        else:
            break
    print(f"  -> empirical largest fp32-exact b (32-bit dividend): {last_ok}")
    print(f"     predicted (RELU_S*15*b<2^24): {const_divmod_fp32_ceiling()}")
    print("     narrowing: a k-bit-bounded dividend caps R' at min(16b, 2^k), so a")
    print("     tight in_width makes ANY b exact while the dividend stays < ~2^24/15.")

    print(f"\n{'ALL BYTE-EXACT (q and r)' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
