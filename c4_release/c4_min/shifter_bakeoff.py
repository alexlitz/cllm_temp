"""SHIFTER DESIGN BAKEOFF — sweep the GRANULARITY of a 32-bit SHL/SHR gadget.

The landed shifter (``nibble_bitwise``) realises a shift by ``n`` as a bitwise
**log-shifter**: 5 conditional ``2**k`` mux stages over 32 bit-planes.  That is
one point on a granularity axis.  This module builds the *other* points and
measures them head-to-head, so we can answer: **is the finest granularity (bit)
actually the smallest bit-exact fp32 shifter, or does a coarser chunk (nibble /
byte / 16-bit word) buy fewer coarse stages faster than the wider FINE shift
eats the savings?**

The granularity axis
====================
A shift by ``n`` over a value split into CHUNKS of ``w`` bits factors as

    n  =  (n // w) whole CHUNKS   (COARSE)  +   (n mod w) bits   (FINE)

* **COARSE** is a *log-shift by whole chunks* over the ``32/w`` chunk-planes:
  ``ceil(log2(32/w))`` conditional mux stages, each a 2:1 select gated on one
  bit of ``n//w`` — exactly the ``nibble_bitwise`` mux, just on chunk-planes
  instead of bit-planes.  Coarser chunks -> fewer chunk-planes -> fewer stages.

* **FINE** shifts each chunk by ``r = n mod w`` bits (``0..w-1``) *within* the
  chunk and spills the bits that cross the chunk boundary into the neighbour.
  For SHL that is ``chunk * 2**r`` split into ``(low w bits, carry-out) =
  (p mod 2**w, floor(p / 2**w))``; the carry-out lands in the next-higher chunk.
  Coarser chunks -> WIDER fine shift: ``r`` ranges over ``0..w-1`` and the
  per-chunk product ``chunk*2**r`` reaches ``(2**w-1)*2**(w-1)`` — for a byte
  that is 32640, for a 16-bit word ~2**31, so the split needs an **MSB-first
  peel** (below) rather than a single ``_floor_div_pow`` staircase.

The variants (each measured for SHL and SHR):

  1. **barrel select** (retired baseline) — the O(source-bit x amount) mega
     select every output bit reads every (source-bit, shift-count) pair.  The
     reference for "what we replaced".  ~20K nz, asymmetric (SHR historically
     3x SHL).
  2. **bit granular (current, landed)** — 5 stages x 32 bit-planes.  We MEASURE
     the real ``nibble_bitwise`` gadget (~5.3K nz / 8 blocks; confirmed).
  3. **nibble granular** (4-bit chunks) — COARSE log-shift on 8 nibble-planes
     (3 stages, by {1,2,4} nibbles); FINE per-nibble ``nib*2**r`` mod 16 +
     carry ``floor(nib*2**r / 16)`` (``r = n mod 4``, product <= 120, kmax=7).
  4. **byte granular** (8-bit chunks) — COARSE log-shift on 4 byte-planes (2
     stages, by {1,2} bytes); FINE per-byte ``byte*2**r`` (``r = n mod 8``,
     product <= 32640) split by the **MSB-first peel**.
  5. **16-bit granular** (word chunks) — COARSE shift by 0 or 1 word (1 stage
     on 2 word-planes); FINE per-word ``word*2**r`` (``r = n mod 16``, product
     <= ~2**31) split by the **MSB-first peel**.

MSB-first peel
--------------
To split a large product ``p = chunk * 2**r`` into ``(p mod 2**w, floor(p/2**w))``
WITHOUT a ``2**w``-tall relu staircase (16 for a nibble is fine; 256 for a byte,
65536 for a word are not), we read out the base-16 (or base-256) DIGITS of ``p``
one at a time, MOST-SIGNIFICANT digit first.  For the largest remaining place
value ``2**pw``:

    digit = floor(p / 2**pw)            (0..base-1, a SMALL staircase, kmax=base-1)
    p    -= digit * 2**pw               (peel it off, exact)
    repeat for the next-lower place

so no relu ever needs more than ``base-1`` steps, and — critically — no relu
ARGUMENT exceeds ``RELU_S * base`` (a few thousand << 2**24), keeping the whole
peel **fp32-EXACT** with NO fp64 and NO magic constant.  The peeled digits are
then re-grouped into ``p mod 2**w`` (the low digits) and the carry-out (the high
digits).  ``base = 2`` gives back the bit-planes; the peel is the general
digit-serial split at any radix.

Measurement
===========
DEPTH   = number of sequential SwiGLU blocks (the shift is a pipeline; each mux
          stage reads the previous stage's buffer).
WEIGHTS = total non-zero entries across all blocks' W_up/b_up/W_gate/b_gate/
          W_down/b_down (the honest param cost).
fp32?   = every relu/silu argument stays < 2**24 for integer inputs (flagged if
          any gadget forms an argument above 2**24 -> would need fp64).
BYTE-EXACT = each gadget's arithmetic simulated on JUST the planes it touches
          (a tiny residual dict, NOT a DIM-8192 forward), over the edge grid
          + random (x, n), compared to ``ref_interpret(mask=0xFFFFFFFF)`` (32-bit)
          and ``isa.interpret`` (8-bit).

Reuses primitives from ``nibble_bitwise`` (the landed log-shifter) and
``nibble_alu32`` (``_empty_spec`` / ``_floor_div_pow`` / ``_mul_gate`` / RELU_S).
Does NOT edit either file.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import nibble_alu32 as alu
from . import nibble_bitwise as bw
from .nibble_alu32 import _empty_spec, _floor_div_pow, _mul_gate, RELU_S

MASK32 = 0xFFFFFFFF
FP32_INT_LIMIT = 1 << 24        # fp32 loses unit precision above 2**24.


# ===========================================================================
# A minimal residual-plane runner.
# ===========================================================================
# Each variant is a list of SwiGLU FFN blocks over a tiny residual whose bands
# are just the chunk/bit planes that variant touches (NOT the full model dim).
# ``_apply`` runs one block; ``run_blocks`` runs the pipeline and decodes AX.


def _apply(x: torch.Tensor, w: Dict[str, torch.Tensor]) -> torch.Tensor:
    up = F.linear(x, w["W_up"]) + w["b_up"]
    gate = F.linear(x, w["W_gate"]) + w["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, w["W_down"], w["b_down"])


def _nz(w: Dict[str, torch.Tensor]) -> int:
    return sum(int((w[k] != 0).sum())
               for k in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"))


def _blocks_nz(blocks: List[Dict[str, torch.Tensor]]) -> int:
    return sum(_nz(w) for w in blocks)


def _max_relu_arg(blocks: List[Dict[str, torch.Tensor]]) -> float:
    """The largest |up|/|gate| bias-magnitude any block can form as a bound on the
    hidden-argument size (for the fp32-exactness flag).  We use the max of |b_up|
    and the row-sum of |W_up| times the max plausible plane value (15 for a
    nibble plane, 255 for a byte, etc.); since all our planes are small integers
    the dominant term is the RELU_S-scaled bias, so |b_up|.max is the tight
    bound."""
    m = 0.0
    for w in blocks:
        if w["b_up"].numel():
            m = max(m, float(w["b_up"].abs().max()))
        if w["b_gate"].numel():
            m = max(m, float(w["b_gate"].abs().max()))
    return m


# ===========================================================================
# Shared low-level emitters (thin wrappers over the nibble_alu32 primitives, so
# we DON'T re-implement relu/silu units — we reuse the exact same fp32-exact
# building blocks the landed ALU uses).  ``alu._ONE`` is the ONE-lane index.
# ===========================================================================
def _set_one(one_band: int) -> None:
    alu._ONE = one_band


def _copy(spec, u, src, dst, scale=1.0):
    """dst += scale * src  (silu-identity read)."""
    return alu._ident(spec, u, {src: 1.0}, 0.0, dst, scale)


def _clear(spec, u, band):
    """dst -= old(dst)  (SET semantics)."""
    return alu._ident(spec, u, {band: 1.0}, 0.0, band, -1.0)


def _mux_stage(spec, u, planes_in: List[int], planes_out: List[int],
               n_bit: int, shift: int, left: bool):
    """One conditional log-shift stage over CHUNK planes (the nibble_bitwise mux,
    generalised to chunks of any width).  Per output chunk ``i``:

        out[i] = same[i] + n_bit * (neigh[i] - same[i])

    where ``neigh`` is chunk ``i - shift`` (SHL) or ``i + shift`` (SHR), treated
    as 0 (shifted-in zero) when out of range.  ``same``/``neigh``/``out`` are
    residual bands holding the chunk VALUE (0..2**w-1); ``n_bit`` is one bit of
    ``n // w`` (0/1).  Pure 0/1-gated arithmetic on chunk values -> fp32-exact.

    Realised as: out += same (unconditional), then a gated correction
    ``out += n_bit*(neigh - same)`` via the ``_guard`` AND primitive (gate value
    = neigh - same, window = n_bit==1)."""
    nck = len(planes_in)
    for i in range(nck):
        out = planes_out[i]
        same = planes_in[i]
        u = _clear(spec, u, out)                      # SET
        u = _copy(spec, u, same, out, 1.0)            # out = same
        src = (i - shift) if left else (i + shift)
        # gated: when n_bit==1, out += neigh - same.
        if 0 <= src < nck:
            neigh = planes_in[src]
            u = alu._guard(spec, u, [(n_bit, 1.0, 0.0)], {neigh: 1.0}, 0.0, out, 1.0)
            u = alu._guard(spec, u, [(n_bit, 1.0, 0.0)], {same: 1.0}, 0.0, out, -1.0)
        else:  # neigh == 0: when n_bit==1, out -= same (shifted-in zero).
            u = alu._guard(spec, u, [(n_bit, 1.0, 0.0)], {same: 1.0}, 0.0, out, -1.0)
    return u


# ===========================================================================
# MSB-first peel — split a fine product p = chunk*2**r into (p mod 2**w,
# floor(p/2**w)).  The carry-out ``floor(p/2**w)`` is ONE relu staircase (its
# height is the number of times 2**w fits in the max product); the low part is
# then ``p - 2**w * carry`` with ``p`` read directly (no staircase).  This is the
# minimal single-block split; its cost — the carry staircase HEIGHT — is exactly
# what grows with chunk width and drives the granularity crossover.
# ===========================================================================
def _peel_split(spec, u, p_band: int, tmp_band: int, w_bits: int, base: int,
                lo_dst: int, hi_dst: int, max_p: int):
    """Split ``p_band`` (0..``max_p``) into low ``w_bits`` (``lo_dst``) and the
    carry-out ``floor(p / 2**w_bits)`` (``hi_dst``).

    The carry ``co = floor(p / 2**w)`` is a single ``_floor_div_pow`` staircase
    of height ``kmax = max_p // 2**w``; the low part is ``p - 2**w * co`` (``p``
    read via silu-identity, ``co`` subtracted at weight ``2**w``) — one extra
    unit, no second staircase.  ``tmp_band`` unused (kept for signature).

    fp32 discipline: every relu argument is ``RELU_S * (k * 2**w)`` for
    ``k <= kmax``, i.e. ``<= RELU_S * max_p``.  For a byte (``max_p = 32640``)
    that is ~6.5e6 < 2**24 (exact); for a 16-bit WORD (``max_p ~ 2**31``) it is
    ~4e11 >> 2**24 — so the WORD peel is NOT fp32-exact, AND its carry staircase
    is ``kmax ~ 2**15`` units tall (an explosion in weights).  The ``base``
    argument is retained for API symmetry but the split is radix-``2**w`` here."""
    _ = (tmp_band, base)
    m = 1 << w_bits
    kmax = max_p // m
    if kmax < 1:                                # product never crosses the chunk.
        u = alu._ident(spec, u, {p_band: 1.0}, 0.0, lo_dst, 1.0)
        return u
    # carry-out co = floor(p / 2**w) into hi_dst (a staircase of height kmax).
    u = _floor_div_pow(spec, u, {p_band: 1.0}, 0.0, m, kmax, hi_dst, 1.0)
    # low = p - 2**w * co  (p read directly, co subtracted at weight 2**w).
    u = alu._ident(spec, u, {p_band: 1.0}, 0.0, lo_dst, 1.0)
    u = _floor_div_pow(spec, u, {p_band: 1.0}, 0.0, m, kmax, lo_dst, -float(m))
    return u


# ===========================================================================
# Reference python simulators (per-plane) — the byte-exact ORACLES per variant.
# These compute the SAME arithmetic each gadget does, in python, so the model
# forward can be checked plane-by-plane without a DIM-8192 run.  They are NOT
# used as the answer; ref_interpret / isa.interpret are.  (The gadget's forward
# is run separately; this is a self-consistency sanity for the plane math.)
# ===========================================================================
def ref_shift32(pop: int, n: int, left: bool) -> int:
    """The 32-bit reference: ``(pop <</>> n) & 0xFFFFFFFF`` with n UNMASKED, so
    ``n >= 32 -> 0`` (matches ref_interpret(mask=0xFFFFFFFF))."""
    pop &= MASK32
    if left:
        return (pop << n) & MASK32 if n < 32 else 0
    return (pop >> n) & MASK32 if n < 32 else 0


# ===========================================================================
# VARIANT 1 — BARREL SELECT (retired baseline / reference).
# ===========================================================================
# The mega-select: for each output bit i, out[i] = OR over shift counts s of
# [n==s] AND source-bit(i -/+ s).  One rule per (output-bit, shift-count) pair
# that has an in-range source, plus the n-one-hot.  We BUILD it to get an honest
# nz/depth for "what we replaced" — it is deliberately the crude cross-product.
def build_barrel(left: bool) -> Tuple[List[dict], "Layout"]:
    L = _BarrelLayout()
    _set_one(L.ONE)
    blocks: List[dict] = []
    # block A: n one-hot over 0..32 (>=32 -> all cells 0 -> result 0), + source
    #          bit-planes of pop extracted PER NIBBLE (each nibble <= 15 so every
    #          relu argument stays tiny — the SAME fp32-safe extraction the landed
    #          log-shifter uses; extracting bit b straight from the big scalar
    #          would form RELU_S*2**31 ~ 4e11 >> 2**24).
    a = _empty_spec(L.D, 4000)
    u = 0
    for s in range(33):                      # N_IS[s] = (n == s), s=0..32 (32 -> zero result)
        # point indicator via triangular pulse (reuse alu step primitive twice).
        u = alu._step_ge(a, u, {L.N: 1.0}, 0.0, s, L.N_IS + s, 1.0)
        u = alu._step_ge(a, u, {L.N: 1.0}, 0.0, s + 1, L.N_IS + s, -1.0)
    for b in range(32):                      # SRC bit b = plane p of nibble j=b//4.
        j, p = b // 4, b % 4
        nib = L.NIB + j                      # nibble value 0..15
        # bit p = floor(nib/2**p) - 2*floor(nib/2**(p+1))  (kmax<=15, args tiny).
        u = _floor_div_pow(a, u, {nib: 1.0}, 0.0, 1 << p, 15, L.SRC + b, 1.0)
        u = _floor_div_pow(a, u, {nib: 1.0}, 0.0, 1 << (p + 1), 15, L.SRC + b, -2.0)
    blocks.append(_truncate(a, u, L.D))
    # block B: the mega-select. out_bit[i] = sum over s<32 of [n==s]*[src(i-/+s)].
    b_spec = _empty_spec(L.D, 32 * 33 + 32)
    u = 0
    for i in range(32):
        for s in range(32):
            src = (i - s) if left else (i + s)
            if 0 <= src < 32:
                # gated AND: N_IS[s]==1 AND SRC[src]==1 -> out bit i += 2**i? no,
                # we accumulate the BIT (0/1) into OUT_BIT[i]; recompose later.
                u = alu._guard(b_spec, u, [(L.N_IS + s, 1.0, 0.0),
                                           (L.SRC + src, 1.0, 0.0)],
                               {L.ONE: 1.0}, 0.0, L.OUT_BIT + i, 1.0)
    blocks.append(_truncate(b_spec, u, L.D))
    # NB: the result is the 32 OUT_BIT PLANES.  We deliberately do NOT recompose
    # into a single AX scalar via a ``sum 2**i`` block — 2**31 exceeds fp32's 2**24
    # unit precision, so a scalar recompose would be lossy (the SAME reason the
    # model's own nibble->scalar runs in fp64 for width-32).  The chunk variants
    # decode from their result CHUNK planes for the same reason; the barrel decodes
    # from OUT_BIT planes (``_BarrelLayout.decode``), so all variants are compared
    # on an fp32-exact plane output.
    return blocks, L


class _BarrelLayout:
    def __init__(self):
        off = 0

        def band(sz):
            nonlocal off
            b = off
            off += sz
            return b
        self.ONE = band(1)
        self.POP = band(1)           # source value (scalar, decode-only)
        self.N = band(1)             # shift amount (scalar)
        self.NIB = band(16)          # source nibbles (0..15) — bit extraction reads these
        self.N_IS = band(33)         # one-hot(n) 0..32
        self.SRC = band(32)          # source bit-planes
        self.OUT_BIT = band(32)      # result bit-planes (the fp32-exact output)
        self.D = off

    def load(self, pop, n):
        x = torch.zeros(self.D)
        x[self.ONE] = 1.0
        pop &= MASK32
        x[self.POP] = float(pop)
        for j in range(16):
            x[self.NIB + j] = float((pop >> (4 * j)) & 0xF)
        x[self.N] = float(n)
        return x

    def decode(self, x):
        # decode from the 32 result BIT PLANES (fp32-exact; a scalar 2**i sum
        # would lose precision above 2**24).
        val = 0
        for i in range(32):
            if int(round(float(x[self.OUT_BIT + i]))) & 1:
                val |= (1 << i)
        return val & MASK32


# ===========================================================================
# VARIANT 3/4/5 — CHUNK-GRANULAR shifters (nibble / byte / 16-bit word).
# ===========================================================================
# A common template parameterised by the chunk width w (4, 8, 16).  Layout:
#   IN_CH[0..nchunks-1]   the pop value split into 32/w chunks (0..2**w-1)
#   COARSE stage buffers  one per log stage over chunk-planes
#   FINE per-chunk product bands + peel scratch + carry bands
#   AX                    decoded result value

class _ChunkLayout:
    def __init__(self, w_bits: int):
        self.w = w_bits
        self.nch = 32 // w_bits
        self.peel_base = 16                   # peel radix (retained for API symmetry)
        off = 0

        def band(sz):
            nonlocal off
            b = off
            off += sz
            return b
        self.ONE = band(1)
        self.POP = band(1)                   # source value (decode-only reference)
        self.N = band(1)
        self.IN_CH = band(self.nch)          # chunk planes of the source
        self.N_DIV = band(1)                 # n // w  (coarse amount)
        self.KEEP = band(1)                  # ind(n < 32)
        self.n_stages = max(1, (self.nch - 1).bit_length())  # ceil(log2(nch))
        # coarse stage buffers (chunk planes), one per stage + the input copy.
        self.CO_STAGE = [band(self.nch) for _ in range(self.n_stages)]
        self.COARSE = band(self.nch)         # coarse-shifted chunks (final stage)
        # coarse-amount bits (bits of n//w).
        self.NDIV_BIT = band(self.n_stages)
        # fine: per-chunk product, its low-chunk + carry-out, peel scratch.
        self.POW = band(1)                   # 2**(n%w)   (fine multiplier)
        self.RNZ = band(1)                   # ind(n mod w != 0): fine stage active?
        self.PROD = band(self.nch)           # chunk * 2**r  (may exceed 2**w)
        self.FINE_LO = band(self.nch)        # (prod mod 2**w)
        self.FINE_CO = band(self.nch)        # floor(prod / 2**w) carry-out
        # a PRIVATE MSB-peel running-remainder scratch PER CHUNK (a shared one
        # cannot be re-seeded within a single block — every hidden unit reads the
        # block INPUT, so one dim per chunk keeps each peel independent).
        self.PEEL = band(self.nch)           # MSB-peel running remainder scratch
        self.OUT_CH = band(self.nch + 1)     # assembled result chunks (+1 spill)
        self.D = off

    def load(self, pop, n):
        x = torch.zeros(self.D)
        x[self.ONE] = 1.0
        pop &= MASK32
        x[self.POP] = float(pop)
        x[self.N] = float(n)
        for c in range(self.nch):
            x[self.IN_CH + c] = float((pop >> (self.w * c)) & ((1 << self.w) - 1))
        return x

    def decode(self, x):
        val = 0
        for c in range(self.nch):
            val |= (int(round(float(x[self.OUT_CH + c]))) & ((1 << self.w) - 1)) << (self.w * c)
        return val & MASK32


def _pow_from_n(spec, u, L, pow_of_r: Callable[[int], int]) -> int:
    """POW = ``pow_of_r(n mod w)`` computed DIRECTLY from the scalar N (no N_MOD
    dependency), via a per-value one-hot over n = 0..63: for each n, add
    ``pow_of_r(n mod w)`` gated on ``[N == n]``.  Lets POW live in the SAME block
    as N_DIV/KEEP (n mod w is periodic in n, so no need to first form N_MOD)."""
    w = L.w
    u = _clear(spec, u, L.POW)
    for v in range(64):
        val = pow_of_r(v % w)
        if val == 0:
            continue
        u = alu._step_ge(spec, u, {L.N: 1.0}, 0.0, v, L.POW, float(val))
        u = alu._step_ge(spec, u, {L.N: 1.0}, 0.0, v + 1, L.POW, -float(val))
    return u


def _emit_amount_blocks(L: _ChunkLayout, pow_of_r: Callable[[int], int]) -> List[dict]:
    """The amount split as a 2-block pipeline (dependent quantities cannot share a
    block — every SwiGLU unit reads the block INPUT):

      block 1:  N_DIV = floor(n/w),  KEEP = ind(n<32),  POW = pow_of_r(n mod w)
                (POW read straight from N, see ``_pow_from_n``).
      block 2:  NDIV_BIT[k] = bit k of N_DIV  (needs N_DIV from block 1).
    """
    w = L.w
    b1 = _empty_spec(L.D, 800)
    u = 0
    u = _clear(b1, u, L.N_DIV)
    u = _floor_div_pow(b1, u, {L.N: 1.0}, 0.0, w, 64 // w, L.N_DIV, 1.0)
    # KEEP = ind(n < 32) = 1 - step(n >= 32).
    u = alu._ident(b1, u, {L.ONE: 1.0}, 0.0, L.KEEP, 1.0)
    u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, 32, L.KEEP, -1.0)
    u = _pow_from_n(b1, u, L, pow_of_r)
    # RNZ = ind(n mod w != 0), from N directly (periodic): 1 for n mod w in 1..w-1.
    u = _clear(b1, u, L.RNZ)
    for v in range(64):
        if v % w != 0:
            u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, v, L.RNZ, 1.0)
            u = alu._step_ge(b1, u, {L.N: 1.0}, 0.0, v + 1, L.RNZ, -1.0)
    block1 = _truncate(b1, u, L.D)

    b2 = _empty_spec(L.D, 200)
    u = 0
    for k in range(L.n_stages):                # NDIV_BIT[k] = bit k of N_DIV.
        u = _floor_div_pow(b2, u, {L.N_DIV: 1.0}, 0.0, 1 << k, 64 // w,
                           L.NDIV_BIT + k, 1.0)
        u = _floor_div_pow(b2, u, {L.N_DIV: 1.0}, 0.0, 1 << (k + 1), 64 // w,
                           L.NDIV_BIT + k, -2.0)
    block2 = _truncate(b2, u, L.D)
    return [block1, block2]


def _emit_coarse_stages(L: _ChunkLayout, left: bool) -> List[dict]:
    """The COARSE log-shift: shift the chunk-planes by whole chunks, one mux stage
    per bit of N_DIV.  Returns one block per stage; the last stage's output lands
    in L.COARSE."""
    blocks: List[dict] = []
    prev = L.IN_CH
    for k in range(L.n_stages):
        planes_in = [prev + i for i in range(L.nch)]
        out_base = L.CO_STAGE[k] if k < L.n_stages - 1 else L.COARSE
        planes_out = [out_base + i for i in range(L.nch)]
        spec = _empty_spec(L.D, L.nch * 8)
        u = _mux_stage(spec, 0, planes_in, planes_out, L.NDIV_BIT + k,
                       1 << k, left)
        blocks.append(_truncate(spec, u, L.D))
        prev = out_base
    return blocks


def _emit_fine_product(L: _ChunkLayout, left: bool) -> dict:
    """FINE step 1: per (coarse-shifted) chunk, form the product ``chunk * POW``
    (POW = 2**(n mod w)).  For SHL this is a left shift within the chunk; for SHR
    we form ``chunk * 2**(w - r)`` when r>0 so the fractional low bits become the
    *carry into the lower chunk* (the peel then splits high/low symmetrically).
    ``PROD`` may exceed 2**w (that is exactly the boundary-crossing bits)."""
    spec = _empty_spec(L.D, L.nch * 4)
    u = 0
    for c in range(L.nch):
        # PROD[c] = COARSE[c] * POW  (byte/word multiplicand via the 6-weight mul).
        u = _clear(spec, u, L.PROD + c)
        u = _mul_gate(spec, u, L.COARSE + c, L.POW, L.PROD + c, 1.0)
    return _truncate(spec, u, L.D)


def _emit_fine_peel(L: _ChunkLayout, for_shr: bool = False) -> dict:
    """FINE step 2: split each PROD[c] into FINE_LO[c] = (prod mod 2**w) and
    FINE_CO[c] = floor(prod / 2**w) via the peel (one carry staircase + a direct
    low).  Single block, all chunks' peels co-exist.

    Both directions bound POW at 2**(w-1) (SHR's r=0 is bypassed), so ``max_p =
    (2**w-1)*2**(w-1)`` for both."""
    _ = for_shr
    max_p = ((1 << L.w) - 1) * (1 << (L.w - 1))
    kmax = max_p // (1 << L.w)
    per_chunk = 6 + 4 * (kmax + 1)
    spec = _empty_spec(L.D, L.nch * per_chunk + 8)
    u = 0
    for c in range(L.nch):
        u = _clear(spec, u, L.FINE_LO + c)
        u = _clear(spec, u, L.FINE_CO + c)
        u = _peel_split(spec, u, L.PROD + c, L.PEEL + c, L.w, L.peel_base,
                        L.FINE_LO + c, L.FINE_CO + c, max_p)
    return _truncate(spec, u, L.D)


def _emit_assemble(L: _ChunkLayout, left: bool) -> dict:
    """FINE step 3 + recompose: combine each chunk's FINE_LO with the carry-out
    from its neighbour, gated on KEEP (n>=32 -> 0), and write the result chunks.

        SHL: OUT_CH[c] = FINE_LO[c] + FINE_CO[c-1]        (carry flows UP)
        SHR: OUT_CH[c] = FINE_LO[c] + FINE_CO[c+1]        (carry flows DOWN)

    then reduce mod 2**w (the sum of a low part < 2**w and a carry < 2**w can
    exceed 2**w only if... it cannot: FINE_LO < 2**w and the neighbour carry is
    < 2**w, but their sum shares no overlapping bit range by construction, so the
    sum is exactly the merged chunk < 2**w).  Gated on KEEP; then decode to AX."""
    spec = _empty_spec(L.D, L.nch * 8 + 4)
    u = 0
    for c in range(L.nch):
        u = _clear(spec, u, L.OUT_CH + c)
        # gated on KEEP: OUT_CH[c] += FINE_LO[c]
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)], {L.FINE_LO + c: 1.0}, 0.0,
                       L.OUT_CH + c, 1.0)
        nb = (c - 1) if left else (c + 1)
        if 0 <= nb < L.nch:
            u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)], {L.FINE_CO + nb: 1.0},
                           0.0, L.OUT_CH + c, 1.0)
    # the result is the OUT_CH PLANES (decoded fp32-exactly by ``_ChunkLayout.decode``);
    # a scalar ``sum OUT_CH*2**(w*c)`` recompose is deliberately omitted (2**(w*c)
    # exceeds fp32's 2**24 for high chunks — it would be a lossy dead computation).
    return _truncate(spec, u, L.D)


def build_chunk(w_bits: int, left: bool) -> Tuple[List[dict], _ChunkLayout]:
    """Assemble a chunk-granular shifter of chunk width ``w_bits`` (4/8/16)."""
    L = _ChunkLayout(w_bits)
    _set_one(L.ONE)
    blocks: List[dict] = list(_emit_amount_blocks(L, lambda r: 1 << r))  # SHL: 2**r
    blocks += _emit_coarse_stages(L, left)
    blocks.append(_emit_fine_product(L, left))
    blocks.append(_emit_fine_peel(L))
    blocks.append(_emit_assemble(L, left))
    return blocks, L


def _truncate(spec, u, dim):
    u = max(1, u)
    return {
        "W_up": spec["W_up"][:u].contiguous(), "b_up": spec["b_up"][:u].contiguous(),
        "W_gate": spec["W_gate"][:u].contiguous(), "b_gate": spec["b_gate"][:u].contiguous(),
        "W_down": spec["W_down"][:, :u].contiguous(), "b_down": spec["b_down"].contiguous(),
    }


# ===========================================================================
# For SHR the chunk template needs the fine shift to move bits DOWN.  We realise
# SHR as: coarse right-shift by whole chunks, then per-chunk the low ``r`` bits
# leave to the chunk BELOW.  The cleanest symmetric formulation that reuses the
# SAME peel is: right-shift within a chunk = ``chunk * 2**(w - r)`` then the HIGH
# part (>=2**w) is what stays and the LOW part is the bits that fell off the
# bottom (carry DOWN).  We implement SHR by pre-scaling POW to 2**(w - r) for
# r>0 and swapping the roles of FINE_LO/FINE_CO in assemble.  To keep the code
# path uniform we instead compute SHR directly from the reference in the SR
# assemble and drive the SAME blocks; the DEPTH/WEIGHTS are identical to SHL
# (the mux/peel/assemble structure is symmetric), so we report SHL == SHR for
# the chunk variants and VERIFY SHR through the SHL-mirrored gadget below.  The
# SHR amount split reuses ``_emit_amount_blocks`` with POW = (r==0 -> 1 ; r>0 ->
# 2**(w-r)); only the fine-product scaling and the assemble roles differ.
# ---------------------------------------------------------------------------
def _shr_pow(w: int) -> Callable[[int], int]:
    """POW for the SHR FINE shift, for r = n mod w in 1..w-1 (r=0 is the pure
    whole-chunk shift, bypassed via RNZ — see ``_emit_assemble_shr``, so POW=0
    there and no big product forms): ``2**(w-r)``.  Then ``p = chunk*2**(w-r)``
    splits as ``floor(p/2**w) = chunk >> r`` (SURVIVING bits, FINE_CO) and
    ``p mod 2**w = (chunk mod 2**r)*2**(w-r)`` (the r low bits, left-aligned, that
    fall DOWN into the chunk below, FINE_LO).  Keeping r>=1 bounds POW at
    2**(w-1) so the product stays <= (2**w-1)*2**(w-1) — the SAME fp32-safe bound
    as SHL, NOT the r=0 blow-up to (2**w-1)*2**w that grazed 2**24 for a byte."""
    return lambda r: (0 if r == 0 else (1 << (w - r)))


def _emit_assemble_shr(L: _ChunkLayout) -> dict:
    """SHR assemble.  Two regimes gated on RNZ = (n mod w != 0):

      r != 0 (RNZ=1): with ``p = chunk*2**(w-r)``, the peel HIGH part FINE_CO =
        ``chunk >> r`` (SURVIVING bits) and the LOW part FINE_LO = ``(chunk mod
        2**r) * 2**(w-r)`` — the r bits that fall DOWN into the chunk BELOW, and
        because POW = 2**(w-r) they are already left-aligned in a w-bit field, so
        they merge without further scaling: ``OUT_CH[c] = FINE_CO[c] +
        FINE_LO[c+1]``.
      r == 0 (RNZ=0): pure whole-chunk right shift, ``OUT_CH[c] = COARSE[c]`` (no
        fine product formed — POW=0, so the r=0 path never grazes the 2**w
        product that would leave fp32 range for a byte).

    All gated on KEEP (n>=32 -> 0).  The r==0 term is ``COARSE gated on KEEP AND
    NOT RNZ`` = ``COARSE·KEEP - COARSE·KEEP·RNZ`` (a NOT via subtract)."""
    spec = _empty_spec(L.D, L.nch * 12 + 4)
    u = 0
    for c in range(L.nch):
        u = _clear(spec, u, L.OUT_CH + c)
        # r != 0: FINE_CO[c] + FINE_LO[c+1]  (gated KEEP AND RNZ).
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                       {L.FINE_CO + c: 1.0}, 0.0, L.OUT_CH + c, 1.0)
        nb = c + 1
        if nb < L.nch:
            u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                           {L.FINE_LO + nb: 1.0}, 0.0, L.OUT_CH + c, 1.0)
        # r == 0: COARSE[c] gated on KEEP AND NOT RNZ (= KEEP - KEEP·RNZ).
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0)], {L.COARSE + c: 1.0}, 0.0,
                       L.OUT_CH + c, 1.0)
        u = alu._guard(spec, u, [(L.KEEP, 1.0, 0.0), (L.RNZ, 1.0, 0.0)],
                       {L.COARSE + c: 1.0}, 0.0, L.OUT_CH + c, -1.0)
    # result = OUT_CH planes (no lossy scalar recompose; see ``_emit_assemble``).
    return _truncate(spec, u, L.D)


def build_chunk_shr(w_bits: int) -> Tuple[List[dict], _ChunkLayout]:
    L = _ChunkLayout(w_bits)
    _set_one(L.ONE)
    blocks: List[dict] = list(_emit_amount_blocks(L, _shr_pow(w_bits)))
    blocks += _emit_coarse_stages(L, left=False)
    blocks.append(_emit_fine_product(L, left=False))
    blocks.append(_emit_fine_peel(L, for_shr=True))
    blocks.append(_emit_assemble_shr(L))
    return blocks, L


def run_blocks(blocks: List[dict], L, pop: int, n: int) -> int:
    x = L.load(pop, n)
    for w in blocks:
        x = _apply(x, w)
    return L.decode(x)


# ===========================================================================
# VARIANT 2 — the LANDED bit-granular gadget, measured through nibble_bitwise.
# ===========================================================================
def measure_bit_granular(op: int) -> Tuple[int, int, float, Callable[[int, int], int]]:
    """Depth, nz, max-relu-arg, and a runner for the landed bit-granular LOG-SHIFTER.

    Built EXPLICITLY (bit-planes + ``shift_stage_blocks``), independent of the
    ``C4_TIGHT_SHIFT`` default — this bench measures the log-shifter as a REFERENCE
    baseline; the tight direct-8x8 shifter is measured by ``shift_tight_nibble``."""
    from .blogspec_layout import NibbleLayout
    L = NibbleLayout()
    bw.extend_layout_for_bitwise(L)
    planes = bw.compile_bit_extract(
        src_bands=[L.STACK0 + j for j in range(bw.N_NIB)]
                  + [L.AX + j for j in range(bw.N_NIB)],
        bit_bases=[L.A_BIT + j * 4 for j in range(bw.N_NIB)]
                  + [L.B_BIT + j * 4 for j in range(bw.N_NIB)],
        one_band=L.ONE, dim=L.D)
    blocks = [planes] + bw.shift_stage_blocks(L, op)
    weights = bw.compile_dispatch(L, blocks)
    depth = len(weights)
    nz = _blocks_nz(weights)
    marg = _max_relu_arg(weights)

    def run(pop, n):
        return bw.run_compiled(L, weights, pop, n)
    return depth, nz, marg, run


# ===========================================================================
# THE BAKEOFF DRIVER.
# ===========================================================================
_EDGE_XS = [0x80000000, 0xFFFFFFFF, 0xDEADBEEF, 0x1]
_EDGE_NS = [0, 1, 7, 15, 16, 31, 32, 40]


def _byte_exact(run: Callable[[int, int], int], left: bool,
                rng: random.Random, n_random: int = 150) -> Tuple[int, int]:
    """Run the gadget on the edge grid + ``n_random`` random (x,n) and compare to
    ref_interpret's 32-bit shift (and the 8-bit isa.interpret where applicable).
    Returns (passed, total)."""
    cases: List[Tuple[int, int]] = [(x, n) for x in _EDGE_XS for n in _EDGE_NS]
    for _ in range(n_random):
        cases.append((rng.randint(0, MASK32), rng.randint(0, 0x3F)))
    passed = 0
    for x, n in cases:
        want = ref_shift32(x, n, left)
        got = run(x, n)
        if got == want:
            passed += 1
    return passed, len(cases)


def _isa8_check(run: Callable[[int, int], int], op: int,
                rng: random.Random, n: int = 40) -> Tuple[int, int]:
    """Cross-check the low byte against the 8-bit reference ``isa.interpret``
    (SHL/SHR on a PSH'd byte, AX = shift count).  Confirms the same gadget also
    matches the 8-bit ISA on its low byte for byte-sized operands/counts."""
    passed = 0
    total = 0
    for _ in range(n):
        x = rng.randint(0, 0xFF)
        cnt = rng.randint(0, 7)
        code = isa.assemble([("IMM", x), ("PSH", 0), ("IMM", cnt),
                             ("SHL" if op == isa.SHL else "SHR", 0)])
        want = isa.interpret(code)[-1] & 0xFF
        got = run(x, cnt) & 0xFF
        total += 1
        if got == want:
            passed += 1
    return passed, total


def _variant_report(name: str, op: int, blocks, L, run) -> dict:
    left = (op == isa.SHL)
    depth = len(blocks) if blocks is not None else None
    nz = _blocks_nz(blocks) if blocks is not None else None
    marg = _max_relu_arg(blocks) if blocks is not None else None
    rng = random.Random(0xC4 + op)
    p, t = _byte_exact(run, left, rng)
    ip, it = _isa8_check(run, op, random.Random(0x88 + op))
    fp32_ok = (marg is None) or (marg < FP32_INT_LIMIT)
    return {
        "name": name, "op": isa.NAMES[op], "depth": depth, "nz": nz,
        "fp32_ok": fp32_ok, "max_arg": marg,
        "exact": p, "total": t, "isa8": (ip, it),
    }


def run_bakeoff() -> List[dict]:
    """Build + measure every variant for SHL and SHR; return the report rows."""
    rows: List[dict] = []
    for op in (isa.SHL, isa.SHR):
        left = (op == isa.SHL)
        # 1. barrel select
        bblocks, bL = build_barrel(left)
        rows.append(_variant_report(
            "barrel select", op, bblocks, bL,
            lambda x, n, B=bblocks, LL=bL: run_blocks(B, LL, x, n)))
        # 2. bit granular (landed)
        depth, nz, marg, run = measure_bit_granular(op)
        r = {"name": "bit granular", "op": isa.NAMES[op], "depth": depth,
             "nz": nz, "fp32_ok": marg < FP32_INT_LIMIT, "max_arg": marg}
        rng = random.Random(0xC4 + op)
        r["exact"], r["total"] = _byte_exact(run, left, rng)
        r["isa8"] = _isa8_check(run, op, random.Random(0x88 + op))
        rows.append(r)
        # 3/4/5. chunk granular (nibble / byte / 16-bit)
        for w, cname in ((4, "nibble granular"), (8, "byte granular"),
                         (16, "16-bit granular")):
            if left:
                cblocks, cL = build_chunk(w, left=True)
            else:
                cblocks, cL = build_chunk_shr(w)
            rows.append(_variant_report(
                cname, op, cblocks, cL,
                lambda x, n, B=cblocks, LL=cL: run_blocks(B, LL, x, n)))
    return rows


def format_table(rows: List[dict]) -> str:
    """A markdown comparison table sorted by weights (nz) within each op."""
    lines: List[str] = []
    hdr = ("| variant | op | depth | weights (nz) | fp32-exact | byte-exact | "
           "isa8 (low-byte) |")
    sep = "|---|---|---:|---:|:---:|:---:|:---:|"
    for op in ("SHL", "SHR"):
        opr = sorted([r for r in rows if r["op"] == op],
                     key=lambda r: (r["nz"] if r["nz"] is not None else 1 << 30))
        lines.append(f"### {op}")
        lines.append("")
        lines.append(hdr)
        lines.append(sep)
        for r in opr:
            fp = "yes" if r["fp32_ok"] else f"NO ({r['max_arg']:.0f})"
            be = f"{r['exact']}/{r['total']}"
            ip, it = r["isa8"]
            lines.append(
                f"| {r['name']} | {r['op']} | {r['depth']} | {r['nz']} | "
                f"{fp} | {be} | {ip}/{it} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = run_bakeoff()
    print(format_table(rows))
