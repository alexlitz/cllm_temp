"""RADIX-256 SRT (ESTIMATE, normalized-domain remainder) 32-bit DIVIDE — bakeoff.

Standalone MEASURE-ONLY.  The VARIABLE-divisor DIV *instruction* (arbitrary
runtime 32-bit divisor) as a base-256 (byte-at-a-time) digit recurrence: 4
iterations vs the radix-16 path's 8.  Divide-by-constant / magic-multiply is
OFF-TARGET and is NOT used.

Algorithm (all nibble-level, fp32-safe by construction)
=======================================================
NORMALIZE d ONCE: ``sh = CLZ(d)`` so ``dn = d << sh`` has its MSB in bit 31
(``dn in [2^31, 2^32)``).  The running remainder is kept in the NORMALIZED domain
``Rn = R << sh`` throughout, so the SRT estimate index ``Ahat = Rn >> 24`` is just
the TOP 4 NIBBLES of Rn — no per-iteration variable shift.  Per byte iteration:

  * bring-down (normalized):  ``Rn = 256*Rn + (byte << sh)``.
  * estimate ``qhat = floor(Ahat / (Bhat+1))`` (``Ahat = Rn>>24`` 16-bit,
    ``Bhat = dn>>24`` = top byte of dn) — the FIXED SRT select, one rule per
    normalized divisor, error ``q_true - qhat in {0,1,2}`` (proven 2M pairs).
  * SRT correction by a 3-LANE PARALLEL select: form ``Rn - qc*dn`` for
    ``qc in {qhat, qhat+1, qhat+2}`` (lane1/2 = lane0 - dn / - 2dn), pick the
    largest non-negative lane, ``Rn <- that``, emit ``qc``.

DE-NORMALIZE the final remainder ONCE (``R = Rn >> sh``) for MOD.

The variable shifts (``d<<sh``, ``byte<<sh``, ``Rn>>sh``) are the honest
CLZ-normalize overhead.  Each is a one-hot(sh)-GATED raw-column placement
(``2^(sh%4) * nib`` into column ``i + sh//4``, raw <= 120 < 256) then a base-16
carry-normalise — fp32-safe (no column >= 256, no ``16^p`` scalar recompose).

fp32 discipline
===============
Estimate staircase forms <= ``255*(Bhat+1) <= 65280`` (``RELU_S*65280 = 1.3e7 <
2^24``).  Subtract borrow uses the ``div_radix16_lean`` 3-nibble-limb pattern
(limb values <= 4095).  ``qhat*dn`` is a nibble schoolbook (columns < 256).

MEASURE-ONLY: builds the real SwiGLU FFN blocks, CPU forward fp64 AND fp32 over
the edge grid + adversarial classes + random 32-bit pairs, reports depth / nz /
byte-exact + the depth BREAKDOWN.  No full-model bake.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .nibble_alu32 import (
    _floor_div_pow, _floor_div_pow2, _guard, _step_ge, _ident, _clear,
    _empty_spec as _base_empty_spec, _truncate, _mul_gate, RELU_S, S,
)
from .nibble_vm_layout import NibbleVMLayout


def _empty_spec(dim, n_units):
    """Padded allocator: reserve generous headroom (the exact unit count of each
    block is hand-estimated; ``_truncate`` trims the spec back to the units actually
    written, so over-allocating is free while under-allocating errors)."""
    return _base_empty_spec(dim, 2 * n_units + 64)

MASK32 = 0xFFFFFFFF
_RN = 10          # normalized remainder nibbles (Rn < 256*dn < 2^40)
_NITERS = 4
_NLANE = 3        # SRT correction lanes: qc in {qhat, qhat+1, qhat+2}
_NIB_KMAX = 15


def _set_one(L):
    from . import nibble_alu32 as m
    m._ONE = L.ONE


class R256Bands:
    def __init__(self, L):
        self.RN = _RN
        RN = _RN
        self.SH_OH = L._band("R256E_SH_OH", 32)     # one-hot(sh), sh = CLZ(d)
        self.HN = L._band("R256E_HN", 9)             # hi_nonzero[c] = [Σ_{c'>=c} d_nib >= 1]
        self.DN = L._band("R256E_DN", RN)           # dn = d<<sh nibbles
        self.DN2 = L._band("R256E_DN2", RN)         # 2*dn nibbles
        self.BP = L._scalar("R256E_BP")             # (dn>>24)+1
        self.BZ = L._scalar("R256E_BZ")             # d==0
        self.R = L._band("R256E_R", RN)             # normalized remainder Rn
        self.SHIN = L._band("R256E_SHIN", RN)       # (byte<<sh) nibbles, shift scratch
        self.SHRAW = L._band("R256E_SHRAW", RN)     # shift raw columns scratch
        self.AHAT = L._scalar("R256E_AHAT")         # Rn>>24 (16-bit)
        self.QHAT = L._scalar("R256E_QHAT")         # estimated quotient byte
        self.QD = L._band("R256E_QD", RN)           # qhat*dn nibbles
        self.QDRAW = L._band("R256E_QDRAW", RN)     # qhat*dn raw columns
        # lane subtrahends and remainders (nibbles).
        self.LSUB = L._band("R256E_LSUB", _NLANE * RN)
        self.LR = L._band("R256E_LR", _NLANE * RN)
        self.LNEG = L._band("R256E_LNEG", _NLANE)   # [lane < 0]
        # 3-nibble limb borrow scratch (per lane): limb diffs and borrows.
        self.LB0 = L._band("R256E_LB0", _NLANE)
        self.LB1 = L._band("R256E_LB1", _NLANE)
        self.LB2 = L._band("R256E_LB2", _NLANE)
        # clean limb floors F16 / F256 per lane per limb (for the residue-snapping split).
        self.LF16 = L._band("R256E_LF16", _NLANE * RN)
        self.LF256 = L._band("R256E_LF256", _NLANE * RN)
        self.QC = L._scalar("R256E_QC")             # chosen quotient byte
        self.ROUT = L._band("R256E_ROUT", 8)        # de-normalized remainder nibbles
        self.DIV_RES = L._band("R256E_DIV", 8)
        self.MOD_RES = L._band("R256E_MOD", 8)
        self.IT = L._scalar("R256E_IT")
        self.IT_OH = L._band("R256E_IT_OH", _NITERS)


def extend_layout(L):
    if getattr(L, "R256E", None) is not None:
        return L.R256E
    L.R256E = R256Bands(L)
    while L._off % L.n_heads != 0:
        L._scalar(f"_r256epad{L._off}")
    L.D = L._off
    return L.R256E


def _new_layout(code_size: int = 8, n_heads: int = 4):
    L = NibbleVMLayout(code_size, n_heads=n_heads)
    extend_layout(L)
    return L


def _nibbles(v, n):
    return [(v >> (4 * j)) & 0xF for j in range(n)]


def _it_oh_units(spec, u, a, it_const=0.0):
    for j in range(_NITERS):
        u = _clear(spec, u, a.IT_OH + j)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j, a.IT_OH + j, 1.0)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j + 1, a.IT_OH + j, -1.0)
    return u


# ===========================================================================
# Shared carry-normalise (columns < 256, kmax=15), a settled column is a fixed pt.
# ===========================================================================
def _carry_round_units(spec, u, src, dst, n):
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)
        u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, -16.0)
        if c + 1 < n:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c + 1, 1.0)
    return u


# ===========================================================================
# 1. NORMALIZE prologue.
# ===========================================================================
def _hinonzero_block(L, dim) -> Dict[str, torch.Tensor]:
    """HN[c] = [Σ_{c'>=c} d_nib >= 1] (any divisor nibble at/above c is nonzero),
    c = 0..8 (HN[8] = 0, no nibble at/above 8)."""
    a = L.R256E
    spec = _empty_spec(dim, 9 * 2)
    u = 0
    for c in range(9):
        u = _clear(spec, u, a.HN + c)
        if c <= 7:
            u = _step_ge(spec, u, {L.AX + cc: 1.0 for cc in range(c, 8)}, 0.0, 1, a.HN + c, 1.0)
    return _truncate(spec, u, dim)


def _clz_block(L, dim) -> Dict[str, torch.Tensor]:
    """One-hot(sh), sh = CLZ(d).  ``sh_oh[j] = [d >= 2^(31-j)] - [d >= 2^(32-j)]``.
    ``[d >= 2^p]`` (p = bit index) is EXACT via ``[16*HN[n+1] + d_nib[n] >= 2^(p%4)]``
    where n = p//4 and HN[c] = [any nibble at/above c nonzero] (a PRIOR block wrote
    HN — avoids the same-block stale read).  fp32-trivial (forms <= 16 + 15 = 31)."""
    a = L.R256E
    spec = _empty_spec(dim, 32 + 32 * 4)
    u = 0
    for j in range(32):
        u = _clear(spec, u, a.SH_OH + j)
    for j in range(32):
        p_hi = 31 - j
        n = p_hi // 4; b = p_hi % 4
        u = _step_ge(spec, u, {a.HN + n + 1: 16.0, L.AX + n: 1.0}, 0.0, 2 ** b, a.SH_OH + j, 1.0)
        p_lo = 32 - j
        if p_lo < 32:
            n2 = p_lo // 4; b2 = p_lo % 4
            u = _step_ge(spec, u, {a.HN + n2 + 1: 16.0, L.AX + n2: 1.0}, 0.0, 2 ** b2, a.SH_OH + j, -1.0)
    return _truncate(spec, u, dim)


def _shift_left_raw_units(spec, u, a, src_terms_per_nib, dst_raw, n_out):
    """Emit gated raw columns for ``y = x << sh``: for shift j = 4*nq + nr,
    ``y_col[i+nq] += 2^nr * x_nib[i]`` gated by sh_oh[j].  ``src_terms_per_nib(i)``
    returns the residual band holding x's nibble i.  Raw column <= 8*15 = 120."""
    for c in range(n_out):
        u = _clear(spec, u, dst_raw + c)
    for j in range(32):
        nq, nr = j // 4, j % 4
        g = (a.SH_OH + j, 1.0, 0.0)
        for i in range(8):            # x has 8 nibbles
            c = i + nq
            if c >= n_out:
                continue
            u = _guard(spec, u, [g], {src_terms_per_nib(i): float(2 ** nr)}, 0.0, dst_raw + c, 1.0)
    return u


def _dn_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    """Raw columns of dn = d<<sh (d = AX nibbles), gated by sh_oh."""
    a = L.R256E
    spec = _empty_spec(dim, a.RN + 32 * 8)
    u = 0
    u = _shift_left_raw_units(spec, u, a, lambda i: L.AX + i, a.DN, a.RN)
    return _truncate(spec, u, dim)


def _dn_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    spec = _empty_spec(dim, a.RN * (2 + 15 * 2 + 15 * 2))
    u = _carry_round_units(spec, 0, a.DN, a.DN, a.RN)
    return _truncate(spec, u, dim)


def _dn2_bp_block(L, dim) -> Dict[str, torch.Tensor]:
    """DN2 = 2*dn (nibble raw = 2*dn_nib, <=30 -> carry in same block via a 2nd pass
    is not possible in one block; keep raw then a carry block).  Also BP = (dn>>24)+1
    = nibble dn[6] + 16*dn[7] + 1 (top byte of dn) — dn is normalized so dn[7]>=8."""
    a = L.R256E
    spec = _empty_spec(dim, a.RN * 2 + 4)
    u = 0
    for c in range(a.RN):
        u = _clear(spec, u, a.DN2 + c)
        u = _ident(spec, u, {a.DN + c: 2.0}, 0.0, a.DN2 + c, 1.0)   # raw 2*dn_nib (<=30)
    u = _clear(spec, u, a.BP)
    u = _ident(spec, u, {a.DN + 6: 1.0, a.DN + 7: 16.0, L.ONE: 1.0}, 0.0, a.BP, 1.0)  # top byte + 1
    return _truncate(spec, u, dim)


def _dn2_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    spec = _empty_spec(dim, a.RN * (2 + 15 * 2 + 15 * 2))
    u = _carry_round_units(spec, 0, a.DN2, a.DN2, a.RN)
    return _truncate(spec, u, dim)


def _bz_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    spec = _empty_spec(dim, 2 + 8)
    u = 0
    u = _clear(spec, u, a.BZ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)
    u = _step_ge(spec, u, {L.AX + c: 1.0 for c in range(8)}, 0.0, 1, a.BZ, -1.0)
    return _truncate(spec, u, dim)


def _init_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, RN + 8 + 1 + _NITERS * 5)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
    for c in range(8):
        u = _clear(spec, u, a.DIV_RES + c)
    u = _clear(spec, u, a.IT)
    u = _it_oh_units(spec, u, a)
    return _truncate(spec, u, dim)


def _normalize_blocks(L, dim):
    return [
        ("r256e-hn", _hinonzero_block(L, dim)),
        ("r256e-clz", _clz_block(L, dim)),
        ("r256e-dnraw", _dn_raw_block(L, dim)),
        ("r256e-dnc0", _dn_carry_block(L, dim)),
        ("r256e-dnc1", _dn_carry_block(L, dim)),
        ("r256e-dn2bp", _dn2_bp_block(L, dim)),
        ("r256e-dn2c", _dn2_carry_block(L, dim)),
        ("r256e-bz", _bz_block(L, dim)),
        ("r256e-init", _init_block(L, dim)),
    ]


# ===========================================================================
# 2. ITERATION BODY.
# ===========================================================================
def _shift_insert_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    """Rn = 256*Rn + (byte<<sh).  Step 1 (this block): shift Rn up 2 nibbles
    (the *256) into raw columns AND add the (byte<<sh) raw columns.  ``byte`` =
    STACK0[2*(3-it)], STACK0[2*(3-it)+1] via one-hot(IT).  ``byte<<sh`` raw =
    ``2^(sh%4) * byte_nib`` placed at column ``bi + sh//4``.  Rn shift: Rn[c-2]
    into raw col c.  Raw column <= 15 (shifted Rn) + 8*15 (shift) < 256."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, RN * 2 + 32 * 2 * _NITERS)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.SHRAW + c)
    # Rn << 8 : Rn[c-2] -> raw col c (drop top 2 nibbles, they overflow past 2^40 and are 0 anyway).
    for c in range(RN - 1, 1, -1):
        u = _ident(spec, u, {a.R + c - 2: 1.0}, 0.0, a.SHRAW + c, 1.0)
    # + (byte << sh) : for iter it, byte nibbles are STACK0[2*bidx], STACK0[2*bidx+1], bidx=3-it.
    for it in range(_NITERS):
        bidx = 3 - it
        git = (a.IT_OH + it, 1.0, 0.0)
        for j in range(32):
            nq, nr = j // 4, j % 4
            gsh = (a.SH_OH + j, 1.0, 0.0)
            for half in range(2):    # byte's two nibbles: low (bitpos 4*half within byte)
                # bit offset of this nibble within the shifted byte = 4*half + j.  nibble goes to
                # column (4*half + j)//4 = half + nq (+ carry from nr).  Value = 2^nr * nibble... but
                # 4*half already nibble-aligned; the nr shift mixes.  Handle via: contribution value =
                # 2^nr * STACK0[2*bidx+half], placed at column half + nq.
                col = half + nq
                if col >= RN:
                    continue
                u = _guard(spec, u, [git, gsh],
                           {L.STACK0 + 2 * bidx + half: float(2 ** nr)}, 0.0,
                           a.SHRAW + col, 1.0)
    return _truncate(spec, u, dim)


def _shift_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    """Carry-normalise the shift raw columns into Rn nibbles."""
    a = L.R256E
    spec = _empty_spec(dim, a.RN * (2 + 15 * 2 + 15 * 2))
    u = _carry_round_units(spec, 0, a.SHRAW, a.R, a.RN)
    return _truncate(spec, u, dim)


def _ahat_block(L, dim) -> Dict[str, torch.Tensor]:
    """AHAT = Rn>>24 = the top 4 nibbles of Rn as ONE clean 16-bit scalar (Rn[6..9]
    with weights 1,16,256,4096).  Materialising AHAT as a SINGLE dim (bound 65535)
    lets the estimate staircase use the form ``AHAT - k*Bp`` (coeff 1 on AHAT)
    instead of the 4096*nib9 recompose — keeping every relu W_up coeff small so the
    fp32 accumulation ``RELU_S*AHAT - RELU_S*k*Bp`` never catastrophically cancels a
    4096-scaled term."""
    a = L.R256E
    spec = _empty_spec(dim, 2)
    u = 0
    u = _clear(spec, u, a.AHAT)
    u = _ident(spec, u, {a.R + 6: 1.0, a.R + 7: 16.0, a.R + 8: 256.0, a.R + 9: 4096.0},
               0.0, a.AHAT, 1.0)
    return _truncate(spec, u, dim)


def _estimate_block(L, dim) -> Dict[str, torch.Tensor]:
    """qhat = floor(AHAT / Bp) = Σ_{k=1..255}[AHAT - k*Bp >= 0].  Reads the clean
    AHAT scalar (prior block) and Bp; the form ``AHAT - k*Bp`` is bounded by
    ``255*256 = 65280`` (``RELU_S*65280 = 1.3e7 < 2^24``) — fp32-exact."""
    a = L.R256E
    spec = _empty_spec(dim, 8 + 255 * 2)
    u = 0
    u = _clear(spec, u, a.QHAT)
    for k in range(1, 256):
        u = _step_ge(spec, u, {a.AHAT: 1.0, a.BP: -float(k)}, 0.0, 0, a.QHAT, 1.0)
    return _truncate(spec, u, dim)


def _qmul_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    """QDRAW = qhat * dn (raw columns).  qhat is a byte (0..255), dn nibbles.  A
    byte*nibble product qhat*dn[c] <= 255*15 = 3825; laid into column c raw.  (One
    multiply per column via _mul_gate: qhat as the byte multiplicand, dn[c] gate.)"""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, RN + RN * 2)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.QDRAW + c)
        u = _mul_gate(spec, u, a.QHAT, a.DN + c, a.QDRAW + c, 1.0)   # qhat * dn[c]
    return _truncate(spec, u, dim)


def _qmul_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    """Carry-normalise qhat*dn columns (each <= 3825 -> kmax 255) into QD nibbles.
    One round; column c carry into c+1.  A single round is NOT enough for cols up to
    3825 (carry ~239 ripples several columns), so we call this twice."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, RN * (2 + 255 * 2 + 255 * 2))
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.QD + c)
        u = _ident(spec, u, {a.QDRAW + c: 1.0}, 0.0, a.QD + c, 1.0)
        u = _floor_div_pow(spec, u, {a.QDRAW + c: 1.0}, 0.0, 16, 255, a.QD + c, -16.0)
        if c + 1 < RN:
            u = _floor_div_pow(spec, u, {a.QDRAW + c: 1.0}, 0.0, 16, 255, a.QD + c + 1, 1.0)
    return _truncate(spec, u, dim)


def _qmul_carry2_block(L, dim) -> Dict[str, torch.Tensor]:
    """Second carry-normalise pass over QD (now < 256 per column) -> clean nibbles."""
    a = L.R256E
    spec = _empty_spec(dim, a.RN * (2 + 15 * 2 + 15 * 2))
    u = _carry_round_units(spec, 0, a.QD, a.QD, a.RN)
    return _truncate(spec, u, dim)


def _lane_sub_block(L, dim) -> Dict[str, torch.Tensor]:
    """Form the 3 lane subtrahends (nibbles): lane0 = QD (=qhat*dn), lane1 = QD+dn,
    lane2 = QD+2dn (raw sums, carry-normalised next block).  qc = qhat + lane."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, _NLANE * RN * 3)
    u = 0
    for lane in range(_NLANE):
        base = a.LSUB + lane * RN
        for c in range(RN):
            u = _clear(spec, u, base + c)
            u = _ident(spec, u, {a.QD + c: 1.0}, 0.0, base + c, 1.0)     # + qhat*dn
            if lane >= 1:
                u = _ident(spec, u, {a.DN + c: float(lane)}, 0.0, base + c, 1.0)  # + lane*dn (raw <=15+2*15=45)
    return _truncate(spec, u, dim)


def _lane_sub_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, _NLANE * RN * (2 + 15 * 2 + 15 * 2))
    u = 0
    for lane in range(_NLANE):
        u = _carry_round_units(spec, u, a.LSUB + lane * RN, a.LSUB + lane * RN, RN)
    return _truncate(spec, u, dim)


# --- 3-nibble-limb borrow of Rn - LSUB[lane], per lane (fp32-safe limb <= 4095) ---
# 10 remainder nibbles -> 4 limbs of 3,3,3,1 nibbles (top limb 1 nibble).  Every limb
# value <= 16^3-1 = 4095, so the split floors have kmax <= 255 (fp32-trivial), the
# same discipline div_radix16_lean proved (875/875 fp32).
_LIMBS = [(0, 3), (3, 3), (6, 3), (9, 1)]


def _limb_val(band_base, start, cnt):
    return {band_base + start + j: float(16 ** j) for j in range(cnt)}


# Borrow-out band for each limb (0,1,2,3); the top limb's borrow-out = LNEG.
_LB_ALL = None   # set lazily from bands in the block builders


def _lb_band(a, li):
    return [a.LB0, a.LB1, a.LB2, a.LNEG][li]


def _lane_borrow_limb_block(L, dim, li) -> Dict[str, torch.Tensor]:
    """ONE limb of the inter-limb borrow, for ALL lanes (parallel bands).  Borrow-out
    ``LB[li][lane] = [ (Rn_limb - sub_limb) - Bin < 0 ]`` where Bin = LB[li-1][lane]
    (0 for li=0).  Because Bin is read from the RESIDUAL (a PRIOR block wrote it), the
    within-block stale-read bug is avoided — each limb is its own block (the lean
    variant's borrow0/borrow1 discipline).  Every limb value <= 4095 (fp32-safe)."""
    a = L.R256E
    RN = a.RN
    st, cnt = _LIMBS[li]
    spec = _empty_spec(dim, _NLANE * 3)
    u = 0
    for lane in range(_NLANE):
        sub = a.LSUB + lane * RN
        f = _limb_val(a.R, st, cnt)
        f = dict(f)
        s = _limb_val(sub, st, cnt)
        for kk, vv in s.items():
            f[kk] = f.get(kk, 0.0) - vv
        if li >= 1:
            f[_lb_band(a, li - 1) + lane] = f.get(_lb_band(a, li - 1) + lane, 0.0) - 1.0
        bout = _lb_band(a, li) + lane
        u = _clear(spec, u, bout)
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, bout, 1.0)
        u = _step_ge(spec, u, f, 0.0, 0, bout, -1.0)       # [limb_diff - Bin < 0]
    return _truncate(spec, u, dim)


def _addk(d, key, coeff):
    r = dict(d)
    r[key] = r.get(key, 0.0) + coeff
    return r


def _lane_limb_V(a, L, lane, li):
    """The (form dict of the) limb result value V = (Rn_limb - sub_limb) - Bin +
    16^cnt*Bout, in [0, 16^cnt - 1].  Bin = LB[li-1][lane] (0 for li=0), Bout =
    LB[li][lane].  Read entirely from RESIDUAL bands (prior blocks)."""
    RN = a.RN
    st, cnt = _LIMBS[li]
    sub = a.LSUB + lane * RN
    V = dict(_limb_val(a.R, st, cnt))
    for kk, vv in _limb_val(sub, st, cnt).items():
        V[kk] = V.get(kk, 0.0) - vv
    if li >= 1:
        V[_lb_band(a, li - 1) + lane] = V.get(_lb_band(a, li - 1) + lane, 0.0) - 1.0
    V[_lb_band(a, li) + lane] = V.get(_lb_band(a, li) + lane, 0.0) + float(16 ** cnt)
    return V, st, cnt


def _lane_floors_block(L, dim) -> Dict[str, torch.Tensor]:
    """Clean floors F16 = floor(V/16), F256 = floor(V/256) per lane per limb, via
    SHARP staircases that SNAP the ``16^j`` recompose residue (the div_radix16_lean
    fp32-hygiene: never carry a sub-integer residue into the next iteration's 256x
    shift).  V <= 4095 -> F16 kmax 255, F256 kmax 15."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, _NLANE * len(_LIMBS) * (2 + 255 * 2 + 15 * 2))
    u = 0
    for lane in range(_NLANE):
        for li, (st, cnt) in enumerate(_LIMBS):
            V, st, cnt = _lane_limb_V(a, L, lane, li)
            f16 = a.LF16 + lane * RN + st
            f256 = a.LF256 + lane * RN + st
            u = _clear(spec, u, f16)
            u = _floor_div_pow(spec, u, V, 0.0, 16, 255, f16, 1.0)       # floor(V/16)
            u = _clear(spec, u, f256)
            u = _floor_div_pow(spec, u, V, 0.0, 256, 15, f256, 1.0)      # floor(V/256)
    return _truncate(spec, u, dim)


def _lane_nibbles_block(L, dim) -> Dict[str, torch.Tensor]:
    """Snapped nibbles LR from V + the clean floors:
        nib0 = floor((V - 16*F16)/1)   nib1 = floor((F16 - 16*F256)/1)   nib2 = F256
    each a sharp kmax=15 staircase (snaps any leftover residue).  For a 1-nibble top
    limb (cnt=1), nib0 = floor(V/1) snapped, kmax 15 (V<16)."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, _NLANE * RN * (2 + 15 * 2))
    u = 0
    for lane in range(_NLANE):
        for li, (st, cnt) in enumerate(_LIMBS):
            V, st, cnt = _lane_limb_V(a, L, lane, li)
            out = a.LR + lane * RN
            f16 = a.LF16 + lane * RN + st
            f256 = a.LF256 + lane * RN + st
            if cnt == 1:
                u = _clear(spec, u, out + st + 0)
                u = _floor_div_pow(spec, u, V, 0.0, 1, 15, out + st + 0, 1.0)     # V snapped
                continue
            # nib0 = V - 16*F16  (sharp)
            u = _clear(spec, u, out + st + 0)
            u = _floor_div_pow(spec, u, _addk(V, f16, -16.0), 0.0, 1, 15, out + st + 0, 1.0)
            # nib1 = F16 - 16*F256  (sharp)
            u = _clear(spec, u, out + st + 1)
            u = _floor_div_pow(spec, u, {f16: 1.0, f256: -16.0}, 0.0, 1, 15, out + st + 1, 1.0)
            if cnt >= 3:
                # nib2 = F256 (re-snapped)
                u = _clear(spec, u, out + st + 2)
                u = _floor_div_pow(spec, u, {f256: 1.0}, 0.0, 1, 15, out + st + 2, 1.0)
    return _truncate(spec, u, dim)


def _select_emit_block(L, dim) -> Dict[str, torch.Tensor]:
    """Pick the largest lane with LNEG=0 (Rn - qc*dn >= 0): lane2 preferred, then
    lane1, then lane0 (which is always valid since qhat <= q_true).  Set Rn = the
    chosen LR; QC = qhat + chosen_lane; emit QC's two nibbles into DIV_RES; IT++.
    Select one-hot: pick_l2 = [LNEG2==0]; pick_l1 = [LNEG1==0]*(1-pick_l2);
    pick_l0 = 1 - pick_l2 - pick_l1."""
    a = L.R256E
    RN = a.RN
    spec = _empty_spec(dim, 8 + RN * 3 + 8 * 3 + _NITERS * 5 + 2 + 8)
    u = 0
    # pick flags (0/1): valid[l] = 1 - LNEG[l].  chosen = highest valid lane.
    #   p2 = valid2 ; p1 = valid1 AND NOT valid2 ; p0 = NOT valid1 AND NOT valid2.
    # (lane0 always valid, so p0 covers the fallthrough.)
    # Use guards on LNEG lanes (0/1).
    g_v2 = (a.LNEG + 2, -1.0, 1.0)      # valid2 = 1 - LNEG2
    g_nv2 = (a.LNEG + 2, 1.0, 0.0)      # NOT valid2 = LNEG2
    g_v1 = (a.LNEG + 1, -1.0, 1.0)
    g_nv1 = (a.LNEG + 1, 1.0, 0.0)
    # Set Rn (10 nibbles) from the selected lane.
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
        u = _guard(spec, u, [g_v2], {a.LR + 2 * RN + c: 1.0}, 0.0, a.R + c, 1.0)                 # lane2
        u = _guard(spec, u, [g_nv2, g_v1], {a.LR + 1 * RN + c: 1.0}, 0.0, a.R + c, 1.0)          # lane1
        u = _guard(spec, u, [g_nv2, g_nv1], {a.LR + 0 * RN + c: 1.0}, 0.0, a.R + c, 1.0)         # lane0
    # Emit the chosen quotient byte QC = QHAT + chosen_lane into DIV_RES[2*(3-it)]
    # (raw byte, split into nibbles by _emit_split_block).  QHAT and the lane flags
    # are all block INPUTS, so the emit is computed directly (no stale QC read):
    #   DIV_RES[lo] += IT_OH[it] * ( QHAT + 2*[lane2] + 1*[lane1] ).
    for it in range(_NITERS):
        bidx = 3 - it
        lo = a.DIV_RES + 2 * bidx
        hi = a.DIV_RES + 2 * bidx + 1
        g = (a.IT_OH + it, 1.0, 0.0)
        u = _guard(spec, u, [g], {lo: -1.0}, 0.0, lo, 1.0)                    # clear lo (gated)
        u = _guard(spec, u, [g], {hi: -1.0}, 0.0, hi, 1.0)                    # clear hi (gated)
        u = _guard(spec, u, [g], {a.QHAT: 1.0}, 0.0, lo, 1.0)                 # + QHAT
        u = _guard(spec, u, [g, g_v2], {L.ONE: 2.0}, 0.0, lo, 1.0)           # + 2 if lane2
        u = _guard(spec, u, [g, g_nv2, g_v1], {L.ONE: 1.0}, 0.0, lo, 1.0)    # + 1 if lane1
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IT, 1.0)
    u = _it_oh_units(spec, u, a, it_const=1.0)
    return _truncate(spec, u, dim)


def _emit_split_block(L, dim) -> Dict[str, torch.Tensor]:
    """Split the raw quotient byte in each DIV_RES low-nibble slot into (byte mod 16,
    floor(byte/16)).  Idempotent on settled 0..15 nibbles."""
    a = L.R256E
    spec = _empty_spec(dim, _NITERS * (15 * 2 + 15 * 2))
    u = 0
    for it in range(_NITERS):
        bidx = 3 - it
        lo = a.DIV_RES + 2 * bidx
        hi = a.DIV_RES + 2 * bidx + 1
        u = _floor_div_pow(spec, u, {lo: 1.0}, 0.0, 16, 15, hi, 1.0)
        u = _floor_div_pow(spec, u, {lo: 1.0}, 0.0, 16, 15, lo, -16.0)
    return _truncate(spec, u, dim)


def _iteration_body(L, dim):
    borrow = [(f"r256e-lb{li}", _lane_borrow_limb_block(L, dim, li))
              for li in range(len(_LIMBS))]
    return [
        ("r256e-shiftraw", _shift_insert_raw_block(L, dim)),
        ("r256e-shiftc", _shift_carry_block(L, dim)),
        ("r256e-ahat", _ahat_block(L, dim)),
        ("r256e-est", _estimate_block(L, dim)),
        ("r256e-qmulraw", _qmul_raw_block(L, dim)),
        ("r256e-qmulc", _qmul_carry_block(L, dim)),
        ("r256e-qmulc2", _qmul_carry2_block(L, dim)),
        ("r256e-lanesub", _lane_sub_block(L, dim)),
        ("r256e-lanesubc", _lane_sub_carry_block(L, dim)),
        ("r256e-lanesubc2", _lane_sub_carry_block(L, dim)),   # 2nd round: carry can ripple 2 cols
    ] + borrow + [
        ("r256e-lanefloors", _lane_floors_block(L, dim)),
        ("r256e-lanenibs", _lane_nibbles_block(L, dim)),
        ("r256e-selemit", _select_emit_block(L, dim)),
        ("r256e-emitsplit", _emit_split_block(L, dim)),
    ]


# ===========================================================================
# 3. REMAINDER + FINALIZE.  MOD = a - q*d  (avoids the normalized-domain de-shift).
#
# ``q*d`` <= a < 2^32 (q = floor(a/d) exactly), so it is an 8-nibble product; the
# low 8 nibbles of the schoolbook are exact.  ``MOD = a - q*d`` is then an 8-nibble
# borrow.  q lives in DIV_RES (8 nibbles), a in STACK0, d in AX.  This reuses the
# fp32-safe nibble schoolbook + limb-borrow discipline and needs NO variable shift.
# ===========================================================================
def _mod_qd_raw_block(L, dim) -> Dict[str, torch.Tensor]:
    """QDRAW = low-32 columns of q*d = Σ_{i+j<8} q_nib[i]*d_nib[j] into column i+j
    (raw <= 8*225 = 1800 < 2^24).  q = DIV_RES nibbles, d = AX nibbles."""
    a = L.R256E
    spec = _empty_spec(dim, 8 + 8 * 8 * 3)
    u = 0
    for c in range(8):
        u = _clear(spec, u, a.QDRAW + c)
    for i in range(8):
        for j in range(8):
            if i + j < 8:
                u = _mul_gate(spec, u, a.DIV_RES + i, L.AX + j, a.QDRAW + i + j, 1.0)
    return _truncate(spec, u, dim)


def _mod_qd_carry_block(L, dim) -> Dict[str, torch.Tensor]:
    """Carry-normalise q*d columns (<= 1800 -> kmax 255) into QD nibbles (8)."""
    a = L.R256E
    spec = _empty_spec(dim, 8 * (2 + 255 * 2 + 255 * 2))
    u = 0
    for c in range(8):
        u = _clear(spec, u, a.QD + c)
        u = _ident(spec, u, {a.QDRAW + c: 1.0}, 0.0, a.QD + c, 1.0)
        u = _floor_div_pow(spec, u, {a.QDRAW + c: 1.0}, 0.0, 16, 255, a.QD + c, -16.0)
        if c + 1 < 8:
            u = _floor_div_pow(spec, u, {a.QDRAW + c: 1.0}, 0.0, 16, 255, a.QD + c + 1, 1.0)
    return _truncate(spec, u, dim)


def _mod_qd_carry2_block(L, dim) -> Dict[str, torch.Tensor]:
    a = L.R256E
    spec = _empty_spec(dim, 8 * (2 + 15 * 2 + 15 * 2))
    u = _carry_round_units(spec, 0, a.QD, a.QD, 8)
    return _truncate(spec, u, dim)


# MOD = a - q*d, an 8-nibble subtract, via 3-nibble-limb borrow (limbs [3,3,2]).
_MOD_LIMBS = [(0, 3), (3, 3), (6, 2)]


def _mod_sub_borrow_block(L, dim, li) -> Dict[str, torch.Tensor]:
    """ONE limb of the MOD borrow (a - q*d): borrow-out ``MB[li] = [(a_limb -
    qd_limb) - Bin < 0]``.  Reuses LB0/LB1/LB2 bands (lane 0)."""
    a = L.R256E
    MB = [a.LB0, a.LB1, a.LB2]
    st, cnt = _MOD_LIMBS[li]
    spec = _empty_spec(dim, 4)
    u = 0
    f = {L.STACK0 + st + m: float(16 ** m) for m in range(cnt)}
    for m in range(cnt):
        f[a.QD + st + m] = f.get(a.QD + st + m, 0.0) - float(16 ** m)
    if li >= 1:
        f[MB[li - 1]] = f.get(MB[li - 1], 0.0) - 1.0
    u = _clear(spec, u, MB[li])
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, MB[li], 1.0)
    u = _step_ge(spec, u, f, 0.0, 0, MB[li], -1.0)
    return _truncate(spec, u, dim)


def _mod_floors_block(L, dim) -> Dict[str, torch.Tensor]:
    """Clean floors of each MOD limb value V = (a_limb - qd_limb) - Bin + 16^cnt*Bout
    (>= 0)."""
    a = L.R256E
    MB = [a.LB0, a.LB1, a.LB2]
    spec = _empty_spec(dim, len(_MOD_LIMBS) * (2 + 255 * 2 + 15 * 2))
    u = 0
    for li, (st, cnt) in enumerate(_MOD_LIMBS):
        V = {L.STACK0 + st + m: float(16 ** m) for m in range(cnt)}
        for m in range(cnt):
            V[a.QD + st + m] = V.get(a.QD + st + m, 0.0) - float(16 ** m)
        if li >= 1:
            V[MB[li - 1]] = V.get(MB[li - 1], 0.0) - 1.0
        V[MB[li]] = V.get(MB[li], 0.0) + float(16 ** cnt)
        f16 = a.LF16 + st
        f256 = a.LF256 + st
        u = _clear(spec, u, f16)
        u = _floor_div_pow(spec, u, V, 0.0, 16, 255, f16, 1.0)
        u = _clear(spec, u, f256)
        u = _floor_div_pow(spec, u, V, 0.0, 256, 15, f256, 1.0)
    return _truncate(spec, u, dim)


def _finalize_block(L, dim) -> Dict[str, torch.Tensor]:
    """MOD_RES nibbles from the clean floors (sharp snapped), honour d==0 -> (0,0).
        nib0 = V - 16*F16 ; nib1 = F16 - 16*F256 ; nib2 = F256  (each sharp kmax 15).
    Then MOD_RES *= (1 - BZ), DIV_RES *= (1 - BZ)."""
    a = L.R256E
    MB = [a.LB0, a.LB1, a.LB2]
    spec = _empty_spec(dim, len(_MOD_LIMBS) * 3 * (2 + 15 * 2) + 8 * 4)
    u = 0
    for li, (st, cnt) in enumerate(_MOD_LIMBS):
        V = {L.STACK0 + st + m: float(16 ** m) for m in range(cnt)}
        for m in range(cnt):
            V[a.QD + st + m] = V.get(a.QD + st + m, 0.0) - float(16 ** m)
        if li >= 1:
            V[MB[li - 1]] = V.get(MB[li - 1], 0.0) - 1.0
        V[MB[li]] = V.get(MB[li], 0.0) + float(16 ** cnt)
        f16 = a.LF16 + st
        f256 = a.LF256 + st
        u = _clear(spec, u, a.MOD_RES + st + 0)
        u = _floor_div_pow(spec, u, _addk(V, f16, -16.0), 0.0, 1, 15, a.MOD_RES + st + 0, 1.0)
        if cnt >= 2:
            u = _clear(spec, u, a.MOD_RES + st + 1)
            u = _floor_div_pow(spec, u, {f16: 1.0, f256: -16.0}, 0.0, 1, 15, a.MOD_RES + st + 1, 1.0)
        if cnt >= 3:
            u = _clear(spec, u, a.MOD_RES + st + 2)
            u = _floor_div_pow(spec, u, {f256: 1.0}, 0.0, 1, 15, a.MOD_RES + st + 2, 1.0)
    return _truncate(spec, u, dim)


def _bz_zero_block(L, dim) -> Dict[str, torch.Tensor]:
    """d == 0 -> (DIV, MOD) = (0, 0): subtract BZ*value (reads the block-input
    MOD_RES/DIV_RES, written by prior blocks — no stale read)."""
    a = L.R256E
    spec = _empty_spec(dim, 8 * 2)
    u = 0
    gz = (a.BZ, 1.0, 0.0)
    for c in range(8):
        u = _guard(spec, u, [gz], {a.MOD_RES + c: -1.0}, 0.0, a.MOD_RES + c, 1.0)
        u = _guard(spec, u, [gz], {a.DIV_RES + c: -1.0}, 0.0, a.DIV_RES + c, 1.0)
    return _truncate(spec, u, dim)


def _epilogue_blocks(L, dim):
    borrow = [(f"r256e-modsub{li}", _mod_sub_borrow_block(L, dim, li))
              for li in range(len(_MOD_LIMBS))]
    return [
        ("r256e-modqdraw", _mod_qd_raw_block(L, dim)),
        ("r256e-modqdc", _mod_qd_carry_block(L, dim)),
        ("r256e-modqdc2", _mod_qd_carry2_block(L, dim)),
    ] + borrow + [
        ("r256e-modfloors", _mod_floors_block(L, dim)),
        ("r256e-finalize", _finalize_block(L, dim)),
        ("r256e-bzzero", _bz_zero_block(L, dim)),
    ]


# ===========================================================================
# 4. Public builders.
# ===========================================================================
def compile_blocks_unrolled(L, dim, n_iters: int = _NITERS):
    _set_one(L)
    blocks = list(_normalize_blocks(L, dim))
    body = _iteration_body(L, dim)
    for it in range(n_iters):
        for name, spec in body:
            blocks.append((f"{name}{it}", spec))
    blocks += _epilogue_blocks(L, dim)
    return blocks


def compile_blocks_recurrent(L, dim, n_iters: int = _NITERS):
    _set_one(L)
    prefix = _normalize_blocks(L, dim)
    body = _iteration_body(L, dim)
    epi = _epilogue_blocks(L, dim)
    unique = prefix + body + epi
    apply_names = [n for n, _ in prefix]
    for _ in range(n_iters):
        apply_names += [n for n, _ in body]
    apply_names += [n for n, _ in epi]
    return unique, apply_names


# ===========================================================================
# 5. CPU forward SIM.
# ===========================================================================
def _apply_block(x, spec):
    up = x @ spec["W_up"].T + spec["b_up"]
    gate = x @ spec["W_gate"].T + spec["b_gate"]
    hidden = F.silu(up) * gate
    return x + hidden @ spec["W_down"].T + spec["b_down"]


def simulate(a_val, b_val, unique=None, apply_names=None, L=None, dim=None,
             dtype=torch.float64):
    if L is None:
        L = _new_layout(); dim = L.D
    if unique is None:
        unique, apply_names = compile_blocks_recurrent(L, dim)
    by_name = {n: s for n, s in unique}
    x = torch.zeros(dim, dtype=dtype)
    x[L.ONE] = 1.0
    for j, nv in enumerate(_nibbles(b_val & MASK32, 8)):
        x[L.AX + j] = float(nv)
    for j, nv in enumerate(_nibbles(a_val & MASK32, 8)):
        x[L.STACK0 + j] = float(nv)
    for name in apply_names:
        spec = {k: v.to(dtype) for k, v in by_name[name].items()}
        x = _apply_block(x, spec)
    a = L.R256E
    q = sum(int(round(float(x[a.DIV_RES + c]))) << (4 * c) for c in range(8))
    r = sum(int(round(float(x[a.MOD_RES + c]))) << (4 * c) for c in range(8))
    return q & MASK32, r & MASK32


def _ref(a, b):
    a &= MASK32; b &= MASK32
    if b == 0:
        return 0, 0
    return a // b, a % b


def _spec_nnz(spec):
    n = 0
    for key in ("W_up", "b_up", "W_gate", "b_gate", "W_down", "b_down"):
        n += int((spec[key] != 0).sum())
    return n


if __name__ == "__main__":
    from .div_radix256_est_measure import measure
    measure()
