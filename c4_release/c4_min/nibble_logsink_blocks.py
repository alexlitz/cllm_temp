"""§653 LOG-SINK division as NEURAL BLOCKS — the shallow divide (~14 blocks) that
replaces the 262-block base-16 recurrent long division.

In-model wiring of ``nibble_logsink_div`` (Python reference verified 56,514/56,514
byte-exact vs //,%).  ``a // b`` / ``a % b`` for the full 32-bit range as a stack of
SwiGLU FFN blocks + ONE dedicated attention head (the softmax1 sink → ``1/b``):

  1. **bm1**       (FFN) ``(b-1)`` nibbles ``d_j`` (closed-form borrow) + the
                   divisor-zero flag ``BZ``.
  2. **log-query** (FFN) sink QUERY lanes ``LOGQ_j = j·log16 + log(d_j)`` (a 16-way
                   log lookup, ``d_j==0 → NEG``) + the query self flag.
  3. **recip-attn**(ATTN) softmax1 sink over 8 PRE-SEEDED reserved-KV log-key rows:
                   the sink weight is ``1/(1+Σ_j 16^j·d_j) = 1/b`` — the log-key rows
                   are seeded at INIT (position-independent), so a leading DIV works.
  4. **newton**    (FFN) one Newton step ``r ← r·(2 − b·r)`` lifts the ~5e-7 sink
                   reciprocal (the residual RoPE phase) to fp64-exact.
  5. **qf**        (FFN) ``QF = a·r`` (the quotient-scale scalar, carried at K_DIV).
  6. **q-scalar**  (FFN) ``QSC = ⌊QF⌋`` and the quotient nibbles ``Q_c``.
  7. **qb**        (FFN×few) ``QB = q·b`` via the schoolbook nibble product +
                   carry-normalise + recombine to a scalar (bounded, ``< 2^34``).
  8. **rem**       (FFN) ``REM = a − QB`` (scalar), ``[REM<0]`` (q too big),
                   ``[REM≥b]`` (q too small).
  9. **correct**   (FFN) ``QSC ± 1`` ; ``REM ± b`` — a single ±1 step (Newton makes
                   ``q0`` within ±1 of the true quotient).
 10. **finalize**  (FFN) decompose corrected ``QSC`` → DIV_RES, ``REM`` → MOD_RES
                   (nibbles the AX mux copies out); ``b==0 → 0`` (ISA_SPEC 4.2).

fp64: the reciprocal (softmax1 precision) and the correction ``q·b`` compare
(``q·b < 2^34`` after Newton — beyond fp32's 2^24) both need doubles, so the div
blocks (and the model that runs them) are fp64.  The quotient-scale scalars (``QF``,
``QSC``, ``QB``, ``REM``) live on a band whose containing blocks use a LARGER RMSNorm
compensator ``K_DIV`` so RMSNorm stays ≈ identity for a value up to 2^34
(``1−r_norm ~ v²/2K_DIV² < 1e-10`` at K_DIV=1e15).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from . import isa
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF

# ------------------------------------------------------------------ constants
NEG = -80.0            # log(d_j==0) sink mask (e^NEG ~ 0 under softmax)
BIG = 200.0            # query self / non-log-key penalty
LOG16 = math.log(16.0)
# RMSNorm compensator K for the whole div_logsink model.  It must dominate the
# quotient-scale scalars (qf, q·b up to ~2^34 = 1.7e10) so the floor is exact
# (1-r < 0.5/1.7e10 -> K > ~2e15).  1e15 is the sweet spot: it also gives the best
# attention-reciprocal precision (a much larger K perturbs the RMSNorm-normalised
# LOGQ query on the sink head).  For the 8-bit operand path (the corpus / isa.interpret
# width) the quotient/remainder are <256 and this is byte-exact through the forward;
# see the report for the 32-bit-through-the-forward precision note.
K_DIV = 1.0e15

_ONE = None            # layout ONE lane, set by the builders


# ==========================================================================
# fp64 SwiGLU unit emitters (weights float64 so log-lookup coefficients keep
# full precision — a float32 log(v)/RELU_S loses ~1e-7 which the reciprocal
# AMPLIFIES ~quotient-fold).
# ==========================================================================
def _empty64(dim: int, n: int) -> Dict[str, torch.Tensor]:
    return {
        "W_up": torch.zeros(n, dim, dtype=torch.float64),
        "b_up": torch.zeros(n, dtype=torch.float64),
        "W_gate": torch.zeros(n, dim, dtype=torch.float64),
        "b_gate": torch.zeros(n, dtype=torch.float64),
        "W_down": torch.zeros(dim, n, dtype=torch.float64),
        "b_down": torch.zeros(dim, dtype=torch.float64),
    }


def _truncate(spec, u, dim):
    u = max(1, u)
    return {
        "W_up": spec["W_up"][:u].contiguous(), "b_up": spec["b_up"][:u].contiguous(),
        "W_gate": spec["W_gate"][:u].contiguous(), "b_gate": spec["b_gate"][:u].contiguous(),
        "W_down": spec["W_down"][:, :u].contiguous(), "b_down": spec["b_down"].contiguous(),
    }


def _grow(spec, u):
    """Ensure the spec's hidden dim covers unit ``u`` (double when exhausted); lets
    the block builders over-shoot their unit estimate without an IndexError."""
    n = spec["W_up"].shape[0]
    if u < n:
        return
    dim = spec["W_up"].shape[1]
    add = max(n, u - n + 1)
    z = lambda r, c: torch.zeros(r, c, dtype=torch.float64)
    spec["W_up"] = torch.cat([spec["W_up"], z(add, dim)], 0)
    spec["b_up"] = torch.cat([spec["b_up"], torch.zeros(add, dtype=torch.float64)], 0)
    spec["W_gate"] = torch.cat([spec["W_gate"], z(add, dim)], 0)
    spec["b_gate"] = torch.cat([spec["b_gate"], torch.zeros(add, dtype=torch.float64)], 0)
    spec["W_down"] = torch.cat([spec["W_down"], z(dim, add)], 1)


def _ident(spec, u, gate_terms, gate_const, dst, scale):
    _grow(spec, u)
    spec["W_up"][u, _ONE] = S
    for band, coeff in gate_terms.items():
        spec["W_gate"][u, band] += coeff
    spec["b_gate"][u] += gate_const
    spec["W_down"][dst, u] += scale / SILU_S
    return u + 1


def _relu(spec, u, terms, const, dst, scale):
    _grow(spec, u)
    for band, coeff in terms.items():
        spec["W_up"][u, band] += RELU_S * coeff
    spec["b_up"][u] += RELU_S * const
    spec["W_gate"][u, _ONE] = 1.0
    spec["W_down"][dst, u] += scale / RELU_S
    return u + 1


def _clear(spec, u, band):
    return _ident(spec, u, {band: 1.0}, 0.0, band, -1.0)


_RAMP_W = 0.25


def _step_ge(spec, u, terms, const, thr, dst, scale):
    """dst += scale * [ (terms·x + const) >= thr ]  (sharp ramp at thr-0.5)."""
    c = thr - 0.5
    w = _RAMP_W
    u = _relu(spec, u, terms, const - (c - w), dst, scale / w)
    u = _relu(spec, u, terms, const - c, dst, -scale / w)
    return u


def _floor_div_pow(spec, u, terms, const, m, kmax, dst, scale):
    """dst += scale * floor(form/m) = scale * sum_{k=1..kmax}[form >= k*m]."""
    for k in range(1, kmax + 1):
        u = _step_ge(spec, u, terms, const, k * m, dst, scale)
    return u


_SHARP_W = 0.02          # steep ramp half-width (for rounding a fractional scalar)


def _step_ge_sharp(spec, u, terms, thr, dst, scale):
    """dst += scale * [ (terms·x) >= thr ] with a STEEP ramp (transition width
    ``_SHARP_W``) centred just below ``thr``.  For rounding a fractional scalar at a
    half-integer ``thr``: a generic fractional value is >> ``_SHARP_W`` away from a
    half-integer, so the output is a CLEAN 0/1."""
    w = _SHARP_W
    u = _relu(spec, u, terms, -(thr - w), dst, scale / w)
    u = _relu(spec, u, terms, -thr, dst, -scale / w)
    return u


def _step_ge_mid(spec, u, terms, center, dst, scale):
    """dst += scale * [ (terms·x) >= center ] with a SHARP ramp (width ``_SHARP_W``)
    centred exactly at ``center`` (a bucket midpoint).  The running rem's integer part
    is a multiple of 16^c plus a sub-bucket value; ``center`` sits half a bucket below
    the boundary, so a rem in the higher bucket is >> center and one in the lower is
    << center — the fractional part cannot flip the result."""
    w = _SHARP_W
    u = _relu(spec, u, terms, -(center - w), dst, scale / w)
    u = _relu(spec, u, terms, -center, dst, -scale / w)
    return u


def _mul_gate(spec, u, a_band, b_band, dst, scale):
    """dst += scale * (A * B), A a byte multiplicand.  6-weight SiLU multiply."""
    _grow(spec, u + 1)
    spec["W_up"][u, a_band] += S
    spec["W_gate"][u, b_band] += 1.0
    spec["W_down"][dst, u] += scale / S
    u += 1
    spec["W_up"][u, a_band] += -S
    spec["W_gate"][u, b_band] += 1.0
    spec["W_down"][dst, u] += scale / S
    u += 1
    return u


def _nibble_carry_round(spec, u, src, dst, n):
    """One base-16 carry-normalise round on n columns kept < 256."""
    for c in range(n):
        u = _clear(spec, u, dst + c)
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)
        u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, 15, dst + c, -16.0)
        if c > 0:
            u = _floor_div_pow(spec, u, {src + c - 1: 1.0}, 0.0, 16, 15, dst + c, 1.0)
    return u


# ==========================================================================
# Scratch layout (attaches onto L, past the ALU32 bands).
# ==========================================================================
class LogSinkBands:
    def __init__(self, L, div_res=None, mod_res=None):
        # If the model already has DIV_RES/MOD_RES result bands (the nibble_alu32
        # scratch the ax-mux reads), WRITE the log-sink quotient/remainder there so
        # the existing opcode-gated ax-mux picks them up unchanged.  Otherwise allocate
        # private result bands (the standalone-test path).
        self._shared_res = div_res is not None
        self.BM1 = L._band("LS_BM1", 8)            # nibbles of (b-1)
        self.BZ = L._scalar("LS_BZ")               # [b == 0]
        self.LOGQ = L._band("LS_LOGQ", 8)          # sink query LOGQ_j
        self.LOGKEY = L._band("LS_LOGKEY", 8)      # 8 reserved log-key row inds
        self.IS_RECIPQ = L._scalar("LS_IS_RECIPQ")  # query self flag (sink penalty)
        self.IS_SINK = L._scalar("LS_IS_SINK")     # the sink row flag (value 1)
        self.IS_RECIP_ROW = L._scalar("LS_IS_RECIP_ROW")  # 1 on log-key + sink rows
        self.IS_RECIP_Q = L._scalar("LS_IS_RECIP_Q")      # 1 on the recip QUERY row
        self.RECIP = L._scalar("LS_RECIP")         # sink weight ~= 1/b
        self.RECIP2 = L._scalar("LS_RECIP2")       # Newton-refined 1/b (fp64-exact)
        self.QF = L._scalar("LS_QF")               # a*RECIP2 (quotient est, ~2^32)
        self.QSC = L._scalar("LS_QSC")             # floor(QF) scalar quotient
        self.Q = L._band("LS_Q", 8)                # quotient nibbles
        self.QB_PP = L._band("LS_QB_PP", 45)       # raw nibble products q_i*b_j
        self.QB_COL = L._band("LS_QB_COL", 9)      # 9 nibble columns of q*b
        self.QB_C1 = L._band("LS_QB_C1", 9)        # carry double-buffer
        self.QB = L._scalar("LS_QB")               # q*b recombined scalar (<2^34)
        self.REM = L._scalar("LS_REM")             # a - q*b (scalar; may be <0)
        self.REM_NEG = L._scalar("LS_REM_NEG")     # [rem < 0]  (q too big)
        self.REM_GEB = L._scalar("LS_REM_GEB")     # [rem >= b] (q too small)
        self.QSC2 = L._scalar("LS_QSC2")           # corrected quotient scalar
        self.REM2 = L._scalar("LS_REM2")           # corrected remainder scalar
        self.QREM = L._scalar("LS_QREM")           # running-rem for the Q decompose
        self.DREM = L._scalar("LS_DREM")           # running-rem for DIV_RES decompose
        self.MREM = L._scalar("LS_MREM")           # running-rem for MOD_RES decompose
        # result nibble bands: reuse the ALU32 ones (ax-mux reads them) if given.
        self.DIV_RES = div_res if div_res is not None else L._band("LS_DIV_RES", 8)
        self.MOD_RES = mod_res if mod_res is not None else L._band("LS_MOD_RES", 8)


def extend_layout_for_logsink(L, div_res=None, mod_res=None):
    if getattr(L, "LOGSINK", None) is not None:
        return L.LOGSINK
    L.LOGSINK = LogSinkBands(L, div_res=div_res, mod_res=mod_res)
    while L._off % L.n_heads != 0:
        L._scalar(f"_lspad{L._off}")
    L.D = L._off
    return L.LOGSINK


# ==========================================================================
# 1. bm1 : (b-1) nibbles + divisor-zero flag.  b lives in the AX nibble band.
#    Closed-form borrow (all reads are of the block INPUT, so single block):
#      borrow_in_j  = [b_0..b_{j-1} all zero]  (=[sum of lower nibbles == 0])
#      borrow_out_j = [b_0..b_j   all zero]
#      d_j = b_j - borrow_in_j + 16*borrow_out_j    (in [0,15])
# ==========================================================================
def compile_bm1(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 8 * 8 + 8)
    u = 0
    for j in range(8):
        dst = a.BM1 + j
        u = _clear(spec, u, dst)
        u = _ident(spec, u, {L.AX + j: 1.0}, 0.0, dst, 1.0)                 # + b_j
        # borrow_in_j = [b_0..b_{j-1} all zero].  For j==0 there are no lower nibbles,
        # so borrow_in_0 = 1 UNCONDITIONALLY (the -1 of b-1 always hits nibble 0).
        lower = {L.AX + k: 1.0 for k in range(j)}                          # sum lower
        if lower:
            u = _ident(spec, u, {L.ONE: 1.0}, 0.0, dst, -1.0)             # -1 (borrow_in)
            u = _step_ge(spec, u, lower, 0.0, 1, dst, 1.0)                 # +[sum_lower>=1]
        else:
            u = _ident(spec, u, {L.ONE: 1.0}, 0.0, dst, -1.0)             # -1 (borrow_in_0=1)
        lower_j = {L.AX + k: 1.0 for k in range(j + 1)}                    # incl b_j
        # borrow_out_j = [b_0..b_j all zero] = 1 - [sum_{0..j} >= 1]
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, dst, 16.0)                  # +16*borrow_out
        u = _step_ge(spec, u, lower_j, 0.0, 1, dst, -16.0)                # -16*[sum>=1]
    # BZ = [b == 0] = 1 - [sum(b nibbles) >= 1]
    u = _clear(spec, u, a.BZ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)
    u = _step_ge(spec, u, {L.AX + c: 1.0 for c in range(8)}, 0.0, 1, a.BZ, -1.0)
    return _truncate(spec, u, dim)


# ==========================================================================
# 2. log-query : the sink QUERY lanes LOGQ_j = j*log16 + log(d_j), d_j==0 -> NEG.
#    16-way log lookup: LOGQ_j = (NEG + j*log16) + sum_{v=1..15}(log v - prev)*[d_j>=v].
#    Plus the query self flag IS_RECIPQ (the sink head penalises the query row).
# ==========================================================================
def compile_log_query(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 8 * (1 + 15 * 2) + 2)
    u = 0
    for j in range(8):
        dst = a.LOGQ + j
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, dst, NEG + j * LOG16)
        prev = NEG
        for v in range(1, 16):
            lv = math.log(v)
            u = _step_ge(spec, u, {a.BM1 + j: 1.0}, 0.0, v, dst, lv - prev)
            prev = lv
    u = _clear(spec, u, a.IS_RECIPQ)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IS_RECIPQ, 1.0)
    return _truncate(spec, u, dim)


# ==========================================================================
# 4. newton : RECIP2 = RECIP*(2 - b*RECIP) = 2*RECIP - RECIP*(b*RECIP).
#    b reconstructed from its 8 AX nibbles (linear form, up to 2^32 in the GATE).
#    BR = b*RECIP ~ 1 (bounded).  All silu-identity multiplies (RECIP, BR >= 0).
# ==========================================================================
def compile_newton_br(L, dim, seed) -> Dict[str, torch.Tensor]:
    """Compute BR = b*seed into the a.QB scratch (does NOT touch the reciprocals).
    ``b`` (LARGE integer) on ``up`` -> silu(S*b)/SILU_S = b exact (b>=1; b==0 -> 0,
    handled by BZ); ``seed`` on the ``gate``."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    b_up = {L.AX + c: float(S * 16 ** c) for c in range(8)}
    spec = _empty64(dim, 2)
    u = 0
    u = _clear(spec, u, a.QB)
    for band, coeff in b_up.items():
        spec["W_up"][u, band] += coeff
    spec["W_gate"][u, seed] += 1.0
    spec["W_down"][a.QB, u] += 1.0 / SILU_S
    u += 1
    return _truncate(spec, u, dim)


def compile_newton_step(L, dim, seed, dst) -> Dict[str, torch.Tensor]:
    """Newton refine: dst = seed*(2 - BR) = 2*seed - seed*BR  (BR = b*seed ~ 1 in the
    a.QB scratch, from ``compile_newton_br``).  All reads are of the block INPUT, so
    seed/BR are the pre-refinement values.  BR (>=~0.5) on ``up``, seed on ``gate``."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 4)
    u = 0
    u = _clear(spec, u, dst)
    u = _ident(spec, u, {seed: 1.0}, 0.0, dst, 2.0)               # 2*seed
    spec["W_up"][u, a.QB] = S                                     # up = S*BR
    spec["W_gate"][u, seed] += 1.0                               # gate = seed
    spec["W_down"][dst, u] += -1.0 / SILU_S                       # - seed*BR
    u += 1
    return _truncate(spec, u, dim)


def compile_newton(L, dim) -> Dict[str, torch.Tensor]:
    """(legacy 2-block Newton, kept for the standalone test path.)  RECIP2 = 2*RECIP
    and BR = b*RECIP into a.QB."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    b_up = {L.AX + c: float(S * 16 ** c) for c in range(8)}
    spec = _empty64(dim, 4)
    u = 0
    u = _clear(spec, u, a.RECIP2)
    u = _ident(spec, u, {a.RECIP: 1.0}, 0.0, a.RECIP2, 2.0)
    u = _clear(spec, u, a.QB)
    for band, coeff in b_up.items():
        spec["W_up"][u, band] += coeff
    spec["W_gate"][u, a.RECIP] += 1.0
    spec["W_down"][a.QB, u] += 1.0 / SILU_S
    u += 1
    return _truncate(spec, u, dim)


def compile_newton2(L, dim) -> Dict[str, torch.Tensor]:
    """(legacy) RECIP2 -= RECIP*BR."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 4)
    u = 0
    spec["W_up"][u, a.QB] = S
    spec["W_gate"][u, a.RECIP] += 1.0
    spec["W_down"][a.RECIP2, u] += -1.0 / SILU_S       # -RECIP*BR
    u += 1
    u = _clear(spec, u, a.QB)                           # clear the scratch
    return _truncate(spec, u, dim)


# ==========================================================================
# 5. qf : QF = a*RECIP2  (a on the ``up`` side, RECIP2 on the gate; a==0 -> QF=0).
#    Runs at K_DIV.
# ==========================================================================
def compile_qf(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    a_up = {L.STACK0 + c: float(S * 16 ** c) for c in range(8)}   # up = S*a_form
    spec = _empty64(dim, 2)
    u = 0
    u = _clear(spec, u, a.QF)
    for band, coeff in a_up.items():
        spec["W_up"][u, band] += coeff
    spec["W_gate"][u, a.RECIP2] += 1.0
    spec["W_down"][a.QF, u] += 1.0 / SILU_S
    u += 1
    return _truncate(spec, u, dim)


# ==========================================================================
# 6. q-decompose : the quotient nibbles Q_c from the QF scalar, MSB-first with a
#    RUNNING REMAINDER (each block extracts nibble c = floor(rem/16^c) with kmax=15
#    — bounded — then subtracts it out; the running scalar rem stays < 16^(c+1)).
#    One block per nibble (8), each reading the running-rem scalar (input) and
#    writing nib_c + the reduced rem.  QF may sit a hair below the exact integer
#    for an exact division, so the first block subtracts 0.5 (the ±1 correction
#    fixes any slip).  Runs at K_DIV (the running rem is a quotient-scale scalar).
#    The generic helper ``_msb_decompose_block`` is reused for QSC2->DIV_RES and
#    REM2->MOD_RES.
# ==========================================================================
def _msb_extract_block(L, dim, rem_band, nib_dst, c, round_low=False) -> Dict:
    """EXTRACT nibble ``c`` into ``nib_dst + c`` (0..15) from the running rem WITHOUT
    reducing rem — the reduction is a SEPARATE block reading the SNAPPED clean nibble,
    so the sharp ramp's ~4e-7 residue never accumulates into the running rem and gets
    amplified by 16^c (the residual-cascade bug).

    ``rem`` was seeded ``QF - 0.5`` so its fractional part sits in [-0.5, 0.5).  For a
    MIDDLE nibble the ``k*16^c - 0.5`` boundary is then clean.  For the LOW nibble
    (``round_low``) rem is ``q_low + frac - 0.5`` and we ROUND (ramp at the half
    integers ``k-0.5``) -> a CLEAN integer within +-1 of q_low (the iterated snap +
    the +-1 correction handle any residue / off-by-one)."""
    global _ONE
    _ONE = L.ONE
    m = 16 ** c
    spec = _empty64(dim, 1 + 16 * 2)
    u = 0
    rem = {rem_band: 1.0}
    u = _clear(spec, u, nib_dst + c)
    if round_low:
        for k in range(1, 17):
            u = _step_ge_sharp(spec, u, rem, k - 0.5, nib_dst + c, 1.0)
    else:
        for k in range(1, 16):
            u = _step_ge_sharp(spec, u, rem, k * m - 0.5, nib_dst + c, 1.0)
    return _truncate(spec, u, dim)


def _msb_reduce_block(L, dim, rem_band, nib_dst, c) -> Dict:
    """REDUCE the running rem by the CLEAN (already snapped-to-integer) nibble:
        rem_band -= nib_dst[c] * 16^c.
    ``nib_dst[c]`` is an exact integer, so this subtraction is exact and the running
    rem stays clean for the next nibble."""
    global _ONE
    _ONE = L.ONE
    m = 16 ** c
    spec = _empty64(dim, 2)
    u = 0
    u = _ident(spec, u, {nib_dst + c: 1.0}, 0.0, rem_band, -float(m))
    return _truncate(spec, u, dim)


def compile_snap_nibbles(L, dim, band, n) -> Dict[str, torch.Tensor]:
    """Re-round each nibble ``band + c`` (c=0..n-1) to a CLEAN integer (the sharp
    round leaves a ~4e-7 residue on the first pass; re-rounding drives it to <1e-12,
    so the schoolbook ``q·b`` — which multiplies by up to 2^32 — does not amplify it
    into a several-hundred error)."""
    global _ONE
    _ONE = L.ONE
    spec = _empty64(dim, n * (1 + 16 * 2))
    u = 0
    for c in range(n):
        u = _clear(spec, u, band + c)
        for k in range(1, 17):
            u = _step_ge_sharp(spec, u, {band + c: 1.0}, k - 0.5, band + c, 1.0)
    return _truncate(spec, u, dim)


def compile_seed_rem(L, dim, src, rem_band, offset=0.0) -> Dict[str, torch.Tensor]:
    """Seed a running-remainder scratch ``rem_band := src + offset`` (a scalar copy
    the MSB-decompose blocks reduce in place without destroying ``src``)."""
    global _ONE
    _ONE = L.ONE
    spec = _empty64(dim, 3)
    u = 0
    u = _clear(spec, u, rem_band)
    u = _ident(spec, u, {src: 1.0}, 0.0, rem_band, 1.0)
    if offset != 0.0:
        u = _ident(spec, u, {L.ONE: 1.0}, 0.0, rem_band, offset)
    return _truncate(spec, u, dim)


# ==========================================================================
# 7. qb : QB = q*b via schoolbook nibble products (bounded) + carry + recombine
#    to a scalar.  Also QSC = sum_c Q_c*16^c (the scalar quotient, for the ±1).
#    products q_i*b_j, i+j<9 (higher are 0 for the ±1-correct q).
# ==========================================================================
_QB_PAIRS = [(i, j) for i in range(8) for j in range(8) if i + j < 9]


def compile_qb_products(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, len(_QB_PAIRS) * 3 + 2)
    u = 0
    for idx, (i, j) in enumerate(_QB_PAIRS):
        u = _clear(spec, u, a.QB_PP + idx)
        u = _mul_gate(spec, u, a.Q + i, L.AX + j, a.QB_PP + idx, 1.0)
    # QSC = sum_c Q_c*16^c (scalar quotient).
    u = _clear(spec, u, a.QSC)
    u = _ident(spec, u, {a.Q + c: float(16 ** c) for c in range(8)}, 0.0, a.QSC, 1.0)
    return _truncate(spec, u, dim)


def compile_qb_split(L, dim) -> Dict[str, torch.Tensor]:
    """Split each product into low/high nibble into columns c=i+j and c+1."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 9 + len(_QB_PAIRS) * (1 + 15 * 2 + 15 * 2))
    u = 0
    for c in range(9):
        u = _clear(spec, u, a.QB_COL + c)
    for idx, (i, j) in enumerate(_QB_PAIRS):
        c = i + j
        pp = a.QB_PP + idx
        u = _ident(spec, u, {pp: 1.0}, 0.0, a.QB_COL + c, 1.0)
        u = _floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, a.QB_COL + c, -16.0)
        if c + 1 < 9:
            u = _floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, a.QB_COL + c + 1, 1.0)
    return _truncate(spec, u, dim)


def compile_qb_carry(L, dim, src, dst) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    spec = _empty64(dim, 9 * (2 + 15 * 2 + 15 * 2 + 2))
    u = _nibble_carry_round(spec, 0, src, dst, 9)
    return _truncate(spec, u, dim)


def compile_qb_recombine(L, dim, src) -> Dict[str, torch.Tensor]:
    """QB = sum_c col_c*16^c (the q*b scalar, < 2^34).  Runs at K_DIV."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 2)
    u = 0
    u = _clear(spec, u, a.QB)
    u = _ident(spec, u, {src + c: float(16 ** c) for c in range(9)}, 0.0, a.QB, 1.0)
    return _truncate(spec, u, dim)


# ==========================================================================
# 8. rem : REM = a - QB (scalar), REM_NEG=[rem<0], REM_GEB=[rem>=b].  Runs at K_DIV.
#    a reconstructed from STACK0 nibbles; b from AX nibbles.  All scalars < 2^34.
# ==========================================================================
def compile_rem(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    a_form = {L.STACK0 + c: float(16 ** c) for c in range(8)}
    b_form = {L.AX + c: float(16 ** c) for c in range(8)}
    spec = _empty64(dim, 8 + 8 + 8)
    u = 0
    # REM = a - QB
    u = _clear(spec, u, a.REM)
    u = _ident(spec, u, a_form, 0.0, a.REM, 1.0)
    u = _ident(spec, u, {a.QB: 1.0}, 0.0, a.REM, -1.0)
    # REM_NEG = [rem < 0] = 1 - [rem >= 0].  rem is an integer (a-q*b), threshold -0.5.
    u = _clear(spec, u, a.REM_NEG)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.REM_NEG, 1.0)
    # [rem >= 0]: use rem form directly; sharp ramp at -0.5.
    rem_form = dict(a_form)
    rem_form[a.QB] = -1.0
    u = _step_ge(spec, u, rem_form, 0.0, 0, a.REM_NEG, -1.0)
    # REM_GEB = [rem >= b] = [rem - b >= 0] (only meaningful when rem>=0).
    u = _clear(spec, u, a.REM_GEB)
    remb = dict(a_form)
    remb[a.QB] = -1.0
    for band, coeff in b_form.items():
        remb[band] = remb.get(band, 0.0) - coeff
    u = _step_ge(spec, u, remb, 0.0, 0, a.REM_GEB, 1.0)
    return _truncate(spec, u, dim)


# ==========================================================================
# 9. correct : QSC2 = QSC + REM_GEB - REM_NEG ; REM2 = REM + b*REM_NEG - b*REM_GEB.
#    Runs at K_DIV.
# ==========================================================================
def compile_correct(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    b_up = {L.AX + c: float(S * 16 ** c) for c in range(8)}    # up = S*b_form
    spec = _empty64(dim, 8 + 12)
    u = 0
    # QSC2 = QSC + REM_GEB - REM_NEG
    u = _clear(spec, u, a.QSC2)
    u = _ident(spec, u, {a.QSC: 1.0}, 0.0, a.QSC2, 1.0)
    u = _ident(spec, u, {a.REM_GEB: 1.0}, 0.0, a.QSC2, 1.0)
    u = _ident(spec, u, {a.REM_NEG: 1.0}, 0.0, a.QSC2, -1.0)
    # REM2 = REM + b*REM_NEG - b*REM_GEB.  b*flag: LARGE b on ``up`` (silu exact),
    # 0/1 flag on the ``gate`` -> hidden = b when flag=1, 0 when flag=0.
    u = _clear(spec, u, a.REM2)
    u = _ident(spec, u, {a.REM: 1.0}, 0.0, a.REM2, 1.0)
    for band, coeff in b_up.items():                          # + b*REM_NEG
        spec["W_up"][u, band] += coeff
    spec["W_gate"][u, a.REM_NEG] += 1.0
    spec["W_down"][a.REM2, u] += 1.0 / SILU_S
    u += 1
    for band, coeff in b_up.items():                          # - b*REM_GEB
        spec["W_up"][u, band] += coeff
    spec["W_gate"][u, a.REM_GEB] += 1.0
    spec["W_down"][a.REM2, u] += -1.0 / SILU_S
    u += 1
    return _truncate(spec, u, dim)


# ==========================================================================
# 10. finalize : zero DIV_RES/MOD_RES on b==0 (ISA_SPEC 4.2).  The nibbles were
#     produced by the MSB-decompose blocks; this gates them off when b==0.  Runs
#     at K_DIV (reads BZ which is small; K choice is irrelevant for the gate).
# ==========================================================================
def compile_finalize(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    spec = _empty64(dim, 8 * 2 + 8 * 2 + 4)
    u = 0
    for c in range(8):
        u = _guard64(spec, u, a.BZ, {a.DIV_RES + c: -1.0}, a.DIV_RES + c, 1.0, L)
        u = _guard64(spec, u, a.BZ, {a.MOD_RES + c: -1.0}, a.MOD_RES + c, 1.0, L)
    return _truncate(spec, u, dim)


def _guard64(spec, u, gate_band, value_terms, dst, scale, L):
    """dst += scale * value * [gate_band == 1] (single 0/1 window)."""
    _grow(spec, u)
    spec["W_up"][u, gate_band] += S
    spec["b_up"][u] += -S * 0.5
    for band, coeff in value_terms.items():
        spec["W_gate"][u, band] += coeff
    spec["W_down"][dst, u] += scale / SILU_HALF
    return u + 1


# ==========================================================================
# The block list assembler (FFN blocks only; the sink-attn head is baked
# separately by qwen_full_vm._bake_logsink_recip_cam on the recip-attn block).
# ==========================================================================
def compile_logsink_blocks(L, dim) -> Tuple[List[Tuple[str, Dict]], List[str]]:
    """Return ``(blocks, kdiv_names)`` — the FFN block (name, spec) list and the
    names of the blocks that must be baked at ``K_DIV`` (they carry a
    quotient-scale scalar)."""
    global _ONE
    _ONE = L.ONE
    a = L.LOGSINK
    blocks: List[Tuple[str, Dict]] = []
    kdiv: List[str] = []

    def add(name, spec, is_kdiv=False):
        blocks.append((name, spec))
        if is_kdiv:
            kdiv.append(name)

    add("ls-bm1", compile_bm1(L, dim))
    add("ls-logq", compile_log_query(L, dim))
    add("ls-recip-attn", _recip_attn_ffn(L, dim))         # FFN no-op; attn baked separately
    # TWO Newton iterations (r <- r*(2 - b*r)): the sink reciprocal has a ~5e-7 residual
    # RoPE phase; one step -> ~2.5e-13, two -> fp64.  Two steps keep the largest-quotient
    # DIV (b small, a~2^32) within the ±1 correction band.  Iter 1: RECIP -> RECIP2.
    # Iter 2: RECIP2 -> RECIP2 (in place; the step block reads RECIP2 + BR2 from input).
    add("ls-newton-br1", compile_newton_br(L, dim, a.RECIP))
    add("ls-newton-st1", compile_newton_step(L, dim, a.RECIP, a.RECIP2))
    add("ls-newton-br2", compile_newton_br(L, dim, a.RECIP2))
    add("ls-newton-st2", compile_newton_step(L, dim, a.RECIP2, a.RECIP2))
    add("ls-qf", compile_qf(L, dim), True)
    # decompose floor(QF - 0.5) -> Q nibbles (MSB-first, running rem in QREM).  The
    # -0.5 seed offset centres the (arbitrary) fractional part of ``QF = a/b`` into
    # [-0.5, 0.5), so every ``rem mod 16^c`` stays strictly below the ``k*16^c - 0.5``
    # bucket-boundary threshold -> every nibble floor is CLEAN (no fractional-part
    # cascade).  floor(QF-0.5) is within ±1 of the true quotient (the ±1 correction
    # fixes the off-by-one), and — crucially — it is a CLEAN INTEGER (the schoolbook
    # q·b needs integer nibbles).
    add("ls-q-seed", compile_seed_rem(L, dim, a.QF, a.QREM, offset=-0.5), True)

    def decompose(prefix, rem_band, nib_band):
        # MSB-first: for each nibble extract -> snap (iterated round -> CLEAN integer)
        # -> reduce the running rem by the CLEAN nibble (so no floor residue
        # accumulates x16^c).  The LOW nibble needs the iterated snap (a round-boundary
        # value ~d+0.5 converges to a clean d/d+1 in a few passes); the middle nibbles
        # are already near-integer so one snap pass suffices.
        for c in range(7, -1, -1):
            add(f"{prefix}-ext{c}", _msb_extract_block(L, dim, rem_band, nib_band, c, round_low=(c == 0)), True)
            npass = 3 if c == 0 else 1
            for s in range(npass):
                add(f"{prefix}-snap{c}_{s}", compile_snap_nibbles(L, dim, nib_band + c, 1), True)
            if c > 0:
                add(f"{prefix}-red{c}", _msb_reduce_block(L, dim, rem_band, nib_band, c), True)

    decompose("ls-q", a.QREM, a.Q)
    # QSC = sum Q_c*16^c ; QB schoolbook q*b.
    add("ls-qb-products", compile_qb_products(L, dim), True)
    add("ls-qb-split", compile_qb_split(L, dim))
    add("ls-qb-carry0", compile_qb_carry(L, dim, a.QB_COL, a.QB_C1))
    add("ls-qb-carry1", compile_qb_carry(L, dim, a.QB_C1, a.QB_COL))
    add("ls-qb-carry2", compile_qb_carry(L, dim, a.QB_COL, a.QB_C1))
    add("ls-qb-recombine", compile_qb_recombine(L, dim, a.QB_C1), True)
    add("ls-rem", compile_rem(L, dim), True)
    add("ls-correct", compile_correct(L, dim), True)
    # decompose QSC2 -> DIV_RES, REM2 -> MOD_RES.  QSC2/REM2 are CLEAN integers after
    # the correction, so each nibble needs only ONE snap pass (extract -> snap -> reduce
    # the running rem by the CLEAN nibble) — the snap between extract and reduce is what
    # keeps the ~4e-7 sharp-floor residue from accumulating x16^c into the running rem.
    def decompose_clean(prefix, rem_band, nib_band):
        for c in range(7, -1, -1):
            add(f"{prefix}-ext{c}", _msb_extract_block(L, dim, rem_band, nib_band, c), True)
            add(f"{prefix}-snap{c}", compile_snap_nibbles(L, dim, nib_band + c, 1), True)
            if c > 0:
                add(f"{prefix}-red{c}", _msb_reduce_block(L, dim, rem_band, nib_band, c), True)

    add("ls-d-seed", compile_seed_rem(L, dim, a.QSC2, a.DREM, offset=-0.5), True)
    decompose_clean("ls-d", a.DREM, a.DIV_RES)
    add("ls-m-seed", compile_seed_rem(L, dim, a.REM2, a.MREM, offset=-0.5), True)
    decompose_clean("ls-m", a.MREM, a.MOD_RES)
    add("ls-finalize", compile_finalize(L, dim), True)
    return blocks, kdiv


def _recip_attn_ffn(L, dim) -> Dict[str, torch.Tensor]:
    """The recip-attn block's FFN is a no-op passthrough (its work is the baked
    sink attention head).  A single self-clearing zero unit keeps the spec valid."""
    global _ONE
    _ONE = L.ONE
    spec = _empty64(dim, 1)
    spec["W_up"][0, L.ONE] = S
    spec["W_gate"][0, L.ONE] = 0.0
    return _truncate(spec, 1, dim)


# ==========================================================================
# The RECIPROCAL SINK attention head (softmax1 1/b) — baked on the recip-attn
# block's self_attn.  8 PRE-SEEDED reserved-KV log-key rows (one per nibble j of
# (b-1)) score ``LOGQ_j`` against a per-nibble query lane; softmax's implicit BOS
# sink weight is then ``1/(1 + Σ_j 16^j·d_j) = 1/b``.  The log-key rows are seeded
# at model/execution INIT (position-independent), so the FIRST executed DIV works.
#
# Head layout (mirrors _bake_code_cam's slow-RoPE-lane keying):
#   * slow lane (half-1-j) : query = SINK_QGAIN·LOGQ_j ; log-key row j key = 1 there.
#     score_j = (q·k)/sqrt(hd) = SINK_QGAIN·LOGQ_j/sqrt(hd).  Pick SINK_QGAIN so this
#     equals LOGQ_j exactly (so softmax sees exp(LOGQ_j)=16^j·d_j).
#   * gate lane : query +BIG on ONE, -BIG on IS_RECIPQ (the fetch/query row) -> the
#     query self-row scores -BIG^2 and never wins the sink (else it is a 2nd sink@0).
#   * value : the SINK row (a dedicated content-free reserved token, IS_SINK=1)
#     carries value 1 on a value lane; o_proj routes it into RECIP -> RECIP = 1/b.
# ==========================================================================
SINK_QGAIN = None    # set per-build = sqrt(head_dim) so score_j == LOGQ_j


def bake_recip_sink_cam(attn, L, arch, head_idx=1):
    """Bake the reciprocal softmax1 sink head onto ``attn`` head ``head_idx``.

    8 reserved log-key rows (``LS_LOGKEY[j]=1``, ``LS_IS_RECIP_ROW=1``) + 1 sink row
    (``LS_IS_SINK=1``, ``LS_IS_RECIP_ROW=1``) are TOKENS the driver seeds after BOS.
    The div-query row (the last position) carries ``LS_LOGQ_j`` (from the ls-logq FFN)
    on the slow rotary lanes.  Scores (post 1/sqrt(hd) scale):
      * log-key row j : LOGQ_j = log(16^j·d_j)            -> exp = 16^j·d_j
      * sink row      : 0                                  -> exp = 1
      * every OTHER row: -P^2 (a huge penalty, IS_RECIP_ROW=0) -> exp ~ 0
    so the SINK weight = 1/(1 + Σ_j 16^j·d_j) = 1/b, copied into ``LS_RECIP``."""
    a = L.LOGSINK
    hd = arch.head_dim
    half = hd // 2
    base = head_idx * hd
    qw = attn.q_proj.weight
    kw = attn.k_proj.weight
    vw = attn.v_proj.weight
    ow = attn.o_proj.weight
    qgain = math.sqrt(hd)                       # so score_j = LOGQ_j exactly
    for j in range(8):
        lane = half - 1 - j                     # slowest rotary lanes (position-inv)
        qw[base + lane, a.LOGQ + j] = qgain     # query LOGQ_j
        kw[lane, a.LOGKEY + j] = 1.0            # log-key row j (one-hot indicator)
    # EXCLUDE every non-recip row: query +P on ONE ; key -P*(1 - IS_RECIP_ROW) =
    # -P*ONE + P*IS_RECIP_ROW.  Score = P*(-P + P*IS_RECIP_ROW) = -P^2 (non-recip) or
    # 0 (log-key / sink row).  Placed on a slow lane (RoPE ~ identity).
    gate_lane = half - 1 - 8
    P = BIG
    qw[base + gate_lane, L.ONE] = P
    kw[gate_lane, L.ONE] = -P
    kw[gate_lane, a.IS_RECIP_ROW] = P
    # value: the sink row carries value 1 (IS_SINK), routed into RECIP.
    vw[0, a.IS_SINK] = 1.0
    ow[a.RECIP, base + 0] = 1.0
