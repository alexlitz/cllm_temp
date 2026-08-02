"""FULL 32-bit ALU folded as PERSISTENT fp32 FFN blocks (BLOG_SPEC-faithful).

Replaces the unified model's **8-bit MUL/DIV/MOD lookup table** (the 256x256 ->
byte table that inflated the dense param count to ~8B) with the spec's genuine
32-bit gadgets, materialised as **persistent SwiGLU FFN weights** applied by
``model.forward`` — no functional per-call Python iteration, no lookup table, no
fp64.

Design
======
Every op operates on the register **nibble bands** (16 nibbles per register = the
low 8 nibbles carry the 32-bit result), so no scalar value ever needs to be exact
above 2^24.  Each op writes its result into its **own** dedicated scratch nibble
band, UNCONDITIONALLY (a fixed circuit — same weights fire every step).  A single
final **opcode-gated multiplexer** copies the selected op's 8 result nibbles into
the AX nibble bands (``AX_nib[j] += OP_IS[op] * RES[op][j]``), so the blocks never
interfere with each other or with non-ALU ops.

  * **ADD / SUB** (``§Addition``) — per-byte carry chain.  Each of the 4 result
    bytes is ``A_i + B_i (+carry_in)``; result byte = ``mod 256``, carry-out =
    ``(sum >= 256)``, threaded to byte i+1.  SUB = ``A + (~B + 1)`` through the
    same chain.  Every hidden value is a byte sum (<= 511) -> fp32-exact for the
    full 32-bit range.

  * **MUL** (``§Multiplication``) — byte schoolbook.  The 10 partial products
    ``a_i * b_j`` with ``i + j < 4`` (each <= 65025) accumulate into 4 byte
    columns; 3 carry rounds normalise.  Each partial product is one 6-weight
    SiLU-gated multiply.  Column accumulators <= ~260100 < 2^24 -> fp32-exact.

  * **DIV / MOD** (``§Division``) — base-16 long division, 8 nibble iterations
    (MSB first).  Per iteration ``r = 16 r + nib``, quotient digit
    ``q = sum_{k=1..15}[r >= k*b]``, ``r -= q*b``.  The threshold ``k*b`` reaches
    ~15*2^32 ~ 6e10 where a naive fp32 magnitude compare breaks; we keep it
    **fp32-EXACT** by doing every ``r >= k*b`` as a **byte-wise (multi-precision)
    comparison** — r and k*b compared byte-by-byte from the MSB, so no compared
    quantity exceeds 255.  ``b == 0 -> 0`` (ISA_SPEC 4.2).

fp32 discipline: no hidden unit ever holds a value >= 2^24.  ADD/SUB byte sums
<= 511; MUL column accumulators <= 260100; DIV/MOD byte-wise compare keeps every
compared quantity <= 255 and the remainder/quotient live as bytes/nibbles.

The public builders return ``(name, spec)`` FFN sub-block lists (SwiGLU tensor
dicts, the ``nibble_vm.compile_ffn`` container) that the unified model wires like
any other op block.  ``extend_layout_for_alu32`` allocates the scratch bands.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch

from . import isa
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF, _empty_spec

# silu at the AND-gate operating point silu(0.5*S) ~= 30.
_SILU_G = SILU_HALF
# the layout's ONE lane, set by the builders before emitting (keeps emitters terse).
_ONE = None


# ---------------------------------------------------------------------------
# C4_MUL_LOOKAHEAD — the general MUL carry resolve.
#
# DEFAULT ON: ``compile_mul_blocks`` resolves the 8 post-split carry-save columns
# with a base-16 Kogge-Stone parallel-prefix carry (round-1 | G/P | 3 prefix
# combine | fused apply) — ``ceil(log2 8) = 3`` prefix stages + a fused result
# write = 6 resolve blocks, vs the 7 SERIAL ripple rounds + a separate result copy
# = 8 blocks the ripple used.  Byte-identical result (both compute the exact
# carry-propagate over the same split columns).
#
# OFF (``C4_MUL_LOOKAHEAD=0``): the historical 7 serial ripple rounds + result
# copy — byte-identical to the pre-lookahead build (SAME weights, SAME ``L.D``: the
# resolve's private scratch bands are only allocated when the flag is ON).
#
# SCOPE: this flag governs ONLY ``compile_mul_blocks``' own resolve.  The DIV/MOD
# path (``_kb_precompute_blocks`` / the QB normalise) calls ``_carry_round_block``
# DIRECTLY and is UNAFFECTED — it always ripples (its 9-column QB is already only 6
# rounds, a wash for prefix; see mul_lookahead.div_knockon).
# ---------------------------------------------------------------------------
def mul_lookahead() -> bool:
    """True iff the general-MUL carry resolve uses the Kogge-Stone parallel prefix
    (``C4_MUL_LOOKAHEAD != '0'``, default ON).  OFF -> the 7-ripple path, which is
    byte-identical to the pre-lookahead build (same weights, same ``L.D``)."""
    return os.environ.get("C4_MUL_LOOKAHEAD", "1") != "0"


def _signed_divmod_enabled() -> bool:
    """``C4_DIVMOD_SIGNED`` (#790, DEFAULT OFF): compute DIV/MOD with native-c4
    SIGNED truncation-toward-zero semantics instead of the golden UNSIGNED floor.
    The base-16 long divider is UNSIGNED, so signed div/mod is done by NEGATING each
    negative operand to its magnitude BEFORE the divide (recording its sign) and
    conditionally negating the quotient (sign = SGN_A^SGN_B) / remainder (sign =
    SGN_A) AFTER — all via a two's-complement carry chain over the operand / result
    nibble bands.  OFF -> DIV/MOD stay unsigned-floor -> golden byte-IDENTICAL."""
    return os.environ.get("C4_DIVMOD_SIGNED", "0") not in ("0", "", "false", "False")


# ---------------------------------------------------------------------------
# C4_KB_BATCHED — the divide's KB-precompute (``KB[k]=k*b`` normalise) resolve.
#
# DEFAULT ON: ``_kb_precompute_blocks`` resolves all 15 ``KB[k]`` with ONE batched
# base-16 Kogge-Stone parallel prefix (raw | round-1 | G/P | ceil(log2 9)=4 KS
# stages | fused apply | BZ = 9 blocks), vs the 15 x _KB_CARRY_ROUNDS serial
# ripples (~92 blocks) the historical path used.  Byte-identical result (both
# compute the exact carry-propagate over the same raw columns); validated
# byte-exact for all 15 KB[k] over the edge set + random b<2^32 incl 0xFFFFFFFF.
# Benefits EVERY _kb_precompute_blocks user — the long-division DIV/MOD fallback
# AND the radix-16 KB threshold table share this prologue.
#
# OFF (``C4_KB_BATCHED=0``): the historical 15x6 serial ripple — byte-identical to
# the pre-lookahead build (SAME weights, SAME ``L.D``: the batched prefix scratch
# bands are only allocated when the flag is ON, allocated LAST).
# ---------------------------------------------------------------------------
def kb_batched() -> bool:
    """True iff the KB-precompute uses the batched Kogge-Stone parallel prefix
    (``C4_KB_BATCHED != '0'``, default ON).  OFF -> the 15x6 serial ripple, which
    is byte-identical to the pre-batched build (same weights, same ``L.D``)."""
    return os.environ.get("C4_KB_BATCHED", "1") != "0"


# ---------------------------------------------------------------------------
# C4_RECURRENT_CORE — NESTED weight-tie inside the recurrent divmod body.
#
# DEFAULT OFF.  The recurrent divmod body already stores ONE iteration body and
# applies it 8x (the OUTER recurrence, ``compile_divmod_blocks_recurrent``).
# Within that ONE body the 6 QB carry-normalise rounds (``alu-div-qbc-*``) are
# BYTE-IDENTICAL blocks (each is the same position-independent ``_carry_round_block(
# L, dim, QB, QB, RN)`` — src==dst==QB, so applying it repeatedly just re-runs the
# same in-place carry-propagate, exactly the intent of an N-round ripple).  This
# flag ties those 6 into ONE stored block applied 6x (an INNER recurrence), so the
# recurrent build stores 6-1=5 FEWER physical blocks with ZERO change to the applied
# sequence -> byte-EXACT.  Kept OFF by default so the production/golden build is
# unchanged; the census showed this is the ONLY remaining byte-identical tie in the
# 95-physical-block recurrent build (all 63 non-divmod blocks are byte-DISTINCT).
# ---------------------------------------------------------------------------
def recurrent_core() -> bool:
    """True iff the recurrent divmod body ties its 6 byte-identical QB carry-round
    blocks into ONE stored block applied 6x (``C4_RECURRENT_CORE != '0'``, default
    OFF).  Byte-EXACT: the tied block is position-independent (src==dst==QB), so
    the applied sequence is unchanged; only the STORED physical-block set shrinks."""
    return os.environ.get("C4_RECURRENT_CORE", "0") != "0"


# ===========================================================================
# Low-level SwiGLU unit emitters — all fp32-exact for integer arguments.
# A hidden unit computes ``silu(up . x + b_up) * (gate . x + b_gate)`` and its
# W_down column routes that product (scaled) into a destination band.  Three
# building blocks compose the whole ALU:
# ===========================================================================
def _ident(spec, u, gate_terms, gate_const, dst, scale):
    """dst += scale * (gate_terms . x + gate_const).  up = S*ONE so silu(up)=S;
    hidden = S * form; W_down routes form*scale into dst."""
    spec["W_up"][u, _ONE] = S
    for band, coeff in gate_terms.items():
        spec["W_gate"][u, band] += coeff
    spec["b_gate"][u] += gate_const
    spec["W_down"][dst, u] += scale / SILU_S
    return u + 1


def _relu(spec, u, terms, const, dst, scale):
    """dst += scale * relu(terms . x + const).  up = RELU_S*form (silu-relu),
    gate = ONE; hidden/RELU_S = relu(form)."""
    for band, coeff in terms.items():
        spec["W_up"][u, band] += RELU_S * coeff
    spec["b_up"][u] += RELU_S * const
    spec["W_gate"][u, _ONE] = 1.0
    spec["W_down"][dst, u] += scale / RELU_S
    return u + 1


def _clear(spec, u, band):
    """SET semantics: dst -= old(dst)."""
    return _ident(spec, u, {band: 1.0}, 0.0, band, -1.0)


_RAMP_W = 0.25          # sharp-ramp half-width; the 0->1 transition at thr-0.5.


def _step_ge(spec, u, terms, const, thr, dst, scale):
    """dst += scale * [ (terms.x + const) >= thr ]  for an integer form.

    Uses a SHARP unit ramp centred at ``thr - 0.5`` (a half-integer, safely
    between achievable integer forms): ``(relu(form-(c-w)) - relu(form-c))/w`` with
    ``c = thr-0.5`` and small ``w``.  Because achievable forms are near-integers
    and the ramp flips at a half-integer, the result is cleanly 0 or 1 and immune
    to the ~0.1 fp residue that accumulates across carry rounds (a naive unit-wide
    ramp anchored at integer thresholds emits a *fractional* value when the form
    lands a noisy 16k, which then poisons downstream carries)."""
    c = thr - 0.5
    w = _RAMP_W
    u = _relu(spec, u, terms, const - (c - w), dst, scale / w)
    u = _relu(spec, u, terms, const - c, dst, -scale / w)
    return u


def _floor_div_pow(spec, u, terms, const, m, kmax, dst, scale):
    """dst += scale * floor(form / m) = scale * sum_{k=1..kmax} [form >= k*m].

    Byte-bounded ``m`` (16 or 256) and small ``kmax``: ``kmax`` MUST be tight —
    ``kmax >= floor(max_form / m)`` — because each threshold is one SwiGLU hidden
    unit and the model's uniform hidden width is the widest block.  A carry column
    kept ``< 16*m`` needs only ``kmax = 15`` (the whole reason the ALU keeps every
    accumulator ``< 256`` and carries a single nibble per round)."""
    for k in range(1, kmax + 1):
        u = _step_ge(spec, u, terms, const, k * m, dst, scale)
    return u


def _floor_div_pow2(spec, u, terms, const, m, kmax, dst_a, scale_a, dst_b, scale_b):
    """Emit ``floor(form/m)`` ONCE (the shared relu staircase) and route it to TWO
    destinations: ``dst_a += scale_a*floor`` and ``dst_b += scale_b*floor``.  Used
    to split a product ``p`` into ``p mod m`` (``-m*floor`` into column c) and
    ``floor(p/m)`` (``+floor`` into column c+1) with a single staircase — half the
    units of two separate ``_floor_div_pow`` calls."""
    for k in range(1, kmax + 1):
        # one _step_ge = two relu units routed to BOTH dsts.
        c = k * m - 0.5
        w = _RAMP_W
        u = _relu2(spec, u, terms, const - (c - w), dst_a, scale_a / w, dst_b, scale_b / w)
        u = _relu2(spec, u, terms, const - c, dst_a, -scale_a / w, dst_b, -scale_b / w)
    return u


def _relu2(spec, u, terms, const, dst_a, scale_a, dst_b, scale_b):
    """One relu unit routed to two destinations (shared up/gate, two W_down)."""
    for band, coeff in terms.items():
        spec["W_up"][u, band] += RELU_S * coeff
    spec["b_up"][u] += RELU_S * const
    spec["W_gate"][u, _ONE] = 1.0
    spec["W_down"][dst_a, u] += scale_a / RELU_S
    spec["W_down"][dst_b, u] += scale_b / RELU_S
    return u + 1


# floor(x/16) for a column kept < 256 (single-nibble carry-out 0..15): kmax=15.
_NIB_KMAX = 15
# carry-round counts to FULLY settle a column stack whose raw entries are < 256.
# A carry ripples one column per round, so the count is bounded by the ripple
# length, NOT the width RN.  Verified worst-cases (random 300k, §carry-settle sim):
# KB/QB raw = k*b_nib <= 225 settle in <= 5 rounds; MUL split partials <= 6.
# We add +1 headroom (settled columns are idempotent under further rounds).
_KB_CARRY_ROUNDS = 6
_QB_CARRY_ROUNDS = 6
_MUL_CARRY_ROUNDS = 7
# NB: an already-settled column (all nibbles < 16) is a FIXED POINT of a carry
# round, so extra rounds are harmless — the +1 headroom never corrupts a result.


def _nibble_carry_round(spec, u, src, dst, n):
    """One base-16 carry-normalise round on ``n`` columns kept ``< 256``:
        dst[c] = src[c] mod 16 + floor(src[c-1] / 16)     (carry-in from below)
    Every column stays ``< 256`` so ``floor(./16) <= 15`` (kmax=15).  When
    ``src == dst`` (in place) the block reads the pre-round residual — safe because
    all units read the block INPUT and write deltas.  SET semantics.

    CARRY-SHARING: each column's ``floor(src[c]/16)`` is the SAME value the baseline
    computed twice (once as its own ``-16*floor`` mod, once as the ``+floor`` carry
    into column c+1).  We emit it ONCE via ``_floor_div_pow2`` and route it to BOTH
    destinations — half the staircase units per column, byte-identical result
    (the top column's carry-out overflows past the kept nibbles and is dropped,
    exactly as the baseline dropped it).  The +1 carry into ``dst[c+1]`` is a delta
    on the block INPUT, so it survives that column's later ``_clear`` (which only
    subtracts the input value)."""
    for c in range(n):
        u = _clear(spec, u, dst + c)                                                # SET -old
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)                       # + col
        if c + 1 < n:
            # ONE floor(col/16) staircase -> -16 into col c (mod), +1 into c+1 (carry).
            u = _floor_div_pow2(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX,
                                dst + c, -16.0, dst + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, -16.0)  # top: mod only
    return u


def _carry_round_block(L, dim, src, dst, n) -> Dict[str, torch.Tensor]:
    """A standalone FFN block wrapping ONE ``_nibble_carry_round`` (columns < 256,
    kmax=15).  Used by MUL, KB and QB carry stacks; every such block is tiny."""
    spec = _empty_spec(dim, n * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = _nibble_carry_round(spec, 0, src, dst, n)
    return _truncate(spec, u, dim)


def _mul_gate(spec, u, a_band, b_band, dst, scale):
    """dst += scale * (A * B) via the 6-weight SiLU-gated multiply
    ``(silu(S*A)+silu(-S*A))*B/S`` (§Basic Arithmetic).  Two hidden units share
    the gate B; exact for a byte multiplicand A (0..255)."""
    # unit 1: silu(S*A) * B
    spec["W_up"][u, a_band] += S
    spec["W_gate"][u, b_band] += 1.0
    spec["W_down"][dst, u] += scale / S
    u += 1
    # unit 2: silu(-S*A) * B  (~0 for A>=0, but keeps the primitive exact/general)
    spec["W_up"][u, a_band] += -S
    spec["W_gate"][u, b_band] += 1.0
    spec["W_down"][dst, u] += scale / S
    u += 1
    return u


def _guard(spec, u, windows, value_terms, value_const, dst, scale):
    """dst += scale * value * AND(windows).  Each window (band, coeff, const) is a
    0/1 indicator; the AND fires iff all are 1.  up = S*(sum windows - (n-0.5));
    gate = value; when all windows=1 up=0.5S -> silu=0.5S -> /SILU_HALF => value,
    else up<=-0.5S -> silu~=0.  ``value`` must be byte-bounded (the gate path)."""
    n = len(windows)
    for band, coeff, const in windows:
        spec["W_up"][u, band] += S * coeff
        spec["b_up"][u] += S * const
    spec["b_up"][u] += -S * (n - 0.5)
    for band, coeff in value_terms.items():
        spec["W_gate"][u, band] += coeff
    spec["b_gate"][u] += value_const
    spec["W_down"][dst, u] += scale / _SILU_G
    return u + 1


def _truncate(spec, u, dim):
    u = max(1, u)
    return {
        "W_up": spec["W_up"][:u].contiguous(), "b_up": spec["b_up"][:u].contiguous(),
        "W_gate": spec["W_gate"][:u].contiguous(), "b_gate": spec["b_gate"][:u].contiguous(),
        "W_down": spec["W_down"][:, :u].contiguous(), "b_down": spec["b_down"].contiguous(),
    }


# ===========================================================================
# Scratch layout for the 32-bit ALU (appended to the unified layout).
# ===========================================================================
class ALU32Bands:
    """Allocate the scratch bands the 32-bit ALU blocks use (attaches onto L)."""

    def __init__(self, L, recurrent_divmod: bool = False,
                 shift_via_mul: bool = False, mul_lookahead: bool = False,
                 kb_batched: bool = False):
        self.A = L._band("ALU_A", 4)          # operand-A bytes (STACK0 = popped)
        self.B = L._band("ALU_B", 4)          # operand-B bytes (AX = accumulator)
        self.NOTB = L._band("ALU_NOTB", 4)    # ~B bytes (for SUB two's complement)
        # ADD / SUB
        self.ADD_C = L._band("ALU_ADD_C", 5)  # carry chain (C[0]=cin .. C[4]=drop)
        self.SUB_C = L._band("ALU_SUB_C", 5)
        self.ADD_RES = L._band("ALU_ADD_RES", 8)   # 8 result nibbles
        self.SUB_RES = L._band("ALU_SUB_RES", 8)
        # MUL: nibble-schoolbook column accumulators (8 nibble columns, kept small
        # so every relu argument stays fp32-exact), double-buffered, result nibbles
        self.PP = L._band("ALU_PP", 36)       # the 36 raw nibble products a_i*b_j (i+j<8)
        self.MCOL = L._band("ALU_MCOL", 8)    # 8 nibble columns (col c = weight 16^c)
        self.MC1 = L._band("ALU_MC1", 8)      # double buffer
        self.MUL_RES = L._band("ALU_MUL_RES", 8)
        # DIV / MOD: base-16 long division on NIBBLES (all quantities 0..15 so
        # every relu argument stays byte-bounded -> fp32-EXACT even though k*b
        # reaches ~6e10, the value that forced the standalone gadget onto fp64).
        _RN = 9                                # remainder nibbles (r < 16*b < 2^36)
        self.RN = _RN
        self.R = L._band("ALU_R", _RN)         # running remainder nibbles (LSB first)
        self.R2 = L._band("ALU_R2", _RN)       # double buffer (post-subtract)
        # k*b nibbles for k=1..15, precomputed once from B: KB + _RN*k + nibble.
        self.KB = L._band("ALU_KB", _RN * 16)
        # per-nibble compare lanes for R vs KB[k]: GT[k,i], EQ_N[k,i], k=1..15.
        self.GT = L._band("ALU_GT", 15 * _RN)
        self.EQ_N = L._band("ALU_EQ_N", 15 * _RN)
        self.QD = L._scalar("ALU_QD")          # current quotient digit (0..15)
        self.QB = L._band("ALU_QB", _RN)       # q*b nibbles (for the subtract)
        self.SUBB = L._band("ALU_SUBB", _RN + 1)   # subtract borrow chain
        self.DIV_RES = L._band("ALU_DIV_RES", 8)   # quotient nibbles (result)
        self.MOD_RES = L._band("ALU_MOD_RES", 8)   # remainder nibbles (result)
        self.BZ = L._scalar("ALU_BZ")          # divisor == 0 predicate
        # RECURRENT DIV/MOD scratch (only allocated when the recurrent path is
        # active, so the unrolled build's dim is byte-identically unchanged): the
        # digit-index counter IT (0..7, threaded across the 8 reused iteration
        # applications like PC/AX) and its 8-lane one-hot IT_OH (used by the
        # index-independent shift-gather and qcopy-scatter).
        self.recurrent_divmod = recurrent_divmod
        if recurrent_divmod:
            self.IT = L._scalar("ALU_IT")          # current iteration index 0..7
            self.IT_OH = L._band("ALU_IT_OH", 8)   # one-hot(IT)
        # SHIFT-VIA-MUL/DIV (retire the barrel shifter): a shift ``x</>> n`` is a
        # multiply / floor-divide by ``2**n`` (§Shifts).  ``SH_N_OH`` is the EXACT-count
        # one-hot of the shift amount ``n`` over s=0..31; the pow2-route block reads it
        # to write ``2**n`` (or 0 when n>=32, no cell fires) into the AX nibble band, so
        # the NATIVE MUL (SHL) / DIV (SHR) gadgets — already applied every forward —
        # compute the shift with NO barrel-shifter weights.  Only allocated when
        # shift_via_mul is on.
        self.shift_via_mul = shift_via_mul
        if shift_via_mul:
            self.SH_N_OH = L._band("ALU_SH_N_OH", 32)   # one-hot(shift count n), s=0..31
        # MUL LOOKAHEAD (Kogge-Stone) — resolve scratch, allocated LAST and ONLY when
        # the flag is on so a flag-OFF build's ``L.D`` (and every earlier band offset)
        # is byte-identical to the pre-lookahead layout.  The resolve reduces the 8
        # post-split columns (< 256) to ``t_c`` in [0,30] (T), computes the binary
        # generate/propagate lanes (G0/P0), Kogge-Stone-combines them across the 3
        # prefix stages in DOUBLE-BUFFERED lanes (G0/P0 <-> G1/P1), and the fused
        # apply writes MUL_RES directly.  See ``_mul_lookahead_resolve_blocks``.
        self.mul_lookahead = mul_lookahead
        if mul_lookahead:
            self.MUL_T = L._band("ALU_MUL_T", _MUL_NCOL)    # t_c in [0,30] (post round-1)
            self.MUL_G0 = L._band("ALU_MUL_G0", _MUL_NCOL)  # generate lane, buffer 0
            self.MUL_P0 = L._band("ALU_MUL_P0", _MUL_NCOL)  # propagate lane, buffer 0
            self.MUL_G1 = L._band("ALU_MUL_G1", _MUL_NCOL)  # generate lane, buffer 1
            self.MUL_P1 = L._band("ALU_MUL_P1", _MUL_NCOL)  # propagate lane, buffer 1
        # KB-PRECOMPUTE LOOKAHEAD (batched base-16 Kogge-Stone) — the divide's
        # ``KB[k]=k*b`` normalise resolves all 15 KB[k] in ONE parallel-prefix pass
        # (raw|round1|G/P|4 KS stages|apply|BZ = 9 blocks) instead of 15 serial
        # 6-round ripples (~92 blocks).  Scratch = t/G0/P0/G1/P1, 15 KB[k] x _RN
        # columns side by side.  Allocated LAST + only when the flag is ON so the
        # ripple path (C4_KB_BATCHED=0) is byte-identical to the pre-lookahead
        # layout (same L.D, same band offsets).  15 = k in 1..15.
        self.kb_batched = kb_batched
        if kb_batched:
            _NK = 15
            self.KB_T = L._band("ALU_KB_T", _NK * _RN)      # t_c in [0,30] (post round-1)
            self.KB_G0 = L._band("ALU_KB_G0", _NK * _RN)    # generate lane, buffer 0
            self.KB_P0 = L._band("ALU_KB_P0", _NK * _RN)    # propagate lane, buffer 0
            self.KB_G1 = L._band("ALU_KB_G1", _NK * _RN)    # generate lane, buffer 1
            self.KB_P1 = L._band("ALU_KB_P1", _NK * _RN)    # propagate lane, buffer 1
        # SIGNED DIV/MOD (C4_DIVMOD_SIGNED, #790) — the sign flags + the conditional
        # two's-complement negate scratch.  Allocated LAST + only when the flag is on
        # so the UNSIGNED (golden) build's L.D + every band offset are byte-identical.
        self.signed_divmod = _signed_divmod_enabled()
        if self.signed_divmod:
            self.SGN_A = L._scalar("ALU_SGN_A")     # 1 iff dividend (STACK0) < 0 (bit 31)
            self.SGN_B = L._scalar("ALU_SGN_B")     # 1 iff divisor  (AX)     < 0 (bit 31)
            self.RES_SGN = L._scalar("ALU_RES_SGN")  # quotient sign = SGN_A ^ SGN_B
            # ZCUM[j] = [ Σ_{k<j} nib_k == 0 ] (the two's-complement carry-in), 9 slots
            # (j=0..8) for whichever register is being negated this block (STACK0/AX/
            # DIV_RES/MOD_RES — one negate per block, so ONE shared scratch suffices).
            self.ZCUM = L._band("ALU_ZCUM", 9)


def extend_layout_for_alu32(L, recurrent_divmod: bool = False,
                            shift_via_mul: bool = False,
                            mul_lookahead: "bool | None" = None,
                            kb_batched: "bool | None" = None):
    """Allocate ALU-32 scratch bands on ``L`` and refresh ``L.D`` (pad to heads).

    ``recurrent_divmod`` adds the digit-index counter band the reused
    division-iteration block reads/threads; OFF (default) is byte-identical to
    the historical unrolled build (same dim, same weights).

    ``shift_via_mul`` (default False, so every EXISTING caller is byte-identical)
    adds the shift-amount one-hot band used to route SHL/SHR through the NATIVE
    MUL/DIV gadgets (``2**n`` power-of-two table) instead of the barrel shifter;
    the qwen_full_vm FULL build passes True (muldiv+bitwise), everything else OFF.

    ``mul_lookahead`` allocates the Kogge-Stone MUL carry-resolve scratch bands so
    ``compile_mul_blocks`` can resolve the columns with a parallel prefix instead of
    7 serial ripples.  ``None`` (default) reads the ``C4_MUL_LOOKAHEAD`` flag (default
    ON), so the resolve bands are present by default and ``compile_mul_blocks`` uses
    them; the bands are allocated LAST, so ``C4_MUL_LOOKAHEAD=0`` (the ripple path)
    is byte-identical to the pre-lookahead layout — same ``L.D``, same band offsets.

    ``kb_batched`` allocates the batched base-16 Kogge-Stone KB-precompute scratch
    bands so ``_kb_precompute_blocks`` normalises all 15 ``KB[k]`` in one prefix
    pass (~9 blocks) instead of 15 serial ripples (~92).  ``None`` (default) reads
    ``C4_KB_BATCHED`` (default ON); the bands are allocated LAST so
    ``C4_KB_BATCHED=0`` (the ripple path) is byte-identical to the pre-batched
    layout."""
    if getattr(L, "ALU32", None) is not None:
        return L.ALU32
    if mul_lookahead is None:
        mul_lookahead = globals()["mul_lookahead"]()   # module-level C4_MUL_LOOKAHEAD reader
    if kb_batched is None:
        kb_batched = globals()["kb_batched"]()         # module-level C4_KB_BATCHED reader
    L.ALU32 = ALU32Bands(L, recurrent_divmod=recurrent_divmod,
                         shift_via_mul=shift_via_mul, mul_lookahead=mul_lookahead,
                         kb_batched=kb_batched)
    while L._off % L.n_heads != 0:
        L._scalar(f"_alupad{L._off}")
    L.D = L._off
    return L.ALU32


# ===========================================================================
# 0. operand expand : AX / STACK0 nibbles -> operand bytes  (A = STACK0, B = AX)
# ===========================================================================
def compile_expand(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    spec = _empty_spec(dim, 4 * (3 + 3 + 4))
    u = 0
    for i in range(4):
        u = _clear(spec, u, a.A + i)
        u = _ident(spec, u, {L.STACK0 + 2 * i: 1.0, L.STACK0 + 2 * i + 1: 16.0}, 0.0, a.A + i, 1.0)
        u = _clear(spec, u, a.B + i)
        u = _ident(spec, u, {L.AX + 2 * i: 1.0, L.AX + 2 * i + 1: 16.0}, 0.0, a.B + i, 1.0)
        # ~B byte = 255 - B (ones' complement, for SUB), read straight off AX
        # nibbles (this block's input) so it does not depend on a.B being written.
        u = _clear(spec, u, a.NOTB + i)
        u = _ident(spec, u, {L.AX + 2 * i: -1.0, L.AX + 2 * i + 1: -16.0}, 255.0, a.NOTB + i, 1.0)
    return _truncate(spec, u, dim)


# ===========================================================================
# 1. ADD / SUB : per-byte carry chain (§Addition).  4 sequential blocks (one per
#    byte): byte i reads A_i + B_i (+ carry_in from C[i]); writes RES nibbles
#    lo/hi and carry-out C[i+1] = [sum >= 256].  SUB uses NOTB and cin C[0]=1.
#    The carry lane C[i+1] is written by block i and READ by block i+1 (a genuine
#    chain across blocks — the intermediate residual carries it).
# ===========================================================================
def _byte_add_block(L, dim, A_band, B_band, carry_band, res_nib, i, cin_const=0.0):
    """One byte of the add chain.  sum = A[i] + B[i] + carry_in.  Writes:
      res_nib[2i]   = sum mod 16      (low nibble)
      res_nib[2i+1] = (sum // 16) mod 16   (high nibble)
      carry_band[i+1] = [sum >= 256]  (carry-out).
    ``carry_in`` is carry_band[i] (a 0/1 lane written by the previous block); for
    byte 0 there is no previous block so the initial carry is the constant
    ``cin_const`` (0 for ADD, 1 for SUB's two's-complement +1)."""
    a = L.ALU32
    spec = _empty_spec(dim, 320)
    u = 0
    sum_terms = {A_band + i: 1.0, B_band + i: 1.0}
    # byte 0 has no previous block: the initial carry is the constant cin_const
    # (0 for ADD, 1 for SUB's two's-complement +1). Bytes 1..3 read the carry
    # lane written by the previous block, and take NO constant carry.
    if i == 0:
        sum_const = cin_const
    else:
        sum_const = 0.0
        sum_terms[carry_band + i] = 1.0        # carry_in from previous block
    # carry-out first (needed as a scalar we can also subtract): [sum >= 256]
    u = _clear(spec, u, carry_band + i + 1)
    u = _step_ge(spec, u, sum_terms, sum_const, 256, carry_band + i + 1, 1.0)
    # low nibble = sum mod 16 = sum - 16*floor(sum/16).  sum in [0, 511].
    u = _clear(spec, u, res_nib + 2 * i)
    u = _ident(spec, u, sum_terms, sum_const, res_nib + 2 * i, 1.0)                    # + sum
    u = _floor_div_pow(spec, u, sum_terms, sum_const, 16, 32, res_nib + 2 * i, -16.0)  # - 16*floor(sum/16)
    # high nibble = floor(sum/16) mod 16 = floor(sum/16) - 16*floor(sum/256).
    u = _clear(spec, u, res_nib + 2 * i + 1)
    u = _floor_div_pow(spec, u, sum_terms, sum_const, 16, 32, res_nib + 2 * i + 1, 1.0)
    u = _floor_div_pow(spec, u, sum_terms, sum_const, 256, 2, res_nib + 2 * i + 1, -16.0)
    return _truncate(spec, u, dim)


def compile_addsub_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    blocks = []
    for i in range(4):
        blocks.append((f"alu-add-b{i}",
                       _byte_add_block(L, dim, a.A, a.B, a.ADD_C, a.ADD_RES, i, cin_const=0.0)))
    for i in range(4):
        blocks.append((f"alu-sub-b{i}",
                       _byte_add_block(L, dim, a.A, a.NOTB, a.SUB_C, a.SUB_RES, i, cin_const=1.0)))
    return blocks


# ===========================================================================
# 2. MUL : NIBBLE schoolbook (§Multiplication).  Operands are 8 nibbles each
#    (32-bit); the partial products a_i*b_j (nibbles, <= 15*15 = 225) go to
#    column i+j (weight 16^(i+j)); only i+j < 8 matter (i+j >= 8 overflows 2^32).
#    A column's raw sum <= 8*225 = 1800, so no accumulator ever exceeds ~1920
#    (raw + carry) and every relu argument stays RELU_S*1920 = 384000 < 2^24
#    -> fp32-EXACT.  8 carry rounds propagate carries fully; the settled columns
#    ARE the 8 result nibbles.  (Byte schoolbook would push column sums to
#    ~260100 where RELU_S*x = 52M overflows fp32 integer exactness — the reason
#    for the finer nibble decomposition.)
# ===========================================================================
_MUL_NCOL = 8           # 8 nibble columns cover the 32-bit result


# the (i,j) pairs with i+j<8, and each pair's slot index in the PP band.
_MUL_PAIRS = [(i, j) for i in range(_MUL_NCOL) for j in range(_MUL_NCOL)
              if i + j < _MUL_NCOL]


def _mul_products_block(L, dim) -> Dict[str, torch.Tensor]:
    """Materialise the 36 raw nibble products ``PP[idx] = a_i*b_j`` (i+j<8, each
    ``<= 225``) into the PP band via the byte-bounded SiLU-gated multiply.  A
    SEPARATE block (``_mul_split_block``) then splits each PP into low/high nibble
    and accumulates the columns — the split MUST be a downstream block because a
    unit reads the BLOCK INPUT, so the product must live in a residual band first."""
    a = L.ALU32
    spec = _empty_spec(dim, len(_MUL_PAIRS) * 3)
    u = 0
    for idx, (i, j) in enumerate(_MUL_PAIRS):
        u = _clear(spec, u, a.PP + idx)
        u = _mul_gate(spec, u, L.STACK0 + i, L.AX + j, a.PP + idx, 1.0)   # a_i*b_j
    return _truncate(spec, u, dim)


def _mul_split_block(L, dim) -> Dict[str, torch.Tensor]:
    """Split each PP product ``p`` (0..225, now in the residual) into
    ``p = (p mod 16) + 16*(p>>4)``: low nibble -> column ``i+j``, high nibble ->
    column ``i+j+1``.  A column receives at most 8 low (<=15) + 8 high (<=14)
    contributions = ``<= 232 < 256`` so every carry round needs only ``kmax=15``.
    One shared ``floor(p/16)`` staircase routes to both columns."""
    a = L.ALU32
    spec = _empty_spec(dim, _MUL_NCOL + len(_MUL_PAIRS) * (1 + 15 * 2))
    u = 0
    for c in range(_MUL_NCOL):
        u = _clear(spec, u, a.MCOL + c)
    for idx, (i, j) in enumerate(_MUL_PAIRS):
        c = i + j
        pp = a.PP + idx
        u = _ident(spec, u, {pp: 1.0}, 0.0, a.MCOL + c, 1.0)              # + p (into col c)
        if c + 1 < _MUL_NCOL:
            # -16*floor(p/16) into col c (=> p mod 16), +floor(p/16) into col c+1.
            u = _floor_div_pow2(spec, u, {pp: 1.0}, 0.0, 16, 15,
                                a.MCOL + c, -16.0, a.MCOL + c + 1, 1.0)
        else:
            u = _floor_div_pow(spec, u, {pp: 1.0}, 0.0, 16, 15, a.MCOL + c, -16.0)
    return _truncate(spec, u, dim)


# ---------------------------------------------------------------------------
# MUL CARRY-LOOKAHEAD (Kogge-Stone parallel prefix) — the DEFAULT resolve of the
# 8 post-split carry-save columns (``a.MCOL`` < 256) into ``a.MUL_RES``.  Ported
# from the proven ``mul_lookahead`` bakeoff (20,685/20,685 byte-exact incl
# 0xFFFFFFFF^2); it reuses the SAME products+split front-end and the SAME library
# ``_nibble_carry_round`` for round-1, then replaces the 7 serial ripple rounds
# with 3 log-depth prefix stages.  See the ``mul_lookahead`` module docstring for
# the full CLA derivation.  All scratch reads/writes are ALU32 bands allocated by
# ``ALU32Bands`` when ``mul_lookahead`` is on.
# ---------------------------------------------------------------------------
def _mul_round1_block(L, dim, src, dst) -> Dict[str, torch.Tensor]:
    """One base-16 carry round reducing the post-split columns (< 256) to
    ``t_c = (src[c] mod 16) + floor(src[c-1]/16)`` in [0, 30].  Reuses the library
    ``_nibble_carry_round`` (the SAME gadget the ripple baseline uses) — after this
    ONE round the carries are pure BINARY, so the prefix that follows is clean CLA."""
    spec = _empty_spec(dim, _MUL_NCOL * (2 + _NIB_KMAX * 2 + _NIB_KMAX * 2 + 2))
    u = _nibble_carry_round(spec, 0, src, dst, _MUL_NCOL)
    return _truncate(spec, u, dim)


def _mul_gp_block(L, dim, t_band, g_band, p_band) -> Dict[str, torch.Tensor]:
    """Generate / propagate lanes from ``t_c`` in [0, 30] (SET each):
        G_c = [t_c >= 16]         (carries out with no incoming carry)
        P_c = [t_c == 15] = [t_c >= 15] - [t_c >= 16]   (carries out iff incoming).
    Both 0/1; thresholds 15/16 -> tiny fp32 args."""
    spec = _empty_spec(dim, _MUL_NCOL * (2 + 2 + 2 + 2))
    u = 0
    for c in range(_MUL_NCOL):
        tc = {t_band + c: 1.0}
        u = _clear(spec, u, g_band + c)
        u = _step_ge(spec, u, tc, 0.0, 16, g_band + c, 1.0)             # G = [t>=16]
        u = _clear(spec, u, p_band + c)
        u = _step_ge(spec, u, tc, 0.0, 15, p_band + c, 1.0)            # +[t>=15]
        u = _step_ge(spec, u, tc, 0.0, 16, p_band + c, -1.0)          # -[t>=16] => [t==15]
    return _truncate(spec, u, dim)


def _mul_ks_stage_block(L, dim, g_src, p_src, g_dst, p_dst, d) -> Dict[str, torch.Tensor]:
    """ONE Kogge-Stone prefix combine stage at distance ``d`` (SET g_dst/p_dst):
        for c >= d:  (G_c, P_c) := (G_c OR (P_c AND G_{c-d}),  P_c AND P_{c-d})
        for c <  d:  carried through.
    0/1 lanes, single-staircase exact forms:
        G_c OR (P_c AND G_{c-d}) = [2*G_c + P_c + G_{c-d} >= 2]
        P_c AND P_{c-d}          = [P_c + P_{c-d} >= 2].
    Reads the block INPUT (the previous stage's g_src/p_src), so one block."""
    spec = _empty_spec(dim, _MUL_NCOL * (2 + 2 + 2 + 2))
    u = 0
    for c in range(_MUL_NCOL):
        if c >= d:
            gform = {g_src + c: 2.0, p_src + c: 1.0, g_src + c - d: 1.0}
            u = _clear(spec, u, g_dst + c)
            u = _step_ge(spec, u, gform, 0.0, 2, g_dst + c, 1.0)
            pform = {p_src + c: 1.0, p_src + c - d: 1.0}
            u = _clear(spec, u, p_dst + c)
            u = _step_ge(spec, u, pform, 0.0, 2, p_dst + c, 1.0)
        else:
            u = _clear(spec, u, g_dst + c)
            u = _ident(spec, u, {g_src + c: 1.0}, 0.0, g_dst + c, 1.0)
            u = _clear(spec, u, p_dst + c)
            u = _ident(spec, u, {p_src + c: 1.0}, 0.0, p_dst + c, 1.0)
    return _truncate(spec, u, dim)


def _mul_apply_block(L, dim, t_band, g_final, res_band) -> Dict[str, torch.Tensor]:
    """Final digits into MUL_RES (SET), fused with the carry-apply (no separate copy):
        b_c = G_final[c-1]  (carry INTO column c; b_0 = 0)
        digit_c = t_c + G_final[c-1] - 16*G_final[c]
    (since [t_c + b_c >= 16] = b_{c+1} = G_final[c]).  The top column's carry-out
    G_final[NCOL-1] overflows past the kept nibbles and is dropped (exactly as the
    ripple baseline dropped the top carry -> result & 0xFFFFFFFF)."""
    spec = _empty_spec(dim, _MUL_NCOL * 4)
    u = 0
    for c in range(_MUL_NCOL):
        u = _clear(spec, u, res_band + c)
        u = _ident(spec, u, {t_band + c: 1.0}, 0.0, res_band + c, 1.0)          # + t_c
        if c >= 1:
            u = _ident(spec, u, {g_final + c - 1: 1.0}, 0.0, res_band + c, 1.0)  # + b_c
        u = _ident(spec, u, {g_final + c: 1.0}, 0.0, res_band + c, -16.0)        # - 16*b_{c+1}
    return _truncate(spec, u, dim)


def _mul_lookahead_resolve_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Resolve the split columns (``a.MCOL`` < 256) into ``a.MUL_RES`` with the
    Kogge-Stone parallel prefix: round-1 (-> t_c in [0,30]) | G/P | 3 prefix combine
    stages (double-buffered) | fused apply.  ``ceil(log2 8) = 3`` prefix stages
    replace the 7 serial ripple rounds; the apply fuses the result write."""
    a = L.ALU32
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    blocks.append(("alu-mul-round1", _mul_round1_block(L, dim, a.MCOL, a.MUL_T)))
    blocks.append(("alu-mul-gp", _mul_gp_block(L, dim, a.MUL_T, a.MUL_G0, a.MUL_P0)))
    (gs, ps), (gd, pd) = (a.MUL_G0, a.MUL_P0), (a.MUL_G1, a.MUL_P1)
    d = 1
    st = 0
    while d < _MUL_NCOL:
        blocks.append((f"alu-mul-ks{st}",
                       _mul_ks_stage_block(L, dim, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)   # swap buffers
        d *= 2
        st += 1
    blocks.append(("alu-mul-apply", _mul_apply_block(L, dim, a.MUL_T, gs, a.MUL_RES)))
    return blocks


def compile_mul_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """32-bit MUL block stack: products + split front-end, then the column carry
    resolve into ``a.MUL_RES``.

    ``C4_MUL_LOOKAHEAD`` (default ON, via ``L.ALU32.mul_lookahead``) selects the
    Kogge-Stone parallel-prefix resolve (round-1 | G/P | 3 prefix stages | fused
    apply = 6 blocks); OFF selects the 7 serial ripple rounds + result copy — the
    two are byte-identical over every operand.  The DIV/MOD path is UNAFFECTED (it
    calls ``_carry_round_block`` directly and always ripples)."""
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    blocks = [("alu-mul-products", _mul_products_block(L, dim)),
              ("alu-mul-split", _mul_split_block(L, dim))]
    if getattr(a, "mul_lookahead", False):
        # KOGGE-STONE resolve: 3 log-depth prefix stages settle the columns and the
        # fused apply writes MUL_RES (no separate result copy).
        blocks += _mul_lookahead_resolve_blocks(L, dim)
        return blocks
    # RIPPLE (flag-OFF, byte-identical to the pre-lookahead build): carry rounds
    # settle every column to a single nibble.  A carry ripples one column per round;
    # split partials keep columns < 256 so _MUL_CARRY_ROUNDS (verified worst-case +
    # headroom) fully settle it, kmax=15 each -> tiny.
    src, dst = a.MCOL, a.MC1
    for r in range(_MUL_CARRY_ROUNDS):
        blocks.append((f"alu-mul-carry{r}", _carry_round_block(L, dim, src, dst, _MUL_NCOL)))
        src, dst = dst, src
    # the settled columns ARE the result nibbles: copy src -> MUL_RES.
    copy = _empty_spec(dim, _MUL_NCOL * 2)
    u = 0
    for c in range(_MUL_NCOL):
        u = _clear(copy, u, a.MUL_RES + c)
        u = _ident(copy, u, {src + c: 1.0}, 0.0, a.MUL_RES + c, 1.0)
    blocks.append(("alu-mul-result", _truncate(copy, u, dim)))
    return blocks


# ===========================================================================
# 3. DIV / MOD : base-16 long division on NIBBLES (§Division).  8 nibble
#    iterations, MSB first.  Per iteration:
#       R = 16*R + dividend_nibble         (a nibble shift + insert)
#       q = sum_{k=1..15} [ R >= k*b ]     (nibble-wise lexicographic compare)
#       R = R - q*b                         (nibble borrow chain)
#    fp32-EXACT: R and every k*b live as NIBBLES (0..15), and the comparison is
#    lexicographic byte/nibble-wise, so no compared quantity ever exceeds 15 even
#    though k*b reaches ~6e10 (the value that forced the standalone gadget onto
#    fp64).  ``b == 0 -> q = r = 0`` (ISA_SPEC 4.2).  DIV returns the assembled
#    quotient nibbles; MOD the final remainder nibbles.
# ===========================================================================
def _kb_precompute_blocks_batched(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """BATCHED base-16 Kogge-Stone KB-precompute: resolve all 15 ``KB[k]=k*b`` in
    ONE parallel-prefix pass (raw | round-1 | G/P | ceil(log2 RN) KS stages | fused
    apply | BZ = ~9 blocks) instead of 15 serial ``_KB_CARRY_ROUNDS`` ripples
    (~92 blocks).  Reuses the proven block builders from ``div_kb_lookahead`` (a
    lazy import to avoid a load-time cycle: that module imports from here); passes
    the ALU32 scratch band offsets (``KB_T``/``KB_G0``/``KB_P0``/``KB_G1``/
    ``KB_P1``) so the resolve lives on the SAME layout, and writes the result into
    the production ``a.KB`` band the compare reads — byte-identical to the ripple."""
    from . import div_kb_lookahead as DKB
    a = L.ALU32
    RN = a.RN
    _ONE_prev = globals().get("_ONE")
    globals()["_ONE"] = L.ONE
    n_stages = DKB._n_prefix_stages(RN)
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    blocks.append(("alu-div-kb-raw", DKB._kb_raw_block(L, dim, RN)))
    blocks.append(("alu-div-kb-round1", DKB._round1_block(L, dim, RN, a.KB, a.KB_T)))
    blocks.append(("alu-div-kb-gp", DKB._gp_block(L, dim, RN, a.KB_T, a.KB_G0, a.KB_P0)))
    (gs, ps), (gd, pd) = (a.KB_G0, a.KB_P0), (a.KB_G1, a.KB_P1)
    d = 1
    st = 0
    while d < RN:
        blocks.append((f"alu-div-kb-ks{st}",
                       DKB._ks_stage_block(L, dim, RN, gs, ps, gd, pd, d)))
        (gs, ps), (gd, pd) = (gd, pd), (gs, ps)
        d *= 2
        st += 1
    blocks.append(("alu-div-kb-apply", DKB._apply_block(L, dim, RN, a.KB_T, gs, a.KB)))
    blocks.append(("alu-div-bz", DKB._bz_block(L, dim)))
    globals()["_ONE"] = _ONE_prev
    return blocks


def _kb_precompute_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Compute KB[k] = k*b nibbles for k=1..15 from the AX (=B) nibble bands, plus
    the divisor-zero predicate BZ.  ``k*b`` = (single-nibble k) * (8-nibble b): a
    per-nibble multiply (b_j*k <= 15*15 = 225 < 256) laid into 9 raw columns, then
    carry-normalised in place with ``kmax=15`` (every column stays ``< 256``).

    Each KB[k] is carried in its OWN tiny block (RN rounds, in place) so no block
    is wide — the uniform-hidden budget is set by the small per-k carry, NOT by a
    fused 15-k round.  This is the fix for the recovered draft's 75k-unit blocks.

    When ``C4_KB_BATCHED`` is ON (default) and the batched scratch bands were
    allocated (``a.kb_batched``), route to the batched Kogge-Stone prefix
    (~9 blocks); OFF -> the 15 x ``_KB_CARRY_ROUNDS`` serial ripple below."""
    a = L.ALU32
    if getattr(a, "kb_batched", False):
        return _kb_precompute_blocks_batched(L, dim)
    RN = a.RN
    blocks = []
    # raw columns per k: KB[k] column c = b_nib[c] * k  (only c<8 for b; <= 225).
    raw = _empty_spec(dim, 16 * RN + 16 * RN * 2)
    u = 0
    for k in range(1, 16):
        base = a.KB + RN * k
        for c in range(RN):
            u = _clear(raw, u, base + c)
        for c in range(8):                    # b has 8 nibbles
            u = _ident(raw, u, {L.AX + c: float(k)}, 0.0, base + c, 1.0)   # + k*b_nib[c]
    blocks.append(("alu-div-kb-raw", _truncate(raw, u, dim)))
    # carry-normalise each KB[k] in place.  A carry ripples one column per round;
    # KB raw = k*b_nib <= 225 settles in <= _KB_CARRY_ROUNDS (verified).
    for k in range(1, 16):
        base = a.KB + RN * k
        for rnd in range(_KB_CARRY_ROUNDS):
            blocks.append((f"alu-div-kb{k}-c{rnd}",
                           _carry_round_block(L, dim, base, base, RN)))
    # BZ = [b == 0] = product over nibbles [b_nib==0]; b<2^32 -> 8 nibbles.
    bz = _empty_spec(dim, 2 + 8 * 4)
    u = 0
    u = _clear(bz, u, a.BZ)
    # [b==0] via 1 - [b_sum >= 1] where b_sum = sum of nibbles (0 iff all zero).
    u = _ident(bz, u, {L.ONE: 1.0}, 0.0, a.BZ, 1.0)                    # + 1
    bsum = {L.AX + c: 1.0 for c in range(8)}
    u = _step_ge(bz, u, bsum, 0.0, 1, a.BZ, -1.0)                      # - [sum>=1]
    blocks.append(("alu-div-bz", _truncate(bz, u, dim)))
    return blocks


# --- per-iteration DIV scratch: gt/eq lanes for R vs KB[k] -----------------
# GT[k,i] = [R[i] > KB_k[i]], EQ[k,i] = [R[i] == KB_k[i]] for k=1..15, i=0..RN-1.
# Laid out in a scratch band GTEQ of size 2*15*RN.  These are the ONLY two-band
# functions in the comparison; everything downstream ANDs these 0/1 lanes.
def _div_gteq_block(L, dim) -> Dict[str, torch.Tensor]:
    """Materialise GT[k,i] and EQ[k,i] from R (current remainder nibbles) and the
    precomputed KB[k] nibbles.  d = R[i] - KB_k[i] in [-15,15]:
        GT = [d >= 1]      EQ = [d == 0] = [d>=0] - [d>=1].
    (SET each lane.)"""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, 15 * RN * (2 + 4 + 4))
    u = 0
    for k in range(1, 16):
        for i in range(RN):
            gt = a.GT + (k - 1) * RN + i
            eq = a.EQ_N + (k - 1) * RN + i
            d = {a.R + i: 1.0, a.KB + RN * k + i: -1.0}
            u = _clear(spec, u, gt)
            u = _step_ge(spec, u, d, 0.0, 1, gt, 1.0)                # [d>=1]
            u = _clear(spec, u, eq)
            u = _step_ge(spec, u, d, 0.0, 0, eq, 1.0)                # +[d>=0]
            u = _step_ge(spec, u, d, 0.0, 1, eq, -1.0)              # -[d>=1] => [d==0]
    return _truncate(spec, u, dim)


def _div_qdigit_block(L, dim) -> Dict[str, torch.Tensor]:
    """QD = sum_{k=1..15} [R >= KB_k], the quotient digit.  For each k the
    lexicographic '>=' expands to a sum of suffix-ANDs of the GT/EQ lanes:
        [R>=KB_k] = sum_{i=RN-1..0} ( GT[k,i] AND EQ[k,i+1..RN-1] )
                    + ( EQ[k,0..RN-1] )                (all-equal term)
    each AND realised by ONE guarded unit.  (SET QD.)"""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, 1 + 15 * (RN + 1))
    u = 0
    u = _clear(spec, u, a.QD)
    for k in range(1, 16):
        gtb = a.GT + (k - 1) * RN
        eqb = a.EQ_N + (k - 1) * RN
        # suffix-AND terms: position i is the highest nibble where X>Y (all higher equal)
        for i in range(RN - 1, -1, -1):
            windows = [(gtb + i, 1.0, 0.0)] + [(eqb + j, 1.0, 0.0) for j in range(i + 1, RN)]
            u = _guard(spec, u, windows, {L.ONE: 1.0}, 0.0, a.QD, 1.0)
        # all-equal term (R == KB_k exactly): also counts as >=
        windows = [(eqb + j, 1.0, 0.0) for j in range(RN)]
        u = _guard(spec, u, windows, {L.ONE: 1.0}, 0.0, a.QD, 1.0)
    return _truncate(spec, u, dim)


def _div_qb_block(L, dim) -> Dict[str, torch.Tensor]:
    """QB = QD * b  (nibbles).  QD is a single digit 0..15; b is 8 nibbles.  Raw
    columns QB_raw[c] = QD * b_nib[c] (<= 15*15 = 225) via the SiLU-gated multiply
    with QD as the byte multiplicand.  (Carry-normalised by the following block.)"""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, RN + 8 * 2)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.QB + c)
    for c in range(8):
        u = _mul_gate(spec, u, a.QD, L.AX + c, a.QB + c, 1.0)      # QD * b_nib[c]
    return _truncate(spec, u, dim)


def _div_sub_nibble_block(L, dim, i) -> Dict[str, torch.Tensor]:
    """One nibble of the borrow subtract R2 = R - QB.  nibble ``i``:
        diff = R[i] - QB[i] - borrow_in(SUBB[i])   in [-16, 15]
        R2[i] = diff + 16*[diff < 0]    borrow_out SUBB[i+1] = [diff < 0]
    ``[diff < 0] = 1 - [diff >= 0]``.  Borrow threaded through SUBB (block input
    carries the previous nibble's borrow)."""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, 40)
    u = 0
    diff = {a.R + i: 1.0, a.QB + i: -1.0}
    if i > 0:
        diff[a.SUBB + i] = -1.0
    # borrow_out = [diff < 0] = 1 - [diff >= 0]
    u = _clear(spec, u, a.SUBB + i + 1)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.SUBB + i + 1, 1.0)      # +1
    u = _step_ge(spec, u, diff, 0.0, 0, a.SUBB + i + 1, -1.0)        # -[diff>=0]
    # R2[i] = diff + 16*borrow_out
    u = _clear(spec, u, a.R2 + i)
    u = _ident(spec, u, diff, 0.0, a.R2 + i, 1.0)                    # + diff
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.R2 + i, 16.0)          # + 16
    u = _step_ge(spec, u, diff, 0.0, 0, a.R2 + i, -16.0)           # - 16*[diff>=0]
    return _truncate(spec, u, dim)


def _div_shift_block(L, dim, div_nib_idx) -> Dict[str, torch.Tensor]:
    """R = 16*R + dividend_nibble[div_nib_idx].  Nibble shift: R[c] <- R[c-1]
    (for c>=1), R[0] <- dividend nibble.  Reads R (block input) so the shift is a
    clean gather; the dividend nibble is STACK0[div_nib_idx] (=A).  (SET R.)"""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, RN * 2 + 2)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
        if c == 0:
            u = _ident(spec, u, {L.STACK0 + div_nib_idx: 1.0}, 0.0, a.R + c, 1.0)
        else:
            u = _ident(spec, u, {a.R + c - 1: 1.0}, 0.0, a.R + c, 1.0)
    return _truncate(spec, u, dim)


def _div_qcopy_block(L, dim, q_out_idx) -> Dict[str, torch.Tensor]:
    """Store the current quotient digit QD into DIV_RES[q_out_idx] (the quotient
    nibble for this iteration).  Iterations run MSB-first: iteration 0 produces
    quotient nibble 7 (the most significant), iteration 7 nibble 0."""
    a = L.ALU32
    spec = _empty_spec(dim, 2)
    u = 0
    u = _clear(spec, u, a.DIV_RES + q_out_idx)
    u = _ident(spec, u, {a.QD: 1.0}, 0.0, a.DIV_RES + q_out_idx, 1.0)
    return _truncate(spec, u, dim)


def _div_r2_to_r_block(L, dim) -> Dict[str, torch.Tensor]:
    """Copy R2 (post-subtract remainder) back into R for the next iteration."""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, RN * 2)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
        u = _ident(spec, u, {a.R2 + c: 1.0}, 0.0, a.R + c, 1.0)
    return _truncate(spec, u, dim)


def _div_init_block(L, dim) -> Dict[str, torch.Tensor]:
    """Zero the running remainder R and the quotient result before iteration 0."""
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, RN + 8)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
    for c in range(8):
        u = _clear(spec, u, a.DIV_RES + c)
    return _truncate(spec, u, dim)


def _div_finalize_block(L, dim) -> Dict[str, torch.Tensor]:
    """Assemble the final results, honouring b==0 -> (0, 0) per ``isa.interpret``
    (``a // b if b else 0`` / ``a % b if b else 0``; matches the 8-bit reference
    the corpus uses):
        MOD_RES[c] = R[c] (remainder)         but if b==0 -> 0
        DIV_RES stays the quotient nibbles    but if b==0 -> 0
    We overwrite with the divide-by-zero fallback GATED on BZ (=[b==0]):
        MOD_RES = (1-BZ)*R      (i.e. subtract BZ*R)
        DIV_RES = (1-BZ)*DIV_RES  (i.e. subtract BZ*DIV_RES)
    """
    a = L.ALU32
    RN = a.RN
    spec = _empty_spec(dim, 8 * 6 + 8 * 4)
    u = 0
    for c in range(8):
        # MOD_RES[c] = (1-BZ)*R[c] : write R[c], then subtract BZ*R[c].  BOTH read
        # the block-input R[c] (all units see the block input), so they compose to
        # (1-BZ)*R even though the -old-MOD_RES trick would read a stale value.
        u = _clear(spec, u, a.MOD_RES + c)
        u = _ident(spec, u, {a.R + c: 1.0}, 0.0, a.MOD_RES + c, 1.0)              # + R[c]
        gz = (a.BZ, 1.0, 0.0)
        u = _guard(spec, u, [gz], {a.R + c: -1.0}, 0.0, a.MOD_RES + c, 1.0)       # - BZ*R[c]
        # DIV_RES := (1-BZ)*DIV_RES : subtract BZ*DIV_RES (DIV_RES already holds
        # the quotient from the iteration blocks -> it IS in the block input).
        u = _guard(spec, u, [gz], {a.DIV_RES + c: -1.0}, 0.0, a.DIV_RES + c, 1.0)  # - BZ*DIV_RES
    return _truncate(spec, u, dim)


# ===========================================================================
# SIGNED DIV/MOD (C4_DIVMOD_SIGNED, #790) — the sign-magnitude wrapper around the
# UNSIGNED base-16 long divider.  Native c4 ``int`` is signed and c4 divides with
# C truncation-toward-zero, so:
#   * |a| = a<0 ? -a : a  ,  |b| = b<0 ? -b : b     (negate operands before divide)
#   * q = trunc(a/b) = (sign(a)^sign(b)) ? -(|a|//|b|) : |a|//|b|
#   * r = a - q*b        = sign(a)       ? -(|a|%|b|) : |a|%|b|
# The divider itself is UNCHANGED (it sees the magnitudes); two extra blocks do the
# conditional two's-complement negate before (operands) and after (results).
# ===========================================================================
def _neg_zcum_block(L, dim, reg_base) -> Dict[str, torch.Tensor]:
    """Materialise ``ZCUM[j] = [ Σ_{k<j} reg_nib_k == 0 ]`` (j=0..8) — the
    two's-complement carry-in for negating the 8-nibble register at ``reg_base``.
    ``ZCUM[0] = 1`` (empty sum), ``ZCUM[j] = relu(1 - Σ_{k<j} nib_k)`` (the nibbles
    are non-negative, so the prefix sum is 0 iff every lower nibble is 0).  UNGATED
    (writes only the private ZCUM scratch; the apply block is what gates on op/sign).
    SET (self-clears ZCUM first)."""
    a = L.ALU32
    spec = _empty_spec(dim, 9 * 2 + 1)
    u = 0
    for j in range(9):
        u = _clear(spec, u, a.ZCUM + j)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.ZCUM + 0, 1.0)      # ZCUM[0] = 1
    for j in range(1, 9):
        low = {reg_base + k: 1.0 for k in range(min(j, 8))}
        u = _relu(spec, u, {b: -c for b, c in low.items()}, 1.0, a.ZCUM + j, 1.0)  # relu(1-Σ)
    return _truncate(spec, u, dim)


def _cond_negate_apply_block(L, dim, reg_base, sign_flag, gate_ops) -> Dict[str, torch.Tensor]:
    """SET ``reg[j] := (sign AND op) ? twoscomp(reg)[j] : reg[j]`` for j=0..7.
    ``twoscomp[j] = (15 - x_j) + ZCUM[j] - 16*ZCUM[j+1]`` (ZCUM from the PRIOR
    :func:`_neg_zcum_block`).  In place: ``reg[j] += g*(y_j - x_j)`` where
    ``g = sign_flag AND (any op in gate_ops)`` and ``y_j - x_j = (15 - 2*x_j) +
    ZCUM[j] - 16*ZCUM[j+1]``.  Every value reads the block INPUT."""
    a = L.ALU32
    wins = [(sign_flag, 1.0, 0.0)] + [(L.OP_IS + op, 1.0, 0.0) for op in gate_ops]
    # NOTE: gate_ops is OR'd — but _guard ANDs its windows.  With DIV and MOD both
    # present the op-gate must be an OR.  Since exactly one op fires per step, we emit
    # ONE _guard per op (each ANDs sign AND that op); their sum = sign AND (DIV|MOD).
    spec = _empty_spec(dim, 8 * len(gate_ops) * 3 + 4)
    u = 0
    for op in gate_ops:
        w = [(sign_flag, 1.0, 0.0), (L.OP_IS + op, 1.0, 0.0)]
        for j in range(8):
            # g*(15 - 2*x_j)
            u = _guard(spec, u, w, {reg_base + j: -2.0}, 15.0, reg_base + j, 1.0)
            # g*ZCUM[j]
            u = _guard(spec, u, w, {a.ZCUM + j: 1.0}, 0.0, reg_base + j, 1.0)
            # g*(-16*ZCUM[j+1])
            u = _guard(spec, u, w, {a.ZCUM + j + 1: -16.0}, 0.0, reg_base + j, 1.0)
    return _truncate(spec, u, dim)


def compile_divmod_sign_prep(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """SIGNED DIV/MOD, PART 1 (before the unsigned divide): detect the operand signs
    and replace STACK0 (dividend a) / AX (divisor b) with their MAGNITUDES |a|/|b|,
    gated on DIV|MOD.  Blocks:
      sd-sign  : SGN_A=[STACK0 nib7>=8], SGN_B=[AX nib7>=8], RES_SGN=SGN_A xor SGN_B.
      sd-zca   : ZCUM for STACK0 ; sd-nega : negate STACK0 if SGN_A (DIV|MOD).
      sd-zcb   : ZCUM for AX     ; sd-negb : negate AX     if SGN_B (DIV|MOD).
    A non-DIV/MOD step is a strict no-op (every negate is op-gated; the sign lanes
    are private scratch read only by these blocks)."""
    a = L.ALU32
    gate = [isa.DIV, isa.MOD]
    # --- sd-sign : sign flags from the operand top nibble (>= 8). ---
    sgn = _empty_spec(dim, 2 * 3)
    u = 0
    u = _clear(sgn, u, a.SGN_A)
    u = _step_ge(sgn, u, {L.STACK0 + 7: 1.0}, 0.0, 8, a.SGN_A, 1.0)     # [STACK0 nib7>=8]
    u = _clear(sgn, u, a.SGN_B)
    u = _step_ge(sgn, u, {L.AX + 7: 1.0}, 0.0, 8, a.SGN_B, 1.0)         # [AX nib7>=8]
    sgn = _truncate(sgn, u, dim)
    # --- sd-ressgn : RES_SGN = SGN_A xor SGN_B — a SEPARATE block so the SGN_A/SGN_B
    # it reads are the values sd-sign WROTE (a same-block read would see the stale
    # block input, i.e. 0, and RES_SGN would never fire). ---
    rs = _empty_spec(dim, 1 + 2 + 2)                    # clear + 2 _step_ge (2 units each)
    u = 0
    u = _clear(rs, u, a.RES_SGN)
    u = _step_ge(rs, u, {a.SGN_A: 1.0, a.SGN_B: 1.0}, 0.0, 1, a.RES_SGN, 1.0)   # +[sum>=1]
    u = _step_ge(rs, u, {a.SGN_A: 1.0, a.SGN_B: 1.0}, 0.0, 2, a.RES_SGN, -1.0)  # -[sum>=2]
    rs = _truncate(rs, u, dim)
    return [
        ("sd-sign", sgn),
        ("sd-ressgn", rs),
        ("sd-zca", _neg_zcum_block(L, dim, L.STACK0)),
        ("sd-nega", _cond_negate_apply_block(L, dim, L.STACK0, a.SGN_A, gate)),
        ("sd-zcb", _neg_zcum_block(L, dim, L.AX)),
        ("sd-negb", _cond_negate_apply_block(L, dim, L.AX, a.SGN_B, gate)),
    ]


def compile_divmod_sign_apply(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """SIGNED DIV/MOD, PART 2 (after the unsigned divide, before ax-mux): the divider
    wrote the UNSIGNED |a|//|b| into DIV_RES and |a|%|b| into MOD_RES.  Apply the
    result signs:
      * quotient  DIV_RES negated iff RES_SGN (= SGN_A xor SGN_B)  — gated on DIV.
      * remainder MOD_RES negated iff SGN_A   (r has the dividend's sign)  — on MOD.
    Only the ACTIVE op's result feeds AX (the ax-mux is op-gated), so negating both
    is safe; we op-gate anyway so a DIV step never perturbs MOD_RES and vice-versa."""
    a = L.ALU32
    return [
        ("sd-zcq", _neg_zcum_block(L, dim, a.DIV_RES)),
        ("sd-negq", _cond_negate_apply_block(L, dim, a.DIV_RES, a.RES_SGN, [isa.DIV])),
        ("sd-zcr", _neg_zcum_block(L, dim, a.MOD_RES)),
        ("sd-negr", _cond_negate_apply_block(L, dim, a.MOD_RES, a.SGN_A, [isa.MOD])),
    ]


# ===========================================================================
# RECURRENT DIV/MOD (the refactor): ONE division-iteration block-body, REUSED
# 8 times by the block-application loop, carrying the running remainder R, the
# quotient nibbles DIV_RES, and the digit-index counter IT through the residual
# across the reused applications — exactly as the VM step is itself recurrent
# (one step-block re-applied per token, threading PC/AX/SP).  The unrolled
# `_div_shift_block(div_nib_idx)` / `_div_qcopy_block(q_out_idx)` baked the
# iteration index into the WEIGHTS (so the 8 iterations were 8 DISTINCT blocks);
# here the index lives in the DATA (the IT counter) and the shift-gather /
# qcopy-scatter select the dividend/quotient nibble via one-hot(IT), so every
# iteration application is byte-identical weights.
# ===========================================================================
def _div_it_oh_units(spec, u, a, it_const=0.0):
    """Recompute IT_OH[j] = [(IT+it_const) == j] = [form>=j] - [form>=j+1] for
    j=0..7 (SET each), where ``form = IT + it_const`` reads the BLOCK INPUT IT.

    ``it_const`` lets the increment block compute IT_OH from IT+1 in the SAME
    block (every FFN unit reads the block input, so IT_OH must be formed from the
    pre-increment IT plus the literal +1, not from the post-increment IT which is
    only a residual delta this block writes)."""
    for j in range(8):
        u = _clear(spec, u, a.IT_OH + j)
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j, a.IT_OH + j, 1.0)       # +[form>=j]
        u = _step_ge(spec, u, {a.IT: 1.0}, it_const, j + 1, a.IT_OH + j, -1.0)  # -[form>=j+1]
    return u


def _div_it_init_block(L, dim) -> Dict[str, torch.Tensor]:
    """Zero R, zero the quotient result, set the digit counter IT = 0, and prime
    IT_OH = one-hot(0).  The recurrent counterpart of ``_div_init_block``."""
    a = L.ALU32
    RN = a.RN
    # units: R clears (RN) + DIV_RES clears (8) + IT clear (1) + IT_OH (8*5).
    spec = _empty_spec(dim, RN + 8 + 1 + 8 * 5)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
    for c in range(8):
        u = _clear(spec, u, a.DIV_RES + c)
    u = _clear(spec, u, a.IT)                          # IT = 0
    u = _div_it_oh_units(spec, u, a)                   # IT_OH = one-hot(0)
    return _truncate(spec, u, dim)


def _div_shift_block_rec(L, dim) -> Dict[str, torch.Tensor]:
    """Index-INDEPENDENT nibble shift: R[c] <- R[c-1] (c>=1); R[0] <- the dividend
    nibble selected by the counter, ``sum_j IT_OH[j] * STACK0[7-j]`` (iteration
    ``it`` folds STACK0 nibble ``7-it``, MSB first).  Reads the block INPUT (R and
    IT_OH), so the same weights fire for every iteration.  (SET R.)"""
    a = L.ALU32
    RN = a.RN
    # units: (RN-1) high-nibble shifts (2 each) + R[0] clear (1) + 8 gather guards.
    spec = _empty_spec(dim, (RN - 1) * 2 + 1 + 8)
    u = 0
    # shift the high nibbles first is irrelevant (all read block input): SET each.
    for c in range(RN - 1, 0, -1):
        u = _clear(spec, u, a.R + c)
        u = _ident(spec, u, {a.R + c - 1: 1.0}, 0.0, a.R + c, 1.0)
    u = _clear(spec, u, a.R + 0)
    for j in range(8):                                 # R[0] = sum_j IT_OH[j]*STACK0[7-j]
        u = _guard(spec, u, [(a.IT_OH + j, 1.0, 0.0)],
                   {L.STACK0 + (7 - j): 1.0}, 0.0, a.R + 0, 1.0)
    return _truncate(spec, u, dim)


def _div_qcopy_block_rec(L, dim) -> Dict[str, torch.Tensor]:
    """Index-INDEPENDENT quotient store: DIV_RES[7-j] += IT_OH[j]*QD (iteration
    ``it`` writes quotient nibble ``7-it``, MSB first).  SET the selected slot by
    first clearing it gated on IT_OH[j], then adding QD gated on IT_OH[j]."""
    a = L.ALU32
    spec = _empty_spec(dim, 8 * 2)
    u = 0
    for j in range(8):
        slot = a.DIV_RES + (7 - j)
        g = (a.IT_OH + j, 1.0, 0.0)
        u = _guard(spec, u, [g], {slot: -1.0}, 0.0, slot, 1.0)     # clear slot gated
        u = _guard(spec, u, [g], {a.QD: 1.0}, 0.0, slot, 1.0)      # + QD gated
    return _truncate(spec, u, dim)


def _div_r2r_incr_block(L, dim) -> Dict[str, torch.Tensor]:
    """Copy R2 -> R for the next iteration, INCREMENT the digit counter IT += 1, and
    recompute IT_OH = one-hot(IT).  This is the recurrent step's state-thread: the
    reused iteration body's tail that advances the counter (like PC += 1)."""
    a = L.ALU32
    RN = a.RN
    # units: R copy (RN*2) + IT increment (1) + IT_OH (8*5).
    spec = _empty_spec(dim, RN * 2 + 1 + 8 * 5)
    u = 0
    for c in range(RN):
        u = _clear(spec, u, a.R + c)
        u = _ident(spec, u, {a.R2 + c: 1.0}, 0.0, a.R + c, 1.0)
    u = _ident(spec, u, {L.ONE: 1.0}, 0.0, a.IT, 1.0)      # IT += 1
    u = _div_it_oh_units(spec, u, a, it_const=1.0)         # IT_OH = one-hot(IT+1)
    return _truncate(spec, u, dim)


def _divmod_iteration_body(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The ONE reusable long-division iteration body (index-independent).  Order
    matches one unrolled iteration: shift, gteq, qdigit, qcopy, qb, qb-carry x N,
    sub-nibble x RN, r2->r + IT++.  Returned as a named block list; the caller
    reuses the SAME specs for all 8 iterations (weight-shared recurrence)."""
    a = L.ALU32
    RN = a.RN
    body: List[Tuple[str, Dict[str, torch.Tensor]]] = [
        ("alu-div-shift", _div_shift_block_rec(L, dim)),
        ("alu-div-gteq", _div_gteq_block(L, dim)),
        ("alu-div-qd", _div_qdigit_block(L, dim)),
        ("alu-div-qcopy", _div_qcopy_block_rec(L, dim)),
        ("alu-div-qb", _div_qb_block(L, dim)),
    ]
    for r in range(_QB_CARRY_ROUNDS):
        body.append((f"alu-div-qbc-{r}", _carry_round_block(L, dim, a.QB, a.QB, RN)))
    for c in range(RN):
        body.append((f"alu-div-sub-{c}", _div_sub_nibble_block(L, dim, c)))
    body.append(("alu-div-r2r", _div_r2r_incr_block(L, dim)))
    return body


def compile_divmod_blocks_recurrent(L, dim, n_iters: int = 8):
    """RECURRENT base-16 long division: KB-precompute + init, then the SINGLE
    iteration body REUSED ``n_iters`` times, then finalize.

    Returns ``(unique_blocks, apply_names)``:
      * ``unique_blocks`` — the DISTINCT block specs to materialise (KB, init, the
        ONE iteration body, finalize).  This is what the model stores.
      * ``apply_names``   — the full application order (the body names repeat
        ``n_iters`` times); the block-application loop applies each named block,
        reusing the shared body weights ``n_iters`` times.  This threads R / IT /
        DIV_RES through the residual across the reused applications, exactly like
        the VM step threads PC/AX/SP through the emitted frame.

    fp/semantics are IDENTICAL to :func:`compile_divmod_blocks` — the same gteq /
    qdigit / qb / carry / sub gadgets, only the iteration index moved from the
    weights (8 distinct shift/qcopy blocks) into the IT counter (one shared body).
    """
    global _ONE
    _ONE = L.ONE
    assert getattr(L.ALU32, "recurrent_divmod", False), \
        "compile_divmod_blocks_recurrent needs extend_layout_for_alu32(recurrent_divmod=True)"
    prefix = _kb_precompute_blocks(L, dim)                 # KB[k]=k*b, BZ
    prefix.append(("alu-div-init", _div_it_init_block(L, dim)))
    body = _divmod_iteration_body(L, dim)                  # the ONE reused body
    finalize = [("alu-div-finalize", _div_finalize_block(L, dim))]

    # C4_RECURRENT_CORE: NESTED tie of the 6 byte-identical ``alu-div-qbc-*`` carry
    # rounds into ONE stored ``alu-div-qbc`` block applied 6x per iteration.  The
    # blocks are position-independent (src==dst==QB), so the applied SEQUENCE is
    # unchanged -> byte-EXACT; only the STORED body shrinks by 5 blocks.
    body_apply_names = [n for n, _ in body]               # per-iteration apply order
    if recurrent_core():
        qbc_specs = [(n, s) for n, s in body if n.startswith("alu-div-qbc-")]
        if qbc_specs:
            n_qbc = len(qbc_specs)
            tied_name = "alu-div-qbc"
            tied_spec = qbc_specs[0][1]                    # all 6 are byte-identical
            # rebuild ``body`` (stored/unique) with ONE tied qbc block; rebuild the
            # per-iteration apply order with the tied name repeated n_qbc times.
            new_body, new_apply, inserted = [], [], False
            for name, spec in body:
                if name.startswith("alu-div-qbc-"):
                    if not inserted:
                        new_body.append((tied_name, tied_spec))
                        new_apply += [tied_name] * n_qbc
                        inserted = True
                    # subsequent qbc-* are dropped from the stored set (tied).
                else:
                    new_body.append((name, spec))
                    new_apply.append(name)
            body, body_apply_names = new_body, new_apply

    unique = prefix + body + finalize                     # DISTINCT specs to store
    prefix_names = [n for n, _ in prefix]
    # application order: prefix (once) | body x n_iters | finalize (once).  The
    # body names REPEAT so the loop reuses the shared iteration weights n_iters x.
    apply_names = list(prefix_names)
    for _ in range(n_iters):
        apply_names += body_apply_names
    apply_names.append("alu-div-finalize")
    return unique, apply_names


def compile_divmod_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """The full base-16 long-division block stack (DIV -> DIV_RES quotient nibbles,
    MOD -> MOD_RES remainder nibbles).  Order:
        kb-precompute (KB[k]=k*b, BZ) | init | 8 x [shift, gteq, qdigit, qcopy,
        qb, 9x qb-carry, RN x sub-nibble, r2->r] | finalize."""
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    RN = a.RN
    blocks: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    blocks += _kb_precompute_blocks(L, dim)
    blocks.append(("alu-div-init", _div_init_block(L, dim)))
    for it in range(8):                          # 8 dividend nibbles, MSB first
        div_nib_idx = 7 - it                     # MSB nibble of the 32-bit dividend
        q_out_idx = 7 - it                       # quotient nibble (MSB first)
        blocks.append((f"alu-div-shift{it}", _div_shift_block(L, dim, div_nib_idx)))
        blocks.append((f"alu-div-gteq{it}", _div_gteq_block(L, dim)))
        blocks.append((f"alu-div-qd{it}", _div_qdigit_block(L, dim)))
        blocks.append((f"alu-div-qcopy{it}", _div_qcopy_block(L, dim, q_out_idx)))
        blocks.append((f"alu-div-qb{it}", _div_qb_block(L, dim)))
        # carry-normalise QB in place (kmax=15: QB raw col = QD*b_nib <= 225 < 256).
        # The read-old / write-delta pattern is safe within a block.
        for r in range(_QB_CARRY_ROUNDS):
            blocks.append((f"alu-div-qbc{it}-{r}",
                           _carry_round_block(L, dim, a.QB, a.QB, RN)))
        for c in range(RN):
            blocks.append((f"alu-div-sub{it}-{c}", _div_sub_nibble_block(L, dim, c)))
        blocks.append((f"alu-div-r2r{it}", _div_r2_to_r_block(L, dim)))
    blocks.append(("alu-div-finalize", _div_finalize_block(L, dim)))
    return blocks


# ===========================================================================
# 4. AX MULTIPLEXER : opcode-gated writeback of the selected op's 8 result
#    nibbles into the AX nibble bands.  Each op wrote its result into its own
#    scratch band (ADD_RES/SUB_RES/MUL_RES/DIV_RES/MOD_RES) UNCONDITIONALLY, so
#    the blocks never interfere; this ONE block picks the active op by OP_IS[op]
#    and copies its nibbles into AX (SET: -old AX nibble + gated result nibble).
#    Also PC += 1 and SP += 4 (the popped operand consumed one stack slot) — the
#    housekeeping the base experts do — gated on any ALU op being active.
# ===========================================================================
_ALU_RESULT = {
    isa.ADD: "ADD_RES", isa.SUB: "SUB_RES", isa.MUL: "MUL_RES",
    isa.DIV: "DIV_RES", isa.MOD: "MOD_RES",
    # SHIFT-VIA-MUL/DIV: SHL reuses the MUL product (x * 2**n), SHR the DIV quotient
    # (x // 2**n).  The pow2-route block put 2**n in AX, so MUL_RES / DIV_RES already
    # hold the shift result; the mux copies it into AX gated on OP_IS[SHL/SHR].
    isa.SHL: "MUL_RES", isa.SHR: "DIV_RES",
}


def compile_ax_mux(L, dim, ops=None) -> Dict[str, torch.Tensor]:
    """Write AX_nib[c] = sum_op OP_IS[op] * RES[op][c]  for c=0..7  (SET), for the
    ALU ops in ``ops`` (default all five).  The old AX nibble is cleared gated on
    'any ALU op active' so a non-ALU step leaves AX untouched.

    When ``ops`` includes SHL/SHR (the shift-via-mul path) their result comes from
    MUL_RES / DIV_RES respectively (the shift computed by the reused MUL/DIV gadget
    with the pow2-routed operand), so the SAME mux delivers it into AX."""
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    if ops is None:
        ops = [isa.ADD, isa.SUB, isa.MUL, isa.DIV, isa.MOD]
    spec = _empty_spec(dim, 8 * (len(ops) + len(ops)) + 4)
    u = 0
    for c in range(8):
        # clear old AX nibble c, gated on each op (so it clears once per active op).
        for op in ops:
            g = (L.OP_IS + op, 1.0, 0.0)
            u = _guard(spec, u, [g], {L.AX + c: -1.0}, 0.0, L.AX + c, 1.0)
            band = getattr(a, _ALU_RESULT[op])
            u = _guard(spec, u, [g], {band + c: 1.0}, 0.0, L.AX + c, 1.0)
    return _truncate(spec, u, dim)


# ===========================================================================
# 5. PSH (32-bit) : copy the full AX 8 nibbles into STACK0, opcode-gated on PSH.
#    The base PSH expert copies AX_VAL -> STK_VAL (scalar, byte only); this makes
#    the pushed operand carry the full 32-bit value so the ALU ops read a 32-bit
#    STACK0 operand.  (SET STACK0 low 8 nibbles, gated.)
# ===========================================================================
def compile_psh_nibble_copy(L, dim) -> Dict[str, torch.Tensor]:
    global _ONE
    _ONE = L.ONE
    spec = _empty_spec(dim, 8 * 2)
    u = 0
    g = (L.OP_IS + isa.PSH, 1.0, 0.0)
    for c in range(8):
        u = _guard(spec, u, [g], {L.STACK0 + c: -1.0}, 0.0, L.STACK0 + c, 1.0)  # clear
        u = _guard(spec, u, [g], {L.AX + c: 1.0}, 0.0, L.STACK0 + c, 1.0)       # = AX
    return _truncate(spec, u, dim)


# ===========================================================================
# 6. SHIFT-VIA-MUL/DIV : retire the ~20K barrel shifter (§Shifts).
#
#   SHL x, n = (x * 2**n) & 0xFFFFFFFF   -> route through the NATIVE MUL gadget
#   SHR x, n = x // 2**n   (logical)     -> route through the NATIVE DIV gadget
#
# The shift amount ``n`` lives in AX (the popped ``x`` is in STACK0).  ONE tiny
# power-of-two block, gated on SHL|SHR, OVERWRITES the AX nibble band with ``2**n``
# (a ~33-entry table: n=0..31 -> 2**n, n>=32 -> 0), so the MUL products block (reads
# STACK0=A, AX=B) then computes ``x * 2**n`` INTO MUL_RES and the DIV blocks compute
# ``x // 2**n`` INTO DIV_RES — no new VM steps, the shift REUSES the mul/div blocks
# already applied every forward (in-step reuse).  The ax-mux copies MUL_RES -> AX on
# SHL and DIV_RES -> AX on SHR.  This replaces the ~14.7K-nz barrel-shifter select
# (``perbit_shift_select_rules``) with a ~few-hundred-weight pow2 table + operand
# routing.  Requires the MUL (and DIV, for SHR) gadget present; behind
# ``shift_via_mul`` (default True when the subset has muldiv) with the barrel shifter
# as the muldiv-less fallback.
# ===========================================================================
def compile_shift_onehot(L, dim) -> Dict[str, torch.Tensor]:
    """Build the EXACT-count one-hot ``SH_N_OH[s] = [n == s]`` (s=0..31) of the shift
    amount ``n`` (AX low byte = ``AX_nib0 + 16*AX_nib1``): the point indicator
    ``[n>=s] - [n>=s+1]``.  Opcode-INDEPENDENT and cheap; only the (opcode-gated)
    pow2-route write consumes it.

    Split from the pow2-route write into its OWN, PRIOR block because every FFN unit
    reads the block INPUT: the write's guard on ``SH_N_OH[s]`` must see it already
    materialised.  Over the EXACT count (not ``n mod 32``): for ``n >= 32`` no cell
    fires, so the pow2 route writes 0 (whole 32-bit word shifted out -> 0)."""
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    n_form = {L.AX + 0: 1.0, L.AX + 1: 16.0}   # AX low byte = shift count n (0..255).
    # per s: _clear (1) + two _step_ge (2 relu units each) = 5 units.
    spec = _empty_spec(dim, 32 * 5 + 4)
    u = 0
    for s in range(32):
        oh = a.SH_N_OH + s
        u = _clear(spec, u, oh)
        u = _step_ge(spec, u, n_form, 0.0, s, oh, 1.0)
        u = _step_ge(spec, u, n_form, 0.0, s + 1, oh, -1.0)
    return _truncate(spec, u, dim)


def compile_shift_pow2_route(L, dim) -> Dict[str, torch.Tensor]:
    """Overwrite AX with ``2**n`` (n = shift amount in AX), gated on SHL|SHR.

    Reads the pre-built one-hot ``SH_N_OH[s] = [n == s]`` (``compile_shift_onehot``, a
    PRIOR block) and, gated on SHL|SHR, SETs the AX nibble band to ``2**s`` via the
    §520 one-hot select over the 33-entry power-of-two table ``{s: 2**s}``.  For
    ``n >= 32`` no cell fires and the AX nibbles stay cleared to 0 — the correct "whole
    word shifted out" result (``x*0`` / ``x//0`` -> 0, ISA §Shifts / §Division b==0).
    Every unit reads the block input, so clearing AX (SET) then adding ``2**n`` composes
    to a clean SET of ``2**n`` over the AX nibbles the MUL (SHL) / DIV (SHR) blocks read
    NEXT.  This ~few-hundred-weight table + operand route REPLACES the ~14.7K-nz
    barrel-shifter select."""
    from .blogspec_vocab import nibbles_of_value
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    # unit budget: SHL|SHR each: 8 AX-clear + <=32*8 one-hot pow2 nibble writes.
    spec = _empty_spec(dim, 2 * (8 + 32 * 8) + 4)
    u = 0
    # gated (SHL|SHR) AX := 2**n : clear the 8 AX nibbles, then add 2**s's nibbles for
    # the (single) active one-hot.  n >= 32 -> no one-hot -> AX stays 0 (word shifted out).
    for op in (isa.SHL, isa.SHR):
        g = (L.OP_IS + op, 1.0, 0.0)
        for c in range(8):
            u = _guard(spec, u, [g], {L.AX + c: -1.0}, 0.0, L.AX + c, 1.0)      # clear AX[c]
        for s in range(32):
            oh = (a.SH_N_OH + s, 1.0, 0.0)
            for c, nv in enumerate(nibbles_of_value(1 << s, 8)):
                if nv:
                    u = _guard(spec, u, [g, oh], {L.ONE: float(nv)}, 0.0, L.AX + c, 1.0)
    return _truncate(spec, u, dim)
