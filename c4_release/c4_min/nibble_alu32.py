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

from typing import Dict, List, Tuple

import torch

from . import isa
from .nibble_vm import S, RELU_S, SILU_S, SILU_HALF, _empty_spec

# silu at the AND-gate operating point silu(0.5*S) ~= 30.
_SILU_G = SILU_HALF
# the layout's ONE lane, set by the builders before emitting (keeps emitters terse).
_ONE = None


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
    all units read the block INPUT and write deltas.  SET semantics."""
    for c in range(n):
        u = _clear(spec, u, dst + c)                                                # SET -old
        u = _ident(spec, u, {src + c: 1.0}, 0.0, dst + c, 1.0)                       # + col
        u = _floor_div_pow(spec, u, {src + c: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, -16.0)  # - 16*floor
        if c > 0:
            u = _floor_div_pow(spec, u, {src + c - 1: 1.0}, 0.0, 16, _NIB_KMAX, dst + c, 1.0)  # + carry
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
                 shift_via_mul: bool = False):
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


def extend_layout_for_alu32(L, recurrent_divmod: bool = False,
                            shift_via_mul: bool = False):
    """Allocate ALU-32 scratch bands on ``L`` and refresh ``L.D`` (pad to heads).

    ``recurrent_divmod`` adds the digit-index counter band the reused
    division-iteration block reads/threads; OFF (default) is byte-identical to
    the historical unrolled build (same dim, same weights).

    ``shift_via_mul`` (default False, so every EXISTING caller is byte-identical)
    adds the shift-amount one-hot band used to route SHL/SHR through the NATIVE
    MUL/DIV gadgets (``2**n`` power-of-two table) instead of the barrel shifter;
    the qwen_full_vm FULL build passes True (muldiv+bitwise), everything else OFF."""
    if getattr(L, "ALU32", None) is not None:
        return L.ALU32
    L.ALU32 = ALU32Bands(L, recurrent_divmod=recurrent_divmod,
                         shift_via_mul=shift_via_mul)
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


def compile_mul_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    global _ONE
    _ONE = L.ONE
    a = L.ALU32
    blocks = [("alu-mul-products", _mul_products_block(L, dim)),
              ("alu-mul-split", _mul_split_block(L, dim))]
    # carry rounds settle every column to a single nibble.  A carry ripples one
    # column per round; split partials keep columns < 256 so _MUL_CARRY_ROUNDS
    # (verified worst-case + headroom) fully settle it, kmax=15 each -> tiny.
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
def _kb_precompute_blocks(L, dim) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Compute KB[k] = k*b nibbles for k=1..15 from the AX (=B) nibble bands, plus
    the divisor-zero predicate BZ.  ``k*b`` = (single-nibble k) * (8-nibble b): a
    per-nibble multiply (b_j*k <= 15*15 = 225 < 256) laid into 9 raw columns, then
    carry-normalised in place with ``kmax=15`` (every column stays ``< 256``).

    Each KB[k] is carried in its OWN tiny block (RN rounds, in place) so no block
    is wide — the uniform-hidden budget is set by the small per-k carry, NOT by a
    fused 15-k round.  This is the fix for the recovered draft's 75k-unit blocks."""
    a = L.ALU32
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

    unique = prefix + body + finalize                     # DISTINCT specs to store
    prefix_names = [n for n, _ in prefix]
    body_names = [n for n, _ in body]
    # application order: prefix (once) | body x n_iters | finalize (once).  The
    # body names REPEAT so the loop reuses the shared iteration weights n_iters x.
    apply_names = list(prefix_names)
    for _ in range(n_iters):
        apply_names += body_names
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
