"""NIBBLE bitwise + shift gadgets (BLOG_SPEC §Bitwise operations, §Shifts).

This is the additive op-fan-out layer the greenfield foundation
(``docs/GREENFIELD_BLOGSPEC_FOUNDATION_2026_07_14.md``) was designed to carry:
the five ops ``OR XOR AND SHL SHR`` realised directly on the spec's **16 4-bit
nibble** representation (``blogspec_layout.NibbleLayout``), plugging into the
nibble-skeleton's dispatch interface — *two 16-nibble operands → a 16-nibble
result written to AX*.

Where the operands live (matching ``blogspec_run._apply_op``):

    STACK0 band  = the popped operand  (``pop``)
    AX     band  = the second operand  (the accumulator, ``AX``)

and the result is written back over the ``AX`` nibble band.

Spec authority
--------------
* **Bitwise** (§683-685): "Bitwise operations, AND, OR, XOR ... just lookup
  tables per nibble embedded in the FFNs ... since they are binary operations
  and each nibble is 4 bits there need to be **256 entries** ... the per nibble
  operations/tables are each **replicated 16 times**." We build exactly that: a
  256-entry table per nibble, one table per each of the 16 nibbles. The table is
  addressed by the one-hot-indicator → table-select pattern of §504-568
  (``building_blocks_dsl.one_hot_indicator_rule`` + ``lookup_table_rules``):
  each operand nibble (0..15) is expanded to a one-hot, the pair one-hots are
  AND-ed to form the 256-way key, and that key selects the result nibble.

* **Shifts** (§597-615): "Left and right shifts could simply be handled by 32
  different multiplications by powers of two ... overflow can be handled via the
  modulus by floor." So SHL = ``(pop * 2**s) mod 2**32`` and SHR =
  ``floor(pop / 2**s)``; the **shift amount is ``AX & 0x1F``** (§ISA table:
  ``SHL/SHR  AX = pop <</>> AX``, C's shift-count masked to 5 bits for a 32-bit
  type). All arithmetic runs through the SwiGLU multiply / clamped-relu-floor
  primitives (§Basic Arithmetic, §Efficient Floor) — no python ``*``/``>>``/
  ``round`` on the values, exactly as ``nibble_add_gadget`` avoids python ``+``.

Two surfaces
------------
1. **Standalone gadgets** (``or_gadget`` / ``xor_gadget`` / ``and_gadget`` /
   ``shl_gadget`` / ``shr_gadget``): pure spec-math over two 32-bit operands via
   their 16-nibble decomposition. Byte-exact vs the python reference; swept in
   ``test_nibble_bitwise.py`` without needing the full model dispatch.

2. **FFN dispatch rules** (``bitwise_dispatch_rules`` / ``shift_dispatch_rules``
   + ``append_bitwise_shift_to_dispatch``): c4_min ``FFNRule`` lists on a
   ``NibbleLayout`` that lower through ``compile_ffn`` to real SwiGLU weights and
   plug into the ISA dispatch. These realise the SAME one-hot-indicator →
   table-select (bitwise) and power-of-two-select (shift) construction the
   standalone gadgets compute, so the two agree by construction.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from . import isa
from .dsl import FFNRule, LinearExpr
from .blogspec_layout import NibbleLayout, NIB_PER_REG


# ===========================================================================
# Spec building-block scalars (shared with blogspec_compiler / control).
# ===========================================================================
RELU_S = 200.0   # relu-via-silu scale: silu(RELU_S*z)/RELU_S ~= max(0,z), exact on ints.
SCALE = 60.0     # silu-identity scale: silu(60)~=60, silu(-60)~=0.

WIDTH_BITS = 32
N_NIB = NIB_PER_REG          # 16 nibbles per 32-bit register (§569-571).
SHIFT_MASK = WIDTH_BITS - 1  # AX & 0x1F — C's 32-bit shift-count mask.
LOG_STAGES = 5               # log2(32): shift amounts 1,2,4,8,16 — one stage per n-bit.


def _relu_t(z: torch.Tensor) -> torch.Tensor:
    """Exact-integer ReLU via silu (BLOG_SPEC §ReLU), vectorised: at RELU_S=200
    the fp result rounds to the exact integer for integer ``z``."""
    return F.silu(RELU_S * z) / RELU_S


def _relu(z: float) -> float:
    return float(_relu_t(torch.tensor(float(z))))


def _point_indicator(x: float, i: int) -> float:
    """Unit point indicator (BLOG_SPEC §510) as an integer-exact triangular
    pulse: ``relu(x-(i-1)) - 2 relu(x-i) + relu(x-(i+1))`` — 1.0 iff ``x == i``
    (integer), else 0.0. This is the discrete realisation the reference DSL's
    ``one_hot_indicator_rule`` (§510) lowers to."""
    return _relu(x - (i - 1)) - 2.0 * _relu(x - i) + _relu(x - (i + 1))


def _onehot_vec(x: float, cells: int) -> torch.Tensor:
    """One-hot of integer ``x`` over ``[0, cells)`` as a vector of §510 point
    indicators, computed in ONE batched silu call (the same triangular-pulse
    math as ``_point_indicator``, just vectorised for speed)."""
    idx = torch.arange(cells, dtype=torch.float32)
    xt = torch.tensor(float(x))
    return (_relu_t(xt - (idx - 1)) - 2.0 * _relu_t(xt - idx)
            + _relu_t(xt - (idx + 1)))


def _nibbles(v: int) -> List[int]:
    """The 16 little-endian 4-bit nibbles of a 32-bit value."""
    return [(v >> (4 * j)) & 0xF for j in range(N_NIB)]


def _from_nibbles(nibs: List[float]) -> int:
    """Reassemble a 32-bit int from 16 (exact-integer) nibble values."""
    v = 0
    for j, nv in enumerate(nibs):
        v |= (int(round(nv)) & 0xF) << (4 * j)
    return v & 0xFFFFFFFF


# ===========================================================================
# §Bitwise — per-nibble 256-entry lookup table (one-hot-indicator → select).
# ===========================================================================
#
# For a binary bitwise op ``⊕`` (∈ {OR, XOR, AND}) and nibble position ``j``:
#
#     result_j = Σ_{a=0}^{15} Σ_{b=0}^{15}  ind(A_j = a) · ind(B_j = b) · (a ⊕ b)
#
# The double sum is the 256-entry lookup table for nibble ``j``: ``ind(A_j=a)``
# and ``ind(B_j=b)`` are the §510 one-hot indicators, their product is the
# §520 one-hot key ``(a,b)``, and ``a ⊕ b`` is the baked table value. Since
# exactly one (a,b) key is active, the sum collapses to the single table entry.
# The identical table is applied to every one of the 16 nibbles (§685
# "replicated 16 times").

_BITWISE_FN: Dict[int, Callable[[int, int], int]] = {
    isa.OR:  lambda a, b: a | b,
    isa.XOR: lambda a, b: a ^ b,
    isa.AND: lambda a, b: a & b,
}


def bitwise_nibble_table(op: int) -> Dict[Tuple[int, int], int]:
    """The 256-entry per-nibble table ``{(a,b): a⊕b}`` for a bitwise ``op``.

    Same table for every nibble (§685). ``a``/``b`` ∈ 0..15, result ∈ 0..15.
    """
    fn = _BITWISE_FN[op]
    return {(a, b): fn(a, b) for a in range(16) for b in range(16)}


_TABLE_CACHE: Dict[int, torch.Tensor] = {}


def _table_tensor(op: int) -> torch.Tensor:
    """The 16×16 baked table ``T[a][b] = a⊕b`` as a tensor (cached). Selecting
    the entry is ``a_onehot · T · b_onehot`` — the §520 lookup-select."""
    if op not in _TABLE_CACHE:
        fn = _BITWISE_FN[op]
        T = torch.tensor([[float(fn(a, b)) for b in range(16)]
                          for a in range(16)])
        _TABLE_CACHE[op] = T
    return _TABLE_CACHE[op]


def _bitwise_gadget(op: int, a_val: int, b_val: int) -> int:
    """32-bit ``a ⊕ b`` via the per-nibble 256-entry lookup (spec math).

    For each of the 16 nibbles the operand nibbles are expanded to §510 one-hot
    indicators and the result nibble is ``a_onehot · T · b_onehot`` — the
    §520 table-select over the 256-entry table ``T`` (§685, same table per
    nibble). NO python bitwise operator is applied to the *values*.
    """
    T = _table_tensor(op)
    an, bn = _nibbles(a_val & 0xFFFFFFFF), _nibbles(b_val & 0xFFFFFFFF)
    out: List[float] = []
    for j in range(N_NIB):
        a_oh = _onehot_vec(float(an[j]), 16)         # §510 point indicators
        b_oh = _onehot_vec(float(bn[j]), 16)
        out.append(float(a_oh @ T @ b_oh))           # §520 table select
    return _from_nibbles(out)


def or_gadget(a: int, b: int) -> int:
    """32-bit ``a | b`` via the per-nibble 256-entry OR lookup (§Bitwise)."""
    return _bitwise_gadget(isa.OR, a, b)


def xor_gadget(a: int, b: int) -> int:
    """32-bit ``a ^ b`` via the per-nibble 256-entry XOR lookup (§Bitwise)."""
    return _bitwise_gadget(isa.XOR, a, b)


def and_gadget(a: int, b: int) -> int:
    """32-bit ``a & b`` via the per-nibble 256-entry AND lookup (§Bitwise)."""
    return _bitwise_gadget(isa.AND, a, b)


# ===========================================================================
# §Shifts — a bitwise LOG-SHIFTER (5 conditional 2^k stages), NOT a barrel
# select and NOT shift-via-mul.
# ===========================================================================
#
# A shift by ``n`` is 5 conditional stages, each gated on one bit of ``n``:
#
#     for k in 0..4:                       # shift amounts 1, 2, 4, 8, 16
#         if n_bit_k:  x = x shifted by 2**k   (left for SHL, right for SHR)
#
# so the whole shift is ``O(bits × log2(32)) = 5`` stages instead of the full
# per-(source-bit × shift-amount) select.  Each stage is a per-output-bit 2:1
# MUX between "same bit" and "the bit 2**k away", gated on ``n``'s bit k — built
# from the SAME boolean AND/OR machinery the bitwise ops use.  ``n`` is the AX
# operand and its bits are exactly the AX bit-planes ``B_BIT``:
#     n_bit_0..4  =  bits 0..4 of AX  =  ``B_BIT[0..4]``.
# Because every value is a 0/1 bit-plane, the whole shifter is fp32-EXACT — no
# scalar 32-bit value, no MUL, no DIV, no fp64.
#
# Direction is the ONLY difference between SHL and SHR (SHR is the logical /
# unsigned right shift matching ``isa.interpret`` / ``ref_interpret``).
#
# ``n >= 32`` -> 0: the 5 stages consume ``n``'s low 5 bits (``n & 0x1F``); when
# ANY higher bit of ``n`` is set (``n >= 32``) the result is forced all-zero, so
# ``pop << n`` / ``pop >> n`` matches the full-width shift of ``ref_interpret``
# (``(pop <</>> ax) & 0xFFFFFFFF``, ax UNMASKED) rather than a bare 5-bit mask.


def _bit_planes(value: int) -> torch.Tensor:
    """The 32 bits of ``value`` as §510 indicators, ONE batched silu call:
    ``bit_k = ind((value>>k)&1 == 1)`` (exact 0/1)."""
    raw = torch.tensor([float((value >> k) & 1) for k in range(WIDTH_BITS)])
    # ind(raw == 1) via the same triangular-pulse point indicator (§510).
    return (_relu_t(raw - 0.0) - 2.0 * _relu_t(raw - 1.0) + _relu_t(raw - 2.0))


def _log_shift_stage(planes: torch.Tensor, n_bit: float, k: int,
                     left: bool) -> torch.Tensor:
    """One conditional log stage over the 32 source ``planes`` (0/1 bit tensor).

    Per output bit ``i`` a 2:1 mux picks between the SAME bit and the bit
    ``2**k`` away, gated on ``n_bit`` (bit ``k`` of the shift amount):

        left  (SHL):  out[i] = n_bit ? in[i - 2**k] : in[i]
        right (SHR):  out[i] = n_bit ? in[i + 2**k] : in[i]

    with the neighbour treated as 0 when it falls outside ``[0, 32)`` (bits
    shifted in are zero).  The mux is the NOT-free boolean identity
    ``out = same + n_bit*(neighbour - same)`` — pure 0/1 arithmetic, fp-exact.
    """
    shift = 1 << k
    out = torch.zeros(WIDTH_BITS)
    for i in range(WIDTH_BITS):
        same = float(planes[i])
        src = (i - shift) if left else (i + shift)
        neigh = float(planes[src]) if 0 <= src < WIDTH_BITS else 0.0
        out[i] = same + n_bit * (neigh - same)
    return out


def _n_bits(ax: int) -> Tuple[List[float], float]:
    """The 5 low shift-amount bits ``n_bit_0..4`` (= AX bits 0..4) as exact 0/1,
    plus ``keep = ind(n < 32)`` (1.0 iff every AX bit at position >= 5 is 0).
    All computed from the §510 AX bit-planes — no python ``&`` on the value."""
    planes = _bit_planes(ax & 0xFFFFFFFF)
    n_bit = [float(planes[k]) for k in range(LOG_STAGES)]
    hi = float(planes[LOG_STAGES:].sum())            # # of set bits at pos >= 5
    keep = float(_point_indicator(hi, 0))            # 1.0 iff hi == 0  (n < 32)
    return n_bit, keep


def _log_shift(pop: int, ax: int, left: bool) -> int:
    """The bitwise log-shifter: ``pop`` shifted by ``ax`` (SHL if ``left`` else
    SHR), via the 5 conditional 2**k stages over the source bit-planes, forced to
    0 when ``ax >= 32``.  Pure 0/1 bit math, fp32-exact."""
    planes = _bit_planes(pop & 0xFFFFFFFF)
    n_bit, keep = _n_bits(ax)
    for k in range(LOG_STAGES):
        planes = _log_shift_stage(planes, n_bit[k], k, left)
    planes = planes * keep                           # n >= 32 -> all-zero
    # recompose the 32 result bits to a 32-bit int (exact-integer place values).
    val = 0
    for i in range(WIDTH_BITS):
        if int(round(float(planes[i]))) & 1:
            val |= (1 << i)
    return val & 0xFFFFFFFF


def shl_gadget(pop: int, ax: int) -> int:
    """32-bit ``pop << ax`` (§Shifts) via the bitwise LOG-SHIFTER: 5 conditional
    left-by-2**k stages gated on the AX bits, ``ax >= 32 -> 0``.  Matches
    ``ref_interpret``'s ``(pop << ax) & 0xFFFFFFFF`` (unsigned, full-width)."""
    return _log_shift(pop, ax, left=True)


def shr_gadget(pop: int, ax: int) -> int:
    """32-bit ``pop >> ax`` (logical, §Shifts / §Chars) via the bitwise
    LOG-SHIFTER: 5 conditional right-by-2**k stages gated on the AX bits,
    ``ax >= 32 -> 0``.  Matches ``ref_interpret``'s ``(pop >> ax) & 0xFFFFFFFF``."""
    return _log_shift(pop, ax, left=False)


# ===========================================================================
# FFN DISPATCH RULES — the one-hot-indicator → table-select realised as
# c4_min FFNRules on a NibbleLayout (lower via compile_ffn to SwiGLU weights).
# ===========================================================================
#
# These are the *dispatch plug-in*: real FFN rules that read the operand nibble
# bands and write the result to AX. They embody the SAME construction as the
# reference building_blocks_dsl.one_hot_indicator_rule (expand) +
# lookup_table_rules (select), but typed to c4_min's light FFNRule/compile_ffn
# so they run on the greenfield nibble skeleton.
#
# Layout requirement: the caller extends NibbleLayout with per-nibble one-hot
# expansion bands for both operands (``extend_layout_for_bitwise``). The FFN
# stack is two phases:
#   Phase A (expand): STACK0+j (0..15) -> A_OH[j][0..15];  AX+j -> B_OH[j][0..15]
#                     via the §510 triangular-pulse point indicator.
#   Phase B (select): for each key (a,b) a rule gated on A_OH[j][a] ∧ B_OH[j][b]
#                     writes (op(a,b) - AX+j) into AX+j  (additive SET).


def extend_layout_for_bitwise(L: NibbleLayout) -> NibbleLayout:
    """Allocate the SHARED bit-plane bands the per-bit bitwise combine and the
    log-shifter both read/write.

    * ``A_BIT``/``B_BIT`` — the 4 bit-planes of each of the 16 STACK0 (A) and AX
      (B) operand nibbles as exact 0/1 indicators (band ``j*4+p`` = bit ``p`` of
      nibble ``j``).  OR/XOR/AND combine these; the log-shifter reads the SOURCE
      planes ``A_BIT`` and takes the shift amount ``n`` straight from the AX
      planes ``B_BIT[0..4]`` (= bits 0..4 of AX).
    * ``SH_STAGE`` — the log-shifter pipeline: ``LOG_STAGES`` fresh 32-plane
      buffers, one per conditional 2**k stage (stage 0's input is ``A_BIT``,
      stage ``k``'s output is ``SH_STAGE[k]``).
    * ``SH_KEEP`` — the ``ind(n < 32)`` bit (result is forced 0 when ``n >= 32``).

    The DENSE per-operand one-hot bands (``A_OH``/``B_OH``, 512 dims) and the
    shift-amount one-hots (``SHIFT_LO_OH``/``SHIFT_N1_OH``/``SHIFT_BIT4``) are NO
    LONGER allocated — the bit-planes serve both the bitwise combine and the
    log-shifter.  Idempotent: only allocates if not already present.  ``A_OH`` /
    ``B_OH`` are kept as ``None`` attributes for the legacy "per-bit path" checks.
    """
    if getattr(L, "A_BIT", None) is not None:
        return L
    L.A_OH = None            # dense operand one-hots retired (log-shifter path).
    L.B_OH = None
    # per-nibble bit PLANES for the shared per-bit OR/XOR/AND gadget AND the
    # log-shifter source: for each of the 16 nibbles, the 4 bits of the STACK0 (A)
    # and AX (B) operand nibble as exact 0/1 indicators.  ONE extraction serves
    # OR/XOR/AND (combine) and SHL/SHR (source + shift-amount bits).
    L.A_BIT = L._band("A_BIT", N_NIB * 4)        # A_BIT[j*4+p] = bit p of STACK0 nib j
    L.B_BIT = L._band("B_BIT", N_NIB * 4)        # B_BIT[j*4+p] = bit p of AX     nib j
    # log-shifter pipeline: one fresh 32-plane buffer per conditional 2**k stage.
    L.SH_STAGE = [L._band(f"SH_STAGE_{k}", WIDTH_BITS) for k in range(LOG_STAGES)]
    L.SH_KEEP = L._scalar("SH_KEEP")             # ind(n < 32) — n>=32 zeroes result.
    L.D = L._off
    return L


def compile_onehot_expand(src_bands: List[int], oh_bases: List[int],
                          cells: int, one_band: int, dim: int) -> dict:
    """Compile the §510 point-indicator one-hot expansion as SwiGLU weights.

    For each ``(src, oh_base)`` pair, materialise ``oh_base + a = ind(src == a)``
    for ``a`` in ``[0, cells)`` via the integer-exact triangular pulse
    ``tri_a(x) = relu(x-(a-1)) - 2 relu(x-a) + relu(x-(a+1))`` — the discrete
    realisation of the blog's ``+1/-2/+1`` point indicator (§510), 1.0 iff
    ``src == a`` else 0.0. Shared relu units per distinct threshold (as
    ``control.compile_pc_fetch`` does for the PC one-hot), because
    ``compile_ffn``'s boolean guard cannot range-check an *integer* band.

    ``compile_ffn`` guards are boolean-only, so this dedicated compiler is what
    turns the integer nibble bands into the one-hot keys the ``select`` lookup
    block consumes. Each expansion band is written as a fresh SET (the block is
    applied once per op, on a zero-initialised one-hot region).
    """
    assert len(src_bands) == len(oh_bases)
    # thresholds needed per source: relu(src - t) for t in [-1, cells].
    thresholds = list(range(-1, cells + 1))
    n_thr = len(thresholds)
    # one relu bank per source band (thresholds are shared across a's within it).
    n_units = n_thr * len(src_bands)
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)

    for si, (src, oh_base) in enumerate(zip(src_bands, oh_bases)):
        base_u = si * n_thr
        thr_unit = {t: base_u + j for j, t in enumerate(thresholds)}
        # up_j = RELU_S*(src - t); hidden_j = relu(src - t); gate = 1 (via ONE).
        for t, u in thr_unit.items():
            W_up[u, src] = RELU_S
            b_up[u] = -RELU_S * t
            W_gate[u, one_band] = 1.0
        # oh_base+a = relu(src-(a-1)) - 2 relu(src-a) + relu(src-(a+1)); /RELU_S.
        for a in range(cells):
            W_down[oh_base + a, thr_unit[a - 1]] += 1.0 / RELU_S
            W_down[oh_base + a, thr_unit[a]] += -2.0 / RELU_S
            W_down[oh_base + a, thr_unit[a + 1]] += 1.0 / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def compile_bit_extract(src_bands: List[int], bit_bases: List[int],
                        one_band: int, dim: int) -> dict:
    """Compile the per-nibble BIT-PLANE extraction as SwiGLU weights (shared, ungated).

    For each ``(src, bit_base)`` pair (``src`` a nibble band 0..15) materialise the
    four bit indicators ``bit_base + p = (src >> p) & 1`` for ``p`` in 0..3.
    Bit ``p`` is the integer-exact ``floor(src/2**p) - 2*floor(src/2**(p+1))``.
    Each ``floor(src/2**p)`` is a staircase of relu steps (one relu per unit rise,
    exactly like ``compile_fold`` / the divmod floor); the two floors share their
    relu banks across the two consecutive planes, so the whole extraction is a
    handful of relu units per nibble — a fraction of the 256-entry (a,b) tables it
    replaces. Purely a function of the operand nibble (no opcode gate), so ONE copy
    serves OR, XOR and AND.
    """
    assert len(src_bands) == len(bit_bases)
    # floor(v/2**p) for a nibble v in 0..15 rises 1 at v=2**p, 2*2**p, ...  We build
    # each floor from unit-rise relu steps: floor(v/M) = sum_{m>=1} step(v >= m*M),
    # step(v>=t) = relu(v-(t-1)) - relu(v-t) (exact 0/1 on integers). For M=2**p and
    # v in 0..15 the thresholds needed are the multiples of M up to 15.
    def floor_terms(M):
        # returns {threshold: coeff} for the relu STEPS composing floor(v/M).
        terms = {}
        m = M
        while m <= 15:
            # step(v>=m) = relu(v-(m-1)) - relu(v-m)
            terms[m - 1] = terms.get(m - 1, 0.0) + 1.0
            terms[m] = terms.get(m, 0.0) - 1.0
            m += M
        return terms

    # bit p = floor(v/2**p) - 2*floor(v/2**(p+1)); collect the net relu-threshold
    # coefficients per source, then emit ONE relu unit per (src, distinct threshold).
    W_rows = []          # (src, threshold, coeff, bit_base+p) accumulation
    per_src_thr = []     # list over srcs of {threshold: {bit_idx: coeff}}
    for src, bit_base in zip(src_bands, bit_bases):
        thr_map: dict = {}
        for p in range(4):
            hi = floor_terms(1 << p)             # +floor(v/2**p)
            lo = floor_terms(1 << (p + 1))       # -2*floor(v/2**(p+1))
            net: dict = {}
            for t, c in hi.items():
                net[t] = net.get(t, 0.0) + c
            for t, c in lo.items():
                net[t] = net.get(t, 0.0) - 2.0 * c
            for t, c in net.items():
                if c == 0.0:
                    continue
                thr_map.setdefault(t, {})[bit_base + p] = \
                    thr_map.setdefault(t, {}).get(bit_base + p, 0.0) + c
        per_src_thr.append((src, thr_map))

    n_units = sum(len(thr_map) for _src, thr_map in per_src_thr) or 1
    W_up = torch.zeros(n_units, dim); b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim); b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units); b_down = torch.zeros(dim)
    u = 0
    for src, thr_map in per_src_thr:
        for t, dst_coeffs in thr_map.items():
            # up = RELU_S*(src - t); hidden = relu(src - t); gate = 1 (via ONE).
            W_up[u, src] = RELU_S
            b_up[u] = -RELU_S * t
            W_gate[u, one_band] = 1.0
            for dst, c in dst_coeffs.items():
                W_down[dst, u] += c / RELU_S
            u += 1
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


# the per-bit boolean combine for each of the three bitwise ops, expressed as
# additive contributions of the result bit (weight 2**p on the nibble):
#   AND(a,b) = a*b                       -> +[A_BIT ∧ B_BIT]
#   OR (a,b) = a + b - a*b               -> +[A_BIT] +[B_BIT] -[A_BIT ∧ B_BIT]
#   XOR(a,b) = a + b - 2*a*b             -> +[A_BIT] +[B_BIT] -2*[A_BIT ∧ B_BIT]
# where each [·] is an FFNRule gated on the listed bit-plane predicate(s).
_PERBIT_COMBINE = {
    isa.AND: [(("A", "B"), 1.0)],
    isa.OR:  [(("A",), 1.0), (("B",), 1.0), (("A", "B"), -1.0)],
    isa.XOR: [(("A",), 1.0), (("B",), 1.0), (("A", "B"), -2.0)],
}


def perbit_select_rules(L: NibbleLayout, op: int) -> List[FFNRule]:
    """Per-nibble OR/XOR/AND result via the SHARED bit-plane gadget (§Bitwise).

    Replaces the op's 256-entry ``(a,b)`` lookup with the boolean per-bit identity
    ``AND=a·b``, ``OR=a+b-a·b``, ``XOR=a+b-2a·b`` on the pre-extracted bit planes
    ``A_BIT``/``B_BIT`` (built once, shared across the three ops). For nibble ``j``
    each of the 4 bit planes ``p`` contributes ``2**p`` weighted by the boolean
    combine, gated on the bit-plane predicates. A single ``-AX+j`` self-cancel per
    nibble keeps SET semantics — identical result values to the table, but a couple
    dozen rules per nibble instead of 670. Bit-exact vs ``bitwise_dispatch_rules``.
    """
    if op not in _PERBIT_COMBINE:
        raise ValueError(f"perbit_select_rules: op {op} not OR/XOR/AND")
    rules: List[FFNRule] = []
    for j in range(N_NIB):
        res = L.AX + j
        # SET semantics: cancel the old AX nibble once (same as the table path).
        rules.append(FFNRule([(L.ONE, 0.5, 1.5)],
                             {res: LinearExpr.of(res, -1.0)}))
        for p in range(4):
            a_bit = L.A_BIT + j * 4 + p
            b_bit = L.B_BIT + j * 4 + p
            weight = float(1 << p)
            for operands, coeff in _PERBIT_COMBINE[op]:
                when = []
                if "A" in operands:
                    when.append((a_bit, 0.5, 1.5))
                if "B" in operands:
                    when.append((b_bit, 0.5, 1.5))
                rules.append(FFNRule(when, {res: LinearExpr.c(coeff * weight)}))
    return rules


# ---------------------------------------------------------------------------
# LOG-SHIFTER FFN rules — 5 conditional 2**k stages + an n>=32 keep bit +
# recompose.  These REPLACE the old barrel select (~17K nz) with ~6 shallow
# blocks (one mux stage each) reading the SHARED source bit-planes ``A_BIT`` and
# taking the shift amount straight from the AX bit-planes ``B_BIT[0..4]``.
# ---------------------------------------------------------------------------

def _stage_in_dim(L: NibbleLayout, k: int, i: int) -> int:
    """Residual dim of source bit ``i`` feeding log stage ``k`` (0..31).  Stage 0
    reads the SOURCE planes ``A_BIT`` (bit i = nibble i//4, plane i%4); stage k>0
    reads the previous stage's fresh 32-plane buffer ``SH_STAGE[k-1]``."""
    if k == 0:
        return L.A_BIT + (i // 4) * 4 + (i % 4)
    return L.SH_STAGE[k - 1] + i


def shift_keep_rules(L: NibbleLayout) -> List[FFNRule]:
    """Compute ``SH_KEEP = ind(n < 32)`` (n = AX): the log stages consume n's low
    5 bits; when ANY AX bit at position >= 5 is set (n >= 32) the result must be
    forced to 0.  ``keep = 1 - OR(high AX bits)``: seed ``+1`` unconditionally,
    then subtract ``1`` gated on EACH high AX bit-plane (``B_BIT`` bits 5..31 and,
    since B_BIT spans all 16 AX nibbles, every plane of AX nibbles >= 2 as well).
    At most one high bit fires for a plain shift-count, and any that do drive keep
    to 0 — exactly the ``n >= 32 -> 0`` fold.  (For arbitrary garbage AX with two
    high bits set, keep can go negative, but the recompose only WRITES when
    ``keep >= 1``, so <1 still zeroes the result.)"""
    rules: List[FFNRule] = [
        FFNRule([(L.ONE, 0.5, 1.5)], {L.SH_KEEP: LinearExpr.c(1.0)})
    ]
    # every AX bit-plane at global position >= 5 is a "high" bit (n >= 32 marker).
    for b in range(LOG_STAGES, N_NIB * 4):
        hi_plane = L.B_BIT + (b // 4) * 4 + (b % 4)
        rules.append(FFNRule([(hi_plane, 0.5, 1.5)],
                             {L.SH_KEEP: LinearExpr.c(-1.0)}))
    return rules


def log_shift_stage_rules(L: NibbleLayout, op: int, k: int) -> List[FFNRule]:
    """One conditional log stage (shift by ``2**k`` gated on AX bit ``k``).

    Per output bit ``i`` a 2:1 MUX picks between the SAME bit and the NEIGHBOUR
    ``2**k`` away, realised NOT-free as ``out = same + n_bit*(neigh - same)``:

        out[i] += same[i]                               (gated on same[i]==1)
        out[i] += neigh[i]        when n_bit ∧ neigh[i]==1
        out[i] -= same[i]         when n_bit ∧ same[i]==1

    ``n_bit = AX bit k = B_BIT[k]`` (bits 0..4 of AX are the shift amount).  The
    neighbour is ``i-2**k`` (SHL) or ``i+2**k`` (SHR), treated as 0 when out of
    ``[0, 32)`` (shifted-in zero — its rules are simply omitted).  Writes into the
    FRESH stage buffer ``SH_STAGE[k]`` (starts at 0), reading ``SH_STAGE[k-1]``
    (or ``A_BIT`` for stage 0).  Pure 0/1 boolean AND/OR machinery — fp32-exact.
    """
    if op not in (isa.SHL, isa.SHR):
        raise ValueError(f"log_shift_stage_rules: op {op} not SHL/SHR")
    left = (op == isa.SHL)
    shift = 1 << k
    n_bit = L.B_BIT + k                              # AX bit k = shift-amount bit k
    out_base = L.SH_STAGE[k]
    rules: List[FFNRule] = []
    for i in range(WIDTH_BITS):
        out = out_base + i
        same = _stage_in_dim(L, k, i)
        src = (i - shift) if left else (i + shift)
        neigh = _stage_in_dim(L, k, src) if 0 <= src < WIDTH_BITS else None
        # out += same          (unconditional copy when n_bit == 0).
        rules.append(FFNRule([(same, 0.5, 1.5)], {out: LinearExpr.c(1.0)}))
        # out -= same          (cancel the copy when n_bit == 1).
        rules.append(FFNRule([(n_bit, 0.5, 1.5), (same, 0.5, 1.5)],
                             {out: LinearExpr.c(-1.0)}))
        # out += neigh         (route the shifted-in bit when n_bit == 1).
        if neigh is not None:
            rules.append(FFNRule([(n_bit, 0.5, 1.5), (neigh, 0.5, 1.5)],
                                 {out: LinearExpr.c(1.0)}))
    return rules


def shift_recompose_rules(L: NibbleLayout, op: int) -> List[FFNRule]:
    """Recompose the final log-stage result bit-planes into the AX nibble band,
    gated on ``SH_KEEP`` (n < 32).

    SET semantics: one ``-AX+j`` self-cancel per nibble, then each result bit
    ``i`` of the last stage buffer ``SH_STAGE[LOG_STAGES-1]`` adds ``2**(i%4)``
    into result nibble ``AX + i//4`` — gated on ``result_bit ∧ SH_KEEP`` so an
    ``n >= 32`` shift leaves AX all-zero.  Reads only 0/1 planes; fp-exact."""
    if op not in (isa.SHL, isa.SHR):
        raise ValueError(f"shift_recompose_rules: op {op} not SHL/SHR")
    last = L.SH_STAGE[LOG_STAGES - 1]
    rules: List[FFNRule] = []
    for j in range(N_NIB):
        rules.append(FFNRule([(L.ONE, 0.5, 1.5)],
                             {L.AX + j: LinearExpr.of(L.AX + j, -1.0)}))
    for i in range(WIDTH_BITS):
        rj, rp = i // 4, i % 4
        rules.append(FFNRule(
            [(last + i, 0.5, 1.5), (L.SH_KEEP, 0.5, 1.5)],
            {L.AX + rj: LinearExpr.c(float(1 << rp))},
        ))
    return rules


def perbit_shift_select_rules(L: NibbleLayout, op: int) -> List[FFNRule]:
    """The complete LOG-SHIFTER rule list for SHL/SHR as ONE flat ``FFNRule``
    sequence (§Shifts).  This SUPERSEDES both the old barrel select AND
    shift-via-mul: ``keep`` bit, then the ``LOG_STAGES`` conditional 2**k stages,
    then the recompose — the mux stages are wired to read/write the ``SH_STAGE``
    pipeline buffers, so the caller must apply them as SEPARATE sequential FFN
    blocks (see ``append_bitwise_shift_to_dispatch`` / ``build_bitwise_blocks``).
    Kept as one entry point (the barrel select's old name) for callers that only
    need the rule list / a rule count; the shift is ~6 shallow blocks, no dense
    per-value table and no MUL/DIV dependency (works in a muldiv-less subset)."""
    if op not in (isa.SHL, isa.SHR):
        raise ValueError(f"perbit_shift_select_rules: op {op} not SHL/SHR")
    rules: List[FFNRule] = list(shift_keep_rules(L))
    for k in range(LOG_STAGES):
        rules += log_shift_stage_rules(L, op, k)
    rules += shift_recompose_rules(L, op)
    return rules


def shift_stage_blocks(L: NibbleLayout, op: int) -> List[List[FFNRule]]:
    """The log-shifter as an ORDERED list of per-block rule lists (the shift is a
    pipeline — each mux stage reads the previous stage's buffer, so the stages
    CANNOT be one FFN block).  Order: keep bit, ``LOG_STAGES`` mux stages, then
    the keep-gated recompose.  ``LOG_STAGES + 2`` blocks."""
    blocks: List[List[FFNRule]] = [list(shift_keep_rules(L))]
    for k in range(LOG_STAGES):
        blocks.append(log_shift_stage_rules(L, op, k))
    blocks.append(shift_recompose_rules(L, op))
    return blocks


# NOTE: the DENSE per-nibble 256-entry (a,b) bitwise lookup table
# (``bitwise_dispatch_rules``) and the DENSE per-value shift lookup table
# (``shift_dispatch_rules``) have been REMOVED, as has the intermediate BARREL
# SELECT (the per-(source-bit × shift-amount) cross-product, ~17K nz).  The
# per-bit gadget (``perbit_select_rules`` for OR/XOR/AND) and the LOG-SHIFTER
# (the 5 conditional 2**k stages for SHL/SHR) — both reading the shared
# ``compile_bit_extract`` bit-planes — are the SOLE bitwise/shift path and were
# proven byte-exact to ``ref_interpret`` / ``isa.interpret`` before removal.


def append_bitwise_shift_to_dispatch(L: NibbleLayout, op: int) -> List:
    """Single entry point for the ISA dispatch: return the ordered list of FFN
    blocks that implement ``op`` on the nibble skeleton.

    Each block is either a compiled SwiGLU weight-dict (the one-hot / bit-plane
    expansions) or an ``FFNRule`` list (lowered by the caller via ``compile_ffn``).
    Applying the blocks in order transforms ``(STACK0=pop, AX=operand2)`` into
    ``AX=result``.

    ALL five ops (OR/XOR/AND/SHL/SHR) use the per-BIT path — the shared bit-plane
    extraction (``compile_bit_extract`` -> ``A_BIT``/``B_BIT``) plus a tiny per-bit
    combine (OR/XOR/AND via ``perbit_select_rules``) or the LOG-SHIFTER
    (SHL/SHR via the 5 conditional 2**k stages, ``shift_stage_blocks``).  There is
    NO dense per-value lookup table and NO barrel select any more; the shift also
    has no MUL/DIV dependency (works in a muldiv-less bitwise subset).  In a real
    fan-out each block's rules are additionally gated on the opcode one-hot.
    """
    extend_layout_for_bitwise(L)
    # Shared bit-plane extraction of the source (STACK0) and operand-2 (AX) nibbles.
    # For the shifter this ALSO yields the shift amount: AX bits 0..4 = B_BIT[0..4].
    planes = compile_bit_extract(
        src_bands=[L.STACK0 + j for j in range(N_NIB)]
                  + [L.AX + j for j in range(N_NIB)],
        bit_bases=[L.A_BIT + j * 4 for j in range(N_NIB)]
                  + [L.B_BIT + j * 4 for j in range(N_NIB)],
        one_band=L.ONE, dim=L.D)
    if op in _PERBIT_COMBINE:                          # OR / XOR / AND
        return [planes, perbit_select_rules(L, op)]
    if op in (isa.SHL, isa.SHR):
        # bit-planes (source + shift amount), then the log-shifter pipeline
        # (keep bit + LOG_STAGES conditional 2**k mux stages + keep-gated recompose).
        return [planes] + shift_stage_blocks(L, op)
    raise ValueError(f"append_bitwise_shift_to_dispatch: unsupported op {op}")


# ===========================================================================
# Reference SwiGLU applier — run a dispatch block on a residual (test/demo).
# ===========================================================================
def _apply_swiglu(x: torch.Tensor, w: dict) -> torch.Tensor:
    """One SwiGLU FFN block over residual ``x`` (matches blogspec_model.FFN)."""
    up = F.linear(x, w["W_up"]) + w["b_up"]
    gate = F.linear(x, w["W_gate"]) + w["b_gate"]
    hidden = F.silu(up) * gate
    return x + F.linear(hidden, w["W_down"], w["b_down"])


def compile_dispatch(L: NibbleLayout, blocks: List) -> List[dict]:
    """Compile an ``append_bitwise_shift_to_dispatch`` block list ONCE into a
    list of SwiGLU weight dicts (compile the rule blocks via ``compile_ffn``,
    pass the already-compiled expansion dicts through). Call this once, then run
    many operands through ``run_compiled`` — avoids re-lowering the (large) shift
    select block per sample."""
    from .compile_ffn import compile_ffn
    return [b if isinstance(b, dict) else compile_ffn(b, L.D) for b in blocks]


def run_compiled(L: NibbleLayout, weights: List[dict], pop: int,
                 operand2: int) -> int:
    """Run pre-compiled dispatch ``weights`` on a residual with ``STACK0 = pop``
    and ``AX = operand2``; return the decoded 32-bit AX (real SwiGLU path)."""
    from .blogspec_vocab import nibbles_of_value

    x = torch.zeros(L.D)
    x[L.ONE] = 1.0
    for j, nv in enumerate(nibbles_of_value(pop & 0xFFFFFFFF, N_NIB)):
        x[L.STACK0 + j] = float(nv)
    for j, nv in enumerate(nibbles_of_value(operand2 & 0xFFFFFFFF, N_NIB)):
        x[L.AX + j] = float(nv)

    for w in weights:
        x = _apply_swiglu(x, w)

    val = 0
    for j in range(N_NIB):
        val |= (int(round(float(x[L.AX + j]))) & 0xF) << (4 * j)
    return val & 0xFFFFFFFF


def run_dispatch(L: NibbleLayout, blocks: List, pop: int, operand2: int) -> int:
    """Compile ``blocks`` and run one operand pair (convenience for a single
    call). For many samples, prefer ``compile_dispatch`` once + ``run_compiled``.

    Applies the real SwiGLU weights (expansion blocks + ``compile_ffn``-lowered
    rule blocks) in sequence — the genuine FFN path, not the standalone gadget
    math — so a test can confirm the dispatch agrees with the reference.
    """
    return run_compiled(L, compile_dispatch(L, blocks), pop, operand2)
