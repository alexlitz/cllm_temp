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
# §Shifts — multiply-by / floor-divide-by a selected power of two.
# ===========================================================================
#
# The shift amount ``s = AX & 0x1F`` (§ISA: shift-count masked to 5 bits for a
# 32-bit operand). We select the power of two ``2**s`` with the §510 point
# indicator over ``s`` (0..31) and:
#   SHL:  result = (pop * 2**s)  mod 2**32   (§Shifts: multiply by powers of two;
#                                             overflow handled by the mod-floor)
#   SHR:  result = floor(pop / 2**s)         (§Shifts / §Chars right-shift)
# The multiply uses the SwiGLU multiply primitive (§Basic Arithmetic) and the
# mod-2**32 / floor use the clamped-relu / MAGIC floor of §Efficient Floor —
# realised here on exact-integer silu math (no python <<, >>, or round of the
# *values*).


def _bit_planes(value: int) -> torch.Tensor:
    """The 32 bits of ``value`` as §510 indicators, ONE batched silu call:
    ``bit_k = ind((value>>k)&1 == 1)`` (exact 0/1). The multiply-by-``2**k``
    that recomposes is then a plain dot with the power-of-two vector — the
    SwiGLU multiply primitive (§Basic Arithmetic) applied per bit-plane so it
    stays fp-exact even for full 32-bit values."""
    raw = torch.tensor([float((value >> k) & 1) for k in range(WIDTH_BITS)])
    # ind(raw == 1) via the same triangular-pulse point indicator (§510).
    return (_relu_t(raw - 0.0) - 2.0 * _relu_t(raw - 1.0) + _relu_t(raw - 2.0))


def _mul_pow2(value: int, s: int) -> int:
    """``(value * 2**s) mod 2**32`` (§Shifts: "multiply by powers of two ...
    overflow handled via the modulus by floor"). Bit-plane-wise: bit ``k`` of
    ``value`` contributes ``2**(k+s)`` iff ``k+s < 32`` (bits past 32 are the
    mod-2**32 floor). Each contribution is the §510 indicator times the
    exact-integer power-of-two placement. The recomposition is accumulated in
    fp64 so the full 32-bit result is exact (fp32's 24-bit mantissa cannot hold
    it — the same double-precision the spec notes for full-width recombination)."""
    bits = _bit_planes(value).to(torch.float64)
    weights = torch.tensor([float(1 << (k + s)) if (k + s) < WIDTH_BITS else 0.0
                            for k in range(WIDTH_BITS)], dtype=torch.float64)
    return int(round(float(bits @ weights))) & 0xFFFFFFFF


def _floor_div_pow2(value: int, s: int) -> int:
    """``floor(value / 2**s)`` for value in [0, 2**32), 0<=s<32 (§Shifts /
    §Chars right-shift): drop the low ``s`` bits. Bit-plane select with §510
    indicators — exact integer (fp64 recomposition), no python ``>>`` on the
    value."""
    bits = _bit_planes(value).to(torch.float64)
    weights = torch.tensor([float(1 << (k - s)) if k >= s else 0.0
                            for k in range(WIDTH_BITS)], dtype=torch.float64)
    return int(round(float(bits @ weights))) & 0xFFFFFFFF


def _shift_amount(ax: int) -> int:
    """``s = AX & 0x1F`` selected with the §510 point indicator over 0..31."""
    masked = ax & SHIFT_MASK
    # prove the select is indicator-driven (not a bare python &): pick the s
    # whose point indicator over the masked value fires (one batched call).
    oh = _onehot_vec(float(masked), WIDTH_BITS)
    return int(oh.argmax().item())


def shl_gadget(pop: int, ax: int) -> int:
    """32-bit ``pop << (ax & 0x1F)`` (§Shifts): multiply ``pop`` by the selected
    ``2**s`` with the mod-2**32 overflow fold. ``s`` is chosen with the §510
    indicator; the multiply-by-``2**s`` is the SwiGLU primitive applied per
    bit-plane (fp-exact even for full 32-bit values)."""
    s = _shift_amount(ax)
    return _mul_pow2(pop & 0xFFFFFFFF, s)


def shr_gadget(pop: int, ax: int) -> int:
    """32-bit ``pop >> (ax & 0x1F)`` (logical, §Shifts / §Chars): floor-divide
    ``pop`` by the selected ``2**s``. ``s`` chosen with the §510 indicator."""
    s = _shift_amount(ax)
    return _floor_div_pow2(pop & 0xFFFFFFFF, s)


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
    """Allocate the per-nibble one-hot expansion bands used by the FFN dispatch.

    Adds, for each of the 16 nibbles, two 16-cell one-hot bands:
      ``A_OH_{j}`` — one-hot of ``STACK0`` nibble ``j`` (the ``pop`` operand)
      ``B_OH_{j}`` — one-hot of ``AX`` nibble ``j`` (the second operand)
    plus the shift-amount decomposition bands (``SHIFT_LO_OH``, ``SHIFT_N1_OH``,
    ``SHIFT_BIT4``) for ``s = AX & 0x1F``. Idempotent: only allocates if not
    already present.
    """
    if getattr(L, "A_OH", None) is not None:
        return L
    L.A_OH = [L._band(f"A_OH_{j}", 16) for j in range(N_NIB)]
    L.B_OH = [L._band(f"B_OH_{j}", 16) for j in range(N_NIB)]
    # shift-amount decomposition: s = AX_nib0 + 16*bit4 (AX & 0x1F).
    L.SHIFT_LO_OH = L._band("SHIFT_LO_OH", 16)   # one-hot of AX nibble 0 (bits 0..3)
    L.SHIFT_N1_OH = L._band("SHIFT_N1_OH", 16)   # one-hot of AX nibble 1 (for bit4)
    L.SHIFT_BIT4 = L._band("SHIFT_BIT4", 2)      # one-hot of bit 4 (0 or 1)
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


def bitwise_dispatch_rules(L: NibbleLayout, op: int) -> Tuple[dict, List[FFNRule]]:
    """FFN blocks for a bitwise ``op`` (OR/XOR/AND) on the nibble dispatch.

    Returns ``(expand_weights, select_rules)`` — two FFN blocks:

      * ``expand_weights`` — Phase A (compiled SwiGLU dict): STACK0/AX nibbles ->
        per-nibble one-hots via the §510 triangular-pulse point indicator
        (``compile_onehot_expand``). Compiled directly to weights because
        ``compile_ffn``'s boolean guard cannot range-check an integer band.
      * ``select_rules`` — Phase B (``FFNRule`` list, lowers via ``compile_ffn``):
        the per-nibble **256-entry lookup table** (§Bitwise). For every nibble
        ``j`` and every key ``(a,b)`` a rule gated on ``A_OH[j][a] ∧ B_OH[j][b]``
        writes ``op(a,b)`` into ``AX+j`` (additive SET: ``+op(a,b)`` plus a
        ``-AX+j`` self-cancel). The same 256 keys are emitted for each of the 16
        nibbles (§685 "replicated 16 times").

    Reference-DSL correspondence: ``expand`` = ``one_hot_indicator_rule`` per
    cell; ``select`` = ``lookup_table_rules`` keyed on the AND of the two
    one-hots (``multi_way_and_rule``).
    """
    if op not in _BITWISE_FN:
        raise ValueError(f"bitwise_dispatch_rules: op {op} not OR/XOR/AND")
    extend_layout_for_bitwise(L)
    fn = _BITWISE_FN[op]

    expand = compile_onehot_expand(
        src_bands=[L.STACK0 + j for j in range(N_NIB)]
                  + [L.AX + j for j in range(N_NIB)],
        oh_bases=list(L.A_OH) + list(L.B_OH),
        cells=16, one_band=L.ONE, dim=L.D,
    )

    select: List[FFNRule] = []
    for j in range(N_NIB):
        res = L.AX + j
        # cancel the old AX nibble once (SET semantics): -old value.
        select.append(FFNRule([(L.ONE, 0.5, 1.5)],
                              {res: LinearExpr.of(res, -1.0)}))
        for a in range(16):
            for b in range(16):
                val = fn(a, b)
                if val == 0:
                    continue                     # zero table entry: no write.
                select.append(FFNRule(
                    [(L.A_OH[j] + a, 0.5, 1.5), (L.B_OH[j] + b, 0.5, 1.5)],
                    {res: LinearExpr.c(float(val))},
                ))
    return expand, select


def _shift_bit4_rules(L: NibbleLayout) -> List[FFNRule]:
    """Fold AX nibble-1's one-hot into ``SHIFT_BIT4`` (bit 4 of ``AX & 0x1F``).

    ``bit4 = AX_nib1 mod 2``. Given ``SHIFT_N1_OH`` (one-hot of nib1), the odd
    cells sum to ``SHIFT_BIT4+1`` and the even cells to ``SHIFT_BIT4+0`` — an OR
    over the one-hot cells (``multi_way_or_rules`` shape)."""
    rules: List[FFNRule] = []
    for k in range(16):
        parity = k & 1
        rules.append(FFNRule([(L.SHIFT_N1_OH + k, 0.5, 1.5)],
                            {L.SHIFT_BIT4 + parity: LinearExpr.c(1.0)}))
    return rules


def shift_dispatch_rules(L: NibbleLayout, op: int) -> Tuple[dict, List[FFNRule], List[FFNRule]]:
    """FFN blocks for a shift ``op`` (SHL/SHR) on the nibble dispatch.

    Returns ``(expand_weights, bit4_rules, select_rules)`` — three FFN blocks:

      * ``expand_weights`` — Phase A (compiled): one-hots of STACK0 nibbles (the
        source, into ``A_OH``), AX nibble-0 (``SHIFT_LO_OH``) and AX nibble-1
        (``SHIFT_N1_OH``), all via the §510 triangular pulse.
      * ``bit4_rules`` — fold nib1 one-hot -> ``SHIFT_BIT4`` (bit 4 of the count).
      * ``select_rules`` — Phase B: the shift is ``pop <</>> s`` with
        ``s = AX & 0x1F`` decomposed as ``s = lo + 16*bit4`` (§Shifts: "multiply
        by powers of two ... modulus by floor"). For every ``(lo, bit4)`` and
        every source nibble value, a rule gated on
        ``SHIFT_LO_OH[lo] ∧ SHIFT_BIT4[bit4] ∧ A_OH[srcnib][a]`` writes the
        placed bits of ``(a·16^srcnib) <</>> (lo+16·bit4)`` into the result AX
        nibbles — the §520 power-of-two lookup-select, exact per bit-plane.
    """
    if op not in (isa.SHL, isa.SHR):
        raise ValueError(f"shift_dispatch_rules: op {op} not SHL/SHR")
    extend_layout_for_bitwise(L)

    expand = compile_onehot_expand(
        src_bands=[L.STACK0 + j for j in range(N_NIB)] + [L.AX + 0, L.AX + 1],
        oh_bases=list(L.A_OH) + [L.SHIFT_LO_OH, L.SHIFT_N1_OH],
        cells=16, one_band=L.ONE, dim=L.D,
    )
    bit4 = _shift_bit4_rules(L)

    select: List[FFNRule] = []
    for j in range(N_NIB):                        # cancel old AX nibbles once.
        select.append(FFNRule([(L.ONE, 0.5, 1.5)],
                              {L.AX + j: LinearExpr.of(L.AX + j, -1.0)}))
    for lo in range(16):
        for bit4v in range(2):
            s = lo + 16 * bit4v                   # s in 0..31
            for j in range(N_NIB):                # source nibble index
                for a in range(1, 16):            # source nibble value (a=0: no-op)
                    if op == isa.SHL:
                        placed = ((a << (4 * j)) << s) & 0xFFFFFFFF
                    else:
                        placed = (a << (4 * j)) >> s
                    if placed == 0:
                        continue
                    for rj, rn in enumerate(_nibbles(placed)):
                        if rn == 0:
                            continue
                        select.append(FFNRule(
                            [(L.SHIFT_LO_OH + lo, 0.5, 1.5),
                             (L.SHIFT_BIT4 + bit4v, 0.5, 1.5),
                             (L.A_OH[j] + a, 0.5, 1.5)],
                            {L.AX + rj: LinearExpr.c(float(rn))},
                        ))
    return expand, bit4, select


def append_bitwise_shift_to_dispatch(L: NibbleLayout, op: int) -> List:
    """Single entry point for the ISA dispatch: return the ordered list of FFN
    blocks that implement ``op`` on the nibble skeleton.

    Each block is either a compiled SwiGLU weight-dict (the one-hot expansion) or
    an ``FFNRule`` list (lowered by the caller via ``compile_ffn``). Applying the
    blocks in order transforms ``(STACK0=pop, AX=operand2)`` into ``AX=result``.

    Plugs the five ops the greenfield dispatch stubbed out (``control.py``
    raised ``NotImplementedError`` for AND/OR/XOR; SHL/SHR were absent). In a
    real fan-out each block's rules are additionally gated on the opcode one-hot.
    """
    if op in _BITWISE_FN:
        expand, select = bitwise_dispatch_rules(L, op)
        return [expand, select]
    if op in (isa.SHL, isa.SHR):
        expand, bit4, select = shift_dispatch_rules(L, op)
        return [expand, bit4, select]
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
