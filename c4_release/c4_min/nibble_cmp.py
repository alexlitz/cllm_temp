"""Spec-faithful NIBBLE comparison gadgets — EQ / NE / LT / GT / LE / GE
(BLOG_SPEC §Comparisons 573-590 + §Building Blocks 504-518).

Everything reduces to **one primitive**: a zero detector on ``d = a - b``
(§576). Following the spec exactly:

    "The c4 opcodes EQ, LT, GT, BZ, BNZ can all be reduced to a single
     primitive: detecting whether a difference is zero. ... The remaining
     comparisons are built from the sign of a - b together with EQ."

so this file exposes the 6 comparison ops as functions over the substrate's
**16-nibble operand bands** (``blogspec_layout`` — dim ``REG+j`` holds nibble
``j`` of a 32-bit value, little-endian):

    EQ(a,b) = Z(a-b)                 (zero detector)
    NE(a,b) = 1 - EQ(a,b)
    LT(a,b) = sign(a-b < 0)          (most-significant differing nibble)
    GT(a,b) = sign(a-b > 0)
    LE(a,b) = LT(a,b)  OR  EQ(a,b)
    GE(a,b) = GT(a,b)  OR  EQ(a,b)

The output is a **0/1 boolean** (destined for AX nibble 0). Every op is
computed from SiLU/sigmoid nodes only — no python ``==``/``<`` on the values,
no ``round`` (mirrors ``blogspec_compiler``'s exec-path discipline).

The zero detector (§578-586)
----------------------------
Three SiLU nodes sharing a common ``SCALE`` and ``EPS``:

    Node 1: in-weight SCALE, bias +SCALE·EPS, out-weight +1/k
    Node 2: in-weight SCALE, bias  0,         out-weight -2/k
    Node 3: in-weight SCALE, bias -SCALE·EPS, out-weight +1/k
    k = SCALE·EPS·(2σ(SCALE·EPS) - 1)

    Z(d) = [silu(SCALE·d + SCALE·EPS)
            - 2·silu(SCALE·d)
            + silu(SCALE·d - SCALE·EPS)] / k

The +1/-2/+1 pattern is a finite second difference: far from 0 the three
SiLUs are in their linear regime and cancel (Z≈0); at 0 the curvature breaks
the cancellation to give a sharp bump with **Z(0)=1** exactly. 11 parameters
(9 weights + 2 biases), all shared across nibble groups.

4-nibble packing (§590)
-----------------------
Rather than one detector per nibble, four nibble-differences are packed into a
single detector by scaling the more-significant nibbles and summing:

    P = Σ_{i=0..3} BASE^i · (a_j - b_j)          (j = group·4 + i)

with ``BASE = 31 > 2·15``, so a packed value is exactly zero **iff every**
component nibble-diff is zero (no cross-cancellation is possible since each
diff is in [-15, 15]). Biases and output weights are shared across the four
groups; the 16-nibble EQ is the AND (product, §538 gate-multiply) of the 4
group detectors.

Sign / ordering (§588)
----------------------
``a - b`` cannot be taken as one fp32 subtraction across 32 bits (mantissa is
24 bits), so the sign is recovered lexicographically from the **most-significant
nibble at which a and b differ** — the standard nibble comparison cascade. Per
nibble ``j`` (scanning MS→LS): ``gt_j = step(a_j-b_j ≥ 1)``,
``lt_j = step(b_j-a_j ≥ 1)`` (smoothed Heaviside via sigmoid), and the running
``eq_above`` product (the zero detector on each higher nibble, exactly 1 when
equal) gates so only the first differing nibble contributes. This is unsigned
32-bit ordering, matching ``isa.interpret``'s masked comparisons.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

# ---------------------------------------------------------------------------
# Shared building-block constants (BLOG_SPEC §578-590, §504-506).
# ---------------------------------------------------------------------------
SCALE = 60.0     # common detector/step scale (as blogspec_compiler.SCALE)
EPS = 0.5        # detector half-width ε; integer diffs land cleanly outside ±ε
PACK_BASE = 31   # 4-nibble packing radix; > 2·15 ⇒ no cancellation (§590)
NIB_PER_REG = 16 # a value is 16 little-endian nibbles (§569-571)


def _sigmoid(x: float) -> float:
    return float(torch.sigmoid(torch.as_tensor(x, dtype=torch.float64)))


def _silu(x) -> torch.Tensor:
    """SiLU in fp64 (the detector's linear-regime cancellation needs the extra
    mantissa when the packed input is large — §590 fp64 remark)."""
    return torch.nn.functional.silu(torch.as_tensor(x, dtype=torch.float64))


# ---------------------------------------------------------------------------
# §578-586 — the zero detector Z(d)  (the single primitive)
# ---------------------------------------------------------------------------
def zero_detector(d: float, S: float = SCALE, eps: float = EPS) -> float:
    """The +1/-2/+1 finite-second-difference zero indicator: Z(0)=1, Z(d≠0)≈0.

    Direct transcription of BLOG_SPEC §584-586. 11 parameters: input weights
    ``S`` (×3), biases ``+Sε, 0, -Sε``, output weights ``+1/k, -2/k, +1/k``.
    """
    k = S * eps * (2.0 * _sigmoid(S * eps) - 1.0)
    z = (_silu(S * d + S * eps)
         - 2.0 * _silu(S * d)
         + _silu(S * d - S * eps)) / k
    return float(z)


# ---------------------------------------------------------------------------
# §504-506 — smoothed Heaviside step (used for the per-nibble sign test)
# ---------------------------------------------------------------------------
def _step_ge1(d: float, S: float = SCALE) -> float:
    """1.0 iff integer ``d`` ≥ 1, else 0.0 — a clean 0/1 gate (sigmoid at 0.5,
    the spec's smoothed step). For nibble diffs d ∈ {-15..15} this is exact."""
    return _sigmoid(S * (d - 0.5))


# ---------------------------------------------------------------------------
# Operand access: a "16-nibble operand band" is just a length-16 nibble list.
# ---------------------------------------------------------------------------
def nibbles_of_value(v: int, n: int = NIB_PER_REG) -> List[int]:
    """The ``n`` little-endian 4-bit nibbles of a 32-bit value (the operand-band
    encoding the dispatch feeds in)."""
    return [(v >> (4 * j)) & 0xF for j in range(n)]


def _as_nibbles(operand: Sequence[int]) -> List[int]:
    """Accept a raw int (convenience) or an already-nibbled length-16 band."""
    if isinstance(operand, int):
        return nibbles_of_value(operand)
    nb = list(operand)
    if len(nb) != NIB_PER_REG:
        raise ValueError(f"operand band must be {NIB_PER_REG} nibbles, got {len(nb)}")
    return nb


# ---------------------------------------------------------------------------
# EQ — zero detector on a-b with 4-nibble packing (§590)
# ---------------------------------------------------------------------------
def eq_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """EQ(a,b) = Z(a-b) over all 16 nibbles, done **4 nibbles at a time** and
    AND-combined (§590). Returns ~1.0 iff a==b else ~0.0.

    Packs each group of 4 nibble-diffs into one detector input
    ``P = Σ 31^i·(a_j-b_j)`` (zero iff the group matches), then multiplies the
    4 group detectors (§538: the SwiGLU gate multiplies two [0,1] indicators).
    """
    an, bn = _as_nibbles(a), _as_nibbles(b)
    result = 1.0
    for group in range(NIB_PER_REG // 4):
        packed = 0.0
        for i in range(4):
            j = group * 4 + i
            packed += (PACK_BASE ** i) * (an[j] - bn[j])
        result *= zero_detector(float(packed), S)
    return result


def ne_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """NE(a,b) = 1 - EQ(a,b)  (§588)."""
    return 1.0 - eq_gadget(a, b, S)


# ---------------------------------------------------------------------------
# LT / GT — sign of a-b via the most-significant differing nibble (§588)
# ---------------------------------------------------------------------------
def _sign_cascade(a: Sequence[int], b: Sequence[int], S: float = SCALE):
    """Return ``(lt, gt, eq)`` ∈ [0,1]³ from the nibble comparison cascade.

    Scan MS→LS. ``eq_above`` (product of per-nibble zero detectors on the
    higher nibbles, exactly 1.0 while they match) gates each nibble so only the
    **first** differing nibble decides. ``lt``/``gt`` accumulate that nibble's
    verdict; whatever ``eq_above`` survives to the end is the all-equal case.
    """
    an, bn = _as_nibbles(a), _as_nibbles(b)
    eq_above = 1.0
    lt = 0.0
    gt = 0.0
    for j in range(NIB_PER_REG - 1, -1, -1):
        d = an[j] - bn[j]
        gt_j = _step_ge1(d, S)          # a_j > b_j
        lt_j = _step_ge1(-d, S)         # a_j < b_j
        lt += eq_above * lt_j
        gt += eq_above * gt_j
        eq_above *= zero_detector(float(d), S)   # 1.0 iff a_j==b_j
    return lt, gt, eq_above


def lt_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """LT(a,b) = sign(a-b < 0), unsigned 32-bit (§588)."""
    return _sign_cascade(a, b, S)[0]


def gt_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """GT(a,b) = sign(a-b > 0), unsigned 32-bit (§588)."""
    return _sign_cascade(a, b, S)[1]


# ---------------------------------------------------------------------------
# LE / GE — LT∨EQ, GT∨EQ  (§588)
# ---------------------------------------------------------------------------
def _bool_or(x: float, y: float) -> float:
    """Boolean OR of two [0,1] indicators: x + y - x·y (§538 gate multiply
    handles the product). For disjoint indicators (LT xor EQ) this is just the
    sum, but the full form is robust to any residual overlap."""
    return x + y - x * y


def le_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """LE(a,b) = LT(a,b) ∨ EQ(a,b)  (§588)."""
    lt, _gt, eq = _sign_cascade(a, b, S)
    return _bool_or(lt, eq)


def ge_gadget(a: Sequence[int], b: Sequence[int], S: float = SCALE) -> float:
    """GE(a,b) = GT(a,b) ∨ EQ(a,b)  (§588)."""
    _lt, gt, eq = _sign_cascade(a, b, S)
    return _bool_or(gt, eq)


# ---------------------------------------------------------------------------
# Dispatch plug-in: name -> gadget, and a boolean-to-AX-nibbles packer.
#
# The nibble-skeleton agent owns the MoE opcode dispatch (§566). Its contract
# (per blogspec_compiler / the ISA) is: given the two live 16-nibble operand
# bands (pop and AX), produce the next AX value. A comparison op's result is a
# 0/1 boolean written to AX. So the dispatch calls ``compare(op_name, pop, ax)``
# and lays the returned bit into the AX nibble band via ``bool_to_ax_nibbles``
# (bit in nibble 0, all higher nibbles 0) — exactly the shape
# ``blogspec_compiler``'s emit head decodes.
# ---------------------------------------------------------------------------
CMP_GADGETS = {
    "EQ": eq_gadget,
    "NE": ne_gadget,
    "LT": lt_gadget,
    "GT": gt_gadget,
    "LE": le_gadget,
    "GE": ge_gadget,
}


def compare(op_name: str, a: Sequence[int], b: Sequence[int],
            S: float = SCALE) -> float:
    """Dispatch entry point: run comparison ``op_name`` on operand bands ``a``
    (pop) and ``b`` (AX). Returns the raw [0,1] indicator (≈0 or ≈1)."""
    try:
        gadget = CMP_GADGETS[op_name]
    except KeyError:
        raise NotImplementedError(f"{op_name} is not a comparison op") from None
    return gadget(a, b, S)


def to_bit(indicator: float) -> int:
    """Quantize a [0,1] indicator to a hard 0/1 bit (the boolean AX writes)."""
    return 1 if indicator > 0.5 else 0


def bool_to_ax_nibbles(indicator: float) -> List[int]:
    """Lay a comparison result into a 16-nibble AX band: the boolean bit in
    nibble 0, all higher nibbles 0 — the AX shape the emit head decodes."""
    bit = to_bit(indicator)
    nb = [0] * NIB_PER_REG
    nb[0] = bit
    return nb


__all__ = [
    "SCALE", "EPS", "PACK_BASE", "NIB_PER_REG",
    "zero_detector",
    "eq_gadget", "ne_gadget", "lt_gadget", "gt_gadget", "le_gadget", "ge_gadget",
    "CMP_GADGETS", "compare", "to_bit", "bool_to_ax_nibbles",
    "nibbles_of_value",
]
