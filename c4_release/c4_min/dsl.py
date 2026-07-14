"""c4_min compact DSL: FFNRule + AttentionSpec + LinearExpr.

The MINIMUM to express the ISA. See DESIGN.md (d).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LinearExpr:
    """dst += (sum coeff*src_band) + const, evaluated when the rule fires."""
    terms: Dict[int, float] = field(default_factory=dict)  # band -> coeff
    const: float = 0.0

    @staticmethod
    def of(band: int, coeff: float = 1.0) -> "LinearExpr":
        return LinearExpr(terms={band: coeff})

    @staticmethod
    def c(const: float) -> "LinearExpr":
        return LinearExpr(const=const)

    def __add__(self, other: "LinearExpr") -> "LinearExpr":
        t = dict(self.terms)
        for b, c in other.terms.items():
            t[b] = t.get(b, 0.0) + c
        return LinearExpr(terms=t, const=self.const + other.const)


# A guard window: band value must lie in [lo, hi]. Used to gate a rule.
Window = Tuple[int, float, float]  # (band, lo, hi)


@dataclass
class FFNRule:
    """When ALL windows hold, add each write-expr into its destination band."""
    when: List[Window]
    write: Dict[int, LinearExpr]


@dataclass
class AttentionSpec:
    """Copy value band ``v`` from the position whose ``k`` band matches ``q``.

    Written into ``dst`` at the query position. ``alibi_slope`` biases toward
    nearer source positions (0 = content-only match).
    """
    q_band: int
    k_band: int
    v_band: int
    dst_band: int
    gain: float = 30.0        # sharpness of the softmax match
    alibi_slope: float = 0.0
    # optional constant offset added to the query key (to match k = q + offset)
    q_offset: float = 0.0
