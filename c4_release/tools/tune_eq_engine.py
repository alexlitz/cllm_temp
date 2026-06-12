#!/usr/bin/env python3
"""Offline threshold tuner for the per-nibble EQ engine (NO model build).

Uses the probed AX-row operand values for eq_true (5==5) and eq_false (5!=7)
to find condition weights + threshold for the 4-way AND so that:
  - the TRUE (h=0,l=5) unit FIRES for eq_true
  - the SAME unit does NOT fire for eq_false (AXC_LO+5 absent there)
  - NO spurious unit fires (esp. the index-0 artifact units (h=0,l=0))
The AND term sum is `sum_i w_i * value_i`; the unit fires iff sum >= threshold.
The OP_EQ gate is multiplicative (always 1 here since both are EQ).
"""

# Probed AX-row operand bands (value at each nibble index). 0 if absent.
EQ_TRUE = {
    "ALU_HI":  {0: 11.38, 15: 0.47},
    "AXC_HI":  {0: 1.26},
    "ALU_LO":  {0: 5.39, 5: 6.0, 8: 0.45},
    "AXC_LO":  {0: 0.32, 5: 0.94},
}
EQ_FALSE = {
    "ALU_HI":  {0: 11.38, 15: 0.47},
    "AXC_HI":  {0: 1.26},
    "ALU_LO":  {0: 5.39, 5: 5.99, 8: 0.45},
    "AXC_LO":  {0: 0.32, 7: 0.94},
}
MARK_AX = 1.0  # marker value at AX row

def g(state, band, idx):
    return state.get(band, {}).get(idx, 0.0)

def unit_sum(state, h, l, w):
    """AND sum for unit (h,l): MARK_AX + ALU_HI+h + AXC_HI+h + ALU_LO+l + AXC_LO+l"""
    return (
        w["mark"] * MARK_AX
        + w["alu_hi"] * g(state, "ALU_HI", h)
        + w["axc_hi"] * g(state, "AXC_HI", h)
        + w["alu_lo"] * g(state, "ALU_LO", l)
        + w["axc_lo"] * g(state, "AXC_LO", l)
    )

def scan(w, thr):
    """Return (eqtrue_fired_units, eqfalse_fired_units) as lists of (h,l)."""
    out = {}
    for label, state in (("eq_true", EQ_TRUE), ("eq_false", EQ_FALSE)):
        fired = []
        for h in range(16):
            for l in range(16):
                if unit_sum(state, h, l, w) >= thr:
                    fired.append((h, l))
        out[label] = fired
    return out

def evaluate(w):
    """For weights w, return (true_unit_sum, max_nonfire_sum, margin, thr).
    A GOOD design: eq_true fires ONLY (0,5); eq_false fires NOTHING.
    The threshold sits in the gap between true(0,5) and the highest
    non-target unit sum across BOTH programs."""
    st_true = unit_sum(EQ_TRUE, 0, 5, w)
    # every non-(0,5) unit in eq_true + every unit in eq_false must NOT fire.
    nonfire = []
    for h in range(16):
        for l in range(16):
            if (h, l) != (0, 5):
                nonfire.append(unit_sum(EQ_TRUE, h, l, w))
            nonfire.append(unit_sum(EQ_FALSE, h, l, w))
    ceiling = max(nonfire)
    margin = st_true - ceiling
    thr = (st_true + ceiling) / 2.0
    return st_true, ceiling, margin, thr

def main():
    # The 4-way AND must REQUIRE all four nibble matches. ALU is unreliable
    # (artifact at +0 ~= true at +5), so the AND cannot lean on ALU alone;
    # but it also cannot lean on AXC_LO alone (else (0,7) fires for eq_false).
    # Scan a grid; pick the largest-margin weighting where eq_true fires only
    # (0,5) and eq_false fires nothing.
    best = None
    import itertools
    # Wider search emphasising the AXC_LO discriminator (0.32 artifact vs 0.94
    # true) to MAXIMISE the firing margin so build-to-build FP nondeterminism
    # (~0.4) cannot flip the true unit's firing.
    for alu_w in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2):
        for axc_lo_w in (0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0):
            for axc_hi_w in (0.4, 0.6, 0.8, 1.0, 1.2):
                for mark_w in (0.1, 0.3, 0.5, 0.8, 1.0):
                    w = dict(mark=mark_w, alu_hi=alu_w, axc_hi=axc_hi_w,
                             alu_lo=alu_w, axc_lo=axc_lo_w)
                    st_true, ceiling, margin, thr = evaluate(w)
                    if margin <= 0:
                        continue
                    res = scan(w, thr)
                    if res["eq_true"] == [(0, 5)] and res["eq_false"] == []:
                        # symmetric margin = min(true-thr, thr-ceiling) maximised
                        sym = min(st_true - thr, thr - ceiling)
                        if best is None or sym > best[0]:
                            best = (sym, w, st_true, ceiling, thr, margin)
    if best:
        sym, w, st_true, ceiling, thr, margin = best
        print(f"BEST symmetric_margin={sym:.3f} total_margin={margin:.3f}")
        print(f"  weights={w}")
        print(f"  true(0,5)={st_true:.3f} ceiling(nonfire)={ceiling:.3f} thr={thr:.3f}")
        print(f"  headroom: true above thr = {st_true-thr:.3f}; thr above ceiling = {thr-ceiling:.3f}")
        res = scan(w, thr)
        print(f"  @thr={thr:.3f}: eq_true fires {res['eq_true']}  eq_false fires {res['eq_false']}")
    else:
        print("NO clean separating weighting found in grid.")

if __name__ == "__main__":
    main()
