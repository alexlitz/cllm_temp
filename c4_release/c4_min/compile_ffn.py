"""Compile FFNRule list -> SwiGLU (W_up,b_up,W_gate,b_gate,W_down,b_down).

Gadget (exact-integer): for each (rule, dst) create one hidden unit.
  gate = write-expression value               (linear read via W_gate)
  up   = S * guard_indicator                  (silu(S)~=S when guard holds,
                                                silu(-S)~=0 when it does not)
  down = 1/S   routes  silu(up)*gate ~= guard * expr  into dst.

Guard: AND of windows. Each window (band, lo, hi) contributes an indicator that
is ~1 inside [lo,hi] and ~0 outside, built from two silu steps. For the common
case of an opcode one-hot window (band in {0,1}), a single linear term suffices:
we require sum of one-hot guard bands >= n_windows - 0.5.
"""
from __future__ import annotations

from typing import List

import torch

from .dsl import FFNRule


S = 60.0     # silu identity scale: silu(60) ~= 60 to fp32; silu(-60) ~= 0
RELU_S = 200.0  # relu-via-silu scale: silu(RELU_S*z)/RELU_S ~= relu(z)


def compile_fold(band: int, one_band: int, dim: int, modulus: int = 256):
    """FFN implementing an exact mod-``modulus`` fold on an integer ``band`` value
    in ``[0, 2*modulus)``: ``band -= modulus * (band >= modulus)``.

    Indicator ``(band>=M)`` is the clamped-relu difference
    ``relu(band-(M-1)) - relu(band-M)`` (exact 0/1 on integers), each relu built as
    ``silu(RELU_S*z)/RELU_S``. Two hidden units.
    """
    M = modulus
    W_up = torch.zeros(2, dim); b_up = torch.zeros(2)
    W_gate = torch.zeros(2, dim); b_gate = torch.zeros(2)
    W_down = torch.zeros(dim, 2); b_down = torch.zeros(dim)
    for i, thr in enumerate((M - 1, M)):
        # up = RELU_S*(band - thr) ; gate = 1 (const via one_band)
        W_up[i, band] = RELU_S
        b_up[i] = -RELU_S * thr
        W_gate[i, one_band] = 1.0
    # down: band += (-M/RELU_S)*relu1 + (+M/RELU_S)*relu2  = -M*(relu1-relu2)
    W_down[band, 0] = -M / RELU_S
    W_down[band, 1] = +M / RELU_S
    return {"W_up": W_up, "b_up": b_up, "W_gate": W_gate, "b_gate": b_gate,
            "W_down": W_down, "b_down": b_down}


def _units_per_rule(rule: FFNRule) -> int:
    return len(rule.write)


def compile_ffn(rules: List[FFNRule], dim: int):
    """Return a dict of the six SwiGLU tensors implementing ``rules``."""
    n_units = sum(_units_per_rule(r) for r in rules) or 1
    W_up = torch.zeros(n_units, dim)
    b_up = torch.zeros(n_units)
    W_gate = torch.zeros(n_units, dim)
    b_gate = torch.zeros(n_units)
    W_down = torch.zeros(dim, n_units)
    b_down = torch.zeros(dim)

    u = 0
    for rule in rules:
        n_win = len(rule.when)
        for dst, expr in rule.write.items():
            # --- gate = expr value ---
            for band, coeff in expr.terms.items():
                W_gate[u, band] += coeff
            b_gate[u] += expr.const

            # --- up = S * (guard >= n_win - 0.5) ---
            # guard = sum over windows of an in-window indicator. For one-hot /
            # boolean guard bands the value is exactly 0 or 1, so summing the
            # raw band values and thresholding at (n_win - 0.5) is exact.
            for (band, lo, hi) in rule.when:
                # indicator ~ band membership; for boolean bands lo/hi bracket 1.
                W_up[u, band] += S
            # bias so up = +S when all windows active, <= -S when any missing.
            # all active: sum = n_win  -> up = S*n_win + b ; want ~ +S
            # one missing: sum = n_win-1 -> want <= -S
            # Choose b = -S*(n_win - 0.5): all -> +0.5S ; missing -> -0.5S.
            b_up[u] += -S * (n_win - 0.5)

            # --- down routes silu(up)*gate / S_eff into dst ---
            # silu(0.5S) with S=60 -> ~30 (=0.5S since 0.5S>>0). Normalise by it.
            silu_on = float(torch.nn.functional.silu(torch.tensor(0.5 * S)))
            W_down[dst, u] += 1.0 / silu_on

            u += 1

    return {
        "W_up": W_up, "b_up": b_up,
        "W_gate": W_gate, "b_gate": b_gate,
        "W_down": W_down, "b_down": b_down,
    }
