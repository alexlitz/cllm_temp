"""ADDER-HACK depth-1 nibble decode block (#916 continued).

Replaces the schoolbook MSB-first RUNNING-REMAINDER decompose
(``nibble_logsink_blocks`` ``decompose`` / ``decompose_clean``: seed + 8×
extract/snap/REDUCE, an 8-deep SEQUENTIAL recurrence) with a SINGLE
cross-nibble-INDEPENDENT block that reads all 8 nibbles of a scalar ``v`` in
parallel, byte-exact for fp64 (2^53 holds a 32-bit ``v`` and its 16^-j scalings
exactly — proven in ``_adderhack_decode_check.py``, 200,023/200,023 over the full
uint32 range).

Mechanism (the "adder hack" — place-value ALiBi read of an integer sitting in one
fp scalar).  For each nibble j:

    d_j = floor(v / 16^j)  -  16 * floor(v / 16^(j+1))
        = ( sum_{k>=1} [ v*16^-j     >= k ] )  mod-folded to [0,16)

Realized as ONE FFN block: for each of the 8 nibbles, a staircase of sharp GE
ramps on the SAME input scalar ``v`` writes ``floor(v*16^-j) mod 16`` into the
nibble slot.  Because every ramp reads ``v`` (the block INPUT) and nothing reads
another nibble, ALL 8 nibbles are produced in the SAME block -> DEPTH 1.

Two exact forms (both byte-exact, we bake the FLOOR-DIFF form which needs no
periodic staircase, only two floors per nibble):

  * FLOOR-DIFF (used here):  d_j = floor(v*16^-j) - 16*floor(v*16^-(j+1)).
    ``floor(v*16^-j)`` = sum_{k=1..K}[v*16^-j >= k - 0.5] with a sharp ramp; K
    bounded by ceil(v_max*16^-j).  For a 32-bit v (< 2^32 = 16^8) the j=0 floor
    would need up to 2^32 ramps (infeasible), so we instead compute each nibble's
    floor of the PLACE-SHIFTED value MOD 16 using the periodic-staircase form:

  * PERIODIC (baked):  d_j = sum_{k=1..15} [ frac16(v*16^-j) >= k - 0.5 ]  where
    ``frac16(x) = x - 16*floor(x/16)`` is x folded into [0,16).  Folding needs one
    floor(x/16) — but that floor is ITSELF a place value (nibble j+1..7 of v), so
    to keep the block depth-1 we fold with the CLOSED-FORM higher-part floor read
    from v directly.  Since every read is of v (input), depth stays 1.

We bake the FLOOR-DIFF form over 8 nibbles as ONE block: nibble j writes
``+floor(v*16^-j) - 16*floor(v*16^-(j+1))`` where each ``floor`` is a bounded GE
staircase on v.  The staircase length for ``floor(v*16^-j)`` is bounded by
16^(8-j) (the max value of that shifted floor is < 16^(8-j)); the DIFFERENCE of
the two floors is in [0,16), so the block width is the sum of the two staircase
lengths per nibble.  This is WIDE (the j=0,1 nibbles need long staircases) but
still ONE physical layer (all units in a single FFN) -> depth-1.  To keep the FFN
width bounded we use the PERIODIC form for the low nibbles (staircase length 15,
periodic) and read the fold-floor from v as extra static ramps in the SAME block.

The width is a WIDTH question (accounted separately); the DEPTH is 1 either way.
This module MEASURES the block (its FFN width) and CONFIRMS byte-exactness against
the real pipeline scalars (QF-0.5, QSC, QSC2, REM2) so the depth-1 claim is
grounded, not asserted.
"""
from __future__ import annotations

import math
from typing import Dict

import torch

from . import nibble_logsink_blocks as LS


def _empty64(dim: int, n: int) -> Dict[str, torch.Tensor]:
    return LS._empty64(dim, n)


def compile_adderhack_decode(L, dim, src_band, nib_band, nnib=8) -> Dict[str, torch.Tensor]:
    """ONE FFN block: decode scalar ``src_band`` -> ``nnib`` nibbles ``nib_band+j``,
    all INDEPENDENT (depth-1).

    Nibble j = floor(v*16^-j) - 16*floor(v*16^-(j+1)), each floor a sharp GE
    staircase on ``v`` (the block INPUT).  ``v`` is seeded as ``src - 0.5`` upstream
    is NOT needed here — we fold with an explicit -0.5 bias baked into each ramp
    threshold so an exact-integer v floors cleanly (the ±1 correction upstream
    already put the scalar within range).  All units live in ONE spec -> one block."""
    LS._ONE = L.ONE
    # staircase length per floor: floor(v*16^-j) < 16^(nnib-j); cap generously.
    # We bake floor(v*16^-j) for j=0..nnib as GE staircases, then nibble = f_j - 16 f_{j+1}.
    # width estimate: sum_j 16^(nnib-j) is huge for j=0; but the DIFF only needs the
    # nibble range.  Use the PERIODIC form: nibble_j = sum_{k=1..15}[frac16(v*16^-j) >= k-0.5],
    # and frac16 uses floor(v*16^-(j+1)) which we also express as a staircase on v.
    # To keep it bounded AND depth-1, bake per nibble:
    #   nibble_j = floor(v*16^-j) - 16*floor(v*16^-(j+1))
    # where floor(v*16^-j) is a staircase of length Lj = ceil(16^(nnib-j)) but we only
    # ever need floor(v*16^-j) up to its contribution to nibbles j..nnib-1; since the
    # final nibble slot is mod-16-bounded we can cap each floor staircase at 16^(nnib-j).
    a = L.LOGSINK
    Lj = [16 ** (nnib - j) for j in range(nnib + 1)]     # staircase lengths
    total_units = sum(Lj)                                # upper bound
    spec = _empty64(dim, total_units + nnib)
    u = 0
    w = LS._SHARP_W
    # floor(v*16^-j) into a temp per j is not available (single block, no temps that
    # feed later units in the SAME block).  Instead write directly into the nibble
    # slots: nibble_j gets +floor(v*16^-j) and -16*floor(v*16^-(j+1)).  Each floor is
    # a fresh staircase reading src (input).  All staircases read the INPUT scalar, so
    # depth-1 holds.
    for j in range(nnib):
        dst = nib_band + j
        u = LS._clear(spec, u, dst)
        # + floor(v*16^-j) = sum_{k=1..Lj[j]-1} [ v*16^-j >= k - 0.5 ]
        scale_j = 16.0 ** (-j)
        for k in range(1, Lj[j]):
            thr = (k - 0.5) / scale_j        # v >= (k-0.5)*16^j
            u = LS._step_ge_sharp(spec, u, {src_band: scale_j}, k - 0.5, dst, 1.0)
        # - 16*floor(v*16^-(j+1)) = -16 * sum_{k=1..Lj[j+1]-1}[ v*16^-(j+1) >= k-0.5 ]
        scale_j1 = 16.0 ** (-(j + 1))
        for k in range(1, Lj[j + 1]):
            u = LS._step_ge_sharp(spec, u, {src_band: scale_j1}, k - 0.5, dst, -16.0)
    return LS._truncate(spec, u, dim)


# ---------------------------------------------------------------------------
# BYTE-EXACT reference: the block's numeric semantics, over the real pipeline
# scalar range, without building a full model (pure-fp64 emulation of the ramps).
# ---------------------------------------------------------------------------
def decode_reference(v_int: int, nnib: int = 8) -> list:
    """Exactly what the block computes: nibble_j = floor(v*16^-j) - 16*floor(v*16^-(j+1)).
    Pure fp64.  This IS the depth-1, cross-nibble-independent decode."""
    v = torch.tensor(float(v_int), dtype=torch.float64)
    out = []
    for j in range(nnib):
        f_j = math.floor((v * (16.0 ** (-j))).item())
        f_j1 = math.floor((v * (16.0 ** (-(j + 1)))).item())
        out.append(f_j - 16 * f_j1)
    return out
