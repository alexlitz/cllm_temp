"""C4_ATTN_TABLES — the reciprocal-divide extension (ops-via-lookup).

The most expensive ISA op is DIV: the recurrent base-16 long-division megablock
is 168 of the VM's 242 blocks (``c4_min.batched_block_skip``), and every DIV step
pays the whole division-layer stack.

If the DIVISOR ranges over a small known set (doom.c divides by ``FP=1024`` and by
per-frame magnitudes ``mag`` that fit a byte), we can replace the DIV with a
LOOKUP + MUL:

    a / b  ==  (a * recip[b]) >> SHIFT      where recip[b] = round(2^SHIFT / b)

The LOOKUP is BYTE-EXACT (attention CAM, proven in ``_agent_attn_tables``); the
reciprocal-multiply is APPROXIMATE (a fixed-point reciprocal, like DIV-light),
with a bounded error we measure here.  For an exact power-of-two divisor
(``FP=1024`` is ``2^10``) the reciprocal-multiply is EXACT (it degenerates to a
shift), so doom's dominant ``/ FP`` divides become exact lookup-free shifts and
the residual per-frame ``/ mag`` divides use the reciprocal table.

We measure, per divisor set:
  * steps saved  — one 168-block DIV replaced by one O(1) LOOKUP + one MUL,
  * exactness    — fraction of (a, b) pairs whose reciprocal-multiply equals the
                   true floor division, and the max abs error where it differs.
"""
from __future__ import annotations

from typing import List, Tuple

from _agent_attn_tables import AttentionTable
from c4_min.nibble_muldivmod import mul32, div32


def build_reciprocal_table(max_divisor: int, shift: int) -> AttentionTable:
    """recip[b] = round(2^shift / b) for b in 0..max_divisor (recip[0]=0, DIV-by-0).

    Baked as a fixed attention KV table keyed on the divisor ``b``.
    """
    recip = [0] * (max_divisor + 1)
    for b in range(1, max_divisor + 1):
        recip[b] = round((1 << shift) / b)
    return AttentionTable(recip, base_addr=0x30000, stride=4,
                          name=f"recip_s{shift}").bake()


def recip_divide(a: int, b: int, table: AttentionTable, shift: int) -> int:
    """a // b via LOOKUP(recip, b) + MUL + shift.  Uses the SAME 32-bit MUL the ISA
    uses (``mul32``); the reciprocal comes from the byte-exact attention lookup."""
    if b == 0:
        return 0                                   # ISA: div-by-0 -> 0
    r = table.lookup(b)                            # byte-exact attention lookup
    # (a * recip) >> shift.  HONEST: a*recip can exceed 32 bits (recip~2^shift, a
    # up to 2^32), so this needs a WIDENING MUL (the high word) — the blog MUL
    # primitive (nibble_muldivmod.nibble_mul) computes all 10 partial products so
    # the hi bits ARE available; the ISA's mul32 discards them (32-bit mask), so a
    # 2-word product + a cross-word shift is the real VM cost (still << a 168-block
    # DIV).  We use the full python product here to isolate the RECIPROCAL
    # approximation error from the (mechanical, exact) wide-multiply/shift.
    prod = a * r
    return prod >> shift


def measure_recip_divide(max_divisor: int, shift: int,
                         a_samples: List[int]) -> dict:
    """Compare recip-divide vs the ISA's true floor DIV over a grid of (a, b).

    Returns exactness fraction, max abs error, and the step saving.
    """
    table = build_reciprocal_table(max_divisor, shift)
    total = 0
    exact = 0
    max_err = 0
    worst: Tuple[int, int, int, int] = (0, 0, 0, 0)
    for b in range(1, max_divisor + 1):
        for a in a_samples:
            true_q = div32(a, b)                   # ISA floor division
            approx_q = recip_divide(a, b, table, shift)
            total += 1
            err = abs(approx_q - true_q)
            if err == 0:
                exact += 1
            elif err > max_err:
                max_err = err
                worst = (a, b, true_q, approx_q)
    return {
        "max_divisor": max_divisor,
        "shift": shift,
        "pairs": total,
        "exact_pairs": exact,
        "exact_fraction": exact / total if total else 0.0,
        "max_abs_error": max_err,
        "worst_case": {"a": worst[0], "b": worst[1],
                       "true": worst[2], "approx": worst[3]},
        "table_entries": max_divisor + 1,
        "steps_per_divide_DIV": "1 DIV op = 168/242-block long-division megablock",
        "steps_per_divide_LOOKUP_MUL": "1 O(1) attention lookup + 1 MUL",
    }
