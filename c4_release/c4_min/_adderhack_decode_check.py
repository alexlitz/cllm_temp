"""Byte-exact check of the ADDER-HACK depth-1 nibble decode (#916 continued).

The claim under test (from the task brief): a 32-bit integer sitting in ONE fp64
scalar ``v`` can be decoded to its 8 hex nibbles ``d_j = floor(v/16^j) mod 16``
with each nibble INDEPENDENT of the others (no running remainder / no cross-nibble
dependency), so the read is DEPTH-1 (all 8 nibbles in parallel in one layer).

This is the DIRECT-FLOOR-MOD decode, contrasted with the schoolbook MSB-first
running-remainder decode (``nibble_logsink_blocks._msb_extract_block`` + reduce),
which is an 8-block SEQUENTIAL recurrence (each ``reduce`` feeds the next
``extract``).

We verify TWO things, both byte-exact over a stress sample of the full uint32
range (all 8 nibbles must match Python's exact ``(v>>(4j))&0xF``):

  (A) DIRECT floor/mod is exact in fp64 for v in [0, 2^32):
        d_j = floor(v * 16^-j) - 16*floor(v * 16^-(j+1))
      This is exactly what an ALiBi place-value read + a difference-of-floors
      realizes as a per-nibble, cross-nibble-INDEPENDENT operation.  fp64's 2^53
      mantissa holds v (< 2^32) AND v*16^-j (a dyadic-scaled subset) exactly, so
      every floor is exact.

  (B) The floor itself, realized as the difference-min / half-shifted argmax the
      adder-hack decode head uses (``d = argmax_c -|value - (c+0.5)|`` over the
      bucket value ``value = v/16^j``), gives the SAME nibble — i.e. the decode
      head mechanism (not just abstract floor) is byte-exact and needs NO running
      remainder.  We realize it as floor(value) via the +0.5 recentred bucket, per
      nibble, INDEPENDENTLY.

If both pass for the full 32-bit stress set, the adder-hack decode is a valid
DEPTH-1 replacement for the 8-block schoolbook recurrence, byte-exact, fp64.
"""
from __future__ import annotations

import math
import random

import torch


def true_nibbles(v: int) -> list:
    return [(v >> (4 * j)) & 0xF for j in range(8)]


def direct_floor_mod_fp64(v_int: int) -> list:
    """(A) Per-nibble INDEPENDENT floor/mod in fp64 — no running remainder.

    d_j = floor(v/16^j) - 16*floor(v/16^(j+1)).  Every term is computed straight
    from the ORIGINAL v (all 8 in parallel: depth-1), so there is NO cross-nibble
    dependency."""
    v = torch.tensor(float(v_int), dtype=torch.float64)
    out = []
    for j in range(8):
        lo = torch.floor(v * (16.0 ** (-j)))
        hi = torch.floor(v * (16.0 ** (-(j + 1))))
        d = lo - 16.0 * hi
        out.append(int(d.item()))
    return out


def adderhack_argmax_decode_fp64(v_int: int) -> list:
    """(B) The adder-hack decode-HEAD mechanism, per nibble, INDEPENDENT.

    For nibble j the bucket value is ``value = v/16^j`` and the nibble is
    ``floor(value) mod 16``.  The decode head selects ``floor(value mod 16)`` via
    the +0.5-recentred negative-abs-difference argmax over candidates 0..15 — the
    SAME selection the minimal-adder decode uses, but applied to the ALREADY
    place-shifted value with an explicit mod-16 fold (``value - 16*floor(value/16)``).
    All 8 nibbles read the original v independently (depth-1)."""
    v = torch.tensor(float(v_int), dtype=torch.float64)
    cand = torch.arange(16, dtype=torch.float64)          # 0..15
    half = 0.5
    tie = 1e-12 * cand                                    # floor tie-break (favor larger)
    out = []
    for j in range(8):
        value = v * (16.0 ** (-j))                        # place-shifted value
        # fold to this nibble's bucket: frac_bucket in [0,16)
        bucket = value - 16.0 * torch.floor(value / 16.0)
        logits = -(bucket.unsqueeze(-1) - (cand + half)).abs() + tie   # (16,)
        d = int(logits.argmax(dim=-1).item())
        out.append(d)
    return out


def _sample(n_random: int = 200000, seed: int = 0) -> list:
    rng = random.Random(seed)
    hard = [0, 1, 15, 16, 255, 256, 4095, 4096, 65535, 65536,
            0x0FFFFFFF, 0x10000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
            0xF0F0F0F0, 0x0F0F0F0F, 0xDEADBEEF, 0xCAFEBABE, 0x12345678,
            0xFEDCBA98, 0xFFFFFFF0, 0x0000000F]
    vals = list(hard)
    for _ in range(n_random):
        vals.append(rng.randrange(0, 1 << 32))
    return vals


def run(n_random: int = 200000) -> dict:
    vals = _sample(n_random)
    a_ok = b_ok = 0
    a_bad = b_bad = None
    for v in vals:
        t = true_nibbles(v)
        da = direct_floor_mod_fp64(v)
        db = adderhack_argmax_decode_fp64(v)
        if da == t:
            a_ok += 1
        elif a_bad is None:
            a_bad = (v, t, da)
        if db == t:
            b_ok += 1
        elif b_bad is None:
            b_bad = (v, t, db)
    return {
        "total": len(vals),
        "direct_floor_mod_ok": a_ok, "direct_first_bad": a_bad,
        "adderhack_argmax_ok": b_ok, "adderhack_first_bad": b_bad,
    }


if __name__ == "__main__":
    import resource
    r = run(200000)
    print(f"ADDER-HACK depth-1 decode byte-exact check (fp64), {r['total']} values:")
    print(f"  (A) direct floor/mod (independent per nibble): "
          f"{r['direct_floor_mod_ok']}/{r['total']}"
          f"  {'ALL PASS' if r['direct_floor_mod_ok']==r['total'] else 'FAIL '+str(r['direct_first_bad'])}")
    print(f"  (B) adder-hack argmax decode-head:             "
          f"{r['adderhack_argmax_ok']}/{r['total']}"
          f"  {'ALL PASS' if r['adderhack_argmax_ok']==r['total'] else 'FAIL '+str(r['adderhack_first_bad'])}")
    print(f"  peak RSS MB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024}")
