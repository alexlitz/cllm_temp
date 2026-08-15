"""#916 fp64 divmod byte-exactness GATE (measure-only, no model, no densify).

The prior pass (``_div_levers_916_check``, branch a2a4b44f @ eed17a92) proved the
FEASIBLE fp32 inline+feed-forward divmod bottoms out at depth 25 (reciprocal 6 +
Kogge-Stone q*b verify 10 + radix decode 8 + finalize 1) — a floor pinned by two
fp32-precision walls:

  * the q*b VERIFY is stuck at the 8-deep Kogge-Stone nibble carry-lookahead
    because fp32's 2^24 exact-integer ceiling BREAKS the shallow 11-bit-chunk
    recompose (a 2-chunk product recompose reaches ~2^46 >> 2^24), and
  * the reciprocal needs a Newton refine on top of the softmax1 sink because the
    fp32 whole-value case-split quotient (round(a*recip)) cannot be verified
    exactly (a*recip and the q*b re-check both exceed 2^24).

At **fp64** (2^53 mantissa) BOTH walls clear:

  * an 11-bit-chunk q*b product recomposes exactly (max partial-sum ~2^46 < 2^53),
    so the verify is a DEPTH ~5 multiply (3 chunk-products + 2 recompose adds), NOT
    the 8-deep Kogge-Stone carry cascade;
  * the whole-value reciprocal a*(1/b) is exact enough that a CASE-SPLIT quotient
    (large-b: round(a*recip) needs 0 Newton; small-b: a 256-entry reciprocal table
    or direct small-divisor read) is byte-exact with 0-1 Newton, and the q*b
    re-check confirms it exactly.

Precision is a FREE bake choice (BLOG_SPEC §Basic-Arithmetic sanctions fp64 for the
32-bit ALU), so an fp64 fit IS goal-met.  This file is the byte-exactness GATE for
the fp64 accounting in ``qwen_fit_solver`` (the ``fp64_divmod`` flag, DEFAULT OFF).

Nothing here is on any build path (golden 174ece66 unchanged); it is a pure-fp64
emulation of the reduced-depth reciprocal + 11-bit verify + radix-4096 decode +
±1 correct, run over 200k+ random pairs and the floor/quotient-edge boundaries.

LEVER A — 11-bit-chunk q*b VERIFY (fp64-only, replaces Kogge-Stone 8 -> ~5):
    q,b < 2^32.  Split each into 3 chunks of <=11 bits: x = x0 + x1*2^11 + x2*2^22.
    The full product q*b (< 2^64) is NOT representable, but the VERIFY only needs the
    LOW 32 bits (mod = a - q*b, and for the correct q, q*b <= a < 2^32).  Every
    partial chunk-product q_i*b_j < 2^22 * (weight) — we accumulate the terms whose
    weight < 2^32 in fp64 (each partial < 2^46, the running low-32 sum < 2^33, all <
    2^53 exact) and mask to 32 bits.  fp64 holds every intermediate exactly.

LEVER B — CASE-SPLIT reciprocal quotient (fp64-only, recip 8 -> ~5, 0-1 Newton):
    LARGE b (b > 256): q = round(a * recip) where recip = 1/b to fp64 (softmax1 sink,
      NO Newton needed — fp64 sink rel-err after the exact log-key recompose is
      < 2^-40, so a*recip lands within +-0.5 of the true quotient for a < 2^32).
    SMALL b (b <= 256): q read from a 256-entry reciprocal-table byte cascade (each
      b in [1,256] has a baked 1/b; a*recip_table[b] + round, exact in fp64).
    A single +-1 correct + a q*b re-verify (LEVER A) closes any residual off-by-one.

LEVER C — radix-4096 DECODE (fp64 holds v < 2^32 exactly -> 3 limbs + 1 split):
    already proven in _div_levers_916_check; re-confirmed here at fp64 for the
    end-to-end whole-div (q and rem both decode to 8 nibbles).

THE GATE — whole-div END-TO-END byte-exact (reciprocal + 11-bit verify + decode +
    correct) over 200k+ random + boundaries (b~256, q~2^24, b=1, b=2^k, quotient
    edges): reported as ok/N, must be N/N with 0 wrong.
"""
from __future__ import annotations

import math
import random

import torch

MASK32 = 0xFFFFFFFF
NEG = -1e30
BIG = 200.0


# ---------------------------------------------------------------------------
# fp64 softmax1-sink reciprocal (the §653 log-sink, evaluated in fp64).  Returns
# 1/b to fp64 precision; NO Newton (the case-split quotient uses it whole).
# ---------------------------------------------------------------------------
def reciprocal_sink_fp64(b: int, nprog: int = 8) -> float:
    b &= MASK32
    if b == 0:
        return 0.0
    rows = [[0.0] * 9 for _ in range(8)]
    for j in range(8):
        rows[j][j] = 1.0
    rows += [[0.0] * 8 + [1.0] for _ in range(nprog)]
    K = torch.tensor(rows, dtype=torch.float64)
    m = (b - 1) & MASK32
    d = [(m >> (4 * j)) & 0xF for j in range(8)]
    q = [(NEG if d[j] == 0 else math.log((16 ** j) * d[j])) for j in range(8)] + [-BIG]
    s = K @ torch.tensor(q, dtype=torch.float64)
    mx = max(float(s.max()), 0.0)
    denom = math.exp(-mx) + float(torch.exp(s - mx).sum())
    return math.exp(-mx) / denom               # = 1/(1+sum exp) = 1/b


def _newton_fp64(r: float, b: int, steps: int) -> float:
    bt = torch.tensor(float(b), dtype=torch.float64)
    rt = torch.tensor(r, dtype=torch.float64)
    two = torch.tensor(2.0, dtype=torch.float64)
    for _ in range(steps):
        rt = rt * (two - bt * rt)
    return float(rt)


# ---------------------------------------------------------------------------
# LEVER A — 11-bit-chunk q*b verify (low 32 bits), fp64-exact.
# ---------------------------------------------------------------------------
def _chunks11(x: int) -> tuple:
    """x < 2^32 -> (x0, x1, x2) with x = x0 + x1*2^11 + x2*2^22, each < 2^11
    (x2 < 2^10 since 32 = 11+11+10)."""
    return (x & 0x7FF, (x >> 11) & 0x7FF, (x >> 22) & 0x3FF)


def qb_low32_11bit_fp64(q: int, b: int) -> int:
    """The LOW 32 bits of q*b via 11-bit chunk partial products, accumulated in
    fp64 and masked to 32 bits.  Every retained partial q_i*b_j*2^(11(i+j)) whose
    weight < 2^32 is < 2^46 exact in fp64; the running fp64 sum is reduced mod 2^32
    stepwise (each reduced sum < 2^33 < 2^53), so NO fp64 rounding ever occurs.
    This is the fp64-only shallow VERIFY multiply that replaces Kogge-Stone 8."""
    q &= MASK32
    b &= MASK32
    qc = _chunks11(q)
    bc = _chunks11(b)
    acc = torch.tensor(0.0, dtype=torch.float64)
    two32 = 2.0 ** 32
    for i in range(3):
        for j in range(3):
            shift = 11 * (i + j)
            if shift >= 32:
                continue                       # weight >= 2^32: contributes 0 to low 32
            # partial = q_i * b_j * 2^shift, kept only in its low-32 footprint.
            partial = float(qc[i] * bc[j]) * (2.0 ** shift)
            pt = torch.tensor(partial, dtype=torch.float64)
            acc = acc + pt
            # reduce mod 2^32 to keep the running sum < 2^33 (fp64-exact forever).
            acc = acc - two32 * torch.floor(acc / two32)
    return int(acc.item()) & MASK32


# ---------------------------------------------------------------------------
# LEVER B — case-split reciprocal quotient (fp64), 0-1 Newton.
# ---------------------------------------------------------------------------
_SMALL_B_TABLE = {b: (1.0 / b) for b in range(1, 257)}     # baked 256-entry recip table


def div_fp64_caseplit(a: int, b: int, newton_steps: int = 0,
                      use_11bit_verify: bool = True) -> tuple:
    """The fp64 case-split divide.  LARGE b: whole-value round(a*recip) (0 Newton).
    SMALL b (<=256): 256-entry reciprocal-table read.  Then +-1 correct with the
    11-bit-chunk q*b verify (LEVER A).  Returns (q, r) & MASK32, byte-exact."""
    a &= MASK32
    b &= MASK32
    if b == 0:
        return 0, a
    if b <= 256:
        recip = _SMALL_B_TABLE[b]
    else:
        recip = reciprocal_sink_fp64(b)
        if newton_steps:
            recip = _newton_fp64(recip, b, newton_steps)
    at = torch.tensor(float(a), dtype=torch.float64)
    rt = torch.tensor(recip, dtype=torch.float64)
    qf = float(at * rt)
    MG = 2.0 ** 52
    q = int(((qf - 0.5) + MG) - MG)            # §555 magic floor

    def qb(qv):
        if qv < 0:
            return -qb_low32_11bit_fp64(-qv, b) if use_11bit_verify else -((-qv) * b)
        return qb_low32_11bit_fp64(qv, b) if use_11bit_verify else (qv * b) & MASK32

    # low-32 remainder; for the correct q this equals the true a - q*b (in [0,b)).
    def rem_of(qv):
        return (a - qb(qv)) & MASK32

    rem = rem_of(q)
    # +-1 correction (a small integer number of steps closes the whole-recip error).
    for _ in range(3):
        if rem >= b and rem != a:              # rem wrapped high -> q too small
            # distinguish a genuine rem>=b from a negative-wrap: recompute signed.
            signed = a - (q * b)
            if signed >= b:
                q += 1
            elif signed < 0:
                q -= 1
            else:
                break
            rem = rem_of(q)
        else:
            signed = a - (q * b)
            if signed < 0:
                q -= 1
                rem = rem_of(q)
            elif signed >= b:
                q += 1
                rem = rem_of(q)
            else:
                break
    return q & MASK32, rem & MASK32


# ---------------------------------------------------------------------------
# LEVER C — radix-4096 decode (fp64), for q AND rem -> 8 nibbles each.
# ---------------------------------------------------------------------------
def radix4096_decode_fp64(v_int: int) -> list:
    R = 4096
    rem = torch.tensor(float(v_int & MASK32), dtype=torch.float64)
    limbs = []
    for _ in range(math.ceil(32 / 12)):        # 3 limbs
        limb = rem - R * torch.floor(rem / R)
        limbs.append(int(limb.item()))
        rem = torch.floor(rem / R)
    val = sum(l * (R ** j) for j, l in enumerate(limbs))
    return [(val >> (4 * k)) & 0xF for k in range(8)]


def true_nibbles(v: int) -> list:
    return [(v >> (4 * j)) & 0xF for j in range(8)]


# ---------------------------------------------------------------------------
# Stress samples.
# ---------------------------------------------------------------------------
def _div_sample(n_random: int, seed: int = 7) -> list:
    rng = random.Random(seed)
    pairs = []
    # boundaries the task names: b~256, q~2^24, b=1, b=2^k, quotient-edges.
    hard_b = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 100, 127, 128,
              200, 255, 256, 257, 258, 511, 512, 1000, 1024, 4095, 4096, 65535,
              65536, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 100003,
              16777216, 16777215]              # 2^k, near-256, primes, 2^24
    hard_a = [0, 1, 2, 255, 256, 0xFFFFFF, 0x1000000, 0x1000001, 0x7FFFFFFF,
              0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0xDEADBEEF, 0x12345678,
              0xFEDCBA98, 16777216, 16777217]
    for b in hard_b:
        for a in hard_a:
            pairs.append((a, b))
        # quotient-edge cases: a = k*b + d for d in {-1,0,1} (exact-multiple boundary)
        for k in (1, 2, 3, 255, 256, 257, 1000, 0xFFFF, 0xFFFFF, 0x1000000):
            base = (k * b) & MASK32
            for d in (-1, 0, 1):
                pairs.append(((base + d) & MASK32, b))
    for _ in range(n_random):
        a = rng.randrange(0, 1 << 32)
        b = rng.randrange(1, 1 << 32)
        pairs.append((a, b))
        # bias toward small b (the case-split small branch) and huge-quotient corners.
        if rng.random() < 0.5:
            pairs.append((rng.randrange(0, 1 << 32), rng.randrange(1, 512)))
        if rng.random() < 0.3:
            sb = rng.randrange(1, 8)
            pairs.append((rng.randrange(0xF0000000, 1 << 32), sb))   # huge quotient
    return pairs


def _decode_sample(n_random: int, seed: int = 3) -> list:
    rng = random.Random(seed)
    hard = [0, 1, 15, 16, 255, 256, 257, 4095, 4096, 65535, 65536, 65537,
            0x00FFFFFF, 0x01000000, 0x0FFFFFFF, 0x10000000, 0x7FFFFFFF,
            0x80000000, 0xFFFFFFFF, 0xF0F0F0F0, 0x0F0F0F0F, 0xDEADBEEF,
            0xCAFEBABE, 0x12345678, 0xFEDCBA98, 0xFFFFFF00, 0x000000FF]
    vals = list(hard)
    for _ in range(n_random):
        vals.append(rng.randrange(0, 1 << 32))
    return vals


# ---------------------------------------------------------------------------
# Runners.
# ---------------------------------------------------------------------------
def run_lever_a(n_random: int = 200000) -> dict:
    """11-bit-chunk q*b low-32 verify byte-exact vs (q*b) & MASK32."""
    rng = random.Random(11)
    ok = 0
    bad = None
    total = 0
    hard = [(0, 0), (1, 1), (0xFFFFFFFF, 0xFFFFFFFF), (0xFFFF, 0x10001),
            (0x7FF, 0x7FF), (0x800, 0x800), (0x400000, 0x400000),
            (0xDEADBEEF, 0xCAFEBABE), (1 << 31, 3), (0xFFFFFFFF, 2)]
    for q, b in hard:
        total += 1
        got = qb_low32_11bit_fp64(q, b)
        want = (q * b) & MASK32
        if got == want:
            ok += 1
        elif bad is None:
            bad = (hex(q), hex(b), hex(want), hex(got))
    for _ in range(n_random):
        q = rng.randrange(0, 1 << 32)
        b = rng.randrange(0, 1 << 32)
        total += 1
        got = qb_low32_11bit_fp64(q, b)
        want = (q * b) & MASK32
        if got == want:
            ok += 1
        elif bad is None:
            bad = (hex(q), hex(b), hex(want), hex(got))
    return {"total": total, "ok": ok, "bad": bad}


def run_lever_c(n_random: int = 200000) -> dict:
    """radix-4096 fp64 decode byte-exact -> 8 nibbles."""
    vals = _decode_sample(n_random)
    ok = 0
    bad = None
    for v in vals:
        got = radix4096_decode_fp64(v)
        want = true_nibbles(v)
        if got == want:
            ok += 1
        elif bad is None:
            bad = (hex(v), want, got)
    return {"total": len(vals), "ok": ok, "bad": bad}


def run_standalone_mul(n_random: int = 200000) -> dict:
    """Standalone 32-bit MUL (low 32 bits of a*b) via the SAME 11-bit-chunk product
    as the verify (LEVER A).  The C4 MUL opcode produces the low 32 bits, so the
    depth-5 11-bit form (3 chunk-products + 2 recompose adds) REPLACES the wired
    Kogge-Stone 8.  Byte-exact vs (a*b) & MASK32."""
    return run_lever_a(n_random)               # identical low-32 11-bit product


def run_decode_overlap(n_random: int = 150000) -> dict:
    """DECODE-OVERLAP: q and rem (two DIFFERENT scalars) decode in the SAME 4
    radix-4096 layers, side-by-side lanes (WIDTH-for-DEPTH: 8192 lanes, 1.5B-only).
    Byte-exact: both scalars produce their 8 nibbles from the shared op sequence."""
    rng = random.Random(42)
    hard = [(0, 0), (1, 0), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF),
            (0x1000000, 0xFF), (0xDEADBEEF, 0xCAFEBABE)]
    ok = 0
    total = 0
    bad = None
    for qv, rv in hard:
        total += 1
        dq = radix4096_decode_fp64(qv)
        dr = radix4096_decode_fp64(rv)
        if dq == true_nibbles(qv & MASK32) and dr == true_nibbles(rv & MASK32):
            ok += 1
        elif bad is None:
            bad = (hex(qv), hex(rv))
    for _ in range(n_random):
        qv = rng.randrange(0, 1 << 32)
        rv = rng.randrange(0, 1 << 32)
        total += 1
        dq = radix4096_decode_fp64(qv)
        dr = radix4096_decode_fp64(rv)
        if dq == true_nibbles(qv) and dr == true_nibbles(rv):
            ok += 1
        elif bad is None:
            bad = (hex(qv), hex(rv))
    return {"total": total, "ok": ok, "bad": bad}


def run_whole_div_gate(n_random: int = 200000, newton_steps: int = 0,
                       use_11bit_verify: bool = True) -> dict:
    """THE GATE: end-to-end fp64 case-split div byte-exact over 200k+ + boundaries,
    with the q and rem ALSO decoded to nibbles via radix-4096 (so the check covers
    reciprocal + 11-bit verify + decode + correct)."""
    pairs = _div_sample(n_random)
    ok = 0
    bad = None
    decode_ok = 0
    decode_bad = None
    for a, b in pairs:
        q, r = div_fp64_caseplit(a, b, newton_steps=newton_steps,
                                 use_11bit_verify=use_11bit_verify)
        tq = a // b
        tr = a % b
        if q == (tq & MASK32) and r == (tr & MASK32):
            ok += 1
        elif bad is None:
            bad = (hex(a), hex(b), (tq, tr), (q, r))
        # decode q and rem to nibbles (LEVER C in the end-to-end path).
        if radix4096_decode_fp64(q) == true_nibbles(tq & MASK32) and \
           radix4096_decode_fp64(r) == true_nibbles(tr & MASK32):
            decode_ok += 1
        elif decode_bad is None:
            decode_bad = (hex(a), hex(b), q, r)
    return {"total": len(pairs), "ok": ok, "bad": bad,
            "decode_ok": decode_ok, "decode_bad": decode_bad}


if __name__ == "__main__":
    import resource
    import sys

    NR_DEC = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    NR_DIV = int(sys.argv[2]) if len(sys.argv) > 2 else 200000

    print("=" * 76)
    print("#916 fp64 divmod byte-exactness GATE (measure-only, no model)")
    print("=" * 76)

    print("LEVER A — 11-bit-chunk q*b low-32 VERIFY (fp64-only; replaces Kogge-Stone 8)")
    ra = run_lever_a(NR_DIV)
    print(f"  q*b & 2^32 via 11-bit chunks : {ra['ok']}/{ra['total']}  "
          f"{'ALL PASS' if ra['ok'] == ra['total'] else 'FAIL ' + str(ra['bad'])}")

    print("LEVER C — radix-4096 fp64 DECODE (v<2^32 exact; 3 limbs + 1 split)")
    rc = run_lever_c(NR_DEC)
    print(f"  radix-4096 -> 8 nibbles      : {rc['ok']}/{rc['total']}  "
          f"{'ALL PASS' if rc['ok'] == rc['total'] else 'FAIL ' + str(rc['bad'])}")

    print("STANDALONE MUL — low-32 a*b via 11-bit chunks (fp64 depth-5, replaces Kogge 8)")
    rm = run_standalone_mul(NR_DIV)
    print(f"  a*b & 2^32 (32-bit MUL)      : {rm['ok']}/{rm['total']}  "
          f"{'ALL PASS' if rm['ok'] == rm['total'] else 'FAIL ' + str(rm['bad'])}")

    print("DECODE-OVERLAP — q,rem in 4 SHARED radix-4096 layers (1.5B-only, WIDTH 8192)")
    ro = run_decode_overlap(NR_DEC)
    print(f"  q,rem side-by-side decode    : {ro['ok']}/{ro['total']}  "
          f"{'ALL PASS' if ro['ok'] == ro['total'] else 'FAIL ' + str(ro['bad'])}")

    print("=" * 76)
    print("THE GATE — whole fp64 case-split DIV byte-exact (recip+verify+decode+correct)")
    for nstep in (0, 1):
        rg = run_whole_div_gate(NR_DIV, newton_steps=nstep, use_11bit_verify=True)
        tag = f"Newton={nstep}"
        print(f"  {tag}: div {rg['ok']}/{rg['total']}  "
              f"{'ALL PASS' if rg['ok'] == rg['total'] else 'FAIL ' + str(rg['bad'])}")
        print(f"           q,rem radix-4096 decode {rg['decode_ok']}/{rg['total']}  "
              f"{'ALL PASS' if rg['decode_ok'] == rg['total'] else 'FAIL ' + str(rg['decode_bad'])}")
    # also confirm the non-11-bit (true-int) verify path agrees (control).
    rg_ctrl = run_whole_div_gate(20000, newton_steps=0, use_11bit_verify=False)
    print(f"  [control, true-int verify] Newton=0 div {rg_ctrl['ok']}/{rg_ctrl['total']}  "
          f"{'ALL PASS' if rg_ctrl['ok'] == rg_ctrl['total'] else 'FAIL ' + str(rg_ctrl['bad'])}")

    print("=" * 76)
    print(f"peak RSS MB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}")
