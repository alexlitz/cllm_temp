"""Byte-exact validation of the three #916 divmod depth levers (measure-only).

Confirms each lever's byte-exactness over a LARGE uint32 stress sample so the
depth accounting in ``qwen_fit_solver`` is grounded, not asserted.  Nothing here
is on any build path (golden 174ece66 unchanged); it is a pure-fp64 emulation of
the reduced-depth divide + decode.

LEVER 1 — radix-256 BYTE decode (4 feasible layers, not depth-1-infeasible):
    A 32-bit v decodes to 4 BYTES b_j = floor(v/256^j) mod 256, each a 256-way
    per-byte read; then each byte splits into 2 nibbles.  The byte read is the
    SAME floor/mod as the nibble decode but with a 256-entry staircase (<= 4864
    intermediate, FEASIBLE) instead of 4.87e9 for the single-layer 8-nibble form.
    Cross-byte INDEPENDENT (all 4 bytes read the original v) -> the 4 place-value
    bytes are depth-1-in-math; bounded-width realization is a 4-deep radix-256
    cascade (byte j's fold needs floor(v/256^(j+1)), so we bake it sequentially,
    but with a BOUNDED 256-wide staircase per layer).  We verify BYTE-EXACTness of
    the radix-256 byte decode + nibble split.

LEVER 2 — minimum Newton steps in the softmax1 reciprocal:
    The sink gives 1/b to a residual RoPE-phase rel-error ~5e-7 (~2^-21).  Newton
    r<-r(2-b*r) doubles precision/step (e -> e^2).  We model the sink error at its
    WORST 5e-7 and confirm the FULL divide (qf floor + ±1 + refine) is byte-exact
    with N=0,1,2 Newton steps, over the full uint32 range.

LEVER 3 — shallow verify multiply + tight base:
    The q*b verify only needs the LOW 32 bits (mod is a - q*b, and q*b <= a < 2^32
    for the correct q).  We confirm a truncated-to-32-bit q*b gives byte-exact
    div+mod.  (The base-pipeline depth is a config-accounting question, handled in
    the solver, not here.)
"""
from __future__ import annotations

import math
import random

import torch

MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# LEVER 1 — radix-256 byte decode (feasible 4-layer cascade), byte-exact.
# ---------------------------------------------------------------------------
def radix256_byte_decode_fp64(v_int: int) -> list:
    """4 BYTES b_j = floor(v/256^j) mod 256, each read from the ORIGINAL v (fp64).

    b_j = floor(v*256^-j) - 256*floor(v*256^-(j+1)).  fp64's 2^53 mantissa holds
    v (< 2^32) and its 256^-j dyadic scalings exactly, so every floor is exact.
    This is the FEASIBLE radix-256 read (256-wide staircase per byte)."""
    v = torch.tensor(float(v_int), dtype=torch.float64)
    out = []
    for j in range(4):
        lo = torch.floor(v * (256.0 ** (-j)))
        hi = torch.floor(v * (256.0 ** (-(j + 1))))
        out.append(int((lo - 256.0 * hi).item()))
    return out


def radix256_then_nibble_fp64(v_int: int) -> list:
    """The 4 bytes then split to 8 nibbles: nibble 2j = byte_j mod 16,
    nibble 2j+1 = floor(byte_j/16).  The byte->nibble split is a 16-way read of a
    value < 256 (staircase length 15, trivially feasible)."""
    bytes_ = radix256_byte_decode_fp64(v_int)
    nibs = []
    for bj in bytes_:
        b = torch.tensor(float(bj), dtype=torch.float64)
        lo = torch.floor(b * (16.0 ** -0)) - 16.0 * torch.floor(b * (16.0 ** -1))
        hi = torch.floor(b * (16.0 ** -1))
        nibs.append(int(lo.item()))
        nibs.append(int(hi.item()))
    return nibs


def true_bytes(v: int) -> list:
    return [(v >> (8 * j)) & 0xFF for j in range(4)]


def true_nibbles(v: int) -> list:
    return [(v >> (4 * j)) & 0xF for j in range(8)]


def high_radix_decode_fp64(v_int: int, bits: int) -> list:
    """FEASIBLE high-radix (radix R=2^bits) SEQUENTIAL running-remainder decode:
    ceil(32/bits) limb layers (each rem mod R via an R-wide staircase, R <= 4864
    intermediate FEASIBLE) + 1 nibble-split layer.  Returns the 8 hex nibbles.
    radix-4096 (12-bit) = 3 limbs -> 4 decode layers/pass (vs radix-256's 5)."""
    R = 1 << bits
    rem = torch.tensor(float(v_int), dtype=torch.float64)
    n = math.ceil(32 / bits)
    limbs = []
    for _ in range(n):
        limb = rem - R * torch.floor(rem / R)
        limbs.append(int(limb.item()))
        rem = torch.floor(rem / R)
    val = sum(l * (R ** j) for j, l in enumerate(limbs))
    return [(val >> (4 * k)) & 0xF for k in range(8)]


# ---------------------------------------------------------------------------
# LEVER 2 — reduced-Newton reciprocal, full divide byte-exact.
# ---------------------------------------------------------------------------
def _sink_reciprocal_with_error(b: int, worst_relerr: float, rng: random.Random,
                                dtype=torch.float64) -> float:
    """Model the softmax1 sink reciprocal 1/b PERTURBED by a worst-case residual
    RoPE-phase relative error (the ~5e-7 the newton steps must lift).  We inject
    the error at the FULL magnitude (both signs, uniformly up to worst_relerr) so
    the byte-exact check is a genuine stress of the reduced-Newton correction, not
    an idealized exact reciprocal."""
    if b == 0:
        return 0.0
    true_r = 1.0 / b
    err = rng.uniform(-worst_relerr, worst_relerr)
    return float(torch.tensor(true_r * (1.0 + err), dtype=dtype))


def _newton(r: float, b: int, steps: int, dtype=torch.float64) -> float:
    """r <- r*(2 - b*r), ``steps`` times (fp64)."""
    bt = torch.tensor(float(b), dtype=dtype)
    rt = torch.tensor(r, dtype=dtype)
    two = torch.tensor(2.0, dtype=dtype)
    for _ in range(steps):
        rt = rt * (two - bt * rt)
    return float(rt)


def div_reduced_newton(a: int, b: int, newton_steps: int, worst_relerr: float,
                       rng: random.Random, truncate_qb_32: bool = False,
                       refine_R: int = 512, dtype=torch.float64) -> tuple:
    """The log-sink divide with the reciprocal computed by the perturbed sink +
    ``newton_steps`` Newton steps, then qf floor + ±1 + REFINE (delta = floor of
    REM*RECIP2, CLAMPED to |delta| < refine_R exactly as compile_refine_add bakes
    it — the staircase only spans ±R integers).  Returns (q, r) & MASK32.  Mirrors
    nibble_logsink_blocks: qf floor, correct, refine (clamped), correct again."""
    a &= MASK32
    b &= MASK32
    if b == 0:
        return 0, a
    r0 = _sink_reciprocal_with_error(b, worst_relerr, rng, dtype)
    recip2 = _newton(r0, b, newton_steps, dtype)
    at = torch.tensor(float(a), dtype=dtype)
    r2t = torch.tensor(recip2, dtype=dtype)
    qf = float(at * r2t)
    MG = 2.0 ** 52
    q = int(((qf - 0.5) + MG) - MG)              # §555 magic floor

    def qb(qv):
        p = qv * b
        return (p & MASK32) if truncate_qb_32 else p

    # ±1 correction (compile_correct): fix an off-by-one.
    rem = a - qb(q)
    if rem < 0:
        q -= 1; rem = a - qb(q)
    elif rem >= b:
        q += 1; rem = a - qb(q)
    # REFINE (compile_refine + refine_add): delta = floor(rem * recip2) absorbs an
    # error LARGER than ±1, but the baked staircase only spans |delta| < refine_R,
    # so a larger error is NOT recovered (this is the honest clamp).
    delta = math.floor((a - qb(q)) * recip2)
    delta = max(-refine_R + 1, min(refine_R - 1, delta))   # baked staircase clamp
    q += delta
    rem = a - qb(q)
    for _ in range(2):
        if rem < 0:
            q -= 1; rem = a - qb(q)
        elif rem >= b:
            q += 1; rem = a - qb(q)
        else:
            break
    return q & MASK32, rem & MASK32


# ---------------------------------------------------------------------------
# Stress sample: full uint32 range + adversarial floor-boundary cases.
# ---------------------------------------------------------------------------
def _decode_sample(n_random: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    hard = [0, 1, 15, 16, 255, 256, 257, 4095, 4096, 65535, 65536, 65537,
            0x00FFFFFF, 0x01000000, 0x0FFFFFFF, 0x10000000, 0x7FFFFFFF,
            0x80000000, 0xFFFFFFFF, 0xF0F0F0F0, 0x0F0F0F0F, 0xDEADBEEF,
            0xCAFEBABE, 0x12345678, 0xFEDCBA98, 0xFFFFFF00, 0x000000FF,
            0xFF00FF00, 0x00FF00FF]
    vals = list(hard)
    for _ in range(n_random):
        vals.append(rng.randrange(0, 1 << 32))
    return vals


def _div_sample(n_random: int, seed: int = 1) -> list:
    rng = random.Random(seed)
    pairs = []
    # adversarial: large quotient (b small, a large) — the reciprocal-error stressor.
    hard_b = [1, 2, 3, 7, 10, 255, 256, 257, 1000, 65535, 65536,
              0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 6, 9, 17, 100003]
    hard_a = [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFE,
              0xDEADBEEF, 0x12345678]
    for b in hard_b:
        for a in hard_a:
            pairs.append((a, b))
        for k in (1, 2, 3, 1000, 100000, 0xFFFF, 0xFFFFF):
            base = (k * b) & MASK32
            for d in (-1, 0, 1):
                pairs.append(((base + d) & MASK32, b))
    for _ in range(n_random):
        a = rng.randrange(0, 1 << 32)
        b = rng.randrange(1, 1 << 32)
        pairs.append((a, b))
        if rng.random() < 0.5:
            pairs.append((rng.randrange(0, 1 << 32), rng.randrange(1, 1024)))
    return pairs


def run_lever1(n_random: int = 200000) -> dict:
    vals = _decode_sample(n_random)
    byte_ok = nib_ok = hi_ok = 0
    byte_bad = nib_bad = hi_bad = None
    for v in vals:
        tb = true_bytes(v)
        db = radix256_byte_decode_fp64(v)
        if db == tb:
            byte_ok += 1
        elif byte_bad is None:
            byte_bad = (hex(v), tb, db)
        tn = true_nibbles(v)
        dn = radix256_then_nibble_fp64(v)
        if dn == tn:
            nib_ok += 1
        elif nib_bad is None:
            nib_bad = (hex(v), tn, dn)
        # radix-4096 (12-bit limb, 3 limbs -> 4 decode layers): the MAXIMAL feasible
        # radix (staircase 4096 <= 4864 intermediate), one fewer limb than radix-256.
        dh = high_radix_decode_fp64(v, 12)
        if dh == tn:
            hi_ok += 1
        elif hi_bad is None:
            hi_bad = (hex(v), tn, dh)
    return {"total": len(vals), "byte_ok": byte_ok, "byte_bad": byte_bad,
            "nib_ok": nib_ok, "nib_bad": nib_bad,
            "hi_radix_ok": hi_ok, "hi_radix_bad": hi_bad}


def run_lever2(n_random: int = 60000, worst_relerr: float = 5e-7) -> dict:
    """Byte-exact div+mod with N=0,1,2 Newton steps, sink error at worst 5e-7."""
    pairs = _div_sample(n_random)
    out = {}
    for nstep in (0, 1, 2):
        rng = random.Random(12345 + nstep)     # fresh perturbation stream per N
        ok = 0
        bad = None
        for a, b in pairs:
            q, r = div_reduced_newton(a, b, nstep, worst_relerr, rng)
            tq = (a // b) if b else 0
            tr = (a % b) if b else a
            if q == (tq & MASK32) and r == (tr & MASK32):
                ok += 1
            elif bad is None:
                bad = (hex(a), hex(b), (tq, tr), (q, r))
        out[nstep] = {"total": len(pairs), "ok": ok, "bad": bad}
    return out


def run_lever3(n_random: int = 60000, newton_steps: int = 1,
               worst_relerr: float = 5e-7) -> dict:
    """Byte-exact div+mod with the q*b verify TRUNCATED to the low 32 bits
    (mod only needs low 32; for the correct q, q*b <= a < 2^32 so no wrap)."""
    pairs = _div_sample(n_random)
    rng = random.Random(999)
    ok = 0
    bad = None
    for a, b in pairs:
        q, r = div_reduced_newton(a, b, newton_steps, worst_relerr, rng,
                                  truncate_qb_32=True)
        tq = (a // b) if b else 0
        tr = (a % b) if b else a
        if q == (tq & MASK32) and r == (tr & MASK32):
            ok += 1
        elif bad is None:
            bad = (hex(a), hex(b), (tq, tr), (q, r))
    return {"total": len(pairs), "ok": ok, "bad": bad}


class _ExtremeRng:
    """Forces the sink reciprocal error to its worst magnitude/sign, so the qf
    floor is pushed maximally away from the true quotient — the adversarial corner
    for the Newton-step minimum (the random sample under-hits it)."""
    def __init__(self, sign):
        self.sign = sign
    def uniform(self, lo, hi):
        return hi if self.sign > 0 else lo
    def random(self):
        return 0.0
    def randrange(self, *a):
        return 0


def run_lever2_worst_corner(worst_relerr: float = 5e-7, refine_R: int = 512) -> dict:
    """ADVERSARIAL corner: reciprocal error pinned to ±worst_relerr, huge quotient
    (b small, a near 2^32).  With the refine staircase clamped to |delta| < R, the
    initial qf error (q*e) must land within ±(R + a-few ±1) for the divide to be
    exact.  This is where too-few Newton steps break — the honest minimum test."""
    corner = []
    for b in (1, 2, 3, 5, 7, 11, 13):
        for a in (0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFF01, 0x80000000, 0xC0000000,
                  0xE0000000, 0xFFFF0000):
            corner.append((a, b))
    out = {}
    for nstep in (0, 1, 2):
        ok = 0
        bad = None
        for a, b in corner:
            for sign in (+1, -1):
                q, r = div_reduced_newton(a, b, nstep, worst_relerr,
                                          _ExtremeRng(sign), refine_R=refine_R)
                tq, tr = a // b, a % b
                if q == (tq & MASK32) and r == (tr & MASK32):
                    ok += 1
                elif bad is None:
                    bad = (hex(a), b, sign, (tq, tr), (q, r))
        out[nstep] = {"total": len(corner) * 2, "ok": ok, "bad": bad}
    return out


if __name__ == "__main__":
    import resource

    print("=" * 72)
    print("LEVER 1 — feasible bounded-width decode (radix-256 + max-radix), byte-exact")
    r1 = run_lever1(200000)
    print(f"  radix-256 byte decode   : {r1['byte_ok']}/{r1['total']}  "
          f"{'ALL PASS' if r1['byte_ok'] == r1['total'] else 'FAIL ' + str(r1['byte_bad'])}")
    print(f"  radix-256 -> 8 nibbles  : {r1['nib_ok']}/{r1['total']}  "
          f"{'ALL PASS' if r1['nib_ok'] == r1['total'] else 'FAIL ' + str(r1['nib_bad'])}")
    print(f"  radix-4096 (3-limb) dec : {r1['hi_radix_ok']}/{r1['total']}  "
          f"{'ALL PASS' if r1['hi_radix_ok'] == r1['total'] else 'FAIL ' + str(r1['hi_radix_bad'])}")

    print("=" * 72)
    print("LEVER 2 — reduced Newton steps (sink worst-case rel-err 5e-7), div byte-exact")
    print("  (a) random full-uint32 stress sample:")
    r2 = run_lever2(60000, worst_relerr=5e-7)
    for nstep in (0, 1, 2):
        d = r2[nstep]
        print(f"    Newton={nstep}: {d['ok']}/{d['total']}  "
              f"{'ALL PASS' if d['ok'] == d['total'] else 'FAIL ' + str(d['bad'])}")
    print("  (b) ADVERSARIAL worst-sign corner (huge q, refine staircase R=512 clamp):")
    r2c = run_lever2_worst_corner(worst_relerr=5e-7, refine_R=512)
    for nstep in (0, 1, 2):
        d = r2c[nstep]
        print(f"    Newton={nstep}: {d['ok']}/{d['total']}  "
              f"{'ALL PASS' if d['ok'] == d['total'] else 'FAIL ' + str(d['bad'])}")

    print("=" * 72)
    print("LEVER 3 — truncated-to-32-bit q*b verify (Newton=1), div byte-exact")
    r3 = run_lever3(60000, newton_steps=1)
    print(f"  q*b mod 2^32: {r3['ok']}/{r3['total']}  "
          f"{'ALL PASS' if r3['ok'] == r3['total'] else 'FAIL ' + str(r3['bad'])}")

    print("=" * 72)
    print(f"peak RSS MB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}")
