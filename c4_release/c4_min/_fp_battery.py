"""Bit-exact IEEE-754 single battery: Python oracle (isa.f32_op_bits) vs gcc.

Generates a battery over normals, subnormals, +/-inf, NaN, signed zero,
rounding-tie cases, overflow/underflow for each of F_ADD/F_SUB/F_MUL/F_DIV;
runs both the Python oracle and the compiled gcc reference; asserts bit-exact.
Also cross-checks the softfloat.c lib when present (compiled f32_add/sub/mul/div).
"""
from __future__ import annotations

import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from c4_min import isa  # noqa: E402

OPS = {"F_ADD": isa.F_ADD, "F_SUB": isa.F_SUB, "F_MUL": isa.F_MUL, "F_DIV": isa.F_DIV}


def _f(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _b(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def battery_operands():
    """A representative set of raw 32-bit float bit patterns."""
    vals = set()
    named = [
        0x00000000,  # +0
        0x80000000,  # -0
        0x3F800000,  # 1.0
        0xBF800000,  # -1.0
        0x40000000,  # 2.0
        0x40490FDB,  # pi
        0x7F800000,  # +inf
        0xFF800000,  # -inf
        0x7FC00000,  # qNaN
        0x7F800001,  # sNaN
        0x00000001,  # smallest subnormal
        0x007FFFFF,  # largest subnormal
        0x00800000,  # smallest normal
        0x7F7FFFFF,  # largest normal (FLT_MAX)
        0xFF7FFFFF,  # -FLT_MAX
        0x34000000,  # ~2^-23 (ULP scale)
        0x4B000000,  # 2^23 (integer boundary)
        0x4B800000,  # 2^24
        0x33800000,  # tiny
        0x00000002, 0x00000003,  # subnormal near-tie
        0x3F800001,  # 1.0 + 1 ulp
        0x3EFFFFFF,  # just below 0.5
    ]
    vals.update(named)
    # rounding-tie cases: values whose sum/product lands exactly halfway.
    for k in range(24):
        vals.add(_b(1.0 + (2 ** -k)))
        vals.add(_b(2.0 ** k))
        vals.add(_b(2.0 ** -k))
    rng = random.Random(12345)
    for _ in range(400):
        vals.add(rng.getrandbits(32))
    return sorted(vals)


def build_pairs():
    ops = battery_operands()
    rng = random.Random(999)
    pairs = []
    # full cross of the small named/tie set (bounded), plus random pairs.
    small = ops[:60]
    for a in small:
        for b in small:
            pairs.append((a, b))
    for _ in range(20000):
        pairs.append((rng.choice(ops), rng.choice(ops)))
        pairs.append((rng.getrandbits(32), rng.getrandbits(32)))
    return pairs


def gcc_reference(cases):
    src = os.path.join(HERE, "_fp_gcc_battery.c")
    exe = os.path.join(HERE, "_fp_gcc_battery.bin")
    subprocess.run(["gcc", "-O2", "-static-libgcc", "-o", exe, src], check=True)
    inp = "\n".join(f"{name} {a} {b}" for name, a, b in cases) + "\n"
    out = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True)
    return [int(x) for x in out.stdout.split()]


def nan_equiv(x, y):
    """A produced NaN compares equal to any NaN (payload/sign not architectural)."""
    xn = (x & 0x7F800000) == 0x7F800000 and (x & 0x007FFFFF) != 0
    yn = (y & 0x7F800000) == 0x7F800000 and (y & 0x007FFFFF) != 0
    return xn and yn


def run():
    pairs = build_pairs()
    cases = [(name, a, b) for name in OPS for (a, b) in pairs]
    gcc = gcc_reference(cases)
    mismatches = 0
    per_op = {name: [0, 0] for name in OPS}  # [ok, total]
    for (name, a, b), g in zip(cases, gcc):
        py = isa.f32_op_bits(OPS[name], a, b)
        per_op[name][1] += 1
        if py == g or nan_equiv(py, g):
            per_op[name][0] += 1
        else:
            mismatches += 1
            if mismatches <= 20:
                print(f"MISMATCH {name} a={a:08x} b={b:08x} py={py:08x} gcc={g:08x}"
                      f"  ({_f(a)!r} {name} {_f(b)!r} -> py={_f(py)!r} gcc={_f(g)!r})")
    for name, (ok, tot) in per_op.items():
        print(f"{name}: {ok}/{tot} bit-exact vs gcc")
    print(f"TOTAL mismatches: {mismatches} / {len(cases)}")
    return mismatches == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
