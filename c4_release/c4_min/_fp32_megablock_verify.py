"""BIT-EXACT verification of the F_ADD/F_SUB/F_MUL/F_DIV transformer MEGABLOCKS.

Runs each op's dedicated FFN block stack (``compile_fp_{mul,add,sub,div}_blocks``)
through the standalone SwiGLU forward harness (``_fp32_blockrun.apply_blocks``),
seeding operand ``a`` into the STACK0 nibble band and ``b`` into the AX nibble band,
then reads the 8 F_RES result nibbles back and asserts they equal the raw bits of
the ``isa.f32_op_bits`` oracle (itself proven bit-exact vs gcc by ``_fp_battery``).

This is the load-bearing check: it proves the *neural weights* (not just the Python
reference) compute correctly-rounded IEEE-754 single.  Coverage is reported per op
and per operand class (normal/subnormal/inf/nan/zero/tie), HONEST on any gap.
"""
from __future__ import annotations

import os
import random
import struct
import sys

os.environ.setdefault("C4_FLOAT_OPS", "1")

import torch  # noqa: E402

from . import isa  # noqa: E402
from . import nibble_fp32 as fp  # noqa: E402
from . import _fp32_blockrun as BR  # noqa: E402


def _f(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _classify(bits):
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    if e == 0:
        return "zero" if m == 0 else "subnormal"
    if e == 0xFF:
        return "inf" if m == 0 else "nan"
    return "normal"


def build_op_blocks(L, dim, op):
    if op == isa.F_MUL:
        return fp.compile_fp_mul_blocks(L, dim)
    if op == isa.F_ADD:
        return fp.compile_fp_add_blocks(L, dim)
    if op == isa.F_SUB:
        return fp.compile_fp_sub_blocks(L, dim)
    if op == isa.F_DIV:
        return fp.compile_fp_div_blocks(L, dim)
    raise ValueError(op)


def run_op(L, dim, blocks, a_bits, b_bits):
    x = BR.new_residual(L, dim)
    BR.seed_bits(x, L.STACK0, a_bits)
    BR.seed_bits(x, L.AX, b_bits)
    x = BR.apply_blocks(x, blocks)
    return BR.read_nibbles(x, L.FP32.F_RES, 8)


def nan_equiv(x, y):
    xn = (x & 0x7F800000) == 0x7F800000 and (x & 0x007FFFFF) != 0
    yn = (y & 0x7F800000) == 0x7F800000 and (y & 0x007FFFFF) != 0
    return xn and yn


def operands():
    vals = set()
    named = [
        0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x40000000, 0x40490FDB,
        0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0x00000001, 0x007FFFFF,
        0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x34000000, 0x4B000000, 0x4B800000,
        0x33800000, 0x00000002, 0x00000003, 0x3F800001, 0x3EFFFFFF, 0xC0490FDB,
        0x41200000, 0x42F60000, 0x3DCCCCCD, 0xBDCCCCCD, 0x477FE000, 0x3F000000,
    ]
    vals.update(named)
    for k in range(24):
        vals.add(struct.unpack("<I", struct.pack("<f", 1.0 + 2.0 ** -k))[0])
        vals.add(struct.unpack("<I", struct.pack("<f", 2.0 ** k))[0])
        vals.add(struct.unpack("<I", struct.pack("<f", 2.0 ** -k))[0])
    return sorted(vals)


def build_pairs(n_random):
    ops = operands()
    rng = random.Random(2024)
    pairs = []
    for a in ops:
        for b in ops:
            pairs.append((a, b))
    for _ in range(n_random):
        pairs.append((rng.choice(ops), rng.choice(ops)))
        pairs.append((rng.getrandbits(32), rng.getrandbits(32)))
    return pairs


def main(n_random=1500, ops_to_run=None):
    L = BR.make_layout()
    fp.extend_layout_for_fp32(L)
    dim = L.D
    torch.set_grad_enabled(False)

    all_ops = ops_to_run or [isa.F_MUL, isa.F_ADD, isa.F_SUB, isa.F_DIV]
    pairs = build_pairs(n_random)
    grand_ok = True
    for op in all_ops:
        name = isa.NAMES[op]
        blocks = build_op_blocks(L, dim, op)
        ok = 0
        bad = 0
        by_class = {}       # (class_a, class_b) -> [ok, total]
        first_bad = []
        for (a, b) in pairs:
            got = run_op(L, dim, blocks, a, b)
            want = isa.f32_op_bits(op, a, b)
            ca, cb = _classify(a), _classify(b)
            key = (ca, cb)
            slot = by_class.setdefault(key, [0, 0])
            slot[1] += 1
            if got == want or nan_equiv(got, want):
                ok += 1
                slot[0] += 1
            else:
                bad += 1
                if len(first_bad) < 15:
                    first_bad.append(
                        f"  {name} a={a:08x}({_f(a)!r},{ca}) b={b:08x}({_f(b)!r},{cb})"
                        f" -> got={got:08x}({_f(got)!r}) want={want:08x}({_f(want)!r})")
        tot = ok + bad
        print(f"\n=== {name}: {ok}/{tot} megablock bit-exact vs oracle ===")
        if bad:
            grand_ok = False
            # per-class breakdown of where the misses are.
            print("  class breakdown (ok/total), classes with any miss:")
            for key in sorted(by_class):
                o, t = by_class[key]
                if o != t:
                    print(f"    {key[0]:>9} x {key[1]:<9}: {o}/{t}")
            print("  first mismatches:")
            for line in first_bad:
                print(line)
    print("\nGRAND:", "ALL BIT-EXACT" if grand_ok else "MISMATCHES PRESENT")
    return grand_ok


if __name__ == "__main__":
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    only = None
    if len(sys.argv) > 2:
        only = [isa.BY_NAME[x] for x in sys.argv[2:]]
    ok = main(nr, only)
    sys.exit(0 if ok else 1)
