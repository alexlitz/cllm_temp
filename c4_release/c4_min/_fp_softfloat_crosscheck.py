"""Cross-check the F_ADD/F_SUB/F_MUL/F_DIV megablocks against the Berkeley SoftFloat
reference vectors (round-to-nearest-even) at
``/home/alexlitz/Documents/misc/c4_doom/id_port/softfloat/refvectors_small.txt``.

Vector line format (7 fields):  op  a_lo a_hi  b_lo b_hi  res_lo res_hi
(operands/result are 64-bit lo/hi pairs; for f32* the hi halves are 0).

We run the neural megablock stack for each pair and compare to res_lo, treating any
produced NaN as equivalent (SoftFloat emits sign-1 qNaN ``ffc00000`` where x86/our
oracle canonicalises to ``7fc00000``).  Reports per-op bit-exact counts, HONEST on any
class that misses.
"""
from __future__ import annotations

import os
import struct
import sys

os.environ.setdefault("C4_FLOAT_OPS", "1")

import torch  # noqa: E402

from . import isa  # noqa: E402
from . import nibble_fp32 as fp  # noqa: E402
from . import _fp32_blockrun as BR  # noqa: E402

REF = "/home/alexlitz/Documents/misc/c4_doom/id_port/softfloat/refvectors_small.txt"

OPMAP = {
    "f32add": isa.F_ADD,
    "f32sub": isa.F_SUB,
    "f32mul": isa.F_MUL,
    "f32div": isa.F_DIV,
}


def is_nan(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def classify(bits):
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    if e == 0:
        return "zero" if m == 0 else "subnormal"
    if e == 0xFF:
        return "inf" if m == 0 else "nan"
    return "normal"


def build_blocks(L, dim):
    return {
        isa.F_ADD: fp.compile_fp_add_blocks(L, dim),
        isa.F_SUB: fp.compile_fp_sub_blocks(L, dim),
        isa.F_MUL: fp.compile_fp_mul_blocks(L, dim),
        isa.F_DIV: fp.compile_fp_div_blocks(L, dim),
    }


def run(L, dim, blocks, a, b):
    x = BR.new_residual(L, dim)
    BR.seed_bits(x, L.STACK0, a)
    BR.seed_bits(x, L.AX, b)
    x = BR.apply_blocks(x, blocks)
    return BR.read_nibbles(x, L.FP32.F_RES, 8)


def main(limit_per_op=None, only=None):
    L = BR.make_layout()
    fp.extend_layout_for_fp32(L)
    dim = L.D
    torch.set_grad_enabled(False)
    blocks = build_blocks(L, dim)

    # collect vectors (keyed by op int)
    vecs = {op: [] for op in OPMAP.values()}
    for line in open(REF):
        p = line.split()
        if not p or p[0] not in OPMAP:
            continue
        op = OPMAP[p[0]]
        if only and op not in only:
            continue
        a = int(p[1], 16)
        b = int(p[3], 16)
        r = int(p[5], 16)
        vecs[op].append((a, b, r))

    grand = True
    results = {}
    for opname, op in OPMAP.items():
        if only and op not in only:
            continue
        items = vecs[op]
        if limit_per_op:
            items = items[:limit_per_op]
        ok = bad = 0
        by_class = {}
        firstbad = []
        for (a, b, want) in items:
            got = run(L, dim, blocks[op], a, b)
            ca, cb = classify(a), classify(b)
            key = (ca, cb)
            slot = by_class.setdefault(key, [0, 0])
            slot[1] += 1
            match = (got == want) or (is_nan(got) and is_nan(want))
            if match:
                ok += 1
                slot[0] += 1
            else:
                bad += 1
                if len(firstbad) < 12:
                    firstbad.append(
                        f"  {opname} a={a:08x} b={b:08x} -> got={got:08x} want={want:08x} ({ca}x{cb})")
        tot = ok + bad
        results[opname] = (ok, tot)
        print(f"\n=== {opname}: {ok}/{tot} bit-exact vs SoftFloat ===")
        if bad:
            grand = False
            for key in sorted(by_class):
                o, t = by_class[key]
                if o != t:
                    print(f"    {key[0]:>9} x {key[1]:<9}: {o}/{t}")
            for ln in firstbad:
                print(ln)
    print("\nGRAND:", "ALL BIT-EXACT vs SoftFloat" if grand else "MISMATCHES PRESENT")
    return grand, results


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    only = [isa.BY_NAME[x] for x in sys.argv[2:]] if len(sys.argv) > 2 else None
    ok, _ = main(lim, only)
    sys.exit(0 if ok else 1)
