"""Measurement harness for the radix-256 SRT divide (table variant).

Prints the depth (blocks) breakdown, nz weights, fp32-safety, and byte-exact
counts over the edge grid + adversarial classes + random 32-bit pairs, in fp64
AND fp32.  Kept separate from the block builders so the builders stay lean.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from . import div_radix256_srt as R
from .nibble_alu32 import RELU_S, S


def _band_value_bounds(L):
    """Per-DIM |value| upper bound for the fp32 relu-arg audit.  Every band in the
    TABLE variant is nibble/predicate-bounded (<=15) — there is NO 16^p scalar
    recompose.  KB carry columns transiently hold k*d_nib <= 255*15 = 3825 during
    the precompute rounds, so bound KB by 3825; everything else <= 15, ONE = 1."""
    a = L.R256
    b = torch.full((L.D,), 15.0)
    b[L.ONE] = 1.0
    for k in range(R._NK):
        for c in range(a.RN):
            b[a.KB + a.RN * k + c] = 3825.0     # pre-settle carry column bound
    b[a.QD] = 255.0                              # quotient byte
    for c in range(8):
        b[a.DIV_RES + c] = 255.0                 # transiently holds the raw byte
    return b


def _max_relu_arg(unique, L):
    bounds = _band_value_bounds(L)
    max_arg = 0.0
    worst = None
    for name, spec in unique:
        wup = spec["W_up"]
        if not wup.numel():
            continue
        per_unit = wup.abs() @ bounds + spec["b_up"].abs()
        m = float(per_unit.max())
        if m > max_arg:
            max_arg = m
            worst = name
    return max_arg, worst


def _adversarial_cases():
    cases = []
    M = 2 ** 32 - 1
    for d in (2, 3, 7):
        cases += [(M, d), (M - 1, d), (M, d + 1)]
    for k in range(0, 33):
        for delta in (-1, 1):
            div = (1 << k) + delta
            if 1 <= div <= M:
                rng = random.Random(k * 7 + delta)
                cases += [(rng.randint(0, M), div), (div, div), (div - 1, div),
                          (div * 5 + 3, div), (M, div)]
    for a in (0, 1, 2, 12345, 2 ** 31, M):
        cases.append((a, 1))
    rng = random.Random(20260723)
    for _ in range(300):
        b = rng.randint(1, M)
        k = rng.randint(0, 255)
        for a in (k * b, k * b + b - 1, k * b + b):
            cases.append((a & M, b))
    for a in (0, 1, M, 2 ** 31):
        cases.append((a, 0))
    return cases


def _run_batch(L, dim, specs, cases, dtype):
    a = L.R256
    B = len(cases)
    x = torch.zeros(B, dim, dtype=dtype)
    x[:, L.ONE] = 1.0
    for bi, (av, bv) in enumerate(cases):
        for j, nv in enumerate(R._nibbles(bv & R.MASK32, 8)):
            x[bi, L.AX + j] = float(nv)
        for j, nv in enumerate(R._nibbles(av & R.MASK32, 8)):
            x[bi, L.STACK0 + j] = float(nv)
    for spec in specs:
        up = x @ spec["W_up"].T + spec["b_up"]
        gate = x @ spec["W_gate"].T + spec["b_gate"]
        x = x + (F.silu(up) * gate) @ spec["W_down"].T + spec["b_down"]
    out = []
    for bi in range(B):
        q = sum(int(round(float(x[bi, a.DIV_RES + c]))) << (4 * c) for c in range(8))
        r = sum(int(round(float(x[bi, a.MOD_RES + c]))) << (4 * c) for c in range(8))
        out.append((q & R.MASK32, r & R.MASK32))
    return out


def measure(verbose: bool = True, n_random: int = 2000, batch: int = 64):
    L = R._new_layout()
    dim = L.D
    unrolled = R.compile_blocks_unrolled(L, dim)
    unique, apply_names = R.compile_blocks_recurrent(L, dim)

    body = R._iteration_body(L, dim)
    blocks_per_iter = len(body)
    depth_unrolled = len(unrolled)
    depth_applied = len(apply_names)
    stored = len(unique)
    nnz = sum(R._spec_nnz(s) for _, s in unique)
    max_arg, worst = _max_relu_arg(unique, L)
    n_kb = len(R._kb_precompute_blocks(L, dim))

    cases = list(R._edge_grid()) + _adversarial_cases()
    rng = random.Random(1234)
    for _ in range(n_random):
        cases.append((rng.randint(0, 2 ** 32 - 1), rng.randint(0, 2 ** 32 - 1)))
    total = len(cases)

    def _run(dtype):
        by = {n: {k: v.to(dtype) for k, v in s.items()} for n, s in unique}
        specs = [by[n] for n in apply_names]
        p, fs = 0, []
        for i in range(0, total, batch):
            chunk = cases[i:i + batch]
            outs = _run_batch(L, dim, specs, chunk, dtype)
            for (av, bv), (q, r) in zip(chunk, outs):
                rq, rr = R._ref(av, bv)
                if (q, r) == (rq, rr):
                    p += 1
                elif len(fs) < 16:
                    fs.append((av, bv, (q, r), (rq, rr)))
        return p, fs

    passed, fails = _run(torch.float64)
    passed32, fails32 = _run(torch.float32)

    fp32_safe = max_arg < 2 ** 24
    if verbose:
        print("=" * 76)
        print("RADIX-256 SRT (TABLE, byte-at-a-time) 32-bit DIVIDE — bakeoff")
        print("=" * 76)
        print(f"d_model (residual dim)    : {dim}")
        print(f"stored (unique) blocks    : {stored}")
        print(f"depth (UNROLLED, straight): {depth_unrolled} blocks   (GOAL < 40)")
        print(f"depth (applied, recurrent): {depth_applied} block-applications")
        print(f"blocks / iteration        : {blocks_per_iter}  (shift, gteq, qdigit,"
              f" qbsel, gp, ks0..{blocks_per_iter}, apply, emitsplit)")
        print(f"iterations                : {R._NITERS}  (radix-256, byte digits)")
        print(f"one-time overhead         : {n_kb} KB-precompute + 1 init + 1 finalize "
              f"= {n_kb + 2} blocks")
        print("DEPTH BREAKDOWN:")
        n_borrow = len(R._borrow_prefix_blocks(L, dim)[0])
        print(f"  normalize/precompute (KB table + BZ) : {n_kb} blocks")
        print(f"  per-iter select (shift+gteq+qd+qbsel): 4 blocks x {R._NITERS} = {4*R._NITERS}")
        print(f"  per-iter borrow (gp+ks+apply)        : {n_borrow} blocks x {R._NITERS} = {n_borrow*R._NITERS}")
        print(f"  per-iter emit-split                  : 1 block x {R._NITERS} = {R._NITERS}")
        print(f"  init + finalize                      : 2 blocks")
        print(f"nz (nonzero weights)      : {nnz}")
        print(f"max relu arg (worst |up|) : {max_arg:.1f}  @ {worst}  "
              f"(fp32-safe < 2^24 = {2**24}) -> {'YES' if fp32_safe else 'NO'}")
        print(f"byte-exact (fp64 sim)     : {passed}/{total}  "
              f"({'ALL PASS' if passed == total else 'FAIL'})")
        print(f"byte-exact (fp32 sim)     : {passed32}/{total}  "
              f"({'ALL PASS' if passed32 == total else 'RESIDUE FLOOR'})")
        if fails:
            print("  first fp64 fails (a,b,got,exp):")
            for f in fails[:8]:
                print("   ", f)
        if fails32:
            print("  first fp32 fails (a,b,got,exp):")
            for f in fails32[:8]:
                print("   ", f)
        print("-" * 76)
        print(f"VERDICT: radix-256 depth = {depth_unrolled} blocks vs radix-16 lean 80, "
              f"{'UNDER' if depth_unrolled < 40 else 'OVER'} the 40 goal.")
        print("=" * 76)
    return dict(dim=dim, depth_unrolled=depth_unrolled, depth_applied=depth_applied,
                blocks_per_iter=blocks_per_iter, nnz=nnz, max_relu_arg=max_arg,
                fp32_safe=fp32_safe, byte_exact_pass=passed, byte_exact_total=total,
                byte_exact_pass_fp32=passed32)


if __name__ == "__main__":
    measure()
