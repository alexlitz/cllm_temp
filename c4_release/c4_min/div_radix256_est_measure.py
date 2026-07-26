"""Measurement harness for the radix-256 SRT ESTIMATE divide.

Prints depth (with breakdown), nz, fp32-safety, and byte-exact counts over the
edge grid + adversarial classes + random 32-bit pairs in BOTH fp64 and fp32.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from . import div_radix256_est as R
from .nibble_alu32 import RELU_S, S


def _band_value_bounds(L):
    """Per-DIM |value| bound for the fp32 relu-arg audit.  Nibble / predicate bands
    <= 15; the estimate reads a 16-bit Ahat (Rn top 4 nibbles) — the estimate forms
    ``Ahat - k*Bp`` with Bp <= 256 reach ~65280; the qhat*dn / q*d raw columns reach
    ~3825.  Bound the wide lanes accordingly; ONE = 1."""
    a = L.R256E
    b = torch.full((L.D,), 15.0)
    b[L.ONE] = 1.0
    b[a.QHAT] = 255.0
    b[a.BP] = 256.0
    for c in range(8):
        b[a.DIV_RES + c] = 255.0
    # QDRAW columns transiently hold raw products up to ~3825.
    for c in range(a.RN):
        b[a.QDRAW + c] = 3825.0
    return b


def _max_relu_arg(unique, L):
    bounds = _band_value_bounds(L)
    max_arg, worst = 0.0, None
    for name, spec in unique:
        wup = spec["W_up"]
        if not wup.numel():
            continue
        per_unit = wup.abs() @ bounds + spec["b_up"].abs()
        m = float(per_unit.max())
        if m > max_arg:
            max_arg, worst = m, name
    return max_arg, worst


def _edge_grid():
    edges = [0, 1, 2, 2 ** 31, 2 ** 32 - 1]
    edges += [1 << p for p in range(0, 32)]
    bs = [1, 2, 3, 7, 10, 16, 255, 256, 65535, 65536, 2 ** 31, 2 ** 32 - 1]
    cases = []
    for a in edges:
        for b in bs:
            cases.append((a, b))
    for v in edges:
        cases.append((v, v))
        cases.append((v, 0))
        cases.append((max(0, v - 1), v))
    return cases


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
    # divisor near 2^31 (SRT normalize boundary), tiny divisors, dividend<divisor.
    for _ in range(150):
        b = rng.randint(2 ** 30, 2 ** 31 + 2 ** 20)
        cases.append((rng.randint(0, M), b))
    for b in (1, 2, 3, 4, 5, 255, 256):
        for _ in range(20):
            cases.append((rng.randint(0, M), b))
    for a in (0, 1, M, 2 ** 31):
        cases.append((a, 0))
    return cases


def _run_batch(L, dim, specs, cases, dtype):
    a = L.R256E
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


def measure(verbose: bool = True, n_random: int = 4000, batch: int = 128):
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
    n_norm = len(R._normalize_blocks(L, dim))
    n_epi = len(R._epilogue_blocks(L, dim))
    n_borrow = len(R._LIMBS)

    cases = list(_edge_grid()) + _adversarial_cases()
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
        print("=" * 78)
        print("RADIX-256 SRT (ESTIMATE, normalized-domain) 32-bit DIVIDE — bakeoff")
        print("=" * 78)
        print(f"d_model (residual dim)    : {dim}")
        print(f"depth (UNROLLED, straight): {depth_unrolled} blocks   (GOAL < 40; radix-16 lean = 80)")
        print(f"depth (applied, recurrent): {depth_applied} block-applications")
        print(f"blocks / iteration        : {blocks_per_iter}")
        print(f"iterations                : {R._NITERS}  (radix-256, byte digits)")
        print("DEPTH BREAKDOWN:")
        print(f"  normalize (CLZ + dn<<sh + 2dn + Bp + BZ + init): {n_norm} blocks")
        print(f"  per-iter x{R._NITERS}: shift-insert(2) + estimate(1) + q*d(3) + "
              f"lane-form(2) + borrow({n_borrow}) + split(2) + select-emit(2)")
        print(f"                = {blocks_per_iter}/iter -> {blocks_per_iter * R._NITERS} blocks")
        print(f"  epilogue (MOD = a - q*d: mul(3) + borrow(3) + floors(1) + finalize(2)): {n_epi} blocks")
        print(f"nz (nonzero weights)      : {nnz}")
        print(f"max relu arg (worst |up|) : {max_arg:.1f}  @ {worst}  "
              f"(fp32-safe < 2^24 = {2**24}) -> {'YES' if fp32_safe else 'NO'}")
        print(f"byte-exact (fp64 sim)     : {passed}/{total}  "
              f"({'ALL PASS' if passed == total else 'FAIL'})")
        print(f"byte-exact (fp32 sim)     : {passed32}/{total}  "
              f"({'ALL PASS' if passed32 == total else 'RESIDUE FLOOR'})")
        if fails:
            print("  first fp64 fails:", fails[:6])
        if fails32:
            print("  first fp32 fails:", fails32[:6])
        print("-" * 78)
        verdict = "UNDER" if depth_unrolled < 40 else "OVER"
        print(f"VERDICT: variable-DIV depth = {depth_unrolled} blocks -> {verdict} the 40 goal.")
        print("=" * 78)
    return dict(dim=dim, depth_unrolled=depth_unrolled, depth_applied=depth_applied,
                blocks_per_iter=blocks_per_iter, nnz=nnz, max_relu_arg=max_arg,
                fp32_safe=fp32_safe, byte_exact_pass=passed, byte_exact_total=total,
                byte_exact_pass_fp32=passed32)


if __name__ == "__main__":
    measure()
