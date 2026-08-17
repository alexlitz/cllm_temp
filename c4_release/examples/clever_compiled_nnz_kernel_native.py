#!/usr/bin/env python3
r"""clever_compiled_nnz_kernel_native.py — NATIVIZED compiled-nnz kernel.

This is the follow-on to examples/clever_compiled_nnz_kernel.py. That kernel
codegens the clever min-param VM's nonzero compute DAG to native C, but each op's
C body FAITHFULLY REPRODUCES THE MODEL'S INTERMEDIATE ARITHMETIC — MUL is an
fp128 (x86 80-bit long double) depth-20 `powl`-per-place decode chain, DIV/MOD are
fp64 digit-extraction long-division loops, ADD/SUB decode the whole value place by
place. That reproduces the model's slow *method*, and MUL (fp128 powl-per-place) is
the sole heavy op: ~137 K/sec, dragging the mixed average.

THE INSIGHT
-----------
Byte-exactness is a property of the OUTPUT (the register values: the 32-bit result,
the low+high words of the 64-bit product, the quotient/remainder), NOT of the
INTERMEDIATE arithmetic. Every one of these ops is, semantically, an EXACT integer
function of its operands:
  * ADD    = a + b                     (whole value, depth 11 == no 32-bit mask)
  * SUB    = (a - b) mod 2^32
  * MUL    = a * b (full 64-bit)       low = product & M32, high = product >> 32
  * DIV    = a // b ,  MOD = a % b     (integer long division)
  * CMP x6 = sign(signed32(a) - signed32(b))
  * SHL    = (a << n) & M32 ,  SHR = signed32(a) >> n  (arithmetic, mod 2^32)
  * LEA    = (signed32(a) + signed32(b)) mod 2^32
  * OR/XOR/AND = the 8-nibble bitwise result (== the direct a|b / a^b / a&b)
The fp128 powl decode of MUL and the fp64 digit-extraction of DIV/MOD are just slow
ways to COMPUTE those exact integer functions. So the nativized kernel replaces the
slow inner loops with the NATIVE C integer form (`__int128` for the product,
`uint64_t` divmod, native add/sub/mask) that yields the BYTE-IDENTICAL OUTPUT.

WHAT NATIVIZATION CHANGES / DOES NOT CHANGE (honesty gate)
---------------------------------------------------------
* CHANGES: the kernel no longer bit-reproduces the MODEL'S INTERMEDIATE fp
  arithmetic (the powl-per-place / digit-extraction loops). That is the POINT.
* DOES NOT CHANGE: the per-op OUTPUT (AX / registers / low+high MUL words /
  quotient+remainder) is byte-identical to the reference transformer cells
  (ArithCell/BitwiseCell) and hence to the 32-bit c4 reference semantics. We prove
  this N/N over the SAME battery the faithful kernel is checked against, AND we
  cross-check the nativized C against the faithful fp128/fp64 C on the same inputs.
* SCOPE unchanged: still the clever min-param SUBSET (full 32-bit per op, MUL
  byte-exact 64-bit product), NOT the production nibble whole-ISA (golden 174ece66).

Both the faithful and the nativized kernels are BUILT and MEASURED here so the
before->after per-op table is a single-run apples-to-apples comparison.

Run:
    python examples/clever_compiled_nnz_kernel_native.py
    python examples/clever_compiled_nnz_kernel_native.py --n 100000 --threads 8
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import subprocess
import tempfile
import time

import numpy as np

LN10 = math.log(10.0)
M32 = (1 << 32) - 1
SIGN32 = 1 << 31

# Opcode ids handed to the compiled kernel (must match the C switch).
OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD = 0, 1, 2, 3, 4
OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE = 5, 6, 7, 8, 9, 10
OP_SHL, OP_SHR, OP_LEA = 11, 12, 13
OP_OR, OP_XOR, OP_AND = 14, 15, 16
OP_NAMES = {OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL", OP_DIV: "DIV",
            OP_MOD: "MOD", OP_EQ: "EQ", OP_NE: "NE", OP_LT: "LT", OP_GT: "GT",
            OP_LE: "LE", OP_GE: "GE", OP_SHL: "SHL", OP_SHR: "SHR",
            OP_LEA: "LEA", OP_OR: "OR", OP_XOR: "XOR", OP_AND: "AND"}
ARITH_OPS = (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD)
CMP_OPS = (OP_EQ, OP_NE, OP_LT, OP_GT, OP_LE, OP_GE)
SHIFT_OPS = (OP_SHL, OP_SHR)
BITWISE_OPS = (OP_OR, OP_XOR, OP_AND)


# ============================================================================ #
# (A) FAITHFUL kernel — bit-reproduces the model's intermediate arithmetic.
#     This is the prior kernel's C body (fp128 powl MUL, fp64 divmod loops).
# ============================================================================ #
C_SOURCE_FAITHFUL = r"""
#include <stdint.h>
#include <math.h>

#define M32 4294967295.0
#define LN10 2.302585092994045901
#define SIGN32 2147483648LL

static int decode_digit(double value) {
    int best = 0; double best_logit = -1e300;
    for (int c = 0; c < 10; c++) {
        double logit = -fabs(value - ((double)c + 0.5)) + 1e-12 * (double)c;
        if (logit > best_logit) { best_logit = logit; best = c; }
    }
    return best;
}
static int64_t decode_whole(double S, int depth) {
    double R = S; int64_t out = 0;
    for (int p = depth - 1; p >= 0; p--) {
        double scale = pow(10.0, (double)p);
        int d = decode_digit(R / scale);
        out += (int64_t)d * (int64_t)llround(scale);
        R -= (double)d * scale;
    }
    return out;
}
static double signed32(int64_t x) {
    double xf = (double)x;
    return (x >= SIGN32) ? xf - 4294967296.0 : xf;
}
static void decode_mul(uint64_t a, uint64_t b, uint64_t *out) {
    long double R = (long double)a * (long double)b;
    unsigned long long res = 0;
    for (int p = 19; p >= 0; p--) {
        long double place = powl(10.0L, (long double)p);
        long double val = R / place;
        int best = 0; long double best_logit = -1e300L;
        for (int c = 0; c < 10; c++) {
            long double logit = -fabsl(val - ((long double)c + 0.5L)) + 1e-15L * (long double)c;
            if (logit > best_logit) { best_logit = logit; best = c; }
        }
        unsigned long long pw = 1ULL;
        for (int k = 0; k < p; k++) pw *= 10ULL;
        res += (unsigned long long)best * pw;
        R -= (long double)best * place;
    }
    *out = res;
}
static int nib_lut(int op, int na, int nb) {
    if (op == 2) return na & nb;
    if (op == 1) return na ^ nb;
    return na | nb;
}
int64_t vm_step(int op, uint64_t a, uint64_t b, uint64_t n, uint64_t *out_hi) {
    *out_hi = 0;
    switch (op) {
        case 0: { return decode_whole((double)a + (double)b, 11); }
        case 1: { double S = (double)a - (double)b; if (S < 0) S += 4294967296.0; return decode_whole(S, 11); }
        case 2: { uint64_t prod; decode_mul(a, b, &prod); *out_hi = prod >> 32; return (int64_t)(prod & 0xFFFFFFFFULL); }
        case 3:
        case 4: {
            double R = (double)a; double bf = (double)b; int64_t q = 0;
            for (int p = 9; p >= 0; p--) {
                double place = pow(10.0, (double)p);
                double bp = bf * place;
                double val = R / bp;
                int d = decode_digit(val);
                unsigned long long pw = 1ULL; for (int k=0;k<p;k++) pw*=10ULL;
                q += (int64_t)d * (int64_t)pw;
                R -= (double)d * bp;
            }
            if (op == 3) return q;
            return (int64_t)llround(R);
        }
        case 5: case 6: case 7: case 8: case 9: case 10: {
            double delta = signed32((int64_t)a) - signed32((int64_t)b);
            int lt = delta < 0, eq = delta == 0, gt = delta > 0;
            switch (op) {
                case 5: return eq; case 6: return !eq; case 7: return lt;
                case 8: return gt; case 9: return lt || eq; case 10: return gt || eq;
            }
        }
        case 11: {
            double af = (double)a; double two_n = pow(2.0, (double)n);
            double prod = af * two_n; double mod = 4294967296.0;
            double res = prod - floor(prod / mod) * mod; return (int64_t)llround(res);
        }
        case 12: {
            double a_s = signed32((int64_t)a); double two_n = pow(2.0, (double)n);
            double q = floor(a_s / two_n); double mod = 4294967296.0;
            if (q < 0) q += mod; return (int64_t)llround(q);
        }
        case 13: {
            double s = signed32((int64_t)a) + signed32((int64_t)b); double mod = 4294967296.0;
            double r = s - floor(s / mod) * mod; return (int64_t)llround(r);
        }
        case 14: case 15: case 16: {
            int kop = (op == 16) ? 2 : (op == 15) ? 1 : 0;
            int64_t out = 0;
            for (int k = 0; k < 8; k++) {
                int na = (int)((a >> (4*k)) & 0xF); int nb = (int)((b >> (4*k)) & 0xF);
                out |= ((int64_t)nib_lut(kop, na, nb) << (4*k));
            }
            return out;
        }
    }
    return 0;
}
void vm_run_batch(const int32_t *ops, const uint64_t *a, const uint64_t *b,
                  const uint64_t *n, int64_t *results, uint64_t *results_hi,
                  int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        uint64_t hi;
        results[i] = vm_step(ops[i], a[i], b[i], n[i], &hi);
        results_hi[i] = hi;
    }
}
"""


# ============================================================================ #
# (B) NATIVIZED kernel — each op is the NATIVE C integer form producing the
#     BYTE-IDENTICAL OUTPUT. No fp128 powl, no digit-extraction loops.
# ============================================================================ #
# Per op, mapping to the exact integer function (see module docstring):
#   ADD  : (int64_t)(a + b)                      (whole value; NO 32-bit mask,
#                                                  matches decode_whole depth 11)
#   SUB  : (a - b) mod 2^32                       (matches sub's +2^32 wrap)
#   MUL  : full = (unsigned __int128)a * b -> the product is < 2^64 for 32-bit
#          operands, so lo = full & M32, hi = full >> 32. (__int128 keeps it
#          exact; identical to the fp128 whole product.)
#   DIV  : a / b (unsigned) ,  MOD : a % b        (b != 0 by battery construction)
#   CMP  : sign of signed32(a)-signed32(b) via int64 (exact, no fp)
#   SHL  : (a << n) & M32                          (n in 0..31; a<<n as uint64)
#   SHR  : arithmetic signed32(a) >> n, then mod 2^32
#   LEA  : (int64)(signed32(a)+signed32(b)) & M32
#   OR/XOR/AND : the direct 32-bit bitwise op (the 8-nibble LUT collapses to it)
# The bitwise / CMP / LEA ops in the faithful kernel were already integer/LUT
# (fast), but we nativize them too so the WHOLE kernel is native-integer and the
# per-op table is uniform. SHL/SHR were fp (pow(2,n)+floor); nativized to shifts.
C_SOURCE_NATIVE = r"""
#include <stdint.h>

#define M32 0xFFFFFFFFULL
#define SIGN32 2147483648LL

int64_t vm_step(int op, uint64_t a, uint64_t b, uint64_t n, uint64_t *out_hi) {
    *out_hi = 0;
    switch (op) {
        case 0: { /* ADD: whole a+b (no 32-bit mask; matches depth-11 decode) */
            return (int64_t)(a + b);
        }
        case 1: { /* SUB: (a - b) mod 2^32 */
            uint64_t s = (a - b) & M32;
            return (int64_t)s;
        }
        case 2: { /* MUL: native full 64-bit product; lo & hi words */
            unsigned __int128 full = (unsigned __int128)a * (unsigned __int128)b;
            uint64_t prod = (uint64_t)full;              /* low 64 bits (exact, <2^64) */
            *out_hi = (prod >> 32) & M32;
            return (int64_t)(prod & M32);
        }
        case 3: { /* DIV: integer quotient */
            return (int64_t)(a / b);
        }
        case 4: { /* MOD: integer remainder */
            return (int64_t)(a % b);
        }
        case 5: case 6: case 7: case 8: case 9: case 10: { /* CMP x6: signed */
            int64_t as = (a >= (uint64_t)SIGN32) ? (int64_t)a - 4294967296LL : (int64_t)a;
            int64_t bs = (b >= (uint64_t)SIGN32) ? (int64_t)b - 4294967296LL : (int64_t)b;
            int lt = as < bs, eq = as == bs, gt = as > bs;
            switch (op) {
                case 5: return eq; case 6: return !eq; case 7: return lt;
                case 8: return gt; case 9: return lt || eq; case 10: return gt || eq;
            }
        }
        case 11: { /* SHL: (a << n) & M32 */
            uint64_t res = (n >= 64) ? 0ULL : ((a << n) & M32);
            return (int64_t)res;
        }
        case 12: { /* SHR: arithmetic signed32(a) >> n, mod 2^32 */
            int64_t a_s = (a >= (uint64_t)SIGN32) ? (int64_t)a - 4294967296LL : (int64_t)a;
            int64_t q = (n >= 63) ? (a_s >> 63) : (a_s >> n);   /* arithmetic shift */
            uint64_t r = ((uint64_t)q) & M32;
            return (int64_t)r;
        }
        case 13: { /* LEA: (signed32(a)+signed32(b)) mod 2^32 */
            int64_t as = (a >= (uint64_t)SIGN32) ? (int64_t)a - 4294967296LL : (int64_t)a;
            int64_t bs = (b >= (uint64_t)SIGN32) ? (int64_t)b - 4294967296LL : (int64_t)b;
            uint64_t r = ((uint64_t)(as + bs)) & M32;
            return (int64_t)r;
        }
        case 14: { /* OR  */ return (int64_t)((a | b) & M32); }
        case 15: { /* XOR */ return (int64_t)((a ^ b) & M32); }
        case 16: { /* AND */ return (int64_t)((a & b) & M32); }
    }
    return 0;
}
void vm_run_batch(const int32_t *ops, const uint64_t *a, const uint64_t *b,
                  const uint64_t *n, int64_t *results, uint64_t *results_hi,
                  int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        uint64_t hi;
        results[i] = vm_step(ops[i], a[i], b[i], n[i], &hi);
        results_hi[i] = hi;
    }
}
"""


# ============================================================================ #
# Codegen + build.
# ============================================================================ #
def build_kernel(workdir: str, src_text: str, tag: str, extra_cflags=()):
    src = os.path.join(workdir, f"clever_nnz_{tag}.c")
    so = os.path.join(workdir, f"clever_nnz_{tag}.so")
    with open(src, "w") as f:
        f.write(src_text)
    cflags = ["-O3", "-march=native", "-fPIC", "-shared", "-ffp-contract=off",
              "-fno-fast-math", "-static-libgcc"]
    cflags += list(extra_cflags)
    cmd = ["gcc"] + cflags + [src, "-o", so, "-lm"]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    build_s = time.perf_counter() - t0
    lib = ctypes.CDLL(so)
    lib.vm_step.restype = ctypes.c_int64
    lib.vm_step.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.vm_run_batch.restype = None
    lib.vm_run_batch.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int64]
    return lib, build_s, so


def run_batch(lib, ops, a, b, n):
    count = ops.shape[0]
    ops32 = np.ascontiguousarray(ops, dtype=np.int32)
    a64 = np.ascontiguousarray(a, dtype=np.uint64)
    b64 = np.ascontiguousarray(b, dtype=np.uint64)
    n64 = np.ascontiguousarray(n, dtype=np.uint64)
    res = np.zeros(count, dtype=np.int64)
    res_hi = np.zeros(count, dtype=np.uint64)
    lib.vm_run_batch(
        ops32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        a64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        b64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        n64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        res_hi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        ctypes.c_int64(count))
    return res, res_hi


# ============================================================================ #
# REFERENCE: the byte-exact clever transformer cells (torch).
# ============================================================================ #
def reference_forward(ops, a, b, n):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from examples.clever_realtime_cells import ArithCell, BitwiseCell
    arith = ArithCell(torch.float64)
    bit = {"OR": BitwiseCell("OR"), "XOR": BitwiseCell("XOR"), "AND": BitwiseCell("AND")}
    count = ops.shape[0]
    res = np.zeros(count, dtype=np.int64)
    res_hi = np.zeros(count, dtype=np.uint64)
    ta = torch.from_numpy(a.astype(np.int64))
    tb = torch.from_numpy(b.astype(np.int64))
    tn = torch.from_numpy(n.astype(np.int64))
    for op in np.unique(ops):
        idx = np.where(ops == op)[0]
        if idx.size == 0:
            continue
        sa, sb, sn = ta[idx], tb[idx], tn[idx]
        if op == OP_ADD:
            r = arith.add(sa, sb)
        elif op == OP_SUB:
            r = arith.sub(sa, sb)
        elif op == OP_MUL:
            prod = arith.mul(sa, sb)
            r = np.array([int(x) & M32 for x in prod], dtype=np.int64)
            res_hi[idx] = np.array([(int(x) >> 32) & M32 for x in prod], dtype=np.uint64)
            res[idx] = r
            continue
        elif op == OP_DIV:
            q, _ = arith.divmod(sa, sb)
            r = q
        elif op == OP_MOD:
            _, rem = arith.divmod(sa, sb)
            r = rem
        elif op in CMP_OPS:
            r = arith.cmp(sa, sb, OP_NAMES[op])
        elif op == OP_SHL:
            r = arith.shift(sa, sn, "SHL")
        elif op == OP_SHR:
            r = arith.shift(sa, sn, "SHR")
        elif op == OP_LEA:
            r = arith.lea(sa, sb)
        elif op in BITWISE_OPS:
            r = bit[OP_NAMES[op]](sa, sb)
        else:
            raise ValueError(op)
        res[idx] = np.asarray(r if not hasattr(r, "numpy") else r.numpy(), dtype=np.int64)
    return res, res_hi


# ============================================================================ #
# Battery generation.
# ============================================================================ #
def make_battery(n, rng):
    ops_pool = list(OP_NAMES.keys())
    ops = rng.choice(ops_pool, size=n).astype(np.int64)
    a = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    n_sh = rng.integers(0, 32, size=n, dtype=np.uint64)
    divmod_mask = (ops == OP_DIV) | (ops == OP_MOD)
    zero_b = divmod_mask & (b == 0)
    b[zero_b] = 1
    # inject edge operands (full 32-bit boundaries)
    edges_a = np.array([0, 1, M32, SIGN32, 0x7FFFFFFF, 0xAAAAAAAA, 2, 100], dtype=np.uint64)
    edges_b = np.array([M32, 1, 0, 0, 1, 0x55555555, 3, 7], dtype=np.uint64)
    k = min(edges_a.size, n)
    a[:k] = edges_a[:k]; b[:k] = edges_b[:k]
    return ops, a, b, n_sh


def make_op_battery(op, n, rng):
    """A single-op battery (for per-op throughput). Full 32-bit operands; for
    MUL both operands full 32-bit so the 64-bit product is exercised; for DIV/MOD
    b!=0; boundaries injected."""
    ops = np.full(n, op, dtype=np.int64)
    a = rng.integers(0, 1 << 32, size=n, dtype=np.uint64)
    b = rng.integers(1, 1 << 32, size=n, dtype=np.uint64)  # b>=1 (safe for div/mod)
    n_sh = rng.integers(0, 32, size=n, dtype=np.uint64)
    edges_a = np.array([0, 1, M32, SIGN32, 0x7FFFFFFF, 0xAAAAAAAA, 2, 100], dtype=np.uint64)
    edges_b = np.array([M32, 1, 1, 1, 1, 0x55555555, 3, 7], dtype=np.uint64)
    k = min(edges_a.size, n)
    a[:k] = edges_a[:k]; b[:k] = edges_b[:k]
    return ops, a, b, n_sh


def _rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def _time_batch(lib, ops, a, b, n, reps):
    run_batch(lib, ops, a, b, n)  # warm
    best = 1e300
    for _ in range(reps):
        t0 = time.perf_counter()
        run_batch(lib, ops, a, b, n)
        best = min(best, time.perf_counter() - t0)
    return ops.shape[0] / best


def _bench_multithread(so_path, ops, a, b, n, threads, reps):
    import threading
    count = ops.shape[0]
    ops32 = np.ascontiguousarray(ops, dtype=np.int32)
    a64 = np.ascontiguousarray(a, dtype=np.uint64)
    b64 = np.ascontiguousarray(b, dtype=np.uint64)
    n64 = np.ascontiguousarray(n, dtype=np.uint64)
    res = np.zeros(count, dtype=np.int64)
    res_hi = np.zeros(count, dtype=np.uint64)
    libs = []
    for _ in range(threads):
        L = ctypes.CDLL(so_path)
        L.vm_run_batch.restype = None
        L.vm_run_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int64]
        libs.append(L)
    bounds = [(i * count // threads, (i + 1) * count // threads) for i in range(threads)]

    def worker(t):
        lo, hi = bounds[t]
        if hi <= lo:
            return
        cnt = hi - lo
        libs[t].vm_run_batch(
            ops32[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            a64[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            b64[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            n64[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            res[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            res_hi[lo:hi].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_int64(cnt))

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for th in ths: th.start()
    for th in ths: th.join()
    best = 1e300
    for _ in range(reps):
        t0 = time.perf_counter()
        ths = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
        for th in ths: th.start()
        for th in ths: th.join()
        best = min(best, time.perf_counter() - t0)
    return count / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90000, help="mixed battery size")
    ap.add_argument("--per-op-n", type=int, default=200000, help="per-op battery size")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    peak_rss = _rss_mb()
    print("=" * 96)
    print("NATIVIZED COMPILED-NNZ KERNEL — byte-exact OUTPUT, native-C integer ops")
    print("=" * 96)

    # ---- build both kernels ----
    workdir = tempfile.mkdtemp(prefix="clever_nnz_nat_")
    lib_f, build_f, so_f = build_kernel(workdir, C_SOURCE_FAITHFUL, "faithful")
    lib_n, build_n, so_n = build_kernel(workdir, C_SOURCE_NATIVE, "native")
    print(f"\n[1] BUILD  faithful (fp128 powl / fp64 divmod loops): {build_f*1000:.0f} ms, "
          f"{os.path.getsize(so_f)} B")
    print(f"    BUILD  nativized (native-C integer ops)          : {build_n*1000:.0f} ms, "
          f"{os.path.getsize(so_n)} B")

    # ---- mixed battery ----
    ops, a, b, n = make_battery(args.n, rng)
    print(f"\n[2] MIXED BATTERY: {args.n:,} steps across all {len(OP_NAMES)} opcodes "
          f"(full 32-bit operands + boundary edges)")

    # ---- byte-exact verification: nativized vs reference cells, and vs faithful ----
    kres_n, khi_n = run_batch(lib_n, ops, a, b, n)
    kres_f, khi_f = run_batch(lib_f, ops, a, b, n)
    rres, rhi = reference_forward(ops, a, b, n)
    match_nat_ref = (kres_n == rres) & (khi_n == rhi)
    match_nat_faith = (kres_n == kres_f) & (khi_n == khi_f)
    n_ok_ref = int(match_nat_ref.sum())
    n_ok_faith = int(match_nat_faith.sum())
    print(f"\n[3] OUTPUT-BYTE-EXACT (nativized kernel):")
    print(f"    nativized vs REFERENCE cells (ArithCell/BitwiseCell): "
          f"{n_ok_ref:,}/{args.n:,} {'ALL EXACT' if n_ok_ref==args.n else 'MISMATCH'}")
    print(f"    nativized vs FAITHFUL fp128/fp64 kernel             : "
          f"{n_ok_faith:,}/{args.n:,} {'ALL EXACT' if n_ok_faith==args.n else 'MISMATCH'}")
    if n_ok_ref != args.n:
        bad = np.where(~match_nat_ref)[0][:10]
        for i in bad:
            print(f"      step {i}: op={OP_NAMES[ops[i]]} a={a[i]} b={b[i]} n={n[i]} "
                  f"nat={kres_n[i]}(hi{khi_n[i]}) ref={rres[i]}(hi{rhi[i]})")
    # per-op breakdown of the exactness
    print(f"    per-op exactness (nativized vs reference):")
    for op in sorted(OP_NAMES):
        idx = np.where(ops == op)[0]
        if idx.size == 0:
            continue
        ok = int(((kres_n[idx] == rres[idx]) & (khi_n[idx] == rhi[idx])).sum())
        print(f"        {OP_NAMES[op]:4s}: {ok:,}/{idx.size:,}")
    peak_rss = max(peak_rss, _rss_mb())

    # ---- per-op throughput: faithful -> nativized (single thread) ----
    print(f"\n[4] PER-OP THROUGHPUT (single-thread, {args.per_op_n:,} steps/op, "
          f"best of {args.reps}):")
    print(f"    {'op':4s} {'faithful (ips)':>18s} {'nativized (ips)':>18s} {'speedup':>10s}")
    per_op = {}
    for op in sorted(OP_NAMES):
        o2, a2, b2, n2 = make_op_battery(op, args.per_op_n, rng)
        ips_f = _time_batch(lib_f, o2, a2, b2, n2, args.reps)
        ips_n = _time_batch(lib_n, o2, a2, b2, n2, args.reps)
        per_op[op] = (ips_f, ips_n)
        print(f"    {OP_NAMES[op]:4s} {ips_f:>18,.0f} {ips_n:>18,.0f} {ips_n/ips_f:>9,.1f}x")
    peak_rss = max(peak_rss, _rss_mb())

    # ---- mixed-battery throughput single + multi ----
    faith_single = _time_batch(lib_f, ops, a, b, n, args.reps)
    nat_single = _time_batch(lib_n, ops, a, b, n, args.reps)
    faith_multi = _bench_multithread(so_f, ops, a, b, n, args.threads, args.reps)
    nat_multi = _bench_multithread(so_n, ops, a, b, n, args.threads, args.reps)
    print(f"\n[5] MIXED-BATTERY THROUGHPUT ({args.n:,} steps, all ops):")
    print(f"    {'':22s} {'faithful':>16s} {'nativized':>16s} {'speedup':>10s}")
    print(f"    {'single-thread (ips)':22s} {faith_single:>16,.0f} {nat_single:>16,.0f} "
          f"{nat_single/faith_single:>9,.1f}x")
    print(f"    {args.threads}-thread (ips){'':10s} {faith_multi:>16,.0f} {nat_multi:>16,.0f} "
          f"{nat_multi/faith_multi:>9,.1f}x")

    peak_rss = max(peak_rss, _rss_mb())
    print(f"\n[6] PEAK RSS: {peak_rss:.0f} MB  ({'OK <4GB' if peak_rss < 4000 else 'ABORT >4GB'})")
    assert peak_rss < 4000, "RSS exceeded 4 GB"

    # ---- ladder placement ----
    py_interp = 1.3e6
    c_port = 540e6
    print(f"\n[7] SPEED LADDER (byte-exact model computation, single-thread mixed):")
    print(f"    faithful fp128 compiled kernel (prior)   : {faith_single:>14,.0f} ips")
    print(f"    NATIVIZED compiled kernel (THIS)         : {nat_single:>14,.0f} ips  "
          f"(single) / {nat_multi:,.0f} ({args.threads}-thread)")
    print(f"    native Python c4 interpreter (prior)     : {py_interp:>14,.0f} ips")
    print(f"    native C c4 port (prior)                 : {c_port:>14,.0f} ips")
    gap_before = c_port / faith_single
    gap_after = c_port / nat_single
    print(f"\n    gap to the native C port (single-thread):")
    print(f"      faithful  : {gap_before:>8,.0f}x slower than the C port")
    print(f"      nativized : {gap_after:>8,.1f}x slower than the C port")
    print(f"      -> nativization closed the gap {gap_before/gap_after:,.0f}x "
          f"(from {gap_before:,.0f}x down to {gap_after:,.1f}x).")
    print(f"      vs native Python interpreter: nativized single is "
          f"{nat_single/py_interp:.2f}x its speed "
          f"({'FASTER' if nat_single>py_interp else 'slower'}).")

    return {"faith_single": faith_single, "nat_single": nat_single,
            "faith_multi": faith_multi, "nat_multi": nat_multi,
            "n_ok_ref": n_ok_ref, "n_ok_faith": n_ok_faith,
            "per_op": per_op, "peak_rss": peak_rss}


if __name__ == "__main__":
    main()
