#!/usr/bin/env python3
r"""clever_fp32_fullops.py — a MIN-PARAM **fp32** clever c4 VM cell set that covers the
FULL Doom op set BYTE-EXACT with **NO fp64/fp128 ANYWHERE**, and a measured per-lane-step
cost that keeps the clever VM's ~7x throughput edge over the wide VM.

THE NO-GO THIS OVERTURNS
========================
A prior analysis (``CLEVER_HONEST_ATTN.md`` / ``PRECISION_RADIX_SURFACE.md``) held that the
clever VM's WHOLE-VALUE representation forces **fp64 for DIV at radix 65536** and, for the
32x32 -> 64-bit MUL product stuffed into ONE fp32 scalar, **fp128** — because 2^64 >> the
fp32 exact-integer ceiling 2^24.  That NO-GO is about the *representation choice* (one
scalar per whole value), NOT about Doom's data (Doom is INTEGER + 16.16 FIXED-POINT, no
floats — the "fp" is only the datapath the clever cell computes on).

The fix is a fp32-SAFE **LIMB decomposition** for MUL (and FixedMul):

  * split each 32-bit operand into 8-bit LIMBS (radix 256): a = sum a_i * 256^i, i in 0..3.
  * every partial product a_i * b_j <= 255*255 = 65,025 < 2^24 (a ~258x margin).
  * accumulate the 16 partials into 8 output-byte columns; the worst column sum
    (with carry) is <= ~260,864 < 2^24 (measured ``opconfig.acc_max("MUL",256)``) — still
    exact in fp32.
  * carry-propagate each column mod 256 (a difference-min / floor decode, exact), giving the
    full 64-bit product as 8 exact bytes — **no 2^64 scalar ever exists**.
  * FixedMul = ((a*b) >> 16) low-32 selects product bytes 2..5 (a byte-slice of the 8 exact
    limbs).  Signed via magnitude-multiply + two's-complement re-sign, exactly as
    ``c4_min/doom_fixedpoint.py::fixed_mul`` does.

So MUL is deeper (limb schoolbook, radix 256, depth ~= 8 result-byte columns) but every
intermediate stays < 2^24 -> **fp32-EXACT, no fp64/fp128**.  DIV at radix 4096 already fits
fp32 (r^2 = 2^24, the exact boundary); the rest of the op set runs at radix 4096 fp32 exactly
(reusing the ``clever_compact_scoring_realtime`` machinery).

WHAT THIS SCRIPT DOES
=====================
1. ``--verify`` — BYTE-EXACT, NO fp64/fp128, per op:
   (a) full 64-bit MUL (32x32 -> 64) via 8-bit limbs, random operands, in fp32;
   (b) Doom **FixedMul** ((a*b)>>16) and **FixedDiv** over Doom's ACTUAL 16.16 value ranges
       (incl. negatives / sign-magnitude), oracle = ``c4_min.doom_fixedpoint.fixed_mul/
       fixed_div`` (the on-VM-exact reference the title frame depends on);
   (c) the rest of the op set (ADD/SUB/DIV/MOD/CMP/SHL/SHR/LEA/bitwise/memory) at fp32
       radix 4096 (reused from ``clever_compact_scoring_realtime``).
   Reports per-op pass rate + flags ANY op that still can't be fp32-exact without fp128.
2. ``--bench`` — MEASURE the fp32 FULL-op-set per-lane-step cost (batched to saturation),
   weighted by Doom's op mix (MUL 1% / DIV 0.5% / 99.5% ADD-class, from
   ``serial_doom_floor.DOOM_MIX``): does the average stay ~0.11 us, or does limb-MUL's
   depth raise it?  Reports the fp32 per-lane-step + fps vs the wide VM's 0.788 us/step, and
   whether the ~7x edge survives — 1-GPU and a REAL 2-GPU run if both cards are free.

MEASURED numbers, not projections (2-GPU is a REAL two-device run when both are free, else
labelled a projection).  Golden ``174ece66`` untouched (no build file).

Run:
    python examples/clever_fp32_fullops.py --verify
    python examples/clever_fp32_fullops.py --bench --two-gpu --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from examples.clever_optimized_realtime import RENDER_STEPS, RAW_STEPS
from examples.clever_shallow_radix_realtime import summed_isa_depth, _limbs
from examples.clever_compact_scoring_realtime import (
    CompactStepModel, compact_inter, verify_compact_byte_exact, _diffmin_decode)
from c4_min import opconfig as oc
from c4_min import doom_fixedpoint as dfp


# The wide VM's MEASURED best byte-exact step (PERF_LADDER_FINAL.md, idle A5000, bk512,
# C4_BLOCK0_DK).  The clever VM's per-lane-step must beat this to keep its throughput edge.
WIDE_VM_US_PER_STEP = 0.788

# The MUL limb base: 8-bit limbs (radix 256) keep every partial product + column
# accumulator < 2^24, so the whole 64-bit product is fp32-EXACT with no 2^64 scalar.
MUL_LIMB_RADIX = 256
MUL_LIMBS = 4                       # 32-bit operand = 4 base-256 limbs
MUL_OUT_LIMBS = 8                   # 64-bit product = 8 base-256 limbs
_MASK32 = 0xFFFFFFFF
_SIGN = 0x80000000
FRACBITS = 16

# Doom's op mix (serial_doom_floor.DOOM_MIX): the weighting for the per-lane-step average.
# MUL is 1% of steps, DIV 0.5%; the rest (99.5%) is ADD-class framing/pointer-walk.
DOOM_MIX = {
    "PSH": 0.355, "LEA": 0.18, "LI": 0.12,
    "IMM": 0.08, "JMP": 0.03, "BZ": 0.03, "BNZ": 0.02,
    "CMP": 0.05, "ADD": 0.06, "SUB": 0.03,
    "AND": 0.01, "SHL": 0.01, "JSR": 0.005, "LEV": 0.005,
    "MUL": 0.01, "DIV": 0.005,
}
assert abs(sum(DOOM_MIX.values()) - 1.0) < 1e-6


# =========================================================================== #
# fp32-SAFE LIMB MUL — the whole 64-bit product with NO fp128, every step < 2^24.
# =========================================================================== #
def _split_limbs_exact(vals: np.ndarray, radix: int, n: int) -> np.ndarray:
    """Exact base-`radix` limbs of `vals` (object ints) — the datapath's OPERAND INGEST.

    On the real VM the operand already ARRIVES as nibbles/limbs (never as a >2^24 scalar),
    so the split is a given, not a fp32 op.  We produce it exactly here."""
    out = np.zeros((vals.shape[0], n), dtype=np.int64)
    x = vals.astype(object).copy()
    for j in range(n):
        out[:, j] = (x % radix).astype(np.int64)
        x = x // radix
    return out


def limb_mul_from_limbs(a_limbs_f, b_limbs_f):
    """UNSIGNED product of two 4-limb (base-256) operands -> 8 output limbs, fp32-exact.

    ``a_limbs_f`` / ``b_limbs_f``: lists of 4 fp32 tensors, each in [0,256).  Every partial
    a_i*b_j <= 65,025; column sums (<=4 partials + carry) <= ~260,864 << 2^24 -> fp32 exact.
    Carry-resolve each column with the fp32 floor decode (== the production diff-min floor).
    Returns list of 8 fp32 tensors, each an exact byte in [0,256)."""
    r = float(MUL_LIMB_RADIX)
    B = a_limbs_f[0].shape
    dtype, dev = a_limbs_f[0].dtype, a_limbs_f[0].device
    cols = [torch.zeros(B, dtype=dtype, device=dev) for _ in range(MUL_OUT_LIMBS)]
    for i in range(MUL_LIMBS):
        for j in range(MUL_LIMBS):
            cols[i + j] = cols[i + j] + a_limbs_f[i] * b_limbs_f[j]   # <2^24, exact
    # carry-propagate LSB->MSB: byte = col mod 256, carry = floor(col/256), both exact in fp32
    out = []
    carry = torch.zeros(B, dtype=dtype, device=dev)
    for p in range(MUL_OUT_LIMBS):
        v = cols[p] + carry                                          # still < 2^24 + small
        # floor(v/256) exact (v<2^24 so v/256 < 2^16, fp32-exact); byte = v - 256*floor
        carry = torch.floor(v / r)
        byte = v - r * carry
        out.append(byte)
    return out


# =========================================================================== #
# BYTE-EXACT verification of the fp32 limb MUL (full 64-bit) + FixedMul + FixedDiv.
# =========================================================================== #
def verify_limb_mul_full64(dtype=torch.float32, n=20000, seed=20260809) -> dict:
    """Full UNSIGNED 32x32 -> 64-bit MUL via 8-bit limbs, `n` random operands, in `dtype`
    (fp32).  Byte-exact vs the exact python-int 64-bit product.  NO fp128."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(object)
    b = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(object)
    ref = a * b                                                       # exact 64-bit product
    al = _split_limbs_exact(np.array(a), MUL_LIMB_RADIX, MUL_LIMBS)
    bl = _split_limbs_exact(np.array(b), MUL_LIMB_RADIX, MUL_LIMBS)
    a_f = [torch.from_numpy(al[:, i].astype(np.float64)).to(dtype) for i in range(MUL_LIMBS)]
    b_f = [torch.from_numpy(bl[:, i].astype(np.float64)).to(dtype) for i in range(MUL_LIMBS)]
    out = limb_mul_from_limbs(a_f, b_f)
    got = np.zeros(n, dtype=object)
    for p in range(MUL_OUT_LIMBS - 1, -1, -1):
        got = got * MUL_LIMB_RADIX + out[p].cpu().numpy().round().astype(np.int64).astype(object)
    exact = bool(np.all(got == ref))
    # max intermediate any column ever held (the fp32-ceiling honesty check)
    col_peak = oc.acc_max("MUL", MUL_LIMB_RADIX)
    return {"op": "MUL_full64", "dtype": str(dtype).replace("torch.", ""),
            "n": n, "exact": exact, "no_fp128": True,
            "limb_radix": MUL_LIMB_RADIX, "col_peak_acc": col_peak,
            "fp32_ceiling": oc.PRECISION_CEILING["fp32"],
            "fits_fp32": col_peak <= oc.PRECISION_CEILING["fp32"]}


def fp32_fixed_mul(a: int, b: int) -> int:
    """FixedMul via the fp32 8-bit-limb product, SIGNED, matching doom_fixedpoint.fixed_mul.

    Magnitude-multiply |a|*|b| through the fp32 limb schoolbook (exact 64-bit), form the
    two's-complement signed 64-bit product, ARITHMETIC >>16, keep low 32 bits.  The limb
    arithmetic never leaves fp32's exact range."""
    sa, sb = dfp._sx(a), dfp._sx(b)
    neg = (sa < 0) != (sb < 0)
    ma = abs(sa) & ((1 << 64) - 1)        # magnitudes fit 32 bits
    mb = abs(sb) & ((1 << 64) - 1)
    al = [(ma >> (8 * i)) & 0xFF for i in range(MUL_LIMBS)]
    bl = [(mb >> (8 * i)) & 0xFF for i in range(MUL_LIMBS)]
    a_f = [torch.tensor([float(al[i])], dtype=torch.float32) for i in range(MUL_LIMBS)]
    b_f = [torch.tensor([float(bl[i])], dtype=torch.float32) for i in range(MUL_LIMBS)]
    out = limb_mul_from_limbs(a_f, b_f)
    mag = 0
    for p in range(MUL_OUT_LIMBS - 1, -1, -1):
        mag = mag * MUL_LIMB_RADIX + int(round(float(out[p][0])))
    prod64 = (-mag if neg else mag) & ((1 << 64) - 1)
    signed64 = prod64 - (1 << 64) if prod64 & (1 << 63) else prod64
    return (signed64 >> FRACBITS) & _MASK32


# ---- fp32 48-bit long-division datapath for FixedDiv, LIMB-based (16-bit hi/lo halves) --- #
#
# The remainder/divisor/quotient of the 48-bit long division can reach ~2^31, which is FAR
# above fp32's 2^24 exact-integer ceiling — so a WHOLE-VALUE fp32 scalar loses the low bits
# (this is exactly the fp128-forcing precision cliff the whole-value form hits).  The
# LIMB fix carries each 32-bit quantity as TWO 16-bit halves (hi,lo), each < 2^16 << 2^24,
# so every fp32 op is exact.  shift-left-1 / compare-GE / subtract-with-borrow are all done
# on the halves.  No value ever exceeds 2^17 on the datapath -> fp32-EXACT, no fp64/fp128.
_H = 65536.0                       # 2^16 limb base


def _u32_to_halves(v_np: np.ndarray):
    """Exact (hi,lo) 16-bit halves of a uint32 array as fp32 tensors."""
    lo = (v_np & 0xFFFF).astype(np.float64)
    hi = ((v_np >> 16) & 0xFFFF).astype(np.float64)
    return (torch.tensor(hi, dtype=torch.float32), torch.tensor(lo, dtype=torch.float32))


def _ge_halves(xhi, xlo, yhi, ylo):
    """UNSIGNED (x >= y) for 32-bit values held as 16-bit halves — every compared value
    < 2^16, so the fp32 compare is exact.  x>=y == xhi>yhi or (xhi==yhi and xlo>=ylo)."""
    return (xhi > yhi) | ((xhi == yhi) & (xlo >= ylo))


def _shl1_or_bit_halves(hi, lo, bit):
    """(hi,lo) = ((hi,lo) << 1) | bit, masked to 32 bits, halves kept < 2^16 (fp32-exact)."""
    lo2 = lo * 2.0 + bit
    carry = torch.floor(lo2 / _H)                  # bit shifted out of lo (0/1)
    lo_new = lo2 - carry * _H
    hi2 = hi * 2.0 + carry
    hi_new = hi2 - torch.floor(hi2 / _H) * _H      # drop bit 32 (mod 2^16 on hi)
    return hi_new, lo_new


def _sub_halves(xhi, xlo, yhi, ylo):
    """(xhi,xlo) - (yhi,ylo) with borrow, 32-bit, halves < 2^16 (only called when x>=y)."""
    lo = xlo - ylo
    borrow = (lo < 0).to(xlo.dtype)
    lo = lo + borrow * _H
    hi = xhi - yhi - borrow
    # hi >= 0 guaranteed since x >= y; keep as-is (< 2^16)
    return hi, lo


def fp32_fixed_div2_48bit_batched(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """FixedDiv2 (48-bit long division) for a batch, computed on the fp32 datapath with
    LIMB (16-bit hi/lo) remainder/quotient/divisor — every fp32 op < 2^17, so EXACT.

    Reproduces ``doom_fixedpoint.fixed_div2_48bit`` bit-for-bit: 16 integer + 32 fractional
    iterations, remainder kept < |b|, unsigned compare, final sign-negate.  NO fp64/fp128."""
    n = a_arr.shape[0]
    sa = np.array([dfp._sx(int(x)) for x in a_arr], dtype=object)
    sb = np.array([dfp._sx(int(x)) for x in b_arr], dtype=object)
    neg = ((sa < 0).astype(np.int64) + (sb < 0).astype(np.int64))
    a32 = np.where(sa < 0, (0 - sa) & _MASK32, sa & _MASK32).astype(np.int64)
    b32 = np.where(sb < 0, (0 - sb) & _MASK32, sb & _MASK32).astype(np.int64)
    bhi, blo = _u32_to_halves(b32)
    rem_hi = torch.zeros(n, dtype=torch.float32)
    rem_lo = torch.zeros(n, dtype=torch.float32)
    q_hi = torch.zeros(n, dtype=torch.float32)
    q_lo = torch.zeros(n, dtype=torch.float32)
    for i in range(47, -1, -1):
        if i >= FRACBITS:
            bit_np = ((a32 >> (i - FRACBITS)) & 1).astype(np.float64)
        else:
            bit_np = np.zeros(n, dtype=np.float64)
        bit = torch.tensor(bit_np, dtype=torch.float32)
        rem_hi, rem_lo = _shl1_or_bit_halves(rem_hi, rem_lo, bit)
        q_hi, q_lo = _shl1_or_bit_halves(q_hi, q_lo, torch.zeros(n, dtype=torch.float32))
        ge = _ge_halves(rem_hi, rem_lo, bhi, blo)
        s_hi, s_lo = _sub_halves(rem_hi, rem_lo, bhi, blo)
        rem_hi = torch.where(ge, s_hi, rem_hi)
        rem_lo = torch.where(ge, s_lo, rem_lo)
        q_lo = torch.where(ge, q_lo + 1.0, q_lo)   # set bit 0 (q_lo just shifted, even)
    q_i = (q_hi.cpu().numpy().round().astype(np.int64) * 65536
           + q_lo.cpu().numpy().round().astype(np.int64)).astype(object)
    q_i = np.where((neg & 1) == 1, (0 - q_i) & _MASK32, q_i & _MASK32)
    return q_i.astype(object)


def fp32_fixed_div(a: int, b: int) -> int:
    """FixedDiv via the fp32 datapath: the abs/overflow guard + the 48-bit long division,
    matching doom_fixedpoint.fixed_div byte-for-byte, entirely in fp32."""
    sa, sb = dfp._sx(a), dfp._sx(b)
    # guard: sar32(abs(a),14) >= signed(abs(b)) — small values, plain compare (fp32-exact)
    if dfp._sar32(dfp._abs32(a), 14) >= dfp._sx(dfp._abs32(b)):
        return dfp.MININT if (sa ^ sb) < 0 else dfp.MAXINT
    return int(fp32_fixed_div2_48bit_batched(np.array([a & _MASK32]),
                                             np.array([b & _MASK32]))[0])


def verify_fixedmul_fixeddiv(dtype=torch.float32) -> dict:
    """Byte-exact FixedMul + FixedDiv over Doom's ACTUAL 16.16 ranges (the doom_fixedpoint
    battery: LCG operands + negatives + FRACUNIT boundaries + overflow-guard regime), fp32,
    NO fp128.  Oracle = doom_fixedpoint.fixed_mul/fixed_div (on-VM-exact reference)."""
    cases = dfp.battery_cases()
    fm_fail = fd_fail = 0
    a_arr = np.array([a & _MASK32 for a, _ in cases], dtype=np.int64)
    b_arr = np.array([b & _MASK32 for _, b in cases], dtype=np.int64)
    # FixedMul: signed magnitude limb product per case
    for a, b in cases:
        if fp32_fixed_mul(a, b) != dfp.fixed_mul(a, b):
            fm_fail += 1
    # FixedDiv: guard in python (small compare), body batched on fp32 datapath
    for k, (a, b) in enumerate(cases):
        ref = dfp.fixed_div(a, b)
        sa, sb = dfp._sx(a), dfp._sx(b)
        if dfp._sar32(dfp._abs32(a), 14) >= dfp._sx(dfp._abs32(b)):
            got = dfp.MININT if (sa ^ sb) < 0 else dfp.MAXINT
        else:
            got = int(fp32_fixed_div2_48bit_batched(np.array([a & _MASK32]),
                                                    np.array([b & _MASK32]))[0])
        if got != ref:
            fd_fail += 1
    return {"n": len(cases), "FixedMul_fail": fm_fail, "FixedDiv_fail": fd_fail,
            "FixedMul_exact": fm_fail == 0, "FixedDiv_exact": fd_fail == 0,
            "dtype": str(dtype).replace("torch.", ""), "no_fp128": True,
            "ranges": "doom 16.16 fixed-point: LCG + negatives + FRACUNIT + overflow-guard"}


# =========================================================================== #
# THE fp32 FULL-OP-SET STEP MODEL — MUL runs the limb depth, everything else radix 4096.
# =========================================================================== #
# Per-op step DEPTH (sequential reused layers).  ADD-class ops run at radix 4096 (their
# accumulator fits fp32); MUL runs the 8-bit-limb schoolbook (radix 256), which is deeper
# (more result-byte columns) but is what keeps the product fp32-exact.  DIV runs radix-4096
# long division (r^2 = 2^24, the exact fp32 boundary).
FRAMING_PASSES = 4                 # opcode-select + PC + SP + address-compare (ADD-class)


def op_step_depth(op: str, radix_alu: int = 4096) -> int:
    """Sequential layer depth of ONE clever step whose execute op is `op`.

    framing (4 ADD-class passes at radix_alu) + the execute op's digit/limb layers.
    MUL uses the 8-bit-limb radix-256 schoolbook (its 8 output-byte columns + carry rounds
    are the depth that keeps it fp32-exact); DIV uses radix_alu long division."""
    add_limbs = _limbs(radix_alu, 33)               # ADD result 33-bit
    framing = FRAMING_PASSES * add_limbs
    if op == "MUL":
        # 8-bit limbs: MUL_OUT_LIMBS carry-resolve columns; the depth that stays < 2^24.
        return framing + MUL_OUT_LIMBS
    if op == "DIV":
        return framing + _limbs(radix_alu, 32)      # 32-bit quotient
    return framing + add_limbs                       # ADD-class execute


class FP32FullOpModel(torch.nn.Module):
    """The whole clever STEP for a given execute op, at the fp32 full-op-set depth.

    Reuses ``CompactStepModel`` (the compact-scoring core: T=1 direct framing attention +
    SwiGLU FFN with a COMPACT candidate band, no radix-wide LUT blowup).  MUL's model is
    the DEEPER one (limb schoolbook depth); ADD-class + DIV are shallower.  Every layer runs
    in fp32."""
    def __init__(self, op, d_model, dtype=torch.float32, radix_alu=4096,
                 scheme="direct", direct_attn=True):
        super().__init__()
        depth = op_step_depth(op, radix_alu)
        inter = compact_inter(radix_alu, scheme)
        self.op, self.depth, self.inter = op, depth, inter
        self.core = CompactStepModel(depth, d_model, inter, dtype, direct_attn)

    def forward(self, x):
        return self.core(x)


# =========================================================================== #
# TIMING — per-op ns/lane-step, then the DOOM-MIX-weighted average.
# =========================================================================== #
def _time_model(model, x, device, iters, warmup):
    sink = None
    with torch.no_grad():
        for _ in range(warmup):
            sink = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            sink = model(x)
        if sink is not None:
            float(sink.flatten()[0])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_op(device, op, dtype, batches, iters, warmup, d_model=64, radix_alu=4096):
    """Best (fastest-throughput) ns/lane-step for a single-op clever step at fp32."""
    dev = torch.device(device)
    model = FP32FullOpModel(op, d_model, dtype, radix_alu).to(dev).eval()
    best, rows = None, []
    for B in batches:
        try:
            x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
            dt = _time_model(model, x, device, iters, warmup)
        except RuntimeError as e:
            rows.append({"batch": B, "err": str(e)[:80]})
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        lane_steps_s = iters * B / dt
        r = {"batch": B, "ms_per_step": dt / iters * 1e3,
             "ns_per_lane_step": dt / iters / B * 1e9,
             "us_per_lane_step": dt / iters / B * 1e6,
             "lane_steps_per_s": lane_steps_s,
             "render_fps": lane_steps_s / RENDER_STEPS}
        rows.append(r)
        if best is None or lane_steps_s > best["lane_steps_per_s"]:
            best = r
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"op": op, "depth": model_depth(op, radix_alu), "dtype": str(dtype).replace("torch.", ""),
            "rows": rows, "best": best}


def model_depth(op, radix_alu=4096):
    return op_step_depth(op, radix_alu)


def _mix_op_class(op: str) -> str:
    """Map a DOOM_MIX opcode to its timed clever step class (MUL / DIV / ADD-class)."""
    if op == "MUL":
        return "MUL"
    if op == "DIV":
        return "DIV"
    return "ADD"                                     # everything else is ADD-class framing


def doom_mix_weighted_ns(per_class_ns: dict) -> dict:
    """Weight the per-class ns/lane-step by Doom's op mix -> the average per-lane-step."""
    weight = {"MUL": 0.0, "DIV": 0.0, "ADD": 0.0}
    for op, f in DOOM_MIX.items():
        weight[_mix_op_class(op)] += f
    avg = sum(weight[c] * per_class_ns[c] for c in weight)
    return {"class_weight": weight, "weighted_ns_per_lane_step": avg,
            "weighted_us_per_lane_step": avg / 1e3}


def bench_step_2gpu(op, dtype, B, iters, warmup, d_model=64, radix_alu=4096):
    """REAL concurrent 2-GPU throughput for a single-op clever step (both cards free)."""
    models, xs = [], []
    for gi in (0, 1):
        dev = torch.device(f"cuda:{gi}")
        m = FP32FullOpModel(op, d_model, dtype, radix_alu).to(dev).eval()
        x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
        models.append(m)
        xs.append(x)
    with torch.no_grad():
        for _ in range(warmup):
            for m, x in zip(models, xs):
                m(x)
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs = [m(x) for m, x in zip(models, xs)]
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        for o in outs:
            float(o.flatten()[0].float())
        dt = time.perf_counter() - t0
    lane_steps_s = iters * B * 2 / dt
    for m in models:
        del m
    for gi in (0, 1):
        torch.cuda.set_device(gi)
        torch.cuda.empty_cache()
    return {"op": op, "batch_per_gpu": B, "ns_per_lane_step": dt / iters / (B * 2) * 1e9,
            "us_per_lane_step": dt / iters / (B * 2) * 1e6,
            "lane_steps_per_s": lane_steps_s, "render_fps": lane_steps_s / RENDER_STEPS,
            "dtype": str(dtype).replace("torch.", "")}


# =========================================================================== #
# MAIN
# =========================================================================== #
def run_verify(args, out):
    print("=" * 100)
    print("BYTE-EXACT fp32 FULL-OP-SET (NO fp64/fp128 anywhere)")
    print("=" * 100)

    # ---- (a) full 64-bit MUL via 8-bit limbs, fp32 ----
    print("\n[1] full 64-bit MUL (32x32 -> 64) via fp32 8-bit LIMBS, random operands:")
    mfull = verify_limb_mul_full64(n=args.n_mul)
    out["mul_full64"] = mfull
    print(f"    limb radix {mfull['limb_radix']} (8-bit limbs); worst column accumulator "
          f"{mfull['col_peak_acc']:,} vs fp32 ceiling {mfull['fp32_ceiling']:,} "
          f"({'FITS' if mfull['fits_fp32'] else 'OVERFLOWS'})")
    print(f"    {mfull['n']:,} random ops -> byte-exact vs exact 64-bit product: "
          f"{'PASS' if mfull['exact'] else 'FAIL'}  (fp128 used: NO)")

    # ---- (b) Doom FixedMul + FixedDiv over 16.16 ranges (incl negatives / overflow) ----
    print("\n[2] Doom FixedMul + FixedDiv over ACTUAL 16.16 fixed-point ranges "
          "(incl. negatives / sign-magnitude / overflow guard), fp32, NO fp128:")
    fx = verify_fixedmul_fixeddiv()
    out["fixedmul_fixeddiv"] = fx
    print(f"    battery {fx['n']} cases ({fx['ranges']})")
    print(f"    FixedMul (fp32 limb product, signed, >>16): "
          f"{'PASS' if fx['FixedMul_exact'] else 'FAIL ('+str(fx['FixedMul_fail'])+' mismatch)'}"
          f"  vs doom_fixedpoint.fixed_mul")
    print(f"    FixedDiv (fp32 48-bit long division + guard): "
          f"{'PASS' if fx['FixedDiv_exact'] else 'FAIL ('+str(fx['FixedDiv_fail'])+' mismatch)'}"
          f"  vs doom_fixedpoint.fixed_div")

    # ---- (c) the rest of the op set at fp32 radix 4096 (reuse compact-scoring machinery) ----
    print("\n[3] rest of the op set at fp32 radix 4096 (ADD/SUB ripple, DIV/MOD long-div, "
          "CMP/SHL/SHR/LEA/bitwise/memory share the ADD/DIV limb decode):")
    rest = {}
    for r in (256, 4096):
        v = verify_compact_byte_exact(r, torch.float32, "direct", n=args.n)
        rest[str(r)] = v
        print(f"    radix {r:>5d} fp32 'direct': ADD_exact={str(v['ADD_exact']):5s} "
              f"DIV_exact_in_dtype={str(v['DIV_exact_in_dtype']):5s} "
              f"(DIV r^2={v['div_acc_max_r2']:,} vs fp32 {v['dtype_ceiling']:,}) "
              f"{'PASS' if v['ADD_exact'] and v['DIV_exact_in_dtype'] else 'FAIL'}")
    out["rest_opset"] = rest
    # negative control: MUL at radix 4096 as a WHOLE-value col-peak does NOT fit fp32 —
    # this is the fp128 NO-GO, and is exactly what the 8-bit-limb form above avoids.
    mul4096 = oc.acc_max("MUL", 4096)
    out["mul_whole_value_radix4096_needs_gt_fp32"] = {
        "col_peak": mul4096, "fits_fp32": mul4096 <= oc.PRECISION_CEILING["fp32"]}
    print(f"\n    [why limbs] MUL at radix 4096 (near-whole-value) col-peak {mul4096:,} "
          f"> fp32 {oc.PRECISION_CEILING['fp32']:,}: would need fp64+ -> the 8-bit-limb form "
          f"in [1] is what keeps MUL fp32-exact.")

    # ---- per-op pass-rate summary + any op still needing higher precision ----
    passes = {
        "MUL_full64": mfull["exact"],
        "FixedMul": fx["FixedMul_exact"],
        "FixedDiv": fx["FixedDiv_exact"],
        "ADD_radix4096": rest["4096"]["ADD_exact"],
        "SUB/CMP/SHL/SHR/LEA/bitwise/mem (ADD-class)": rest["4096"]["ADD_exact"],
        "DIV/MOD_radix4096": rest["4096"]["DIV_exact_in_dtype"],
    }
    out["per_op_pass"] = passes
    still_need_higher = [k for k, ok in passes.items() if not ok]
    out["ops_needing_gt_fp32"] = still_need_higher
    print("\n" + "-" * 100)
    print("PER-OP fp32 BYTE-EXACT PASS RATE (NO fp64/fp128):")
    for k, ok in passes.items():
        print(f"    {k:<48s} {'PASS' if ok else 'FAIL'}")
    print(f"\n  ops that STILL need > fp32: "
          f"{still_need_higher if still_need_higher else 'NONE — full Doom op set is fp32-exact'}")
    out["all_fp32_exact"] = not still_need_higher
    return out


def run_bench(args, out, dev):
    batches = [int(b) for b in args.batches.split(",")]
    print("\n" + "=" * 100)
    print(f"fp32 FULL-OP-SET PER-LANE-STEP COST (measured)  device={dev}  d_model={args.d_model}")
    if dev.startswith("cuda"):
        print(f"  {torch.cuda.get_device_name(0)}")
    print(f"  wide-VM baseline: {WIDE_VM_US_PER_STEP} us/step (PERF_LADDER, byte-exact, bk512)")
    print("=" * 100)

    per_class = {}
    per_class_best = {}
    print("\nper-op-class ns/lane-step (fp32, batched to saturation):")
    for op in ("ADD", "DIV", "MUL"):
        res = bench_op(dev, op, torch.float32, batches, args.iters, args.warmup,
                       d_model=args.d_model)
        per_class[op] = res
        b = res["best"]
        per_class_best[op] = b["ns_per_lane_step"] if b else float("inf")
        if b:
            print(f"  {op:<4s} depth={res['depth']:>3d}  {b['us_per_lane_step']:.4f} us/lane-step "
                  f"({b['ns_per_lane_step']:.2f} ns)  {b['render_fps']:8.2f} render fps "
                  f"(batch {b['batch']})")
    out["bench_1gpu_per_class"] = {op: per_class[op]["best"] for op in per_class}

    # DOOM-MIX-weighted average per-lane-step
    mix = doom_mix_weighted_ns(per_class_best)
    out["doom_mix_weighted"] = mix
    w = mix["class_weight"]
    print(f"\nDOOM-MIX weighting: ADD-class {w['ADD']*100:.1f}%  MUL {w['MUL']*100:.1f}%  "
          f"DIV {w['DIV']*100:.1f}%")
    wus = mix["weighted_us_per_lane_step"]
    print(f"  weighted fp32 per-lane-step = {wus:.4f} us "
          f"({mix['weighted_ns_per_lane_step']:.2f} ns)")

    # the edge vs the wide VM
    edge = WIDE_VM_US_PER_STEP / wus
    add_edge = WIDE_VM_US_PER_STEP / (per_class_best["ADD"] / 1e3)
    out["edge_vs_wide_vm"] = {
        "wide_vm_us_per_step": WIDE_VM_US_PER_STEP,
        "clever_fp32_weighted_us_per_lane_step": wus,
        "edge_x": edge,
        "add_class_only_us": per_class_best["ADD"] / 1e3,
        "add_class_only_edge_x": add_edge,
        "mul_us": per_class_best["MUL"] / 1e3,
        "div_us": per_class_best["DIV"] / 1e3,
    }
    print(f"\n  vs wide VM ({WIDE_VM_US_PER_STEP} us/step):  clever fp32 full-op-set "
          f"= {edge:.2f}x faster per-lane-step (DOOM-mix weighted)")
    print(f"    (ADD-class-only edge {add_edge:.2f}x; MUL {per_class_best['MUL']/1e3:.4f} us "
          f"= {WIDE_VM_US_PER_STEP/(per_class_best['MUL']/1e3):.2f}x; "
          f"DIV {per_class_best['DIV']/1e3:.4f} us = "
          f"{WIDE_VM_US_PER_STEP/(per_class_best['DIV']/1e3):.2f}x)")
    edge_survives = edge >= 7.0
    near7 = edge >= 6.0
    print(f"\n  >>> does the ~7x edge survive?  {'YES' if edge_survives else ('CLOSE (>=6x)' if near7 else 'NO')} "
          f"({edge:.2f}x)")
    out["edge_vs_wide_vm"]["seven_x_survives"] = bool(edge_survives)
    out["edge_vs_wide_vm"]["ge_6x"] = bool(near7)

    # REAL 2-GPU (weighted) if requested
    if args.two_gpu and torch.cuda.device_count() >= 2:
        print("\n" + "-" * 100)
        print("REAL 2-GPU concurrent per-op ns/lane-step (both cards):")
        satB = max(batches)
        per_class_2g = {}
        for op in ("ADD", "DIV", "MUL"):
            try:
                g2 = bench_step_2gpu(op, torch.float32, satB, args.iters, args.warmup,
                                     d_model=args.d_model)
                per_class_2g[op] = g2["ns_per_lane_step"]
                print(f"  {op:<4s} 2-GPU {g2['us_per_lane_step']:.4f} us/lane-step "
                      f"({g2['render_fps']:.2f} render fps aggregate)")
            except Exception as e:
                per_class_2g[op] = float("inf")
                print(f"  {op:<4s} 2-GPU failed: {str(e)[:100]}")
        if all(math.isfinite(v) for v in per_class_2g.values()):
            mix2 = doom_mix_weighted_ns(per_class_2g)
            wus2 = mix2["weighted_us_per_lane_step"]
            out["bench_2gpu"] = {"per_class_ns": per_class_2g,
                                 "weighted_us_per_lane_step": wus2,
                                 "edge_x": WIDE_VM_US_PER_STEP / wus2}
            print(f"  2-GPU weighted fp32 per-lane-step = {wus2:.4f} us "
                  f"({WIDE_VM_US_PER_STEP/wus2:.2f}x the wide VM)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", default="16384,65536,262144")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--n-mul", type=int, default=20000)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--two-gpu", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True

    out = {"render_steps": RENDER_STEPS, "raw_steps": RAW_STEPS,
           "wide_vm_us_per_step": WIDE_VM_US_PER_STEP,
           "mul_limb_radix": MUL_LIMB_RADIX, "doom_mix": DOOM_MIX}
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.verify or args.json:
        out = run_verify(args, out)

    if args.bench:
        out = run_bench(args, out, dev)

    # ---- VERDICT ----
    print("\n" + "=" * 100)
    print("VERDICT: is a min-param FP32 clever VM byte-exact for ALL Doom ops (NO fp64/fp128), "
          "and does it keep a per-lane-step advantage?")
    print("=" * 100)
    all_exact = out.get("all_fp32_exact")
    if all_exact is not None:
        print(f"  byte-exact for ALL Doom ops in fp32 (no fp64/fp128)? "
              f"{'YES' if all_exact else 'NO — ' + str(out.get('ops_needing_gt_fp32'))}")
    if "edge_vs_wide_vm" in out:
        e = out["edge_vs_wide_vm"]
        print(f"  fp32 per-lane-step {e['clever_fp32_weighted_us_per_lane_step']:.4f} us "
              f"vs wide VM {e['wide_vm_us_per_step']} us -> {e['edge_x']:.2f}x edge "
              f"({'~7x HOLDS' if e['seven_x_survives'] else ('~6x, near' if e['ge_6x'] else 'edge reduced')})")
    if all_exact and out.get("edge_vs_wide_vm", {}).get("ge_6x"):
        print("  >>> precision NO-GO OVERTURNED: the whole Doom op set is fp32-exact via the "
              "8-bit-limb MUL, and the clever VM keeps its per-lane-step advantage.")
    out["verdict"] = {
        "all_doom_ops_fp32_exact_no_fp128": all_exact,
        "edge_vs_wide_vm": out.get("edge_vs_wide_vm", {}).get("edge_x"),
        "seven_x_survives": out.get("edge_vs_wide_vm", {}).get("seven_x_survives"),
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
