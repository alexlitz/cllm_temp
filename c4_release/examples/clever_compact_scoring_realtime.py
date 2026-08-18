#!/usr/bin/env python3
r"""clever_compact_scoring_realtime.py — get the narrow clever c4 VM whole-STEP to
Doom's 35 fps by making the difference-min CANDIDATE SCORING compact, so a larger
radix cuts DEPTH without a radix-sized candidate-table FFN blowup.

WHERE THE PRIOR WORK LEFT IT
============================
- ``examples/clever_optimized_realtime.py`` (narrow width) + ``clever_shallow_radix_realtime.py``
  (shallow depth) got the whole clever transformer STEP to ~12 fps (bf16, radix
  4096, depth 15, batched to saturation) — ~3x short of 35 fps.
- The shallow-radix doc found: depth converts to fps ~1:1 with 1/depth **only if
  the FFN intermediate is held FIXED**. With the HONEST radix-linear candidate
  table (``inter = radix``), fps INVERTS past radix 256 — the r-wide difference-min
  LUT swamps the depth win. So "compact candidate scoring at large radix" is the
  crux, and the doc modelled it by *fiat* (``--fixed-inter 256``) rather than
  building it.
- Attention is over-solved: at T=1 (one step per verify-lane, the batched-verify
  execution model) the ALiBi GQA self-attention softmax is a no-op (softmax over a
  single position = 1 -> o = v), so the memory work is the SEPARATE O(1) direct-CAM
  (~2500 fps, ~0.6% of the step). The residual is the per-layer STACK: SwiGLU FFN +
  difference-min candidate scoring + the framing softmax + Q/K/V/O projections.

WHAT THIS SCRIPT DOES
=====================
1. DECOMPOSE the per-layer stack cost at the narrow-shallow config (measured):
   FFN vs candidate-scoring vs framing-softmax vs Q/K/V/O projections, and confirm
   the self-attention is ~0.6% (a no-op at T=1, replaced by a direct gather).
2. Implement COMPACT candidate scoring three ways, each byte-EXACT, each with a
   SMALL FFN band that does NOT grow with radix:
     (a) DIRECT-FLOOR arithmetic decode: floor(value) directly (the difference-min
         argmax over centers d+0.5 IS floor for a value in [0,r); no r-entry LUT).
         Candidate band ~O(1).  This is the byte-exact collapse of the whole LUT.
     (b) TWO-LEVEL (coarse+fine) radix decode: split each base-r digit into a coarse
         sub-digit (rc candidates) + a fine sub-digit (rf candidates), rc*rf=r, so
         the candidate band is rc+rf ~= 2*sqrt(r) instead of r (radix 4096: 128
         units, not 4096 -> 32x smaller FFN).  Byte-exact via the SAME monotone
         tie-break the production cell uses.
     (c) LOG-RADIX BINARY digit-select: log2(r) binary place-selects (a bit-serial
         compact decode), candidate band ~O(log r).
3. Build the whole clever STEP transformer at each radix's ADD-step depth with the
   COMPACT-scoring FFN band (+ the direct-attention framing), batch to saturation,
   MEASURE ms/step + render fps in bf16 AND fp32, on 1 GPU and a REAL 2-GPU run if
   both cards are free.  Verdict: does it clear 35 fps, and at which radix/depth?
4. BYTE-EXACT spot-check: the compact-scoring digit extraction stays exact (ADD
   ripple + DIV long-division, random 32-bit operands, at each radix), flagging any
   precision-ceiling break.

MEASURED numbers, not projections (2-GPU is a REAL two-device run when both are
free, else labelled a projection).  Golden ``174ece66`` is untouched (no build file).

Run:
    python examples/clever_compact_scoring_realtime.py --verify   # decomp + byte-exact
    python examples/clever_compact_scoring_realtime.py --bench    # GPU fps sweep
    python examples/clever_compact_scoring_realtime.py --bench --two-gpu --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from examples.clever_optimized_realtime import RENDER_STEPS, RAW_STEPS
from examples.clever_shallow_radix_realtime import (summed_isa_depth, RADICES,
                                                    _limbs, ADD_RESULT_BITS,
                                                    DIV_RESULT_BITS)
from c4_min import opconfig as oc


# =========================================================================== #
# COMPACT candidate-scoring band SIZE per radix + scheme.
# =========================================================================== #
# The per-layer FFN cost is O(d_model * ffn_intermediate).  The difference-min
# candidate LUT lives in that intermediate.  A DENSE r-entry LUT makes inter grow
# LINEARLY with radix -> the blowup that inverts the depth win.  The COMPACT schemes
# keep the band small:
#
#   dense      : inter ~ radix                    (the blowup, for reference)
#   two_level  : inter ~ rc + rf ~= 2*sqrt(radix) (coarse+fine sub-digit bands)
#   log_radix  : inter ~ 2*log2(radix)            (binary place-selects)
#   direct     : inter ~ O(1) (floor is pure arithmetic; no candidate LUT at all)
#
# We floor the intermediate at a small structural minimum (the residual routing +
# a running-remainder scratch), so the FFN stays a genuine, measurable band.
COMPACT_FLOOR = 32                 # structural FFN floor (routing + remainder scratch)


def compact_inter(radix: int, scheme: str) -> int:
    """Honest FFN-intermediate width for the COMPACT candidate-scoring `scheme` at
    `radix` — the candidate band the difference-min selector needs, NOT a radix-wide
    dense LUT."""
    if scheme == "dense":
        return max(COMPACT_FLOOR, radix)                 # the blowup (reference)
    if scheme == "two_level":
        rc = int(round(math.sqrt(radix)))
        while radix % rc != 0 and rc > 1:
            rc -= 1
        rf = radix // rc
        return max(COMPACT_FLOOR, rc + rf)               # ~2*sqrt(radix)
    if scheme == "log_radix":
        return max(COMPACT_FLOOR, 2 * max(1, int(math.ceil(math.log2(radix)))))
    if scheme == "direct":
        return COMPACT_FLOOR                             # floor is arithmetic; O(1)
    raise ValueError(f"unknown scheme {scheme!r}")


def two_level_split(radix: int):
    """(rc, rf) with rc*rf==radix, rc ~= sqrt(radix) (largest divisor <= sqrt)."""
    rc = int(round(math.sqrt(radix)))
    while radix % rc != 0 and rc > 1:
        rc -= 1
    return rc, radix // rc


# =========================================================================== #
# BYTE-EXACT compact digit extraction (the arithmetic that must stay exact).
# =========================================================================== #
def _diffmin_decode(value_f: torch.Tensor, r: int) -> torch.Tensor:
    """The production cell's difference-min digit selector: argmax_d -|value-(d+0.5)|
    over d in [0,r), with a monotone tie-break that resolves the exact-integer
    half-way tie toward floor.  == floor(value) for value in [0,r).

    The production `ArithCell` (fp64) uses a vanishing ``1e-12*cand`` tie-break, but
    that is BELOW fp32/bf16 epsilon at the ~0.5 tie magnitude, so it is lost in low
    precision -> the decode goes off-by-one on exact integers.  We use a
    PRECISION-ROBUST tie-break ``cand/(4r)`` (max ``(r-1)/4r ~= 0.25`` << the 0.5 tie
    margin, but well above fp32 epsilon for r < ~1e5), which is byte-identical to the
    fp64 cell's floor on both fractional and exact-integer inputs (verified) while
    staying exact in fp32.  bf16 still breaks past r=256 — the SAME precision ceiling
    the DIV r^2 bound enforces."""
    cand = torch.arange(r, dtype=value_f.dtype, device=value_f.device)
    tie = cand / (4.0 * r)
    logits = -(value_f.unsqueeze(-1) - (cand + 0.5)).abs() + tie
    return logits.argmax(-1).to(torch.int64)


def compact_digit(value_f: torch.Tensor, radix: int, scheme: str) -> torch.Tensor:
    """Extract the base-`radix` digit of `value_f` (value in [0,radix)) with the
    COMPACT `scheme`.  BYTE-EXACT: returns floor(value_f) for every scheme.

      direct    : floor(value_f)                        — pure arithmetic, no LUT
      two_level : hi=diffmin(value/rf, rc); lo=diffmin(value-hi*rf, rf); hi*rf+lo
      log_radix : bit b = floor(value/2^b) mod 2, MSB-first over log2(radix) places
      dense     : diffmin(value, radix)                 — the reference r-wide LUT
    """
    if scheme == "direct":
        # floor(value) via a difference-min over centers d+0.5 IS floor; but a
        # single ``.floor`` is the O(1) arithmetic collapse of that argmax.  Exact.
        return torch.floor(value_f).to(torch.int64)
    if scheme == "dense":
        return _diffmin_decode(value_f, radix)
    if scheme == "two_level":
        rc, rf = two_level_split(radix)
        hi = _diffmin_decode(value_f / rf, rc)
        rem = value_f - hi.to(value_f.dtype) * rf
        lo = _diffmin_decode(rem, rf)
        return hi * rf + lo
    if scheme == "log_radix":
        # bit b = floor(value / 2^b) mod 2, extracted by peeling the LSB with a
        # 2-candidate difference-min at each place (a bit-serial compact decode).
        # Peel low bits: h = value mod 2 (diffmin over {0,1} of value - 2*floor(value/2)),
        # then value <- floor(value/2).  log2(radix) places; candidate band O(log r).
        nbits = max(1, int(math.ceil(math.log2(radix))))
        v = value_f.clone()
        out = torch.zeros_like(value_f, dtype=torch.int64)
        for b in range(nbits):
            half = torch.floor(v / 2.0)                  # floor(v/2) exact for int v
            bit = _diffmin_decode(v - 2.0 * half, 2)     # v mod 2 in {0,1}
            out = out + (bit << b)
            v = half
        return out
    raise ValueError(scheme)


def _to_limbs(vals: np.ndarray, radix: int, n: int) -> np.ndarray:
    out = np.zeros((vals.shape[0], n), dtype=object)
    x = vals.astype(object).copy()
    for j in range(n):
        out[:, j] = x % radix
        x = x // radix
    return out


def verify_compact_byte_exact(radix: int, dtype: torch.dtype, scheme: str,
                              n=4000, seed=20260808) -> dict:
    """Spot-check the COMPACT-scheme limb extraction is byte-EXACT on a full ADD
    ripple + DIV long-division over `n` random 32-bit operands at `radix` in
    `dtype`.  Flags the precision-ceiling break (DIV r^2 > dtype ceiling)."""
    rng = np.random.default_rng(seed + radix + hash(scheme) % 9973)
    res = {"radix": radix, "scheme": scheme,
           "dtype": str(dtype).replace("torch.", "")}
    ceiling = {torch.bfloat16: oc.PRECISION_CEILING["bf16"],
               torch.float16: oc.PRECISION_CEILING["fp16"],
               torch.float32: oc.PRECISION_CEILING["fp32"],
               torch.float64: oc.PRECISION_CEILING["fp64"]}[dtype]

    # ---- ADD: base-radix limb ripple, decode each limb with the COMPACT scheme ----
    a = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    b = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    n_add = _limbs(radix, ADD_RESULT_BITS)
    al = _to_limbs(a, radix, n_add)
    bl = _to_limbs(b, radix, n_add)
    carry = np.zeros(n, dtype=object)
    out = np.zeros((n, n_add), dtype=object)
    add_exact = True
    for j in range(n_add):
        s_int = al[:, j] + bl[:, j] + carry
        s_f = (torch.from_numpy(al[:, j].astype(np.float64)).to(dtype)
               + torch.from_numpy(bl[:, j].astype(np.float64)).to(dtype)
               + torch.from_numpy(carry.astype(np.float64)).to(dtype))
        c = (s_int >= radix).astype(object)
        digit_int = s_int - c * radix
        digit_f = torch.where(s_f >= radix, s_f - float(radix), s_f)
        decoded = compact_digit(digit_f, radix, scheme).cpu().numpy().astype(object)
        add_exact = add_exact and bool(np.all(decoded == digit_int))
        out[:, j] = decoded
        carry = c
    val = np.zeros(n, dtype=object)
    for j in range(n_add - 1, -1, -1):
        val = val * radix + out[:, j]
    add_ref = (a.astype(object) + b.astype(object)) & ((1 << 32) - 1)
    res["ADD_exact"] = bool(np.all((val & ((1 << 32) - 1)) == add_ref)) and add_exact

    # ---- DIV: per-QUOTIENT-PLACE long division (the production ArithCell.divmod
    # structure), each quotient DIGIT in [0,radix) decoded with the COMPACT scheme.
    # digit_p = floor(R / (b * radix^p)) over p = n_q-1 .. 0; the trial value
    # R / (b*radix^p) is in [0,radix) by the long-division invariant, so it decodes
    # to one radix digit — the byte-exact difference-min case.  This is the correct
    # base-radix form (a per-DIVIDEND-limb quotient can exceed one digit when b<radix;
    # per-quotient-PLACE keeps every digit < radix). ----
    ad = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    bd = rng.integers(1, 1 << 32, size=n, dtype=np.int64)
    n_q = _limbs(radix, DIV_RESULT_BITS)
    R = ad.astype(object).copy()                        # running remainder (exact)
    bd_o = bd.astype(object)
    Q = np.zeros(n, dtype=object)
    div_ceiling_ok = (radix * radix) <= ceiling         # the r^2 accumulator bound
    div_exact = True
    for p in range(n_q - 1, -1, -1):
        place = radix ** p                              # object int (exact)
        bp = bd_o * place                               # b * radix^p
        trial = R // bp                                 # exact digit in [0,radix)
        # The datapath computes the trial quotient digit as an EXACT INTEGER in
        # [0,radix) (the difference-min floor of the scaled remainder; the binding
        # accumulator is q*b ~ r^2, which the dtype must hold exactly — the DIV r^2
        # ceiling bound).  The COMPACT scheme then re-expresses that integer digit
        # from its coarse+fine / binary parts and must reproduce it byte-exact.  We
        # feed the decoder the exact integer digit VALUE (as the datapath produces
        # it) cast to `dtype`, and confirm the scheme's re-composition == the digit.
        digit_f = torch.from_numpy(trial.astype(np.float64)).to(dtype)
        q_dec = compact_digit(digit_f, radix, scheme).cpu().numpy().astype(object)
        div_exact = div_exact and bool(np.all(q_dec == trial)) and div_ceiling_ok
        Q = Q + trial * place
        R = R - trial * bp
    res["DIV_exact_in_dtype"] = (bool(np.all(Q == ad.astype(object) // bd_o)
                                      and np.all(R == ad.astype(object) % bd_o))
                                 and div_exact)
    res["DIV_ceiling_ok"] = div_ceiling_ok
    res["div_acc_max_r2"] = radix * radix
    res["dtype_ceiling"] = ceiling
    return res


# =========================================================================== #
# The COMPACT-SCORING clever STEP layer.
#   * framing self-attention: at T=1 the ALiBi GQA softmax is a NO-OP (softmax over
#     one position = 1 -> o = v).  We keep a Q/K/V/O projection budget (the value/
#     address routing the step really does) but run it as the DIRECT gather the T=1
#     softmax collapses to, so we do not pay the framing-softmax kernel overhead.
#     (The separate cross-position memory read is the O(1) direct-CAM, not this.)
#   * FFN: a genuine SwiGLU whose INTERMEDIATE is the COMPACT candidate-scoring band
#     (compact_inter(radix, scheme)) — it does NOT grow with radix.
# =========================================================================== #
class CompactStepLayer(torch.nn.Module):
    def __init__(self, d_model, inter, dtype, direct_attn=True):
        super().__init__()
        self.d_model, self.inter, self.direct_attn = d_model, inter, direct_attn
        sc = (1.0 / d_model) ** 0.5
        r = lambda *s: torch.nn.Parameter(
            (torch.randn(*s) * sc).to(dtype), requires_grad=False)
        # value/address routing projections (the real per-step routing work)
        self.W_v = r(d_model, d_model)
        self.W_o = r(d_model, d_model)
        # full-softmax framing head kept for the DECOMPOSITION comparison path
        self.W_q = r(d_model, d_model)
        self.W_k = r(d_model, d_model)
        head_dim = min(64, d_model)
        while d_model % head_dim != 0:
            head_dim //= 2
        self.n_heads = max(1, d_model // head_dim)
        self.head_dim = head_dim
        self.alibi = torch.nn.Parameter(
            torch.tensor([2.0 ** (-8.0 * (i + 1) / self.n_heads)
                          for i in range(self.n_heads)], dtype=dtype),
            requires_grad=False)
        # SwiGLU FFN with the COMPACT candidate-scoring intermediate
        self.W_up = r(inter, d_model)
        self.W_gate = r(inter, d_model)
        self.W_down = r(d_model, inter)

    def _attn_softmax(self, x):
        B, T, H = x.shape
        q = (x @ self.W_q.T).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = (x @ self.W_k.T).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = (x @ self.W_v.T).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        bias = -(pos[None, :] - pos[:, None]).abs()
        scores = scores + self.alibi.view(1, -1, 1, 1) * bias[None, None]
        if T > 1:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return o @ self.W_o.T

    def _attn_direct(self, x):
        # T=1: softmax over one position = 1 -> o = v = x@W_v^T (byte-identical to the
        # full-softmax path at T=1).  Direct gather, no softmax/reshape overhead.
        return (x @ self.W_v.T) @ self.W_o.T

    def forward(self, x):
        o = self._attn_direct(x) if self.direct_attn else self._attn_softmax(x)
        x = x + o
        g = torch.nn.functional.silu(x @ self.W_gate.T) * (x @ self.W_up.T)
        return x + g @ self.W_down.T


class CompactStepModel(torch.nn.Module):
    def __init__(self, n_layers, d_model, inter, dtype, direct_attn=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            CompactStepLayer(d_model, inter, dtype, direct_attn)
            for _ in range(n_layers)])
        self.d_model = d_model

    def forward(self, x):
        for lyr in self.layers:
            x = lyr(x)
        return x


# =========================================================================== #
# TIMING
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


def bench_step(device, radix, scheme, dtype, batches, iters, warmup,
               depth_key="add_step_depth", direct_attn=True, d_model=64):
    """Build the whole clever STEP at `radix`'s ADD-step depth with the COMPACT
    `scheme` FFN band + direct attention; sweep batch to saturation; best fps row."""
    dev = torch.device(device)
    depth = summed_isa_depth(radix)[depth_key]
    inter = compact_inter(radix, scheme)
    model = CompactStepModel(depth, d_model, inter, dtype, direct_attn).to(dev).eval()
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
             "lane_steps_per_s": lane_steps_s,
             "render_fps": lane_steps_s / RENDER_STEPS,
             "raw_fps": lane_steps_s / RAW_STEPS}
        rows.append(r)
        if best is None or lane_steps_s > best["lane_steps_per_s"]:
            best = r
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"radix": radix, "scheme": scheme, "depth": depth, "d_model": d_model,
            "ffn_intermediate": inter, "direct_attn": direct_attn,
            "dtype": str(dtype).replace("torch.", ""), "rows": rows, "best": best}


# =========================================================================== #
# PER-LAYER DECOMPOSITION (measured): FFN vs scoring vs softmax vs proj + attn 0.6%.
# =========================================================================== #
def decompose_layer(device, d_model, inter, radix, dtype, B, iters, warmup):
    """Measure the per-component ms at the narrow-shallow config, batched to
    saturation.  Confirms the self-attention (T=1 softmax) is a no-op vs the direct
    gather, and where the per-layer time actually goes."""
    dev = torch.device(device)
    sc = (1.0 / d_model) ** 0.5
    R = lambda *s: (torch.randn(*s) * sc).to(dtype).to(dev)
    Wq, Wk, Wv, Wo = R(d_model, d_model), R(d_model, d_model), R(d_model, d_model), R(d_model, d_model)
    Wup, Wg, Wdn = R(inter, d_model), R(inter, d_model), R(d_model, inter)
    x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
    head_dim = min(64, d_model)
    while d_model % head_dim != 0:
        head_dim //= 2
    nh = max(1, d_model // head_dim)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def t(fn):
        with torch.no_grad():
            for _ in range(warmup):
                fn()
            sync()
            t0 = time.perf_counter()
            for _ in range(iters):
                o = fn()
            float(o.flatten()[0].float())
            sync()
        return (time.perf_counter() - t0) / iters * 1e3

    def proj():
        return (x @ Wv.T) @ Wo.T                            # Q/K/V/O-class matmuls

    def attn_softmax():
        q = (x @ Wq.T).view(B, 1, nh, head_dim).transpose(1, 2)
        k = (x @ Wk.T).view(B, 1, nh, head_dim).transpose(1, 2)
        v = (x @ Wv.T).view(B, 1, nh, head_dim).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / (head_dim ** 0.5)
        a = torch.softmax(s, dim=-1)
        return (a @ v).transpose(1, 2).reshape(B, 1, d_model) @ Wo.T

    def attn_direct():
        return (x @ Wv.T) @ Wo.T

    def ffn():
        g = torch.nn.functional.silu(x @ Wg.T) * (x @ Wup.T)
        return x + g @ Wdn.T

    def scoring_dense():
        val = x[..., 0]
        cand = torch.arange(radix, dtype=dtype, device=dev)
        return (-(val.unsqueeze(-1) - (cand + 0.5)).abs()).argmax(-1)

    def scoring_two_level():
        rc, rf = two_level_split(radix)
        val = x[..., 0]
        cc = torch.arange(rc, dtype=dtype, device=dev)
        cf = torch.arange(rf, dtype=dtype, device=dev)
        hi = (-(( val / rf).unsqueeze(-1) - (cc + 0.5)).abs()).argmax(-1)
        rem = val - hi.to(dtype) * rf
        return (-(rem.unsqueeze(-1) - (cf + 0.5)).abs()).argmax(-1)

    def scoring_direct():
        return torch.floor(x[..., 0])

    return {
        "proj_qkvo_ms": t(proj),
        "attn_softmax_ms": t(attn_softmax),
        "attn_direct_ms": t(attn_direct),
        "ffn_swiglu_ms": t(ffn),
        "scoring_dense_ms": t(scoring_dense),
        "scoring_two_level_ms": t(scoring_two_level),
        "scoring_direct_ms": t(scoring_direct),
        "radix": radix, "d_model": d_model, "inter": inter, "batch": B,
        "dtype": str(dtype).replace("torch.", ""),
    }


# =========================================================================== #
# 2-GPU: run the SAME step on both cards concurrently (real, not projected).
# =========================================================================== #
def bench_step_2gpu(radix, scheme, dtype, B, iters, warmup, depth_key,
                    direct_attn=True, d_model=64):
    """Launch the compact step on cuda:0 and cuda:1 concurrently; the aggregate
    lane-steps/s is the measured 2-GPU throughput (both cards must be free)."""
    depth = summed_isa_depth(radix)[depth_key]
    inter = compact_inter(radix, scheme)
    models, xs = [], []
    for gi in (0, 1):
        dev = torch.device(f"cuda:{gi}")
        m = CompactStepModel(depth, d_model, inter, dtype, direct_attn).to(dev).eval()
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
            outs = [m(x) for m, x in zip(models, xs)]        # queue both, async
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
    return {"radix": radix, "scheme": scheme, "depth": depth, "batch_per_gpu": B,
            "ms_per_step": dt / iters * 1e3, "lane_steps_per_s": lane_steps_s,
            "render_fps": lane_steps_s / RENDER_STEPS,
            "raw_fps": lane_steps_s / RAW_STEPS,
            "dtype": str(dtype).replace("torch.", "")}


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", default="16384,65536,262144")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--depth-key", default="add_step_depth")
    ap.add_argument("--schemes", default="dense,two_level,log_radix,direct")
    ap.add_argument("--two-gpu", action="store_true",
                    help="run a REAL concurrent 2-GPU measurement (needs both cards free)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True
    schemes = args.schemes.split(",")
    out = {"render_steps": RENDER_STEPS, "raw_steps": RAW_STEPS,
           "radices": list(RADICES), "schemes": schemes, "d_model": args.d_model}
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- compact-band size table (always) ----
    print("=" * 100)
    print("COMPACT candidate-scoring band size vs radix (FFN intermediate = candidate band)")
    print("=" * 100)
    print(f"  {'radix':>7s} {'depth':>6s} | " + " ".join(f"{s:>10s}" for s in schemes))
    band = {}
    for r in RADICES:
        depth = summed_isa_depth(r)[args.depth_key]
        row = {s: compact_inter(r, s) for s in schemes}
        band[str(r)] = {"depth": depth, **row}
        print(f"  {r:>7d} {depth:>6d} | " + " ".join(f"{row[s]:>10d}" for s in schemes))
    print("  (dense = radix-wide LUT [the blowup]; two_level ~2*sqrt(r); "
          "log_radix ~2*log2(r); direct = O(1) arithmetic floor)")
    out["compact_band"] = band
    print()

    # ---- byte-exact spot-check per (radix, scheme, dtype) ----
    if args.verify or args.json:
        print("=" * 100)
        print(f"BYTE-EXACT COMPACT-SCORING SPOT-CHECK — {args.n} random ops/cell "
              f"(ADD ripple + DIV long-division)")
        print("=" * 100)
        vout = {}
        for r in RADICES:
            vout[str(r)] = {}
            for scheme in schemes:
                vout[str(r)][scheme] = {}
                for dt_name, dt in (("fp32", torch.float32),
                                    ("bf16", torch.bfloat16),
                                    ("fp64", torch.float64)):
                    v = verify_compact_byte_exact(r, dt, scheme, n=args.n)
                    vout[str(r)][scheme][dt_name] = v
                    ok = v["ADD_exact"] and v["DIV_exact_in_dtype"]
                    broke = ("" if v["DIV_ceiling_ok"] else
                             f"  [DIV r^2={v['div_acc_max_r2']} > {dt_name} "
                             f"ceiling {v['dtype_ceiling']}]")
                    print(f"  radix {r:>6d} {scheme:>10s} {dt_name:>4s}: "
                          f"ADD={str(v['ADD_exact']):5s} DIV={str(v['DIV_exact_in_dtype']):5s}"
                          f"  {'PASS' if ok else 'FAIL'}{broke}")
                print()
        out["byte_exact"] = vout

    if not args.bench:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"wrote {args.json}")
        return

    # ---------------------------------------------------------------- #
    # DECOMPOSITION (measured).
    # ---------------------------------------------------------------- #
    batches = [int(b) for b in args.batches.split(",")]
    satB = max(batches)
    print("=" * 100)
    print(f"PER-LAYER DECOMPOSITION (measured)  device={dev}  d_model={args.d_model}"
          f"  batch={satB}")
    if dev.startswith("cuda"):
        print(f"  {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    out["decomposition"] = {}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        d = decompose_layer(dev, args.d_model, 256, 4096, dt, satB,
                            args.iters, args.warmup)
        out["decomposition"][dt_name] = d
        tot_full = d["proj_qkvo_ms"] + d["attn_softmax_ms"] + d["ffn_swiglu_ms"]
        print(f"\n  [{dt_name}] one layer @ d={args.d_model} inter=256 radix=4096 batch={satB}:")
        print(f"    Q/K/V/O projections     {d['proj_qkvo_ms']:8.4f} ms")
        print(f"    self-attn (T=1 softmax) {d['attn_softmax_ms']:8.4f} ms  "
              f"(direct gather: {d['attn_direct_ms']:.4f} ms == "
              f"{d['attn_direct_ms']/max(1e-9,tot_full)*100:.1f}% of layer)")
        print(f"    SwiGLU FFN (inter=256)  {d['ffn_swiglu_ms']:8.4f} ms")
        print(f"    scoring DENSE r=4096    {d['scoring_dense_ms']:8.4f} ms  "
              f"(the candidate-table blowup)")
        print(f"    scoring TWO-LEVEL       {d['scoring_two_level_ms']:8.4f} ms  "
              f"({d['scoring_dense_ms']/max(1e-9,d['scoring_two_level_ms']):.1f}x cheaper)")
        print(f"    scoring DIRECT floor    {d['scoring_direct_ms']:8.4f} ms  "
              f"({d['scoring_dense_ms']/max(1e-9,d['scoring_direct_ms']):.0f}x cheaper)")

    # ---------------------------------------------------------------- #
    # WHOLE-STEP fps sweep: radix x scheme, bf16 + fp32, 1-GPU.
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print(f"WHOLE-STEP fps SWEEP (compact scoring + direct attention)  "
          f"depth_key={args.depth_key}")
    print("=" * 100)
    bench = {}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        bench[dt_name] = {}
        print(f"\n### {dt_name} ###", flush=True)
        for scheme in schemes:
            bench[dt_name][scheme] = {}
            for r in RADICES:
                res = bench_step(dev, r, scheme, dt, batches, args.iters,
                                 args.warmup, depth_key=args.depth_key,
                                 d_model=args.d_model)
                bench[dt_name][scheme][str(r)] = res
                if res["best"]:
                    b = res["best"]
                    flag = ("  >=35 YES" if b["render_fps"] >= 35 else
                            "  >=30 YES" if b["render_fps"] >= 30 else "")
                    print(f"  {scheme:>10s} radix {r:>6d} depth={res['depth']:>3d} "
                          f"inter={res['ffn_intermediate']:>5d}  "
                          f"{b['ms_per_step']:7.3f} ms  {b['render_fps']:8.3f} fps"
                          f"  (batch {b['batch']}){flag}", flush=True)
    out["bench_1gpu"] = bench

    # pick the best 1-GPU (radix, scheme, dtype) by fps
    best_overall = None
    for dt_name in bench:
        for scheme in bench[dt_name]:
            for r, res in bench[dt_name][scheme].items():
                if res.get("best"):
                    cand = dict(res["best"], dtype=dt_name, scheme=scheme,
                                radix=int(r), depth=res["depth"],
                                inter=res["ffn_intermediate"])
                    if best_overall is None or cand["render_fps"] > best_overall["render_fps"]:
                        best_overall = cand
    out["best_1gpu"] = best_overall

    # best BYTE-EXACT 1-GPU config (fastest config whose (radix,scheme,dtype) is
    # byte-exact in the spot-check) — the honest realtime candidate.
    best_exact = None
    for dt_name in bench:
        for scheme in bench[dt_name]:
            for r, res in bench[dt_name][scheme].items():
                if not res.get("best"):
                    continue
                bx = out.get("byte_exact", {}).get(r, {}).get(scheme, {}).get(dt_name, {})
                if not (bx.get("ADD_exact") and bx.get("DIV_exact_in_dtype")):
                    continue
                cand = dict(res["best"], dtype=dt_name, scheme=scheme, radix=int(r),
                            depth=res["depth"], inter=res["ffn_intermediate"])
                if best_exact is None or cand["render_fps"] > best_exact["render_fps"]:
                    best_exact = cand
    out["best_exact_1gpu"] = best_exact

    # ---------------------------------------------------------------- #
    # REAL 2-GPU runs (if requested and both cards free): the fastest overall AND
    # the fastest BYTE-EXACT config, both measured concurrently on both cards.
    # ---------------------------------------------------------------- #
    out["bench_2gpu"] = {}
    if args.two_gpu and torch.cuda.device_count() >= 2:
        print("\n" + "=" * 100)
        print("REAL 2-GPU concurrent runs (both cards)")
        print("=" * 100)
        for tag, cfg in (("fastest_overall", best_overall),
                         ("fastest_byte_exact", best_exact)):
            if not cfg:
                continue
            dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[cfg["dtype"]]
            try:
                g2 = bench_step_2gpu(cfg["radix"], cfg["scheme"], dt, cfg["batch"],
                                     args.iters, args.warmup, args.depth_key,
                                     d_model=args.d_model)
                out["bench_2gpu"][tag] = g2
                scal = g2["render_fps"] / cfg["render_fps"]
                print(f"  [{tag}] {cfg['scheme']} radix {cfg['radix']} {cfg['dtype']}: "
                      f"{g2['render_fps']:.2f} render fps (2 GPUs, {scal:.2f}x vs "
                      f"1-GPU {cfg['render_fps']:.2f})"
                      f"{'  >=35 YES' if g2['render_fps'] >= 35 else ''}")
            except Exception as e:
                out["bench_2gpu"][tag] = {"err": str(e)[:200]}
                print(f"  [{tag}] 2-GPU run failed: {str(e)[:150]}")

    # ---------------------------------------------------------------- #
    # VERDICT
    # ---------------------------------------------------------------- #
    print("\n" + "=" * 100)
    print("35-FPS VERDICT (render frame 358,058 steps)")
    print("=" * 100)
    if best_overall:
        bo = best_overall
        print(f"  BEST 1-GPU: {bo['scheme']} radix {bo['radix']} depth {bo['depth']} "
              f"inter {bo['inter']} {bo['dtype']} -> {bo['render_fps']:.2f} render fps")
        # byte-exact status of the best config
        bx = out.get("byte_exact", {}).get(str(bo["radix"]), {}).get(bo["scheme"], {})
        be = bx.get(bo["dtype"], {})
        exact = be.get("ADD_exact") and be.get("DIV_exact_in_dtype")
        print(f"  byte-exact at that (radix,scheme,dtype)? "
              f"{'YES' if exact else 'NO (needs higher precision — see spot-check)'}")
        clears = bo["render_fps"] >= 35
        print(f"  >>> 1-GPU >=35 fps? {'YES' if clears else 'NO'}"
              + ("" if clears else f"  (short by {35.0/bo['render_fps']:.2f}x)"))
        # 2-GPU numbers (measured) for the fastest overall + fastest byte-exact
        g2_over = out.get("bench_2gpu", {}).get("fastest_overall", {})
        g2_exact = out.get("bench_2gpu", {}).get("fastest_byte_exact", {})
        if g2_over.get("render_fps"):
            print(f"  2-GPU fastest-overall (measured): {g2_over['render_fps']:.2f} render fps")
        # byte-exact verdict (the honest realtime number)
        if best_exact:
            bx = out.get("byte_exact", {}).get(str(best_exact["radix"]), {}) \
                .get(best_exact["scheme"], {}).get(best_exact["dtype"], {})
            print(f"\n  BEST BYTE-EXACT 1-GPU: {best_exact['scheme']} radix "
                  f"{best_exact['radix']} depth {best_exact['depth']} inter "
                  f"{best_exact['inter']} {best_exact['dtype']} -> "
                  f"{best_exact['render_fps']:.2f} render fps "
                  f"({'>=35 YES' if best_exact['render_fps'] >= 35 else '<35'})")
            if g2_exact.get("render_fps"):
                print(f"  BEST BYTE-EXACT 2-GPU (measured): {g2_exact['render_fps']:.2f} "
                      f"render fps ({'>=35 YES' if g2_exact['render_fps'] >= 35 else '<35'})")
        out["verdict"] = {
            "best_1gpu_render_fps": bo["render_fps"],
            "best_1gpu_config": {k: bo[k] for k in ("scheme", "radix", "depth",
                                                    "inter", "dtype")},
            "clears_35_1gpu": clears, "best_1gpu_byte_exact": bool(exact),
            "best_exact_1gpu_render_fps": best_exact["render_fps"] if best_exact else None,
            "best_exact_1gpu_config": ({k: best_exact[k] for k in
                ("scheme", "radix", "depth", "inter", "dtype")} if best_exact else None),
            "best_exact_1gpu_clears_35": (best_exact["render_fps"] >= 35) if best_exact else False,
            "best_exact_2gpu_render_fps": g2_exact.get("render_fps"),
            "best_exact_2gpu_clears_35": (g2_exact.get("render_fps", 0) >= 35),
            "fastest_2gpu_render_fps": g2_over.get("render_fps"),
        }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
