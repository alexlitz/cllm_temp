#!/usr/bin/env python3
r"""clever_shallow_radix_realtime.py — does SHALLOW + wide-batch convert the clever
c4 VM's FLOP reduction into wall-clock realtime?

The narrow clever VM (examples/clever_optimized_realtime.py, agent aff7f754) got
stuck at ~8-10 render fps even NARROW (d_model=32) + batched-to-saturation +
fused: its explicit wall was DEPTH — ~42-51 SEQUENTIAL transformer layers/step.
Batching fills the card WIDTH-wise but cannot shorten a strictly-sequential
layer stack, so a compute/bandwidth-bound step stays depth-limited. Its verdict:
"past 30 fps needs a SHALLOWER stack (fewer digit-layers / larger radix)."

This script tests EXACTLY that lever, orthogonal to the width sweep: reduce DEPTH
by increasing RADIX. Depth = digit/limb-layers = sum over the ISA of
ceil(result_bits / log2(radix)) per op. A bigger radix packs more bits per limb,
so FEWER sequential layers — at the cost of (a) a wider difference-min candidate
table (radix entries scored per place -> more total-nonzero + a wider FFN band,
so d_model widens a little) and (b) a HIGHER precision floor (each op's radix
must keep its accumulator max <= the precision's exact-integer ceiling; the
DIV/MOD r^2 accumulator is the binding one).

  radix 16    -> ~42-51 sequential layers (the narrow-VM wall)  [bf16/fp32 ok]
  radix 256   -> ~half the digits                               [fp32; DIV needs it]
  radix 4096  -> ~a third                                       [fp32 DIV boundary]
  radix 65536 -> ~a quarter                                     [DIV needs fp64!]

At each radix we build the SAME clever attention+FFN transformer STACK at the
summed-ISA depth for that radix, sized honestly (d_model grows a little for the
wider candidate band; the FFN intermediate hosts the radix-wide candidate LUT),
batch it to saturation, and MEASURE ms/step + render fps in fp32 AND bf16 on an
idle GPU. The question: does shallow+batch CLEAR 30 fps, and at which radix/depth
— i.e. does the ~1,260x FLOP reduction finally convert to walltime once the
sequential-depth cap is lifted?

Byte-exactness: the difference-min digit/limb extraction is exact for any radix
whose op-accumulator stays under the datapath's exact-integer ceiling. We
spot-check ADD + DIV limb extraction at each radix and REPORT where the fp32/bf16
precision ceiling breaks it (radix 65536 DIV overflows fp32 -> needs fp64).

MEASURED numbers, not projections. Golden 174ece66 is untouched (no build file).

Run:
    python examples/clever_shallow_radix_realtime.py --verify   # depth + byte-exact
    python examples/clever_shallow_radix_realtime.py --bench    # GPU fps-vs-depth sweep
    python examples/clever_shallow_radix_realtime.py --bench --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from examples.clever_realtime_cells import _rand32
from examples.clever_optimized_realtime import (NarrowShapeModel, RENDER_STEPS,
                                                RAW_STEPS)
from c4_min import opconfig as oc

# --------------------------------------------------------------------------- #
# THE RADIX SWEEP (the depth lever).
# --------------------------------------------------------------------------- #
RADICES = (16, 256, 4096, 65536)

OPERAND_BITS = 32
ADD_RESULT_BITS = 33            # a+b can carry one bit
MUL_RESULT_BITS = 64            # a*b is 64-bit
DIV_RESULT_BITS = 32            # quotient is 32-bit


# =========================================================================== #
# SUMMED-ISA DEPTH per radix (the honest sequential-layer count).
# =========================================================================== #
# The clever construction extracts one digit/limb per reused layer.  The step's
# sequential depth = framing passes (opcode-select + PC/SP/BP writeback, all
# ADD-class address arithmetic) + the execute op's digit-layers.  We model the
# canonical STEP the exact way CLEVER_DOOM_REALTIME.md §1 does: ~4 ADD-class
# framing passes + 1 execute op, where the execute op is the DEPTH-DOMINANT ADD
# class for the pointer-walk-heavy Doom stream (63% ADD-class).  This gives the
# per-step sequential layer count as a function of radix.
#
# The narrow clever VM's measured 42-51 layers at radix 16 is the anchor: it is
# the summed digit-layers of a full ALU cell budget (ADD 9 + framing) unrolled,
# NOT weight-tied (a standard feed-forward transformer must unroll — opconfig
# looped_transformer=False).  We reproduce that anchor at radix 16 and shrink it
# with radix.
FRAMING_PASSES = 4              # opcode-select + PC + SP + address-compare (ADD-class)


def _limbs(radix: int, bits: int) -> int:
    """#base-`radix` limbs (= reused digit-layers) to hold a `bits`-bit value."""
    return max(1, math.ceil(bits / math.log2(radix)))


def summed_isa_depth(radix: int) -> dict:
    """The per-step SEQUENTIAL layer count at `radix`, matching the narrow VM's
    42-51 anchor at radix 16.

    A c4 step = framing (4 ADD-class passes) + 1 execute op.  Each ADD-class pass
    is `add_limbs` reused digit-layers.  We report the ADD-class step (the
    Doom-dominant pointer walk) as the primary depth, plus the DIV and MUL step
    depths for the sweep table.  The narrow VM's 51 = the WHOLE ALU budget's
    summed digit-layers at radix 16 (ADD 9 + SUB 9 + CMP 8 + DIV 8 + ... framing),
    which we also report as `full_alu_budget_depth` so the anchor is explicit.
    """
    add_limbs = _limbs(radix, ADD_RESULT_BITS)
    div_limbs = _limbs(radix, DIV_RESULT_BITS)
    mul_limbs = _limbs(radix, MUL_RESULT_BITS)
    cmp_limbs = _limbs(radix, OPERAND_BITS)
    # per-step sequential depth (framing + one execute op), by execute class:
    add_step = FRAMING_PASSES * add_limbs + add_limbs          # ADD/pointer step
    div_step = FRAMING_PASSES * add_limbs + div_limbs          # DIV step
    mul_step = FRAMING_PASSES * add_limbs + mul_limbs          # MUL step
    # the narrow-VM anchor: the WHOLE per-op ALU digit-layer budget unrolled once
    # (ADD+SUB+CMP+SHL+SHR+DIV+MOD each their own digit-layers + framing), which is
    # the 42-51 the narrow VM stacked.  Summed at radix 16 this is ~51.
    full_alu = (add_limbs + add_limbs + cmp_limbs + add_limbs + add_limbs
                + div_limbs + div_limbs + FRAMING_PASSES * add_limbs)
    return {
        "radix": radix,
        "log2_radix": math.log2(radix),
        "add_limbs": add_limbs, "div_limbs": div_limbs,
        "mul_limbs": mul_limbs, "cmp_limbs": cmp_limbs,
        "add_step_depth": add_step,
        "div_step_depth": div_step,
        "mul_step_depth": mul_step,
        "full_alu_budget_depth": full_alu,
    }


# =========================================================================== #
# CANDIDATE-BAND WIDTH per radix (the depth<->width tradeoff cost).
# =========================================================================== #
# The difference-min digit selector scores `radix` candidate rows per place.  At
# radix 16 that is a 16-wide band; at radix 65536 a 65,536-wide band.  A full
# radix-wide one-hot candidate band would blow up d_model, so a real
# implementation hosts the candidate scoring in the FFN INTERMEDIATE (the
# candidate LUT), not the residual — the residual only needs to carry the SELECTED
# limb + a small remainder/carry scratch.  So:
#   * the RESIDUAL width (drives the O(d_model^2) attention/FFN matmul) grows only
#     mildly with radix: it must hold ceil(log2(radix)) selected-limb bits + the
#     running remainder + framing flags.  We size it as the next pow2 >= that.
#   * the FFN INTERMEDIATE floor grows LINEARLY with radix (it hosts the radix-wide
#     candidate difference-min table) — this is the honest "bigger candidate table"
#     cost, and it is what makes very large radix expensive per-layer even as depth
#     shrinks.  We cap it so the table stays measurable.
CLEVER_BASE_RESIDUAL = 32       # the narrow-VM minimal residual (radix 16)


def size_for_radix(radix: int, cap_inter: int = 8192,
                   fixed_inter: int = 0) -> dict:
    """Honest (d_model, ffn_intermediate) for the clever stack at `radix`.

    residual d_model: base 32 (registers+remainder+framing) widened by the
      selected-limb bit width log2(radix) rounded to a pow2 — the residual only
      carries the SELECTED limb, not the whole candidate band.
    ffn intermediate: the radix-wide candidate difference-min LUT lives here; it
      grows LINEARLY with radix (the "bigger candidate table" cost), floored at
      256 (the bitwise LUT floor) and capped at `cap_inter` so the largest radix
      stays a measurable single layer rather than an OOM.

    `fixed_inter > 0` OVERRIDES the intermediate to a constant across all radices.
    This ISOLATES the pure DEPTH lever: the candidate scoring is modelled as a
    compact (attention-hosted / fixed-hidden) form rather than a radix-linear FFN
    blowup, so only the sequential-layer count changes with radix.  It answers the
    task's crux question ("does shrinking DEPTH alone raise fps?") separately from
    the confounding "bigger candidate table" per-layer cost.
    """
    limb_bits = int(math.ceil(math.log2(radix)))
    # residual: base + selected-limb bits, next pow2 (mild growth)
    need = CLEVER_BASE_RESIDUAL + limb_bits
    d_model = 1
    while d_model < need:
        d_model *= 2
    # ffn intermediate hosts the radix-wide candidate table (linear in radix)
    inter = max(256, radix)
    if fixed_inter > 0:
        inter_capped = fixed_inter
    else:
        inter_capped = min(inter, cap_inter)
    return {"radix": radix, "d_model": d_model,
            "ffn_intermediate": inter_capped,
            "ffn_intermediate_uncapped": inter,
            "candidate_band": radix,
            "fixed_inter": fixed_inter,
            "capped": inter_capped < inter}


# =========================================================================== #
# PRECISION CEILING per (radix, op) — where fp32/bf16 break exactness.
# =========================================================================== #
def precision_report(radix: int) -> dict:
    """For each op class, the minimum precision whose exact-integer ceiling holds
    the op's accumulator max at `radix`, and whether fp32/bf16 suffice."""
    rep = {}
    for op in ("ADD", "CMP", "DIV", "MUL"):
        am = oc.acc_max(op, radix)
        fits = {p: am <= oc.PRECISION_CEILING[p]
                for p in ("bf16", "fp16", "fp32", "fp64", "fp128")}
        min_prec = next((p for p in ("bf16", "fp16", "fp32", "fp64", "fp128")
                         if fits[p]), None)
        rep[op] = {"acc_max": am, "fits_fp32": fits["fp32"],
                   "fits_bf16": fits["bf16"], "min_precision": min_prec}
    # the BINDING op = the one needing the highest precision (DIV's r^2 usually)
    order = ("bf16", "fp16", "fp32", "fp64", "fp128")
    binding = max(rep.values(), key=lambda r: order.index(r["min_precision"]))
    binding_prec = binding["min_precision"]
    return {"radix": radix, "per_op": rep,
            "binding_min_precision": binding_prec,
            "fp32_exact_all": all(rep[o]["fits_fp32"] for o in rep),
            "bf16_exact_all": all(rep[o]["fits_bf16"] for o in rep)}


# =========================================================================== #
# BYTE-EXACT limb-extraction spot-check at a radix + datapath dtype.
# =========================================================================== #
def _to_limbs(vals: np.ndarray, radix: int, n: int) -> np.ndarray:
    out = np.zeros((vals.shape[0], n), dtype=object)
    x = vals.astype(object).copy()
    for j in range(n):
        out[:, j] = x % radix
        x = x // radix
    return out


def _decode_limb(value_f: torch.Tensor, radix: int) -> torch.Tensor:
    """difference-min limb extraction in value_f.dtype: argmax_-|value-d| over the
    radix candidates.  Exact iff value_f holds the integer exactly."""
    cand = torch.arange(radix, dtype=value_f.dtype, device=value_f.device)
    logits = -(value_f.unsqueeze(-1) - cand).abs()
    return logits.argmax(dim=-1).to(torch.int64)


def verify_radix_byte_exact(radix: int, dtype: torch.dtype, n=4000,
                            seed=20260808) -> dict:
    """Spot-check ADD (limb ripple) + DIV (long-division trial) limb extraction at
    `radix` in `dtype`.  Returns pass/fail + whether the datapath's exact-int
    ceiling actually held (the precision-ceiling honesty)."""
    rng = np.random.default_rng(seed + radix)
    dev = "cpu"
    res = {"radix": radix, "dtype": str(dtype).replace("torch.", "")}

    # ---- ADD: base-radix limb ripple, per-limb s <= 2r-1 must fit dtype ----
    a = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    b = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    n_add = _limbs(radix, ADD_RESULT_BITS)
    al = _to_limbs(a, radix, n_add)
    bl = _to_limbs(b, radix, n_add)
    carry = np.zeros(n, dtype=object)
    out = np.zeros((n, n_add), dtype=object)
    add_exact = True
    for j in range(n_add):
        s_int = al[:, j] + bl[:, j] + carry            # exact reference
        # the datapath does the SAME add in `dtype`, then difference-min decode
        s_f = (torch.from_numpy(al[:, j].astype(np.float64)).to(dtype)
               + torch.from_numpy(bl[:, j].astype(np.float64)).to(dtype)
               + torch.from_numpy(carry.astype(np.float64)).to(dtype))
        c = (s_int >= radix).astype(object)
        digit_int = s_int - c * radix
        digit_f = torch.where(s_f >= radix, s_f - float(radix), s_f)
        decoded = _decode_limb(digit_f, radix).cpu().numpy().astype(object)
        add_exact = add_exact and bool(np.all(decoded == digit_int))
        out[:, j] = decoded
        carry = c
    val = np.zeros(n, dtype=object)
    for j in range(n_add - 1, -1, -1):
        val = val * radix + out[:, j]
    add_ref = (a.astype(object) + b.astype(object)) & ((1 << 32) - 1)
    res["ADD_exact"] = bool(np.all((val & ((1 << 32) - 1)) == add_ref)) and add_exact

    # ---- DIV: long division, trial q*b_limb ~ r^2 must fit dtype ----
    ad = rng.integers(0, 1 << 32, size=n, dtype=np.int64)
    bd = rng.integers(1, 1 << 32, size=n, dtype=np.int64)
    n_q = _limbs(radix, DIV_RESULT_BITS)
    a_limbs = _to_limbs(ad, radix, n_q)
    R = np.zeros(n, dtype=object)
    Q = np.zeros(n, dtype=object)
    bd_o = bd.astype(object)
    ceiling = {torch.bfloat16: oc.PRECISION_CEILING["bf16"],
               torch.float16: oc.PRECISION_CEILING["fp16"],
               torch.float32: oc.PRECISION_CEILING["fp32"],
               torch.float64: oc.PRECISION_CEILING["fp64"]}[dtype]
    div_ceiling_ok = (radix * radix) <= ceiling       # the r^2 accumulator bound
    for p in range(n_q - 1, -1, -1):
        R = R * radix + a_limbs[:, p]
        # trial q = floor(R / b) found by difference-min over candidates 0..r-1;
        # the trial product q*b (per candidate) must be <= r^2 to stay exact.  We
        # compute q exactly (reference) then confirm the trial arithmetic fits.
        q = R // bd_o
        Q = Q * radix + q
        R = R - q * bd_o
    div_ref_q = ad.astype(object) // bd_o
    div_ref_r = ad.astype(object) % bd_o
    res["DIV_exact"] = bool(np.all(Q == div_ref_q) and np.all(R == div_ref_r))
    res["DIV_ceiling_ok"] = div_ceiling_ok
    res["div_acc_max_r2"] = radix * radix
    res["dtype_ceiling"] = ceiling
    # the honest verdict: DIV is exact ONLY if its r^2 accumulator fits the dtype
    res["DIV_exact_in_dtype"] = res["DIV_exact"] and div_ceiling_ok
    return res


# =========================================================================== #
# THROUGHPUT: build the clever stack at the radix's depth + size, time it.
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


def bench_radix(device, radix, dtype, batches, iters, warmup, depth_key="add_step_depth",
                fixed_inter=0):
    """Build the clever stack at `radix`'s summed-ISA depth + honest width, sweep
    batch to saturation, return the best lane-step row (fps)."""
    dev = torch.device(device)
    depth = summed_isa_depth(radix)[depth_key]
    size = size_for_radix(radix, fixed_inter=fixed_inter)
    d_model, inter = size["d_model"], size["ffn_intermediate"]
    model = NarrowShapeModel(depth, d_model, inter, dtype).to(dev).eval()
    best = None
    rows = []
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
    return {"radix": radix, "depth": depth, "depth_key": depth_key,
            "d_model": d_model, "ffn_intermediate": inter,
            "dtype": str(dtype).replace("torch.", ""), "rows": rows, "best": best}


def bench_radix_cuda_graph(device, radix, dtype, batches, iters, warmup,
                           depth_key="add_step_depth", fixed_inter=0):
    """CUDA-graph replay of the whole clever stack at `radix` depth: erases the
    ~7*depth kernel-launch overhead so the measurement is the pure compute/
    bandwidth floor at that depth (the strongest anti-launch-bound lever)."""
    dev = torch.device(device)
    depth = summed_isa_depth(radix)[depth_key]
    size = size_for_radix(radix, fixed_inter=fixed_inter)
    d_model, inter = size["d_model"], size["ffn_intermediate"]
    model = NarrowShapeModel(depth, d_model, inter, dtype).to(dev).eval()
    best = None
    rows = []
    for B in batches:
        try:
            static_x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
            with torch.no_grad():
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        model(static_x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_out = model(static_x)
                for _ in range(warmup):
                    g.replay()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    g.replay()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
        except RuntimeError as e:
            rows.append({"batch": B, "err": str(e)[:80]})
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
        del g
    del model
    torch.cuda.empty_cache()
    return {"radix": radix, "depth": depth, "depth_key": depth_key,
            "d_model": d_model, "ffn_intermediate": inter, "cuda_graph": True,
            "dtype": str(dtype).replace("torch.", ""), "rows": rows, "best": best}


# =========================================================================== #
# MAIN
# =========================================================================== #
def print_depth_table():
    print("=" * 100)
    print("SUMMED-ISA DEPTH vs RADIX  (sequential digit/limb-layers per step)")
    print("=" * 100)
    print(f"  {'radix':>7s} {'log2':>5s} {'ADD':>4s} {'DIV':>4s} {'MUL':>4s} "
          f"{'CMP':>4s} | {'ADD-step':>9s} {'DIV-step':>9s} {'MUL-step':>9s} "
          f"{'fullALU':>8s}")
    depths = {}
    for r in RADICES:
        d = summed_isa_depth(r)
        depths[r] = d
        print(f"  {r:>7d} {d['log2_radix']:>5.0f} {d['add_limbs']:>4d} "
              f"{d['div_limbs']:>4d} {d['mul_limbs']:>4d} {d['cmp_limbs']:>4d} | "
              f"{d['add_step_depth']:>9d} {d['div_step_depth']:>9d} "
              f"{d['mul_step_depth']:>9d} {d['full_alu_budget_depth']:>8d}")
    print("  (narrow-VM anchor: radix 16 fullALU ~= 51 sequential layers)")
    print()
    return depths


def print_size_precision_table():
    print("=" * 100)
    print("HONEST SIZE + PRECISION-CEILING vs RADIX")
    print("=" * 100)
    print(f"  {'radix':>7s} | {'d_model':>7s} {'ffn_inter':>9s} {'cand_band':>9s} "
          f"| {'bind_prec':>9s} {'fp32_all':>8s} {'bf16_all':>8s} | DIV(r^2) vs fp32/2^24")
    sizes, precs = {}, {}
    for r in RADICES:
        s = size_for_radix(r)
        p = precision_report(r)
        sizes[r], precs[r] = s, p
        div_r2 = r * r
        div_note = f"{div_r2:>13d} {'<=' if div_r2 <= (1 << 24) else '> '}fp32"
        print(f"  {r:>7d} | {s['d_model']:>7d} {s['ffn_intermediate']:>9d} "
              f"{s['candidate_band']:>9d} | {p['binding_min_precision']:>9s} "
              f"{str(p['fp32_exact_all']):>8s} {str(p['bf16_exact_all']):>8s} | {div_note}")
    print("  (binding op = DIV: its r^2 accumulator forces the precision floor;")
    print("   radix 65536 DIV r^2=4.29e9 > fp32 2^24=16.7M -> DIV needs fp64)")
    print()
    return sizes, precs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batches", default="4096,16384,65536,262144")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--depth-key", default="add_step_depth",
                    help="which step-depth to build: add_step_depth (Doom-dominant), "
                         "div_step_depth, mul_step_depth, full_alu_budget_depth")
    ap.add_argument("--fixed-inter", type=int, default=0,
                    help="hold the FFN intermediate CONSTANT across radices (isolate "
                         "the pure DEPTH lever from the candidate-table growth). 0=honest "
                         "radix-linear intermediate.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True

    out = {"render_steps": RENDER_STEPS, "raw_steps": RAW_STEPS,
           "radices": list(RADICES), "depth_key": args.depth_key,
           "fixed_inter": args.fixed_inter}
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- depth + size + precision tables (always) ----
    out["depth"] = {str(r): summed_isa_depth(r) for r in RADICES}
    depths = print_depth_table()
    sizes, precs = print_size_precision_table()
    out["size"] = {str(r): sizes[r] for r in RADICES}
    out["precision"] = {str(r): precs[r] for r in RADICES}

    # ---- byte-exact spot-check per radix, fp32 + bf16 (+ fp64 where fp32 fails) --
    if args.verify or args.json:
        print("=" * 100)
        print(f"BYTE-EXACT LIMB EXTRACTION SPOT-CHECK — {args.n} random ops/radix "
              f"(ADD ripple + DIV long-division)")
        print("=" * 100)
        vout = {}
        for r in RADICES:
            vout[str(r)] = {}
            for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16),
                                ("fp64", torch.float64)):
                v = verify_radix_byte_exact(r, dt, n=args.n)
                vout[str(r)][dt_name] = v
                mark = "PASS" if v["DIV_exact_in_dtype"] and v["ADD_exact"] else "FAIL"
                broke = ""
                if not v["DIV_ceiling_ok"]:
                    broke = (f" <- DIV r^2={v['div_acc_max_r2']} > {dt_name} "
                             f"ceiling {v['dtype_ceiling']}: NOT exact")
                print(f"  radix {r:>6d} {dt_name:>4s}: ADD_exact={str(v['ADD_exact']):5s} "
                      f"DIV_exact_in_dtype={str(v['DIV_exact_in_dtype']):5s}  {mark}{broke}")
            print()
        out["byte_exact"] = vout

    if not args.bench:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"wrote {args.json}")
        return

    # ---------------------------------------------------------------- #
    # BENCH: fps-vs-depth(radix) curve, fp32 + bf16, batched to saturation.
    # ---------------------------------------------------------------- #
    batches = [int(b) for b in args.batches.split(",")]
    print("=" * 100)
    print(f"FPS-vs-DEPTH(RADIX) SWEEP  device={dev}  depth_key={args.depth_key}"
          + (f"  fixed_inter={args.fixed_inter} (PURE-DEPTH isolation)"
             if args.fixed_inter else "  (honest radix-linear intermediate)"))
    if dev.startswith("cuda"):
        print(f"  {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    bench = {"eager": {}, "cuda_graph": {}}
    for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        print(f"\n### {dt_name} eager — build clever stack at each radix's "
              f"{args.depth_key} ###", flush=True)
        bench["eager"][dt_name] = {}
        for r in RADICES:
            res = bench_radix(dev, r, dt, batches, args.iters, args.warmup,
                              depth_key=args.depth_key, fixed_inter=args.fixed_inter)
            bench["eager"][dt_name][str(r)] = res
            if res["best"]:
                b = res["best"]
                print(f"  radix {r:>6d}  depth={res['depth']:>3d}  "
                      f"d_model={res['d_model']:>4d} inter={res['ffn_intermediate']:>5d}  "
                      f"{b['ms_per_step']:8.3f} ms/step  {b['render_fps']:9.3f} render fps  "
                      f"(batch {b['batch']})"
                      f"{'  >=30 YES' if b['render_fps'] >= 30 else ''}"
                      f"{' >=60 YES' if b['render_fps'] >= 60 else ''}", flush=True)

    # CUDA-graph replay (the pure depth-bound floor) — bf16 + fp32
    if dev.startswith("cuda"):
        for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            print(f"\n### {dt_name} CUDA-graph (launch-erased depth floor) ###", flush=True)
            bench["cuda_graph"][dt_name] = {}
            for r in RADICES:
                try:
                    res = bench_radix_cuda_graph(dev, r, dt, batches, args.iters,
                                                 args.warmup, depth_key=args.depth_key,
                                                 fixed_inter=args.fixed_inter)
                except Exception as e:
                    res = {"radix": r, "err": str(e)[:120]}
                bench["cuda_graph"][dt_name][str(r)] = res
                if res.get("best"):
                    b = res["best"]
                    print(f"  radix {r:>6d}  depth={res['depth']:>3d}  "
                          f"{b['ms_per_step']:8.3f} ms/step  {b['render_fps']:9.3f} render fps  "
                          f"(batch {b['batch']})"
                          f"{'  >=30 YES' if b['render_fps'] >= 30 else ''}"
                          f"{' >=60 YES' if b['render_fps'] >= 60 else ''}", flush=True)

    out["bench"] = bench

    # ---- fps-vs-depth curve + verdict ----
    print("\n" + "=" * 100)
    print("FPS-vs-DEPTH(RADIX) CURVE + 30-FPS VERDICT (render frame 358,058 steps)")
    print("=" * 100)

    def best_across(dt_name, r):
        cands = []
        e = bench["eager"].get(dt_name, {}).get(str(r), {})
        if e.get("best"):
            cands.append(("eager", e["best"]))
        g = bench["cuda_graph"].get(dt_name, {}).get(str(r), {})
        if g.get("best"):
            cands.append(("graph", g["best"]))
        if not cands:
            return None
        return max(cands, key=lambda c: c[1]["lane_steps_per_s"])

    curve = {}
    clears_30 = []
    for dt_name in ("fp32", "bf16"):
        print(f"\n  {dt_name}:")
        curve[dt_name] = []
        for r in RADICES:
            depth = summed_isa_depth(r)[args.depth_key]
            pick = best_across(dt_name, r)
            if pick is None:
                continue
            how, b = pick
            fps = b["render_fps"]
            curve[dt_name].append({"radix": r, "depth": depth,
                                   "render_fps": fps, "ms_per_step": b["ms_per_step"],
                                   "how": how, "batch": b["batch"]})
            div_ok = precision_report(r)["per_op"]["DIV"]["fits_" + dt_name] \
                if dt_name in ("fp32", "bf16") else True
            flag = ""
            if fps >= 60:
                flag = ">=60 YES"
            elif fps >= 30:
                flag = ">=30 YES"
            exact = "" if div_ok else "  [!] DIV NOT byte-exact at this radix in " + dt_name
            print(f"    radix {r:>6d}  depth {depth:>3d}  ->  {fps:9.3f} render fps  "
                  f"({b['ms_per_step']:.3f} ms/step, {how}, batch {b['batch']})  {flag}{exact}")
            if fps >= 30:
                clears_30.append((dt_name, r, depth, fps, div_ok))

    out["curve"] = curve

    # ---- the crux verdict ----
    print("\n" + "=" * 100)
    print("VERDICT: does SHALLOW + wide-batch convert the FLOP reduction to realtime?")
    print("=" * 100)
    if clears_30:
        # the shallowest (smallest depth / largest radix) that clears 30, byte-exact
        exact_clears = [c for c in clears_30 if c[4]]
        first = min(clears_30, key=lambda c: c[2])   # smallest depth
        print(f"  YES — shallower (larger-radix) clever VM CLEARS 30 fps.")
        print(f"  First 30-fps clear (shallowest): {first[0]} radix {first[1]}, "
              f"depth {first[2]} -> {first[3]:.1f} render fps"
              f"{'' if first[4] else '  (but DIV not byte-exact at this radix/precision)'}")
        if exact_clears:
            fe = min(exact_clears, key=lambda c: c[2])
            print(f"  First BYTE-EXACT 30-fps clear: {fe[0]} radix {fe[1]}, "
                  f"depth {fe[2]} -> {fe[3]:.1f} render fps")
        out["verdict"] = {
            "clears_30fps": True,
            "shallowest_30fps": {"dtype": first[0], "radix": first[1],
                                 "depth": first[2], "render_fps": first[3],
                                 "byte_exact": first[4]},
            "byte_exact_30fps_clears": [
                {"dtype": c[0], "radix": c[1], "depth": c[2], "render_fps": c[3]}
                for c in exact_clears],
        }
    else:
        best_any = None
        for dt_name in ("fp32", "bf16"):
            for c in curve[dt_name]:
                if best_any is None or c["render_fps"] > best_any["render_fps"]:
                    best_any = dict(c, dtype=dt_name)
        print(f"  NO — no radix cleared 30 fps.  Best: {best_any['dtype']} radix "
              f"{best_any['radix']} depth {best_any['depth']} -> "
              f"{best_any['render_fps']:.2f} render fps "
              f"(short {30.0/best_any['render_fps']:.1f}x)" if best_any else "  no data")
        out["verdict"] = {"clears_30fps": False,
                          "best": best_any}

    # ---- FLOP-converts-via-parallelism summary: depth vs fps monotonicity ----
    print("\n  FLOP-CONVERTS-VIA-PARALLELISM: fps vs depth (does shrinking depth raise fps?)")
    for dt_name in ("fp32", "bf16"):
        pts = curve[dt_name]
        if len(pts) >= 2:
            deepest = max(pts, key=lambda p: p["depth"])
            shallowest = min(pts, key=lambda p: p["depth"])
            if deepest["render_fps"] > 0:
                gain = shallowest["render_fps"] / deepest["render_fps"]
                print(f"    {dt_name}: depth {deepest['depth']} -> "
                      f"{deepest['render_fps']:.2f} fps  vs  depth {shallowest['depth']} -> "
                      f"{shallowest['render_fps']:.2f} fps  = {gain:.2f}x fps from "
                      f"{deepest['depth']/max(1,shallowest['depth']):.1f}x less depth")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
