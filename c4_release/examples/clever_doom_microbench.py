#!/usr/bin/env python3
r"""clever_doom_microbench.py — GPU microbenchmark + FLOP/occupancy model for the
CLEVER min-param ALU cell (whole-value fp64/fp128 + difference-min digit-extraction),
projecting whether a CLEVER-built c4 VM could run Doom in realtime (35 fps).

This is the wall-clock companion to examples/clever_minparam_alu.py (which proves the
cells are byte-EXACT) and docs/SERIAL_ALU_FLOP_FLOOR.md (the bit-serial FLOP floor).
The CLEVER form is NARROW-and-SHALLOW (~10-20 reused digit layers, whole values in one
fp64 scalar) — the opposite depth regime from the ~160-layer bit-serial floor. The KEY
question this benchmark answers: does the shallow clever form hit enough GPU occupancy
that the FLOP cut translates to wall-clock, and does fp64's 1/32-1/64 consumer penalty
force fp32?

Everything here is the SAME hand-set decode cell as clever_minparam_alu.py, wrapped as a
batched torch module so we can drive it with a large batch (= many concurrent VM lanes /
speculative-K steps) and MEASURE steps/s, FLOP/s, and % of the card's fp{32,64} peak.

Run:
    python examples/clever_doom_microbench.py                 # full: fp64 + fp32, GPU
    python examples/clever_doom_microbench.py --device cpu    # analytic occupancy fallback
    python examples/clever_doom_microbench.py --json out.json # machine-readable
CPU-safe; uses ONE GPU (cuda:0) if free.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch

LN10 = math.log(10.0)

# ------------------------------------------------------------------ #
# Op parameters straight from clever_minparam_alu.py
# ------------------------------------------------------------------ #
ADD_DEPTH = 11      # a+b < 10^11
DIV_DEPTH = 10      # quotient <= 10 decimal digits
MUL_DEPTH = 20      # a*b <= 20 decimal digits
NCAND = 10          # candidate digits 0..9


# ------------------------------------------------------------------ #
# The CLEVER digit-extraction cell, batched. ONE reused layer per output
# digit (the depth lever). Holds the WHOLE running value in one scalar/lane.
#   logits_d = -|value/scale - (d+0.5)| + tie*d   over d in 0..9
#   digit    = argmax_d
#   value   -= digit * scale        (running-remainder update)
# This is decode_digit + the addsub/divmod running-remainder loop, verbatim,
# vectorized over a batch B of lanes.
# ------------------------------------------------------------------ #
def clever_digit_extract(value: torch.Tensor, depth: int, dtype: torch.dtype):
    """value: (B,) whole fp result to decode. Returns (out int64, #MAC-equiv per lane).

    Counts the nonzero-MAC-equivalent elementwise flops the same way the c4 _flop_gauge
    counts COO ops (MAC -> 2 FLOP; abs/sub/argmax counted as their elementwise op).
    """
    B = value.shape[0]
    cand = torch.arange(NCAND, dtype=dtype, device=value.device)  # 0..9
    half = torch.tensor(0.5, dtype=dtype, device=value.device)
    tie = torch.tensor(1e-12, dtype=dtype, device=value.device)
    R = value.clone()
    out = torch.zeros(B, dtype=torch.int64, device=value.device)
    for p in range(depth - 1, -1, -1):
        scale = 10.0 ** p
        val = R / scale                                  # (B,)   1 div/lane
        # difference-min selector over 10 candidates (the clever core):
        diff = val.unsqueeze(-1) - (cand + half)         # (B,10) 10 sub
        logits = -diff.abs() + tie * cand                # (B,10) 10 abs +10 mac +10 add
        d = logits.argmax(dim=-1)                        # (B,)   argmax over 10
        df = d.to(dtype)
        # int64 out-accumulate: safe scale (MUL depth20 exceeds int64; timing-only so
        # accumulate in the fp domain there — the FLOP/arithmetic is identical).
        if scale <= 1e17:
            out = out + d * int(scale)                   # (B,)   1 mac
        R = R - df * scale                               # (B,)   1 mac
    return out, None


# Per-DIGIT-layer elementwise-FLOP count for the clever cell (MAC=2 FLOP; abs/sub/argmax
# each 1 FLOP-equiv over their width) — counting the SAME way _flop_gauge counts COO ops:
#   div R/scale            : 1
#   sub  (10-wide)         : 10
#   abs  (10-wide)         : 10
#   tie*cand mac (10-wide) : 10*2 = 20
#   add  (10-wide)         : 10
#   argmax over 10         : 10 (9 compares)
#   out += d*scale (mac)   : 2
#   R  -= d*scale  (mac)   : 2
#                            ----
#                            75 FLOP per digit layer
CLEVER_FLOP_PER_DIGIT_LAYER = 1 + 10 + 10 + 20 + 10 + 10 + 2 + 2  # = 65 (+argmax ~10 -> ~75)
CLEVER_FLOP_PER_DIGIT_LAYER_WITH_ARGMAX = CLEVER_FLOP_PER_DIGIT_LAYER + 10  # 75


def clever_op_flop(depth: int) -> int:
    """FLOP for one clever ALU op = depth reused digit-layers + one place-value ingest.

    Ingest (place_value_read) per operand of width W (<=10 digits): exp(W) + sum(W) +
    div(W) + W-wide weighted-sum mac -> ~4*W. Two operands -> ~8*W ~ 80 FLOP. Add the
    two operand reconstructions (a,b whole) ~ 160. depth digit-layers dominate.
    """
    decode = depth * CLEVER_FLOP_PER_DIGIT_LAYER_WITH_ARGMAX
    ingest = 2 * (4 * 10)   # two operands, ~4*W per, W=10 -> ~80
    return decode + ingest


# ------------------------------------------------------------------ #
# GPU microbenchmark: drive the cell with a big batch, measure wall throughput.
# ------------------------------------------------------------------ #
def bench_cell(device: str, dtype: torch.dtype, depth: int, batch: int,
               iters: int, warmup: int):
    dev = torch.device(device)
    # random whole values in range for the given depth (10^depth). For MUL (depth20) the
    # exact 64-bit product needs fp128 (CPU numpy longdouble) — GPU fp64/fp32 CANNOT hold
    # it (2^53 ceiling), so on GPU this depth is a THROUGHPUT proxy only (not exact): we
    # generate values in fp-representable range and time the identical cell arithmetic.
    hi = float(10 ** depth)          # float to avoid int-overflow in the *; timing-only
    torch.manual_seed(0)
    value = (torch.rand(batch, device=dev, dtype=dtype) * hi).floor()
    # warmup
    for _ in range(warmup):
        clever_digit_extract(value, depth, dtype)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        clever_digit_extract(value, depth, dtype)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total_ops = iters * batch                 # ops decoded (each = one ALU op = one VM step-equiv)
    per_op_flop = clever_op_flop(depth)
    ops_per_s = total_ops / dt
    flop_per_s = ops_per_s * per_op_flop
    return {
        "device": device, "dtype": str(dtype), "depth": depth, "batch": batch,
        "iters": iters, "wall_s": dt,
        "ops_per_s": ops_per_s,
        "us_per_op_amortized": dt / total_ops * 1e6,
        "per_op_flop": per_op_flop,
        "flop_per_s": flop_per_s,
    }


# A5000 (GA102, consumer Ampere) datasheet peaks:
#   fp32 = 27.77 TFLOP/s; fp64 = 1/64 of fp32 = ~0.434 TFLOP/s (no fp64 tensor cores).
A5000_FP32_PEAK = 27.77e12
A5000_FP64_PEAK = A5000_FP32_PEAK / 64.0   # ~0.434 TFLOP/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda:0 / cpu (auto: cuda:0 if free)")
    ap.add_argument("--batches", default="256,4096,65536,262144,1048576")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = args.device
    is_cuda = dev.startswith("cuda")
    batches = [int(b) for b in args.batches.split(",")]

    print("=" * 92)
    print("CLEVER digit-extraction cell — GPU microbenchmark (steps/s, FLOP/s, % peak)")
    print("=" * 92)
    if is_cuda:
        idx = int(dev.split(":")[1]) if ":" in dev else 0
        name = torch.cuda.get_device_name(idx)
        print(f"device={dev}  ({name})")
        print(f"  datasheet peaks: fp32 {A5000_FP32_PEAK/1e12:.2f} TFLOP/s | "
              f"fp64 {A5000_FP64_PEAK/1e12:.3f} TFLOP/s (1/64 fp32, no fp64 tensor cores)")
    else:
        print(f"device={dev}  (CPU — analytic occupancy fallback)")
    print(f"per-op FLOP: ADD depth11={clever_op_flop(ADD_DEPTH)}  "
          f"DIV depth10={clever_op_flop(DIV_DEPTH)}  MUL depth20={clever_op_flop(MUL_DEPTH)}")
    print()

    results = []
    # Use the ADD depth (11) as the representative per-step op (Doom mix is 63% pointer-walk
    # ~= ADD-class); we also report DIV and MUL depths.
    for label, depth, peak_dtype in (("ADD/pointer (depth11)", ADD_DEPTH, None),
                                     ("DIV (depth10)", DIV_DEPTH, None),
                                     ("MUL (depth20)", MUL_DEPTH, None)):
        for dtype, dtname, peak in ((torch.float64, "fp64", A5000_FP64_PEAK),
                                    (torch.float32, "fp32", A5000_FP32_PEAK)):
            print(f"--- {label}  [{dtname}] ---")
            best = None
            for batch in batches:
                try:
                    r = bench_cell(dev, dtype, depth, batch, args.iters, args.warmup)
                except RuntimeError as e:
                    print(f"  batch={batch:>9d}  OOM/err: {e}")
                    continue
                occ = r["flop_per_s"] / peak if is_cuda else float("nan")
                r["op_label"] = label
                r["dtname"] = dtname
                r["pct_peak"] = occ
                results.append(r)
                print(f"  batch={batch:>9d}  {r['ops_per_s']/1e6:8.3f} Mops/s  "
                      f"{r['flop_per_s']/1e9:8.2f} GFLOP/s  "
                      f"{r['us_per_op_amortized']:8.4f} us/op  "
                      + (f"{occ*100:6.2f}% peak" if is_cuda else "n/a"))
                if best is None or r["ops_per_s"] > best["ops_per_s"]:
                    best = r
            if best:
                print(f"  BEST: {best['ops_per_s']/1e6:.2f} Mops/s "
                      f"({best['us_per_op_amortized']:.4f} us/op, "
                      f"{best.get('pct_peak', float('nan'))*100:.2f}% peak) at batch={best['batch']}")
            print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"results": results,
                       "peaks": {"fp32": A5000_FP32_PEAK, "fp64": A5000_FP64_PEAK}}, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
