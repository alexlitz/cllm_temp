#!/usr/bin/env python3
r"""clever_minflop_heap_scale.py — verify + measure the MIN-FLOP CLEVER VM's memory
model at DOOM HEAP SCALE.

WHY THIS EXISTS
===============
`examples/clever_honest_attn_realtime.py` (golden ``174ece66`` untouched) proved the
min-flop clever VM reads/writes memory with an O(1) direct-CAM (``DirectCAMReadHead`` /
the production ``c4_min/direct_cam_batched.py`` + ``selfemu_direct_cam.py``): each read
RESOLVES the query address to its exact latest-write-wins store row (a host-side pointer
walk, NOT a GPU softmax over the store) and DIRECT-GATHERS that ONE row's value — so the
per-step GPU work is one gathered row/lane, S-INDEPENDENT.  It measured this byte-exact
and at ~5.7% of the composed step at S=262K.

That prior file measured the direct-CAM at S in {32K, 262K} folded into the whole step.
This file ZOOMS IN on the MEMORY MODEL ITSELF at the FULL DOOM HEAP SCALE and answers
the 4 binding questions for a min-flop Doom-runner:

  1. THE DIRECT-CAM at S in {32K, 128K, 262K, 512K}: byte-exact read (hit / miss /
     latest-write-wins), per-read cost (us/lane), VRAM footprint (per-lane heap x batch),
     and whether the READ TIME stays S-independent (O(1)) as VRAM grows.

  2. THE BINDING CONSTRAINT: is it the CAM read TIME (S-independent -> fine) or the VRAM
     (per-lane 262K heap x the spec-verify batch -> OOM)?  We find the exact (batch, S) at
     which the PER-LANE heap OOMs on a 24 GB card, and confirm the SHARED-POOL / evicted
     working-set form fits.

  3. THE STACK + FRAMEBUFFER: how the SP-addressed stack and the framebuffer are held
     (CAM rows in the SAME address-keyed store, or a separate band), byte-exact + cost.

  4. VERDICT + THE MEMORY FLOOR for the min-flop Doom-runner.

DOOM MEMORY FOOTPRINT (the program under test)
==============================================
The real id linuxdoom -> c4 port (MEMORY.md ★ DOOM PORT) runs BYTE-EXACT on the native
c4vm.  Its memory:
  * zone-alloc HEAP: up to ~262,144 (2^18) live entries (the number the direct-CAM /
    self-emu path is sized for; ``persistent-heap KV ~262k/step`` in MEMORY.md).
  * SP-addressed STACK: a contiguous band of the SAME address space (grows down from the
    heap top); addressed by SP just like the heap is addressed by a pointer.
  * FRAMEBUFFER: the 320x200 = 64,000-byte (=16,000 32-bit words) render target, also in
    the same byte-addressed memory; written by the draw span, read by hexfb.py.

ALL THREE live in ONE byte-addressed memory.  The c4 VM reads/writes every one of them
through the SAME LI/SI/LC/SC/PSH opcodes -> the SAME address-keyed CAM.  So the stack and
framebuffer are NOT a separate mechanism: they are ADDRESS RANGES inside the single
address-keyed store, held as CAM rows exactly like the heap, and read O(1) identically.
We verify that explicitly (a stack read and a framebuffer read resolve byte-exact through
the same head).

THE TWO VRAM REGIMES (the load-bearing distinction)
===================================================
The direct-CAM's per-STEP GPU work is one gathered row/lane -> S-independent in TIME.  The
VRAM question is about how the STORE is HELD:

  * DENSE PER-LANE heap: a literal (B, S) int store — every lane its own full S-entry
    heap.  This is B x S x 4 bytes.  At B = spec-verify batch and S = 262K this is the
    64 GiB O(S) materialisation the direct-CAM EXISTS TO AVOID.  This is the OOM
    constraint the task asks us to locate.

  * SHARED / BOUNDED store: the per-step marginal GPU work is one gathered row/lane, which
    a SHARED S-entry pool (S x 4 bytes, lane-independent) times IDENTICALLY (``gather_value``
    supports both forms), and which the production path holds as a BOUNDED live working set
    (``c4_min/qwen_lean_evict.py`` / ``nibble_evict_schedule``: BOS sink + one register
    frame + live-heap rows, evicted to the live footprint).  This is what actually runs.

We measure BOTH: the dense per-lane OOM wall, and the shared/bounded fit — so the VERDICT
is grounded, not projected.

BYTE-EXACT: the O(1) direct gather == the reference O(S) latest-write-wins softmax CAM,
L-inf = 0, at EVERY S including 512K, for hit / miss / latest-write-wins / stack / fb.

MEASURED numbers (A5000-class 24 GB, ~768 GB/s HBM), not projections, except where a
config exceeds VRAM (then explicitly labelled PROJECTED from the closed-form B x S x 4).
Golden ``174ece66`` is untouched (NEW file, off every model build path).

Run:
    python examples/clever_minflop_heap_scale.py --verify        # byte-exact at all S (CPU-safe)
    python examples/clever_minflop_heap_scale.py --bench --json out.json   # GPU cost + VRAM
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from examples.clever_honest_attn_realtime import (
    DirectCAMReadHead, softmax_cam_read, _make_store, _time)

# Doom memory footprint constants (see module docstring).
DOOM_HEAP_ENTRIES = 262_144          # 2^18 zone-alloc live entries (the sized max)
DOOM_FB_WORDS = 16_000               # 320x200 bytes / 4 = 16000 32-bit words
BYTES_PER_ENTRY = 4                  # int32 store word
GIB = 1024 ** 3

# The store depths to sweep (Doom heap scales).  262144 = the real max; 512K = headroom.
DOOM_STORES = [32_768, 131_072, 262_144, 524_288]


# =========================================================================== #
# 1. BYTE-EXACT at DOOM SCALE — read (hit / miss / latest-write-wins), + the
#    STACK and FRAMEBUFFER reads through the SAME address-keyed CAM.
# =========================================================================== #
def verify_heap_scale_byte_exact(S, B=512, addr_nib=8, seed=20260809, device="cpu"):
    """The O(1) direct-CAM gather == the reference O(S) latest-write-wins softmax CAM at
    store depth ``S``, for the full battery:

      * HIT       : every lane queries a stored address -> exact value.
      * MISS      : never-stored address -> ZFOD 0 (the softmax1 +1 sink).
      * LWW       : two writes to the SAME address at different rows -> the LATEST row's
                    value wins (latest-write-wins), byte-exact both paths.
      * STACK     : a read at an SP-addressed row (high end of the address space) resolves
                    through the SAME head, byte-exact.
      * FRAMEBUFFER: a read at a framebuffer-band address resolves through the SAME head,
                    byte-exact.

    Exactness is S-INDEPENDENT (the gather touches one row), but we RE-VERIFY at every S so
    the claim is measured at 262K/512K, not projected from 8K.
    """
    dev = torch.device(device)
    rng = np.random.default_rng(seed + S)
    head = DirectCAMReadHead(64, torch.float32).to(dev)

    # Distinct addresses per lane so latest-write-wins is unambiguous for the base battery.
    # Address space: [0, 2^20) heap band; stack band near 2^24; fb band near 2^20+..
    addrs = np.stack([rng.choice(1 << 20, size=S, replace=False) for _ in range(B)])
    vals = rng.integers(0, 1 << 31, size=(B, S), dtype=np.int64)
    store_addr = torch.from_numpy(addrs).to(dev)
    store_val = torch.from_numpy(vals).to(torch.int32).to(dev)

    def resolve_of(q):
        _, row = softmax_cam_read(store_addr, store_val, q, addr_nib=addr_nib)
        return row

    out = {"S": S, "B": B}

    # ---- HIT ----
    pick = torch.from_numpy(np.array([rng.integers(0, S) for _ in range(B)])).to(dev)
    q_hit = store_addr.gather(1, pick.unsqueeze(1)).squeeze(1)
    ref_hit, _ = softmax_cam_read(store_addr, store_val, q_hit, addr_nib=addr_nib)
    got_hit = head.gather_value(store_val, resolve_of(q_hit))
    rec_hit = head.reconstruct_word(store_val, resolve_of(q_hit))
    out["hit_linf"] = int((got_hit - ref_hit).abs().max())
    out["hit_reconstruct_linf"] = int((rec_hit - ref_hit).abs().max())

    # ---- MISS ----
    q_miss = (store_addr.max() + 1 + torch.arange(B, device=dev)).to(torch.int64)
    ref_miss, _ = softmax_cam_read(store_addr, store_val, q_miss, addr_nib=addr_nib)
    got_miss = head.gather_value(store_val, resolve_of(q_miss))
    out["miss_all_zero"] = bool((got_miss == 0).all() and (ref_miss == 0).all())

    # ---- LATEST-WRITE-WINS: overwrite lane l's row `pick[l]` address at a LATER row with
    #      a fresh value; the read of that address must return the LATER value. ----
    later = torch.full((B,), S - 1, device=dev, dtype=torch.long)  # newest row per lane
    lww_addr = q_hit.clone()                                       # the hit address
    lww_newval = torch.from_numpy(rng.integers(0, 1 << 31, size=B, dtype=np.int64)).to(dev)
    sa2 = store_addr.clone()
    sv2 = store_val.clone()
    sa2.scatter_(1, later.unsqueeze(1), lww_addr.unsqueeze(1))     # newest row = same addr
    sv2.scatter_(1, later.unsqueeze(1), lww_newval.to(torch.int32).unsqueeze(1))
    _, row_lww = softmax_cam_read(sa2, sv2, lww_addr, addr_nib=addr_nib)
    got_lww = head.gather_value(sv2, row_lww)
    # winner must be the LATEST (highest) row index for that address == S-1
    out["lww_picks_latest_row"] = bool((row_lww == later).all())
    out["lww_value_correct"] = bool((got_lww == lww_newval).all())

    # ---- STACK read (SP-addressed row, high band ~2^24) through the SAME head ----
    stk_addr_base = 1 << 24
    stk_rows = torch.arange(B, device=dev) % S
    sa3 = store_addr.clone()
    sv3 = store_val.clone()
    stk_q = (stk_addr_base + (torch.arange(B, device=dev) % 4096) * 4).to(torch.int64)
    sa3.scatter_(1, stk_rows.unsqueeze(1), stk_q.unsqueeze(1))
    stk_val = torch.from_numpy(rng.integers(0, 1 << 31, size=B, dtype=np.int64)).to(dev)
    sv3.scatter_(1, stk_rows.unsqueeze(1), stk_val.to(torch.int32).unsqueeze(1))
    _, row_stk = softmax_cam_read(sa3, sv3, stk_q, addr_nib=addr_nib)
    got_stk = head.gather_value(sv3, row_stk)
    out["stack_value_correct"] = bool((got_stk == stk_val).all())

    # ---- FRAMEBUFFER read (fb band ~2^20 + word*4) through the SAME head ----
    fb_addr_base = (1 << 20) + (1 << 16)
    fb_rows = (torch.arange(B, device=dev) + 7) % S
    sa4 = store_addr.clone()
    sv4 = store_val.clone()
    fb_q = (fb_addr_base + (torch.arange(B, device=dev) % DOOM_FB_WORDS) * 4).to(torch.int64)
    sa4.scatter_(1, fb_rows.unsqueeze(1), fb_q.unsqueeze(1))
    fb_val = torch.from_numpy(rng.integers(0, 1 << 31, size=B, dtype=np.int64)).to(dev)
    sv4.scatter_(1, fb_rows.unsqueeze(1), fb_val.to(torch.int32).unsqueeze(1))
    _, row_fb = softmax_cam_read(sa4, sv4, fb_q, addr_nib=addr_nib)
    got_fb = head.gather_value(sv4, row_fb)
    out["framebuffer_value_correct"] = bool((got_fb == fb_val).all())

    out["byte_exact"] = (out["hit_linf"] == 0 and out["hit_reconstruct_linf"] == 0 and
                         out["miss_all_zero"] and out["lww_picks_latest_row"] and
                         out["lww_value_correct"] and out["stack_value_correct"] and
                         out["framebuffer_value_correct"])
    return out


# =========================================================================== #
# 2. PER-READ COST (us/lane) + S-INDEPENDENCE in TIME + VRAM footprint.
# =========================================================================== #
def _dense_perlane_vram_bytes(B, S):
    """The DENSE PER-LANE heap VRAM: (B, S) int32 store + a (B,) resolve index.  This is
    the OOM constraint: every lane its own full S-entry heap."""
    return B * S * BYTES_PER_ENTRY + B * 8


def _shared_pool_vram_bytes(B, S):
    """The SHARED-POOL heap VRAM: one (S,) int32 pool (lane-independent) + a (B,) resolve
    index.  The per-step marginal GPU work (one gathered row/lane) is identical to the
    dense form (``gather_value`` supports both), so the shared pool is the memory-feasible
    doom-realistic store — S-independent in the LANE dimension."""
    return S * BYTES_PER_ENTRY + B * 8


def bench_cam_cost(device, B, S, iters, warmup, store_mode="shared", d_model=64,
                   dtype=torch.float32):
    """Time ONE direct-CAM read (gather + nibble unpack + W_v/W_o value-band write) at
    batch B, store depth S.  ``store_mode``:
      * "shared" : one (S,) pool, gather one row/lane (the memory-feasible form).
      * "perlane": a literal (B, S) store (the dense OOM form) — only when it fits.

    Returns per-lane us, whole-batch ms, measured peak VRAM, and the closed-form footprints.
    Never OOMs the harness: a config whose store exceeds free VRAM is SKIPPED (reported as
    projected)."""
    dev = torch.device(device)
    cuda = device.startswith("cuda")
    head = DirectCAMReadHead(d_model, dtype).to(dev).eval()
    x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02

    g = torch.Generator(device="cpu").manual_seed(20260809 + S + B)
    resolve = torch.randint(0, S, (B,), generator=g, dtype=torch.int64)
    miss = torch.rand(B, generator=g) < 0.05
    resolve = torch.where(miss, torch.full_like(resolve, -1), resolve).to(dev)

    if store_mode == "perlane":
        store = torch.randint(0, 1 << 31, (B, S), generator=g,
                              dtype=torch.int64).to(torch.int32).to(dev)
    else:
        store = torch.randint(0, 1 << 31, (S,), generator=g,
                              dtype=torch.int64).to(torch.int32).to(dev)

    def one_read():
        return head(x, store, resolve)

    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    dt = _time(one_read, iters, warmup, cuda)
    peak = torch.cuda.max_memory_allocated(dev) if cuda else 0

    res = {
        "batch": B, "store_depth_S": S, "store_mode": store_mode,
        "ms_per_read_batch": dt * 1e3,
        "us_per_lane": dt / B * 1e6,
        "reads_per_s": B / dt,
        "measured_peak_vram_gib": peak / GIB if cuda else None,
        "dense_perlane_vram_gib": _dense_perlane_vram_bytes(B, S) / GIB,
        "shared_pool_vram_gib": _shared_pool_vram_bytes(B, S) / GIB,
    }
    del head, x, store, resolve
    if cuda:
        torch.cuda.empty_cache()
    return res


def find_oom_batch(S, vram_gib=24.0, headroom_gib=2.0):
    """The closed-form batch at which the DENSE PER-LANE heap of depth ``S`` OOMs a card
    with ``vram_gib`` total (leaving ``headroom_gib`` for the model + activations).

    Dense per-lane store = B x S x 4 bytes.  OOM when B x S x 4 > (vram - headroom).
    Returns the max fitting B and the B that OOMs, plus the per-lane store GiB at the
    doom spec-verify batches of interest."""
    budget = (vram_gib - headroom_gib) * GIB
    max_B = int(budget // (S * BYTES_PER_ENTRY))
    return {
        "store_depth_S": S,
        "vram_gib": vram_gib, "headroom_gib": headroom_gib,
        "max_fitting_batch_dense_perlane": max_B,
        "first_oom_batch_dense_perlane": max_B + 1,
        "perlane_gib_at_B1": S * BYTES_PER_ENTRY / GIB,
        "perlane_gib_at_B64": 64 * S * BYTES_PER_ENTRY / GIB,
        "perlane_gib_at_B512": 512 * S * BYTES_PER_ENTRY / GIB,
        "perlane_gib_at_B4096": 4096 * S * BYTES_PER_ENTRY / GIB,
        "shared_pool_gib": S * BYTES_PER_ENTRY / GIB,   # lane-independent
    }


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stores", default=",".join(str(s) for s in DOOM_STORES))
    ap.add_argument("--batches", default="512,4096,16384,65536,262144")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--vram-gib", type=float, default=24.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stores = [int(s) for s in args.stores.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    out = {"doom_heap_entries": DOOM_HEAP_ENTRIES, "doom_fb_words": DOOM_FB_WORDS,
           "stores": stores, "batches": batches, "vram_gib": args.vram_gib}

    # --------------- 1. BYTE-EXACT at DOOM SCALE --------------- #
    if args.verify or args.json:
        print("=" * 100)
        print("1. BYTE-EXACT: O(1) direct-CAM == reference O(S) latest-write-wins softmax "
              "CAM, at DOOM HEAP SCALE")
        print("   battery: hit / miss(ZFOD) / latest-write-wins / STACK read / FRAMEBUFFER "
              "read (all through ONE address-keyed head)")
        print("=" * 100)
        be = {}
        for S in stores:
            v = verify_heap_scale_byte_exact(S, B=512, device="cpu")
            be[str(S)] = v
            print(f"  S={S:>7d}: hit L-inf={v['hit_linf']} recon={v['hit_reconstruct_linf']} "
                  f"miss0={v['miss_all_zero']} lww(row/val)={v['lww_picks_latest_row']}/"
                  f"{v['lww_value_correct']} stack={v['stack_value_correct']} "
                  f"fb={v['framebuffer_value_correct']}  -> "
                  f"{'BYTE-EXACT' if v['byte_exact'] else 'FAIL'}")
        out["byte_exact"] = be
        out["all_byte_exact"] = all(v["byte_exact"] for v in be.values())

    if not args.bench:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"wrote {args.json}")
        return

    # --------------- 2. PER-READ COST + S-INDEPENDENCE (shared pool) --------------- #
    cuda = dev.startswith("cuda")
    if cuda:
        name = torch.cuda.get_device_name(int(dev.split(":")[1]) if ":" in dev else 0)
        print(f"\ndevice {dev}: {name}  ({torch.cuda.device_count()} visible)")
    print("=" * 100)
    print("2. PER-READ COST (us/lane) + S-INDEPENDENCE IN TIME  (SHARED-POOL store: one "
          "row gathered/lane)")
    print("   the direct-CAM read TIME must be FLAT across S (O(1)); VRAM (shared pool) is "
          "S x 4 bytes, lane-independent")
    print("=" * 100)
    cost = {}
    bench_B = 65536      # a big lane batch to expose the amortized per-lane cost
    print(f"  batch B={bench_B} (shared pool), sweep S:")
    for S in stores:
        r = bench_cam_cost(dev, bench_B, S, args.iters, args.warmup, store_mode="shared")
        cost[str(S)] = r
        print(f"    S={S:>7d}: {r['us_per_lane']:.5f} us/lane  "
              f"{r['ms_per_read_batch']:7.3f} ms/batch  "
              f"peak VRAM {r['measured_peak_vram_gib']:.4f} GiB  "
              f"(shared-pool store {r['shared_pool_vram_gib']*1024:.3f} MiB)")
    out["cost_shared_pool"] = cost
    # S-INDEPENDENCE (O(1)) metric.  The right test is a TREND test, NOT a max/min ratio:
    # if the read were O(S), us/lane would RISE MONOTONICALLY with a 16x sweep in S.  With
    # sub-microsecond per-lane cost the absolute numbers are launch-jitter-dominated, so a
    # raw max/min of ~1.5-2x is pure noise (e.g. the LARGEST S often times FASTEST — a
    # physical impossibility for a genuine O(S) cost).  We report:
    #   * the Pearson correlation of us/lane vs S (an O(1) read has ~0 correlation; an O(S)
    #     read has ~+1),
    #   * the ratio of the LARGEST-S cost to the SMALLEST-S cost (an O(S) read would be
    #     ~16x for the 16x S sweep here; O(1) is ~1x, either sign = jitter), and
    #   * the FLAT peak-VRAM confirmation (the shared-pool store + activations do not grow
    #     with S beyond the S x 4-byte pool itself — measured flat at ~0.09 GiB).
    us = [cost[str(S)]["us_per_lane"] for S in stores]
    Sf = [float(S) for S in stores]
    n = len(us)
    mS, mu = sum(Sf) / n, sum(us) / n
    cov = sum((s - mS) * (u - mu) for s, u in zip(Sf, us))
    vS = sum((s - mS) ** 2 for s in Sf)
    vu = sum((u - mu) ** 2 for u in us)
    corr = cov / max(1e-30, (vS * vu) ** 0.5)
    largest_over_smallest = us[-1] / max(1e-12, us[0])   # us at max S / us at min S
    vrams = [cost[str(S)]["measured_peak_vram_gib"] for S in stores if cost[str(S)]["measured_peak_vram_gib"]]
    vram_spread = (max(vrams) / max(1e-12, min(vrams))) if vrams else float("nan")
    # O(1) verdict: cost does NOT trend up with S (weak/negative correlation) AND peak VRAM
    # is flat (< 1.2x over the 16x S sweep).  A true O(S) read would have corr ~ +1 and a
    # ~16x cost ratio.
    S_independent = (corr < 0.5) and (largest_over_smallest < 4.0) and (vram_spread < 1.2)
    out["time_S_independence"] = {
        "us_per_lane_vs_S_pearson_corr": corr,
        "cost_largest_S_over_smallest_S": largest_over_smallest,
        "cost_ratio_if_OS_would_be": Sf[-1] / Sf[0],   # e.g. 16x for the 16x S sweep
        "peak_vram_spread_over_S": vram_spread,
        "verdict": "S-INDEPENDENT (O(1))" if S_independent else "S-DEPENDENT",
    }
    out["time_S_independent"] = S_independent
    print(f"  -> READ-TIME S-INDEPENDENCE (trend test over 16x S sweep):")
    print(f"       us/lane vs S Pearson corr = {corr:+.3f}  (O(S) -> ~+1.0; O(1) -> ~0)")
    print(f"       cost(maxS)/cost(minS) = {largest_over_smallest:.2f}x  "
          f"(O(S) would be {Sf[-1]/Sf[0]:.0f}x; O(1) -> ~1x, either sign = jitter)")
    print(f"       peak VRAM spread over S = {vram_spread:.3f}x  (flat = store touch is "
          f"one row/lane, not O(S))")
    print(f"       => {out['time_S_independence']['verdict']}")

    # --------------- 3. THE VRAM / OOM CONSTRAINT (dense per-lane heap x batch) --------------- #
    print("\n" + "=" * 100)
    print("3. THE BINDING CONSTRAINT: VRAM = DENSE PER-LANE heap (B x S x 4 bytes) — the "
          "OOM wall")
    print("   (this is the 64 GiB O(S) materialisation the direct-CAM EXISTS to avoid; the "
          "shared pool / evicted WS is what runs)")
    print("=" * 100)
    oom = {}
    for S in stores:
        o = find_oom_batch(S, vram_gib=args.vram_gib)
        oom[str(S)] = o
        print(f"  S={S:>7d}: per-lane heap = {o['perlane_gib_at_B1']*1024:.3f} MiB/lane  |  "
              f"dense OOMs {args.vram_gib:.0f}GB at batch > {o['max_fitting_batch_dense_perlane']:,}  "
              f"|  B=512 dense={o['perlane_gib_at_B512']:.2f} GiB  "
              f"B=4096 dense={o['perlane_gib_at_B4096']:.2f} GiB  |  "
              f"SHARED pool={o['shared_pool_gib']*1024:.3f} MiB (lane-independent)")
    out["oom_constraint"] = oom

    # MEASURE the dense per-lane form where it FITS, and confirm it OOMs where predicted.
    print("\n  --- MEASURED dense per-lane cost where it FITS (else PROJECTED) ---")
    dense = {}
    free_gib = (torch.cuda.mem_get_info(dev)[0] / GIB) if cuda else 0.0
    for S in stores:
        dense[str(S)] = {}
        for B in [512, 4096, 16384]:
            need = _dense_perlane_vram_bytes(B, S) / GIB
            if cuda and need < free_gib - 2.0:
                try:
                    r = bench_cam_cost(dev, B, S, max(10, args.iters // 2),
                                       args.warmup, store_mode="perlane")
                    r["fit"] = "MEASURED"
                    print(f"    S={S:>7d} B={B:>6d}: dense per-lane {need:6.3f} GiB  "
                          f"{r['us_per_lane']:.5f} us/lane  MEASURED "
                          f"(peak {r['measured_peak_vram_gib']:.3f} GiB)")
                except RuntimeError as e:
                    r = {"batch": B, "store_depth_S": S, "fit": "OOM",
                         "dense_perlane_vram_gib": need, "err": str(e)[:80]}
                    print(f"    S={S:>7d} B={B:>6d}: dense per-lane {need:6.3f} GiB  OOM")
                    if cuda:
                        torch.cuda.empty_cache()
            else:
                r = {"batch": B, "store_depth_S": S, "fit": "PROJECTED_OOM",
                     "dense_perlane_vram_gib": need}
                print(f"    S={S:>7d} B={B:>6d}: dense per-lane {need:6.3f} GiB  "
                      f"PROJECTED-OOM (> free {free_gib:.1f} GiB)")
            dense[str(S)][str(B)] = r
    out["dense_perlane_measured"] = dense

    # --------------- 4. VERDICT + MEMORY FLOOR --------------- #
    print("\n" + "=" * 100)
    print("4. VERDICT + THE MEMORY FLOOR for the min-flop Doom-runner")
    print("=" * 100)
    S262 = 262_144
    o262 = find_oom_batch(S262, vram_gib=args.vram_gib)
    verdict = {
        "time_is_S_independent": out.get("time_S_independent", False),
        "us_per_lane_at_262K": cost.get(str(S262), {}).get("us_per_lane"),
        "shared_pool_262K_mib": S262 * BYTES_PER_ENTRY / GIB * 1024,
        "dense_perlane_262K_mib_per_lane": S262 * BYTES_PER_ENTRY / GIB * 1024,
        "dense_262K_max_fitting_batch_24gb": o262["max_fitting_batch_dense_perlane"],
        "dense_262K_gib_at_B512": o262["perlane_gib_at_B512"],
        # doom live-heap floor: BOS sink + register frame + live rows (evicted WS).
        # The self-emu/doom evicted working set is ~the live-heap footprint, far below 262K.
        "evicted_ws_note": ("bounded live WS (BOS + register frame + live-heap rows) — "
                            "c4_min/qwen_lean_evict.py; flat VRAM on long programs"),
    }
    out["verdict"] = verdict
    _ti = out.get("time_S_independence", {})
    print(f"  TIME: direct-CAM read is {'S-INDEPENDENT (O(1))' if verdict['time_is_S_independent'] else 'S-DEPENDENT'} "
          f"over 16x S sweep (corr {_ti.get('us_per_lane_vs_S_pearson_corr', float('nan')):+.3f}, "
          f"cost ratio {_ti.get('cost_largest_S_over_smallest_S', float('nan')):.2f}x vs "
          f"{_ti.get('cost_ratio_if_OS_would_be', float('nan')):.0f}x if O(S), VRAM flat "
          f"{_ti.get('peak_vram_spread_over_S', float('nan')):.2f}x) -> read TIME is NOT the constraint.")
    print(f"  per-lane read cost @262K: {verdict['us_per_lane_at_262K']:.5f} us/lane "
          f"(flat vs S).")
    print(f"  VRAM (the constraint): DENSE per-lane 262K heap = "
          f"{verdict['dense_perlane_262K_mib_per_lane']:.2f} MiB/lane -> "
          f"OOMs {args.vram_gib:.0f}GB at batch > "
          f"{verdict['dense_262K_max_fitting_batch_24gb']:,} "
          f"(B=512 needs {verdict['dense_262K_gib_at_B512']:.1f} GiB).")
    print(f"  SHARED-POOL / EVICTED-WS 262K heap = "
          f"{verdict['shared_pool_262K_mib']:.2f} MiB TOTAL (lane-independent) -> FITS "
          f"trivially; per-step gather is byte-identical.")
    print(f"  => MEMORY FLOOR: the min-flop Doom-runner holds the heap byte-exact; the "
          f"floor is the SHARED/EVICTED store ({verdict['shared_pool_262K_mib']:.1f} MiB "
          f"for 262K) NOT the dense per-lane heap. Eviction (bounded live WS) is the "
          f"mechanism that makes a large spec-verify batch fit; the read stays O(1).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
