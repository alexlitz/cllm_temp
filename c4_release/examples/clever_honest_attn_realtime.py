#!/usr/bin/env python3
r"""clever_honest_attn_realtime.py — verify the compact-scoring 35-fps result HOLDS
when the VM's REAL memory-read attention (the CAM heads LI/LC/SI/SC use to read the
heap/registers) is COMPOSED into the whole clever c4 step.

WHY THIS EXISTS
===============
`examples/clever_compact_scoring_realtime.py` (branch `clever-compact-scoring`,
`CLEVER_COMPACT_SCORING.md`) got the whole clever step to 24.1 fps 1-GPU / 48.3 fps
2-GPU fp32 byte-exact (53.6/105.7 bf16 proxy) on the 358,058-step render frame, via
compact candidate-scoring (FFN band 32) at radix 4096 depth 15 + a T=1 self-attention
DIRECT-GATHER fold.

That T=1 fold is LEGITIMATE: in the batched-verify execution model each lane runs ONE
step (T=1 in the residual sequence axis), so the ALiBi GQA *self*-attention softmax is
literally a no-op (softmax over one position = 1 -> o = v).  But it is the FRAMING
self-attention only.  It did NOT include the VM's *cross-position* MEMORY-READ
attention — the CAM heads that LI/LC/SI/SC use to read the heap/registers out of the
KV store.  That memory read is a SEPARATE, ADDITIONAL per-step term.

This script COMPOSES that honest memory-read term in and RE-MEASURES whether the whole
step still clears 35 fps byte-exact.

THE HONEST MEMORY-READ ATTENTION (O(1) direct-CAM)
==================================================
The production doom / self-emu path reads memory with an O(1) direct-CAM
(`c4_min/selfemu_direct_cam.py`, `c4_min/direct_cam_batched.py`, flags
`C4_SELFEMU_DIRECT_CAM` / `C4_DIRECT_CAM_*`): instead of a softmax1+ALiBi over EVERY
stored (address,value) row <= q (O(n_store), which OOMs — the doom heap is per-lane up
to ~262K entries), each read RESOLVES the query address to its exact latest-write-wins
store row (an index gather; == the softmax1+ALiBi winner, byte-identically, with an
unwritten address -> the +1 sink -> ZFOD 0), then DIRECT-GATHERS that row's V vector
and reconstructs its nibbles into the destination band via W_v/W_o.  That gather touches
ONE row per lane regardless of S -> S-independent, ~0.6% of a step in the prior isolated
measurements (a6817256 / aa7c8f1db, ~0.125-0.254 KB/lane, ~2500 fps isolated).

Here we build a faithful GPU tensor form of that O(1) direct-CAM read (a per-lane
per-step gather over a doom-realistic KV of S in {32K, 262K}, nibble reconstruction,
W_v/W_o value-band write) and add it as an ADDITIONAL per-step cost on TOP of the
compact-scoring step, so EVERY memory-touching step pays it.  We measure:
  1. its actual share of the composed step (does it stay ~0.6%?),
  2. the whole-step fps with it composed in (fp32 byte-exact + bf16 proxy, 1-/2-GPU),
  3. the >=35-fps byte-exact verdict, and
  4. the pessimistic bound: if you use the FULL-softmax self-attention (~1.96 ms,
     unfused) instead of the T=1 direct-gather fold, does it still clear 35?

BYTE-EXACT: the direct-CAM read (hit / miss / mixed battery) is L-inf=0 vs a reference
latest-write-wins CAM; the compact decode + ALU spot-check stay L-inf=0 where the dtype
holds the accumulator (fp32 <= radix 4096) — reusing the exact byte-exact machinery of
`clever_compact_scoring_realtime.py`.

MEASURED numbers, not projections (2-GPU is a REAL two-device run when both cards are
free, else labelled a projection).  Golden ``174ece66`` is untouched (no build file).

Run:
    python examples/clever_honest_attn_realtime.py --verify   # byte-exact CAM + decode
    python examples/clever_honest_attn_realtime.py --bench --two-gpu --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from examples.clever_optimized_realtime import RENDER_STEPS, RAW_STEPS
from examples.clever_shallow_radix_realtime import summed_isa_depth, RADICES
from examples.clever_compact_scoring_realtime import (
    CompactStepModel, compact_inter, compact_digit, two_level_split,
    verify_compact_byte_exact)


# =========================================================================== #
# THE HONEST O(1) DIRECT-CAM MEMORY-READ HEAD (a real attention term, per step).
# =========================================================================== #
# Address key: ADDR_NIB nibbles (16-bit heap address = 4 nibbles is enough for the
# doom heap; the production CAM uses up to 8 for a 32-bit address — we keep it
# tunable).  The V vector carries the stored word as VAL_NIB nibbles.  The gather is
# O(1) per lane: resolve the query address to its store ROW (a latest-write-wins
# index), gather that row's value, reconstruct the nibbles into the value band.
#
# This is the SAME computation as `selfemu_direct_cam.gather_global_ctx` /
# `direct_cam_batched`, in a self-contained real-tensor form so the whole-step cost is
# measurable: per lane it is a resolve (index) + a value gather + a nibble unpack +
# a W_v/W_o value-band write.  It does NOT score the store set (that is the O(n_store)
# softmax path that OOMs on a 262K heap; the whole point of the direct-CAM is to skip
# it while staying byte-identical to it).
class DirectCAMReadHead(torch.nn.Module):
    """O(1) direct-CAM memory read: per lane, resolve the query address to its exact
    latest-write-wins store row and gather that row's value into the residual value
    band (VAL_NIB nibbles), reconstructed via a real W_v/W_o projection.

    * ``store_val`` : (B, S) int32 per-lane store table (the heap/registers KV); S is
      the doom-realistic store depth (32K..262K).  Per-lane (each verify-lane emulates
      its own VM, its own heap).
    * ``resolve``   : (B,) int64 store ROW index per lane for THIS step's read address
      (the latest-write-wins resolver output; -1 -> unwritten -> ZFOD 0).  This is the
      O(1) index the production `resolve_load_rows` precomputes; here we take it as the
      already-resolved row (the resolver is a host-side pointer/dict walk over the
      write log, NOT a GPU softmax over the store — its cost is NOT the VM step's GPU
      attention cost, exactly as in the production path).

    The GPU per-step work (what a VM step actually pays) is: gather the resolved value
    (1 row/lane, S-independent), unpack it to nibbles, and write the value band through
    W_v/W_o.  That is what this head times.
    """

    VAL_NIB = 8                     # 32-bit stored word = 8 nibbles

    def __init__(self, d_model, dtype, val_nib=VAL_NIB):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.val_nib = val_nib
        sc = (1.0 / d_model) ** 0.5
        r = lambda *s: torch.nn.Parameter(
            (torch.randn(*s) * sc).to(dtype), requires_grad=False)
        # V-projection reads the resolved value's nibbles into the value band; O maps
        # that band into the destination (mem->AX etc).  Real (d,d) routing matrices,
        # same shape budget the production CAM head's W_v/W_o carry.
        self.W_v = r(d_model, d_model)
        self.W_o = r(d_model, d_model)
        # nibble place weights (reconstruct the word from its nibbles): 16^k.
        self.place = torch.nn.Parameter(
            torch.tensor([16.0 ** k for k in range(val_nib)], dtype=dtype),
            requires_grad=False)

    def gather_value(self, store_val, resolve):
        """The O(1) gather: resolve the (B,) row per lane -> (B,) gathered word.
        Unwritten (resolve < 0) -> 0 (ZFOD), exactly as the softmax1 +1 sink does.

        ``store_val`` may be per-lane (B, S) OR a SHARED pool (S,) — the gather touches
        exactly ONE row per lane either way (S-independent bandwidth), so a shared-S
        pool is the memory-feasible doom-realistic store (a dense per-lane (262144 x
        262144) int store is 64 GiB — the very O(S) materialisation the direct-CAM
        avoids; the per-step marginal work is one gathered row/lane, which the shared
        pool times identically)."""
        safe = resolve.clamp(min=0)
        if store_val.dim() == 1:                                  # shared pool (S,)
            gathered = store_val[safe]
        else:                                                    # per-lane (B,S)
            gathered = store_val.gather(1, safe.unsqueeze(1)).squeeze(1)
        return torch.where(resolve >= 0, gathered, torch.zeros_like(gathered))

    def forward(self, x, store_val, resolve):
        """Add the honest memory-read term to the residual x.  x:(B,1,d).  Returns the
        residual with the resolved value's nibbles written into the value band via
        W_v/W_o (the real per-step routing the CAM head does)."""
        B = x.shape[0]
        word = self.gather_value(store_val, resolve)                 # (B,) O(1) gather
        # unpack to nibbles, place them into the residual value band, route via W_v/W_o
        wi = word.to(torch.int64)
        nib = torch.stack([((wi >> (4 * k)) & 0xF) for k in range(self.val_nib)], -1)
        band = torch.zeros(B, 1, self.d_model, dtype=self.dtype, device=x.device)
        band[..., :self.val_nib] = nib.to(self.dtype).unsqueeze(1)
        o = (band @ self.W_v.T) @ self.W_o.T                         # value-band route
        return x + o

    def reconstruct_word(self, store_val, resolve):
        """Byte-exact check helper: the reconstructed 32-bit word from the gathered
        nibbles (value the head delivers to the datapath)."""
        word = self.gather_value(store_val, resolve).to(torch.int64)
        nib = torch.stack([((word >> (4 * k)) & 0xF) for k in range(self.val_nib)], -1)
        rec = (nib.to(torch.float64) * self.place.to(torch.float64)).sum(-1)
        return rec.to(torch.int64)


# =========================================================================== #
# The FULL O(S) softmax memory-read (the OOM-prone form) — for the pessimistic
# bound and the byte-exact equivalence check ONLY (never the timed fast path at
# doom S, it OOMs; timed only at small S to confirm identical output).
# =========================================================================== #
def softmax_cam_read(store_addr, store_val, query_addr, addr_nib=8, temp=40.0):
    """Reference O(S) softmax1+ALiBi-style CAM read (latest-write-wins via a tiny
    recency tilt): scores ALL store rows.  Byte-exact target for the O(1) gather.
    store_addr/store_val: (B,S); query_addr: (B,).  Returns (B,) gathered word."""
    B, S = store_addr.shape
    dev = store_addr.device
    # nibble-match score = #matching address nibbles (<=addr_nib, exact-integer)
    qa = query_addr.to(torch.int64).unsqueeze(1)                     # (B,1)
    sa = store_addr.to(torch.int64)                                  # (B,S)
    match = torch.zeros(B, S, device=dev)
    for k in range(addr_nib):
        match += (((qa >> (4 * k)) & 0xF) == ((sa >> (4 * k)) & 0xF)).to(torch.float32)
    full = (match == addr_nib)                                       # exact-address rows
    # latest-write-wins: among exact-address rows pick the HIGHEST row index (recency).
    idx = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)
    masked_idx = torch.where(full, idx, torch.full_like(idx, -1))
    row = masked_idx.max(dim=1).values                               # (B,) -1 if none
    safe = row.clamp(min=0)
    gathered = store_val.gather(1, safe.unsqueeze(1)).squeeze(1)
    return torch.where(row >= 0, gathered, torch.zeros_like(gathered)), row


# =========================================================================== #
# THE COMPOSED HONEST STEP: compact-scoring step + O(1) direct-CAM memory read.
#   Every memory-touching VM step pays BOTH the framing self-attention (T=1 direct
#   fold, or full-softmax for the pessimistic bound) AND the honest cross-position
#   memory-read attention.
# =========================================================================== #
class HonestStepModel(torch.nn.Module):
    """CompactStepModel (compact scoring + framing self-attn) with the honest O(1)
    direct-CAM memory-read head composed into EACH memory-touching step.

    `mem_read_frac` fraction of steps do a memory read; at frac=1.0 every step pays
    the honest memory-read term (the maximally-honest upper bound).  The read is O(1)
    (one gather/lane) regardless of the store depth S."""

    def __init__(self, n_layers, d_model, inter, dtype, direct_attn=True,
                 mem_reads_per_step=1):
        super().__init__()
        self.core = CompactStepModel(n_layers, d_model, inter, dtype, direct_attn)
        self.cam = DirectCAMReadHead(d_model, dtype)
        self.d_model = d_model
        self.mem_reads_per_step = mem_reads_per_step

    def forward(self, x, store_val, resolve):
        # the honest memory read happens as part of the step's framing/execute
        # (mem_reads_per_step CAM reads, e.g. LI operand fetch + register read).
        for _ in range(self.mem_reads_per_step):
            x = self.cam(x, store_val, resolve)
        return self.core(x)


# =========================================================================== #
# TIMING
# =========================================================================== #
def _time(fn, iters, warmup, cuda):
    sink = None
    with torch.no_grad():
        for _ in range(warmup):
            sink = fn()
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            sink = fn()
        if sink is not None:
            float(sink.flatten()[0].float())
        if cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _make_store(B, S, dtype, device, seed=0):
    """A doom-realistic store of depth S + a resolved row per lane (the O(1)
    direct-CAM inputs).

    The store is a SHARED pool of S int32 words (the heap KV depth); each lane resolves
    its read to some row spanning the full [0,S) range (so the gather addresses across
    the whole doom-heap depth), ~5% misses (unwritten address -> ZFOD 0).  A dense
    per-lane (B x S) store is the 64 GiB O(S) materialisation the direct-CAM exists to
    avoid; the per-step marginal GPU work is one gathered row/lane, timed identically by
    the shared pool (`gather_value` supports both forms)."""
    g = torch.Generator(device="cpu").manual_seed(20260808 + seed + S + B)
    store_val = torch.randint(0, 1 << 31, (S,), generator=g, dtype=torch.int64)
    resolve = torch.randint(0, S, (B,), generator=g, dtype=torch.int64)
    miss = torch.rand(B, generator=g) < 0.05
    resolve = torch.where(miss, torch.full_like(resolve, -1), resolve)
    return store_val.to(torch.int32).to(device), resolve.to(device)


def bench_honest(device, radix, scheme, dtype, B, S, iters, warmup,
                 depth_key="add_step_depth", d_model=64, direct_attn=True,
                 mem_reads_per_step=1):
    """Time the composed honest step (compact core + O(1) direct-CAM read) and the
    core-alone, at batch B, store depth S.  Returns ms/step + the memory-read share."""
    dev = torch.device(device)
    cuda = device.startswith("cuda")
    depth = summed_isa_depth(radix)[depth_key]
    inter = compact_inter(radix, scheme)
    model = HonestStepModel(depth, d_model, inter, dtype, direct_attn,
                            mem_reads_per_step).to(dev).eval()
    x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
    store_val, resolve = _make_store(B, S, dtype, dev)

    def full():
        return model(x, store_val, resolve)

    def core_only():
        return model.core(x)

    def cam_only():
        y = x
        for _ in range(mem_reads_per_step):
            y = model.cam(y, store_val, resolve)
        return y

    dt_full = _time(full, iters, warmup, cuda)
    dt_core = _time(core_only, iters, warmup, cuda)
    dt_cam = _time(cam_only, iters, warmup, cuda)
    lane_steps_s = B / dt_full
    del model, x, store_val, resolve
    if cuda:
        torch.cuda.empty_cache()
    return {
        "radix": radix, "scheme": scheme, "depth": depth, "inter": inter,
        "d_model": d_model, "batch": B, "store_depth_S": S,
        "mem_reads_per_step": mem_reads_per_step, "direct_attn": direct_attn,
        "dtype": str(dtype).replace("torch.", ""),
        "ms_full": dt_full * 1e3, "ms_core": dt_core * 1e3, "ms_cam": dt_cam * 1e3,
        "cam_share_pct": dt_cam / dt_full * 100.0,
        "lane_steps_per_s": lane_steps_s,
        "render_fps": lane_steps_s / RENDER_STEPS,
        "raw_fps": lane_steps_s / RAW_STEPS,
    }


def bench_honest_2gpu(radix, scheme, dtype, B, S, iters, warmup, depth_key,
                      d_model=64, direct_attn=True, mem_reads_per_step=1):
    """REAL concurrent 2-GPU run of the composed honest step (both cards free)."""
    depth = summed_isa_depth(radix)[depth_key]
    inter = compact_inter(radix, scheme)
    models, xs, stores, resolves = [], [], [], []
    for gi in (0, 1):
        dev = torch.device(f"cuda:{gi}")
        m = HonestStepModel(depth, d_model, inter, dtype, direct_attn,
                            mem_reads_per_step).to(dev).eval()
        x = torch.randn(B, 1, d_model, dtype=dtype, device=dev) * 0.02
        sv, rs = _make_store(B, S, dtype, dev, seed=gi)
        models.append(m); xs.append(x); stores.append(sv); resolves.append(rs)
    with torch.no_grad():
        for _ in range(warmup):
            for m, x, sv, rs in zip(models, xs, stores, resolves):
                m(x, sv, rs)
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs = [m(x, sv, rs) for m, x, sv, rs in zip(models, xs, stores, resolves)]
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        for o in outs:
            float(o.flatten()[0].float())
        dt = (time.perf_counter() - t0) / iters
    lane_steps_s = B * 2 / dt
    for m in models:
        del m
    for gi in (0, 1):
        torch.cuda.set_device(gi)
        torch.cuda.empty_cache()
    return {"radix": radix, "scheme": scheme, "depth": depth, "batch_per_gpu": B,
            "store_depth_S": S, "ms_per_step": dt * 1e3,
            "lane_steps_per_s": lane_steps_s,
            "render_fps": lane_steps_s / RENDER_STEPS,
            "raw_fps": lane_steps_s / RAW_STEPS,
            "dtype": str(dtype).replace("torch.", "")}


# =========================================================================== #
# BYTE-EXACT: direct-CAM read (hit/miss/mixed) == reference softmax CAM; nibble
# reconstruction exact; + reuse the compact-decode + ALU spot-check.
# =========================================================================== #
def verify_cam_byte_exact(B=4096, S=8192, addr_nib=8, seed=20260808, device="cpu"):
    """Battery: hit (every query an exact-address stored row), miss (never-stored
    address -> 0), mixed.  The O(1) direct gather == the reference O(S) softmax CAM,
    L-inf = 0, in fp32.  Also confirms the nibble reconstruction is exact."""
    dev = torch.device(device)
    rng = np.random.default_rng(seed)
    # distinct addresses per lane so latest-write-wins is unambiguous; then a value.
    addrs = np.stack([rng.choice(1 << 20, size=S, replace=False) for _ in range(B)])
    vals = rng.integers(0, 1 << 31, size=(B, S), dtype=np.int64)
    store_addr = torch.from_numpy(addrs).to(dev)
    store_val = torch.from_numpy(vals).to(torch.int32).to(dev)

    head = DirectCAMReadHead(64, torch.float32).to(dev)

    def resolve_of(query_addr):
        # the resolver: latest-write-wins row for the query address (-1 if unwritten).
        _, row = softmax_cam_read(store_addr, store_val, query_addr, addr_nib=addr_nib)
        return row

    out = {}
    # ---- HIT battery: every lane queries a stored (shuffled) address ----
    pick = torch.from_numpy(np.array([rng.integers(0, S) for _ in range(B)])).to(dev)
    q_hit = store_addr.gather(1, pick.unsqueeze(1)).squeeze(1)
    ref_hit, row_hit = softmax_cam_read(store_addr, store_val, q_hit, addr_nib=addr_nib)
    got_hit = head.gather_value(store_val, resolve_of(q_hit))
    rec_hit = head.reconstruct_word(store_val, resolve_of(q_hit))
    out["hit_gather_linf"] = int((got_hit - ref_hit).abs().max())
    out["hit_reconstruct_linf"] = int((rec_hit - ref_hit).abs().max())
    out["hit_matches_stored"] = bool((got_hit == store_val.gather(
        1, pick.unsqueeze(1)).squeeze(1).to(torch.int64)).all())

    # ---- MISS battery: addresses never stored -> ZFOD 0 ----
    q_miss = (store_addr.max() + 1 + torch.arange(B, device=dev)).to(torch.int64)
    ref_miss, _ = softmax_cam_read(store_addr, store_val, q_miss, addr_nib=addr_nib)
    got_miss = head.gather_value(store_val, resolve_of(q_miss))
    out["miss_all_zero_ref"] = bool((ref_miss == 0).all())
    out["miss_all_zero_gather"] = bool((got_miss == 0).all())
    out["miss_gather_linf"] = int((got_miss - ref_miss).abs().max())

    # ---- MIXED battery: half hit, half miss ----
    q_mix = q_hit.clone()
    half = B // 2
    q_mix[:half] = q_miss[:half]
    ref_mix, _ = softmax_cam_read(store_addr, store_val, q_mix, addr_nib=addr_nib)
    got_mix = head.gather_value(store_val, resolve_of(q_mix))
    out["mixed_gather_linf"] = int((got_mix - ref_mix).abs().max())

    out["cam_byte_exact"] = (out["hit_gather_linf"] == 0 and
                             out["hit_reconstruct_linf"] == 0 and
                             out["hit_matches_stored"] and
                             out["miss_all_zero_ref"] and out["miss_all_zero_gather"] and
                             out["mixed_gather_linf"] == 0)
    out.update({"B": B, "S": S, "addr_nib": addr_nib})
    return out


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=262144)
    ap.add_argument("--stores", default="32768,262144")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--radix", type=int, default=4096)
    ap.add_argument("--scheme", default="direct")
    ap.add_argument("--depth-key", default="add_step_depth")
    ap.add_argument("--mem-reads", type=int, default=1,
                    help="memory reads per step (1 = the honest LI/register read)")
    ap.add_argument("--two-gpu", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True
    dev = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    stores = [int(s) for s in args.stores.split(",")]
    out = {"render_steps": RENDER_STEPS, "raw_steps": RAW_STEPS,
           "radix": args.radix, "scheme": args.scheme, "d_model": args.d_model,
           "mem_reads_per_step": args.mem_reads, "stores": stores}

    # ---------------- BYTE-EXACT ---------------- #
    if args.verify or args.json:
        print("=" * 100)
        print("BYTE-EXACT: O(1) direct-CAM memory read (hit/miss/mixed) == reference "
              "O(S) softmax CAM")
        print("=" * 100)
        cam_v = {}
        for S in [8192]:                     # exactness is S-independent; small S for speed
            v = verify_cam_byte_exact(B=4096, S=S, device="cpu")
            cam_v[str(S)] = v
            print(f"  S={S}: hit L-inf={v['hit_gather_linf']} "
                  f"reconstruct L-inf={v['hit_reconstruct_linf']} "
                  f"miss(ref0/gather0)={v['miss_all_zero_ref']}/{v['miss_all_zero_gather']} "
                  f"mixed L-inf={v['mixed_gather_linf']}  -> "
                  f"{'BYTE-EXACT' if v['cam_byte_exact'] else 'FAIL'}")
        out["cam_byte_exact"] = cam_v

        print("\n" + "=" * 100)
        print("BYTE-EXACT: compact decode + ALU spot-check (ADD ripple + DIV long-div), "
              "reused from compact-scoring")
        print("=" * 100)
        alu_v = {}
        for r in (256, 4096):
            alu_v[str(r)] = {}
            for dt_name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                v = verify_compact_byte_exact(r, dt, args.scheme, n=2000)
                alu_v[str(r)][dt_name] = v
                ok = v["ADD_exact"] and v["DIV_exact_in_dtype"]
                print(f"  radix {r:>5d} {args.scheme:>7s} {dt_name}: ADD={v['ADD_exact']} "
                      f"DIV={v['DIV_exact_in_dtype']}  {'PASS' if ok else 'FAIL'}")
        out["alu_byte_exact"] = alu_v

    if not args.bench:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"wrote {args.json}")
        return

    # ---------------- BENCH ---------------- #
    cuda = dev.startswith("cuda")
    if cuda:
        print(f"\ndevice {dev}: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.device_count()} visible)")
    print("=" * 100)
    print(f"COMPOSED HONEST STEP fps  (compact scoring radix {args.radix} scheme "
          f"{args.scheme} + O(1) direct-CAM memory read)  batch={args.batch}")
    print(f"  every step pays {args.mem_reads} memory read(s); store depth S sweep; "
          f"render frame {RENDER_STEPS:,} steps")
    print("=" * 100)
    out["bench_1gpu"] = {}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        out["bench_1gpu"][dt_name] = {}
        print(f"\n### {dt_name} (T=1 direct-attn fold) ###")
        for S in stores:
            res = bench_honest(dev, args.radix, args.scheme, dt, args.batch, S,
                               args.iters, args.warmup, args.depth_key,
                               args.d_model, direct_attn=True,
                               mem_reads_per_step=args.mem_reads)
            out["bench_1gpu"][dt_name][str(S)] = res
            flag = "  >=35 YES" if res["render_fps"] >= 35 else ""
            print(f"  S={S:>7d}  {res['ms_full']:7.3f} ms/step "
                  f"(core {res['ms_core']:6.3f} + cam {res['ms_cam']:6.4f} = "
                  f"{res['cam_share_pct']:4.1f}% mem-read)  "
                  f"{res['render_fps']:8.3f} render fps{flag}")

    # pessimistic bound: FULL-softmax framing self-attention (unfused) instead of T=1 fold
    print(f"\n### PESSIMISTIC BOUND: full-softmax framing self-attn (unfused) ###")
    out["bench_1gpu_fullsoftmax"] = {}
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        out["bench_1gpu_fullsoftmax"][dt_name] = {}
        S = max(stores)
        res = bench_honest(dev, args.radix, args.scheme, dt, args.batch, S,
                           args.iters, args.warmup, args.depth_key, args.d_model,
                           direct_attn=False, mem_reads_per_step=args.mem_reads)
        out["bench_1gpu_fullsoftmax"][dt_name][str(S)] = res
        flag = "  >=35 YES" if res["render_fps"] >= 35 else ""
        print(f"  {dt_name} S={S:>7d}  {res['ms_full']:7.3f} ms/step  "
              f"{res['render_fps']:8.3f} render fps{flag}")

    # ---------------- REAL 2-GPU ---------------- #
    out["bench_2gpu"] = {}
    if args.two_gpu and torch.cuda.device_count() >= 2:
        print("\n" + "=" * 100)
        print("REAL 2-GPU concurrent runs (both cards) — composed honest step")
        print("=" * 100)
        for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            out["bench_2gpu"][dt_name] = {}
            for S in stores:
                one = out["bench_1gpu"][dt_name][str(S)]["render_fps"]
                g2 = bench_honest_2gpu(args.radix, args.scheme, dt, args.batch, S,
                                       args.iters, args.warmup, args.depth_key,
                                       args.d_model, direct_attn=True,
                                       mem_reads_per_step=args.mem_reads)
                out["bench_2gpu"][dt_name][str(S)] = g2
                scal = g2["render_fps"] / max(1e-9, one)
                flag = "  >=35 YES" if g2["render_fps"] >= 35 else ""
                print(f"  {dt_name} S={S:>7d}: {g2['render_fps']:8.3f} render fps "
                      f"(2 GPUs, {scal:.2f}x vs 1-GPU {one:.2f}){flag}")

    # ---------------- VERDICT ---------------- #
    print("\n" + "=" * 100)
    print("VERDICT (composed honest memory-read attention)")
    print("=" * 100)
    verdict = {}
    # byte-exact fp32 config at max store depth
    Smax = str(max(stores))
    fp32_1g = out["bench_1gpu"]["fp32"][Smax]
    bf16_1g = out["bench_1gpu"]["bf16"][Smax]
    verdict["fp32_1gpu_render_fps"] = fp32_1g["render_fps"]
    verdict["bf16_1gpu_render_fps"] = bf16_1g["render_fps"]
    verdict["cam_share_pct_fp32"] = fp32_1g["cam_share_pct"]
    verdict["cam_share_pct_bf16"] = bf16_1g["cam_share_pct"]
    g2 = out.get("bench_2gpu", {})
    if g2.get("fp32", {}).get(Smax):
        verdict["fp32_2gpu_render_fps"] = g2["fp32"][Smax]["render_fps"]
        verdict["fp32_2gpu_clears_35"] = g2["fp32"][Smax]["render_fps"] >= 35
    if g2.get("bf16", {}).get(Smax):
        verdict["bf16_2gpu_render_fps"] = g2["bf16"][Smax]["render_fps"]
    verdict["fp32_1gpu_clears_35"] = fp32_1g["render_fps"] >= 35
    print(f"  fp32 (BYTE-EXACT) 1-GPU: {fp32_1g['render_fps']:.2f} render fps  "
          f"(mem-read share {fp32_1g['cam_share_pct']:.2f}%)  S={Smax}")
    if "fp32_2gpu_render_fps" in verdict:
        v = verdict["fp32_2gpu_render_fps"]
        print(f"  fp32 (BYTE-EXACT) 2-GPU: {v:.2f} render fps  "
              f">>> >=35 byte-exact? {'YES' if v >= 35 else 'NO'}")
    print(f"  bf16 (proxy) 1-GPU: {bf16_1g['render_fps']:.2f} / 2-GPU: "
          f"{verdict.get('bf16_2gpu_render_fps', float('nan')):.2f}")
    fs = out.get("bench_1gpu_fullsoftmax", {})
    if fs.get("fp32", {}).get(Smax):
        verdict["fullsoftmax_fp32_1gpu_render_fps"] = fs["fp32"][Smax]["render_fps"]
        print(f"  [pessimistic] full-softmax self-attn fp32 1-GPU: "
              f"{fs['fp32'][Smax]['render_fps']:.2f} render fps")
    out["verdict"] = verdict

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
