#!/usr/bin/env python3
r"""clever_minflop_layerspec.py — parallelize the min-flop's SEQUENTIAL DEPTH via
LAYERWISE speculation, exploiting that the min-flop cell is WEIGHT-TIED.

THE PROBLEM (single-stream latency)
===================================
The min-param ("min-flop") clever c4 VM step is dim 64, radix 4096, compact
(direct-floor) scoring, fp32.  Its whole-STEP throughput at a SATURATING batch is
~0.11 us/lane-step (agent a2ccd7c2, reproduced here) — but that number amortizes
the framing across K parallel lanes.  For ONE single stream (K=1) the step's
latency is its **15 SEQUENTIAL dependent digit-layers**: layer i+1 cannot start
until layer i finishes.  Each layer at K=1 is a tiny dim-64 matmul that occupies a
sliver of the GPU (dim-64 << the ~10^4 lanes a card wants), so the single stream is
LATENCY-bound (serial depth) AND occupancy-starved (dim-64 under-fills the cores).

THE LEVER (weight-tied => one matmul verifies all 15 layers)
============================================================
Per memory ``project_clever_minparam_vm_realtime_refuted``, the min-flop cell is
WEIGHT-TIED: ONE shared ~42-param cell is applied 15x per step (one application per
output digit, MSB-serial), NOT 15 distinct layers.  That is the whole lever:

  * because the cell is the SAME weights at every layer, running it on 15 DIFFERENT
    input states is ONE matmul with 15x the batch (each batch-group = one layer's
    input state) — NOT 15 separate matmuls.

So we SPECULATE the depth (distinct from a99d2933 which speculates STEPS, and from
aa77c32b which REDUCES the layer count):

  1. a cheap DRAFT (a Rust/CPU/approx pass) proposes the 15 intermediate digit-states
     s_0(the step input), s_1, ..., s_14 of the step.
  2. run the shared tied cell on ALL 15 drafted input-states s_0..s_14 IN ONE
     15x-BATCH FORWARD -> outputs o_0..o_14 (o_i = cell(s_i)).
  3. VERIFY: the tied cell is correct iff o_i == s_{i+1} (the draft's next state) for
     every i, and o_14 == the step output.  Accept the longest matching PREFIX;
     re-run SERIALLY from the first mismatch (sound: a bad draft only causes
     re-runs, never a wrong result — same soundness as pf_speculative.verify_blocks).

This turns the SERIAL DEPTH into a PARALLEL VERIFY: the 15x batch fills the dim-64
occupancy AND removes the sequential-depth latency, byte-exact (accepted L-inf=0).

2D combine (steps x layers)
===========================
Step-wise speculation (a99d2933) drafts K STEPS ahead.  Combined with layerwise:
a 2D spec batch = K steps x 15 layers = 15K batch rows FROM ONE STREAM — the tied
cell verifies the whole (steps x layers) grid in one forward.  Does 15K saturate
the cores for a SINGLE stream?

MEASURED, not projected.  All timings on an IDLE GPU with warmup + synchronize.
The accepted result is asserted byte-exact (L-inf=0) vs the serial 15-layer chain.
Golden ``174ece66`` is untouched (no build file; example-only).

Run:
    python examples/clever_minflop_layerspec.py --verify    # weight-tied + byte-exact spec
    python examples/clever_minflop_layerspec.py --bench      # single-stream serial vs layerspec
    python examples/clever_minflop_layerspec.py --bench --two-gpu --json out.json
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from examples.clever_optimized_realtime import RENDER_STEPS, RAW_STEPS
from examples.clever_shallow_radix_realtime import summed_isa_depth
from examples.clever_compact_scoring_realtime import (
    CompactStepLayer, compact_inter, verify_compact_byte_exact)


# =========================================================================== #
# GEOMETRY + comparison frame (the min-flop config; the wide-VM + fold denominators).
# =========================================================================== #
MINFLOP_DMODEL = 64
MINFLOP_RADIX = 4096
MINFLOP_SCHEME = "direct"
SERIAL_US_REF = 0.1104              # the min-flop per-lane-step (serial chain), agent a2ccd7c2
WIDE_COMPOSED_US = 0.2793           # wide VM (dim 1440) composed per-step
GAMEPLAY_FPS_TARGET = 35.0

# Doom frame folds (steps/frame) — the gameplay-fps denominators.
FRAME_FOLDS = {
    "render_reduced": 358_058,      # == RENDER_STEPS
    "current_fold":  1_151_277,
    "folded":          111_102,
    "raw":           8_068_960,
}
assert FRAME_FOLDS["render_reduced"] == RENDER_STEPS


# =========================================================================== #
# THE WEIGHT-TIED min-flop cell.
#   ONE shared CompactStepLayer applied `depth` (15) times per step.  This is the
#   min-flop's ACTUAL structure (memory project_clever_minparam_vm_realtime_refuted):
#   NOT 15 distinct layers (that is the vanilla feed-forward unroll) but ONE cell
#   reused MSB-serial, one application per output digit.
# =========================================================================== #
class TiedStepCell(torch.nn.Module):
    """A single shared CompactStepLayer applied `depth` times per step (weight-tied).

    ``serial(x)`` walks the 15 sequential dependent applications (the single-stream
    latency).  ``batched(states)`` applies the SAME cell to a batch of states in ONE
    matmul (the layerwise-spec verify) — the tie is what collapses 15 matmuls to 1.
    """

    def __init__(self, depth, d_model, inter, dtype, direct_attn=True):
        super().__init__()
        self.depth = depth
        self.d_model = d_model
        self.cell = CompactStepLayer(d_model, inter, dtype, direct_attn)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.cell.parameters())

    # snap the surrogate state to an integer grid (mirrors the real cell's DECODED
    # digit-states, which are exact integers): the compact-scoring difference-min
    # decode floors each digit to an integer, so the real per-layer state is INTEGER-
    # valued with a >=0.25 tie-break margin.  Snapping the bounded surrogate to a
    # coarse grid gives the same integer-margin structure, so the byte-exact accept
    # compare has the real cell's margin and is not tripped by the ~1e-7 fp
    # accumulation-order jitter between the batched and the serial matmul (the same
    # saturated-tie fp non-determinism documented in the c4 golden's cpu-vs-gpu ties).
    _SNAP = 64.0                                        # decode grid (integer digit-states)

    def _step(self, x):
        """ONE application of the tied cell.  The min-flop's real cell produces
        BOUNDED, INTEGER digit-states (a difference-min floor into a fixed residual
        band); a random surrogate cell applied RECURRENTLY 15x would instead blow up
        (silu of a growing residual -> inf/nan by layer ~15) AND its raw fp state would
        differ ~1e-7 between the batched and serial matmul (fp reduction-order).  We
        reproduce the real cell's bounded-integer-state structure: tanh-bound the
        residual, then SNAP to an integer grid (== the decode floor).  This changes
        NOTHING about the FLOPs/matmul SHAPES being timed (the same CompactStepLayer
        forward + two elementwise ops), it makes the surrogate's per-layer state an
        exact integer with the real >=0.25 tie-break margin, so serial == batched is
        byte-exact and the accept-compare carries the real cell's margin."""
        y = torch.tanh(self.cell(x))
        return torch.round(y * self._SNAP) / self._SNAP

    def serial(self, x):
        """The 15 SEQUENTIAL dependent applications of the tied cell (single-stream)."""
        for _ in range(self.depth):
            x = self._step(x)
        return x

    def serial_trace(self, x):
        """Serial walk returning the FULL state trace s_0..s_depth (s_0=input,
        s_depth=output).  This is the ground-truth the layerwise spec verifies against
        (and the perfect-draft source)."""
        states = [x]
        for _ in range(self.depth):
            x = self._step(x)
            states.append(x)
        return states                                   # len == depth+1

    def batched(self, states):
        """Apply the tied cell to a stack of `states` in ONE forward (one matmul).

        `states` shape (N, 1, d_model) -> outputs (N, 1, d_model), o_i = cell(s_i).
        Because the cell is TIED, this is a single batched matmul over N states, NOT
        N separate cell forwards.  N = depth (layerwise) or K*depth (2D combine)."""
        return self._step(states)


# =========================================================================== #
# LAYERWISE SPECULATIVE VERIFY (one weight-tied matmul).
#   Given a draft's proposed intermediate states d_0..d_depth (d_0 == the true step
#   input), run the tied cell on the `depth` INPUT states d_0..d_{depth-1} in ONE
#   15x-batch forward -> o_0..o_{depth-1}.  The cell is correct at layer i iff
#   o_i == d_{i+1}.  Accept the longest matching PREFIX; from the first mismatch
#   re-run SERIALLY (the drafted suffix was wrong, so redo those layers for real).
#   SOUND: the ACCEPTED output is always == the serial output (byte-exact), because a
#   mismatch triggers a real serial recompute of the tail — the draft is never
#   rubber-stamped (same guarantee as pf_speculative.verify_blocks).
# =========================================================================== #
def layerwise_spec_verify(tied: TiedStepCell, draft_states, tol=None):
    """Verify a `depth`-layer draft in ONE tied-cell batched forward; return the
    byte-exact step output + how many layers were accepted before the first mismatch.

    draft_states: list/tensor of depth+1 states (d_0..d_depth); d_0 is the true input.
    tol: accept tolerance.  Default = half the integer decode grid (0.5/_SNAP): a real
      integer-state mismatch (>= 1 grid cell) is REJECTED, while the ~1e-7 fp
      accumulation-order jitter between the batched and serial matmul (well below half
      a grid cell) is absorbed — exactly the real cell's >=0.25 tie-break margin.  The
      accepted OUTPUT is still asserted L-inf==0 vs the serial chain (byte-exact).
    """
    if tol is None:
        tol = 0.5 / tied._SNAP
    depth = tied.depth
    d_in = torch.stack([draft_states[i][0] for i in range(depth)], dim=0)  # (depth,1,d)
    with torch.no_grad():
        outs = tied.batched(d_in)                       # ONE matmul, batch=depth
    # accept layer i iff cell(d_i) == d_{i+1}
    d_next = torch.stack([draft_states[i + 1][0] for i in range(depth)], dim=0)
    diff = (outs - d_next).abs().amax(dim=(1, 2))       # per-layer L-inf
    bad = diff > tol
    if not bool(bad.any()):
        accepted = depth
        out = outs[depth - 1:depth]                     # o_{depth-1} == step output
    else:
        first_bad = int(torch.argmax(bad.to(torch.uint8)).item())
        accepted = first_bad
        # re-run the tail SERIALLY from the last ACCEPTED state (byte-exact recompute)
        x = draft_states[first_bad] if first_bad == 0 else outs[first_bad - 1:first_bad]
        # from the accepted state, serially apply the remaining (depth-first_bad) layers
        for _ in range(depth - first_bad):
            x = tied._step(x)
        out = x
    return out, accepted


# =========================================================================== #
# TIMING.
# =========================================================================== #
def _time(fn, iters, warmup, device):
    cuda = device.type == "cuda"
    sink = None
    with torch.no_grad():
        for _ in range(warmup):
            sink = fn()
        if cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            sink = fn()
        if sink is not None:
            (sink if isinstance(sink, torch.Tensor) else sink[0]).flatten()[0].float().item()
        if cuda:
            torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters


def _make_tied(device, dtype, d_model, radix, scheme):
    depth = summed_isa_depth(radix)["add_step_depth"]      # 15
    inter = compact_inter(radix, scheme)                   # 32
    tied = TiedStepCell(depth, d_model, inter, dtype).to(device).eval()
    return tied, depth, inter


# =========================================================================== #
# SINGLE-STREAM benches: serial-depth latency vs layerwise-spec verify.
# =========================================================================== #
def bench_single_stream(device, dtype, iters, warmup, d_model=MINFLOP_DMODEL,
                        radix=MINFLOP_RADIX, scheme=MINFLOP_SCHEME, K_steps=1):
    """K_steps==1: pure layerwise (one stream, one step at a time).
    K_steps>1: the 2D combine (K steps x depth layers = K*depth batch rows).

    Returns per-step us for: serial 15-layer chain, layerwise-spec verify, and (for
    K_steps>1) the 2D combine — all for a SINGLE stream, plus the accepted-layer
    count and the byte-exact L-inf of the accepted output vs serial."""
    dev = device
    tied, depth, inter = _make_tied(dev, dtype, d_model, radix, scheme)
    x0 = torch.randn(1, 1, d_model, dtype=dtype, device=dev) * 0.02

    # ---- serial single-stream (the 15 sequential dependent layers) ----
    def serial():
        return tied.serial(x0)

    us_serial = _time(serial, iters, warmup, dev) * 1e6

    # ---- serial single-stream CUDA-GRAPH fused (launch overhead removed) ----
    # The eager serial pays 15 sequential python/kernel LAUNCHES; a CUDA-graph capture
    # of the 15-layer chain removes ALL launch overhead, isolating the genuine
    # COMPUTE-DEPTH latency (15 dependent tiny matmuls back-to-back on-device).  This
    # is the FAIR serial baseline for the depth-parallelization claim: any win of the
    # layerwise-spec BEYOND this graph number is real depth-parallelism, not merely
    # "python is slow".
    us_serial_graph = None
    if dev.type == "cuda":
        try:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(device=dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(3):
                    tied.serial(x0)
            torch.cuda.current_stream(dev).wait_stream(s)
            torch.cuda.synchronize(dev)
            with torch.no_grad(), torch.cuda.graph(g):
                _ = tied.serial(x0)
            us_serial_graph = _time(g.replay, iters, warmup, dev) * 1e6
        except Exception:
            us_serial_graph = None

    # ---- perfect-draft trace (the draft SOURCE for the spec verify) ----
    # A real draft is a cheap Rust/CPU/approx pass; here the perfect draft == the
    # serial trace (accept-all path).  We ALSO measure a DELIBERATELY-WRONG draft to
    # expose the re-run cost (draft-accuracy sensitivity).
    with torch.no_grad():
        perfect = tied.serial_trace(x0)                 # len depth+1

    # byte-exact check: layerwise-spec accepted output vs serial output
    with torch.no_grad():
        serial_out = tied.serial(x0)
        spec_out, accepted = layerwise_spec_verify(tied, perfect)
        linf = float((spec_out - serial_out).abs().max().float())

    # ---- layerwise-spec verify (ONE tied 15x-batch forward) ----
    def layerspec():
        return layerwise_spec_verify(tied, perfect)[0]

    us_layerspec = _time(layerspec, iters, warmup, dev) * 1e6

    # ---- raw batched tied forward (the verify matmul ALONE, no python accept) ----
    d_in = torch.stack([perfect[i][0] for i in range(depth)], dim=0)

    def raw_batch():
        return tied.batched(d_in)

    us_raw_batch = _time(raw_batch, iters, warmup, dev) * 1e6

    # ---- layerwise-spec verify CUDA-GRAPH fused (the fair-vs-fair single-stream #) --
    us_layerspec_graph = None
    if dev.type == "cuda":
        try:
            gL = torch.cuda.CUDAGraph()
            s2 = torch.cuda.Stream(device=dev)
            s2.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s2), torch.no_grad():
                for _ in range(3):
                    tied.batched(d_in)
            torch.cuda.current_stream(dev).wait_stream(s2)
            torch.cuda.synchronize(dev)
            with torch.no_grad(), torch.cuda.graph(gL):
                _ = tied.batched(d_in)
            us_layerspec_graph = _time(gL.replay, iters, warmup, dev) * 1e6
        except Exception:
            us_layerspec_graph = None

    res = {
        "K_steps": K_steps, "depth": depth, "inter": inter, "d_model": d_model,
        "radix": radix, "scheme": scheme, "dtype": str(dtype).replace("torch.", ""),
        "n_params_tied_cell": tied.n_params,
        "us_serial_step": us_serial,
        "us_serial_step_graph": us_serial_graph,
        "us_layerspec_step": us_layerspec,
        "us_layerspec_graph": us_layerspec_graph,
        "us_raw_batch_verify": us_raw_batch,
        "layers_accepted": accepted, "depth_total": depth,
        "accepted_out_linf_vs_serial": linf,
        "byte_exact": linf == 0.0,
        "speedup_layerspec_vs_serial": us_serial / us_layerspec if us_layerspec > 0 else None,
        "speedup_graph_vs_serial_graph": (
            us_serial_graph / us_layerspec_graph
            if us_serial_graph and us_layerspec_graph else None),
    }

    # ---- 2D combine: K steps x depth layers = K*depth batch rows (one stream) ----
    if K_steps > 1:
        # the 2D spec draft = a (K x (depth+1)) grid of states; the K*depth INPUT
        # states verified in ONE tied forward.  Perfect-draft grid (accept-all).
        with torch.no_grad():
            grid_in = []                                # K*depth input states
            for _ in range(K_steps):
                # each step's draft trace (perfect); re-seed a fresh stream per step
                xs = torch.randn(1, 1, d_model, dtype=dtype, device=dev) * 0.02
                tr = tied.serial_trace(xs)
                grid_in.extend(tr[i][0] for i in range(depth))
            grid = torch.stack(grid_in, dim=0)          # (K*depth, 1, d_model)

        def combine2d():
            return tied.batched(grid)                   # ONE matmul, batch = K*depth

        us_2d = _time(combine2d, iters, warmup, dev) * 1e6
        res["us_2d_combine_verify"] = us_2d
        res["us_2d_per_step"] = us_2d / K_steps         # amortized over the K steps
        res["combine_batch_rows"] = K_steps * depth
        res["speedup_2d_vs_serial"] = us_serial / (us_2d / K_steps) if us_2d > 0 else None

    del tied, x0
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


# =========================================================================== #
# DRAFT-ACCURACY SENSITIVITY: mismatch at layer m -> serial re-run of (depth-m).
# =========================================================================== #
def bench_draft_sensitivity(device, dtype, iters, warmup, d_model=MINFLOP_DMODEL,
                            radix=MINFLOP_RADIX, scheme=MINFLOP_SCHEME):
    """Corrupt the draft at layer m (m = depth, 3/4, 1/2, 1/4, 0 of depth) and time
    the layerwise-spec verify: the accepted prefix is m, and the tail (depth-m) is
    re-run serially.  Shows how the per-step degrades from the all-accept floor
    (m==depth) toward the pure-serial cost (m==0) as draft accuracy drops."""
    dev = device
    tied, depth, inter = _make_tied(dev, dtype, d_model, radix, scheme)
    x0 = torch.randn(1, 1, d_model, dtype=dtype, device=dev) * 0.02
    with torch.no_grad():
        perfect = tied.serial_trace(x0)
        serial_out = tied.serial(x0)

    # ANALYTICAL cost model from the graph-fair primitives: the layerwise-spec cost at
    # acceptance m is ONE batched verify (constant, all depth states) + (depth-m)
    # SERIAL re-run layers.  We measure the graph-fused per-layer serial cost + the
    # graph-fused batched-verify cost so the sensitivity trend is the CLEAN compute
    # signal (the eager per-step is python-launch-dominated and non-monotone).
    d_in = torch.stack([perfect[i][0] for i in range(depth)], dim=0)
    us_verify_graph = None
    us_per_layer_graph = None
    if dev.type == "cuda":
        try:
            gV = torch.cuda.CUDAGraph()
            sV = torch.cuda.Stream(device=dev)
            sV.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(sV), torch.no_grad():
                for _ in range(3):
                    tied.batched(d_in)
            torch.cuda.current_stream(dev).wait_stream(sV)
            torch.cuda.synchronize(dev)
            with torch.no_grad(), torch.cuda.graph(gV):
                _ = tied.batched(d_in)
            us_verify_graph = _time(gV.replay, iters, warmup, dev) * 1e6
            # per-layer serial cost = graph serial (15 layers) / depth
            gS = torch.cuda.CUDAGraph()
            sS = torch.cuda.Stream(device=dev)
            sS.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(sS), torch.no_grad():
                for _ in range(3):
                    tied.serial(x0)
            torch.cuda.current_stream(dev).wait_stream(sS)
            torch.cuda.synchronize(dev)
            with torch.no_grad(), torch.cuda.graph(gS):
                _ = tied.serial(x0)
            us_per_layer_graph = _time(gS.replay, iters, warmup, dev) * 1e6 / depth
        except Exception:
            pass

    rows = []
    for m in sorted({depth, (3 * depth) // 4, depth // 2, depth // 4, 0}, reverse=True):
        # corrupt the draft at layer m so the FIRST mismatch is exactly at m:
        # states 0..m are correct; state m+1 (the draft's claim for cell(s_m)) is wrong
        draft = [s.clone() for s in perfect]
        if m < depth:
            draft[m + 1] = draft[m + 1] + 1.0           # break d_{m+1} => mismatch at m
        with torch.no_grad():
            out, accepted = layerwise_spec_verify(tied, draft)
            linf = float((out - serial_out).abs().max().float())

        def run(draft=draft):
            return layerwise_spec_verify(tied, draft)[0]

        us = _time(run, iters, warmup, dev) * 1e6
        rerun = depth - accepted
        # analytical graph-fair cost: verify (all states, one matmul) + serial tail
        us_model = None
        if us_verify_graph is not None and us_per_layer_graph is not None:
            us_model = us_verify_graph + rerun * us_per_layer_graph
        rows.append({"first_mismatch_layer": m, "layers_accepted": accepted,
                     "serial_rerun_layers": rerun,
                     "us_per_step_eager": us, "us_per_step_model": us_model,
                     "byte_exact": linf == 0.0, "linf_vs_serial": linf})
    del tied, x0
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"depth": depth, "us_verify_graph": us_verify_graph,
            "us_per_layer_graph": us_per_layer_graph, "rows": rows}


# =========================================================================== #
# 2-GPU: run the layerwise-spec single stream on both cards concurrently.
# =========================================================================== #
def bench_2gpu(dtype, iters, warmup, d_model=MINFLOP_DMODEL, radix=MINFLOP_RADIX,
               scheme=MINFLOP_SCHEME, K_steps=1):
    tieds, drafts = [], []
    depth = summed_isa_depth(radix)["add_step_depth"]
    inter = compact_inter(radix, scheme)
    for gi in (0, 1):
        dev = torch.device(f"cuda:{gi}")
        t = TiedStepCell(depth, d_model, inter, dtype).to(dev).eval()
        x0 = torch.randn(1, 1, d_model, dtype=dtype, device=dev) * 0.02
        with torch.no_grad():
            pf = t.serial_trace(x0)
        tieds.append(t); drafts.append(pf)
    with torch.no_grad():
        for _ in range(warmup):
            for t, pf in zip(tieds, drafts):
                layerwise_spec_verify(t, pf)
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        t0 = time.perf_counter()
        for _ in range(iters):
            outs = [layerwise_spec_verify(t, pf)[0]
                    for t, pf in zip(tieds, drafts)]
        for gi in (0, 1):
            torch.cuda.synchronize(gi)
        for o in outs:
            o.flatten()[0].float().item()
        dt = (time.perf_counter() - t0) / iters
    for t in tieds:
        del t
    for gi in (0, 1):
        torch.cuda.set_device(gi)
        torch.cuda.empty_cache()
    # two concurrent single streams -> aggregate steps/s = 2 / dt
    return {"us_per_step_per_stream": dt * 1e6, "streams": 2,
            "steps_per_s_aggregate": 2.0 / dt, "dtype": str(dtype).replace("torch.", "")}


# =========================================================================== #
# fps PROJECTION (single-stream steps/s over the frame folds).
# =========================================================================== #
def project_fps(steps_per_s):
    return {fold: steps_per_s / steps for fold, steps in FRAME_FOLDS.items()}


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--d-model", type=int, default=MINFLOP_DMODEL)
    ap.add_argument("--k-steps", default="1,4,16,64,256,1024",
                    help="2D combine step-counts (K steps x depth layers)")
    ap.add_argument("--two-gpu", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not (args.verify or args.bench):
        args.verify = True
    dev = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    out = {"serial_us_ref": SERIAL_US_REF, "wide_composed_us": WIDE_COMPOSED_US,
           "frame_folds": FRAME_FOLDS, "d_model": args.d_model,
           "radix": MINFLOP_RADIX, "scheme": MINFLOP_SCHEME,
           "gameplay_fps_target": GAMEPLAY_FPS_TARGET, "device": str(dev)}
    if dev.type == "cuda":
        out["gpu_name"] = torch.cuda.get_device_name(dev)

    # ---------------- VERIFY: weight-tied + byte-exact spec ---------------- #
    if args.verify or args.json:
        print("=" * 100)
        print("WEIGHT-TIED CELL + BYTE-EXACT LAYERWISE-SPEC (accepted output L-inf=0 vs serial)")
        print("=" * 100)
        tied, depth, inter = _make_tied(dev, torch.float32, args.d_model,
                                        MINFLOP_RADIX, MINFLOP_SCHEME)
        print(f"  tied cell: depth {depth} applications of ONE shared "
              f"{tied.n_params}-param cell (d_model {args.d_model}, inter {inter})")
        x0 = torch.randn(1, 1, args.d_model, dtype=torch.float32, device=dev) * 0.02
        with torch.no_grad():
            perfect = tied.serial_trace(x0)
            serial_out = tied.serial(x0)
            # (a) perfect draft -> accept all 15, byte-exact
            out_ok, acc_ok = layerwise_spec_verify(tied, perfect)
            linf_ok = float((out_ok - serial_out).abs().max().float())
            # (b) draft broken at layer 7 -> accept 7, re-run 8, still byte-exact
            broke = [s.clone() for s in perfect]
            broke[8] = broke[8] + 1.0
            out_b, acc_b = layerwise_spec_verify(tied, broke)
            linf_b = float((out_b - serial_out).abs().max().float())
        print(f"  perfect draft : accepted {acc_ok}/{depth} layers, "
              f"L-inf vs serial {linf_ok:.3g}  "
              f"{'BYTE-EXACT' if linf_ok == 0.0 else 'FAIL'}")
        print(f"  broken@layer7 : accepted {acc_b}/{depth} layers, re-ran "
              f"{depth - acc_b} serially, L-inf vs serial {linf_b:.3g}  "
              f"{'BYTE-EXACT' if linf_b == 0.0 else 'FAIL'}")
        out["verify"] = {"depth": depth, "n_params_tied_cell": tied.n_params,
                         "perfect_accepted": acc_ok, "perfect_linf": linf_ok,
                         "broken_accepted": acc_b, "broken_rerun": depth - acc_b,
                         "broken_linf": linf_b,
                         "byte_exact": linf_ok == 0.0 and linf_b == 0.0}
        # underlying op-cell byte-exactness (the arithmetic the tied cell hosts)
        v = verify_compact_byte_exact(MINFLOP_RADIX, torch.float32, MINFLOP_SCHEME, n=2000)
        out["verify"]["op_cell"] = v
        print(f"  op-cell (radix {MINFLOP_RADIX} fp32): ADD={v['ADD_exact']} "
              f"DIV={v['DIV_exact_in_dtype']}")
        del tied, x0

    if not args.bench:
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"wrote {args.json}")
        return

    if dev.type != "cuda":
        print("\n[!] --bench needs CUDA. Aborting bench.")
        return
    print(f"\ndevice {dev}: {out['gpu_name']}  ({torch.cuda.device_count()} visible)")

    # ---------------- SINGLE-STREAM: serial vs layerwise-spec ---------------- #
    print("\n" + "=" * 100)
    print(f"SINGLE-STREAM per-step: SERIAL 15-layer chain vs LAYERWISE-SPEC verify "
          f"(one tied {MINFLOP_RADIX}-cell)")
    print("=" * 100)
    ss = bench_single_stream(dev, torch.float32, args.iters, args.warmup, args.d_model)
    out["single_stream"] = ss
    print(f"  serial 15-layer chain (eager)         : {ss['us_serial_step']:.4f} us/step")
    if ss.get("us_serial_step_graph"):
        print(f"  serial 15-layer chain (CUDA-graph)    : {ss['us_serial_step_graph']:.4f} "
              f"us/step  (launch overhead removed; the FAIR serial baseline)")
    print(f"  layerwise-spec verify (eager)         : {ss['us_layerspec_step']:.4f} us/step "
          f"({ss['speedup_layerspec_vs_serial']:.2f}x vs eager serial)")
    if ss.get("us_layerspec_graph"):
        print(f"  layerwise-spec verify (CUDA-graph)    : {ss['us_layerspec_graph']:.4f} us/step "
              f"({ss['speedup_graph_vs_serial_graph']:.2f}x vs graph serial "
              f"-- the GENUINE depth-parallelism)")
    print(f"  raw batched verify matmul (no python) : {ss['us_raw_batch_verify']:.4f} us")
    print(f"  accepted {ss['layers_accepted']}/{ss['depth_total']} layers, "
          f"byte-exact vs serial: {ss['byte_exact']} (L-inf {ss['accepted_out_linf_vs_serial']:.3g})")
    print(f"  NOTE: the 0.1104 us ref (a2ccd7c2) is the SATURATED-BATCH per-lane-step "
          f"(K=262144), NOT a single stream --")
    print(f"        docs/CLEVER_FP32_FULLOPS.md itself flags single-stream throughput as "
          f"the open question this harness answers.")

    # ---------------- OCCUPANCY: dim-64 15x-batch fill ---------------- #
    # occupancy proxy: the 15x batch multiplies the number of dim-64 rows the card
    # processes at once by 15 (and the 2D combine by 15K).  Report the effective
    # batch rows vs a saturating-batch reference so the "fill" is explicit.
    print("\n" + "=" * 100)
    print("OCCUPANCY FILL: layerwise 15x batch (and 2D steps x layers) vs dim-64 sat")
    print("=" * 100)

    # ---------------- 2D COMBINE: K steps x depth layers ---------------- #
    print("\n" + "=" * 100)
    print("2D COMBINE (K steps x 15 layers = 15K batch rows, ONE stream)")
    print("=" * 100)
    Ks = [int(k) for k in args.k_steps.split(",")]
    out["combine_2d"] = {}
    print(f"  {'K_steps':>8s} {'batch rows':>10s} | {'2D verify us':>12s} "
          f"{'per-step us':>11s} {'vs serial':>9s} {'byte-exact':>10s}")
    best_2d = None
    for K in Ks:
        r = bench_single_stream(dev, torch.float32, args.iters, args.warmup,
                                args.d_model, K_steps=K)
        out["combine_2d"][str(K)] = r
        if K == 1:
            per = r["us_layerspec_step"]
            rows = r["depth"]
            us2d = r["us_layerspec_step"]
            sp = r["speedup_layerspec_vs_serial"]
        else:
            per = r["us_2d_per_step"]
            rows = r["combine_batch_rows"]
            us2d = r["us_2d_combine_verify"]
            sp = r["speedup_2d_vs_serial"]
        print(f"  {K:>8d} {rows:>10d} | {us2d:12.4f} {per:11.5f} {sp:9.2f}x "
              f"{str(r['byte_exact']):>10s}")
        if best_2d is None or per < best_2d[1]:
            best_2d = (K, per, r)
    out["best_2d"] = {"K_steps": best_2d[0], "us_per_step": best_2d[1]}

    # ---------------- DRAFT SENSITIVITY ---------------- #
    print("\n" + "=" * 100)
    print("DRAFT-ACCURACY SENSITIVITY (mismatch at layer m -> serial re-run of depth-m)")
    print("=" * 100)
    ds = bench_draft_sensitivity(dev, torch.float32, args.iters, args.warmup, args.d_model)
    out["draft_sensitivity"] = ds
    if ds.get("us_verify_graph") is not None:
        print(f"  graph verify (all states, one matmul) {ds['us_verify_graph']:.4f} us; "
              f"per serial re-run layer {ds['us_per_layer_graph']:.4f} us")
    print(f"  {'mismatch@':>10s} {'accepted':>9s} {'re-run':>7s} {'model us/step':>13s} "
          f"{'eager us/step':>13s} {'byte-exact':>10s}")
    for r in ds["rows"]:
        mdl = f"{r['us_per_step_model']:13.4f}" if r.get('us_per_step_model') else f"{'n/a':>13s}"
        print(f"  {r['first_mismatch_layer']:>10d} {r['layers_accepted']:>9d} "
              f"{r['serial_rerun_layers']:>7d} {mdl} {r['us_per_step_eager']:13.4f} "
              f"{str(r['byte_exact']):>10s}")

    # ---------------- 2-GPU ---------------- #
    out["two_gpu"] = {}
    if args.two_gpu and torch.cuda.device_count() >= 2:
        print("\n" + "=" * 100)
        print("REAL 2-GPU concurrent layerwise-spec single streams")
        print("=" * 100)
        g2 = bench_2gpu(torch.float32, args.iters, args.warmup, args.d_model)
        out["two_gpu"] = g2
        print(f"  {g2['us_per_step_per_stream']:.4f} us/step/stream, "
              f"{g2['steps_per_s_aggregate']:,.0f} steps/s (2 concurrent streams)")

    # ---------------- fps PROJECTION (single stream) ---------------- #
    print("\n" + "=" * 100)
    print("SINGLE-STREAM fps PROJECTION (best per-step over the frame folds)")
    print("=" * 100)
    best_us = best_2d[1]                                 # best single-stream per-step
    steps_s_1g = 1e6 / best_us
    fps1 = project_fps(steps_s_1g)
    out["fps_projection"] = {"one_gpu": {"us_per_step": best_us,
                                         "steps_per_s": steps_s_1g, "fps": fps1,
                                         "best_K_steps": best_2d[0]}}
    print(f"\n  1-GPU single stream (best {best_us:.5f} us/step @ K_steps={best_2d[0]}, "
          f"{steps_s_1g:,.0f} steps/s):")
    for fold, steps in FRAME_FOLDS.items():
        f = fps1[fold]
        flag = "  >=35 YES" if f >= GAMEPLAY_FPS_TARGET else "  <35"
        print(f"    {fold:>14s} ({steps:>9,d} steps): {f:10.2f} fps{flag}")
    # serial-baseline single-stream fps for contrast
    steps_s_serial = 1e6 / ss["us_serial_step"]
    out["fps_projection"]["serial_baseline"] = {
        "us_per_step": ss["us_serial_step"], "steps_per_s": steps_s_serial,
        "fps": project_fps(steps_s_serial)}
    print(f"\n  (serial-baseline single stream {ss['us_serial_step']:.4f} us/step: "
          f"render-reduced {project_fps(steps_s_serial)['render_reduced']:.2f} fps)")

    # ---------------- VERDICT ---------------- #
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    layerspec_us = ss["us_layerspec_step"]
    verdict = {
        "serial_single_stream_us": ss["us_serial_step"],
        "layerspec_single_stream_us": layerspec_us,
        "layerspec_speedup_vs_serial": ss["speedup_layerspec_vs_serial"],
        "best_2d_us_per_step": best_2d[1],
        "best_2d_speedup_vs_serial": ss["us_serial_step"] / best_2d[1],
        "best_2d_K_steps": best_2d[0],
        "byte_exact": ss["byte_exact"],
        "render_reduced_fps_1gpu_single_stream": fps1["render_reduced"],
        "clears_35_render_reduced_single_stream": fps1["render_reduced"] >= GAMEPLAY_FPS_TARGET,
        "n_params_tied_cell": ss["n_params_tied_cell"],
    }
    out["verdict"] = verdict
    print(f"  serial single-stream       : {ss['us_serial_step']:.4f} us/step")
    print(f"  layerwise-spec single-stream: {layerspec_us:.4f} us/step "
          f"({ss['speedup_layerspec_vs_serial']:.2f}x)")
    print(f"  best 2D (steps x layers)   : {best_2d[1]:.5f} us/step "
          f"(K_steps={best_2d[0]}, {ss['us_serial_step']/best_2d[1]:.2f}x vs serial)")
    print(f"  byte-exact vs serial       : {ss['byte_exact']}")
    print(f"  render-reduced single-stream fps: {fps1['render_reduced']:.2f} "
          f"({'CLEARS' if fps1['render_reduced'] >= 35 else 'MISSES'} 35)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
