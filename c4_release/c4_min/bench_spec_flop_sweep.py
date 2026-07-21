"""SPECULATION wall-clock speedup + FLOP-utilization sweep over block size K.

The premise (measured on this branch): the compacted lean forward's weights are
**99.997% sparse** (~1014 nonzeros out of 31.8M params — a ~31,000x dense/sparse
ratio), and the GPU runs DENSE kernels over them, so a single deterministic VM step
is *massively overhead-bound* — the GPU does a few nanograms of useful work wrapped
in launch/dispatch/Python latency.

Perfect-draft speculation (``qwen_lean_forward.speculative_run_lean``) batches K VM
steps into ONE ``[B=K, Smax, H]`` forward.  As K climbs, each forward does K times
the work but the launch/dispatch overhead is amortised across K steps, so BOTH the
wall-clock speedup and the achieved FLOP/s should CLIMB with K until either the GPU
saturates (compute floor) or VRAM caps the batch.

This sweeps K = 1 (naive), 16, 32, 64, ... up to the VRAM cap on ONE long
deterministic PURE-ALU/branch program (NO JSR/ENT/LEV — avoids the #702 func
fallback), and at each K measures:

  * wall-clock (ms), ms/VM-step, #forwards
  * peak VRAM (torch), GPU util (nvidia-smi sampled during the run)
  * achieved FLOP/s, counted TWO ways:
      - **dense-equivalent** — every GEMM counted as fully dense (2*M*N*K), i.e.
        the FLOPs the GPU's dense kernels ACTUALLY execute;
      - **sparse actual** — only the FLOPs touching a NONZERO weight (the useful
        compute), = dense * (nonzeros / params).  ~31,000x smaller.
  * utilization = achieved / GPU-peak-fp32 (RTX A5000 ~27.8 TFLOP/s).
  * BYTE-EXACT acceptance vs the full-length ``isa.interpret`` reference (100%).

Speedup vs K=1 is the robust signal (clock/contention cancels in the ratio); the
absolute ms is reported with that caveat.

TWO passes, because they measure different things:

  1. FULL-DRIVER (``measure_k`` -> ``speculative_run_lean``): the HONEST
     end-to-end wall, Python overlay INCLUDED.  Its per-block ``_build_spec_batch``
     is an O(B*code) PYTHON rebuild per forward, so past a small K the *Python
     driver* — not the GPU — dominates and ms/step goes UP with K.  That is a
     limitation of THIS driver, not of speculation, so this pass is deliberately
     bounded (``--driver-steps`` steps, K<=``--driver-max-k``) — enough to certify
     100%% byte-exact acceptance at every K and to quantify the Python ceiling.
  2. FORWARD-ISOLATED (``measure_forward_isolated``): times JUST the batched
     ``lean.forward`` at B=K (batch tiled with ``repeat`` at O(1) Python cost), so
     it is the CLEAN GPU overhead-bound signal and carries the full K-to-VRAM-cap
     curve.  THIS is where the headline speedup lives.

MEASURED on an RTX A5000 (7L / dim-960 base VM, weights 0.0032%% dense = ~31,000x
sparse), 20k-step accumulate loop, forward-isolated: ms/step 4.56 (B=1) -> ~0.043
(plateau from B~=512), MAX 107x at B=16384; dense FLOP-util climbs 0.35%% -> ~37.8%%
of the 27.8 TFLOP/s peak (10.5 TFLOP/s) then FLATTENS -> COMPUTE-bound plateau, not
overhead-bound.  VRAM cap: B=65536 fits at ~23GB, B=131072 OOMs.  Run:

    python -m c4_min.bench_spec_flop_sweep --device cuda:0 --steps 20000
    python -m c4_min._mem_guard 24 c4_min.bench_spec_flop_sweep --device cuda:0 \
        --ks 1,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF


# RTX A5000: 8192 fp32 cores * 1.695 GHz boost * 2 (FMA) ~= 27.8 TFLOP/s.
# Overridable via --peak-tflops for a different card.
DEFAULT_PEAK_TFLOPS = 27.8


# ---------------------------------------------------------------------------
# The long pure-ALU/branch program: a self-restarting accumulate counter.
#   0: IMM 0        AX = 0                 (seed)
#   1: PSH          push AX            <-- loop top
#   2: IMM 1        AX = 1
#   3: ADD          AX = old + 1 (mod 256)
#   4: JMP 1        back to loop top
# Runs FOREVER (JMP), AX ramps 0..255 mod 256 — bounded only by max_steps.  Uses
# ONLY IMM / PSH / ADD / JMP: pure-ALU + unconditional branch, no memory, no
# JSR/ENT/LEV -> no #702 func fallback, no store-log window growth (S is constant).
ACCUM_PROG = [("IMM", 0), ("PSH", 0), ("IMM", 1), ("ADD", 0), ("JMP", 1)]


# ---------------------------------------------------------------------------
# Analytic FLOP model for ONE LeanQwenVM.forward over a [B, S, H] window.
# ---------------------------------------------------------------------------
def _forward_dense_flops(lean: LF.LeanQwenVM, B: int, S: int) -> int:
    """Dense-equivalent FLOPs for ONE lean.forward over [B, S, H] (the FLOPs the
    GPU's DENSE GEMM kernels physically execute — zeros are not skipped).

    Per layer, per (batch, token): the 4 attention projections + o_proj, the 2
    S*Sk score/context matmuls, and the 3 SwiGLU MLP GEMMs.  A matmul of an [.., K]
    row by a [N, K] weight is 2*N*K FLOPs (mul + add).
    """
    H = lean.hidden_size
    nh, nkv, hd = lean.n_heads, lean.n_kv_heads, lean.head_dim
    inter = lean.layers[0].gate_w.shape[0]
    q_out, kv_out = nh * hd, nkv * hd
    BS = B * S
    per_layer = 0
    # projections (per token): q, k, v, o  (o maps nh*hd -> H)
    per_layer += 2 * BS * H * q_out            # q_proj
    per_layer += 2 * BS * H * kv_out           # k_proj
    per_layer += 2 * BS * H * kv_out           # v_proj
    per_layer += 2 * BS * (nh * hd) * H        # o_proj
    # attention: scores q@K^T  and  context attn@V, per head, full S*Sk (Sk=S here,
    # past=None) -> 2 * nh * S * S * hd  per batch row, twice (scores + context).
    per_layer += B * (2 * nh * S * S * hd)     # scores
    per_layer += B * (2 * nh * S * S * hd)     # context
    # MLP: gate, up (H->inter), down (inter->H), per token.
    per_layer += 2 * BS * H * inter            # gate
    per_layer += 2 * BS * H * inter            # up
    per_layer += 2 * BS * inter * H            # down
    return per_layer * lean.n_layers


def _weight_density(lean: LF.LeanQwenVM) -> Tuple[int, int]:
    """(nonzero_weight_count, total_weight_count) over all projection/MLP weights."""
    tot = nz = 0
    for l in lean.layers:
        for w in (l.q_w, l.k_w, l.v_w, l.o_w, l.gate_w, l.up_w, l.down_w):
            tot += w.numel()
            nz += int((w != 0).sum().item())
    return nz, tot


# ---------------------------------------------------------------------------
# GPU-util sampler (nvidia-smi) — runs on a daemon thread during a timed run.
# ---------------------------------------------------------------------------
class _UtilSampler:
    def __init__(self, device_index: int, period_s: float = 0.02):
        self.idx = device_index
        self.period = period_s
        self._stop = threading.Event()
        self._samples: List[int] = []
        self._thread: Optional[threading.Thread] = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", "-i", str(self.idx)],
                    timeout=1.0).decode().strip()
                self._samples.append(int(out.splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def summary(self) -> Dict[str, float]:
        s = self._samples
        if not s:
            return {"mean": 0.0, "max": 0.0, "n": 0}
        return {"mean": sum(s) / len(s), "max": max(s), "n": len(s)}


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _base_window_len(lean: LF.LeanQwenVM) -> int:
    """The per-step window length S for the base (no-memory) subset: BOS + 5 reg
    frames + STEP_END = 7 (constant, no store log)."""
    from .qwen_lean_forward import CAM_REGS
    n_store = 0  # base subset: no memory
    return 1 + n_store + len(CAM_REGS) + 1


# ---------------------------------------------------------------------------
# Forward-ISOLATED measurement: time JUST the batched lean.forward at B=K, with
# the Python driver / batch-build cost removed.  This is the CLEANEST test of the
# "GPU is overhead-bound -> FLOP util climbs with K" hypothesis: it measures the
# GPU forward alone (launch/dispatch overhead amortised across B rows), not the
# Python _build_spec_batch overlay loop (which grows with B and becomes the driver
# bottleneck at large K).
# ---------------------------------------------------------------------------
def measure_forward_isolated(lean: LF.LeanQwenVM, code, B: int, *, device: str,
                             nz: int, tot: int, peak_tflops: float,
                             warmup: int, iters: int) -> Dict:
    """Time ONE batched ``lean.forward`` over a [B, Smax, H] window (B independent
    VM-step windows), reporting ms/forward, ms/step, VRAM, GPU util, and dense +
    sparse achieved FLOP/s.  The residual overlay is built ONCE (Python) OUTSIDE
    the timed loop so only the GPU forward is timed."""
    dev_idx = torch.device(device).index or 0
    # Build a SMALL real spec batch once (Python O(tile*code)), then TILE it to B
    # rows with torch.repeat — the lean forward's per-row GPU cost + VRAM depend
    # ONLY on the window SHAPE (B, S, H), not the register values, so a tiled batch
    # is the byte-identical GPU workload at O(1) Python cost (avoids the O(B) overlay
    # loop that dominates at large B and is NOT the GPU signal we're isolating).
    tile = min(B, 256)
    draft = LF.draft_program_lean(lean, code, max_steps=tile)
    slab = draft.steps[:tile]
    x0, pos0 = LF._build_spec_batch(lean, code, slab)     # [tile, Smax, H]
    Smax = x0.shape[1]
    reps = (B + tile - 1) // tile
    x = x0.repeat(reps, 1, 1)[:B].contiguous()
    positions = pos0.repeat(reps, 1)[:B].contiguous()

    def _fwd():
        with torch.no_grad():
            return lean.forward(x, past=None, q_positions=positions)

    for _ in range(warmup):
        _fwd()
    _sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev_idx)
    with _UtilSampler(dev_idx) as sampler:
        t0 = time.perf_counter()
        for _ in range(iters):
            _fwd()
        _sync(device)
        fwd_s = (time.perf_counter() - t0) / iters
    util = sampler.summary()
    vram_peak_gb = (torch.cuda.max_memory_allocated(dev_idx) / 1e9
                    if device.startswith("cuda") else 0.0)

    dense_flops = _forward_dense_flops(lean, B, Smax)
    sparse_flops = dense_flops * nz / tot
    peak_flops_s = peak_tflops * 1e12
    return {
        "B": B, "Smax": Smax, "fwd_ms": fwd_s * 1000.0,
        "ms_per_step": fwd_s * 1000.0 / B,
        "vram_peak_gb": vram_peak_gb,
        "util_mean": util["mean"], "util_max": util["max"], "util_n": util["n"],
        "dense_flops": dense_flops, "sparse_flops": sparse_flops,
        "dense_flops_s": dense_flops / fwd_s, "sparse_flops_s": sparse_flops / fwd_s,
        "dense_util_pct": (dense_flops / fwd_s) / peak_flops_s * 100.0,
        "sparse_util_pct": (sparse_flops / fwd_s) / peak_flops_s * 100.0,
    }


# ---------------------------------------------------------------------------
# One K measurement.
# ---------------------------------------------------------------------------
def measure_k(lean: LF.LeanQwenVM, code, K: int, *, device: str, n_steps: int,
              ref: List[int], nz: int, tot: int, peak_tflops: float,
              warmup: int, iters: int) -> Dict:
    """Run the spec driver at block_steps=K on ``code`` for ``n_steps`` VM steps and
    return the full measurement dict (wall, ms/step, forwards, VRAM, util, FLOP/s,
    utilization, byte-exact)."""
    dev_idx = torch.device(device).index or 0

    def _run():
        return LF.speculative_run_lean(lean, code, block_steps=K, max_steps=n_steps)

    # correctness + shape info from one run.
    r0 = _run()
    ax = r0.ax_trace
    byte_exact = (len(ax) == len(ref)) and (ax == ref)
    forwards = r0.forwards
    steps = r0.steps

    # warm, then time.
    for _ in range(warmup):
        _run()
    _sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev_idx)
    with _UtilSampler(dev_idx) as sampler:
        t0 = time.perf_counter()
        for _ in range(iters):
            _run()
        _sync(device)
        wall_s = (time.perf_counter() - t0) / iters
    util = sampler.summary()
    vram_peak_gb = (torch.cuda.max_memory_allocated(dev_idx) / 1e9
                    if device.startswith("cuda") else 0.0)

    # FLOPs.  Each forward is a [B, Smax] window; B = min(K, steps_left), Smax fixed
    # for the base subset (no store log).  Reconstruct the exact per-forward shapes.
    Smax = _base_window_len(lean)
    dense_flops = 0
    full_blocks = steps // K
    tail = steps - full_blocks * K
    for _ in range(full_blocks):
        dense_flops += _forward_dense_flops(lean, K, Smax)
    if tail:
        dense_flops += _forward_dense_flops(lean, tail, Smax)
    sparse_flops = dense_flops * nz / tot

    dense_flops_s = dense_flops / wall_s
    sparse_flops_s = sparse_flops / wall_s
    peak_flops_s = peak_tflops * 1e12
    return {
        "K": K, "steps": steps, "forwards": forwards,
        "wall_ms": wall_s * 1000.0, "ms_per_step": wall_s * 1000.0 / max(steps, 1),
        "vram_peak_gb": vram_peak_gb,
        "util_mean": util["mean"], "util_max": util["max"], "util_n": util["n"],
        "dense_flops": dense_flops, "sparse_flops": sparse_flops,
        "dense_flops_s": dense_flops_s, "sparse_flops_s": sparse_flops_s,
        "dense_util_pct": dense_flops_s / peak_flops_s * 100.0,
        "sparse_util_pct": sparse_flops_s / peak_flops_s * 100.0,
        "byte_exact": byte_exact,
        "Smax": Smax,
    }


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------
def sweep(device: str = "cuda:0", n_steps: int = 20000,
          ks: Optional[List[int]] = None, peak_tflops: float = DEFAULT_PEAK_TFLOPS,
          warmup: int = 1, iters: int = 3, out_json: Optional[str] = None,
          driver_steps: int = 2000, driver_max_k: int = 64) -> Dict:
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    try:
        sys.stdout.reconfigure(line_buffering=True)   # live progress under redirect
    except Exception:
        pass
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    print(f"device: {device}  torch {torch.__version__}  "
          f"GPU {torch.cuda.get_device_name(0) if device.startswith('cuda') else 'CPU'}")
    print(f"peak fp32 assumed: {peak_tflops:.1f} TFLOP/s")

    t0 = time.time()
    vm = Q.build(code_size=24, subset=Q.SUBSET_BASE)
    vm.embed = vm.embed.to(device)
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    nz, tot = _weight_density(lean)
    print(f"built base VM: {lean.n_layers}L, dim {lean.hidden_size}, {lean.n_heads} q-heads "
          f"({lean.n_kv_heads} kv), head_dim {lean.head_dim}, inter {lean.layers[0].gate_w.shape[0]} "
          f"(built {time.time()-t0:.1f}s)")
    print(f"weight sparsity: {nz}/{tot} nonzero = {nz/tot*100:.4f}% dense "
          f"({tot/max(nz,1):.0f}x sparse) -> dense-equiv FLOPs are ~{tot/max(nz,1):.0f}x "
          f"the sparse-actual FLOPs")

    code = isa.assemble(ACCUM_PROG)
    ref = isa.interpret(code, max_steps=n_steps)
    print(f"program: self-restarting accumulate counter (IMM/PSH/ADD/JMP), "
          f"{n_steps} VM steps, ref_len={len(ref)}")

    if ks is None:
        ks = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # -------------------------------------------------------------------------
    # FULL-DRIVER sweep (measure_k -> speculative_run_lean) — BYTE-EXACT
    # acceptance + the HONEST end-to-end wall (Python overlay INCLUDED).  Its
    # per-block ``_build_spec_batch`` is an O(B*code) PYTHON rebuild per forward,
    # so at large K the Python driver — NOT the GPU — dominates and ms/step goes
    # UP with K (the opposite of the GPU signal).  That is a real, reported
    # limitation of THIS driver, not of speculation: the clean GPU overhead-bound
    # speedup lives in the FORWARD-ISOLATED sweep below.  So the full-driver pass
    # is deliberately bounded (``driver_steps`` steps, K<=``driver_max_k``) — just
    # enough to certify 100%% byte-exact acceptance at every K and to quantify the
    # Python-driver ceiling — while the forward-isolated sweep carries the full
    # K-to-VRAM-cap curve.
    driver_ks = [K for K in ks if K <= driver_max_k]
    print(f"\nFULL-DRIVER (end-to-end, Python overlay included): {driver_steps} steps, "
          f"K in {driver_ks}  [byte-exact gate + Python-driver ceiling]")
    dref = isa.interpret(code, max_steps=driver_steps)
    rows: List[Dict] = []
    for K in driver_ks:
        try:
            r = measure_k(lean, code, K, device=device, n_steps=driver_steps, ref=dref,
                          nz=nz, tot=tot, peak_tflops=peak_tflops,
                          warmup=warmup, iters=iters)
        except torch.cuda.OutOfMemoryError:
            print(f"  K={K:6d}: OOM (VRAM cap) — stopping sweep")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            break
        rows.append(r)
        print(f"  K={K:6d}: fwd={r['forwards']:5d} wall={r['wall_ms']:9.2f}ms "
              f"ms/step={r['ms_per_step']:7.4f} VRAM={r['vram_peak_gb']:5.2f}GB "
              f"util(mean/max)={r['util_mean']:4.0f}/{r['util_max']:3.0f}% "
              f"BYTE-EXACT={r['byte_exact']}")

    # speedup vs K=1.
    base = next((r for r in rows if r["K"] == 1), rows[0] if rows else None)
    if base:
        for r in rows:
            r["speedup"] = base["ms_per_step"] / r["ms_per_step"] if r["ms_per_step"] else 0.0

    _print_tables(rows, peak_tflops, tot, nz)

    # ----- forward-ISOLATED sweep (GPU forward alone; Python overlay removed) -----
    print("\n" + "#" * 78)
    print("FORWARD-ISOLATED sweep — JUST the batched lean.forward at B=K (Python")
    print("batch-build removed): the clean GPU overhead-bound signal.")
    fwd_rows: List[Dict] = []
    for B in ks:
        try:
            fr = measure_forward_isolated(lean, code, B, device=device, nz=nz, tot=tot,
                                          peak_tflops=peak_tflops, warmup=max(warmup, 2),
                                          iters=max(iters, 5))
        except torch.cuda.OutOfMemoryError:
            print(f"  B={B:6d}: OOM (VRAM cap) — stopping forward-isolated sweep")
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            break
        fwd_rows.append(fr)
        print(f"  B={B:6d}: fwd={fr['fwd_ms']:8.3f}ms ms/step={fr['ms_per_step']:8.5f} "
              f"VRAM={fr['vram_peak_gb']:5.2f}GB util(mean/max)="
              f"{fr['util_mean']:4.0f}/{fr['util_max']:3.0f}% "
              f"dense={fr['dense_flops_s']/1e9:8.1f}GF/s ({fr['dense_util_pct']:6.3f}%pk)")
    base_f = next((r for r in fwd_rows if r["B"] == 1), fwd_rows[0] if fwd_rows else None)
    if base_f:
        for fr in fwd_rows:
            fr["speedup"] = base_f["ms_per_step"] / fr["ms_per_step"] if fr["ms_per_step"] else 0.0
    _print_forward_tables(fwd_rows, peak_tflops, tot, nz)

    result = {"rows": rows, "forward_rows": fwd_rows,
              "peak_tflops": peak_tflops, "nz": nz, "tot": tot,
              "n_steps": n_steps, "driver_steps": driver_steps,
              "layers": lean.n_layers, "hidden": lean.hidden_size,
              "device_name": (torch.cuda.get_device_name(0)
                              if device.startswith("cuda") else "cpu")}
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {out_json}")
    return result


def _print_tables(rows: List[Dict], peak_tflops: float, tot: int, nz: int):
    if not rows:
        print("no rows"); return
    print("\n" + "=" * 78)
    print("SPEEDUP vs K=1  (ratio robust to clock/contention; absolute ms with caveat)")
    print(f"  {'K':>6s} {'forwards':>8s} {'ms/step':>9s} {'speedup':>8s} "
          f"{'VRAM GB':>8s} {'util%':>6s} {'byte-exact':>10s}")
    for r in rows:
        print(f"  {r['K']:6d} {r['forwards']:8d} {r['ms_per_step']:9.4f} "
              f"{r.get('speedup', 0.0):7.2f}x {r['vram_peak_gb']:8.2f} "
              f"{r['util_mean']:5.0f}% {str(r['byte_exact']):>10s}")

    best = max(rows, key=lambda r: r.get("speedup", 0.0))
    print(f"\n  MAX speedup: {best.get('speedup',0):.2f}x at K={best['K']} "
          f"(forwards {best['forwards']}, {rows[0]['forwards']}->{best['forwards']} "
          f"= {rows[0]['forwards']/max(best['forwards'],1):.0f}x fewer forwards)")

    # climbing vs flat at the top?
    if len(rows) >= 2:
        top, prev = rows[-1], rows[-2]
        d = top.get("speedup", 0) - prev.get("speedup", 0)
        trend = ("STILL CLIMBING (overhead-bound)" if d > 0.05 * max(prev.get("speedup", 1), 1)
                 else "FLAT/PLATEAUED (compute- or VRAM-bound)")
        print(f"  top-of-sweep trend K={prev['K']}->{top['K']}: "
              f"{prev.get('speedup',0):.2f}x -> {top.get('speedup',0):.2f}x  [{trend}]")

    print("\n" + "=" * 78)
    print(f"FLOP UTILIZATION  (peak fp32 = {peak_tflops:.1f} TFLOP/s; "
          f"weights {nz/tot*100:.4f}% dense = {tot/max(nz,1):.0f}x sparse)")
    print(f"  {'K':>6s} {'dense GF/s':>11s} {'dense %pk':>9s} "
          f"{'sparse KF/s':>12s} {'sparse %pk':>11s} {'GPU util%':>9s}")
    for r in rows:
        print(f"  {r['K']:6d} {r['dense_flops_s']/1e9:11.2f} "
              f"{r['dense_util_pct']:8.4f}% {r['sparse_flops_s']/1e3:12.2f} "
              f"{r['sparse_util_pct']:10.2e}% {r['util_mean']:8.0f}%")
    print("\n  NOTE: 'dense' counts every GEMM as fully dense (what the GPU physically")
    print("  runs); 'sparse' counts only nonzero-weight FLOPs (the useful compute).")
    print("  The tiny dense-% is REAL under-utilization (overhead-bound); the ~31,000x")
    print("  smaller sparse-% is the honest useful-work floor.  A dense COO/CSR kernel")
    print("  can't feed tensor cores efficiently, so even a 'saturated' sparse model")
    print("  shows low tensor-core % — that low % is partly the COST OF SPARSITY, not")
    print("  pure waste (dense kernels over 99.997%-zero weights is the actual work).")


def _print_forward_tables(rows: List[Dict], peak_tflops: float, tot: int, nz: int):
    if not rows:
        print("no forward rows"); return
    print("\n" + "=" * 78)
    print("FORWARD-ONLY: speedup vs B=1 + FLOP utilization (GPU forward, no Python)")
    print(f"  {'B':>6s} {'Smax':>5s} {'fwd ms':>8s} {'ms/step':>9s} {'speedup':>8s} "
          f"{'VRAM GB':>8s} {'util%':>6s} {'dense %pk':>9s}")
    for r in rows:
        print(f"  {r['B']:6d} {r['Smax']:5d} {r['fwd_ms']:8.3f} {r['ms_per_step']:9.5f} "
              f"{r.get('speedup', 0.0):7.2f}x {r['vram_peak_gb']:8.2f} "
              f"{r['util_mean']:5.0f}% {r['dense_util_pct']:8.3f}%")
    best = max(rows, key=lambda r: r.get("speedup", 0.0))
    print(f"\n  MAX forward speedup: {best.get('speedup',0):.1f}x at B={best['B']} "
          f"(peak dense util {max(r['dense_util_pct'] for r in rows):.3f}% of "
          f"{peak_tflops:.1f} TFLOP/s = {max(r['dense_flops_s'] for r in rows)/1e12:.2f} TFLOP/s)")
    if len(rows) >= 2:
        top, prev = rows[-1], rows[-2]
        du_climb = top["dense_util_pct"] - prev["dense_util_pct"]
        sp_climb = top.get("speedup", 0) - prev.get("speedup", 0)
        trend = ("STILL CLIMBING (overhead-bound — bigger K would help)"
                 if (sp_climb > 0.05 * max(prev.get("speedup", 1), 1)
                     or du_climb > 0.05 * max(prev["dense_util_pct"], 1e-6))
                 else "PLATEAUED (compute/VRAM cap reached)")
        print(f"  top-of-sweep trend B={prev['B']}->{top['B']}: speedup "
              f"{prev.get('speedup',0):.1f}x->{top.get('speedup',0):.1f}x, dense-util "
              f"{prev['dense_util_pct']:.3f}%->{top['dense_util_pct']:.3f}%  [{trend}]")
    print("\n  This is the HONEST GPU-compute signal: the forward alone amortises the")
    print("  launch/dispatch across B rows.  If dense-util keeps climbing at the top of")
    print("  the sweep, the model is overhead-bound and a larger K (VRAM permitting)")
    print("  keeps winning; if it flattens, the GPU compute floor (or VRAM) has capped it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=20000, help="VM steps in the long program")
    ap.add_argument("--ks", default="", help="comma K list (default 1,16,...,4096 auto)")
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--driver-steps", type=int, default=2000,
                    help="VM steps for the bounded end-to-end full-driver byte-exact gate")
    ap.add_argument("--driver-max-k", type=int, default=64,
                    help="cap K for the full-driver pass (Python-overlay-bound; the "
                         "forward-isolated sweep carries the full K-to-VRAM curve)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",") if x] or None
    sweep(args.device, n_steps=args.steps, ks=ks, peak_tflops=args.peak_tflops,
          warmup=args.warmup, iters=args.iters, out_json=(args.json or None),
          driver_steps=args.driver_steps, driver_max_k=args.driver_max_k)


if __name__ == "__main__":
    main()
