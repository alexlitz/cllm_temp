#!/usr/bin/env python3
r"""clever_forwards_realtime.py — measure the "SHALLOW network x K forwards-per-step"
realtime config (the one that fits a STOCK vanilla 0.5B: L<=24 layers) against the
"DEEP network x 1 forward" config of the SAME effective depth D.

WHY THIS EXISTS (the crux, from docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md §5):
    The clever digit-extraction chain wants ~D "depth" (one digit per layer). Three
    places to put that depth:
      (a) distinct LAYERS, unrolled  -> D layers in ONE forward (one kernel-launch
          chain, one KV read) -- but D~=51 exceeds a stock 24-layer budget.
      (b) looped/tied layers          -> Universal Transformer, NOT vanilla.
      (c) forwards-per-step (this)    -> L<=24 layers but F=ceil(D/L) forwards per VM
          step (one digit per emitted token), threading the running remainder through
          the KV cache between forwards. This is the ORDINARY autoregressive loop, so
          it is genuinely VANILLA and fits a stock 0.5B.
    Compute is conserved (~D layer-applications/step either way) but the realtime cost
    is NOT: the DEEP config pays ONE launch-chain + ONE KV read per step; the SHALLOW
    config pays F launch-chains + F KV re-reads per step, and batches worse (§5's
    "forwards-per-step is more launch/occupancy-bound"). This script MEASURES whether
    that launch/KV overhead breaks realtime for the stock-vanilla-fitting shallow form.

WHAT IT BUILDS (real torch, vanilla shapes -- SHAPE drives walltime):
    A Qwen2.5-0.5B-shaped transformer block: hidden=896, GQA (14 q-heads / 2 kv-heads,
    head_dim 64), SwiGLU FFN (intermediate 4864), RMSNorm. The clever digit-extract
    decode head is appended (a 10-candidate difference-min over the hidden's dim0), so
    each forward realises one/several digit-layers of depth. (If a clever cell is not
    wired, correct-SHAPE zero-init weights are used -- walltime is shape-driven and
    identical; the arithmetic is verified byte-exact separately in clever_minparam_alu.py.)

    * DEEP config:    L = D layers, F = 1 forward per VM step.
    * SHALLOW config: L in {2,3,4} layers, F = ceil(D/L) forwards per VM step. The F
      forwards run SEQUENTIALLY, each appending 1 token to a threaded KV cache (mimicking
      the autoregressive digit-per-token loop). Forward f attends over the f tokens
      emitted so far -- the running-remainder thread.

    Both configs do ~D layer-applications per VM step (compute conserved). The DEEP
    forward advances the sequence by 1 token/step; the SHALLOW config advances it by F
    tokens/step (so KV grows F x faster -- §5/§6 KV tradeoff, measured in §KV below).

WHAT IT MEASURES (respecting a GPU-contention gate):
    per-VM-step walltime for DEEP vs SHALLOW, batched, swept to saturation, at fp32 AND
    bf16; a launch-overhead decomposition for the SHALLOW config (F x per-forward launch
    cost vs actual compute, via a fixed-batch small-vs-large forward probe and a
    CUDA-graph-replay comparison); fps for both at the render-reduced Doom step count
    (358,058 steps/frame, MEASURED, docs/CLEVER_DOOM_REALTIME.md); and the KV-cache size
    for SHALLOW (F x tokens/step) vs DEEP at the doom seq-length.

Run:
    python examples/clever_forwards_realtime.py                 # full sweep, fp32+bf16, GPU
    python examples/clever_forwards_realtime.py --device cuda:1 # pin a specific (idle) GPU
    python examples/clever_forwards_realtime.py --json out.json # machine-readable
    python examples/clever_forwards_realtime.py --quick         # smaller sweep (dev)
    python examples/clever_forwards_realtime.py --no-gpu-gate   # skip the contention poll

Touches NO build files -- the c4 golden (174ece66) is unchanged. CPU-safe (analytic
fallback if no CUDA).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Render-reduced Doom step count -- MEASURED. Source: docs/CLEVER_DOOM_REALTIME.md §3
# ("render-reduced 358,058") and examples/serial_doom_floor.py (RENDER_STEPS = 358_058).
# Each VM step = one c4 instruction (fetch+decode+execute+writeback). In the DEEP config
# one VM step = one forward; in the SHALLOW config one VM step = F forwards.
DOOM_RENDER_STEPS = 358_058
DOOM_RAW_STEPS = 6_889_264      # raw title-redraw frame (for reference)

# Qwen2.5-0.5B ("stock vanilla") architecture -- SHAPE that drives walltime.
HIDDEN = 896
Q_HEADS = 14
KV_HEADS = 2                    # GQA
HEAD_DIM = 64
INTERMEDIATE = 4864            # SwiGLU FFN
NCAND = 10                     # clever decode: candidate digits 0..9

# A5000 (GA102) datasheet fp32 peak, for a % -peak sanity column.
A5000_FP32_PEAK = 27.77e12

STOCK_MAX_LAYERS = 24          # a stock 0.5B has 24 distinct layers


# --------------------------------------------------------------------------- #
# A single vanilla transformer block: RMSNorm -> GQA attention -> RMSNorm -> SwiGLU.
# Real torch tensors at Qwen2.5-0.5B shapes. This is what drives walltime; the clever
# decode head (below) is the only "clever" addition and is cheap (10-wide).
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v.type_as(x)) * self.weight


class VanillaBlock(nn.Module):
    """One Qwen2.5-0.5B-shaped decoder block with a KV cache (GQA + SwiGLU + RMSNorm).

    forward(x, past_kv) -> (y, new_kv). x is (B, 1, H) (one token/forward). past_kv is the
    threaded (K, V) or None. Attention is causal over past+current (ALiBi-free here; the
    clever place-value ALiBi is a per-head bias scalar, negligible for walltime)."""
    def __init__(self, hidden=HIDDEN, q_heads=Q_HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                 inter=INTERMEDIATE):
        super().__init__()
        self.q_heads, self.kv_heads, self.head_dim = q_heads, kv_heads, head_dim
        self.rep = q_heads // kv_heads
        self.ln1 = RMSNorm(hidden)
        self.ln2 = RMSNorm(hidden)
        self.wq = nn.Linear(hidden, q_heads * head_dim, bias=True)
        self.wk = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.wv = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.wo = nn.Linear(q_heads * head_dim, hidden, bias=False)
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, x, past_kv=None):
        B, T, H = x.shape   # T == 1 (one token per forward)
        h = self.ln1(x)
        q = self.wq(h).view(B, T, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)   # thread the KV cache (grow seq)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)
        # GQA: repeat kv heads to match q heads
        kr = k.repeat_interleave(self.rep, dim=1)
        vr = v.repeat_interleave(self.rep, dim=1)
        # SDPA's fused kernels grid on (B x n_heads); B x 14 > 65535 overflows the CUDA
        # grid ("invalid configuration argument"), so chunk the batch to stay under it.
        # Chunking is a batch-dim split -> byte-identical result, just multiple launches
        # (which is realistic: a real large-batch verify would chunk the same way).
        max_bh = 32768
        chunk = max(1, max_bh // self.q_heads)
        if B <= chunk:
            attn = F.scaled_dot_product_attention(q, kr, vr, is_causal=False)
        else:
            parts = []
            for s in range(0, B, chunk):
                e = min(s + chunk, B)
                parts.append(F.scaled_dot_product_attention(
                    q[s:e], kr[s:e], vr[s:e], is_causal=False))
            attn = torch.cat(parts, dim=0)
        attn = attn.transpose(1, 2).reshape(B, T, self.q_heads * self.head_dim)
        x = x + self.wo(attn)
        h2 = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x, new_kv


class CleverDecodeHead(nn.Module):
    """The clever difference-min digit-extractor over the hidden's dim0 (whole value).
    logit_d = -|value - (d+0.5)|; argmax = floor. 10 shared candidates. Cheap (10-wide),
    so it does NOT change the shape-driven walltime -- it is the 'clever' part that makes
    each forward realise one digit-layer of depth."""
    def __init__(self, ncand=NCAND):
        super().__init__()
        self.register_buffer("cand", torch.arange(ncand).float() + 0.5)

    def forward(self, x):
        val = x[..., 0]                                  # whole value in dim0
        logits = -(val.unsqueeze(-1) - self.cand).abs()  # (B,1,10)
        return logits.argmax(-1)                          # (B,1) digit


class CleverStack(nn.Module):
    """A stack of L VanillaBlocks + the clever decode head. One `.step()` runs F forwards,
    threading the KV cache between them (the autoregressive digit loop). L*F ~= D."""
    def __init__(self, n_layers: int, dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([VanillaBlock() for _ in range(n_layers)])
        self.decode = CleverDecodeHead()
        self.n_layers = n_layers
        self.to(dtype)
        self.decode.cand = self.decode.cand.to(dtype)  # keep candidates in-dtype

    def one_forward(self, x, kv_caches):
        """One token forward through all L blocks; returns (x_out, digit, new_kv_caches)."""
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, kv_caches[i] if kv_caches else None)
            new_caches.append(kv)
        digit = self.decode(x)
        return x, digit, new_caches

    @torch.no_grad()
    def step(self, x0, F_forwards: int):
        """One VM step = F sequential forwards, threading the KV cache. Each forward
        appends one token (the emitted digit) -> KV grows by F over the step. Returns the
        final hidden (kept alive so nothing is optimised away)."""
        B = x0.shape[0]
        x = x0
        kv_caches = None
        for _ in range(F_forwards):
            x, digit, kv_caches = self.one_forward(x, kv_caches)
            # thread the running remainder: fold the emitted digit back into dim0 (the
            # autoregressive feedback). Cheap; keeps the loop data-dependent so the F
            # forwards cannot be fused/reordered away.
            x = x.clone()
            x[..., 0] = x[..., 0] - digit.to(x.dtype)
        return x, kv_caches


# --------------------------------------------------------------------------- #
# GPU-contention gate
# --------------------------------------------------------------------------- #
def _gpu_status():
    """Return list of (idx, util%, mem_used_MiB, mem_total_MiB) or None if no nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
    except Exception:
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def pick_and_gate_gpu(requested: str | None, poll_s: int, no_gate: bool):
    """Pick an idle GPU and poll until it is free (or return provisional=True).

    Returns (device_str, provisional, note)."""
    if not torch.cuda.is_available():
        return "cpu", False, "no CUDA -> analytic fallback"
    status = _gpu_status()
    if requested is not None:
        # honour explicit pin, but still flag contention
        idx = int(requested.split(":")[1]) if ":" in requested else 0
        if status:
            for gi, util, mu, mt in status:
                if gi == idx and util >= 30:
                    return requested, True, f"cuda:{idx} pinned but util={util}% -> PROVISIONAL"
        return requested, False, f"cuda:{idx} pinned"
    if no_gate or status is None:
        return "cuda:0", (status is None), "gate skipped" if no_gate else "no smi -> provisional"
    # choose the least-utilised GPU; poll for it to fall below 30% util
    deadline = time.time() + poll_s
    while True:
        status = _gpu_status() or []
        # prefer a genuinely idle card (util<30 and mem headroom)
        cands = sorted(status, key=lambda r: (r[1], r[2]))  # by util, then mem used
        if cands:
            gi, util, mu, mt = cands[0]
            if util < 30 and (mt - mu) > 4000:
                return f"cuda:{gi}", False, f"cuda:{gi} idle (util={util}%, free={mt-mu}MiB)"
        if time.time() >= deadline:
            gi = cands[0][0] if cands else 0
            u = cands[0][1] if cands else -1
            return f"cuda:{gi}", True, (f"cuda:{gi} still contended (util={u}%) after "
                                        f"{poll_s}s -> PROVISIONAL, needs clean re-run")
        time.sleep(5)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def time_step(stack: CleverStack, dev: str, batch: int, F_forwards: int, dtype,
              iters: int, warmup: int):
    """Time one VM step (= F forwards) at a given batch. Returns dict with ms/step etc."""
    device = torch.device(dev)
    x0 = torch.randn(batch, 1, HIDDEN, device=device, dtype=dtype)
    # keep dim0 a plausible 'whole value' so the decode does real argmax work
    x0[..., 0] = torch.rand(batch, 1, device=device, dtype=dtype) * 1e9
    for _ in range(warmup):
        stack.step(x0, F_forwards)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        stack.step(x0, F_forwards)
    _sync(dev)
    dt = time.perf_counter() - t0
    ms_per_step = dt / iters * 1e3
    steps_per_s = iters / dt
    return {
        "batch": batch, "F": F_forwards, "iters": iters,
        "ms_per_step": ms_per_step,
        "steps_per_s": steps_per_s,
        "lane_steps_per_s": steps_per_s * batch,   # batched: each lane is a VM step
    }


def sweep_config(label: str, n_layers: int, F_forwards: int, dev: str, dtype,
                 batches: list, iters: int, warmup: int):
    """Build the stack and sweep batch to saturation. Returns (rows, best)."""
    stack = CleverStack(n_layers, dtype=dtype).to(dev if dev != "cpu" else "cpu")
    stack.eval()
    rows, best = [], None
    for b in batches:
        try:
            r = time_step(stack, dev, b, F_forwards, dtype, iters, warmup)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "invalid configuration" in msg:
                if dev.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass
                rows.append({"batch": b, "F": F_forwards,
                             "oom": "out of memory" in msg,
                             "ceiling": "invalid configuration" in msg})
                break
            raise
        r["label"] = label
        r["n_layers"] = n_layers
        rows.append(r)
        # 'best' = max lane throughput (batched VM steps/s)
        if best is None or r["lane_steps_per_s"] > best["lane_steps_per_s"]:
            best = r
    del stack
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, best


# --------------------------------------------------------------------------- #
# Launch-overhead decomposition for the SHALLOW config.
# --------------------------------------------------------------------------- #
def launch_decompose(n_layers: int, F_forwards: int, dev: str, dtype, batch: int,
                     iters: int, warmup: int):
    """Decompose SHALLOW cost into per-forward-launch overhead vs actual compute.

    Method: time F forwards (the real step) and time 1 forward, both at the SAME batch.
    If cost were pure compute, F-forward time would be exactly F x the 1-forward time.
    Two overhead components are isolated:
      (1) KV re-read overhead: (eager F-step) - (F x eager 1-forward). The 1-forward step
          has an empty KV; the real F forwards re-read a KV cache that grows each forward,
          so this gap is the growing-KV re-read cost the shallow config pays across F.
      (2) fixed per-launch/kernel-dispatch overhead: measured by a launch-bound probe --
          time one forward at a TINY batch (compute ~ 0, so time ~ pure per-forward launch
          + dispatch floor). The shallow config pays this floor F x/step; the deep config
          pays it once/step. This is the robust, graph-free launch decomposition."""
    stack = CleverStack(n_layers, dtype=dtype).to(dev if dev != "cpu" else "cpu")
    stack.eval()

    # eager F-forward step
    r_F = time_step(stack, dev, batch, F_forwards, dtype, iters, warmup)
    # eager 1-forward step (same batch, F=1)
    r_1 = time_step(stack, dev, batch, 1, dtype, iters, warmup)

    out = {
        "n_layers": n_layers, "F": F_forwards, "batch": batch,
        "ms_step_F_eager": r_F["ms_per_step"],
        "ms_1forward_eager": r_1["ms_per_step"],
        "ms_F_x_1forward": r_1["ms_per_step"] * F_forwards,   # ideal linear (pure compute)
    }
    # (1) KV re-read overhead vs ideal linear scaling of a single forward
    out["kv_reread_overhead_ms"] = r_F["ms_per_step"] - out["ms_F_x_1forward"]

    # (2) fixed per-launch floor: one forward at a tiny (launch-bound) batch.
    tiny_batch = 8
    r_tiny = time_step(stack, dev, tiny_batch, 1, dtype, iters, warmup)
    out["ms_1forward_tiny_batch"] = r_tiny["ms_per_step"]
    out["launch_floor_per_forward_ms"] = r_tiny["ms_per_step"]   # ~= pure launch/dispatch
    out["launch_floor_per_step_ms"] = r_tiny["ms_per_step"] * F_forwards
    # what fraction of the shallow step (at the timing batch) is pure launch floor
    out["launch_floor_frac_of_F_step"] = (out["launch_floor_per_step_ms"]
                                          / out["ms_step_F_eager"]) if out["ms_step_F_eager"] else float("nan")
    # the deep config pays the floor ONCE/step -> per-step launch tax the shallow pays extra
    out["extra_launch_tax_vs_deep_ms"] = r_tiny["ms_per_step"] * (F_forwards - 1)

    del stack
    if dev.startswith("cuda"):
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    return out


# --------------------------------------------------------------------------- #
# KV-cache size model.
# --------------------------------------------------------------------------- #
def kv_bytes(n_layers: int, seq_len: int, batch: int, bytes_per_elem: int):
    """KV_bytes = 2 (K+V) x n_layers x kv_heads x head_dim x seq_len x batch x bytes.
    GQA: KV projection is kv_heads (2) not q_heads (14)."""
    return 2 * n_layers * KV_HEADS * HEAD_DIM * seq_len * batch * bytes_per_elem


def kv_comparison(D: int, deep_layers: int, shallow_layers: int, F_forwards: int,
                  doom_steps: int, batch: int, window: int = 4096):
    """KV size for DEEP (1 token/step) vs SHALLOW (F tokens/step) at the doom frame.

    DEEP advances the sequence by 1 token/step -> seq_len = doom_steps (per lane).
    SHALLOW advances by F tokens/step -> seq_len = doom_steps x F (F x more tokens).

    KV_bytes = 2 x n_layers x kv_heads x head_dim x seq_len x batch x bytes. The
    SHALLOW/DEEP ratio folds TWO opposing levers:
        layers:  shallow_layers / deep_layers   (< 1, HELPS -- fewer layers to cache)
        tokens:  F                              (> 1, HURTS -- F x more tokens/step)
    When L x F == D exactly, these cancel (ratio == 1.0) at the FULL-frame seq length.
    But the memory PRESSURE that actually bites is the PER-STEP KV GROWTH (what an
    eviction window must absorb each step) -- there SHALLOW grows F x faster per step,
    partly offset by its fewer layers -> net (F x shallow_layers/deep_layers). We report
    full-frame, per-step growth, AND a bounded-window resident size."""
    def rows(bpe, name):
        deep_seq = doom_steps
        shal_seq = doom_steps * F_forwards
        return {
            "precision": name, "bytes_per_elem": bpe,
            "deep": {"n_layers": deep_layers, "tokens_per_step": 1,
                     "seq_len_full_frame": deep_seq,
                     "kv_bytes_full_frame": kv_bytes(deep_layers, deep_seq, batch, bpe),
                     "kv_bytes_per_step_growth": kv_bytes(deep_layers, 1, batch, bpe),
                     "kv_bytes_window": kv_bytes(deep_layers, window, batch, bpe)},
            "shallow": {"n_layers": shallow_layers, "tokens_per_step": F_forwards,
                        "seq_len_full_frame": shal_seq,
                        "kv_bytes_full_frame": kv_bytes(shallow_layers, shal_seq, batch, bpe),
                        "kv_bytes_per_step_growth": kv_bytes(shallow_layers, F_forwards, batch, bpe),
                        "kv_bytes_window": kv_bytes(shallow_layers, window, batch, bpe)},
        }
    fp32 = rows(4, "fp32")
    bf16 = rows(2, "bf16")
    for r in (fp32, bf16):
        d, s = r["deep"], r["shallow"]
        r["shallow_over_deep_full_frame"] = (s["kv_bytes_full_frame"]
                                             / d["kv_bytes_full_frame"]) if d["kv_bytes_full_frame"] else float("nan")
        r["shallow_over_deep_per_step_growth"] = (s["kv_bytes_per_step_growth"]
                                                  / d["kv_bytes_per_step_growth"]) if d["kv_bytes_per_step_growth"] else float("nan")
        r["shallow_over_deep_window"] = (s["kv_bytes_window"]
                                         / d["kv_bytes_window"]) if d["kv_bytes_window"] else float("nan")
        r["layer_ratio_shallow_over_deep"] = shallow_layers / deep_layers
        r["token_ratio_shallow_over_deep"] = F_forwards
    return {"D": D, "deep_layers": deep_layers, "shallow_layers": shallow_layers,
            "F": F_forwards, "doom_steps": doom_steps, "batch_per_lane": batch,
            "window": window, "fp32": fp32, "bf16": bf16}


# --------------------------------------------------------------------------- #
# fps helpers
# --------------------------------------------------------------------------- #
def fps_from_best(best: dict, doom_steps: int):
    """fps = lane_steps_per_s / steps_per_frame. Batched: each lane finishes one VM step,
    so the frame's steps are processed at lane_steps_per_s (the spec-decode verify model,
    docs/CLEVER_DOOM_REALTIME.md §3)."""
    if best is None:
        return None
    return best["lane_steps_per_s"] / doom_steps


def mark(fps):
    if fps is None:
        return ""
    tags = []
    if fps >= 60:
        tags.append(">=60fps")
    elif fps >= 30:
        tags.append(">=30fps")
    else:
        tags.append("<30fps")
    return " ".join(tags)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="pin e.g. cuda:1 (else auto-pick idle)")
    ap.add_argument("--D", type=int, default=48, help="effective digit-extract depth")
    ap.add_argument("--shallow-layers", type=int, default=3, help="L for the SHALLOW config")
    ap.add_argument("--batches", default=None, help="comma list; default swept to saturation")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--poll", type=int, default=180, help="GPU-contention poll seconds")
    ap.add_argument("--no-gpu-gate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    D = args.D
    deep_layers = D                       # DEEP: one layer per digit
    shallow_layers = args.shallow_layers  # SHALLOW: fits stock 24
    F_forwards = math.ceil(D / shallow_layers)   # forwards/step to reach depth D
    assert shallow_layers <= STOCK_MAX_LAYERS, "shallow must fit a stock 24-layer budget"
    # effective depths
    deep_eff = deep_layers * 1
    shal_eff = shallow_layers * F_forwards

    dev, provisional, note = pick_and_gate_gpu(args.device, args.poll, args.no_gpu_gate)
    is_cuda = dev.startswith("cuda")

    if args.batches:
        batches = [int(b) for b in args.batches.split(",")]
    elif args.quick:
        batches = [256, 1024, 4096]
    elif is_cuda:
        batches = [256, 1024, 4096, 16384]   # saturates by ~4096 (chunked attn past B*H grid)
    else:
        batches = [64, 256, 1024]

    dtypes = [("fp32", torch.float32)]
    if is_cuda:
        dtypes.append(("bf16", torch.bfloat16))

    print("=" * 96)
    print("SHALLOW (K-forwards, stock-vanilla-fitting) vs DEEP (1-forward) -- realtime measure")
    print("=" * 96)
    print(f"device={dev}  ({note})")
    if provisional:
        print("  *** PROVISIONAL: measured under GPU contention -- numbers need a clean re-run ***")
    if is_cuda:
        idx = int(dev.split(':')[1]) if ':' in dev else 0
        print(f"  GPU: {torch.cuda.get_device_name(idx)}")
    print(f"effective depth D={D}")
    print(f"  DEEP   : n_layers={deep_layers}, F=1  (eff depth {deep_eff}) "
          f"-- {'exceeds' if deep_layers > STOCK_MAX_LAYERS else 'fits'} stock {STOCK_MAX_LAYERS}-layer budget")
    print(f"  SHALLOW: n_layers={shallow_layers}, F={F_forwards}  (eff depth {shal_eff}) "
          f"-- fits stock {STOCK_MAX_LAYERS}-layer budget (VANILLA)")
    print(f"doom render-reduced steps/frame = {DOOM_RENDER_STEPS} (MEASURED)")
    print()

    all_results = {"config": {"D": D, "deep_layers": deep_layers,
                              "shallow_layers": shallow_layers, "F": F_forwards,
                              "hidden": HIDDEN, "doom_render_steps": DOOM_RENDER_STEPS,
                              "device": dev, "provisional": provisional, "note": note},
                   "sweeps": {}, "launch_decompose": {}, "fps": {}, "kv": None}

    # ---- sweeps ----
    for dtname, dt in dtypes:
        print(f"########## dtype = {dtname} ##########")
        for label, nl, Ff in (("DEEP (1-fwd)", deep_layers, 1),
                              ("SHALLOW (K-fwd)", shallow_layers, F_forwards)):
            print(f"--- {label}: n_layers={nl}, F={Ff} [{dtname}] ---")
            rows, best = sweep_config(label, nl, Ff, dev, dt, batches, args.iters, args.warmup)
            all_results["sweeps"][f"{dtname}:{label}"] = rows
            for r in rows:
                if r.get("oom"):
                    print(f"  batch={r['batch']:>7d}  OOM")
                    continue
                print(f"  batch={r['batch']:>7d}  {r['ms_per_step']:9.3f} ms/step  "
                      f"{r['lane_steps_per_s']/1e6:8.3f} Msteps/s (batched)")
            if best:
                fps = fps_from_best(best, DOOM_RENDER_STEPS)
                fps_raw = fps_from_best(best, DOOM_RAW_STEPS)
                all_results["fps"][f"{dtname}:{label}"] = {
                    "best_batch": best["batch"], "best_ms_step": best["ms_per_step"],
                    "best_lane_steps_per_s": best["lane_steps_per_s"],
                    "fps_render": fps, "fps_raw": fps_raw}
                print(f"  BEST: {best['ms_per_step']:.3f} ms/step @ batch={best['batch']}  "
                      f"-> {best['lane_steps_per_s']/1e6:.3f} Msteps/s")
                print(f"        RENDER-frame fps = {fps:8.2f}  [{mark(fps)}]   "
                      f"(RAW fps = {fps_raw:.2f})")
            print()

    # ---- deep-vs-shallow fps gap ----
    print("########## DEEP vs SHALLOW fps gap (the crux) ##########")
    for dtname, _ in dtypes:
        dk = f"{dtname}:DEEP (1-fwd)"
        sk = f"{dtname}:SHALLOW (K-fwd)"
        if dk in all_results["fps"] and sk in all_results["fps"]:
            fd = all_results["fps"][dk]["fps_render"]
            fs = all_results["fps"][sk]["fps_render"]
            gap = fd / fs if fs else float("nan")
            all_results["fps"][f"{dtname}:deep_over_shallow"] = gap
            print(f"  [{dtname}] DEEP {fd:7.2f} fps  vs  SHALLOW {fs:7.2f} fps  "
                  f"-> DEEP is {gap:.2f}x faster; SHALLOW realtime? "
                  f"{'YES >=30' if fs>=30 else 'NO'}"
                  f"{' (>=60)' if fs>=60 else ''}")
    print()

    # ---- launch decomposition (shallow) ----
    print("########## SHALLOW launch-overhead decomposition ##########")
    decomp_batch = 4096 if is_cuda else 256
    for dtname, dt in dtypes:
        d = launch_decompose(shallow_layers, F_forwards, dev, dt, decomp_batch,
                             max(args.iters // 2, 10), args.warmup)
        all_results["launch_decompose"][dtname] = d
        print(f"--- [{dtname}] SHALLOW L={shallow_layers} F={F_forwards} batch={decomp_batch} ---")
        print(f"  1-forward (eager, big batch): {d['ms_1forward_eager']:.4f} ms")
        print(f"  F x 1-forward (ideal linear): {d['ms_F_x_1forward']:.4f} ms")
        print(f"  F-forward step (eager)      : {d['ms_step_F_eager']:.4f} ms")
        print(f"  (1) KV re-read overhead     : {d['kv_reread_overhead_ms']:+.4f} ms "
              f"(F-step minus F x 1-fwd; growing-KV re-reads across F forwards)")
        print(f"  1-forward @ tiny batch=8    : {d['ms_1forward_tiny_batch']:.4f} ms "
              f"(launch/dispatch floor, compute~0)")
        print(f"  (2) launch floor / step     : {d['launch_floor_per_step_ms']:.4f} ms "
              f"= F x {d['launch_floor_per_forward_ms']:.4f} ms  "
              f"({d['launch_floor_frac_of_F_step']*100:.1f}% of the F-step)")
        print(f"      extra launch tax vs DEEP : {d['extra_launch_tax_vs_deep_ms']:+.4f} ms/step "
              f"(shallow pays the floor F={F_forwards}x, deep pays it 1x)")
        print()

    # ---- KV comparison ----
    print("########## KV cache: SHALLOW (F x tokens/step) vs DEEP ##########")
    kv = kv_comparison(D, deep_layers, shallow_layers, F_forwards,
                       DOOM_RENDER_STEPS, batch=1)
    all_results["kv"] = kv
    for name in ("fp32", "bf16"):
        r = kv[name]
        dd, ss = r["deep"], r["shallow"]
        print(f"--- [{name}] per lane ---")
        print(f"  full frame:  DEEP seq={dd['seq_len_full_frame']:>9d} "
              f"KV={dd['kv_bytes_full_frame']/1e9:7.3f}GB   "
              f"SHALLOW seq={ss['seq_len_full_frame']:>9d} "
              f"KV={ss['kv_bytes_full_frame']/1e9:7.3f}GB   "
              f"ratio={r['shallow_over_deep_full_frame']:.3f}x")
        print(f"  per-step growth: DEEP {dd['kv_bytes_per_step_growth']/1e3:8.3f}KB   "
              f"SHALLOW {ss['kv_bytes_per_step_growth']/1e3:8.3f}KB   "
              f"ratio={r['shallow_over_deep_per_step_growth']:.3f}x "
              f"(F={r['token_ratio_shallow_over_deep']}x tokens x "
              f"{r['layer_ratio_shallow_over_deep']:.3f}x layers)")
        print(f"  window={kv['window']}:  DEEP {dd['kv_bytes_window']/1e6:8.3f}MB   "
              f"SHALLOW {ss['kv_bytes_window']/1e6:8.3f}MB   "
              f"ratio={r['shallow_over_deep_window']:.3f}x (same seq window -> layer ratio only)")
    print()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"wrote {args.json}")

    print("=" * 96)
    print("NOTE: shapes are Qwen2.5-0.5B (hidden 896, GQA 14/2, SwiGLU 4864) -- SHALLOW"
          " fits stock 24 layers (vanilla); DEEP does not. Arithmetic byte-exactness is"
          " proven separately in clever_minparam_alu.py; here SHAPE drives walltime.")
    if provisional:
        print("*** These numbers are PROVISIONAL (GPU contention) -- re-run on an idle GPU. ***")
    print("=" * 96)


if __name__ == "__main__":
    main()
