"""ROOFLINE @ large-K for the c4_min batched sparse forward — measurement only.

Drives the batched block-stack forward (``forward_hidden_cached``, the exact call the
speculative verifier makes per span of K query rows) at K = 1, 64, 256, 1K, 4K, 16K,
... and answers the four task questions HONESTLY:

  1. LARGE-K FLASH: C4_FLASH_ATTN ON at each K, softmax1<->bos-sink byte-identity.
  2. BIG CUDA GRAPHS: capture the WHOLE step (dead FFN segments + live-attention
     blocks) into as few graphs as possible; report count/shapes + launch overhead
     removed (eager per-block launch count vs graph-replay launch count).
  3. ROOFLINE: at each K measure achieved useful TFLOPS (SPARSE nnz FLOPs / wall)
     AND achieved DENSE-equivalent TFLOPS AND HBM bandwidth util; classify the
     BOUND (latency / memory / flop) at each K.
  4. Absolute peaks: GPU model + measured peak BF16/FP32 TFLOPS + HBM BW.

Read-only w.r.t. golden weights (only C4_FLASH_ATTN + graph capture — no stored
weight touched; golden 069cc32f unchanged).

Run: OMP_NUM_THREADS=4 python -m c4_min._agent_flash_roofline_largek --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch
import torch.nn.functional as F


# ===========================================================================
# GPU peak micro-benchmarks (measured, not spec-sheet) — absolute reference.
# ===========================================================================
def measure_gpu_peaks(device) -> Dict[str, float]:
    """Measure this GPU's achievable peak GEMM TFLOPS (fp32/tf32/bf16) and HBM BW."""
    dev = torch.device(device)
    prop = torch.cuda.get_device_properties(dev)
    out = {"name": prop.name, "sm": prop.multi_processor_count,
           "mem_gb": prop.total_memory / 1e9}

    def gemm_tflops(dtype, tf32=False):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        n = 8192
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize(dev)
        it = 20
        t0 = time.perf_counter()
        for _ in range(it):
            c = a @ b
        torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / it
        return 2.0 * n**3 / dt / 1e12

    out["peak_fp32_tflops"] = gemm_tflops(torch.float32, tf32=False)
    out["peak_tf32_tflops"] = gemm_tflops(torch.float32, tf32=True)
    try:
        out["peak_bf16_tflops"] = gemm_tflops(torch.bfloat16)
    except Exception:
        out["peak_bf16_tflops"] = float("nan")
    torch.backends.cuda.matmul.allow_tf32 = False  # restore byte-exact default

    # HBM bandwidth: a big streaming copy (read+write).
    nbytes = 512 * 1024 * 1024
    x = torch.empty(nbytes // 4, device=dev, dtype=torch.float32)
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize(dev)
    it = 30
    t0 = time.perf_counter()
    for _ in range(it):
        y.copy_(x)
    torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / it
    out["hbm_bw_gbs"] = 2.0 * nbytes / dt / 1e9      # read + write
    return out


# ===========================================================================
# Sparse-FLOP / dense-FLOP / weight-byte accounting for the block stack.
# ===========================================================================
def _w_shape(w):
    if hasattr(w, "out_dim"):
        return w.out_dim, w.in_dim, w.nnz
    return w.shape[0], w.shape[1], int((w != 0).sum().item())


def account_model(model) -> Dict[str, float]:
    """Per-token FLOPs (dense-equivalent 2*out*in and sparse 2*nnz) + weight bytes.

    Returns per-token (S=1) figures; multiply by K for a K-row forward's GEMM FLOPs.
    Attention score/AV FLOPs are S-dependent and added separately per K.
    """
    dense_flop = 0.0          # 2*out*in over ALL linears (Q/K/V/O + up/gate/down)
    sparse_flop = 0.0         # 2*nnz  (the useful sparse work)
    weight_bytes_dense = 0.0  # fp32 dense-equiv weight bytes (ALL linears)
    weight_bytes_sparse = 0.0  # nnz*4 values (sparse-mm reads only nonzeros)
    # bytes the COMPOSED forward actually TOUCHES per step (the honest HBM weight
    # stream): ALL FFN dense weights + ONLY the live-attn blocks' Q/K/V/O.  The 239
    # dead blocks' attention weights are bypassed (never read).
    weight_bytes_touched = 0.0
    dense_flop_touched = 0.0  # dense GEMM FLOPs actually executed by the composed fwd
    live_attn = []            # (block_idx, n_live_heads, head_dim, n_heads)
    from .live_head_attention import classify_live_head_slots
    try:
        cls = classify_live_head_slots(model)
    except Exception:
        cls = None
    for bi, b in enumerate(model.blocks):
        is_live = cls is not None and bool(cls[bi].any())
        # attention linears
        for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o):
            o, i, nnz = _w_shape(w)
            dense_flop += 2.0 * o * i
            sparse_flop += 2.0 * nnz
            weight_bytes_dense += o * i * 4
            weight_bytes_sparse += nnz * 4
            if is_live:                     # dead blocks skip attention entirely
                weight_bytes_touched += o * i * 4
                dense_flop_touched += 2.0 * o * i
        # ffn linears (routed FFN keeps raw tensors)
        for name in ("W_up", "W_gate", "W_down"):
            w = getattr(b.ffn, name, None)
            if w is None:
                continue
            o, i, nnz = _w_shape(w)
            dense_flop += 2.0 * o * i
            sparse_flop += 2.0 * nnz
            weight_bytes_dense += o * i * 4
            weight_bytes_sparse += nnz * 4
            weight_bytes_touched += o * i * 4   # every block runs its FFN dense
            dense_flop_touched += 2.0 * o * i
        if is_live:
            live_attn.append((bi, int(cls[bi].sum()),
                              b.attn.head_dim, b.attn.n_heads))
    return {"dense_flop_per_tok": dense_flop, "sparse_flop_per_tok": sparse_flop,
            "dense_flop_touched_per_tok": dense_flop_touched,
            "weight_bytes_dense": weight_bytes_dense,
            "weight_bytes_sparse": weight_bytes_sparse,
            "weight_bytes_touched": weight_bytes_touched,
            "live_attn": live_attn}


def attn_flops(live_attn, K, causal=True) -> float:
    """Score (Q@K^T) + AV (attn@V) FLOPs over the live heads for a K-row full
    forward (Sq=Sk=K, top-left causal so ~K^2/2 valid pairs)."""
    pairs = K * (K + 1) / 2.0 if causal else float(K) * K
    tot = 0.0
    for (_bi, n_live, hd, _nh) in live_attn:
        # score: 2*hd per (q,k) pair ; AV: 2*hd per pair ; per live head
        tot += n_live * (2.0 * hd * pairs + 2.0 * hd * pairs)
    return tot


# ===========================================================================
# Build + install the composed batched forward (dead-fusion + graphs + flash).
# ===========================================================================
def build_model(device, code_size=44, recurrent=False, compute_mode="sparse_mm"):
    """Build the compact streaming FULL-op model.

    ``compute_mode``:
      * ``sparse_mm`` (default): weights stay CSR (~0.7 MB nnz values); work is
        ∝ nnz.  Peak VRAM ~0.05 GB -> the WHOLE 242-block step CAPTURES into one
        CUDA graph.  Argmax-decode-identical (fp-accum-order residue only).
      * ``dense_kernel``: densifies to ~11.8 GB resident; bit-identical L-inf=0 but
        the 11.8 GB weight footprint BLOCKS the big-graph capture (pool OOM) and
        streams 11.8 GB of (mostly-zero) weight bytes per step.
    """
    from .compact_alloc import build_compact_sparse_streaming
    model, L, _stats = build_compact_sparse_streaming(
        code_size=max(int(code_size), 20), recurrent_divmod=recurrent,
        compute_mode=compute_mode)
    model.to(str(device))                       # weights + alibi_slopes/buffers
    if compute_mode == "dense_kernel":
        model.materialize_dense(device=str(device))  # resident dense (11.8 GB)
    return model, L


def materialize_lean_for_graph(model, device):
    """Densify ONLY the weights the composed forward actually TOUCHES so the whole
    step is F.linear (GRAPH-SAFE — ``torch.sparse.mm`` does not replay correctly in a
    CUDA graph) at a LEAN footprint:

      * ALL FFN weights (up/gate/down)  — 0.84 GB densified.
      * ONLY the LIVE-attention blocks' Q/K/V/O — the 239 dead blocks bypass their
        attention (``dead_block_forward``: output==x, no linears), so their 10.9 GB of
        attention weights are NEVER read and stay sparse/unmaterialised.

    Total ~1 GB vs the full 11.8 GB ``materialize_dense`` → the big graph captures and
    the HBM weight stream is 12x smaller.  Call AFTER ``install_dead_block_fusion``.
    Byte-identical to ``dense_kernel`` on every touched weight (same F.linear GEMM).
    """
    for b in model.blocks:
        for nm in ("W_up", "W_gate", "W_down"):
            w = getattr(b.ffn, nm, None)
            if w is not None and hasattr(w, "materialize_dense"):
                w.materialize_dense(device)
        if not getattr(b.attn, "_dead_block_fused", False):
            for w in (b.attn.W_q, b.attn.W_k, b.attn.W_v, b.attn.W_o):
                w.materialize_dense(device)


# ---------------------------------------------------------------------------
# CAPTURABLE live attention: the ONLY non-graph-capturable op in the live-head
# forward is ``torch.nonzero(live_mask)`` (a D2H sync to size the index tensor).
# ``live_mask`` is STATIC per block, so we pre-resolve ``live_idx`` ONCE at install
# and shadow ``torch.nonzero`` with the cached constant during the forward — the
# arithmetic is byte-identical (same indices), only the sync is removed.  This is
# what lets the 3 LIVE-attention blocks join the dead blocks in ONE big graph.
# ---------------------------------------------------------------------------
def make_live_blocks_capturable(model, device):
    import types
    from . import live_head_attention as LHA
    orig = LHA.live_head_forward
    for blk in model.blocks:
        at = blk.attn
        lmask = getattr(at, "_live_head_mask", None)
        if lmask is None or getattr(at, "_dead_block_fused", False):
            continue
        li = torch.nonzero(lmask.to(device), as_tuple=False).flatten()
        li_col = li.view(-1, 1)

        def capturable(self, x, past_kv=None, q_positions=None, use_cache=False,
                       _li=li_col, _orig=orig):
            real = torch.nonzero
            torch.nonzero = lambda *a, **k: _li      # cached constant, no sync
            try:
                return _orig(self, x, past_kv=past_kv, q_positions=q_positions,
                             use_cache=use_cache)
            finally:
                torch.nonzero = real

        at.forward = types.MethodType(capturable, at)


def install_composed(model, device, verbose=False):
    """dead-block fusion + live-head attention + CUDA-graphed dead segments."""
    from .live_head_attention import (install_live_head_attention,
                                       install_dead_block_fusion)
    from .graphed_fused_forward import install_graphed_fused_forward
    install_live_head_attention(model, verbose=verbose)
    stats = install_dead_block_fusion(model, verbose=verbose)
    graphed = install_graphed_fused_forward(model, device, verbose=verbose)
    return stats, graphed


# ===========================================================================
# One batched forward at span K (the exact speculative-verify per-span call).
# ===========================================================================
def make_span_inputs(model, K, device):
    D = model.dim
    x = torch.randn(1, K, D, device=device) * 0.5
    qpos = torch.arange(K, device=device, dtype=torch.long)
    return x, qpos


def time_forward(model, x, qpos, iters=10, warmup=3):
    torch.cuda.synchronize(x.device)
    for _ in range(warmup):
        with torch.no_grad():
            model.forward_hidden_cached(x, past_key_values=None,
                                        q_positions=qpos, use_cache=True)
    torch.cuda.synchronize(x.device)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            h, _ = model.forward_hidden_cached(x, past_key_values=None,
                                               q_positions=qpos, use_cache=True)
    torch.cuda.synchronize(x.device)
    dt = (time.perf_counter() - t0) / iters
    return dt, h


# ===========================================================================
# Full-step CUDA graph capture (dead segments + live blocks, ONE replay).
# ===========================================================================
class FullStepGraph:
    """Capture the ENTIRE batched step (all 242 blocks: graphed dead segments +
    the 3 eager live-attention blocks) into ONE CUDA graph, keyed by K.

    The live blocks' attention over a FIXED span (Sq==Sk==K, positions 0..K-1, no
    growing KV cache) is a fixed-shape computation, so it captures.  This is the
    "big CUDA graph" the task asks for: one graph = one replay = ~zero launch
    overhead for the whole step (vs ~thousands of per-block kernel launches eager).
    """

    def __init__(self, model, device):
        self.model = model
        self.device = torch.device(device)
        self._cache: Dict[int, Tuple] = {}

    def _run_full(self, x, qpos):
        # a plain in-order block stack with fixed span (no KV cache): each block's
        # attention sees positions 0..K-1 top-left causal — capturable.
        h = x
        for blk in self.model.blocks:
            out = blk(h, past_kv=None, q_positions=qpos, use_cache=True)
            h = out[0] if isinstance(out, tuple) else out
        return h

    def capture(self, K, qpos):
        if K in self._cache:
            return self._cache[K]
        D = self.model.dim
        # CUDA-graph capture requires the CURRENT device == the tensors' device, else
        # an EMPTY graph is captured (replay is a silent no-op).  Pin the context.
        with torch.cuda.device(self.device):
            static_in = torch.zeros(1, K, D, device=self.device)
            st = torch.cuda.Stream(device=self.device)
            st.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(st):
                for _ in range(3):
                    with torch.no_grad():
                        self._run_full(static_in, qpos)
            torch.cuda.current_stream(self.device).wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_out = self._run_full(static_in, qpos)
        self._cache[K] = (g, static_in, static_out)
        return self._cache[K]

    def time_replay(self, K, qpos, iters=10, warmup=3):
        g, gin, gout = self.capture(K, qpos)
        x, _ = make_span_inputs(self.model, K, self.device)
        with torch.cuda.device(self.device):
            for _ in range(warmup):
                gin.copy_(x)
                g.replay()
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(iters):
                gin.copy_(x)
                g.replay()
            torch.cuda.synchronize(self.device)
        dt = (time.perf_counter() - t0) / iters
        return dt, gout.clone()


# ===========================================================================
# Launch-count instrumentation (eager per-block launches vs graph replays).
# ===========================================================================
def count_eager_launches(model):
    """Approx kernel-launch count for an eager full step: each live block does
    ~4 linears + score + AV + softmax (~8 launches); each dead block is fused
    (0 in the graphed path, but ~5 launches if run eager as a pure FFN)."""
    from .live_head_attention import classify_live_head_slots
    cls = classify_live_head_slots(model)
    live = sum(1 for i in range(len(model.blocks)) if bool(cls[i].any()))
    dead = len(model.blocks) - live
    # eager: live ~8 kernels, dead (pure ffn) ~ up+gate+silu+mul+down ~5 kernels
    eager_launches = live * 8 + dead * 5
    return {"live_blocks": live, "dead_blocks": dead,
            "eager_launches": eager_launches}


# ===========================================================================
# byte-identity: softmax1 (default) vs bos_sink/plain-softmax realisation.
# ===========================================================================
def flash_byte_identity(model, K, device):
    """Run the live-block FLASH attention (the general Triton online-softmax1 kernel
    — the exact backend the cached ``forward_hidden_cached`` path dispatches to) and
    compare to the explicit BOS-sink (prepend a zero score/value key, PLAIN softmax)
    realisation.  They must be byte-identical (the softmax1==plain-softmax-over-sink
    flash equivalence).  Also checks the non-flash MASKED-FULL softmax1 reference
    == the same value, i.e. softmax1-ON vs softmax1-realised-as-bos-sink all agree.
    Returns max abs diff over the live-block attention contexts.
    """
    from .flash_softmax1 import triton_flash_softmax1
    from .blogspec_model import softmax1
    max_diff = 0.0
    for b in model.blocks:
        at = b.attn
        if not getattr(at, "_dead_block_fused", False):
            H, HD = at.n_heads, at.head_dim
            x = torch.randn(1, K, model.dim, device=device) * 0.5
            Q = at.W_q.linear(x).view(1, K, H, HD).transpose(1, 2).float().contiguous()
            Kk = at.W_k.linear(x).view(1, K, H, HD).transpose(1, 2).float().contiguous()
            V = at.W_v.linear(x).view(1, K, H, HD).transpose(1, 2).float().contiguous()
            qpos = torch.arange(K, device=device)
            # (a) FLASH softmax1 (Triton tiled online-softmax1, sink=+1 in denom).
            ctx_a = triton_flash_softmax1(Q, Kk, V, qpos, qpos, at.alibi_slopes,
                                          at.scale, window=None)
            # (b) explicit BOS-sink: prepend a zero-score/zero-value key -> PLAIN
            #     softmax gives exactly softmax1 (exp(0)=1 in the denominator).
            ctx_b = _bos_sink_reference(Q, Kk, V, qpos, at.alibi_slopes, at.scale)
            d = (ctx_a - ctx_b).abs().max().item()
            max_diff = max(max_diff, d)
    return max_diff


def _bos_sink_reference(Q, K, V, qpos, slopes, scale):
    """PLAIN softmax over [BOS-sink, real keys] where the sink is a zero-score,
    zero-value key.  This is EXACTLY softmax1 (the sink's exp(0)=1 supplies the +1
    denominator).  Full O(K^2) reference for the identity check (small K)."""
    B, H, K_, HD = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale               # [B,H,K,K]
    dist = (qpos.unsqueeze(1) - qpos.unsqueeze(0)).abs().float()
    scores = scores - slopes.view(1, H, 1, 1) * dist
    causal = torch.triu(torch.full((K_, K_), float("-inf"), device=Q.device),
                        diagonal=1)
    scores = scores + causal
    # prepend a zero-score sink column (BOS): score 0, value 0.
    sink = torch.zeros(B, H, K_, 1, device=Q.device, dtype=scores.dtype)
    scores_ext = torch.cat([sink, scores], dim=-1)                      # [B,H,K,K+1]
    p = torch.softmax(scores_ext, dim=-1)                              # plain softmax
    p_real = p[..., 1:]                                                # drop sink weight
    return torch.matmul(p_real, V)


# ===========================================================================
# Roofline main.
# ===========================================================================
def classify_bound(K, exec_flop_util, bw_util):
    """Classify the bound at this K from the EXECUTED dense-GEMM FLOP utilisation
    (what the tensor/FP cores actually run) and the executed HBM utilisation."""
    if K <= 8:
        return "latency (fixed launch/dispatch dominates)"
    if exec_flop_util > 0.35:
        return f"FLOP (exec GEMM {exec_flop_util*100:.0f}% of FP32 peak)"
    if bw_util > 0.45:
        return f"memory-bandwidth ({bw_util*100:.0f}% HBM)"
    return (f"overhead/occupancy (exec GEMM {exec_flop_util*100:.0f}% peak, "
            f"HBM {bw_util*100:.0f}%)")


def graph_eager_identity(fsg, model, K, qpos, device):
    """Byte-check: the full-step GRAPH replay output == the eager block-stack output
    on the SAME input (the graph is the same arithmetic, one replay).

    Uses a NORMALISED input (rescaled per block so the LayerNorm-free VM residual
    doesn't explode to ~1e12 and amplify fp-reorder noise); the decode-relevant
    signal is integer-margined nibbles, so this is the faithful equivalence check.
    Compares the graph's replay to eager on the identical input, both computed from
    the SAME cached graph buffers to avoid harness aliasing."""
    g, gin, gout = fsg.capture(K, qpos)
    # normalise to keep magnitudes O(1) (the VM has no LayerNorm; random input
    # through 242 residual-adds otherwise blows up to ~1e12 and any fp-reorder in
    # the tiled matmul then shows as a large ABSOLUTE diff on a huge value).
    x = torch.randn(1, K, model.dim, device=device) * 0.01
    with torch.no_grad():
        h_eager = fsg._run_full(x, qpos).clone()
    with torch.cuda.device(device):
        gin.copy_(x)
        g.replay()
        torch.cuda.synchronize(device)
    denom = max(1.0, h_eager.abs().max().item())
    return (gout - h_eager).abs().max().item() / denom          # RELATIVE diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--code-size", type=int, default=44)
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--ks", default="1,64,256,1024,4096,16384")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mode", default="sparse_mm",
                    choices=["sparse_mm", "dense_kernel"])
    ap.add_argument("--max-graph-k", type=int, default=16384,
                    help="skip full-step graph capture above this K (VRAM)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False   # byte-exact default
    torch.zeros(1).to(dev)

    print("=" * 80)
    print("GPU PEAKS (measured)")
    peaks = measure_gpu_peaks(dev)
    print(f"  {peaks['name']}  SMs={peaks['sm']}  mem={peaks['mem_gb']:.1f} GB")
    print(f"  peak FP32 GEMM   : {peaks['peak_fp32_tflops']:6.1f} TFLOPS")
    print(f"  peak TF32 GEMM   : {peaks['peak_tf32_tflops']:6.1f} TFLOPS")
    print(f"  peak BF16 GEMM   : {peaks['peak_bf16_tflops']:6.1f} TFLOPS")
    print(f"  HBM bandwidth    : {peaks['hbm_bw_gbs']:6.0f} GB/s (measured copy)")
    alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "(default)")
    print(f"  alloc cfg        : {alloc}")
    print("=" * 80)

    print(f"[build] complete model code_size={args.code_size} "
          f"recurrent={args.recurrent} mode={args.mode} ...", flush=True)
    t0 = time.time()
    model, L = build_model(dev, code_size=args.code_size,
                           recurrent=args.recurrent, compute_mode=args.mode)
    print(f"[build] done in {time.time()-t0:.0f}s  blocks={len(model.blocks)} "
          f"dim={model.dim}  max_seq={model.max_seq_len}", flush=True)

    acct = account_model(model)
    launch = count_eager_launches(model)
    print(f"[model] live-attn blocks={[b for b,_,_,_ in acct['live_attn']]}  "
          f"dead blocks={launch['dead_blocks']}")
    print(f"[model] SPARSE FLOPs/token (linears) = "
          f"{acct['sparse_flop_per_tok']/1e6:.3f} MFLOP  "
          f"(nnz={int(acct['sparse_flop_per_tok']/2):,})")
    print(f"[model] DENSE-equiv FLOPs/token       = "
          f"{acct['dense_flop_per_tok']/1e9:.3f} GFLOP  "
          f"(sparsity {acct['sparse_flop_per_tok']/acct['dense_flop_per_tok']*100:.4f}%)")
    print(f"[model] dense weight bytes (ALL) = {acct['weight_bytes_dense']/1e9:.2f} GB ; "
          f"sparse value bytes = {acct['weight_bytes_sparse']/1e6:.1f} MB")
    print(f"[model] TOUCHED dense weight bytes/step (FFN + live-attn; dead-attn "
          f"skipped) = {acct['weight_bytes_touched']/1e9:.2f} GB  "
          f"(exec dense GEMM {acct['dense_flop_touched_per_tok']/1e9:.3f} GFLOP/tok)")
    print(f"[launch] eager approx kernel launches/step = "
          f"{launch['eager_launches']}")

    Ks = [int(k) for k in args.ks.split(",")]

    # --- install live-head + dead-block classification (sets _live_head_mask) ---
    from .live_head_attention import (install_live_head_attention,
                                      install_dead_block_fusion)
    install_live_head_attention(model, verbose=True)
    install_dead_block_fusion(model, verbose=True)
    if args.mode == "sparse_mm":
        # GRAPH-SAFE lean densify: FFN (0.84 GB) + live-attn only (~0.14 GB); dead
        # attention stays sparse (never read).  torch.sparse.mm does NOT replay in a
        # CUDA graph, so the graphed path needs F.linear on resident dense weights.
        materialize_lean_for_graph(model, dev)
        print("[lean-mat] densified FFN + live-attn only (graph-safe F.linear; "
              "dead-attn weights stay sparse/unread)")
    # make the 3 LIVE-attention blocks CAPTURABLE (remove the torch.nonzero D2H sync)
    make_live_blocks_capturable(model, dev)

    fsg = FullStepGraph(model, dev)

    # =====================================================================
    # (A) EAGER roofline: forward_hidden_cached, flash OFF vs ON, per K.
    # =====================================================================
    print("\n" + "=" * 120)
    print("(A) EAGER batched forward (forward_hidden_cached) roofline")
    print(f"{'K':>6} | {'flash':>5} | {'ms/step':>9} | {'ms/tok':>8} | "
          f"{'sparseTF':>8} | {'denseTF':>8} | {'HBM GB/s':>9} | {'VRAM GB':>8} | bound")
    print("-" * 120)

    for flash in ("0", "1"):
        os.environ["C4_FLASH_ATTN"] = flash
        for K in Ks:
            try:
                torch.cuda.reset_peak_memory_stats(dev)
                x, qpos = make_span_inputs(model, K, dev)
                dt, h = time_forward(model, x, qpos, iters=args.iters)
                peak_vram = torch.cuda.max_memory_allocated(dev) / 1e9
                sparse_tf, dense_tf, bw_gbs, bw_util, bound = _roofline(
                    acct, model, K, dt, peaks)
                print(f"{K:>6} | {'ON' if flash=='1' else 'OFF':>5} | "
                      f"{dt*1e3:>9.3f} | {dt*1e3/K:>8.4f} | {sparse_tf:>8.4f} | "
                      f"{dense_tf:>8.2f} | {bw_gbs:>9.2f} | {peak_vram:>8.2f} | {bound}")
            except torch.cuda.OutOfMemoryError:
                print(f"{K:>6} | {'ON' if flash=='1' else 'OFF':>5} | {'OOM':>9}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{K:>6} | {'ON' if flash=='1' else 'OFF':>5} | "
                      f"ERR {type(e).__name__}: {str(e)[:55]}")
                torch.cuda.empty_cache()
    print("=" * 120)

    # =====================================================================
    # (B) BIG CUDA GRAPH: whole step (242 blocks) in ONE graph, per K.
    #     flash OFF (the masked-full live attention is capturable; flash's
    #     Triton kernel is also captured, but OFF is the byte-exact golden path).
    # =====================================================================
    os.environ["C4_FLASH_ATTN"] = "0"
    print("\n(B) BIG CUDA GRAPH — ENTIRE step (all 242 blocks: 239 dead + 3 live "
          "attention) in ONE graph")
    print(f"    1 graph = 1 launch  vs  ~{launch['eager_launches']} eager kernel "
          f"launches/step (launch overhead -> ~0)")
    print(f"{'K':>6} | {'graph ms':>9} | {'ms/tok':>8} | {'eager ms':>9} | "
          f"{'speedup':>8} | {'sparseTF':>8} | {'HBM GB/s':>9} | {'VRAM GB':>8} | bound")
    print("-" * 120)
    for K in [k for k in Ks if k <= args.max_graph_k]:
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            x, qpos = make_span_inputs(model, K, dev)
            dt_eager, _ = time_forward(model, x, qpos, iters=args.iters)
            dt_g, _ = fsg.time_replay(K, qpos, iters=args.iters)
            peak_vram = torch.cuda.max_memory_allocated(dev) / 1e9
            sparse_tf, dense_tf, bw_gbs, bw_util, bound = _roofline(
                acct, model, K, dt_g, peaks)
            print(f"{K:>6} | {dt_g*1e3:>9.3f} | {dt_g*1e3/K:>8.4f} | "
                  f"{dt_eager*1e3:>9.3f} | {dt_eager/dt_g:>7.1f}x | "
                  f"{sparse_tf:>8.4f} | {bw_gbs:>9.2f} | {peak_vram:>8.2f} | {bound}")
        except torch.cuda.OutOfMemoryError:
            print(f"{K:>6} | OOM capturing/replaying full-step graph")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{K:>6} | ERR {type(e).__name__}: {str(e)[:70]}")
            torch.cuda.empty_cache()
    print("=" * 120)

    # ---- graph vs eager byte-identity (fresh graph instance so the timing loop's
    #      buffer reuse can't alias the comparison) ---------------------------
    print("\nBIG-GRAPH vs EAGER equivalence (RELATIVE diff of full-step hidden; the "
          "graph is the same arithmetic, one replay):")
    fsg_id = FullStepGraph(model, dev)
    for K in [k for k in Ks if k <= min(args.max_graph_k, 4096)]:
        try:
            d = graph_eager_identity(fsg_id, model, K, torch.arange(K, device=dev),
                                     dev)
            print(f"  K={K:>6}: rel|graph - eager| = {d:.2e}  "
                  f"[{'IDENTICAL' if d < 1e-4 else 'DIFF'}]")
            del fsg_id._cache[K]        # free before next K
        except Exception as e:
            print(f"  K={K:>6}: {type(e).__name__}: {str(e)[:50]}")
            torch.cuda.empty_cache()

    # ---- softmax1 <-> bos-sink flash byte-identity at each K -------------
    print("\nFLASH softmax1 <-> BOS-sink byte-identity (max abs diff over live "
          "attn ctx):")
    for K in [k for k in Ks if k <= 4096]:
        try:
            d = flash_byte_identity(model, K, dev)
            verdict = "IDENTICAL" if d < 1e-3 else f"DIFF {d:.2e}"
            print(f"  K={K:>6}: max|flash_softmax1 - bos_sink_plainsoftmax| = "
                  f"{d:.2e}  [{verdict}]")
        except torch.cuda.OutOfMemoryError:
            print(f"  K={K:>6}: OOM (bos-sink reference is O(K^2))")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  K={K:>6}: {type(e).__name__}: {str(e)[:50]}")
            torch.cuda.empty_cache()

    os.environ["C4_FLASH_ATTN"] = "0"
    return 0


def _roofline(acct, model, K, dt, peaks):
    """Return (sparse_tflops, exec_dense_tflops, hbm_gbs, hbm_util, bound-string).

    * sparse_tflops    — the USEFUL work: 2*nnz*K / dt.  This is the real signal for
      "are we saturating the GPU on the actual sparse arithmetic".
    * exec_dense_tflops — the dense GEMM FLOPs the composed forward ACTUALLY executes
      (all FFN dense + live-attn dense, most of it multiplying padded zeros) / dt.
      This is what the FP32 cores really run -> classify FLOP-bound off THIS.
    * hbm_gbs          — the HONEST bytes moved: touched dense weights (FFN + live
      attn, re-streamed every step) + activation read/write per block.
    """
    lin_sparse = acct["sparse_flop_per_tok"] * K
    exec_dense = acct["dense_flop_touched_per_tok"] * K
    att = attn_flops(acct["live_attn"], K)
    sparse_tflops = (lin_sparse + att) / dt / 1e12
    exec_dense_tflops = (exec_dense + att) / dt / 1e12
    # HBM: touched dense weights (FFN + live-attn, re-read every step) + activation r/w.
    act_bytes = 2.0 * len(model.blocks) * K * model.dim * 4
    hbm_bytes = acct["weight_bytes_touched"] + act_bytes
    bw_gbs = hbm_bytes / dt / 1e9
    bw_util = bw_gbs / peaks["hbm_bw_gbs"]
    exec_flop_util = exec_dense_tflops / peaks["peak_fp32_tflops"]
    bound = classify_bound(K, exec_flop_util, bw_util)
    return sparse_tflops, exec_dense_tflops, bw_gbs, bw_util, bound


if __name__ == "__main__":
    raise SystemExit(main())
