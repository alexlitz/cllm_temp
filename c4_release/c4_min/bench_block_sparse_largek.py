"""LARGE-K (large batch) head-to-head: DENSE vs CSR vs BSR(8/16/32) forward of the
LEAN compacted C4 VM, re-chasing the #704 block-sparsity verdict in the regime it
MISSED.

#704 benchmarked only N = 5..224 tokens/GEMM (dense wins, LATENCY-bound: the whole
forward is kernel-launch dominated and a scattered-tile BSR pays a launch-overhead
loss on tiny matrices).  But #703's K-sweep reaches speculation batch B = 16384
single-step windows (GEMM M = B x window ~= 114 k rows/GEMM), FAR past the N~=3000
dense/sparse crossover, COMPUTE-bound at ~37.8% of tensor-core peak, spending
~99.99% of FLOPs on ZEROS.  That is exactly where block sparsity SHOULD win.

This bench measures, at the ACTUAL speculation batch shapes B = 512, 2048, 4096,
8192, 16384 (window S=7 -> M up to ~114 k rows/GEMM):

  * ms/forward and ms/step (= ms/forward / B, the #703 metric) for DENSE, CSR, and
    BSR at blocksize 8/16/32, on the SAME byte-identical clustering permutation;
  * the single BUSIEST GEMM (down_proj, 1152x896, the most-nnz family) isolated, so
    the GEMM-level crossover is visible without the attention/RMSNorm/RoPE overhead;
  * the FLOP accounting (BSR does active_tiles*tile^2 MACs, ~0.05-0.3% of dense) and
    whether the measured speedup tracks it or is floored by per-GEMM launch overhead;
  * DECODE byte-identity at large B (L-inf residue is fp-reduction-order only);
  * VRAM headroom (guarded: skips a B whose activation footprint would OOM cuda:0).

Run:
    python -m c4_min.bench_block_sparse_largek --device cuda:0
    python -m c4_min.bench_block_sparse_largek --device cuda:0 --subset full
    python -m c4_min.bench_block_sparse_largek --device cuda:0 --batch 512,2048,16384

===========================================================================
MEASURED VERDICT (RTX A5000, cuda:0, fp32, TF32 OFF, subset=mem+cmp,
method=pack, n=15 warmup=3).  The large-K regime does NOT overturn #704 —
it CONFIRMS it, and finds a NEW crossover the other way:

FULL FORWARD us/step (= ms/forward / B), 10-layer lean stack, window S=7:
  B      M       dense    csr     bsr8    bsr16   bsr32   winner
  512    3584    76.29    58.71   61.32   61.07   60.74   CSR/BSR (-23%)
  2048   14336   72.85    56.39   59.12   59.44   58.75   CSR/BSR (-23%)
  4096   28672   72.65    59.69   62.17   62.47   61.76   CSR/BSR (-18%)
  8192   57344   73.91    79.44   80.06   79.64   77.82   DENSE   (+5%)
  16384  114688  70.49    100.00  99.04   98.26   95.35   DENSE   (+35%)

BUSIEST-GEMM isolation (down_proj 1152x896), ms/call:
  B=512:   dense 0.47  bsr8 0.23  (BSR 2x FASTER)
  B=4096:  dense 3.90  bsr8 2.33  (BSR ~1.6x faster)
  B=8192:  dense 7.78  bsr8 8.29  (DENSE faster)
  B=16384: dense 15.25 bsr32 24.47 (DENSE 1.6x FASTER)

Byte-identity: L-inf/scale ~1e-13 at every B (fp-reduction-order only).

THE FLOP ARGUMENT IS REAL BUT DOES NOT WIN: BSR does only 0.05-0.29% of
dense FLOPs (active_tiles*tile^2), yet is SLOWER than dense at B>=8192.
Reason: dense down_proj at M=114688 achieves 13.95 TFLOP/s = ~50% of the
A5000's fp32 (non-TF32) peak (27.8 TFLOP/s) via cuBLAS, while the torch BSR
path is a Triton kernel that emits "non-optimal kernel parameters" warnings
and whose cost scales with N (=M) columns per active tile, so at N~114k it
loses to the tensor-core dense GEMM despite doing 350-2000x more MACs.  The
crossover (~B=8192) is where cuBLAS dense saturates the tensor cores; past
it, "spending 99.99% of FLOPs on zeros" is still cheaper than an
under-optimized sparse kernel.

#703 RECONCILE: dense measured here = 0.0705 ms/step at B=16384 (vs #703's
0.0425 ms/step — same order; the delta is the extra RMSNorm/RoPE/attention
overhead this full-forward folds in, plus A5000-vs-#703-GPU + TF32 state).
The rough "BSR ~13 ms vs dense ~696 ms/forward = ~50x" estimate is
REFUTED: BSR full-forward at B=16384 = 1563-1623 ms, dense = 1155 ms, so
BSR is ~1.35x SLOWER, not 50x faster.  The FLOP-count extrapolation ignored
the kernel-efficiency gap.

VERDICT: block sparsity WINS only in the MID-batch band (B ~ 512-4096,
CSR/BSR ~-20% us/step) and LOSES at the large batch the task targeted
(B >= 8192).  The small-N '#704 loses' verdict is NOT overturned by
large K; large K makes dense win by MORE.  (The mid-batch CSR win is the
one genuinely useful result: an UNSTRUCTURED CSR — no clustering perm
needed — beats both dense and BSR at B<=4096.)
===========================================================================
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import isa
from . import qwen_full_vm as Q
from . import qwen_lean_forward as LF
from . import block_sparse_analysis as BSA
from . import block_sparse_forward as BSF
from .bench_block_sparse import _CSRLean, _CSRWeight


# ---------------------------------------------------------------------------
def _make_window(lean, device, B: int):
    """B replicated single-step VM windows (the speculation-batch residual).

    Each row is the SAME byte-identical naive step window (BOS sink + register
    frame + STEP_END query); replicating it is representative of the busiest GEMM
    shape (M = B*S rows) without needing a real drafted trace.  Built on CPU then
    moved to ``device`` so the huge B=16384 residual is materialised once."""
    code = isa.assemble([("IMM", 100), ("PSH", 0), ("IMM", 27), ("ADD", 0), ("HALT", 0)])
    x, pos = LF._build_stream_and_overlay(
        lean, code, {"PC": 0, "AX": 0, "SP": 252, "BP": 252, "STACK0": 0}, [], None)
    x = x.expand(B, -1, -1).contiguous().to(device)
    pos = pos.unsqueeze(0).expand(B, -1).contiguous().to(device)
    return x, pos


def _bench(fwd, x, pos, n: int, warmup: int, cuda: bool) -> float:
    """ms/call, averaged over ``n`` calls after ``warmup``."""
    if cuda:
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            fwd(x, pos)
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            fwd(x, pos)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def _vram_ok(device: str, B: int, S: int, H: int, headroom_gb: float = 3.0) -> bool:
    """Rough guard: the batched residual + attention scratch must fit with a margin.

    Peak activation ~ a handful of [B,S,H] fp32 tensors + [B,nh,S,S] scores; at S=7
    this is dominated by [B,S,H].  We require the estimated peak (~30 live copies for
    the 10-layer stack's transient GEMM outputs) plus a headroom under free VRAM."""
    if not device.startswith("cuda"):
        return True
    free, _tot = torch.cuda.mem_get_info(torch.device(device))
    est = 30 * B * S * H * 4  # ~30 live [B,S,H] fp32 buffers across the stack
    return est + headroom_gb * (1024 ** 3) < free


# ---------------------------------------------------------------------------
# Busiest-GEMM isolation: down_proj (1152x896, the most-nnz family).
# ---------------------------------------------------------------------------
def _busiest_gemm_bench(lean, perm, device, B, S, blocksizes, cuda, n, warmup):
    """Isolate the single busiest GEMM (layer-0 down_proj) dense vs CSR vs BSR at
    M = B*S rows.  This strips attention/RMSNorm so the GEMM-level crossover is
    visible directly (the full-forward number folds in ~10 layers x 7 GEMMs plus the
    fixed attention/RoPE/softmax overhead that is IDENTICAL across evaluators)."""
    clustered = BSA.apply_permutation(lean, perm)
    w = clustered.layers[0].down_w.to(device)          # [1152, 896]
    out_dim, in_dim = w.shape
    M = B * S
    x = torch.randn(M, in_dim, device=device, dtype=w.dtype)

    def dense_fwd(xx, _pp):
        return F.linear(xx, w)
    t_dense = _bench(lambda xx, pp: dense_fwd(xx, pp), x, None, n, warmup, cuda)

    csrw = _CSRWeight(clustered.layers[0].down_w).to(device)
    t_csr = _bench(lambda xx, pp: csrw.linear(xx), x, None, n, warmup, cuda)

    bs_times = {}
    for bsz in blocksizes:
        log: Dict[str, int] = {}
        bw = BSF._make_block_weight(clustered.layers[0].down_w.detach(), bsz, 2, log).to(device)
        bs_times[bsz] = (_bench(lambda xx, pp: bw.linear(xx), x, None, n, warmup, cuda),
                         bw.is_block_sparse, bw.active_tiles, bw.total_tiles)
    return M, out_dim, in_dim, t_dense, t_csr, bs_times


# ---------------------------------------------------------------------------
def run(device="cuda:0", subset_name="mem+cmp",
        batches=(512, 2048, 4096, 8192, 16384),
        blocksizes=(8, 16, 32), method="pack", n=30, warmup=5,
        busiest_only=False):
    subsets = {"base": Q.SUBSET_BASE, "mem+cmp": Q.SUBSET_MEM_CMP,
               "bitwise": Q.SUBSET_BITWISE, "full": Q.SUBSET_FULL}
    subset = subsets[subset_name]
    cuda = device.startswith("cuda")
    print(f"# LARGE-K bench subset={subset_name} device={device} method={method} "
          f"n={n} warmup={warmup}")
    vm = Q.build(code_size=24, subset=subset)
    lean_cpu = LF.LeanQwenVM.from_full_vm(vm, device="cpu")
    H = lean_cpu.hidden_size
    perm = BSA.build_clustering_permutation(lean_cpu, method=method)

    # ---- block density + FLOP accounting (the compute-bound argument) --------
    print(f"\n## BLOCK DENSITY / FLOP RATIO ({subset_name}, method={method})")
    total_nnz = 0
    dense_macs_per_layer = 0
    for l in lean_cpu.layers:
        for nm in ("q_w", "k_w", "v_w", "o_w", "gate_w", "up_w", "down_w"):
            w = getattr(l, nm)
            total_nnz += int((w != 0).sum())
            dense_macs_per_layer += w.numel()
    print(f"  total nnz (7 GEMM fam x {lean_cpu.n_layers} layers) = {total_nnz:,}")
    print(f"  dense MACs/forward-window-row               = {dense_macs_per_layer:,}"
          f"   fill={100.0*total_nnz/dense_macs_per_layer:.5f}%")
    clustered = BSA.apply_permutation(lean_cpu, perm)
    rep = BSA.model_tile_report(clustered, tiles=blocksizes)
    for tile in blocksizes:
        ts = rep[tile]["all"]
        block_macs = ts.active_tiles * tile * tile
        util = total_nnz / block_macs if block_macs else 0.0
        print(f"  bsr{tile:<2}: active={ts.active_tiles}/{ts.total_tiles} "
              f"skip={ts.skip_frac*100:.3f}% tile_fill={ts.tile_fill_frac*100:.2f}% "
              f"flop_ratio(vs dense)={ts.block_flops_ratio:.5f} "
              f"MACs={block_macs:,} useful_util={util*100:.3f}%")

    S = 7  # single-step window length (BOS + 5 register frames + STEP_END)

    # ---- busiest single GEMM (down_proj) -------------------------------------
    print(f"\n## BUSIEST GEMM isolation (down_proj 1152x896), ms/call  (M = B*{S})")
    print(f"  {'B':>6} {'M':>8} {'dense':>9} {'csr':>9} "
          + " ".join(f"{'bsr'+str(b):>16}" for b in blocksizes))
    for B in batches:
        if not _vram_ok(device, B, S, H):
            print(f"  {B:>6}  -- SKIP (VRAM guard) --")
            continue
        M, od, idim, td, tc, bst = _busiest_gemm_bench(
            lean_cpu, perm, device, B, S, blocksizes, cuda, n, warmup)
        cells = [f"{B:>6} {M:>8} {td:8.3f}m {tc:8.3f}m"]
        for b in blocksizes:
            t, is_bs, at, tt = bst[b]
            tag = "bsr" if is_bs else "DENSE-kept"
            cells.append(f"{t:7.3f}m({tag[:5]},{at}t)")
        print("  " + " ".join(cells))
        if cuda:
            torch.cuda.empty_cache()

    if busiest_only:
        return

    # ---- full forward: dense vs csr vs bsr, ms/forward AND ms/step ------------
    lean = LF.LeanQwenVM.from_full_vm(vm, device=device)
    csr = _CSRLean(lean_cpu).to(device)
    bs_models = {}
    for bsz in blocksizes:
        bs_models[bsz] = BSF.PermutedBlockSparseVM(
            lean_cpu, perm, blocksize=bsz, min_active_tiles=2).to(device)

    print(f"\n## FULL FORWARD ms/forward and ms/step  (ms/step = ms/forward / B)")
    hdr = f"  {'B':>6} {'M':>8} | {'dense/fwd':>10} {'dense/stp':>10} | " \
          f"{'csr/fwd':>9} {'csr/stp':>9} |"
    for b in blocksizes:
        hdr += f" {'bsr'+str(b)+'/fwd':>11} {'bsr'+str(b)+'/stp':>11} {'L-inf rel':>10} |"
    print(hdr)
    for B in batches:
        if not _vram_ok(device, B, S, H):
            print(f"  {B:>6}  -- SKIP (VRAM guard) --")
            continue
        x, pos = _make_window(lean, device, B)
        M = B * S
        td = _bench(lambda xx, pp: lean.forward(xx, past=None, q_positions=pp),
                    x, pos, n, warmup, cuda)
        tc = _bench(lambda xx, pp: csr.forward(xx, q_positions=pp), x, pos, n, warmup, cuda)
        r_dense = lean.forward(x, q_positions=pos)[0]
        row = f"  {B:>6} {M:>8} | {td:9.3f}m {td/B*1000:9.4f}u | " \
              f"{tc:8.3f}m {tc/B*1000:8.4f}u |"
        for b in blocksizes:
            bs = bs_models[b]
            tb = _bench(lambda xx, pp: bs.forward(xx, q_positions=pp), x, pos, n, warmup, cuda)
            r_bs = bs.forward(x, q_positions=pos)[0]
            linf = (r_dense - r_bs).abs().max().item()
            rel = linf / (r_dense.abs().max().item() + 1e-30)
            row += f" {tb:10.3f}m {tb/B*1000:10.4f}u {rel:10.1e} |"
        print(row)
        del x, pos, r_dense
        if cuda:
            torch.cuda.empty_cache()
    print("\n  (/stp is us/step; /fwd is ms/whole-batched-forward)")


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", default="mem+cmp",
                    choices=["base", "mem+cmp", "bitwise", "full"])
    ap.add_argument("--batch", default="512,2048,4096,8192,16384")
    ap.add_argument("--blocks", default="8,16,32")
    ap.add_argument("--method", default="pack", choices=["identity", "pack", "rcm"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--busiest-only", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(device=a.device, subset_name=a.subset,
        batches=tuple(int(b) for b in a.batch.split(",")),
        blocksizes=tuple(int(b) for b in a.blocks.split(",")),
        method=a.method, n=a.n, warmup=a.warmup, busiest_only=a.busiest_only)
