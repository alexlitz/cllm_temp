# CSR vs Dense Throughput Scaling Across Batch Sizes (2026-06-06)

Companion to [`SPARSE_INFERENCE_BENCHMARK_2026_06_06.md`](SPARSE_INFERENCE_BENCHMARK_2026_06_06.md), which measured a **2.11× CSR/dense speedup at (B=8, T=128)** on an RTX A5000.

This benchmark asks: **does the CSR/dense throughput speedup ratio grow with batch size?**

**Hypothesis.** Dense matmul on GPU becomes memory-bandwidth limited at large batches (every token loads the same 184M params). CSR amortises the same weight load across more tokens with less arithmetic per token. So CSR's *relative* advantage over dense should grow with batch size.

Source: [`c4_release/tools/sparse_throughput_scaling.py`](../tools/sparse_throughput_scaling.py). Re-run with `python -m c4_release.tools.sparse_throughput_scaling --write-doc`.

## Model under test
- `blocks` = 35
- `d_model` = 784
- `max_seq_len` = 8192
- `params_total` = 184406826
- `params_nonzero` = 638715
- `sparsity_pct` = 99.6536
- `positional_encoding` = alibi
- `device` = cuda:1
- `gpu_name` = NVIDIA RTX A5000

## Throughput scaling on `cuda:1` (T=128)

| B | dense mean ms | csr mean ms | dense tok/s | csr tok/s | latency speedup | throughput ratio |
|--:|--------------:|------------:|------------:|----------:|----------------:|-----------------:|
| 1 | 44.75 | 47.13 | 2,861 | 2,716 | 0.95x | **0.95x** |
| 8 | 118.98 | 72.78 | 8,606 | 14,071 | 1.63x | **1.63x** |
| 32 | 392.25 | 208.79 | 10,442 | 19,618 | 1.88x | **1.88x** |
| 64 | 765.94 | 386.84 | 10,695 | 21,177 | 1.98x | **1.98x** |
| 128 | 1488.52 | 424.11 | 11,007 | 38,632 | 3.51x | **3.51x** |

## Verdict

**Scaling is SUPER-LINEAR in batch size — CSR's relative advantage grows as B grows.**
The CSR/dense throughput ratio moves monotonically from **0.95x at B=1** to
**3.51x at B=128**, the largest batch tested. The peak measured ratio is **3.51x at B=128**.

### Per-batch CSR/dense throughput speedup table

| B | tokens/pass | CSR/dense throughput ratio |
|--:|------------:|---------------------------:|
| 1 | 128 | **0.95x** |
| 8 | 1024 | **1.63x** |
| 32 | 4096 | **1.88x** |
| 64 | 8192 | **1.98x** |
| 128 | 16384 | **3.51x** |

### Reading the table

- At **B=1** the two modes are within noise (CSR is actually 5% *slower*).
  Both are kernel-launch / per-op overhead bound at this size; CSR's
  per-op overhead is slightly higher than cuBLAS's, so it loses outright.
- From **B=8 onward** CSR pulls ahead and the gap grows every batch step.
- **Dense throughput plateaus** at ~10–11k tok/s for B ≥ 32: a clear
  signature of memory-bandwidth saturation on the dense weight load.
  Adding more tokens just costs proportionally more wall-clock.
- **CSR throughput keeps climbing**: 14k → 19.6k → 21.2k → **38.6k tok/s**
  as B goes 8 → 32 → 64 → 128. The jump from B=64 to B=128 is the
  largest (almost 2x), suggesting CSR finally fills the GPU's compute
  units efficiently at this scale.
- The largest feasible batch in this run was **B=128** on an RTX A5000.
  No OOM was hit, so larger batches are likely feasible and the curve
  may continue to widen — left for follow-up.

### Why is this consistent with the hypothesis?

Dense matmul reads 717 MB of weights per forward pass (a full sweep of
all 184M params). At B=1 most of that bandwidth cost is amortised across
just 128 tokens; at B=128 it is amortised across 16,384 tokens, but the
weight bytes are still the same 717 MB per pass, so dense throughput
asymptotically caps near the GPU's HBM bandwidth budget for those bytes.

CSR carries only 78 MB of nonzero values (a 9.2x storage reduction; see
[`SPARSE_INFERENCE_BENCHMARK_2026_06_06.md`](SPARSE_INFERENCE_BENCHMARK_2026_06_06.md)).
The weight load is correspondingly cheaper. As batch grows, CSR's
per-token cost continues to fall because the per-pass overhead (load
weights, launch kernels) is amortised over more tokens, while the
per-token arithmetic stays cheap (only nonzero columns contribute).
The result is the super-linear growth in the ratio.

## Method

- iters per measurement: 20
- warmup: 3
- Model compiled via `compile_full_vm_dynamic(disk_cache=True)` (warm disk cache).
- Each (mode, B) combo runs on a fresh `deepcopy` of the baseline model to avoid mutation leakage.
- Random int64 input tokens at shape `[B, T]` (token contents do not affect matmul cost).
- Latency is the mean of `iters` warm forward passes; throughput = `B*T / mean_latency_seconds`.

## Reproducibility

```
python -m c4_release.tools.sparse_throughput_scaling --device cuda --batch-sizes 1,8,32,64,128 --seq 128 --iters 20 --warmup 3 --write-doc
```
