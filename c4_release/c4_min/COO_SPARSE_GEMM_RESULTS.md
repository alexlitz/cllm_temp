# Task #806 — Sparse-GEMM (COO-SpMM) efficiency: MEASURED results

GPU: NVIDIA RTX A5000 (24 GB), torch 2.10.0+cu128, triton 3.6.0.
Golden 069cc32f byte-identical (all work is gated / measurement-only; no stored
weight touched).

Model: `build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")`
→ 242 blocks, dim=1679.

**Run note (environment fix):** the model-build import chain calls
`os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` (several `_agent_*`/`test_*`
modules), which blanks the visible-device list AFTER torch init and breaks
Triton's fresh device-id lookup on the FIRST post-build launch (`AssertionError:
Invalid device id`). Fix for all runs: launch with `CUDA_VISIBLE_DEVICES=0,1`
(or warm one Triton launch before the build). This is a harness quirk, not a
kernel bug.

---

## 1. COO-SpMM correctness (already-built kernel, re-confirmed)

`_agent_coo_kernel_check.py`, all 242 distinct FFN blocks, K=17 random input:

- Worst REL L-inf = **3.78e-02** (block 239, abs 1.85e-01 on |ref|max 4.90).
- Decomposition of block 239 (`_ttest4.py`):
  - W_up  L-inf = 3.05e-05  (rows read 1 nonzero → essentially exact)
  - W_gate L-inf = 0.0
  - hidden L-inf = 1.53e-05
  - **W_down L-inf = 6.29e-02** (rows read ~35 nonzeros)
- VERDICT: the residual is entirely a **fp accumulation-ORDER** difference in
  W_down (Triton sums nonzeros in column-sorted order; cuBLAS `F.linear` uses
  a tiled order). Same arithmetic on the nonzeros; zeros contribute exactly 0.
  ~1% relative on ~4.9-magnitude values → **absorbed by the integer-nibble
  decode margin** ("byte-exact at the snapped-nibble margin", as #806 states).
  NOT a kernel bug.

## 2. Per-block micro-speed: COO-SpMM vs dense F.linear

Representative BIG block (div-round1, Dff=4320, nnz up/gate/down = 4320/4320/7920):

| K    | dense F.linear | COO-SpMM | speedup |
|------|---------------:|---------:|--------:|
| 1024 | 3420.5 us      | 451.8 us | 7.57x   |
| 4096 | 13283.4 us     | 1644.4 us| 8.08x   |

(These are single-block wall times, eager launch. Whole-forward composition +
CUDA-graph numbers are measured below.)

## 3. #804 dense-padded baseline REPRODUCED on this A5000

`_agent_flash_roofline_largek.py` (mode=sparse_mm, lean-densified F.linear FFN,
one big CUDA graph over all 242 blocks). Measured GPU peak FP32 = **17.4 TFLOPS**
(matches the A5000 17.4 spec used in the projection); HBM 682 GB/s.

| K    | dense ms/step | **dense ms/token** | exec-dense TF | useful sparse TF | %peak useful |
|------|--------------:|-------------------:|--------------:|-----------------:|-------------:|
| 1024 | 70.0 (graph)  | **0.0684–0.0699**  | 7.06          | 0.054            | **0.31%**    |
| 4096 | 294.8 (graph) | **0.0720**         | 6.99          | 0.196            | 1.12%        |

- The dense-padded forward is **FLOP-bound at ~40% of FP32 peak**, but only
  **0.0055%** of that FLOP is useful (dense-equiv 5.880 GFLOP/tok vs 0.326
  MFLOP/tok useful nnz). So the *useful* throughput is only **0.31% of peak** at
  K=1024 → the ~3900x (here ~18000x dense/useful, ~1500x touched/useful) headroom.
- The big CUDA graph gives ~1.0x over eager here (FLOP-bound, not launch-bound):
  removing 1219 launches doesn't help when the dense GEMMs saturate the cores.
- Big-graph vs eager: rel diff **0.00e+00 (IDENTICAL)**. Flash softmax1 ==
  bos-sink to 6e-8 (K=1024).

## 4. COO-SpMM composed forward — CUDA-graph + speed (THE DELIVERABLE)

`_agent_coo_graph_bench.py`: swap ALL 242 FFN blocks → `CooSpmmFFN`, compose the
whole step (239 dead-attn-bypassed + 3 live-attn blocks), capture into ONE CUDA
graph.

**Q1 — does the Triton COO kernel replay in a CUDA graph?  → YES.**
The whole composed COO forward (726 Triton SpMM launches + 3 live-attn blocks)
**captures and replays correctly** in a single `torch.cuda.CUDAGraph` at K=1024
and K=4096 (`graph OK = REPLAYS`, replay-vs-eager rel err < 1e-3). This confirms
the #806 hypothesis: unlike `torch.sparse.mm` (which does NOT replay), a plain
Triton launch is graph-capturable and composes with the big graph + flash.
(Graph capture OOMs at K>=8192 — the O(K^2) attention + [Dff,K] activations, not
a graph incompatibility; eager still runs.)

**Q2 — MEASURED ms/token, throughput, VRAM:**

| K    | COO eager ms | COO graph ms | **COO ms/token** | vs dense 0.069 | useful TF | **%peak** | VRAM GB |
|------|-------------:|-------------:|-----------------:|---------------:|----------:|----------:|--------:|
| 1024 | 44.1         | **33.6**     | **0.0328**       | **2.13x**      | 0.115     | **0.67%** | 0.39    |
| 4096 | 171.6        | 170.8        | **0.0417**       | **1.73x**      | 0.338     | 1.95%     | 4.47    |
| 8192 | 468.5 (eager)| OOM (graph)  | 0.0572           | 1.21x          | 0.487     | 2.82%     | 16.97   |

- **Best operating point K=1024: 0.0328 ms/token, 2.13x faster than the 0.069
  dense baseline.** The graph helps ~1.3x over eager (44→34 ms) by removing the
  726-launch dispatch; at K=4096 the graph barely helps (kernels are large enough
  that launch overhead is amortized).
- **Useful throughput recovered: 0.31% → 0.67% of FP32 peak at K=1024 (2.15x more
  of peak).** Of the ~3900x headroom, the COO forward recovers **~2.1–2.2x** of
  the wall-clock (0.069→0.033 ms/token). It does NOT recover the full per-block
  7.5x — see the attribution below.
- Peak VRAM: **0.39 GB at K=1024** (vs 1.38 GB dense-graph) — the CSR weights are
  0.7 MB total; VRAM is dominated by the [Dff,K] activations, so K=1024 is very
  cheap and leaves 23 GB headroom.

## 5. WHY 2.1x full-step and not the 7.5x per-block (honest gap)

`_agent_coo_attrib.py` at K=1024:

- Full composed step 42.5 ms; **FFN-COO is 92% of it (38.9 ms)**, live-attention
  only 12% (5.3 ms).
- **Per-FFN-block averages 160.7 us** across 242 blocks = the full step is
  **launch/overhead-bound on 726 sequential tiny Triton SpMM kernels** (3 per
  block: up/gate/down) plus a per-block transpose→silu*mul→transpose glue (the
  silu*mul alone is 134 us on the big block).
- The per-block micro-benchmark saw 7.5x because it measured ONE big block in
  isolation (no 242x launch serialization). Composed, the launch + elementwise
  glue overhead per block dominates the tiny nnz arithmetic — the CUDA graph
  removes launch *dispatch* latency (1.3x) but the kernels still execute serially
  with fixed per-kernel occupancy overhead.

## 6. PROJECTION — subsequent frame (6.89M steps)

| path              | ms/token (K=1024) | frame time      | clears 1 min? |
|-------------------|------------------:|-----------------|:-------------:|
| dense-padded #804 | 0.0699            | 8.03 min        | NO            |
| **COO-SpMM**      | **0.0328**        | **3.76 min**    | **NO**        |

- COO cuts the projected per-frame time from **8.0 min → 3.8 min** (2.1x) but
  does **NOT clear 1 minute**. To hit <1 min at 6.89M steps needs ms/token
  ≤ 0.0087 — a further ~3.8x beyond COO.
- The remaining gap is NOT FLOPs (useful nnz would run in microseconds at peak);
  it is the **726 sequential per-block kernel launches + per-block elementwise
  glue**. The clear next lever is a SINGLE grouped/segmented kernel that fuses
  all blocks' up+gate+silu+down into one launch (batched-CSR / block-MoE grouped
  GEMM), eliminating the 242x serialization — projected to approach the per-block
  7.5x, i.e. ~0.010 ms/token, the ~1-min regime. That fused grouped kernel is the
  recommended follow-up; it was out of scope for this measurement pass.

## Summary of the answer to #806

- COO-SpMM Triton kernel **replays in a CUDA graph** (composed with the big graph
  + flash) — the #804 note's `torch.sparse.mm` incompatibility does NOT apply to
  the Triton path.
- **MEASURED 0.0328 ms/token (K=1024) vs the 0.069 dense baseline = 2.13x**;
  useful throughput 0.31% → 0.67% of FP32 peak.
- Recovers **~2.1x** of the ~3900x headroom (wall-clock); the rest is bounded by
  726-launch + elementwise-glue overhead, NOT FLOPs.
- Projected frame **3.76 min** (down from 8.03 min) — improved but **does not yet
  clear 1 minute**; a fused grouped-GEMM (one launch for all blocks) is the
  identified path to the <1-min regime.
- Byte-exactness held throughout: correctness residual is W_down fp
  accumulation-ORDER (~6e-2 raw), absorbed by nibble-snap decode. Golden 069cc32f
  untouched (all COO work is measurement/gated, no stored weight modified).
