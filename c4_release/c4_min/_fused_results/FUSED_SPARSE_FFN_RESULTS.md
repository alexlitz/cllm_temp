# Fused segmented sparse-FFN kernel (#806 follow-up) — MEASURED results

GPU: NVIDIA RTX A5000 (24 GB), torch 2.10.0+cu128, triton 3.6.0.
Golden 069cc32f byte-identical (all work is a runtime FFN-forward swap that only
READS the stored weights; `git diff --stat HEAD` is EMPTY — zero tracked source
touched; the model state-dict hash is UNCHANGED before/after the swap).

Run everything with `CUDA_VISIBLE_DEVICES=0,1` (the build import chain
`os.environ.setdefault("CUDA_VISIBLE_DEVICES","")` otherwise blanks the device
after torch init and crashes Triton's first launch).

Model: `build_compact_sparse_streaming(code_size=44, compute_mode="sparse_mm")`
→ 242 FFN blocks, dim=1679, all share input D=1679.

---

## 1. The fused-kernel design — how 726 launches collapse

### The constraint that shapes the design: the blocks are SEQUENTIAL
A dependency probe (`_agent_dep_probe.py`) shows the c4_min residual is a DEEP
sequential chain: **241/242 blocks READ a residual dim an EARLIER block WROTE**
(204 dependency levels; the largest parallel wave is only 3 blocks). Block i+1's
FFN input is block i's output. So the 242 blocks **cannot** be batched into one
data-parallel grid over a single shared X — a naïve "all-242-in-one-GEMM" is
architecturally impossible here. The fusion that IS available is collapsing the
3-launch + glue structure **within** each block into one segmented launch, and
eliminating the per-block HBM waste.

### The fused kernels (`fused_sparse_ffn.py`)
Two Triton kernels, both segmented over the block's CSR nonzeros:

1. **`_fused_upgate_silu_kernel`** — fuses **up + gate + silu·mul into ONE launch**.
   One program per (Dff-row, K-tile) walks the row's W_up nonzeros then its W_gate
   nonzeros (X read once per row), applies `silu(up)*gate` in registers, and writes
   the fused hidden ONCE. This removes: the separate gate SpMM launch, the
   standalone `F.silu(up)*gate` elementwise GLUE kernel (134 µs on the big block in
   #806's attribution), and the up/gate HBM round-trip (they never touch HBM).

2. **`_down_delta_inplace_kernel`** — the down projection **written as a DELTA,
   in place, only to the residual rows W_down actually touches.** W_down averages
   **0.15 nnz/row**, so a block writes only a handful of the 1679 residual dims
   (Dff=4320 block → 135 rows, not 1679; Dff=288 blocks → 9 rows). The kernel
   launches a program ONLY for those `n_active` rows and does `y[d] += Σ Wdown[d,h]·H[h]`
   in place, fusing the residual add. This eliminates the full [D=1679, K] residual
   COPY that a plain down kernel does for the ~99% untouched rows (that copy was the
   single biggest cost — see §3).

Per block: **2 launches** (up+gate+silu fused; down-delta) instead of 3 SpMM +
1 glue = 4 ops. Across the step: **726 → 484 SpMM launches, and the 242 separate
silu-glue kernels are gone entirely; the down kernel now touches only ~active
rows instead of full [D,K].** Plain per-block launches (not a single grid) because
the residual chain forbids cross-block batching.

Graph-capturable: plain Triton launches with static per-block/K shapes. The whole
composed step (239 dead-attn-bypassed fused-FFN blocks + 3 live-attention blocks
0/7/11) captures into ONE `torch.cuda.CUDAGraph` and **REPLAYS correctly** at
K=1024 and K=4096 (replay-vs-eager rel err verified < 1e-3).

---

## 2. Byte-exactness (snapped-nibble) — CONFIRMED

`_agent_fused_byteexact.py`: 9 real C programs (add/sub/mul/div/mod/cmp/var/func
+ a 42-step **doom-slice** arithmetic chain `320*200 /16 %7` + compare + branch,
the fixed-point op mix a doom frame uses) run through the pure-forward VM in TWO
configs — the unmodified sparse_mm FFN (REF) and the fused-delta FFN (FUSED):

- **EVERY program's full per-step AX trace is IDENTICAL** (`trace_eq=True`), i.e.
  the SNAPPED-nibble decode (`_snap_lane` → integer registers) matches for all
  steps. doomslice: both REF and FUSED decode **67997** over 42 steps.
- The W_down fp accumulation-ORDER residual (~6e-2 raw, per #806) is **absorbed by
  the integer nibble-snap** — exactly the "match the SNAPPED nibble, not raw fp"
  bar the task states. Per-block raw rel L-inf on random input: worst **3.3e-7**
  (up/gate ~exact, down ~1e-7 — actually far below even the COO reference's 3.78e-2
  because the delta path sums fewer terms per row).
- **Golden untouched:** the model state-dict hash is IDENTICAL before/after the
  fused swap (`f1cc6a90…`) — the kernels only READ `W_up/W_gate/W_down` into CSR
  index tensors; no stored weight is modified. `git diff --stat HEAD` is empty.

---

## 3. Attribution — why the delta kernel is the lever

`_agent_fused_attrib.py` (the 2-launch upgate form before the delta fix, K=1024):

- up+gate+silu kernel: 5.0 ms (242 launches, 20.8 µs/launch)
- down kernel:        **6.5 ms** (242 launches, 27.0 µs/launch) — the BIGGER cost.
- The plain down kernel writes a full [D=1679, K] output per block, but W_down has
  mean 0.15 nnz/row → **~99 % of the 1679 rows are a pure residual COPY** (HBM
  bandwidth, zero useful FLOP). The delta-in-place kernel skips them entirely.

This is why the delta form roughly HALVES the fused time (12.7 ms → 7.6 ms graph)
and is the piece that clears 1 minute.

---

## 4. MEASURED ms/token, throughput, VRAM (graph, 30 iters)

| form   | K    | eager ms | graph ms | **ms/tok** | vs COO 0.0328 | vs dense 0.069 | useful TF | **%peak** | VRAM GB | graph |
|--------|------|---------:|---------:|-----------:|--------------:|---------------:|----------:|----------:|--------:|:-----:|
| COO #806 | 1024 | 43.2   | 33.4     | 0.0326     | 1.01×         | 2.14×          | 0.116     | 0.67%     | 0.39    | REPLAYS |
| upgate | 1024 | 19.5     | 12.7     | 0.0124     | 2.65×         | 5.65×          | 0.31      | 1.77%     | 0.41    | REPLAYS |
| **DELTA** | **1024** | **19.5** | **7.57** | **0.0074** | **4.44×** | **9.45×** | **0.51** | **2.93%** | **0.41** | **REPLAYS** |
| COO #806 | 4096 | 170.8  | 170.2    | 0.0416     | 1.00×         | 1.73×          | 0.34      | 1.95%     | 4.47    | REPLAYS |
| **DELTA** | **4096** | **65.4** | **64.9** | **0.0159** | **2.63×** | **4.54×** | **0.89** | **5.11%** | **4.48** | **REPLAYS** |

- **Best operating point K=1024: 0.0074 ms/token — 4.44× faster than the COO
  0.0328 and 9.45× faster than the dense-padded 0.069.**
- Useful throughput recovered: **0.31 % (dense-padded #804) → 0.67 % (COO #806) →
  2.93 % of FP32 peak (fused delta)** = **4.4× more of peak than COO, 9.5× more
  than dense-padded.**
- Peak VRAM **0.41 GB at K=1024** (23 GB headroom); the CSR weights are ~0.9 MB,
  VRAM is dominated by the [Dff,K] hidden buffers.
- `hybrid`/`full` single-launch forms (recompute the hidden to save launches) LOSE
  vs the 2-launch delta: the extra arithmetic on the dense blocks costs more than
  the launch it saves (the GPU is not purely launch-bound at this occupancy).

---

## 5. PROJECTION — subsequent frame (6.89M steps) — CLEARS 1 MINUTE

| path                | ms/token (K=1024) | frame time    | clears 1 min? |
|---------------------|------------------:|---------------|:-------------:|
| dense-padded #804   | 0.0699            | 8.03 min      | NO            |
| COO-SpMM #806       | 0.0328            | 3.76 min      | NO            |
| **FUSED DELTA**     | **0.0074**        | **0.85 min (50.9 s)** | **YES** |

- The fused delta kernel cuts the projected per-frame time from **3.76 min (COO)
  → 0.85 min**, i.e. **50.9 s < 60 s — it CLEARS 1 MINUTE at K=1024.** (The bar was
  ms/token ≤ 0.0087; delta is 0.0074, comfortably under.)
- K=4096 is 1.82 min (does not clear), so K=1024 is the operating point — and it
  also has the smallest VRAM (0.41 GB).

## 6. How much of the ~3900× is now recovered

- **Wall-clock:** dense-padded 0.0699 → fused 0.0074 ms/token = **9.45× recovered**
  (vs COO's 2.1×). Of the ~3900× dense-vs-useful FLOP headroom, the fused forward
  recovers **~9.5× of the wall-clock** (the residual gap is still real occupancy,
  not FLOPs — see below).
- **Useful throughput:** 0.31 % → **2.93 % of FP32 peak = ~9.5× of peak recovered**
  (COO recovered ~2.1×; the fused delta recovers ~4.4× more on top of COO).

## 7. HONEST — residual and what's next

The fused delta forward is still **~2.9 % of FP32 peak**, not saturated. The
remaining gap is NOT FLOPs (the 0.326 MFLOP/tok of useful nnz would run in
microseconds at peak) — it is the **484 sequential per-block Triton launches**
that the DEEP SEQUENTIAL RESIDUAL CHAIN forces (241/242 blocks depend on the
previous block's output, so they cannot be merged into one grid). At 7.57 ms /
484 launches ≈ 15.6 µs/launch, we are still launch/occupancy-bound on the tiny
per-block grids for the small blocks (median Dff=25).

To go materially further would require breaking the sequential-residual barrier
itself — e.g. algebraically pre-composing runs of consecutive dead-attn blocks
into a single fused linear (each dead block is `x + W_downᵢ·silu(W_upᵢ·x)·(W_gateᵢ·x)`,
a nonlinear map, so this is a genuine approximation/re-derivation, not a free
merge), or fusing k consecutive blocks' launches behind one grid via a
persistent-kernel / megakernel that walks the block sequence on-device (one
launch, on-device loop over the 242 blocks, keeping the residual in L2/SRAM).
The megakernel is the clear next lever: it removes the 484 launch boundaries
without changing the arithmetic, and is the path from ~3 % toward the tens-of-%
regime. But it was not needed to hit the goal — **the fused delta kernel already
clears the 1-minute target at K=1024.**

## Summary of the answer to #806's fused-kernel follow-up

- Built a SINGLE fused segmented up+gate+silu kernel (one launch) + a delta-in-place
  down kernel (writes only the ~active residual rows), collapsing the 726 sequential
  tiny SpMM launches + 242 silu-glue kernels + the full-[D,K] residual copies down to
  **484 launches with zero glue and no copy waste.** (All-242-in-one-grid is
  architecturally impossible: the residual chain is 204-deep sequential.)
- **MEASURED 0.0074 ms/token (K=1024) — 4.44× over COO 0.0328, 9.45× over dense
  0.069;** useful throughput 0.67 % → **2.93 % of FP32 peak**.
- **Projected frame 0.85 min (50.9 s) — CLEARS 1 MINUTE.** (COO was 3.76 min.)
- **Byte-exact at the snapped-nibble margin:** 9 real programs incl. a 42-step
  doom-slice decode IDENTICALLY (REF == FUSED). Golden 069cc32f untouched (state
  hash unchanged, no tracked source modified).
- Recovers **~9.5×** of the ~3900× headroom (vs COO's ~2.1×). The residual is the
  484 launch boundaries the sequential residual forces — a persistent/megakernel
  is the identified next lever, but was not needed to clear the goal.
