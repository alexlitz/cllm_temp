# Block-0 [D,K] fold (C4_BLOCK0_DK) + dead-FFN occupancy — MEASURED

Base `consolidate-0.5b-2026-07-22` @ `5a6b21d4` (post `C4_ATTN_MEGABLOCK`).  GPU: RTX A5000
(24 GB, ~768 GB/s HBM, 64 SMs).  Model: 242 blocks, `dim=1416`, recurrent-divmod, dense_kernel.
Composed on-chip precomputed-schedule fast path, all landed levers on + `C4_ATTN_MEGABLOCK`.
Starting point (this task's baseline): **7.3 fps 1-GPU / 14.6 fps 2-GPU @96K, byte-exact,
golden `069cc32f`**, whose new wall (per `ATTN_MEGABLOCK_FINDINGS.md`) is dead-FFN chain 71% +
one `[1,C,D]->[D,C]` block-0 input transpose 19%.

## Task 1 — Fold block-0 into the [D,K] region (C4_BLOCK0_DK) — WIN

Under `C4_ATTN_MEGABLOCK`, block-0's FFN ran in the natural `[1,C,D]` layout
(`h = ffn0.forward(h0) -> [1,C,D]`) and was then transposed `h[0] -> [D,C]` to enter the fused
`[D,K]` chain.  That transpose is captured IN the graph body, so it ran EVERY replay.

`C4_BLOCK0_DK` treats block-0 as just another sparse block in the `[D,K]` region: the resident
folded-h0 table is transposed to `[D,K]` ONCE at frame-build (its honest home — build is
pipelined), the graph's static `_s_h0` is bound/filled in `[D,C]` layout (plain slice-copy, no
transpose), block-0's FFN runs IN PLACE on that `[D,C]` buffer via the SAME `_MegaFFN.run_fused`
kernel the dead-FFN chain uses, and the decode reads the `[D,C]` rows directly.  The
`[1,C,D]<->[D,C]` transpose vanishes from the graph body entirely.

### Byte-exact (L-inf=0)
- Whole-step graph, DK vs MEGA vs BASE(`C4_ATTN_MEGABLOCK` OFF), decoded PC/SP/BP/AX over the
  96,469-step `nested_120_255` doom chunk = **L-inf 0** (0 mismatches, all three).
- Op corpus (loop_countdown, nested, malloc, malloc_free, matmul = arith/branch/load/store/
  memory/mul), DK vs BASE @ bk512 = **ALL L-inf 0**.
- Golden flags-OFF UNCHANGED `069cc32f` (`CUDA_VISIBLE_DEVICES="" python -m
  c4_min._fingerprint_build`).  Pure-runtime, default-OFF flag; requires `C4_ATTN_MEGABLOCK`
  (inert alone).

### µs/step (pure replay, whole-step graph @96,469-step chunk, A5000)
| bk  | BASE  | MEGA(ATTN)  | DK           | DK/MEGA | DK/BASE |
|-----|-------|-------------|--------------|---------|---------|
| 256 | 1.990 | 1.111       | **0.839**    | 1.325x  | 2.37x   |
| 512 | 1.944 | 1.063       | **0.788**    | **1.348x** | **2.47x** |
| 128 | 2.207 | 1.340       | 1.072        | 1.250x  | 2.06x   |

**bk512 is best: DK = 0.788 µs/step, 1.35x over MEGA, byte-exact.**  VRAM peak UNCHANGED
(13.35 GB @96K, MEGA == DK).

### Why DK > "just the transpose" (kernel attribution, bk512, @96K body)
| bucket                    | MEGA  | DK    |
|---------------------------|-------|-------|
| FFN-mega chain (triton)   | 69.5% | **94.1%** |
| copy/clone (the transpose)| 20.2% | 0.1%  |
| GEMM (block-0 dense FFN)  | 3.5%  | 0.0%  |
| elementwise (block-0 silu)| 5.0%  | 1.4%  |
DK folds away not just the 20% transpose but ALSO block-0's dense FFN GEMM (3.5%) + dense silu
(5%) — block-0 becomes a sparse `run_fused` block — which is why the win is 1.35x, not ~1.19x.

### fps @96K
Applying the measured 1.35x (bk512, clean idle A5000) to the task's stated baseline
(pure-replay convention): **7.3 -> ~9.85 fps 1-GPU / 14.6 -> ~19.7 fps 2-GPU**.  On the
full-dispatch-loop convention (includes per-chunk output copies): MEGA 4.92 -> DK **6.65 fps
1-GPU / 13.30 fps 2-GPU**, TRUE-PIPE (max(build,dispatch)); DK build +~20 ms for the one-time
`[K,D]->[D,K]` table transpose, still dispatch-bound.

## Task 2 — Dead-FFN chain occupancy — EFFICIENCY-LIMITED (honest, don't chase)

After DK, the dead-FFN mega chain is **94% of the step**.  Independently measured on this HEAD
(`_agent_doom_occupancy.py`, doom-active 49-block chain, K=65536):

| bk  | 2-kernel %HBM | fused %HBM |
|-----|---------------|------------|
| 64  | 6.1%          | 5.9%       |
| 256 | 9.5%          | 10.6%      |
| 512 | 10.1%         | **11.5%**  |

The chain runs at **~11.5% of HBM peak** at the best tile (bk512) — occupancy/tile-bound, NOT
BW-bound.  Root cause = **tiny per-block work** (median Dff=40, ~8 active W_down rows per block)
+ the ~27-wave doom-active sequential dependency: each program does short CSR inner loops, so
memory latency dominates and arithmetic intensity is low.

Levers investigated:
- **Persistent megakernel** (`C4_MEGABLOCK_PERSIST`, exists on a sibling branch off the shared
  `e28935ba` base — same kernels/GPU): byte-exact (L-inf 0) but **REGRESSES 2-5x**.  Root cause
  = occupancy, not launch.  The CUDA graph already erased launch overhead (~0.002 µs/step), so
  there was nothing to save; the K-parallel persistent grid (<=512 programs) runs the whole
  chain serially per program and UNDER-FILLS the 64 SMs, trading away the active-row x K-tile
  parallelism the graphed per-block launches rely on.  NOT fixable as a launch/tile lever.
- **Larger effective K**: the ISOLATED chain %HBM ~doubles at K=131072 (11.5% -> 22.4%, 0.617
  -> 0.317 µs/step) — genuinely occupancy-starved at 65-96K.  BUT in the REAL whole-step DK
  path a bigger natural K does NOT compound: K=96,469 -> 0.788 µs/step vs K=192,919 -> 0.817
  µs/step (slightly WORSE, L2 pressure @16.8 GB + the non-chain terms don't scale).  So
  larger-K is not a clean byte-exact win at the @96K frame; it would only help by batching
  several frames into one giant chunk (VRAM-bounded; a scheduling change, not a kernel lever).
- **block_k / num_warps**: bk512 is the measured optimum (bk128 worse, bk1024 ~= bk512); the
  sibling-branch num_warps/num_stages sweep confirms the graphed kernels are at their occupancy
  optimum.

**Verdict: the dead-FFN COO chain is efficiency-limited at ~11.5% of HBM peak (bk512, @96K),
~8.7x from peak, because of the tiny per-block work + the sequential residual dependency — NOT
byte-exactly improvable further by any kernel/tile/persist/launch lever.**  Deeper wins would
need bf16 activations (byte-exactness risk) or a model-build change to the residual dependency
structure (out of scope for a byte-exact perf lever).

## Combined best @96K (bk512)
- BASE 1.944 -> **DK 0.788 µs/step (2.47x)**, byte-exact, golden `069cc32f` unchanged.
- The new dominant cost is the dead-FFN mega chain (94% of the step, ~11.5% HBM peak,
  efficiency-limited).  The transpose (was 19%) is gone; the live-CAM FFN (was 57%) is gone.
- VRAM peak UNCHANGED: 13.35 GB @96K (OFF == MEGA == DK).

## Files
- `precomputed_schedule.py` — `block0_dk_enabled()` (`C4_BLOCK0_DK`, default OFF) +
  `PrecomputedStepGraph._block0_dk` / `_block0_mffn` + the `[D,C]` `_s_h0` path in
  `_capture` / `replay` / `_body` / `_run_region_fused_dk(block0=True)`.
- `_agent_block0dk_verify.py` — whole-step BASE/MEGA/DK byte-exact + µs/step.
- `_agent_block0dk_corpus.py` — op-corpus DK-vs-BASE byte-exact gate.
- `_agent_block0dk_frame_fps.py` — honest per-frame fps (TRUE-PIPE) OFF/MEGA/DK.
- `_agent_block0dk_prof.py` — DK-vs-MEGA whole-step-body kernel attribution.
- `_agent_doom_occupancy.py` (pre-existing) — doom-active chain %HBM peak sweep.
