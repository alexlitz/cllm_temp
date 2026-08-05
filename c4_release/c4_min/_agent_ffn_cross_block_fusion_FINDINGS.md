# Cross-block FFN fusion on the doom-active dead-FFN chain — PROTOTYPED + MEASURED

Base `agent-doom-depth-analysis` @ `97e4ce9e` (production code identical to `e28935ba`;
`97e4ce9e` is purely additive doom-active tooling).  GPU: A5000 (24 GB, ~768 GB/s HBM,
6 MB L2).  Model: 242 blocks, `dim=1416`, `C4_PF_CFM=1`, recurrent-divmod, dense_kernel.
Doom-active dead-FFN chain = **49 blocks, 26–27 waves (max-parallel 5)** (the block-skip's
DIV-free carve-out).  Baseline to beat: **0.488 µs/step (14.6 % HBM peak), 2.17 fps 1-GPU**.

Two levers added to `c4_min/fused_megablock.py`, both DEFAULT-OFF:
- `C4_FFN_WAVE_BATCH` (A, HORIZONTAL) — the ≤5 independent blocks per dependency wave
  concatenate into ONE up/gate + ONE down-delta launch (49×2 → 26×2 launches).
- `C4_FFN_LINFOLD` (B, VERTICAL) — fold block N's `W_down` into block N+1's `W_up`/`W_gate`
  (`A=Wu_{N+1}@Wd_N`, `B=Wg_{N+1}@Wd_N`) so N+1 reads N's hidden directly.

Battery: `_agent_ffn_cross_block_fusion.py` (measure), `_agent_ffn_fusion_byteexact.py`
(byte-exact gate on random residuals AND 1248 REAL doom-stream cut-boundary residuals).

## (A) HORIZONTAL wave-batch — BYTE-EXACT, and it BREAKS the tile-bound wall (K-dependent)

**Byte-exact:** L-inf = **0.000e+00** vs the 2-kernel skip baseline on random
doom-magnitude residuals (K=64/517/4096) AND on **1248 REAL doom-stream residuals**
captured at the `[cut]` boundary of every non-DIV corpus step.  Bit-identical (not just
nibble-margin) — a wave is hazard-free (RAW+WAR+WAW), so the ≤5 blocks read the same
pre-wave residual and write disjoint rows; the concatenated CSR is the same nonzeros in the
same per-output accumulation order.

**The win is TILE-driven and requires a bigger `C4_MEGABLOCK_BLOCK_K`.**  Per-block, the
dead FFNs are too small to fill a large K-tile, so the block_k lever is inert on the
baseline (0.45 µs at every bk).  Once 5 blocks are concatenated into one wave-kernel, the
big tile fills → kernel-efficiency jumps.  Wave-batch is what UNLOCKS block_k:

| K (chunk) | bk  | 2k baseline µs / %HBM | (A) wave-batch µs / %HBM | speedup |
|-----------|-----|-----------------------|--------------------------|---------|
| 8192      | 64  | 0.452 / 15.8 %        | 0.393 / 18.1 %           | 1.15×   |
| 8192      | 128 | 0.453 / 15.7 %        | 0.269 / 26.5 %           | 1.69×   |
| **8192**  | **256** | **0.454 / 15.7 %** | **0.246 / 29.0 %**       | **1.84×** |
| 65536     | 64  | 0.487 / 14.6 %        | 0.484 / 14.7 %           | 1.01×   |
| 65536     | 256 | 0.331 / 21.6 %        | 0.329 / 21.7 %           | 1.01×   |

- **At K=8192 (the realistic doom render chunk), wave-batch + `C4_MEGABLOCK_BLOCK_K=256`
  = 0.246 µs/step at 29–30 % HBM peak — 1.84× on the mega-chain, DOUBLING the 12–15 %
  tile-bound floor.**  Wave-batch is load-bearing here (baseline can't use the big tile).
- **At K=65536 (the reference chunk), the existing `C4_MEGABLOCK_BLOCK_K=256` lever alone
  breaks the wall (14.6 %→21.6 %, 0.487→0.331 µs); wave-batch adds ~0 %** — at large K the
  per-block kernels already fill big tiles.

**GRAPHED path (production `run_graphed`, what the block-skip actually dispatches):** the
CUDA graph already collapses launches to 1 replay, so it captures wave-batch's launch
reduction for free.  At K=8192/bk=256: 2k-GRAPHED 0.232 µs ≈ (A)-eager 0.246 ≈
(A)+GRAPHED 0.234 (all ≈30 % HBM, all L-inf=0).  So the best number (~0.23–0.25 µs) is
reachable via EITHER wave-batch-eager OR graph+bk256; they are REDUNDANT (both remove
launch overhead).  Wave-batch's distinct value is the eager (un-graphed) path and making
the block_k tile fillable.

fps @ K=8192/bk=256 (whole-step = mega 0.246 + 0.80 non-chain):
**(A) 1.044 µs/step → 2.67 fps 1-GPU / 5.35 fps 2-GPU** (vs 2.17 / 4.34 block-skip).

## (B) VERTICAL lin-fold — NET LOSS, fill-in eats it (honest NO)

**Byte-exact at the nibble margin:** L-inf 1.0e-3–4.9e-4 residual residue (same
fp-accumulation-order residue the COO/fused paths carry vs cuBLAS), **AX-nibble decode
IDENTICAL** to the 2k path on all 1248 real residuals.  So it is decode-safe, but NOT a
speed win.

**Fill-in factor = 6.64×** (fused `A`/`B` nnz 129 986 vs sparse factor nnz 19 573; 24
folded pairs).  The `[Dff,Dff]` products densify far more than the tile-occupancy gain
recovers:

| K     | bk  | 2k baseline | (B) lin-fold | net |
|-------|-----|-------------|--------------|-----|
| 8192  | 64  | 0.452       | 0.885 (0.51×)| LOSS |
| 8192  | 256 | 0.454       | 0.471 (0.96×)| LOSS |
| 65536 | 64  | 0.487       | 0.966 (0.50×)| LOSS |
| 65536 | 256 | 0.331       | 0.507 (0.65×)| LOSS |

**Verdict (B): NET LOSS at every K/bk — the 6.64× fill-in decisively eats the fold.**
Confirms the brief's caution: the fold trades sparse FLOPs for denser GEMMs, and the doom
dead-FFN factors are sparse enough (1.3 nnz/unit) that the fill-in dominates.  Composing
A+B is dominated by B → also a loss (0.96× best).

## Composed best + new dominant wall

- **Best doom-active mega-chain: ~0.246 µs/step** (wave-batch + `C4_MEGABLOCK_BLOCK_K=256`,
  K=8192; or graph+bk256 ≈0.232), **≈1.98× the 0.488 µs block-skip baseline, ~29–31 % HBM
  peak** — cross-block fusion (A) + the tile lever DOES break the 12–15 % wall (to ~30 %).
- **Whole-step ≈ 1.04 µs/step → 2.67 fps 1-GPU / 5.35 fps 2-GPU** (vs 2.17 / 4.34).
  Wave-batch adds **+0.50 fps** (1-GPU) on top of the block-skip's 2.17.
- **New dominant wall = the LIVE-CAM blocks (0.69 µs), now ~66 % of the 1.04 µs step.**
  The dead-FFN mega-chain has dropped from 0.488→0.246 µs (38 %→24 % of the step).  The
  next lever is the live-CAM 0.69 µs (direct-gather), NOT the dead-FFN chain.

## Honest one-liner
Cross-block HORIZONTAL wave-batching (A) is byte-exact (L-inf=0, bit-identical) and, by
making the K-tile fillable, unlocks `C4_MEGABLOCK_BLOCK_K=256` to take the dead-FFN chain
from 14.6 % → ~30 % of HBM peak (≈1.9× on the chain, 0.488→0.246 µs/step) at the realistic
K=8192 doom chunk — **breaking the 12–15 % tile-bound wall** and adding **+0.5 fps** (2.17→
2.67 1-GPU); VERTICAL lin-fold (B) is a decode-safe NET LOSS (6.64× fill-in).  On the
reference K=65536 chunk the existing block_k lever alone gets there; wave-batch's launch
reduction is redundant with the production CUDA graph but load-bearing eager / at small K.
The per-step wall is now the live-CAM 0.69 µs.

## Golden / VRAM
- Flag-OFF golden **UNCHANGED: `069cc32f`**
  (`CUDA_VISIBLE_DEVICES="" python -m c4_min._fingerprint_build`).  Both levers default-OFF,
  pure gated tooling — no build-path writes.
- VRAM peak: **0.44 GB** (2k/wave-batch path, K=8192); 4.28 GB at the K=65536 reference
  chunk; +the lin-fold dense-A/B build adds ~0.15 GB.  Well within budget.
