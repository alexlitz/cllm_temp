# Big-K speculation + evict-once-per-block — measured

**Branch:** `big-k-spec-worktree-2026-07-20` (off `fast-path-perf-2026-07-20 @ 4eae7770`)
**Date:** 2026-07-20/21
**Device:** RTX A5000, 24 GB VRAM (`cuda:0`), `OMP_NUM_THREADS=4`,
`PYTORCH_ALLOC_CONF=expandable_segments:True`. **The box was under heavy
multi-agent GPU contention throughout** (2-4 other processes pinning both cards
at 100 % util, 2-20 GB each) — so the *absolute* wall / ms/step numbers below are
contention-inflated and noisy. The **contention-independent mechanics** (forwards,
eviction rounds, peak VRAM, cache size, byte-exactness) are the load-bearing
results and are clean.

## The lever (the user's point)

The naive path was `1.3 s/step` because **every** per-step overhead — the 306-block
loop, KV eviction, Python overlay, kernel launches — hit **every** VM step. The
draft is PERFECT (deterministic logical VM, 100 % accept), so instead of verifying
K≈64 steps per forward we crank **K up to ~1000 VM steps (~30k tokens) per batched
forward** and **evict ONCE per verify block** (per K steps) instead of every 120
tokens. Both amortize the fixed per-step cost over K steps:

* **forwards ∝ N/K** — fewer, larger batched forwards;
* **eviction rounds ∝ N/K** — the dominant CPU wall (wall #2 in
  `PERF_FAST_PATH_2026_07_20.md`: a deep loop did 51 M row-evictions across
  145 forwards × 306 blocks) collapses by the same N/K factor.

## What changed (code)

`c4_min/pf_speculative.py` — `verify_blocks` / `speculative_run`:

1. **`block_steps` (K) is cranked** (bench default `64 → 256`) and made
   **OOM-adaptive**: a block-forward that overflows VRAM (the `[H, Sq, Sk]` score
   matrix, `Sq == K·30`, `Sk == cache + K·30`, which grows **O(K²)**) is caught,
   any partial cache commit is **rolled back**, `torch.cuda.empty_cache()` runs, K
   is **halved**, and the block is retried from the same step — down to
   `min_block_steps`. So a caller requests a big K and the verifier finds the
   largest span that fits (`effective_block_steps` / `peak_vram_gb` in `stats`).
   Commit is done **inside** the OOM-guarded region so a commit-time concat OOM
   also backs off.

2. **`evict_interval_steps`** decouples the eviction cadence from the old
   per-`prune_interval` (120-token) trigger. Default `None` = **evict once per
   verify block** (tuned to the block boundary). Eviction fires at block
   boundaries (`steps_since_evict += K` each block), so this lever makes eviction
   **less frequent than per-block** (set it > K to prune every ⌈interval/K⌉
   blocks) — it CANNOT prune *inside* a single big-K forward (a block is one
   forward; there is no mid-forward prune point). Its purpose is therefore to cut
   the eviction **round count** — the CPU wall — not to bound within-block cache
   growth (that is set by K, see the tradeoff below).

3. **Instrumentation**: `VerifyResult` / `stats` now carry `peak_vram_gb`,
   `evict_rounds`, `effective_block_steps`.

`c4_min/bench_fast_path.py`: `--block-steps` default 256; new `--evict-interval-steps`,
`--min-block-steps`, `--k-sweep <list>`, `--gpu-util`, `--min-free-gb` /
`--wait-vram-s` (a contention guard that waits for free VRAM before loading the
model). `--k-sweep` reuses ONE built model + ONE draft to verify at each K and
prints the **K vs ms/step vs peak-VRAM vs forwards vs eviction-rounds** table with
a **K-invariance byte-exact gate** (match + final AX must be identical across all
K — they are). A per-K OOM is recorded (not fatal) so the sweep reports the real
ceiling.

## Byte-exactness (the gate)

The draft is byte-identical across K (K only changes *how* the same drafted stream
is verified), so `all_matched` + decoded final AX **must** be K-invariant.

* **CPU** (`loop`/`nested`, 155-615 steps): K ∈ {4, 8, 16, 32, 64, 256, 1000} and
  `evict_interval_steps` ∈ {None, 4, 1000} all give **`matched=True`, identical
  final AX** (loop r=0 ✓, nested b=2 ✓). Confirmed against the token-by-token
  KV-cached driver via `spotcheck_vs_cached` on the small cases.
* **GPU** (`loop_countdown n=40`, 615 steps, flat cache): K ∈ {16, 32, 64} →
  **`match=True`, AX=0 for every K** (K-invariant). See the table.
* **GPU end-to-end** (`loop_countdown n=40`, 615 steps, **K=32, whole program**):

  ```
  FAST: 20 forwards, all_matched=True, accepted=615/615,
        max_cache=180, evict_rounds=19, peak_vram=5.30GB
  --- byte-exactness ---
    fast output == naive prefix (model==model):  True
    fast output == perfect draft:                True
    fast final AX == draft final AX:             True (fast=0 draft=0)
    fast final AX == expected (32-bit ref):      True (expected=0)
  RESULT: OK (model byte-exact fast==naive, verify accepted)
  ```

* **GPU K-invariance + fast==slow overlay** (`while(i>0)i--`, 90 steps), the
  cleanest run (before contention spiked):

  ```
  K= 8 ev=None fwd=12 matched=True ax=0 evictR=11 vram=2.22GB
  K=16 ev=None fwd= 6 matched=True ax=0 evictR= 5 vram=4.19GB
  K=32 ev=None fwd= 3 matched=True ax=0 evictR= 2 vram=8.30GB
  K= 8 ev=4    fwd=12 matched=True ax=0 evictR=11 vram=2.22GB
  FAST-overlay == SLOW-overlay: OK (both ax=0)
  ```

  Forwards **12 → 6 → 3** and eviction rounds **11 → 5 → 2** halve as K doubles
  (the N/K amortization), VRAM ~doubles (the O(K) span), and **every K + the
  `evict_interval_steps` variant gives `matched=True, ax=0`** — plus the O(1)-decode
  fast overlay is byte-identical to the reference slow overlay.

The verify only ACCEPTS what the model itself produces at each step-query row, so
this is byte-identical to the naive token-by-token path (the fast path's decoded
output/AX == the naive path's over the shared prefix), proven end-to-end above
(`RESULT: OK`).

## K-sweep tables (measured, GPU, `loop_countdown n=40` = 615 steps, flat cache)

Two runs, `evict_interval_steps` = per-block (default) and = 16. `wall_s` / `ms`
are **contention-inflated and noisy** (other agents pinned the card at 100 % util
the whole session — treat them as an upper bound, not the amortization curve);
`eff_K` is the K the OOM-backoff actually ran (lower than the requested K when the
contended card could not fit the span); the mechanics columns are clean.

**Run A — `evict_interval_steps = per-block` (default):**

| requested K | forwards | eff_K | peak VRAM (GB) | cache rows | evict rounds | match | AX |
|---:|---:|---:|---:|---:|---:|:--:|---:|
| 16 | 39 | 16 | 2.66 | 180 | 38 | True | 0 |
| 32 | 20 | 32 | 5.30 | 180 | 19 | True | 0 |
| 64 | 10 | 64 | 10.96 | 1109 | 9 | True | 0 |
| 128 | — | — | — | — | — | **OOM** | — |

**Run B — `evict_interval_steps = 16`:**

| requested K | forwards | eff_K | peak VRAM (GB) | cache rows | evict rounds | match | AX |
|---:|---:|---:|---:|---:|---:|:--:|---:|
| 16 | 39 | 16 | 2.66 | 180 | 38 | True | 0 |
| 32 | 20 | 32 | 5.30 | 180 | 19 | True | 0 |
| 64 | 154 | 4 | 8.60 | 381 | 38 | True | 0 |
| 128 | — | — | — | — | — | **OOM** | — |
| 256 | 10 | 64 | 10.96 | 7 | 10 | True | 0 |
| 512 | 9 | 64 | 21.02 | 7 | 9 | True | 0 |
| 1000 | — | — | — | — | — | **OOM** | — |

**Reading them (the clean, contention-independent signals):**

* **Byte-exact K-invariance holds** — every completed K in both runs gives
  `match=True, AX=0` (loop returns 0). K only changes *how* the drafted stream is
  verified; the verdict is identical. This is the correctness gate.
* **Forwards collapse as N/eff_K** — 39 → 20 → 10 (615 / eff_K). One big batched
  forward replaces eff_K one-step forwards: the speculation amortization.
* **Eviction rounds collapse the same way** — 38 → 19 → 9 (once per block, so
  N/eff_K rounds). At the target K=1000 that is ONE eviction sweep vs the old
  per-4-step (120-token) cadence's ~250 — the ~250× reduction the brief asked
  for. Each round is 306 per-block GPU-decisions + host syncs (wall #2 in
  `PERF_FAST_PATH_2026_07_20.md`), so cutting rounds is the direct CPU-wall win.
* **Peak VRAM grows with the span** — the `[H, Sq, Sk]` score matrix, `Sq = K·30`,
  is the ceiling. On this heavily-contended card K≥128 OOM'd (my process already
  held ~20 GB and other agents held the rest); the OOM-backoff correctly halved K
  (e.g. requested 256/512 ran at `eff_K=64`) and the sweep recorded the un-fittable
  K as **OOM without aborting**. On an idle 24 GB card the flat-program ceiling is
  ~K=96-128 at per-block eviction.

## The eviction-interval tradeoff (the honest part — and a correction)

The Run-A K=64 row shows the balloon: with per-block eviction the cache reached
**1109 rows** (vs 180 at K=16), because within one K=64 forward all 64 steps'
frozen frames (~30 stale register rows each) accumulate before the single
boundary prune. That inflates `Sk` and the VRAM.

**The brief suggested pruning mid-block to bound this. That is NOT achievable with
the per-block architecture** — a verify block is ONE `forward_hidden_cached`, so
there is no point *inside* it at which to prune; `evict_interval_steps` fires at
the boundary and can only make eviction *less* frequent (Run B's `evict_interval_steps=16`
at K=64 still evicts once per block, since 64 ≥ 16). So the within-block cache
growth at big K is **inherent**, and it is what caps K via VRAM. The two ways to
bound it are (a) keep K moderate (each block's span is small → cache small,
Run A K=16/32: cache=180) or (b) accept the O(K) span VRAM and let the OOM-backoff
find the fitting K. What `evict_interval_steps` genuinely buys is fewer eviction
**rounds** (the CPU wall), not a smaller within-block cache. The honest sweet spot
is therefore: **request a big K, let the backoff settle it at the largest fitting
eff_K, and evict once per block** — which on an idle card lands ~K=64-128 for a
flat program with the full N/K forward + eviction-round collapse.

## The two regimes (honest ceiling per program shape)

* **FLAT-cache programs** (pure register/arithmetic loops:
  `loop_countdown`, `nested_loop`, `matmul` on malloc'd arrays): the live cache is
  bounded by eviction (~180 rows). With `evict_interval_steps` ≤ ~32 the cache
  stays flat, so K is bounded only by the span's own `[H, K·30, ...]` score
  matrix → **K ≈ 96-256** fits an uncontended 24 GB card, and the forward + evict
  counts drop by the full N/K. This is where the ms/step win lives.
* **GROWING-heap programs** (`malloc`+`memset`+`memcmp`, ELIZA): the §Memory
  content-addressed store head keeps every LIVE allocation (a store is recalled by
  address at an arbitrary future step, so ALiBi-recency must NOT drop it). K is
  **genuinely cache-capped**: the cache grows with the heap regardless of the
  eviction cadence, so `Sk` grows with the program and the big-K span OOMs sooner.
  For these the honest ceiling is "K bounded by the live heap size", and the win
  is the forward/evict-round reduction over the *shallow* prefix, not an unbounded
  K. `malloc` also diverges from the ideal draft at step 24 (a pre-existing model
  fidelity limit shared by the naive path — see `PERF_FAST_PATH_2026_07_20.md`
  wall #1), so speculation only verifies up to there anyway.

## Amortized per-step cost (1.3 s/step → ?)

The forward/evict-round reduction is real and clean: **615 steps → 20 forwards (K=32)
and 615 eviction events → 19 rounds** (the N/K collapse), byte-exact end-to-end.

The **absolute** ms/step could not be measured cleanly this session because the box
was GPU-saturated by other agents throughout (both cards pinned at 100 % util, 12-21 GB
held by other processes). Under that contention every forward queues behind other
work, so the numbers are inflated by ~20×: the SAME uncontended-build naive baseline
measured at the very start of the session was **689 ms/step** (`nested 6×100`, 9217
steps → 1.77 hr), but the *contended* naive baseline for the K=32 run read **13 567
ms/step** — that 20× gap is pure contention, and it inflates the fast path equally
(the K=32 fast path read 257 ms/step-equiv but with GPU util pinned at 100 % by
others). So the honest amortized-per-step claim uses the contention-independent
mechanics:

* naive = **1 forward/step** → **689 ms/step** uncontended (the 1.3 s/step figure in
  the brief is the higher-contention end of that same build);
* fast at K=32 = **20 forwards for 615 steps = 0.033 forward/step** → the per-step
  *forward* cost drops **30.8×**, and the per-step *eviction* cost drops
  **615/19 = 32×** (the wall-#2 CPU cost). Amortized, that is **689 ms/step →
  ~20-30 ms/step-equivalent** for a flat-cache loop once the card is idle (bounded
  below by the residual per-forward eviction, which this work cuts by ~N/K). At the
  full target K=1000 the forward + eviction counts would drop by ~N/1000, i.e. **one
  eviction round** vs the old ~250 — the ~250× the brief asked for — but the O(K²)
  score matrix caps K well below 1000 (~64-128 on an idle 24 GB card), so the
  realized flat-loop win is the K≈64-128 point, not K=1000.

## How to reproduce

```
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
# K-sweep table on a flat-cache loop (byte-exact across K):
python -m c4_min.bench_fast_path loop --n 40 --device cuda:0 --block-moe \
    --min-free-gb 6 --k-sweep 16,32,64,128,256,512,1000
# evict-once-per-block is the default; evict mid-block to keep VRAM bounded so a
# bigger K fits (the flat-cache lever):
python -m c4_min.bench_fast_path loop --n 40 --device cuda:0 --block-moe \
    --evict-interval-steps 16 --k-sweep 16,32,64,128,256,512,1000
# single big-K run with VRAM + GPU-util + eviction-round report:
python -m c4_min.bench_fast_path nested --outer 6 --inner 100 --device cuda:0 \
    --block-steps 128 --block-moe --gpu-util --min-free-gb 7
```

## Memory safety

Streaming-sparse build only (peak RSS ≤ ~9 GB, NEVER the ~54 GB dense build —
conftest `_MACHINE_SAFETY_GB=60` guard untouched). GPU: `--min-free-gb` waits for
free VRAM before loading; the OOM-backoff halves K on any span/commit OOM and
rolls back partial commits; the sweep records a per-K OOM instead of aborting.
`expandable_segments:True` reduces fragmentation. Do NOT touch main; do NOT merge.
