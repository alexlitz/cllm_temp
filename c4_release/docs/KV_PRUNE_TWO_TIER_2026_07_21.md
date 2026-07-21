# Two-tier KV eviction prune — O(S²) cosine → O(S) hash + tiny cosine residual (2026-07-21)

Branch `kv-prune-hash-two-tier-2026-07-21` (off `combine-fastpath-evict-lea-2026-07-21
@ 77efe6d9`, #683).  Makes the fused KV eviction prune — the **dominant remaining
fast-path cost** (#683: ~30–36 % of the deep-loop wall, ~1.5 s/prune) — cheap by
splitting mechanism-1 (near-duplicate supersession) into two tiers.

## 1. The prune profile (what dominated, confirmed)

The fast path (`pf_speculative.verify_blocks`) prunes every ~48–64 steps via
`evict_all_blocks_fused` → `prune_keep_mask_batched` (one fused on-GPU pass over all
~307 blocks × 23 heads).  Its dominant cost is **mechanism 1**:

* a full `[N, S, S]` **cosine near-dup matmul** (`unit @ unitᵀ > 0.99`), then
* a `[N, S, S]`-per-pass **greedy survivor fixpoint** (`_greedy_survivors_batched`),

where `S` is the WHOLE verify span at a prune (K steps × ~35 tokens ≈ **1 400–1 900
rows**), `N = block×head` (chunked).  Both parts are **O(S²)** and grew with the span.
Measured (this branch, cuda:1, kill-switch `C4_KV_PRUNE_FULL_COSINE=1` = the original
path): nested 3×60 = **86.8 s evict (36.4 %), 1 497 ms/prune**; nested 6×100 headline =
matches #683's **282 s (36.5 %), 1 470 ms/prune**.

### Why hashing wins — the exact-duplicate register frame

The VM re-emits the **same ~30-token register/marker frame every step**.  A key is
`W_k @ x` and `x` is **position-free** (ALiBi injects position only into the attention
*score*, never the key), so a given (role, value) key is **BIT-IDENTICAL across the K
steps that emit it**.  Over a K-step span each of the ~30 distinct role/value keys
appears ~K times **verbatim** → the vast majority of the S rows are exact-duplicate
register frames, evictable by an **O(S) hash**, not the O(S²) cosine.  Only genuine
*distinct* keys (~30) need the cosine — a tiny residual.  (Measured deep-loop caches
collapse to ~7–960 distinct keys out of 1 400–1 900 rows.)

## 2. The two-tier fix (`_mech1_cosine_survivors_dedup`)

For the COSINE (register-marker) groups — the majority — mechanism 1 is:

* **TIER A — O(S) exact-duplicate collapse.**  A per-group lexicographic sort over the
  HD key columns makes verbatim-identical keys adjacent → a per-group **class id**.
  Only the **newest** member of an exact-dup class can survive mechanism 1 (an older
  exact-dup has cosine 1.0 > 0.99 with its newer twin, so the greedy drops it; and if
  the newer twin is dropped by an even-newer near-dup, that near-dup is cosine>thr with
  the older twin too — same key direction — so the older twin drops as well).  Every
  non-representative is dropped for **free**.  ZERO-KEY and padded rows are forced into
  their own singleton classes (a zero key has cosine 0 with *everything*, so the
  reference never near-dup-drops it — even vs another zero key).
* **TIER B — O(R²) cosine on the tiny residual.**  Compact the representatives to
  `[N, R, R]` (`R` = #distinct keys, ~30 ≪ S), run the cosine matmul + greedy fixpoint
  there, scatter the per-representative keep decision back.  `R²` chunked to a memory
  budget so a rare non-collapsing prune (R≈S) never blows VRAM.

The result is **BYTE-IDENTICAL** to the full-cosine greedy (exact-dup is a subset of
cosine near-dup, and a class shares one representative's cosine decision).

The EXACT-metric (content-addressed §Memory) groups — a small subset whose *distinct
heap addresses are not verbatim-equal* — keep the original ULP-exact `cdist` path.

### The kernel-launch fix (why the win only lands with a bigger chunk)

The two-tier path NEVER materialises `[*, S, S]` — its peak is the `[chunkN, S, HD]`
sorted-key tensor + the tiny `[chunkN, R, R]` residual — so its memory bound is `S·HD`,
not `S²`.  `evict_all_blocks_fused` therefore raises `chunkN` for the default path from
`24M/S²` (≈ 12 groups → hundreds of tiny chunks/round, each re-launching the sort) to
`24M/(S·HD)` (≈ one chunk/round).  Without this the 8-pass sort's **kernel-launch
overhead** across hundreds of tiny chunks made the two-tier *slower*; with it the win
lands.

`C4_KV_PRUNE_FULL_COSINE=1` is a byte-identical kill-switch / A-B baseline (the
original full path).

## 3. Byte-identity (the survivors are the SAME)

* `test_kv_cache_equivalence.py` — **16/16 pass**, incl. the pre-existing gates
  (`test_vectorized_prune_matches_reference` 720 trials, `…_large_and_content_addressed`,
  `test_fused_all_block_evict_matches_per_block`) plus two NEW gates:
  `test_two_tier_mech1_dedup_matches_full_cosine` (400 trials: exact-dup, near-cosine,
  zero-key, parallel-distinct, padded) and
  `test_batched_prune_two_tier_matches_full_cosine_kill_switch` (end-to-end two-tier ==
  kill-switch through all mechanisms incl. exact groups).
* Every fast-path bench below: **`all_matched=True`**, `fast output == naive prefix`,
  identical `max_cache` / `evicted` count → **byte-identical decode**.

## 4. Benchmark — prune time + ms/step before/after (cuda:1, sparse_mm, K=48)

Kill-switch `C4_KV_PRUNE_FULL_COSINE=1` = OLD; default = NEW.  forward+overlay wall is
identical (the prune change touches ONLY eviction).

All OLD/NEW pairs below were measured back-to-back on the SAME cuda:1 card (serialized,
GPU-0 idle — no contention) so they are apples-to-apples.

| program            | evict OLD          | evict NEW          | ms/prune OLD→NEW | ms/step OLD→NEW | byte-exact |
|--------------------|--------------------|--------------------|------------------|-----------------|-----------|
| loop n=64 (975 steps)               | 29.0 s (36 %)* | **10.3 s (16.7 %)** | 1 450* → **517** | 82.0* → **63.7** | ✓ |
| nested 3×60 (2 818 steps)           | 86.8 s (36.4 %) | **30.5 s (16.7 %)** | 1 497 → **525**  | 84.5 → **64.6** | ✓ |
| nested 6×100 (9 217 steps, headline)| **312.6 s (38.4 %)** | **100.4 s (16.8 %)** | **1 628 → 523** | **88.3 → 64.8** | ✓ |

*loop OLD = #683 `CLEAN_PERF_2026_07_21.md` (same K, sparse); the 3×60 and 6×100 OLD are
this branch's own kill-switch (`C4_KV_PRUNE_FULL_COSINE=1`) re-measure on the same card.

**The prune is ~2.8–3.1× faster** (headline **312.6 s → 100.4 s**; 3×60 87 s → 30 s;
loop 29 s → 10 s), its share of the wall drops from ~36–38 % to ~17 %, and **ms/step
falls 88 → 65 (−27 % headline; −22 to −24 % on the loops)**.  The forward+overlay wall is
UNCHANGED (headline 501 s OLD vs 497 s NEW), confirming the change touches ONLY eviction.
The wall is now **83 % forward+overlay** — the prune is no longer the bottleneck; the
remaining ceiling is the block forward (the sparse-CSR matmul), as targeted.  Survivor
set is byte-identical (same `max_cache=7`, same 82 118 186 evicted, `all_matched=9217/9217`).

### Honest residual
The prune is not free: ~17 % of the wall remains.  Of that, TIER A's `S·HD`
lexicographic sort is O(S·HD·log S) and TIER B's cosine over the ~R distinct keys is
irreducible (genuine near-duplicate MEMORY merges at cosine>0.99 that are *not*
verbatim, plus the ULP-exact `cdist` on content-addressed §Memory heads, still run).
On a NON-collapsing loop (3×60, R up to ~960) the win is smaller than on a fully
collapsing loop (6×100 headline, R≈7) but still ~2.8×; the two-tier never regresses the
full-cosine baseline now that `chunkN` is raised for the O(S·HD) path.

## Files
* `nibble_pure_forward_cached.py` — `_recency_rank` (factored), the two-tier
  `_mech1_cosine_survivors_dedup`, the `_mech1_full_cosine_survivors` kill-switch
  helper, the routed `prune_keep_mask_batched`, and the `chunkN` raise in
  `evict_all_blocks_fused`.
* `test_kv_cache_equivalence.py` — two new byte-identity gates.
* Bench/prof tools: `_bench_batched_prune.py` (micro A-B), `_prof_prune.py` (real
  fused-prune profiler).
