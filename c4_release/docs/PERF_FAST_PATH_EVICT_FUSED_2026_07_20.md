# Fast path — FUSED KV eviction (the #667 deep-loop bottleneck removed)

**Branch:** `fast-path-evict-fused-2026-07-20` (off `fast-path-perf-2026-07-20 @ 4eae7770`)
**Date:** 2026-07-20
**Follow-up to:** `PERF_FAST_PATH_2026_07_20.md` (wall #2: "the fast path is CPU-bound
on EVICTION for deep programs").

## The bottleneck (measured in the base doc)

The perfect-draft speculation + block-MoE fast path (`verify_blocks`) reduced the
deep `nested_loop 6×100` (9 217 steps) from **91.3 min naive → 9.11 min** (10×), a
**63.6×** forward reduction — but the wall speedup was capped at 10× because
**eviction was CPU-bound**: GPU util ~41 %, and the run did **51 080 642
row-evictions across 145 forwards × 306 blocks**. The base doc named the fix
exactly: *"batch the eviction across blocks (one fused GPU decision for all 306
caches instead of 306 host-synced calls)."* This is that fix.

### Why the old prune stalled the GPU

Every `prune_interval` tokens the driver ran a **Python loop over all 306 block
caches**, and each `BlockKVCacheBatched.evict` → `evict_keep_index`:

  * looped over the block's **23 heads in Python**, and
  * forced **multiple device→host syncs per block** — `head_has_value.tolist()` +
    `cm_frac.tolist()` + `slopes.tolist()` (to pick the per-head metric on the host)
    plus a final `torch.nonzero()` / `.numel()`.

So each prune fired **306 × (≥3 + 2) ≈ 1 500 host round-trips**, and on every one the
GPU sat idle waiting for the CPU. That is the 41 % util.

## The fix — `evict_all_blocks_fused` (one batched on-GPU decision)

New in `nibble_pure_forward_cached.py`:

  * **`prune_keep_mask_batched`** — the EXACT four-mechanism policy of
    `prune_keep_mask_head` (near-dup supersession / ALiBi-recency horizon / dead-value
    head / free-zero), but computed for **N = (block × head) groups at once** over a
    leading batch axis, with per-group `slope` / `dup_metric` (cosine vs exact) /
    `content_addressed` VECTORS and per-group `positions` `[N,S]`. Every mechanism is
    a batched tensor op; the greedy newest-first near-dup selection
    (`_greedy_survivors_batched`) is a vectorised recency-rank fixpoint over N. **No
    per-group Python, no per-group host sync.**
  * **`evict_all_blocks_fused`** — groups the live block caches by size `S` (they
    share `S` in the lockstep-commit common case → ONE group), stacks each group's
    blocks × heads into ONE `[N,S,HD]` tensor, computes the per-(block,head) metric
    (`head_has_value`, `cm_frac` → exact/cosine) as a **batched reduction with NO
    `.tolist()`**, runs `prune_keep_mask_batched` once, and UNIONs across a block's
    heads on-GPU → a `[n_blocks, S]` boolean keep-mask that **never leaves the device
    before compaction**. Chunked (block-stack + near-dup `[chunkN,S,S]`) to a VRAM
    budget so the rare large-`S` first prune is memory-bounded while the flat deep
    tail fuses all 306 blocks × 23 heads into ONE pass.
  * **`BlockKVCacheBatched.apply_keep_mask`** — compacts a block with the boolean mask
    via an on-GPU boolean-index gather.

The driver (`verify_blocks`) now calls `evict_all_blocks_fused` once per prune and
`apply_keep_mask` per block, replacing the `for b: caches[b].evict(...)` loop.

### The one unavoidable residual sync (honest)

Two host syncs remain, and both are structural, not per-block:

  1. **ONE `[n_blocks]` "did this block drop anything" boolean** per size-group
     (`dropped_any = (...).tolist()`) so the driver skips compaction of unchanged
     blocks — a single sync for all 306 blocks, replacing the old ~1 500.
  2. The **boolean-index compaction** `K[:, :, keep_mask, :]` needs the survivor
     count on the host to size the output tensor (the CUDA allocator requirement the
     mission called out). This is per block that actually drops, but it is a tiny
     bool-mask popcount, not the old whole-cache `.cpu()` + per-head Python.

## Measured — the eviction decision itself (isolated, contention-free)

`python -m c4_min.bench_kv_evict --all-blocks --device cuda:0` — a whole-model prune
of **306 blocks × 23 heads (HD=78)**, OLD per-block host-synced loop vs NEW fused:

| S (cache size) | OLD per-block loop | NEW fused | speedup |
|---:|---:|---:|---:|
| 7 (deep-tail flat) | **28 842 ms** | **5.3 ms** | **5 408×** |
| 15 | 39 594 ms | 18.5 ms | **2 140×** |

On CPU (no host-sync penalty, pure Python-loop elimination, 40 blocks): 39.5× / 11.8×.
The whole per-prune eviction cost collapses from **~30 s to ~5–18 ms**.

Two follow-on optimisations were needed to hit that (each measured on the real
306-block model, S=240 pre-flatten cache):

  * **cdist only for the EXACT (content-addressed) heads.** Computing the ULP-exact
    `donot_use_mm` cdist on EVERY head (then selecting) cost ~3.2 s/prune; the
    register-marker MAJORITY only needs the fast cosine matmul, so restricting the
    cdist to the exact-metric heads → **~81 ms/prune** (39×).
  * **fused compaction (ONE `nonzero` for the whole size-group).** Compacting each
    block with its own boolean-index gather re-introduced **306 per-block host
    syncs** (~1.5 s/prune) — the DECISION was fused but the GATHER was not. A single
    batched `nonzero` over the `[Lb,S]` group mask yields every survivor
    `(block,col)` pair at once; each block then compacts with a precomputed long
    index via a sync-free `index_select`.

**End-to-end in the real driver** (`verify_blocks`, small countdown loop, 306
blocks, S=240 per prune, `all_matched=True`, final AX byte-identical): the fused +
optimised prune runs at **~102 ms/prune vs ~1 427 ms/prune** for the naive fused
(pre-optimisation) and the per-block loop was ~41 600 ms/prune at that size — a
**~400× per-prune eviction speedup** at the pre-flatten cache size, and the deep
tail (flat S) is milliseconds.

## Byte-identity (the keep decision does NOT change)

* `test_kv_cache_equivalence.py::test_fused_all_block_evict_matches_per_block` —
  30 random multi-block batteries (divergent per-block sizes, mixed register-marker /
  content-addressed heads, protection positions): the fused survivor K/V/pos is
  **byte-identical** to the per-block `evict` loop.
* `prune_keep_mask_batched` vs the reference `prune_keep_mask_head`: **0 mismatches**
  over thousands of random trials (single-group, padded multi-group, per-group
  positions, both metrics, `content_addressed` on/off, budget-forced chunking).
* The end-to-end `bench_fast_path` gate (`fast output == naive prefix`,
  `fast final AX == draft final AX`) stays green.

## GPU utilisation before/after

* OLD (base doc): **~41 %** during a deep verify (GPU idle waiting on the per-block
  eviction host syncs).
* NEW: sampled **99 % SM** for the process during the fused FAST verify on an
  UNCONTENDED GPU (`nvidia-smi pmon`). The GPU no longer stalls on eviction.

## Wall speedup

The eviction stall is removed; the deep-loop fast wall is now bounded by the ~145
block forwards themselves (the true speculative floor), not the prune. Under a
FREE GPU the fast wall drops toward the forward-count-limited bound. **NOTE (honest):
the shared benchmark box was heavily oversubscribed by other users' jobs during this
work (both 24 GB cards routinely at 100 % util / <4 GB free from external
processes), so a clean end-to-end wall number for the full 9 217-step run was not
reliably reproducible here — the absolute wall is contention-dominated. The
contention-ROBUST evidence is (a) the isolated 5 408× / 2 140× eviction-decision
speedup above, (b) the 99 % SM during the fused verify vs 41 % before, and (c) the
`t_evict / t_fast` split the instrumented bench prints (`FAST wall split: evict=… %`),
which shows eviction is now a small single-digit-% slice of the fast wall instead of
the dominant cost.** Re-run on an idle card to read the full wall speedup:

```
python -m c4_min.bench_fast_path nested --outer 6 --inner 100 --device cuda:0 \
    --block-steps 64 --block-moe --naive-steps 40
# prints:  FAST wall split: evict=… (…%, N prunes, … ms/prune)  forward+overlay=… (…%)
```

## What is now the remaining wall

With eviction fused, the deep-loop fast path is **forward-bound** (the ~forwards/K
block forwards on the 306-block streaming-sparse model), plus the Python frame-role
overlay (`apply_overlay_window_fast`, already O(rows+code)). The eviction — the
base doc's wall #2 — is no longer the bottleneck. The next lever is the per-forward
GEMM cost of the wide-span 306-block stack (sparse-vs-dense, or a leaner block set),
not the KV prune.

## Files

* `c4_min/nibble_pure_forward_cached.py` — `prune_keep_mask_batched`,
  `_greedy_survivors_batched`, `evict_all_blocks_fused`,
  `BlockKVCacheBatched.apply_keep_mask`.
* `c4_min/pf_speculative.py` — `verify_blocks` uses the fused prune + timing split.
* `c4_min/bench_kv_evict.py` — `--all-blocks` OLD-vs-NEW whole-model prune bench.
* `c4_min/bench_fast_path.py` — prints the `evict / forward+overlay` wall split;
  frees the naive cache before the FAST phase.
* `c4_min/_bench_evict_fused.py` — lean fused-eviction validator (timing split +
  byte-identity, skips the multi-minute naive baseline).
* `c4_min/test_kv_cache_equivalence.py::test_fused_all_block_evict_matches_per_block`
  — the byte-identity gate for the fused prune.
