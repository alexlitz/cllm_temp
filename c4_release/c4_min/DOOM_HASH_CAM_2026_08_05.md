# Doom live-CAM attention — genuine O(1) HASH-CAM (C4_HASH_CAM) + the fast-path cost verdict

2026-08-05. Attacks the "live-CAM memory-read attention" wall (0.69 µs/step, ~28% of the
composed step, called out as the next lever after the FFN wave-batch) with the user's
"have the attention find the first matching address" idea realized as a genuine O(1)
FNV-1a hash-CAM. **The load-bearing finding is a cost attribution that decides where a
hash can help.** Golden `069cc32f` unchanged (lever default-OFF, runtime-only).

---

## 1. THE 0.69 µs COST ATTRIBUTION — is it search/gather, or fixed head machinery?

Reproduced the dispatch profile (`_agent_dispatch_profile.py`) on an idle A5000 (GPU 0),
`nested_120_255` = 462,979 DIV-free steps, chunk 65536, full composed stack:

| phase | µs/step | % of step |
|---|---|---|
| mega dead-FFN chain (238 blk) | 1.699 | 68.8 |
| **live-CAM (3 blk scatter+FFN)** | **0.6945** | **28.1** |
| block-0 FFN | 0.081 | 3.3 |
| decode lanes | 0.025 | 1.0 |
| launch / graph-replay | ~0 | ~0 |

The live blocks are **block 2 (mem-cam, n_out=6), block 7 (stack-pop-cam, n_out=16),
block 11 (code-select, n_out=32)** — H=24 heads each, HD=59.

**Split of the 0.69 µs** (`_agent_livecam_split.py`, isolated CUDA-graph timing of the
3 live blocks' region):

| component | µs/step | share |
|---|---|---|
| (a) W_o-delta **SCATTER-add** (the resolved value → residual) | 0.053 | **19%** |
| (b) block **FFN GEMM** (SwiGLU) | 0.220 | **81%** |

### VERDICT: the fast-path 0.69 µs is FIXED head/FFN machinery, NOT a search/gather.

- There is **NO O(S) softmax scan and NO gather-over-stores at dispatch.** On the composed
  single-dispatch path the address→slot→value resolution is *precomputed at BUILD time* into
  the compact `W_o` delta (`precomputed_schedule._LiveCamBlock.forward_static_delta` = a
  scatter-add of the precomputed delta + the block FFN). The dispatch profiler confirms
  "gathers cost ~0 at dispatch (precomputed deltas)".
- The BUILD-time resolution is **already an O(1) hash** — `nibble_evict_schedule.resolve_load_rows`
  keeps a `latest: Dict[int,int]` (a Python hash table) of the latest store frame per address,
  latest-write-wins; `latest.get(addr)` is O(1). There is no binary tree, no O(log S) on this
  path.
- So **a hash-CAM cannot reduce the fast-path 0.69 µs.** (a) is a scatter of 1-2 heads'
  resolved values into ≤32 residual dims; (b) is a dense SwiGLU. Neither is address resolution.

### What fewer / structure-exploiting heads (#842) would buy on the fast path — quantified

The live blocks carry H=24 heads but the composed dispatch **already drives the ~22 dead
local heads to zero** (`direct_cam_batched.py:496-499` `_live_local` filter — a `_zero_attn`
head outputs exactly 0 and is skipped), so only the **1-2 real CAM heads** contribute the
scatter. Head reduction is therefore *already done*: the per-block cost is
`scatter(1-2 heads) + FFN`. A further head cut can only touch the **0.053 µs scatter (19%)**,
not the **0.220 µs FFN (81%)**, which is head-count-independent. The honest fast-path lever is
**fusing the live-block FFN into the mega-chain / improving the sparse-FFN kernel efficiency**
(the whole step runs at ~20% of HBM peak — kernel-efficiency-bound), **not a hash and not
fewer heads** (heads already minimal). Ceiling of a head-only fast-path lever ≈ −0.05 µs/step
(≈ the already-measured in-place-scatter micro-win), i.e. negligible.

---

## 2. WHERE A GENUINE O(1) HASH-CAM ACTUALLY HELPS — the FAITHFUL VALUE re-resolution

The one place a non-O(1) resolution survives is the **genuinely-computing / faithful path**
(`C4_FAITHFUL_SINGLE_DISPATCH` / `C4_FAITHFUL_ATTN_EVICT`), which INDEPENDENTLY re-derives
each read's value at the MODEL's own decoded address over the committed stores (rejecting the
audit's planted-value scenario E). That resolver — `faithful_single_dispatch._genuine_value_at`
— is a `np.lexsort` (O(S log S)) + two `np.searchsorted` bisects (O(R log S)): a
**sorted-array binary search**, i.e. exactly the "O(log S) tree" a hash can beat.

`c4_min/hash_cam.py` (`C4_HASH_CAM`, default OFF) is a genuine O(1) address hash-CAM:
- **`build_hash_index`** — one O(S) pass: an **FNV-1a** address hash (the #828 WAD-name-hash
  constant lineage: offset 2166136261, prime 16777619) into a **vectorized open-addressing**
  probe table (load factor ≤ 0.5, linear probing → collision chains bounded by the live
  working set). Per unique address it keeps that address's store slots in ascending frame
  order.
- **`resolve_value_hashed`** — per read: hash the model's decoded address → its bucket in
  **O(1)** (independently, from the address bits, NOT injected), then latest-write-wins = the
  bucket's store with the greatest frame < the read frame (a bisect over that ONE address's
  short slice, ~1–22 writes for doom's hottest slot — O(1) amortized in the total store count
  S). 0 == ZFOD (the softmax1 +1 sink) if the address is unwritten.

Wired into `build_faithful_precompute` (the faithful value-verify precompute) behind
`C4_HASH_CAM`; OFF → the searchsorted `_genuine_value_at` (golden faithful path).

### (a) BYTE-EXACT (`_agent_hash_cam_byteexact.py`, CPU-only)

`resolve_value_hashed` == `_genuine_value_at` **element-for-element, L-inf=0**, over random
store-logs (200 → 20k stores) AND doom-like logs (50k, 200k stores: hot slots written many
times + a cold single-write tail). Integrated into the faithful path,
`_agent_faithful_precompute_cache_equiv.py` gives a **byte-identical FaithfulPrecompute +
verdict** with `C4_HASH_CAM=1` vs OFF.

### (b) GENUINE — rejects the planted-value scenario E (independent, not draft-trusted)

Audit scenario E: true committed `mem[200]=66`; a self-consistent wrong draft plants value
`777`. The hash resolver INDEPENDENTLY recomputes latest-write-wins from the address bits →
**66, REJECTS the planted 777** (identical to the searchsorted resolver, unlike the
draft-trusted direct-CAM which accepts). Value-stale layer (store 66 then 999, draft injects
stale 66) → hash recomputes the latest **999**, rejects the stale 66. In the integrated
faithful path both the **address layer (`cam_addr`) and value layer (`cam_value`) still
REJECT** the wrong draft with `C4_HASH_CAM=1`. **Genuine O(1) read confirmed.**

### (c) TIMING — does O(1) hash beat O(log S) searchsorted?

Per-frame RESOLVE cost (R=20k reads, doom-like log; `_genuine_value_at` re-lexsorts the whole
log every call, the hash lexsorts once at build):

| S stores | uniq addr | searchsorted resolve ms | hash build ms | hash resolve ms | **resolve speedup** |
|---|---|---|---|---|---|
| 2 000 | 810 | 1.75 | 0.21 | 1.50 | 1.16× |
| 20 000 | 8 010 | 3.21 | 1.38 | 2.12 | 1.52× |
| 100 000 | 40 010 | 10.49 | 10.74 | 3.08 | **3.41×** |
| 400 000 | 160 011 | 40.32 | 53.99 | 4.73 | **8.54×** |

The per-frame resolve is **1.2–8.5× faster and grows with S** (the O(1) hash vs O(log S)
bisect). The hash **build** (lexsort + vectorized probe-insert) is comparable to one full
searchsorted call, so the net win requires the build to amortize — which it does: in a
continuous render `C4_FAITHFUL_PRECOMPUTE_CACHE` already caches the whole precompute (the
build is one-time after frame 0), so both build and resolve become one-time and the per-frame
faithful value-verify is the cached lookup.

### The HONEST caveat on the fps impact

The faithful value re-resolution is **already off the dispatch critical path** — the
`FAITHFUL_SINGLE_DISPATCH` work pipelined it onto the GIL-releasing build thread and vectorized
the critical-path compare to **0.005 µs/step** (298× cut), and the whole precompute is cached
per render. So the hash's per-frame resolve win (1.2–8.5×) accelerates a build-thread cost that
is *already hidden under the next frame's dispatch and cached*. **Net fps impact ≈ 0** on the
composed real-time path — the hash makes the genuine value re-resolution asymptotically O(1)
(matters for larger working sets / an un-cached / un-pipelined faithful build), but it does
NOT move the 2.67 fps fast path (fixed FFN machinery, §1) and does NOT move the pipelined
faithful fps (the resolution was already hidden).

---

## 3. BOTTOM LINE

- **The 0.69 µs live-CAM wall is FIXED head/FFN machinery** (19% W_o-delta scatter + 81% block
  FFN GEMM), **NOT a search/gather.** The address resolution is already O(1) (a latest-write-wins
  hash dict at build) and ~0 at dispatch. **A hash-CAM does not help the fast path.** The fast-path
  lever is FFN-kernel-efficiency / mega-fusing the live-block FFN; **fewer heads (#842) is already
  done** (dead heads driven to 0 → only 1-2 CAM heads' scatter, 19%; the 81% FFN is
  head-independent, ceiling of a further head cut ≈ −0.05 µs/step).
- **`C4_HASH_CAM` gives a genuine O(1) independently-verified read on the FAITHFUL path** —
  byte-exact (L-inf=0), rejects the planted-value scenario E, and is 1.2–8.5× faster than the
  O(log S) searchsorted resolve (growing with the store-log). It converts the faithful value
  re-resolution's `np.searchsorted` (O(log S) sorted-array bisect) into a genuine FNV-1a O(1)
  address probe.
- **But the faithful value re-resolution was ALREADY pipelined + cached off the critical path**
  (0.005 µs/step), so the hash's real-time fps impact is ≈ 0. It is an asymptotic correctness /
  scalability improvement to the genuine read, not a real-time speed lever.

## Golden / VRAM
- Flag-OFF golden **UNCHANGED: `069cc32f`** (`CUDA_VISIBLE_DEVICES="" python -m
  c4_min._fingerprint_build`). `C4_HASH_CAM` is a runtime resolver swap in
  `build_faithful_precompute` — no weight authoring, weight-neutral flag-ON.
- Cost-attribution VRAM peak (dispatch profile, chunk 65536): **~0.44 GB** resident graph
  (well within budget; profiler reported free 25.0/25.3 GB throughout). The hash resolver is
  pure CPU/numpy (no VRAM).
