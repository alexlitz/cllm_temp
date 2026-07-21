# Bounded KV eviction on the LEAN compacted forward

`qwen_lean_evict.py` adds a **persistent-cache, memory-bounded** driver on top of
the lean forward foundation (`qwen_lean_forward.py`, #695). It keeps ONE growing
per-layer KV cache across VM steps and **prunes it so the cache stays bounded even
when a loop runs to millions of steps' worth of tokens** — the memory bound for long
programs. This doc is the design + the honest measured numbers.

## What it does (and why the lean RoPE model needs a different policy)

The naive lean driver (`qwen_lean_forward.run_program_lean`) rebuilds a FRESH
single-frame window every step (`past=None`). The persistent-cache driver here
(`run_program_lean_evict`) keeps one per-layer `(K,V,pos)` cache and only APPENDS the
new frame's rows each step — then evicts so the cache is bounded.

The compacted CAM (`qwen_full_vm._bake_register_cam`) was designed for a **single
frame per window**: within one window each register appears on exactly ONE token, so
a content-match (`CONTENT_GAIN=26`) alone selects it — *"the window IS the recency"*.
RoPE recency (`RECENCY_GAIN=3`) is a mild tie-break, **not** a strong ALiBi
`-slope·dist` suppressor. So on a persistent cache a stale prior frame's ROLE marker
is a same-strength content competitor. Measured, on this branch:

| # stale register frames in cache | decoded register read |
|---|---|
| 0 (the fresh-window design point) | **correct** |
| **1** | **corrupt** (wrong AX) |
| 2+ | corrupt (collapses to a default) |

So — unlike the deep ~128 GB ALiBi backend, where ALiBi recency cleanly suppresses
old frames and eviction is a pure optimization — on the lean RoPE forward some
eviction is **structural (required for correctness)**, and the rest is the
decouple-safe memory bound:

| row class | rule | why | when |
|---|---|---|---|
| **BOS sink** (pos 0) | keep forever | the ZFOD logit-0 sink (unwritten address reads 0) | — |
| **register / STEP_END frame** | keep ONLY the newest frame; supersede the prior one | RoPE can't suppress a stale ROLE marker → 1 stale frame corrupts the read | **per-step, before the decode** (structural) |
| **store row, same-address supersession** | drop the older same-address row when a newer store to it arrives | RoPE can't pick the latest among 2 physically-present same-address rows (an `lww` load reads the WRONG value with both in cache) | **at store-append** (structural, mirrors the naive driver's `store_log` compaction) |
| **store row, freed / zero-value** | drop it | a zero value adds 0 to the softmax numerator and is nil under the BOS sink — **read-tolerable** even un-pruned | **periodic / async** (the decouple-safe bound) |
| **store row, live non-zero, distinct address** | keep | the LIVE HEAP — retrieved by address at an arbitrary future step, never recency-decayed | — |

So the cache is `BOS + one register frame (6 rows) + live-heap store rows`. On a
register loop that is a **constant ~7 rows forever**; on a heap program it tracks the
live heap footprint (grows with distinct live addresses, shrinks on free).

## Decoupling the prune from the step loop

The register-frame supersession and same-address supersession are structural (cheap
boolean masks over ~13 rows). The **freed / zero-value prune** is the part that can
be DECOUPLED: an evicted freed row already loses the attention (nil under the BOS
sink), so pruning it now / later / never (VRAM permitting) is **byte-identical** —
the transient un-pruned rows change no decode (verified: a freed cell persists
un-pruned and later reads are byte-identical).

`AsyncPruner` runs that prune on a **separate CUDA stream**, watermark-triggered
(`watermark_rows`, so the transient un-pruned window can never exceed the VRAM
budget). The host-side keep decision (which rows are freed — a bookkeeping pass over
the python `meta` list) is computed, then the tensor GATHER is issued on the side
stream so it **overlaps the next step's forward** on the default stream; the
compacted cache is swapped in at the next step boundary (before the cache is read).

`run_program_lean_evict(evict=...)` has three modes:

- `"off"`  — grow the cache (structural supersession only; NO freed-row prune). The
  un-pruned baseline.
- `"sync"` — prune the freed rows inline every `prune_interval` tokens (blocking).
- `"async"`— prune on the decoupled CUDA stream, watermark-triggered.

## Byte-identity (the guarantee the bound rests on)

`run_program_lean_evict` decodes **byte-identical** to `run_program_lean` (the naive
fresh-window driver) in ALL three modes, on the arith / cmp / branch / loop / memory
/ latest-write-wins / distinct-heap / free battery — 45 cases in
`test_qwen_lean_evict.py`, all green, and the GPU spot-check
(`bench_lean_evict --only byte-identity`) confirms it on cuda. The transient
un-pruned rows change no output.

## Bounded cache on a LONG program (measured, cuda:1, `mem+cmp`, 10 layers)

**(a) register-only spin loop, 5000 steps** (a naive persistent cache would reach
~35,000 rows):

| evict | steps | max cache rows | evicted | ms/step | peak VRAM over base |
|---|---|---|---|---|---|
| off   | 5000 | **7** | 29,994 | 12.5 | +10.1 MB |
| async | 5000 | **7** | 29,994 | 13.6 | +10.1 MB |

The cache is **FLAT at 7 rows** the whole run (the per-step register supersession
bounds it to a constant), VRAM flat — this is the "millions of steps, bounded cache,
flat RSS/VRAM" result. (Same-cache in `off` and `async` here because the register
supersession is structural; the freed-row prune has nothing to reclaim on a
register-only loop.)

**(b) heap alloc+free churn, 5000 cells** (allocate a cell, then free it, repeat):

| evict | final cache rows | evicted | peak VRAM over base | cache-size trace |
|---|---|---|---|---|
| off   | **257** | 9,744 | +19.5 MB | `[2, 257, 257, 257, …]` (freed rows accumulate to the 256-addr space) |
| async | **21**  | 9,980 | +11.9 MB | `[2, 22, 42, 2, 22, 42, …]` (sawtooth — the prune reclaims freed rows) |

The async periodic prune keeps the freed-heap cache a small **sawtooth (2–42 rows)**
vs `off`'s 257 — the genuine unbounded-vs-bounded contrast (the address space caps
`off` at 256 here; on a wider heap it grows without bound).

## Is the prune even on the critical path? (honest answer)

**No — not for the compacted lean model.** The compacted cache is TINY (7–257 rows),
so the keep-mask + gather is a few percent of one forward (measured, cuda:1):

| cache rows | forward ms | keep_mask ms | mask+gather ms | prune / forward |
|---|---|---|---|---|
| 8    | 6.88 | 0.014 | 0.26  | **3.8 %** |
| 64   | 6.98 | 0.084 | 0.33  | **4.8 %** |
| 257  | 7.02 | 0.33  | 0.64  | 9.2 % |
| 1024 | 7.12 | 2.45  | 3.01  | 42 % |
| 4096 | 6.99 | 9.38  | 11.0  | 158 % |

At the register / small-heap regime the compacted model actually runs, the prune is
**negligible** — this is UNLIKE the ~128 GB deep ALiBi backend where the prune was
~17 % (its live-heap cache reaches ~1e5 rows). **The value here is the MEMORY BOUND
for long programs, not ms/step.** The prune only becomes significant at 1k+ rows,
where the O(S) python `meta` keep-mask loop dominates — and that is exactly where the
async side-stream overlap earns its keep (the gather runs concurrently with the next
forward). On the small caches the async mode carries a small (~1 ms/step) fixed
overhead from the per-step side-stream sync, so on the compacted model `off`/`sync`
are marginally cheaper; async is the right default only once the cache is large
enough that its gather is worth overlapping.

## Entry points

```python
from c4_min import qwen_full_vm as Q, qwen_lean_forward as LF, qwen_lean_evict as EV
vm   = Q.build(code_size=24, subset=Q.SUBSET_MEM_CMP)
lean = LF.LeanQwenVM.from_full_vm(vm, device="cuda:1")
stats = EV.run_program_lean_evict(lean, code, max_steps=10_000, evict="async",
                                  prune_interval=120, watermark_rows=2048)
# stats.exact / stats.ax_trace / stats.max_cache_rows / stats.cache_size_trace / ...
```

- `run_program_lean_evict(lean, code, *, evict, prune_interval, watermark_rows, …)`
  — the persistent-cache + bounded-eviction driver → `EvictStats`.
- `LeanKVCache` — the per-layer `(K,V,pos)` cache + the supersession / freed-row
  keep-mask + `supersede_store_addr`.
- `AsyncPruner` — the decoupled-CUDA-stream freed-row prune with a VRAM watermark.

## Tests + bench

- `test_qwen_lean_evict.py` — 45 cases: byte-identity vs the naive driver in
  off/sync/async on the full battery, the register-loop bound, the async freed-row
  reclaim, the structural same-address supersession.
- `bench_lean_evict.py` — the bounded-cache demo + the honest prune-cost table +
  the GPU byte-identity spot-check. Run:
  `python -m c4_min.bench_lean_evict --device cuda:1`.
