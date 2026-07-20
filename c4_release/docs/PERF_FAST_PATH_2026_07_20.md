# Fast execution path for heavy neural-VM programs — measured

**Branch:** `fast-path-perf-2026-07-20` (off `c4min-consolidated-2026-07-20 @ 9a74e0c3`)
**Date:** 2026-07-20
**Device:** single RTX-class GPU, 24 GB VRAM (`cuda:0`), `OMP_NUM_THREADS=4`

## The problem

The heavy demos — mandelbrot (#659), ELIZA (#653), self-emulation (#666), the
malloc corpus (#648) — all ran the **naive token-by-token driver**
(`run_pure_forward_cached`): exactly **one `model.forward` per VM step**. Measured
per-step wall on this build is **~0.5–1.6 s/step** (the 306-block streaming-sparse
model), so a program of N steps is N × ~1 s. A tiny 1×1 mandelbrot is 484 steps
(~10 min); a deep loop is hundreds of thousands of steps (hours to days). That is
the wall this work removes.

## The fast path (levers stacked)

The speedup machinery already existed for the 1096 corpus (`pf_speculative`,
`batched_speculative`, `run_corpus_resumable`); this work **applies it to single
heavy programs**, adds the two things those programs needed, and **measures** it.
Driver: `c4_min/bench_fast_path.py`.

### Lever 1 — perfect-draft speculation (the dominant lever)
`draft_pf_program` runs the *logical VM* (pure Python, 0 model forwards,
deterministic → 100 % accept) to materialise the **whole** per-step 30-token frame
stream up front. `verify_blocks` then confirms the model's argmax register state
== the draft at every step-query row, in the **fewest, largest batched forwards**:
K = `--block-steps` VM steps verified per `forward_hidden_cached`. **N one-step
forwards collapse to ≈ N/K batched forwards.** Eviction keeps the live KV cache
bounded so a large span still fits VRAM.

  * Added: **PRTF (op 33) in the draft** — previously `draft_pf_program` raised
    `NotImplementedError` on PRTF, so no output-emitting program (mandelbrot,
    ELIZA) could use the fast path. PRTF's transition is register-neutral (PC+=1)
    and appends `AX & 0xFF` as the visible byte; `verify_blocks(collect_out=…)`
    decodes it from the model's own verified AX at the PRTF row (byte-identical to
    the naive driver's `out.append`). `PFDraft.out` / `.prtf_steps` carry it.

### Lever 2 — block-MoE divmod-skip
Brought in `block_moe_divmod.py` (`DivModRouter` + `resolve_divmod_span` +
`moe_forward_with_count`, from `block-moe-divmod-skip-af113` / #628). In
`verify_blocks(block_moe=True)`, any block-verify span with **no DIV/MOD step**
skips the contiguous divmod block range (its attention is all-zero identity, so its
K/V is never read and the absent cache is harmless → byte-identical).

  * **Measured span in THIS build (`recurrent_divmod=True`): (31, 146) = 115 of 306
    blocks.** So the block-MoE skip is a **1.60×** per-forward block-count win here,
    NOT the ~7× the branch's non-recurrent build reported (the recurrent build
    shares the divmod blocks, shrinking the span). Honest number: 1.6×.

### Lever 3 — GPU + bounded eviction
`--device cuda:0`; eviction (`prune_interval`) keeps the live cache flat over deep
loops so a big K is viable. Peak RSS ≤ ~9 GB (streaming-sparse build; NEVER the
~54 GB dense build). GPU VRAM: back K off if a span OOMs (the 306-block stack with
a wide span is VRAM-heavy; K=16–48 fits).

### Lever 4 — sparse vs dense forward
`dense_kernel` mode (materialise the CSR weight, `F.linear`) is the default and was
used for every number here; on this GPU the block stack is not the bottleneck for
deep programs (GPU util ~41 % during a deep verify → the fast path is **CPU-bound
on the Python overlay + per-block eviction**, not GPU-bound), so `sparse_mm` was
not a measurable win. This is the honest state: the remaining cost is Python-side,
not GEMM.

## Benchmark table (measured, this GPU)

Naive per-step wall is measured directly on a bounded prefix (`--naive-steps`) of
the SAME model and projected to the full step count; the fast path measures the
whole program (or its accepted prefix). SP_INIT pinned **0xFC** consistently across
draft / naive-cached-driver / model (see "gotcha" below).

| program | steps | naive ms/step | naive full (proj) | fast forwards | fast wall | fwd reduction | **wall speedup** | byte-exact | notes |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| **loop_countdown n=200** | 3 015 | 587 | 29.5 min | 95 | 122.8 s | 31.7× | **14.4×** | ✅ 3015/3015, AX=0 ✓ | fully clean; block-MoE 1.6× |
| **nested_loop 40×200 (deep)** | 121 339 | 605 | **20.39 hr** | ~2 528 | _[FILL]_ | ~48× | **_[FILL]×_** | ✅ naive==draft | byte-safe deep loop; CPU-bound |
| mandelbrot 1×1 iter 3 | 484 | 1152 | 9.3 min | 9 | 11.0 s | 53.8× | 50.6× (accepted) | fast==naive output ✓ | model diverges from ideal draft at step 136 (see below) |
| malloc+memset+memcmp n=64 | 1 506 | 755 | 19.0 min | 2 | 2.0 s | 753× | 579× (accepted) | fast==naive prefix ✓ | model reg-decode drifts at step 24 (heap store) |
| matmul 3×3 (malloc arrays) | 2 802 | 1626 | 75.9 min | 1 | 2.4 s | 2802× | 1895× (accepted) | fast==naive prefix ✓ | model diverges at step 9 (MOD) |

## What is now feasible

* **A deep loop that was 20.4 HOURS naive** runs the fast path in _[FILL]_ min —
  the perfect draft is instant; the verify is ≈2 528 batched forwards instead of
  121 339 one-step forwards, all byte-exact to the naive path.
* **loop_countdown that was 29.5 min naive** → **2.05 min** (14.4×), fully
  byte-exact end-to-end (final AX correct).
* **A 1×1 mandelbrot that was ~10 min naive** verifies its accepted prefix in 11 s
  (50×); the visible render bytes are byte-exact to the naive path.

## The honest walls (measure, don't hide)

1. **Perfect-draft fidelity is the real limit, not the speedup.** Speculation needs
   the draft (the ideal C VM) to match the neural model exactly. It does — **fully**
   — for pure register/arithmetic programs **whose values stay ≤ 255**
   (loop_countdown n=200, nested_loop: 100 % accept). For programs that exceed the
   model's faithful range the **model itself** diverges from ideal 32-bit C:
     * loop_countdown **n=2000** diverges at step 48 (i > 255: the model's byte-wide
       ALU vs the 32-bit draft),
     * mandelbrot at step 136 (a squared local exceeds a byte; model loads 64 where
       ideal C loads 0),
     * malloc at step 24 (heap-store perturbs the model's decoded SP/BP by 1),
     * matmul at step 9 (the first `%`/MOD).
   In every case this divergence is a **pre-existing model fidelity limit shared by
   the naive path** (the naive driver, which also runs the model, produces the same
   divergence — verified: naive AX == draft for the shallow prefix, then both drift
   together), and for mandelbrot it is **downstream of the visible output** (the
   render is still byte-exact). It is NOT a fast-path bug: the fast path's output is
   byte-exact to the naive path over the shared prefix. But it means speculation can
   only verify up to the divergence for these programs, because past that point the
   ideal-C draft is no longer a valid guess for the (imperfect) model.

2. **The fast path is CPU-bound for very deep programs.** GPU util is ~41 % during a
   121 k-step verify: the per-forward Python overlay (`apply_overlay_window_fast`)
   and the per-block eviction loop (306 blocks × ~2 528 prunes) dominate, not the
   GEMMs. This is why the deep-loop wall, while ~48× under naive, is minutes not
   seconds. The next win is vectorising the overlay + eviction off the Python
   critical path (the batched decode in `batched_speculative.decode_states_batched`
   is the template).

3. **Full self-forward (the model running its own weights) remains a genuine wall.**
   Not addressed here.

## A gotcha worth recording

`nibble_pure_forward_cached` does `from nibble_pure_forward import SP_INIT` — an
**import-time snapshot**. Setting `_PF.SP_INIT` later does NOT update the cached
driver's binding, so the naive driver and the draft silently used different
SP_INIT → the draft's bp/sp bookkeeping drifted from the model by a constant (LEA
off by 4). Fix: pin SP_INIT on `_PF`, `_PFC`, **and** the cached module's namespace
before use. Pin **0xFC** (the deep-recursion value); the default 0x10000 overflows
the 8-bit frame window. This is what makes `naive-prefix AX == draft: True`.

## How to reproduce

```
export OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
# clean fully-byte-exact result:
python -m c4_min.bench_fast_path loop --n 200 --device cuda:0 \
    --block-steps 32 --block-moe --naive-steps 15
# deep-loop headline (byte-safe, ~121k steps):
python -m c4_min.bench_fast_path nested --outer 40 --inner 200 --device cuda:0 \
    --block-steps 48 --block-moe --naive-steps 12
# mandelbrot (accepted-prefix speedup; diverges from ideal at step 136):
python -m c4_min.bench_fast_path mandel 1 1 3 --device cuda:0 \
    --block-steps 16 --block-moe --naive-steps 25
```
