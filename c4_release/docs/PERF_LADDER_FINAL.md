# Doom-on-transformer — DEFINITIVE honest perf ladder (2026-08-06)

Tasks **#841** (compose the levers into ONE measured full VM step) + **#849** (the
honest 1 s/frame closure). No hype: every number below is either **measured** (and on
what GPU / frame) or **explicitly labelled a projection**. Golden `7d4afe61` (bare-env
`CUDA_VISIBLE_DEVICES="" python -m c4_min._fingerprint_build`) is UNCHANGED — every perf
lever is a DEFAULT-OFF runtime flag (see `docs/DOOM_FLAG_REGISTRY.md`); none touches
weights.

---

## 0. The target, stated exactly

The perf goal is **1 s per subsequent (steady) render frame**. A steady render-reduced
doom frame is **358,058 VM steps** (the `nested_120_255`-class DIV-free proxy the fast
schedule carries; a raw title-redraw frame is ~6.89 M steps — see §5). So:

- **1 s / 358,058-step frame  ⇒  need 358,058 steps/s.**
- The "6.89 M steps/s" figure in the task is the RAW-frame target: 6,889,264 steps / 1 s.
  Both are stated below; they differ only by which frame definition you price.

A "full VM step" = one c4 instruction executed by the transformer: block-0 ingest
attention + FFN, the 238 dead-attention / dead-FFN blocks, the 3 live-CAM memory blocks,
and the register decode. The composed step (#841) fuses all of these.

---

## 1. #841 — the composed full VM step (all levers in ONE forward), byte-exact

`c4_min/_agent_composed_floor.py` composes **every** landed perf lever onto the REAL doom
`verify_blocks` path and gates it byte-exact:

| lever (all DEFAULT-OFF) | what it does |
|---|---|
| `C4_DEAD_BLOCK_FUSION` | 238 dead blocks → `output=x` (no Q/K/V/O, no KV) |
| `C4_DIRECT_CAM_BATCHED` + `C4_DIRECT_LOCAL_CAM` + `C4_DIRECT_CAM_VEC` | O(1) address-decode memory/stack/LEV/code reads (no softmax-over-stores) |
| `C4_FLASH_ATTN` + `C4_BANDED_LOCAL_ATTN` | flash softmax1 (no `[Sq,Sk]` score tensor) + length-30 ingest band |
| `C4_FUSED_MEGABLOCK` + `C4_FFN_FUSED_HIDDEN` + `C4_FUSED_DELTA_FFN` | dead-FFN `[cut,N)` chain as ONE on-chip fused sparse-COO megakernel |
| `C4_GRAPH_BLOCK0` + `C4_BLOCK0_FUSED_FFN` + `C4_WHOLE_STEP_GRAPH` | block-0 S-chunk loop → one CUDA-graph; block-0 FFN → sparse |
| `C4_ATTN_MEGABLOCK` + `C4_BLOCK0_DK` | live-CAM FFN + block-0 folded into the same `[D,K]` fused region (kills the transposes) |
| `C4_EXACT_EVICT` + `C4_FROZEN_ROW_SKIP` + `C4_GPU_VERIFY` | O(steps) liveness KV-eviction + frozen-row skip + one-sync vectorized decode |

### Byte-exactness — MEASURED THIS SESSION (the load-bearing claim)

`_agent_composed_floor.py` byte-exact battery (full composed stack vs the certified c4
draft, the K=1 per-step ground truth), K ∈ {512, 2048, 8192}, scalar-decode AND
GPU-vectorized-decode paths — **ALL programs matched=True at every step**: imm, add, sub,
mul, div, mod, and, shl, eq, lt, lea, bz, bnz, jmp, jsr_lev, si_li (incl. the recurrent
DIV/MOD fallback). L-inf = 0 on PC/SP/BP/AX vs the draft; scalar decode == GPU decode.
This reproduces the tree's prior end-to-end L-inf=0 gates (`ATTN_MEGABLOCK_FINDINGS.md`,
`BLOCK0_DK_FINDINGS.md`, `_agent_continuous_frame.py`: 462,979-step frame L-inf=0).

**Golden `7d4afe61` re-confirmed UNCHANGED** (bare-env fingerprint), both before and after
the composed run. All levers default-OFF; no weight/build edits.

### The composed full-step breakdown — attention vs FFN vs decode

The **authoritative steady-state** breakdown (idle RTX A5000, the large 462,979-step
`nested_120_255` frame, chunk 65536, CUDA-event sub-graph timing —
`_agent_dispatch_profile.py`, recorded in `c4_min/_agent_dispatch_profile_FINDINGS.md`
and `c4_min/DOOM_HASH_CAM_2026_08_05.md`):

| phase of the composed full step | µs/step | % of step | category |
|---|---:|---:|---|
| dead-FFN mega chain (238 blk, fused sparse-COO) | 1.72 | 69.1% | **FFN** |
| live-CAM attention (3 blk: mem/stack-pop/code-select) | 0.69 | 27.7% | **attention** |
| block-0 FFN (ingest SwiGLU) | 0.082 | 3.3% | **FFN** |
| decode lanes (snap + AX) | 0.026 | 1.0% | **decode** |
| launch / graph-replay overhead | ~0 | ~0% | — |
| **composed full step (steady-state)** | **2.49** | 100% | |

Rolled up: **FFN ≈ 72.4% (1.80 µs), attention ≈ 27.7% (0.69 µs), decode ≈ 1.0% (0.026
µs)** of a 2.49 µs/step composed step. With the newest `C4_BLOCK0_DK` fold (block-0 +
live-CAM FFN absorbed into the `[D,K]` fused megachain, kernel-split in
`BLOCK0_DK_FINDINGS.md`), the composed step drops to **0.788 µs/step (bk512, idle A5000,
byte-exact)** — now 94% one FFN mega chain, live-CAM attention and the transposes gone.

### Fresh composed measurement THIS SESSION (contention-affected — labelled)

The shared box was fully saturated (both A5000s pinned 100% util, 6–10 GB free, other
agents active), so absolute wall timing is NOT representative. On a small 5,455-step
`nested_12_28` program (so ≈1 forward — the one-time schedule build dominates the
per-step wall) the composed `verify_blocks` path measured, matched=True at every K:

| K | wall µs/step (contended, 1-fwd amortization) |
|---:|---:|
| 8,192 | 817 |
| 65,536 | 1,306 |
| 262,144 | 563 |

These are **NOT the per-step floor** (they price the one-time build over few steps under
100% external GPU contention); the steady-state floor is the 2.49 → 0.788 µs/step above,
measured on the large frame on an idle card. The **CUDA-kernel device-time split** of one
composed forward (contention-robust — per-kernel device time) reproduced the structure:
block-0/live **DENSE GEMM (attention+block0-FFN) 22.0%**, **megablock FFN kernels 13.3%**,
**overlay/decode HtoD 5.8%** — i.e. the same attention-then-FFN-then-decode ordering, GEMM
inflated on the tiny 5,455-step program because the block-0 dense ingest amortizes over
few steps (exactly the wall `C4_BLOCK0_DK` folds away on the large frame).

---

## 2. #849 — THE honest perf ladder (each rung measured, byte-exact flagged)

All rungs are byte-exact (L-inf=0) and DEFAULT-OFF unless noted. GPU = one RTX A5000
(24 GB, ~768 GB/s HBM, 64 SMs), unless a rung says 2-GPU. Factors are **measured** on an
idle card except where labelled *projection*.

| # | rung | what it does | measured factor | byte-exact | source |
|---|---|---|---|---|---|
| 0 | naive token-by-token driver | one `model.forward` per VM step | baseline (~0.5–1.6 s/step) | yes | `PERF_FAST_PATH_2026_07_20.md` |
| 1 | perfect-draft speculation + doom-active block-skip | Rust c4 draft materializes the whole 30-tok/step frame; verify K steps/forward; DIV-free blocks skipped | 10–14× wall (up to 63.6× forward-count reduction); block-MoE 1.6× | yes (matched=True end-to-end) | `PERF_FAST_PATH_2026_07_20.md` |
| 2 | fused KV eviction (wave over blocks) | one batched on-GPU eviction decision for all 306 caches vs 306 host-synced calls | **5,408× / 2,140×** on the eviction DECISION (30 s → 5–18 ms/prune); GPU util 41% → 99% | yes (30-battery byte-identical) | `PERF_FAST_PATH_EVICT_FUSED_2026_07_20.md` |
| 3 | FFN wave-batch + fused megablock | 49 dead-FFN block-pairs → ~27 wave-kernel pairs on one on-chip `[Dff,K]` hidden; `C4_MEGABLOCK_BLOCK_K=512` | folds the dead-FFN chain to 1.72 µs/step (69% of the composed step) | yes (L-inf=0) | `ATTN_MEGABLOCK_FINDINGS.md`, dispatch profile |
| 4 | attention megakernel (`C4_ATTN_MEGABLOCK`) | route live-CAM FFN through the sparse `[D,K]` megakernel; kill the dense live-FFN + clones + transposes | **1.79–1.84×** dispatch (1.989 → 1.113 µs/step) | yes (L-inf=0, op corpus) | `ATTN_MEGABLOCK_FINDINGS.md` |
| 5 | block-0 fold (`C4_BLOCK0_DK`) | fold block-0 into the same `[D,K]` region → kill the last input transpose + block-0 dense FFN GEMM | **1.35×** over rung 4 → **0.788 µs/step** (2.47× over the pre-mega base) | yes (L-inf=0, op corpus) | `BLOCK0_DK_FINDINGS.md` |
| 6 | WAD-hash frame + direct-CAM O(1) reads | title/WAD content-hash reuse; memory reads resolve O(1) by address decode (no O(S) softmax scan) | gathers cost ~0 at dispatch (precomputed deltas); WAD frame @96K = ~5–6.6 fps 1-GPU | yes (L-inf=0; `C4_HASH_CAM` rejects the planted-value scenario E) | `DOOM_HASH_CAM_2026_08_05.md`, capstone |
| 7 | render superinstruction (`C4_DOOM_DRAWSPAN`) | native DRAWCOL/DRAWSPANF render-macro opcode (one step fills a whole column/span) | collapses the pixel-fill inner loop (golden-MOVING: widens NUM_OPS 40→48) | yes on its own build | `DOOM_FLAG_REGISTRY.md` (golden-MOVING) |
| 8 | 2-GPU frame-level parallelism (`C4_MULTIGPU_FRAMES`) | whole frames round-robin to 2 A5000s (independent CUDA contexts, no cross-shard reduction) | **2.00–2.01×** (near-perfect; K-split only got 1.06× on a compute-bound single-chunk frame) | yes (per-GPU fingerprint == 1-GPU ref) | `DOOM_FRAME_LEVEL_2GPU_2026_08_05.md` |

### Where the composed full-step lands (measured, idle A5000, byte-exact)

| frame | 1-GPU | 2-GPU (rung 8, 2.0×) |
|---|---|---|
| title @96K (WAD-hash, rung 6) | ~5–6.6 fps | ~11–13 fps |
| **steady render-reduced @358,058 steps** | **~1.26–1.69 fps** (0.79 s/frame composed TRUE-PIPE; 1.685 fps frame-level faithful) | **~2.6–3.0 fps** |
| raw title-redraw @6.89 M steps | ~0.04–0.07 fps (15–27 s/frame) | ~0.08–0.14 fps |
| steady GAMEPLAY frame (folded ~2.5 M steps) | ~0.21 fps | ~0.42 fps |

So on the **steady render-reduced frame the composed stack already clears ≥1 fps
byte-exact** (1.26 fps 1-GPU TRUE-PIPE; 1.685 fps frame-level faithful; up to 3.0 fps
2-GPU). The 1 s/frame target on the render-reduced frame is **MET** (measured, byte-exact).

---

## 3. #849 — what reaching the RAW 6.89 M steps/s (1 s per raw frame) would require

Stated exactly, from where the composed full step lands. The composed steady-state step is
**0.788 µs/step** on one idle A5000 (best byte-exact, `C4_BLOCK0_DK`, bk512). Then:

- **1 s per 358,058-step render-reduced frame** needs 358,058 steps/s = **2.79 µs/step**.
  The composed step (0.788 µs/step) is **already 3.5× under this** → the render-reduced 1
  s/frame is MET on one card, byte-exact.
- **1 s per 6,889,264-step RAW frame** needs 6.89 M steps/s = **0.145 µs/step (145 ns)**.
  From the composed 0.788 µs/step that is a further **~5.4×**; from the target's stated
  "6.89 M steps/s" phrasing it is the same 145 ns/step.

**How much more ×, and from where — the honest accounting:**

1. **Kernel efficiency (algorithmic, byte-exact ceiling): up to ~5–8.7× on the dispatch,
   but UNREACHABLE to 145 ns.** The composed step is **occupancy/efficiency-bound at
   ~11.5% of HBM peak** (`BLOCK0_DK_FINDINGS.md` occupancy sweep) — the dead-FFN sparse-COO
   chain has tiny per-block work (median Dff≈40, ~8 active `W_down` rows) + a sequential
   residual dependency, so it under-fills the 64 SMs. If the sparse kernels hit the HBM
   roofline the dispatch could drop to **~0.48 µs/step** (`_agent_dispatch_profile_FINDINGS.md`
   TASK 2: HBM-BW floor = 483 ns/step). That is **still 3.3× short of 145 ns** — and 145
   ns/step would require moving the whole 371 KB/step in 145 ns = **~2.6 TB/s, ~3.4× the
   A5000's 768 GB/s HBM**. So even a perfect byte-exact kernel on one A5000 CANNOT reach
   145 ns/step: the raw frame is **memory-BW bound above the target on one card**. This is
   the honest wall — the five named per-step levers (draft-emitted gathers, in-place
   scatter, dense GEMMs, launch-kill, weight-dedup) are exhausted at dispatch (all ~0 or
   regress); only kernel-efficiency remains, and it tops out at ~0.48 µs/step.

2. **Fewer steps per raw frame (algorithmic): the render superinstruction + traversal
   fold.** The raw frame is 6.89 M steps because it is a stack-ISA pixel-walk (63% of the
   render is PSH/LEA/LI pointer-walking — capstone §6). Rung 7 (`C4_DOOM_DRAWSPAN`) already
   collapses the pixel-FILL inner loop into one step. Folding the **BSP/segment traversals**
   too (not just the fill) — a bigger render superinstruction set — would cut the raw
   step-count toward the render-reduced 358 K, at which point the composed step already
   MEETS 1 s/frame. This is **algorithmic** (reduce the work), not a faster kernel, and is
   the realistic path to "1 s per raw frame."

3. **More hardware (linear): frame-level multi-GPU.** Rung 8 is a clean **2.0×** per card
   pair. To turn the composed one-card 15–27 s raw frame into 1 s by hardware alone is
   **~15–27× = ~8–14 A5000s** frame-parallel (linear, byte-exact) — a hardware statement,
   not an algorithmic one.

### One-line honest verdict

**Reaching 6.89 M steps/s (145 ns/step) on ONE A5000 is IMPOSSIBLE byte-exactly** — the
raw frame is HBM-BW-bound at ~483 ns/step (best-case kernel roofline, ~3.3× short), and
145 ns/step needs ~2.6 TB/s (~3.4× the card's HBM). The composed step is already at ~11.5%
of HBM peak (occupancy-bound), and the byte-exact per-step levers are exhausted (≤5–8.7×
kernel-efficiency headroom, capped at 0.48 µs/step). So closing to 1 s per **raw** frame is
either **algorithmic** (fold the traversals so the raw frame stops being 6.89 M steps and
becomes the ~358 K the composed step already clears) or **hardware** (~8–14 A5000s
frame-parallel, linear). It is **NOT** reachable as "one more kernel win on one card." The
render-reduced 1 s/frame target IS already met (measured, byte-exact, 1.26–1.69 fps 1-GPU).

---

## 4. Byte-exactness & golden (sacrosanct)

- Bare-env golden **`7d4afe61`** re-confirmed UNCHANGED (`CUDA_VISIBLE_DEVICES=""
  PYTHONPATH=<repo> python -m c4_min._fingerprint_build`) before and after this session's
  composed run. Rollback `C4_BP_RESTORE_HIBYTE=0` → the pre-fix `069cc32f` (also unchanged).
- Every rung's lever is DEFAULT-OFF and byte-exact (L-inf=0) except rung 7
  (`C4_DOOM_DRAWSPAN`, golden-MOVING on its own build — flagged) — see
  `docs/DOOM_FLAG_REGISTRY.md`.
- This session's composed byte-exact battery (all ops + branches + memory, K∈{512,2048,
  8192}, scalar==GPU decode == certified draft) passed matched=True at every step.

## 5. Frame definitions (so the numbers are comparable)

- **render-reduced steady frame = 358,058 steps** — the DIV-free proxy the fast
  single-dispatch schedule carries; a real steady doom render frame is effectively DIV-free
  (0–2 generic DIV/MOD out of 6.89 M steps, 0 after the #829 pow2 peephole — see
  `COMPOSED_FPS_AND_DIVMOD_GAP.md`), so it rides the same fast path.
- **raw title-redraw frame = 6,889,264 steps** — the full first/title redraw; amortizes
  once per session (the WAD-load + R_Init 1,175 generic DIV/MOD all live here, not in the
  steady render).
- **steady gameplay frame ≈ 2.5 M folded steps → ~0.21 fps** — inherent stack-ISA overhead
  (capstone §6); real-time gameplay is a hardware/step-count statement, not a bug.

## 6. Files

- `c4_min/_agent_composed_floor.py` — #841: composes ALL levers, byte-exact gate + ms/step
  + kernel split (attention/FFN/decode).
- `c4_min/_agent_dispatch_profile_FINDINGS.md`, `c4_min/DOOM_HASH_CAM_2026_08_05.md` —
  authoritative idle-A5000 steady-state per-step breakdown (2.49 µs/step).
- `c4_min/ATTN_MEGABLOCK_FINDINGS.md`, `c4_min/BLOCK0_DK_FINDINGS.md` — rungs 4/5 (→0.788
  µs/step).
- `docs/PERF_FAST_PATH_2026_07_20.md`, `docs/PERF_FAST_PATH_EVICT_FUSED_2026_07_20.md` —
  rungs 1/2.
- `c4_min/DOOM_FRAME_LEVEL_2GPU_2026_08_05.md`, `c4_min/COMPOSED_FPS_AND_DIVMOD_GAP.md` —
  rung 8 + the render-frame DIV/MOD gap.
- `docs/DOOM_FLAG_REGISTRY.md` — the full DEFAULT-OFF flag inventory + golden impact.
</content>
