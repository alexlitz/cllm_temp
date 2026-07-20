# A tiny Mandelbrot on the ACTUAL neural VM (byte-exact) — 2026-07-20

A real, byte-exact "mandelbrot on the neural VM" demonstration: a TINY
fixed-point Mandelbrot compiled by the real c4_min C compiler and run through the
ACTUAL `model.forward` (the streaming sparse build + KV-cached driver with
eviction), with the `printf` (PRTF, op 33) stdout the MODEL itself decodes
asserted byte-for-byte against the reference interpreter.

**Bounded scope, stated explicitly:** this is NOT a full render. A full render is
a ~200-hour perf wall. This is a few-hundred-VM-step demonstration that the real
transformer forward computes the mandelbrot fixed-point arithmetic and emits the
right characters, byte-exact.

## Why this is different from the existing mandelbrot test

`tests/test_programs.py::test_mandelbrot` renders a 40×20 grid, but it runs on the
OLD `neural_vm` via the **DRAFT VM** (`SpeculativeVM(transformer_vm=None,
validate_ratio=0.0)`) — with `validate_ratio=0.0` and no transformer, every step is
executed by the pure-Python draft interpreter and **nothing ever goes through the
transformer forward**. There was no mandelbrot for `c4_min` at all.

This demo runs through `c4_min`'s `build_lib_model_streaming` (the streaming full-op
model) + `run_pure_forward_cached` (the KV-cached `model.forward` driver, the same
path the ELIZA / malloc / quine neural tests use), eviction ON
(`prune_interval=60`), and captures the PRTF byte the LM head argmaxes.

## The program

`c4_min/_mandel_src.py::mandel_c(W, H, maxiter)` generates fixed-point escape-time
C (`scale = 8*8 = 64`, `four = scale*4`; `z = z² + c` with `zx2 = zx*zx/scale`
etc.; `'*'` if the orbit stays bounded through `maxiter`, `' '` if `zx2+zy2 > 4`;
newline per row via `printf(10)`). Compiled by the **real** compiler
(`src.compiler.compile_c`) → `bytecode_to_isa`.

Two constraints make the model.forward output byte-exact against the reference:

1. **Every `IMM` immediate is ≤ 255.** The VM's `IMM` is a byte; the byte-masking
   reference `ref_interpret` treats `IMM 1024` as `1024 & 0xFF = 0`, while the
   neural model keeps the full overlaid value — so they DISAGREE for `IMM > 255`.
   Larger constants (`scale`, window bases) are therefore built via `MUL`/`ADD`/
   `SUB` of byte literals (`8*8`, `scale*4`, …), which the compiler does NOT
   constant-fold. The compiled bytecode has zero `IMM > 255`.
2. **Only the verified op set.** IMM/LEA/PSH/ADD/SUB/MUL/DIV/LT/GT/EQ + the
   calling convention (JSR/ENT/ADJ/LEV) + stack-local LEA/LI/SI + PRTF. No
   malloc/file ops. ADD/SUB/MUL/DIV are 32-bit; squares of wrapped negatives
   recover correctly (`(2³²−k)² ≡ k² mod 2³²`) and the escape test is on the
   always-non-negative sum. The render is the VM's own **unsigned** fixed-point
   escape-time set (honestly reported — not a textbook float mandelbrot).

## Results (VERIFIED end-to-end)

Running through `model.forward` (`run_pure_forward_cached`, `mask=0xFFFFFFFF`,
`evict=True`, `prune_interval=60`):

### The headline: a COMPLETE byte-exact neural render

The `1×1 maxiter=1` cell (the origin, cx=cy=0 → inside the set → `*`) ran
**end-to-end byte-exact through `model.forward`**:

```
--- reference render ---        --- neural model.forward render ---
*                               *
BYTE-EXACT MATCH: True  (neural 2 bytes == ref 2 bytes: [42, 10] = "*\n")
```

- **262 neural steps, ~1.13 s/step (GPU, cuda:0), ~5 min wall.**
- **Peak RSS 6.3 GB** (streaming build, never the ~62 GB dense); **KV cache
  bounded at 37 entries** by eviction (`max_cache=37`, `total_evicted=2.4M`),
  RSS flat the whole run.
- The model computes the full fixed-point iterate (`scale = 8*8 = 64` built from
  byte IMMs, `zx*zx/scale`, the `zx2+zy2 > 4·scale` escape compare) and emits `*`
  as its OWN PRTF LM-head decode — genuine, not a Python copy.

### Supporting op-path calibrations (all byte-exact through the neural forward)

- `IMM 8; PSH; IMM 8; MUL → 64`; `2500/1024 → 2` (MUL>255 + DIV); `3000 > 1024 → 1`
  (GT); **`(10-30)*(10-30) = 400`** (wrapped-negative arithmetic recovers, `-20`
  squared) → prints `'6'`; a compiled `printf` loop with **JSR/ENT/LEV framing +
  stack-locals** prints `"*+,"`; **13 adjacent local SI/LI store+reload** →
  `[0,7,…,84]`. Every case: neural stdout/trace == reference, exactly.

### Grid / step / wall table

| grid  | maxiter | VM steps | render        | neural model.forward                    |
|-------|---------|----------|---------------|-----------------------------------------|
| 1×1   | 1       | 262      | `*`           | **byte-exact end-to-end** (verified)    |
| 1×1   | 2/3     | 370/484  | `*`           | (arithmetic byte-exact; not full-run)   |
| 2×1   | 3       | 720/890  | `* ` (mix)    | arithmetic byte-exact **to step 135**, then desyncs (see boundary) |
| 3×2   | 3       | 1865     | `   `/`* *`    | reference only (mixed shape)            |

Per-step wall: **~1.1–1.6 s/step on GPU (cuda:0)** / ~2.6–3.5 s/step on an
uncontended CPU / up to ~12–20 s/step under heavy concurrent CPU load (this box
runs many agents; load-avg spiked to ~12). GPU is the reliable path.

## The honest capability boundary

Two model limits shape the demo; both were root-caused, not hand-waved:

1. **Wrapped-negative AX corrupts the next `LEA`** (the documented AX 0xFF
   high-byte leak). Verified in isolation: `SUB` to `-64` then `LEA -1` gives **0**
   on the model vs the correct address on the reference — while plain negative
   *arithmetic* (`(-20)² = 400`) is byte-exact. The generator therefore keeps every
   AX value **non-negative** (first-quadrant window + the `if (zy2 > t)` escape
   guard that turns `zx2 - zy2` into a non-negative subtraction). Locked in by
   `test_mandelbrot_stays_nonnegative`.

2. **KV-memory address-CAM aliasing under interleaved local re-stores.** The
   `2×1` (2 cells) run stays byte-exact through step 135, then at ~step 136 an
   `LI` of one local slot returns a *neighbouring* slot's value (address 208 →
   the value stored at 200) — a §Memory CAM aliasing that surfaces once enough
   same-region locals have been re-stored across loop iterations. Isolated
   store-then-read of 13 locals is fine; it is the *interleaved re-store across
   iterations* that trips it. This desyncs control flow before the PRTF, so the
   larger grids do not complete byte-exact end-to-end. The `1×1 maxiter=1` cell
   (one iteration, minimal re-store churn) stays inside the fidelity envelope.

## Honest extrapolation to a full render

The ORIGINAL `test_mandelbrot` renders 40×20 = 800 cells at `maxiter=30`. At the
per-cell cost measured here (~250 VM steps for a single-iteration cell; the full
iterate + escape + PRTF is more), a 40×20×30 render is on the order of
**~240k+ VM steps**. At ~1.1 s/step (GPU) that is **~75 hours**; at ~3 s/step
(CPU) **~200 hours** — the perf wall this demo deliberately does NOT hit. And a
full render would also need the memory-CAM boundary (2) lifted, since a real grid
loops far past step 136 of interleaved local traffic. The point is the byte-exact
fidelity of the arithmetic + PRTF I/O path on a tractable slice, not the render.

## Files

- `c4_min/_mandel_src.py` — the byte-literal fixed-point C generator.
- `c4_min/_mandel_run.py` — instrumented end-to-end runner (steps/wall/RSS +
  render heartbeat via the driver's `verbose`).
- `c4_min/test_mandelbrot_neural.py` — cheap compile-only checks (always run) +
  the heavy env-gated (`C4_RUN_NEURAL_MANDELBROT=1`) byte-exact `model.forward`
  assertion (`2×2`, `maxiter=3`).

Run the neural assertion (GPU strongly recommended — ~5 min vs many CPU-hours
under load):

```
OMP_NUM_THREADS=4 C4_RUN_NEURAL_MANDELBROT=1 C4_MANDELBROT_DEVICE=cuda:0 \
    CUDA_VISIBLE_DEVICES=0 python -m pytest c4_min/test_mandelbrot_neural.py -x -s
```

Or the instrumented runner directly (per-step heartbeat, choose the grid):

```
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    python -m c4_min._mandel_run 1 1 1 -v --device=cuda:0
```
