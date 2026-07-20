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

## Results (verified)

Running through `model.forward` (`run_pure_forward_cached`, `mask=0xFFFFFFFF`,
`evict=True`, `prune_interval=60`), byte-exact vs `ref_interpret`:

- **Build:** streaming sparse model, ~28–31 s, **peak RSS ~5 GB** (never the
  ~62 GB dense build). Eviction holds the KV cache bounded (~36–66 entries) and
  RSS flat at ~5 GB for the whole run.
- **Op path proven byte-exact through the neural forward** (calibration, all
  word-width): `IMM 8; PSH; IMM 8; MUL → 64`; `2500/1024 → 2` (MUL>255 + DIV);
  `3000 > 1024 → 1` (GT); a compiled `printf` loop with JSR/ENT/LEV framing +
  stack-locals prints `"*+,"` — neural stdout == reference stdout, exactly.
- **Tiny mandelbrot** (see the run table below): the model's PRTF stdout equals
  the reference render byte-for-byte, and the render is a genuine mix of inside
  (`*`) / escaped (` `) cells — a filled region, not a solid block.

### Grid / step / wall table

| grid  | maxiter | VM steps | reference render        | notes                    |
|-------|---------|----------|-------------------------|--------------------------|
| 1×1   | 1       | 237      | `*`                     | smallest; single cell    |
| 1×1   | 3       | 295      | ` `                     | escapes at iter 2        |
| 2×2   | 3       | 1143     | `  ` / `* ` (mix)       | asserted test case       |
| 3×2   | 3       | 1725     | `   ` / `* *` (mix)     | clearest mixed shape     |
| 4×3   | 2       | 2919     | mix                     |                          |

Per-step wall on this box: **~2.6–3.5 s/step uncontended** (measured on the
printf loop + DIV calibrations); under heavy concurrent CPU load (multiple agents,
load-avg ~9) it degrades to ~12 s/step. So the tiny grids cost roughly:

- 1×1 (237 steps): ~10–13 min uncontended.
- 2×2 (1143 steps): ~50–65 min uncontended.
- 3×2 (1725 steps): ~75–100 min uncontended.

## Honest extrapolation to a full render

The ORIGINAL `test_mandelbrot` renders 40×20 = 800 cells at `maxiter=30`. At the
per-cell arithmetic cost measured here (~250–350 VM steps for the fixed-point
iterate + escape test + PRTF, times ~10× the maxiter), a 40×20×30 render is on the
order of **~240k+ VM steps**. At ~3 s/step (uncontended neural forward) that is
**~200 hours** of wall clock — the perf wall this demo deliberately does NOT hit.
The point is the byte-exact fidelity of the arithmetic + I/O path on a tractable
slice, not the render size.

## Files

- `c4_min/_mandel_src.py` — the byte-literal fixed-point C generator.
- `c4_min/_mandel_run.py` — instrumented end-to-end runner (steps/wall/RSS +
  render heartbeat via the driver's `verbose`).
- `c4_min/test_mandelbrot_neural.py` — cheap compile-only checks (always run) +
  the heavy env-gated (`C4_RUN_NEURAL_MANDELBROT=1`) byte-exact `model.forward`
  assertion (`2×2`, `maxiter=3`).

Run the neural assertion:

```
OMP_NUM_THREADS=4 C4_RUN_NEURAL_MANDELBROT=1 \
    python -m pytest c4_min/test_mandelbrot_neural.py -x -s
```

Or the instrumented runner directly (with a per-step heartbeat):

```
OMP_NUM_THREADS=4 python -m c4_min._mandel_run 1 1 1 -v
```
