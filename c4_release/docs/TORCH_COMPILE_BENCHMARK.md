# torch.compile Benchmark (2026-05-12, updated)

Measurement of the opt-in `compile_mode="reduce-overhead"` (and other
`torch.compile` modes) path landed in commit `ce2a704`
("Add torch.compile support + grouped-GEMM SoftMoE + .item() blockers").

## Environment

- Hardware: CUDA 12.8, single GPU (auto-detected via `torch.cuda.is_available()`)
- Python: 3.12.6
- PyTorch: bundled with project (inductor + dynamo)
- Branch: `torch-compile-standard-recipe` (this commit)
  - Prior attempts on `enable-compile-mode-default` (commit `b347d9c`).

Test harness: `pytest c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit`
plus a focused benchmark script
`c4_release/scripts/bench_torch_compile.py <recipe>` (added 2026-05-12) that
times only the forward path so the compile cost is isolated from the
~30 s session-fixture batched-runner build.

The smoke session fixture (`_smoke_batched_results` in
`tests/test_smoke.py`) batches ~90 programs through a single shared model.
Each program takes 5-50 forward calls; the whole session is ~350-750 s wall
on this hardware without compile, so the smoke-fixture timing alone is too
noisy to attribute to compile vs other costs.

The focused script holds everything else constant and times only the
forward calls themselves.

## Baseline (no compile)

```text
$ python scripts/bench_torch_compile.py none --steps 8
Build runner: 37.89s, init context len: 20
Model: d_model=728, n_blocks=26

step      seq_len     wall_s
   0           20     0.3996  <- CUDA kernel warmup (cudnn / cublas auto-tune)
   1           21     0.0412
   2           22     0.0381
   3           23     0.0326
   4           24     0.0364
   5           25     0.0355
   6           26     0.0351
   7           27     0.0350

Warmup (step 0):    0.3996s
Steady-state mean:  0.0352s  (last 3)
```

Steady-state un-compiled forward is **~35 ms** for seq_len 20-27.
Build-runner (compile_full_vm + cuda transfer) is **~38 s**.

## Recipe A: `torch.compile(..., dynamic=True)` + `mark_dynamic`

Configuration:

```python
self.model = torch.compile(self.model, mode=compile_mode, dynamic=True)
# At first compiled forward:
torch._dynamo.mark_dynamic(token_ids, 1, min=1, max=2048)
```

Plus the prior `_dynamo.config.recompile_limit = 128` and
`force_parameter_static_shapes = False`.

### Recipe A with `mode="reduce-overhead"`

Run: `python scripts/bench_torch_compile.py A --steps 8 --compile-mode reduce-overhead --budget 800`

| Step | seq_len | wall (s) |
| ---- | ------- | -------- |
| 0    | 20      | (timeout — never finished) |
| ...  | ...     | ...                        |

Hit the 900 s pytest-timeout wall before completing **even the first** of
8 forwards. The Inductor compile log shows multiple `[2/N]` recompile-counter
markers, plus the same "CUDAGraph supports dynamic shapes by recording a
new graph for each distinct input size. **We have observed 9 distinct
sizes.**" warning the prior agent (commit `b347d9c`) saw.

Observation: **`dynamic=True` does not eliminate per-seq-len recompiles**
for this model. Inspecting the trace logs we see:

- The model contains data-dependent control flow (e.g. `.item()` early-outs
  in `efficient_alu_divmod_split.py:459`, `.any()` guard in
  `alu/ops/divmod_longdiv.py:284`) that forces specialization on input
  values during Dynamo guard-building. These are gated behind
  `torch.compiler.is_compiling()` but only at the `.item()` site itself —
  the guard is checked AFTER Dynamo has already started tracing the
  full forward.
- Even when those branches don't fire, the symbolic-shape guard for the
  seq dim still produces ~9 different specializations because each
  block has different parameter shapes (the right-sizing pass leaves
  26 distinct FFN widths).

### Recipe A with `mode="default"`

Run: `python scripts/bench_torch_compile.py A --steps 6 --compile-mode default --budget 540`

Also hit timeout. The Inductor codegen + autotune for the full 26-block
forward is the dominant cost; switching to `mode="default"` (no CUDA
graphs) does not help because Inductor still has to emit a Triton kernel
per right-sized FFN.

Decision: **Recipe A wired into the runner anyway** (commit on this branch)
so the code is ready for future torch / model improvements that make
warmup tractable. Default stays `compile_mode="none"`.

## Recipe B: static padding to power-of-2 buckets

Configuration: `torch.compile(model, mode=..., dynamic=False)`, plus a
per-call `F.pad(token_ids, (0, bucket-S), value=0)` to round seq_len up
to the nearest of `{64, 128, 256, 512, 1024}`. The runner already strips
to `model.max_seq_len` so the largest bucket is bounded.

Run: `python scripts/bench_torch_compile.py B --steps 6 --compile-mode default --budget 540`

Result: also hit the 540 s budget mid-Inductor-codegen. Even compiling
once for a SINGLE bucket (after padding all inputs to 64) takes more
than 9 minutes of Inductor work on this hardware. The bottleneck is
genuinely the codegen pipeline for the 26 right-sized blocks, not the
shape-variation overhead.

Note: Recipe B would also need the `BatchedPureNeuralRunner` to pad
its inputs (currently it does its own padding to the longest active
element). Wiring it in is straightforward (the `_pad_to_tensor` site is
the natural insertion point) but the warmup gate is the same.

## Recipe C: raw CUDA graphs (capture-at-fixed-shape replay)

Configuration: allocate a static input buffer per bucket; warm up the
model 3 times; capture one `torch.cuda.CUDAGraph` per bucket; replay
on subsequent calls.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model(static_in)
# Replay:
static_in.copy_(token_ids_padded)
g.replay()
```

Run: `python scripts/bench_torch_compile.py C --steps 8 --budget 200`

Failed during capture with:

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
  File "neural_embedding.py", line 183, in _prefix_cache_matches
    prefix_slice = token_ids[0, :cache_len].detach().to(cached.device)
```

After monkey-patching `_populate_prefix_cache` to a no-op, capture still
fails further down the model:

```
  File "efficient_alu_divmod_split.py", line 459, in forward
    if float(op_div_max.item()) < 0.1 and float(op_mod_max.item()) < 0.1:
```

Recipe C needs the model purified of all host-device sync points
during capture (`.item()`, `.to(cpu)`, `.cpu()`). The existing
`torch.compiler.is_compiling()` guards do not fire under raw CUDA-graph
capture; we'd need an additional `torch.cuda.is_current_stream_capturing()`
check at every site (or a global runtime flag).

Sites identified (incomplete; from `grep -n "\.item()" neural_vm/`):

- `neural_embedding.py:183` — prefix cache CPU comparison.
- `efficient_alu_divmod_split.py:459` — early-out on inactive DIV/MOD.
- `byte_efficient_alu.py:55,56,60,66,128,…` — multiple host-side branches
  (these are inside `if not torch.compiler.is_compiling()` blocks but
  still call `.item()` when not compiling).
- `efficient_alu_8bit.py:124,125,202,203,293,294,…` — RESULT readback.

Decision: **Recipe C is a substantial refactor** (gate every `.item()`
on `torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()`).
Filed for follow-up; not blocking this branch.

## Recipe D: `mode="max-autotune"` + `dynamic=True`

Not attempted — Inductor's autotune adds MORE compile cost (it tries
multiple kernel variants per op), so Recipe A's >900 s warmup is the
floor. `max-autotune` would extend that, not shrink it.

## Results table

| Recipe                                         | Warmup        | Steady state | Speedup vs none | Notes |
| ---------------------------------------------- | ------------- | ------------ | --------------- | ----- |
| **none** (baseline)                            | 0.40 s        | 0.035 s      | 1.0x            | Eager forward, 26 blocks |
| **A** dynamic=True + RO                        | >900 s (DNF)  | n/a          | n/a             | Inductor codegen + 9 recompiles |
| **A** dynamic=True + default                   | >540 s (DNF)  | n/a          | n/a             | Same; CUDA graphs not the bottleneck |
| **B** padding to buckets + default             | >540 s (DNF)  | n/a          | n/a             | Single-bucket compile still DNF |
| **C** raw CUDA graphs                          | n/a (errored) | n/a          | n/a             | `.item()` graph-capture violations |
| **D** max-autotune + dynamic=True              | not run       | n/a          | n/a             | Strictly worse warmup than A |

## Root cause analysis

The model has **26 right-sized blocks** with completely distinct FFN
widths (4096→7, 4096→5, ..., 4096→4096). Standard transformer compile
recipes assume a stack of N homogeneous blocks: torch.compile traces one
block, Inductor codegens one Triton kernel, then all N blocks share that
kernel.

For this model, every block has a different shape, so Inductor must
codegen 26 distinct Triton kernels per forward. The codegen pass for a
single complex kernel (multi-head attention + soft-MoE FFN + post-ops
+ RMSNorm) is ~10-30 s of CPU compile work, so the full-model compile
is 5-15 minutes of CPU even before CUDA-graph capture.

The "dynamic shapes" warnings the prior agent saw are a symptom, not
the cause. Even if all 26 kernels were re-compiled-from-cache, each per-
shape specialization still has ~50% of the codegen cost (guard generation,
fusion planning, Triton compilation).

## What to do next

For this model:

1. **Don't compile the whole forward.** Wrap only the homogeneous
   "vanilla" blocks (post-Phase-0 expansions; ~17 of them are baked
   into vanilla `PureFFN`). The compiler-generated post-ops can stay
   eager.
2. **Compile per-block individually** and accept the per-block CUDA-graph
   capture cost (the prior agent's note above says this trips
   "tensor output of CUDAGraphs has been overwritten"; the fix is to
   use `mode="default"` (no graphs) for the per-block compile, and one
   `mode="reduce-overhead"` wrap around the whole forward in eager mode).
3. **Purify the `.item()` sites** with an additional
   `torch.cuda.is_current_stream_capturing()` check so Recipe C becomes
   viable without monkey-patching.

For this branch:

- **Recipe A is wired into the runner** (`torch.compile(...,
  dynamic=True)` + `mark_dynamic` on first forward). The default stays
  `compile_mode="none"` since none of the recipes converges in
  acceptable warmup time on this hardware.
- **`C4_COMPILE_MODE` env var still toggles**: users who want to
  experiment with `reduce-overhead`/`max-autotune` can set it. With the
  Recipe A wiring, a future model refactor (smaller block-shape
  variance, fewer right-sized blocks) should make compile tractable
  without further runner changes.
- **Benchmark script `scripts/bench_torch_compile.py`** is committed so
  the next agent doesn't have to rebuild the harness from scratch.

## Gate-suite verification (default off)

```bash
timeout 1200 python -m pytest \
  c4_release/tests/test_smoke.py::TestSmokeBasic::test_imm_exit \
  c4_release/tests/test_runtime_vanilla.py \
  c4_release/tests/test_layer_idx_consistency.py \
  c4_release/tests/test_compile_determinism.py \
  -v --tb=line --timeout=900
```

Result with `C4_COMPILE_MODE` unset (default `"none"`): same as prior
attempt — passes in ~750 s.

The Recipe A wiring change (this branch) does **not** alter the
un-compiled path: `compile_mode == "none"` skips the whole
`torch.compile(...)` block entirely. The new `mark_dynamic` call is
inside the `compile_mode != "none"` arm of `_generate_next_cached`.
