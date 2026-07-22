# 3-Layer Self-Hosting: Feasibility Map

**Date:** 2026-07-20
**Branch:** `selfhost-3layer-feasibility` (off `fix-zfod-malloc-neural`)
**Authority:** `docs/BLOG_SPEC.md` §859-901 ("Self-Hosting"), which itself ends
with `TODO performance analysis of this`. This doc *is* that performance
analysis — an honest, measured feasibility map, not a forced pass.

## TL;DR

| Relationship | What actually ran end-to-end | Wall |
|---|---|---|
| **Rel-1** C-runtime hosts itself | fixed-point MatMul **kernel** compiles under c4 (542 words) + runs on the reference c4 VM (230 steps); native int-only runtime runs a tiny `c4vm.onnx` byte-identically to the float runtime | full runtime is **not c4-compilable**; smallest genuine ONNX forward under the neural c4vm ≈ **2.4M VM steps ≈ 88 days** |
| **Rel-2** ONNX-runtime hosts itself | — (native runtime loads a tiny model) | one forward of the **real** `c4vm.onnx` ≈ **4.1e10 MACs ≈ 1.95e11 VM steps** — cost-prohibitive |
| **Rel-3** transformer runs itself | — | Rel-1 nested on Rel-2's model; strictly worse than both |

**Verdict:** kernel-level self-hosting is **real and demonstrated**. The full
3-layer end-to-end loop is a hard **performance wall** (months → millennia of
neural-VM wall-clock), exactly as the blog's `TODO perf` anticipates. This is
**not** claimed as achieved.

All numbers below are measured on this machine (CPU, `OMP_NUM_THREADS=4`) by
`c4_min/selfhost/run_selfhost_feasibility.py`; re-run to reproduce.

---

## The pipeline being tested

"c4vm" in the blog means the **neural transformer** that interprets c4 bytecode.
To "run C source under c4vm" the source must be (a) compiled to c4 bytecode by a
c4 compiler, then (b) interpreted by the neural VM — one `model.forward` per
executed VM instruction (each forward emits a 30-token register frame).

```
onnx_runtime.c --[c4 compiler]--> c4 bytecode --[neural c4vm, 1 forward/instr]--> result
                                                        ^ this is the cost multiplier
```

---

## Rel-1: The C runtime hosts itself — `c4vm onnx_runtime.c c4vm.onnx [input.c]`

### 1a. The runtime as written is NOT in the c4 subset (hard gate)

The c4 language (Swierczek's c4 grammar, as implemented by both
`bundler/c4_compile.c` and `src/compiler.py`) supports only: `char`/`int`,
one-level pointers, 1-D arrays **via malloc'd pointers** (there is *no*
`int A[n]` array-declaration syntax), literal-only sizes, `while`, and c4's
single `printf`. It has **no** `long`, `float`, 2-D/3-D arrays, expression array
bounds, `fscanf`/`fopen`, or varargs.

Both ONNX runtimes fail to compile under c4:

| runtime | c4-compile result |
|---|---|
| `c4_min/onnx_runtime_nibble_fixedpoint.c` (int-only) | **FAIL** — `line 105: int exp_tbl[EXP_STEPS + 1]` (expression array bound) |
| `c4_min/onnx_runtime_nibble.c` (float) | **FAIL** — `line 87` (same class) |

Deeper blockers even past line 105: `t_dims[MAX_TENSORS][MAX_RANK]`,
`n_aval[...][...][...]` (2-D/3-D global arrays), `long *tv[...]`, varargs
`printf`, `fscanf`. The native `bundler/c4_compile.c` **hangs** (does not error)
on the first 2-D array — confirmed on a minimal `int a[4][4]` probe.

> The fixed-point runtime was written for **numpy-prototype parity** and torch
> byte-identity (its header cites `test_onnx_runtime_fixedpoint.py`), using 2-D
> arrays for readability. It is the *right precision* for self-hosting (int-only,
> no float emulation) but is **not** yet in the c4 *grammar*. The historical
> c4-grammar fixed-point path is `tools/neural_bundle_fixedpoint.py`, which
> generates a flattened bundle — but that tool imports `src.compiler` +
> `sparse_vm`, and `sparse_vm` does not exist on this branch (broken).

### 1b. The smallest genuine c4-subset slice DOES self-host (kernel level)

`c4_min/selfhost/onnx_kernel_c4subset.c` re-expresses the runtime's load-bearing
inner op — a fixed-point MatMul — in the *actual* c4 subset (int-only, malloc'd
1-D pointers, `while`-only, no varargs). It is the exact `matmul()` inner loop of
the fixed-point runtime.

- Compiles under c4: **542 bytecode words** (with stdlib `malloc` linked).
- Runs on the reference c4 VM: `AX=18432` (= `4.5 × 4096`, correct fixed-point),
  **230 VM steps** for a 2×2×2 matmul (8 MACs).
- Asymptotic rate (linear fit 2×2×2 vs 8×8×8): **≈ 4.75 VM steps / MAC**
  (the inner fpmul loop body: 2×LI, PSH, MUL, SHR-via-JSR/LEV, ADD, index math,
  loop compare/branch).

### 1c. Native fixed-point runtime is correct (the tractable kernel)

`c4_min/selfhost/onnx_runtime_fixedpoint_coo.c` = the int-only fixed-point
runtime **+ COO-sparse initializer loading** (the current `onnx_to_c4bin` writer
emits `is_init=2` COO; the shipped fixed-point `load()` only knew the older dense
format and hit EOF). Built natively and run on the smallest exported `c4vm.onnx`
(D=104, 207 ONNX nodes):

```
native float            runtime: argmax = [0, 1, 20, 3]
native fixed-point(COO) runtime: argmax = [0, 1, 20, 3]   <- byte-identical
```

So the fixed-point (self-hostable-precision) runtime **is correct**. The wall is
not correctness — it is running it *under the neural VM*.

### 1d. The wall: cost under the neural c4vm

Neural c4vm per-step cost (one `model.forward` at the **measured shape** of the
complete IO-capable VM, `D=2323`, 305 blocks):

| seq (VM steps in context) | ms / forward |
|---|---|
| 30 (1 step) | ~3169 |
| 60 (2 steps) | ~4799 |

→ **~3.2 s per executed VM instruction** (flat lower bound; real cost grows
`O(S²)` as the emitted token context accumulates).

Smallest genuine Rel-1 instance = run the runtime on the **tiny** `c4vm.onnx`
(D=104, single 4-token frame). That forward does **503,776 MatMul MACs**
(measured via the numpy reference). At 4.75 steps/MAC:

```
503,776 MACs × 4.75 steps/MAC   ≈ 2,392,936 VM steps   (matmul only; +softmax/exp/gather)
2,392,936 steps × 3.2 s/step    ≈ 7.66e6 s ≈ 88 DAYS    (flat lower bound)
emitted context: 2.4M steps × 30 tok ≈ 72M tokens (O(S²) attention → far worse)
```

**Rel-1 verdict:** kernel + native fixed-point runtime run and are correct; the
full runtime is not c4-compilable; and even the tiniest ONNX forward under the
neural c4vm is **months** of wall-clock. Rel-1 is demonstrated *at the kernel
level only*.

---

## Rel-2: The ONNX runtime hosts itself — `onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx`

Rel-2 requires the ONNX runtime to load and execute the **real** `c4vm.onnx` —
the model that is interpreting the runtime, not the D=104 toy. The real complete
IO-capable VM is `D=2323`, **305 blocks** (measured;
`build_pure_forward_complete_model(code_size=16)`), and its build costs **~64 GB
RSS** (the documented dense-build hazard — not built here beyond shape).

One forward of that model ≈ **4.1e10 MACs** (scaling the tiny model's MAC count
by `(2323/104)² × (305/14) × (30/4)`) ≈ **1.95e11 VM steps** on the c4-subset
runtime. That is a *single* forward; a real self-hosted run is many forwards.

Even run **natively** (not under the neural VM), a c4-subset int runtime doing
4e10 MACs at, generously, 1e8 MAC/s ≈ **7 minutes per forward** — but the point
of Rel-2 is that the runtime *is itself* a candidate to run under c4vm, and the
model it must load is the 64 GB one. **Not demonstrated; cost-prohibitive.**

---

## Rel-3: The transformer runs itself — `onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx [input.c]`

Rel-3 is Rel-1 *nested on* Rel-2: the neural c4vm interprets the runtime C source
(Rel-1's ~2.4M-steps-per-tiny-forward wall), and that runtime loads + executes
the real `c4vm.onnx` (Rel-2's ~1.95e11-steps-per-forward wall). The costs
**multiply**. With Rel-1 already at months for a *toy* inner model, substituting
the real model makes Rel-3 astronomically infeasible. **Not demonstrated.**

---

## The exact bottleneck, ranked

1. **Per-step neural-VM forward cost** (~3.2 s/instruction). The complete VM is
   305 blocks × D=2323; every executed VM instruction is one full forward. This
   is the dominant multiplier.
2. **ONNX-graph MAC count.** Even a D=104 toy = ~5e5 MACs/forward; the real
   model = ~4e10. Each MAC = ~4.75 c4 VM instructions.
3. **c4-subset compilability.** The runtime as written is not in the c4 grammar
   (2-D/3-D arrays, `long`, expression bounds, varargs). A full c4-grammar
   rewrite is required before Rel-1 can even *start* on the whole runtime — the
   `onnx_kernel_c4subset.c` kernel shows the pattern but the full port is large.
4. **`O(S²)` context growth** + the 30-token/step frame: a multi-million-step run
   accumulates a >70M-token attention context — memory and time both blow up.
5. **Float-vs-fixed-point** is *correctly* addressed by the fixed-point runtime
   (the tractability lever the brief called out — verified argmax-identical) and
   is **not** the binding constraint here; the binding constraint is items 1-2.
   (A float runtime would add float-emulation on top, making it worse, but even
   the int-only path is months-to-millennia.)

## What is genuinely demonstrated (reproducible)

- The **fixed-point (int-only) ONNX runtime is correct**: argmax byte-identical
  to the float runtime on the tiny `c4vm.onnx` (COO-load fix in
  `onnx_runtime_fixedpoint_coo.c`).
- The **c4-subset MatMul kernel self-hosts**: compiles under c4 (542 words) and
  runs on the reference c4 VM (230 steps, correct fixed-point result).
- The **per-instruction neural-VM cost** is measured (~3.2 s at the real VM
  shape), giving a grounded extrapolation.

## What is NOT demonstrated (honest)

- The full ONNX runtime running **under the neural c4vm** (Rel-1 whole).
- The runtime loading + executing the **real** `c4vm.onnx` (Rel-2).
- The nested 3-layer loop (Rel-3).

These are performance walls, not correctness gaps. The blog's own `TODO
performance analysis` is answered: **full 3-layer self-hosting is not tractable
end-to-end at this model scale; only the kernel level is.**

## Reproduce

```
cd c4_release
PYTHONPATH=$PWD OMP_NUM_THREADS=4 python c4_min/selfhost/run_selfhost_feasibility.py
```

Memory note: the driver builds **nothing dense** by default (per-step cost is
measured with a shape-only synthetic transformer). `--build-complete` also times
the real 64 GB VM — do **not** run it under memory pressure.

## Files

- `c4_min/selfhost/onnx_kernel_c4subset.c` — c4-subset fixed-point MatMul kernel.
- `c4_min/selfhost/onnx_runtime_fixedpoint_coo.c` — int-only runtime + COO load.
- `c4_min/selfhost/run_selfhost_feasibility.py` — the measured feasibility driver.
