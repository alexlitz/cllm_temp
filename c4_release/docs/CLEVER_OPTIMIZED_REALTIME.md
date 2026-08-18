# Does the clever c4 VM reach Doom realtime under OPTIMIZED execution? — MEASURED

**Date:** 2026-08-08 · **Scope:** re-measure the clever transformer's Doom fps after
applying the untapped execution levers the dense-eager baseline left on the table —
**narrow width · adaptive precision · fusion (compile/CUDA-graph) · real int8** — and give
an honest realtime verdict. **All numbers MEASURED** on a clean RTX A5000, not projected.
**Touches no build file** — the c4 golden `174ece66` is unchanged (verified §8).

This is the OPTIMIZED-execution counterpart to
[`docs/CLEVER_REALTIME_MEASURED.md`](CLEVER_REALTIME_MEASURED.md). That doc measured the
clever stack at **hidden 896 with real 896×896 Q/K/V/O + FFN GEMMs in eager PyTorch** —
where **~99.99 % of the FLOPs multiply zeros** (the clever cells are ~3k–26k nonzeros
sprinkled into a 217 M-param zero shell), with **no compile / graph / fusion / sparsity**
and **uniform precision**. The 896 width exists ONLY to fit a stock Qwen2.5-0.5B
checkpoint. This doc removes that multiply-by-zeros tax and measures every lever.

Code (new, under `examples/`, no build files touched):
- [`examples/clever_optimized_realtime.py`](../examples/clever_optimized_realtime.py) —
  the minimal-width derivation, the narrow byte-exact spot-check, and the GPU lever sweep
  (narrow · precision · torch.compile · CUDA-graph · real int8 tensor-core matmul).

Reproduce:
```
python examples/clever_optimized_realtime.py --verify                 # narrow byte-exact (CPU)
python examples/clever_optimized_realtime.py --bench --device cuda:0  # full lever sweep (GPU)
python examples/clever_optimized_realtime.py --bench --json out.json
```

---

## TL;DR verdict

**NO — but the levers buy 34× and close most of the gap.** Optimized execution takes the
clever VM from the dense-eager baseline's **0.24 render fps → 8.3 render fps** (the
`51-layer` byte-exact depth) / **10.2 render fps** (the `42-layer` min-walltime depth) — a
**34–49× speedup** — but that is still **~3× short of 30 fps** on the render-reduced frame.

**Why it does NOT clear realtime:** narrowing removes the multiply-by-zeros tax (19×), but
the remaining wall is **the compute/memory-bandwidth of running 42–51 real transformer
blocks every single VM step**. A CUDA graph (which erases ALL kernel-launch overhead)
recovers only **1–2 %** of the step — so the stack is **NOT launch-bound; it is
compute/bandwidth-bound**. The only way past 30 fps is a *shallower* stack (fewer
digit-layers per step) or a non-transformer datapath, not more of these four levers.

| config (render frame, 358,058 steps) | ms/step | ns/lane-step | **render fps** | ≥30? | vs baseline |
|---|---:|---:|---:|:--:|---:|
| **baseline** dense-eager bf16, hidden **896**, 51L | 11.45 | 11,454 | **0.244** | ❌ | 1.0× |
| + **narrow** d_model **32** (bf16), 51L | 0.59 | 593 | **4.71** | ❌ | 19.3× |
| + **fusion** (torch.compile), d32, 51L | 0.33 | 335 | **8.34** | ❌ | **34.2×** |
| + fusion, d32, **42L** (min-walltime depth) | 0.28 | 275 | **10.16** | ❌ | **48.6×** |
| CUDA-graph d32 51L (launch-erased) | 0.59 | 587 | 4.75 | ❌ | 19.5× |
| int8 tensor-core (torch._int_mm) | — | — | — | — | **0.55–0.59× (slower)** |

All MEASURED on an idle **RTX A5000** (device 0), batched to throughput saturation
(batch 262,144 lanes; fp64 saturates by 4,096). Random-weight forwards (see §7 honesty —
zero weights let compile/graph dead-code-eliminate the whole forward and report physically
impossible ~0-ns "speedups"; small randoms make it genuine work).

---

## 1. The minimal-sufficient width derivation (the big lever)

The 896 width is a checkpoint-fitting artifact. The clever VM's TRUE minimal state is the
**max number of residual dims that must be held CONCURRENTLY across all ops in one step** —
the residual is what flows layer-to-layer and drives the **O(d_model²)** attention Q/K/V/O +
FFN matmul cost:

| live residual band | dims |
|---|---:|
| VM registers/frame the step threads (PC, SP, BP, AX) | 4 |
| operand value axis + running remainder (arith datapath) | 2 |
| ingest flag axis (BOS / op / `=` one-hots) | 3 |
| difference-min candidate band held concurrently (NCAND = 10) | 10 |
| scratch / carry (sign, wrap, tie-break) | ~4 |
| **PEAK concurrent residual dims** | **23** |
| **→ minimal d_model (next power of two)** | **32** |

The **bitwise 256-unit nibble-LUT** and the **128-wide CAM address key** are **FFN-hidden /
attention-projection dims, NOT residual dims** — they cost *inside* a layer's FFN/attn but
do not widen the residual the O(d_model²) matmuls scale with. (The clever FFN intermediate
must still be ≥ 256 to host the bitwise LUT — carried as the honest FFN cost, `inter=256`.)

So **d_model = 32** is the derived minimal sufficient residual width (28× narrower than 896).
We sweep {32, 64, 128, 256} to show the width→walltime curve.

### Narrow byte-exactness (correctness preserved)

The clever cells compute each op at their **natural width** (arith d=4, bitwise 32→256→1,
CAM 8-nibble key); embedding them in a d_model ≥ 23 zero-padded residual changes **nothing**
about the arithmetic. Spot-checked at d_model = 32 on **3,000 random 32-bit operands/op**:

| op | ADD | SUB | DIV | MOD | CMP(EQ/NE/LT/GT) | OR | XOR | AND | CAM(LI) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| narrow byte-exact | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**9/9 op families byte-exact at the narrow width, 0 errors** (the full 14/14 family set is
verified in [`clever_realtime_cells.py`](../examples/clever_realtime_cells.py); this
spot-checks the representative ADD/SUB/DIV/CMP/bitwise/CAM at d=32). The narrow reformulation
is byte-exact-preserving by construction: it only removes zero-padding dims.

---

## 2. The four levers, MEASURED (each vs the dense-eager bf16 baseline)

All at 51 layers unless noted; best (saturating) batch = 262,144; bf16.

### Lever 1 — NARROW WIDTH (the big lever)

| d_model | ns/lane-step | render fps | raw fps | speedup vs baseline (896) |
|---:|---:|---:|---:|---:|
| **896 (baseline)** | 11,454 | 0.244 | 0.0127 | 1.0× |
| **32** (derived min) | **593** | **4.71** | 0.245 | **19.3×** |
| 64 | 790 | 3.54 | 0.184 | 14.5× |
| 128 | 1,269 | 2.20 | 0.114 | 9.0× |
| 256 | 2,422 | 1.15 | 0.060 | 4.7× |

Narrowing 896→32 buys **19.3×**. Note it is **sub-linear** in the 28× width reduction
(19× not 28×) — the first sign that at d=32 the stack is no longer compute-density-bound on
the big GEMMs but is hitting the **elementwise/bandwidth floor** (softmax, ALiBi bias, silu,
residual adds) that does not shrink with width. That floor is exactly what fusion attacks.

### Lever 2 — ADAPTIVE / MIXED PRECISION (at the derived minimal width d=32)

Uniform-precision points at d=32, plus the per-op minimum-precision assignment (from
`c4_min/opconfig.py`'s `acc_max` / `max_safe_radix`):

| precision (uniform, d=32) | ns/lane-step | render fps | vs fp64 | vs fp32 |
|---|---:|---:|---:|---:|
| fp64 (byte-exact whole-value floor) | 11,687 | 0.239 | 1.0× | — |
| fp32 | 1,113 | 2.51 | 10.5× | 1.0× |
| **bf16** | **594** | **4.70** | **19.7×** | **1.87×** |

**The per-op minimum-precision map (digit-extract radix-16 form, the fast byte-exact path):**

| op family | acc_max @ r16 | min sufficient precision | why |
|---|---:|---|---|
| ADD/SUB/LEA/frame (the **63 %+ Doom pointer-walk**) | 32 | **bf16** | 32 ≤ bf16 ceiling 256 |
| CMP ×6 | 17 | **bf16** | 17 ≤ 256 |
| DIV/MOD | 256 | **bf16** (boundary, exact) | 256 = bf16 ceiling |
| MUL (64-bit product) | 1,904 col-peak | **fp16** | 1,904 ≤ fp16 ceiling 2,048 |

**The key finding:** in the digit-extract form **the ENTIRE Doom op mix runs bf16-exact**
(ADD/SUB/CMP/DIV/MOD all fit bf16; only MUL — **1 % of Doom steps** —
[`serial_doom_floor.py` `DOOM_MIX`] needs fp16, the *same* tensor-core speed). **No op needs
fp64 in the fast form.** So adaptive precision's win is not a per-op blend of slow and fast
datapaths — it is that **the whole VM runs on the fast bf16/fp16 tensor-core path and NEVER
pays the ~20× fp64 tax** the whole-value min-nonzero corner would. vs the fp32 the naive
build might pick, that is **1.87×**; the fp64 whole-value byte-exact floor is **19.7× slower**
and adaptive precision's contribution is *avoiding it entirely*. Every op keeps exactness at
its assigned precision (verified by the acc_max ≤ ceiling bound above + the §1 byte-exact
spot-check).

### Lever 3 — FUSION (torch.compile + CUDA graphs)

On the best narrow config (d=32, bf16, 51L):

| fusion lever | ns/lane-step | render fps | speedup vs eager narrow |
|---|---:|---:|---:|
| eager narrow d32 | 593 | 4.71 | 1.0× |
| **torch.compile** | **335** | **8.34** | **1.77×** |
| CUDA graph (`torch.cuda.CUDAGraph`) | 587 | 4.75 | 1.01× |

**torch.compile buys 1.77×** — it fuses the elementwise attention/FFN glue (softmax, ALiBi
bias add, silu-gate multiply, residual adds) into the GEMM epilogues, cutting the
memory-bandwidth round-trips that the narrow stack is bound on. **CUDA graphs buy essentially
nothing (1.01×)** — see the diagnostic below.

### Lever 4 — REAL int8 tensor-core matmul (`torch._int_mm`, INT8×INT8→INT32)

The baseline used fp16 as an int8 *proxy*. Here we measure the **real** INT8 tensor-core GEMM
against a bf16 GEMM of the same FFN-class shape (M × 32 @ 32 × 256):

| batch | bf16 GEMM (ms) | int8 GEMM (ms) | int8 / bf16 |
|---:|---:|---:|---:|
| 4,096 | 0.0091 | 0.0082 | 1.11× |
| 16,384 | 0.0169 | 0.0286 | **0.59×** |
| 65,536 | 0.0606 | 0.107 | **0.57×** |
| 262,144 | 0.229 | 0.413 | **0.55×** |

**Real int8 does NOT realize >bf16 at the saturating batch** — it is **0.55–0.59× (i.e.
~1.8× SLOWER)** at batch ≥ 16k, only marginally faster (1.1×) at the tiny launch-bound batch
4,096. On this Ampere card `torch._int_mm` rides the same tensor cores as bf16 but pays a
wider INT32 accumulator write-out and a less-tuned kernel at these skinny-K shapes (K=32).
**int8 is not a lever here** — matching `CLEVER_REALTIME_MEASURED.md`'s honest finding that
int8's theoretical 2.7× is not realized past bf16 on this hardware.

---

## 3. The remaining wall — compute/bandwidth, NOT launch

The single most diagnostic measurement: a **CUDA graph replays the entire narrow forward with
ZERO per-kernel launch overhead** (one graph launch instead of ~7 × 51 kernel launches). If
the stack were launch/dispatch-bound, the graph would be a large win. Measured:

> eager narrow d32 = **593 ns/step** · CUDA-graph d32 = **587 ns/step** →
> **launch overhead erased = 1.0 % of the step.**

**The clever narrow stack is NOT launch-bound.** The step cost is the genuine
**compute + memory-bandwidth of 51 real transformer blocks** (each: 7 matmuls + softmax +
silu + two residual adds, over batch × 32 × {32, 256}). torch.compile's 1.77× is a
*bandwidth* win (kernel fusion cuts HBM round-trips), not a launch win. This is the opposite
diagnosis from a kernel-launch-bound regime — and it is why no amount of further fusion
clears the gap: **the FLOPs/bytes of 42–51 layers per step are the floor.**

---

## 4. COMBINED best config + the realtime verdict

Combining narrow (d=32) + adaptive precision (bf16, no fp64) + fusion (torch.compile), at the
two clever depths:

| combined config | ns/lane-step | **render fps** | raw fps | ≥30? | ≥60? | vs baseline |
|---|---:|---:|---:|:--:|:--:|---:|
| narrow d32 + fusion, **51L** (byte-exact whole depth) | 335 | **8.34** | 0.434 | ❌ | ❌ | **34.2×** |
| narrow d32 + fusion, **42L** (min-walltime radix-16 depth) | 275 | **10.16** | 0.528 | ❌ | ❌ | **48.6×** |

**REALTIME VERDICT: NO.** Even fully optimized, the clever VM reaches **8.3–10.2 render fps**,
which is **~3.0–3.6× short of 30 fps** (and ~6–7× short of 60 fps) on the render-reduced
frame. To hit 30 fps at 358,058 steps the step must fall to **≤ 93 ns**; the optimized floor
is **275–335 ns**.

### Per-lever speedup breakdown (multiplicative, from the 0.244-fps baseline)

| lever | factor | running fps |
|---|---:|---:|
| dense-eager bf16 @ 896 (baseline) | 1.0× | 0.244 |
| **narrow width** (896 → 32) | **× 19.3** | 4.71 |
| **adaptive precision** (already on the bf16 tensor-core path; avoids the ~20× fp64 whole-value tax — worth **1.87×** vs an fp32 build) | ×1.0 here* | 4.71 |
| **fusion** (torch.compile) | **× 1.77** | 8.34 |
| **int8** | ×1.0 (not realized; 0.55× slower) | 8.34 |
| **CUDA graph** (launch already ~1 %) | ×1.01 | 8.34 |
| *(shallower 42L depth)* | ×1.22 | 10.16 |

\*The baseline is *already bf16*, so adaptive precision does not multiply the baseline again;
its value is that the byte-exact clever VM does **not** have to run whole-value fp64 (which
would be **19.7× slower** — the fp64 d32 row). **Narrow width dominates (19×); fusion is the
only additional multiplier that lands (1.77×); int8 and CUDA-graphs contribute nothing on
this hardware/shape.**

### The remaining wall, stated plainly

**Memory-bandwidth + compute of the multi-layer transformer stack per step.** Not kernel
launch (CUDA graph erases only 1 %), not attention O(T²) (T=1 decode — the score matrix is
1×1 per lane), not the arithmetic cell (it is ~4 scalars). The binding cost is **42–51 real
transformer blocks executed every VM step**. The only levers past it are architectural:
**fewer digit-layers per step** (a larger radix / a shallower decode, e.g. the min-nonzero
corner's shallow whole-value form — but that needs the slow fp64 datapath), or a
**non-transformer** narrow datapath (the bare difference-min cell, ~75 FLOP, which the
`CLEVER_DOOM_REALTIME.md` cell-only projection clocks at ~69 fps — but that is not a
realizable *transformer*, it is the isolated cell).

---

## 5. Raw-frame implication

The raw title-redraw frame is **6,889,264 steps** (~19.2× the render-reduced 358,058). Every
raw verdict is ~19× worse:

| config | render fps | **raw fps** |
|---|---:|---:|
| combined best (narrow+fusion, 42L) | 10.16 | **0.528** |
| combined best (narrow+fusion, 51L) | 8.34 | 0.434 |
| baseline dense-eager 896 | 0.244 | 0.0127 |

The raw frame is **~57× short of 30 fps** even fully optimized — it does not fit and still
needs the render-macro step-fold (as the perf ladder has long held). The optimization levers
close the render-frame gap from ~123× to ~3×, but do not by themselves make the raw frame
realtime.

---

## 6. Why this differs from the dense-eager measurement

`CLEVER_REALTIME_MEASURED.md` reported **0.31 fps** (best) and concluded ~100× short. That
was the **hidden-896, eager, uniform-precision, unfused** shape — 99.99 % zeros multiplied
anyway. This doc keeps the *identical clever cells and byte-exactness* but removes the
checkpoint-fitting overhead:

| | dense-eager (that doc) | OPTIMIZED (this doc) | gain |
|---|---:|---:|---:|
| best render fps | 0.31 (bf16 42L) | **10.2** (narrow+fusion 42L) | **~33×** |
| binding wall | full 896-wide stack/step | 42–51 real blocks/step (bandwidth) | — |
| short of 30 fps | ~100× | **~3×** | — |

Both are honest and consistent: the earlier doc's wall ("the full-width stack per step") is
correct; this doc shows **most of that wall was the multiply-by-zeros width, and removing it
(+ fusion) recovers 33× — leaving a residual ~3× that is the irreducible cost of a
multi-layer transformer datapath per step.**

---

## 7. Honesty — measured vs projected, saturation, the DCE trap

- **MEASURED:** every fps here follows from a measured ns/lane-step on an idle A5000 at
  throughput saturation (swept to batch 262,144; fp64 to 4,096). The narrow byte-exactness
  (9/9 families, 3,000 operands/op at d=32). The int8-vs-bf16 real tensor-core ratio
  (`torch._int_mm`). The CUDA-graph launch-fraction (1 %).
- **The zero-weight DCE trap (caught and fixed):** the dense-eager baseline used **zero-init**
  weights. Under `torch.compile` / CUDA-graph capture, a zero-weight forward is a constant
  no-op that the optimizer **dead-code-eliminates**, reporting physically impossible
  ~0.02-ns/step "speedups" (150,000 fps). This doc's model uses **small random weights**, so
  every forward is genuine work no optimizer can fold — the 1.77× / 1.01× fusion numbers are
  real. (An early zero-weight run produced the bogus 33,000× "graph speedup"; it was
  discarded.)
- **PROJECTED / derived:** the per-op minimum-precision *assignment* is derived from
  `opconfig.acc_max` (the exact accumulator bound), not re-timed per op — but the resulting
  claim ("the whole Doom mix runs bf16/fp16, no fp64") is checked against the measured
  bf16/fp16/fp64 d32 rows. The Doom op-mix weighting (63 % pointer-walk, 1 % MUL, 0.5 % DIV)
  is the `serial_doom_floor.py` `DOOM_MIX` ESTIMATE grounded in the measured 63 % fact.
- **The clever cells are byte-exact; the SHAPE is what is timed.** The random-weight
  transformer is a throughput stand-in for the clever cells' matmul *cost* — the cells' actual
  byte-exactness is proven separately (§1 + `clever_realtime_cells.py` 14/14).

---

## 8. Golden preserved

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo>/c4_release python -m c4_min._fingerprint_build
# -> FINGERPRINT 174ece66edff1bb5ab8e9213e484bb1b9560f44ab6c23077a366491543d05637
```

**`174ece66`, unchanged** (verified before and after this work). This doc + the single new
example script touch **no** build file.
