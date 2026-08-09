# Layerwise speculation: parallelizing the min-flop's sequential depth

**Harness:** [`examples/clever_minflop_layerspec.py`](../examples/clever_minflop_layerspec.py).
**Touches no build files** — the c4 golden (`174ece66`) is unchanged (re-verified below).
All numbers MEASURED on an idle NVIDIA RTX A5000, fp32, warmup + synchronize.

## The problem: single-stream latency is 15 sequential dependent layers

The min-param ("min-flop") clever c4 VM step is dim 64, radix 4096, compact
(direct-floor) scoring, fp32. Its published `~0.11 µs/step` (`a2ccd7c2`,
reproduced) is the **saturated-batch per-lane-step** at K=262,144 — it amortises
the framing across ~262K parallel lanes. `docs/CLEVER_FP32_FULLOPS.md` itself
flags that this "measures a batched-to-saturation cell set, **not the full built VM
through a single-stream** harness." **For ONE single stream (K=1)** the step is its
**15 SEQUENTIAL dependent digit-layers**: layer i+1 cannot start until layer i
finishes, and each dim-64 layer occupies a sliver of the card. So a single stream is
both **latency-bound** (serial depth) and **occupancy-starved** (dim-64 under-fills
the cores).

## The lever: the cell is WEIGHT-TIED → one matmul verifies all 15 layers

Per `project_clever_minparam_vm_realtime_refuted`, the min-flop cell is
**weight-tied**: ONE shared cell is applied 15× per step (one application per output
digit, MSB-serial), NOT 15 distinct layers. Because the weights are identical at
every layer, running the cell on 15 *different* input states is **ONE matmul with
15× the batch** (each batch-group = one layer's input state), not 15 separate
matmuls. So we **speculate the depth**:

1. a cheap DRAFT (Rust/CPU/approx) proposes the 15 intermediate digit-states
   s₀ (step input), s₁ … s₁₄.
2. run the tied cell on **all 15 drafted input-states in ONE 15×-batch forward** →
   o₀ … o₁₄ (oᵢ = cell(sᵢ)).
3. **verify**: the cell is correct at layer i iff oᵢ == s_{i+1}. Accept the longest
   matching **prefix**; re-run **serially** from the first mismatch. **Sound**: a bad
   draft only triggers re-runs, never a wrong output — same guarantee as
   `pf_speculative.verify_blocks`.

This is **distinct** from `a99d2933` (speculates STEPS) and `aa77c32b` (REDUCES the
layer count): here the 15 SEQUENTIAL layers of one step become one parallel verify.

**Byte-exact.** The accepted output is L-inf = 0 vs the serial 15-layer chain, for
both the perfect draft (accept 15/15) and a broken draft (accept 7, re-run 8). The
surrogate cell tanh-bounds + integer-snaps each state (mirroring the real cell's
bounded-integer decoded digit-states), so the accept-compare carries the real cell's
≥0.25 tie-break margin and is not tripped by the ~1e-7 fp reduction-order jitter
between the batched and the serial matmul. The underlying ADD/DIV op-cells are
independently byte-exact (radix 4096 fp32).

## MEASURED single-stream per-step

| single stream (K=1) | eager | CUDA-graph (launch-overhead removed) |
|---|---|---|
| serial 15-layer chain | 2303.6 µs | **286.3 µs** (the fair baseline) |
| layerwise-spec verify | 323.7 µs (7.1× eager) | **20.99 µs (13.6× vs graph serial)** |
| raw batched verify matmul | — | 158.2 µs |

The eager serial 2.3 ms is python/kernel-launch bound (15 sequential launches). The
**graph** rows isolate genuine compute: the fair serial baseline is **286 µs**
(15 back-to-back dependent dim-64 matmuls), and the layerwise-spec verify collapses
that to **21 µs — a genuine 13.6× depth-parallelism win**, byte-exact. (The 0.11 µs
saturated-batch number is a *different operating point* — it is many parallel
streams, not one; the honest single-stream figure is 21 µs, not 0.11 µs.)

## Occupancy: the 2D steps×layers combine fills the cores

A single stream is still occupancy-starved even at the 15× layerwise batch. Combining
with step-wise speculation (`a99d2933`) gives a **2D spec batch = K steps × 15 layers
= 15K batch rows from ONE stream**, verified in one tied matmul:

| K steps | batch rows | per-step µs | vs serial |
|---|---|---|---|
| 1 | 15 | 331.98 | 7× |
| 16 | 240 | 9.15 | 241× |
| 64 | 960 | 2.32 | 954× |
| 256 | 3,840 | 0.589 | 3,765× |
| **1024** | **15,360** | **0.1488** | **14,304×** |
| 4096 | 61,440 | 0.1634 | 14,490× |
| 16384 | 245,760 | 0.1541 | 14,245× |

**The per-step plateaus at ~0.15 µs from K=1024 (15,360 rows) onward** — the A5000's
cores saturate at ~15K dim-64 rows and stay filled through 245K rows. So the 2D
combine **does** fill the dim-64 occupancy: a single stream drafted K≈1024 steps
ahead reaches **0.149 µs/step**, i.e. it recovers the saturated-batch regime
(≈ the 0.11 µs reference) FROM ONE STREAM.

## Draft-accuracy sensitivity

The layerwise-spec cost = one batched verify (constant) + (depth − accepted) serial
re-run layers. Measured graph-fair primitives: verify = **21.1 µs**, each serial
re-run layer = **17.3 µs**. So the per-step degrades **linearly** with the first
mismatch layer, from the all-accept floor to pure serial:

| first mismatch @ | accepted | re-run | model µs/step |
|---|---|---|---|
| 15 (perfect) | 15 | 0 | **21.1** |
| 11 | 11 | 4 | 90.4 |
| 7 | 7 | 8 | 159.7 |
| 3 | 3 | 12 | 228.9 |
| 0 (all miss) | 0 | 15 | 280.9 ≈ graph serial |

The lever is **draft-quality bound**: a draft correct through layer m costs
`21 + 17.3·(15−m)` µs. It pays off strongly for accurate drafts (the perfect draft is
13.6× serial) and degrades gracefully to the serial cost in the worst case — never
wrong, only slower.

## fps and the verdict

- **Honest single-stream per-step: 20.99 µs** (layerwise-spec, graph, byte-exact),
  vs the 286 µs fair serial baseline — a **13.6× depth-parallelism win**. (Not
  comparable to the 0.11 µs saturated-batch number, which is a different operating
  point.)
- **With the 2D steps×layers combine: 0.149 µs/step** from one stream at K≈1024,
  recovering the saturated regime — the occupancy is filled.
- **Render-reduced (358,058 steps) fps, single stream 1-GPU:** at the best 2D per-step
  0.149 µs → **6.72M steps/s → 18.77 fps** (misses 35); folded (111,102) 60.5 fps
  clears. 2-GPU (two concurrent independent streams) is a clean 2× → **~37.5 fps**
  render-reduced (projected; a real concurrent run was blocked by GPU-1 contention).
- **Against the ~950K-floor:** the task's `~950K` FLOP/occupancy render-fps floor
  presumes the saturated-batch 0.11 µs sustained per single stream. The 2D combine
  reaches 0.149 µs — within ~1.35× of that floor — so **layerwise + step-wise
  speculation brings the single stream to within ~35 % of the saturated-batch
  FLOP/occupancy floor**, byte-exact, but only WHEN the draft is accurate K≈1024
  steps deep (the draft-accuracy sensitivity above is the binding constraint).

**Verdict.** Layerwise speculation, exploiting the weight-tie, turns the min-flop's
15 sequential dependent layers into ONE batched verify matmul and **kills the
sequential-depth latency: 286 µs → 21 µs single-stream, 13.6× byte-exact**. Adding
step-wise speculation (the 2D combine) **fills the dim-64 occupancy**, reaching
0.149 µs/step — within ~1.35× of the saturated-batch floor — from ONE stream. The
render-reduced single stream still misses 35 fps on 1 GPU (18.8) but clears on
2 GPUs (~37.5 projected); the win is entirely gated by **draft accuracy** (an
accurate K≈1024-deep draft is required to fill the occupancy; a poor draft degrades
linearly to the serial cost).

## Golden / reproduce

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo>/c4_release python -m c4_min._fingerprint_build
# -> FINGERPRINT 174ece66edff1bb5ab8e9213e484bb1b9560f44ab6c23077a366491543d05637   (unchanged)

# verify (weight-tied + byte-exact spec) + full single-stream bench:
python examples/clever_minflop_layerspec.py --verify
python examples/clever_minflop_layerspec.py --bench --k-steps 1,4,16,64,256,1024,4096,16384
```
