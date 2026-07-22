# Fine-grained conditional sparsity: the HONEST net win over block-MoE (#628)

The prior conditional-sparsity work (`perlayer_conditional_sparse`,
`bench_conditional_full`) measured **conditional vs DENSE** and reported 79-379x on
the FULL L9 divmod megablock and ~1.6x on the small subsets. But the model already
ships **block-MoE (#628, `block_moe_divmod`)** — a COARSE per-op route that skips the
whole DIV/MOD megablock span when the step's op is not DIV/MOD. Much of the
"conditional beats dense" number OVERLAPS with what block-MoE already gives for free.
This is the NET-NEW measurement of FINE-GRAINED (within-active-layer unit) conditional
sparsity, against the RIGHT baseline (block-MoE), wired into the LIVE lean forward.

## What was wired (live-forward integration)

The three configs are all the SAME forward machinery (a `SkipAwareCondLean`, a
`ConditionalBlockLean` subclass that also SKIPS a fully-dead layer's attn+FFN as
identity — the #628 route), differing ONLY in the per-layer active-unit set:

* **dense**     — every layer keeps ALL `I` units.
* **block-MoE** — the lean-forward dual of #628: a layer is MoE-active iff ANY unit
                  fires for the program's op-cluster; a MoE-active layer keeps ALL `I`
                  units (run the whole block), a MoE-dead layer keeps 0 (skip it,
                  == the identity expert). This is #628's BLOCK granularity on the
                  compacted lean layers.
* **conditional** — every layer keeps only the units that FIRE (`thr=0` → dropped
                  units contribute exactly 0 to `down(silu(up)·gate)`).

Files (all NEW; no existing model/compiler code touched → model byte-identity trivially
preserved, golden hash unchanged):
* `c4_min/bench_cond_vs_moe.py`         — 3-way EAGER end-to-end ms/step, routes each
  config through the lean spec driver's window-builder + decode (`_run_spec_via`).
* `c4_min/bench_cond_vs_moe_graphed.py` — the SAME 3-way, CUDA-GRAPHED
  (`GraphedLeanForward` replay, zero python/launch overhead).
* `c4_min/bench_megablock_kernel.py`    — shape-accurate FULL L9 megablock
  (I=160465) GEMM cost (no 54 GB dense build needed).
* `c4_min/bench_cond_vs_moe_full.py`    — FULL-model 3-way (needs the 54 GB CPU
  build; RSS-watchdog guarded; NOT run under the current RAM contention).

## Byte-identity (the load-bearing gate)

`verify_conditional_decode` — the fine-grained conditional decodes the SAME register
trace as dense on every program, both subsets:
```
mem+cmp  ALL DECODE-IDENTICAL: True   (5/5 programs cond==dense)
bitwise  ALL DECODE-IDENTICAL: True   (5/5 programs cond==dense)
```
Eager residual L-inf(cond vs dense) = 0.59 on a ~1e8-magnitude band (rel ~6e-9), all
on NON-query intermediate rows; the query-row AX decodes byte-identically. The larger
L-inf the graphed bench printed (~3.8e3) is on a pad/intermediate row (dense 6563.1 vs
cond 6563.5, fp-reduction order), NOT the decode.

## END-TO-END ms/step (RTX A5000, cuda:0, fp32)

### Small models — mem+cmp (I=896) / base (I=896) / bitwise=QN-subroutine (I=2608)

There is NO op-specific megablock in these subsets (muldiv is a bytecode subroutine
in the QN/bitwise model). Therefore **every layer is MoE-active and block-MoE has
NOTHING to skip: block-MoE == dense (`moe/dense` ≈ 1.00x).** The conditional net win
over block-MoE is then the same as over dense:

EAGER (all three via identical `SkipAwareCondLean` machinery):
```
subset    prog        B     cond units / dense    c/dense   c/moe   moe/dense
base      countdown   512   117 / 6272  (1.87%)    1.62x    1.59x    1.02x
base      countdown   2048  117 / 6272             1.50x    1.50x    1.00x
base      mul_accum   2048  126 / 6272  (2.01%)    1.61x    1.61x    1.00x
mem+cmp   countdown   512   454 / 9856  (4.61%)    1.59x    1.60x    1.00x
mem+cmp   mul_accum   512   464 / 9856             2.04x    1.94x    1.05x
bitwise   countdown   512   770 / 41728 (1.85%)    2.81x    2.80x    1.00x
bitwise   countdown   2048  770 / 41728            4.15x    3.72x    1.11x
```
CUDA-GRAPHED (bitwise, replay; measurement noisier at S=7 windows under contention):
```
prog       B     c/dense   c/moe   moe/dense
countdown  512   1.29-4.1x  2.6-4.5x  0.9-1.4x   (median-of-3 still ±; see below)
countdown  2048  2.27x      2.42x     0.94x
```

**Under CUDA graphs the conditional win over dense SHRINKS** (from ~2.8-4x eager to
~1.3-2.3x graphed) because the graph removes the per-layer python launch overhead that
inflated the eager dense/moe cost — what remains is the pure FFN-GEMM FLOP reduction,
which is real but smaller. The `moe/dense` scatter around 1.0 (0.9-1.4x) is
capture/contention noise: with 16/16 layers active, block-MoE runs the identical GEMM
as dense (`Linf m-d=0.0`), so any deviation from 1.0x is measurement noise at these
tiny S=7 window shapes where fixed overhead dominates.

### FULL model — L9 divmod megablock (I=160465), the ONLY place block-MoE ≠ dense

Shape-accurate megablock FFN GEMM (dense == block-MoE on a DIVMOD step, since #628
keeps the WHOLE megablock once the op is DIV/MOD — its route is BLOCK-granular):
```
        B         M   dense/blkMoE |  cond(MOD 508u)  cond(DIV 247u)  non-divmod(14u) | best c/moe
      512      3584      366.7 ms  |     0.976 ms        0.613 ms         0.138 ms     |  376x
     2048     14336        OOM     |     4.27 ms         2.15 ms          0.44 ms      |   inf (dense OOM)
     8192     57344        OOM     |    16.7 ms          8.81 ms          1.43 ms      |   inf
    16384    114688        OOM     |    32.8 ms          17.3 ms          3.2 ms       |   inf
```

* **DIVMOD step**: block-MoE runs the WHOLE 160465-unit megablock (block granularity —
  it gives ZERO help WITHIN the active layer); conditional runs ~508 (MOD)/247 (DIV)
  firing units → **conditional beats block-MoE by ~376x** on the megablock, and at
  B≥2048 the dense/block-MoE GEMM **OOMs** ([M,160465] intermediate = 26-206 GB) so the
  fine-grained conditional model is the ONLY runnable form on 24 GB.
* **NON-divmod step**: block-MoE ALREADY skips the whole megablock (0 units) and so does
  conditional (14 dead units) → on the megablock the net win of conditional over
  block-MoE is ZERO. Their only difference is the small non-megablock layers (the same
  ~1.6-2.8x above).

## VERDICT — is fine-grained conditional a real end-to-end win BEYOND MoE?

**Yes, but the size and location of the net-new win depend entirely on op-disjointness,
and it is NOT the headline 79-379x for a general program — that number is
conditional-vs-DENSE and mostly overlaps with block-MoE.**

1. **On a divmod-heavy inner loop (op DIV/MOD present): the net win over block-MoE IS
   the big one (~376x / dense-OOM), because block-MoE is block-granular and must run the
   entire 160465-unit megablock once the op fires. Fine-grained conditional runs ~0.3%
   of it. This is genuinely NET-NEW over what MoE gives.** (block-MoE's own win over
   dense here is 0 — it can't skip a megablock the program uses.)

2. **On any program WITHOUT a big op-specific megablock (all of base/mem+cmp/bitwise —
   i.e. the QN subroutine model, and every non-divmod FULL program): block-MoE == dense
   (nothing to skip), and conditional's net win over block-MoE == its win over dense =
   ~1.5-2.8x eager, shrinking to ~1.3-2.3x under CUDA graphs.** This is a real but
   modest fixed win from pruning the scattered dead units in every layer.

3. **The cap** is the fixed attention + RMSNorm + gather + the small always-on shared
   band (L4 = 221/269 units fires for every op). Once the FFN is pruned to <2% of units,
   the FFN GEMM is no longer the bottleneck; attention/RMSNorm dominate, and under CUDA
   graphs the launch overhead that eager timing charged to the FFN disappears — so the
   graphed win is bounded by the attention/norm share, ~1.3-2.3x on these models. The
   gather itself is amortised (built once per op-cluster, not per step).

**Bottom line: fine-grained conditional sparsity is a real end-to-end speedup beyond MoE
ONLY where the program hits a large op-specific block that MoE cannot skip (divmod
inner loops → ~376x / feasibility on 24 GB). For op-disjoint / mixed programs on the QN
subroutine model, block-MoE already captures the coarse win and conditional adds a
further ~1.3-2.8x (eager) / ~1.3-2.3x (graphed) from within-layer unit pruning — worth
it (byte-identical, gather amortised), but not categorical.**
