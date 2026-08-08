# Compact candidate-scoring: getting the clever c4 VM whole-step to 35 fps

**Date:** 2026-08-08 · **Scope:** make the difference-min CANDIDATE SCORING compact so a
larger radix cuts the sequential-layer DEPTH *without* a radix-sized candidate-table FFN
blowup — and MEASURE whether the whole clever VM step then clears Doom's 35 Hz target.
Touches no build files — c4 golden `174ece66` is unchanged (verified before and after).
Code: [`examples/clever_compact_scoring_realtime.py`](../examples/clever_compact_scoring_realtime.py).
All GPU numbers MEASURED on idle RTX A5000s (device 0; 2-GPU = a REAL concurrent
two-device run, both cards free), batched to saturation (batch 262,144).

---

## TL;DR verdict

**YES — compact candidate-scoring clears 35 fps on the render-reduced Doom frame.**

- **Fastest (bf16 throughput proxy):** `direct`-floor scoring, radix 4096, depth 15,
  FFN inter **32** → **53.6 fps 1-GPU · 105.7 fps 2-GPU** (2-GPU measured 1.97×).
- **Fastest BYTE-EXACT (fp32):** same config in fp32 → **24.1 fps 1-GPU · 48.3 fps 2-GPU**
  (measured 2.01×). fp32 is byte-exact through radix 4096, so this clears 35 fps
  **byte-exact on 2 GPUs**; on 1 GPU it is 24 fps (short by 1.5×).

The lever that did it: the difference-min candidate LUT is **not** put in a radix-wide
dense FFN band. Kept compact (arithmetic floor / coarse+fine / bit-serial), the per-layer
FFN stays inter≈32 while the radix grows to 4096–65536, so DEPTH collapses to 15 layers
with **no** candidate-table blowup. The prior shallow-radix result (`CLEVER_SHALLOW_RADIX.md`)
plateaued at ~12 fps because it paid a full framing-softmax per layer AND (honestly) a
radix-linear candidate FFN; removing both (direct T=1 attention + compact scoring) is what
crosses 35 fps.

---

## 1. Where the per-layer step time goes (MEASURED decomposition)

At the narrow-shallow config (d_model 64, radix 4096, batch 262,144), one layer's
components (`decompose_layer`, idle A5000):

| component | bf16 ms | fp32 ms | note |
|---|---:|---:|---|
| Q/K/V/O projections (2 d×d) | 0.40 | 0.83 | the real per-step value/address routing |
| **self-attn (T=1 softmax)** | 1.96 | 2.65 | **a NO-OP at T=1** — softmax over one position = 1 |
| &nbsp;&nbsp;→ direct gather (o=v) | **0.21 (5.0%)** | **0.50 (6.1%)** | byte-identical at T=1; drops the softmax/reshape kernels |
| SwiGLU FFN (inter=256) | 1.92 | 4.58 | scales with FFN intermediate |
| **scoring DENSE r=4096** | **20.15** | **37.91** | **the candidate-table blowup — dominates the whole step** |
| scoring TWO-LEVEL (64×64) | 1.16 (17.4×↓) | 1.60 (23.7×↓) | compact coarse+fine |
| scoring DIRECT floor | 0.018 (1141×↓) | 0.020 (1914×↓) | arithmetic; no LUT |

Two facts fall out:

1. **The self-attention is ~0.6% of the step, as the task states** — but only in the
   *direct-CAM* form. In the batched-verify execution model each lane processes ONE step
   (T=1 in the residual sequence axis), so the ALiBi GQA *self*-attention softmax is
   literally a no-op (o = v). The measured full-softmax path costs ~2.0 ms/layer of pure
   kernel overhead (softmax + reshape/transpose); the equivalent direct gather is 0.21 ms
   (5% of the layer). The cross-position memory read (LI/SI) is the SEPARATE O(1)
   direct-CAM (~2500 fps isolated), not this self-attention.
2. **Dense candidate scoring at large radix is the actual wall** — 20 ms/layer at radix
   4096 (bf16), ~10× the FFN and ~50× the projections. This is exactly the term that
   inverts the depth win in `CLEVER_SHALLOW_RADIX.md`. Compact scoring cuts it 17–1900×.

## 2. Compact candidate-scoring — three byte-exact schemes

The FFN intermediate hosts the candidate band. A dense r-entry LUT makes `inter ∝ radix`
(the blowup). Three compact schemes keep the band small and radix-independent:

| scheme | FFN band | idea |
|---|---|---|
| **dense** (reference) | `radix` | the r-wide difference-min LUT (the blowup) |
| **two_level** | `~2√radix` | split each base-r digit into coarse (rc) + fine (rf), rc·rf=r |
| **log_radix** | `~2 log₂ radix` | bit-serial: peel one binary place per step |
| **direct** | O(1) | `floor(value)` directly — the argmax over centers d+0.5 IS floor |

**Measured band sizes** (`compact_inter`):

| radix | depth | dense | two_level | log_radix | direct |
|---:|---:|---:|---:|---:|---:|
| 16 | 45 | 32 | 32 | 32 | 32 |
| 256 | 25 | 256 | 32 | 32 | 32 |
| 4096 | 15 | **4096** | **128** | **32** | **32** |
| 65536 | 15 | **65536** | **512** | **32** | **32** |

So at radix 4096 the FFN band is **32** (direct/log) or **128** (two_level) instead of
**4096** — a 32–128× smaller per-layer FFN at the same depth-15 shallowness.

## 3. Whole-step fps sweep (MEASURED, render frame 358,058 steps, A5000)

Whole clever STEP transformer at each radix's ADD-step depth, compact-scoring FFN band +
direct T=1 attention, batched to saturation, bf16 + fp32, best batch shown.

### 3a. bf16 (throughput proxy past radix 16)

| scheme | radix | depth | inter | ms/step | render fps | ≥35? |
|---|---:|---:|---:|---:|---:|:--:|
| dense | 4096 | 15 | 4096 | 100.0 | 1.83 | (blowup inverts) |
| two_level | 256 | 25 | 32 | 22.8 | 32.0 | |
| two_level | 4096 | 15 | 128 | 22.5 | 32.5 | |
| log_radix | 4096 | 15 | 32 | 13.9 | **52.7** | ✅ |
| **direct** | **4096** | **15** | **32** | **13.7** | **53.6** | ✅ |
| direct | 65536 | 15 | 32 | 13.8 | 53.1 | ✅ |

### 3b. fp32 (byte-exact through radix 4096)

| scheme | radix | depth | inter | ms/step | render fps | ≥35? |
|---|---:|---:|---:|---:|---:|:--:|
| dense | 4096 | 15 | 4096 | 263.4 | 0.70 | (blowup inverts) |
| two_level | 4096 | 15 | 128 | 53.3 | 13.7 | |
| log_radix | 4096 | 15 | 32 | 30.5 | 24.0 | |
| **direct** | **4096** | **15** | **32** | **30.4** | **24.1** | (1-GPU short) |

- **The dense blowup INVERTS** exactly as `CLEVER_SHALLOW_RADIX.md` found (radix 4096
  dense = 1.8 fps bf16 / 0.7 fps fp32 — *slower* than radix 16). Compact scoring removes
  the inversion: fps now rises monotonically with radix (depth ↓) because the FFN band no
  longer grows.
- **bf16 direct/log_radix at radix 4096/65536 clear 35 fps on 1 GPU** (53 fps).
- **fp32 direct at radix 4096 is 24 fps on 1 GPU** (byte-exact) — short of 35 on one card.

### 3c. REAL 2-GPU runs (both cards, measured concurrently)

| config | dtype | 1-GPU fps | 2-GPU fps | scaling | ≥35 2-GPU? | byte-exact? |
|---|---|---:|---:|---:|:--:|:--:|
| direct radix 4096 depth 15 inter 32 | bf16 | 53.6 | **105.7** | 1.97× | ✅ | proxy (bf16) |
| direct radix 4096 depth 15 inter 32 | fp32 | 24.1 | **48.3** | 2.01× | ✅ | **YES** |

Both 2-GPU scalings are near-perfect (1.97×/2.01×), so these are genuine concurrent-device
throughputs, not a ×2 projection.

## 4. What got it there + the residual per-layer cost

At the compact config (direct scoring, inter 32, direct attn, depth 15, bf16), a single
layer is **0.91 ms** (`decompose_layer`, batch 262,144):

| sub-component | ms | share |
|---|---:|---:|
| direct-attn (2 d×d matmuls) | 0.36 | 40% |
| SwiGLU FFN (3 d×32 matmuls) | 0.54 | 60% |
| (single-tensor bandwidth floor for one residual add) | 0.10 | — |

The layer is ~9× the pure single-tensor-touch bandwidth floor — it is **memory/occupancy
bound** (5 small GEMMs each streaming the residual through DRAM), not FLOP bound (matching
the `CLEVER_DOOM_REALTIME.md` occupancy finding). The three levers that crossed 35 fps,
quantified:

1. **Compact scoring (dominant):** dense→compact removes the 20 ms/layer candidate LUT →
   the layer drops from swamped to ~1 ms. Without it, radix 4096 is 1.8 fps (bf16).
2. **Direct T=1 attention (framing-softmax fold):** full-softmax attn 1.96 ms → direct
   gather 0.21 ms per layer. Halves the non-FFN cost.
3. **FFN narrowing:** inter 256 → 32 (the compact band floor) roughly halves the FFN
   term (0.54 ms vs ~1.0 ms). The bitwise 256-LUT floor only applies to bitwise ops; the
   ADD-class Doom-dominant step needs only the compact scoring band.

Residual gap to close on the byte-exact (fp32) 1-GPU path (24→35 fps, 1.5×): the remaining
cost is the 5 small per-layer GEMMs' DRAM traffic × 15 layers. The two open levers are
(a) **layer fusion** — fuse the 15 layers into one megakernel so the residual is streamed
once, not 15×, and the 5 GEMMs/layer become fused epilogues (this is the same megakernel
lever `CLEVER_DOOM_REALTIME.md`/`PERF_LADDER_FINAL.md` identify); and (b) **fewer layers**
— radix 65536 already floors the ADD-step at depth 15, but a superinstruction that folds
the 4 framing passes into the execute op would cut depth ~5× (to ~3), which alone would
clear 35 fps byte-exact on 1 GPU. bf16 already clears it, but is not byte-exact past radix
16, so the byte-exact single-card route is depth-fold or fusion, not more radix.

## 5. Byte-exactness (compact scoring stays exact within the precision ceiling)

Spot-checked ADD (base-radix limb ripple) + DIV (per-quotient-place long division) digit
extraction on 4,000–8,000 random 32-bit operands per (radix, scheme, dtype)
(`verify_compact_byte_exact`). **Compactness does NOT cost exactness** — every scheme is
byte-exact wherever the datapath dtype holds the op's accumulator; the ONLY failures are
the documented DIV r² precision-ceiling breaks (identical to the dense cell):

| radix | fp32 | bf16 | fp64 | binding limit |
|---:|:--:|:--:|:--:|:--|
| 16 | ✅ | ✅ | ✅ | all accumulators ≤ ceiling |
| 256 | ✅ | ❌ | ✅ | bf16 DIV r²=65,536 > 2⁸ |
| 4096 | ✅ | ❌ | ✅ | fp32 DIV r²=16.7M = 2²⁴ boundary (holds); bf16 overflows |
| 65536 | ❌ | ❌ | ✅ | fp32 DIV r²=4.29e9 > 2²⁴ → needs fp64 |

(all four schemes — dense / two_level / log_radix / direct — pass identically at every
cell where the ceiling holds.)

Two byte-exactness notes worth recording:

- **The difference-min tie-break is precision-sensitive.** The production `ArithCell`
  (fp64) resolves the exact-integer half-way tie (a value at `k+0.5` is equidistant from
  centers `k+0.5±1`) with a vanishing `1e-12·cand` bias. That bias is BELOW fp32/bf16
  epsilon at the ~0.5 tie magnitude, so in low precision the decode goes off-by-one. The
  compact schemes use a **precision-robust** tie-break `cand/(4r)` (max ≈0.25 ≪ the 0.5 tie
  margin, but ≫ fp32 epsilon for r<~1e5), which is byte-identical to the fp64 cell's floor
  on both fractional and exact-integer inputs while staying exact in fp32. This is why the
  fp32 rows above are genuinely byte-exact, not proxies.
- **The bf16 fast rows (radix > 16) are throughput proxies, not byte-exact** — bf16's 2⁸
  integer ceiling breaks the DIV r² accumulator past radix 16 (the same ceiling the dense
  cell hits). The byte-exact fast configuration is fp32 (radix ≤ 4096); the 105 fps bf16
  number is the throughput a byte-exact fp32-speed build would approach with the
  operand-halving rework (2× passes), not a byte-exact number itself.

## 6. Conclusion

- The per-layer wall the shallow-radix work hit (~12 fps) was **candidate-table growth +
  framing softmax**, not depth or batch. Both are removed here: compact scoring keeps the
  FFN band at inter≈32 across radix 16→65536, and the T=1 direct attention drops the
  framing softmax.
- With those, the whole clever VM step **clears 35 fps**: 53.6 fps 1-GPU / 105.7 fps 2-GPU
  in bf16 (throughput proxy), and **48.3 fps 2-GPU byte-exact in fp32** (24.1 fps 1-GPU).
- The candidate scoring at large radix is compact **and** byte-exact simultaneously — the
  precision ceiling (DIV r²) is the only limit, and it is the same one the dense cell has.
  Compactness is free of exactness cost.
- The remaining byte-exact-single-card gap (24→35 fps) is the memory-bound per-layer GEMM
  traffic × 15 layers; the levers are megakernel fusion (stream the residual once) or a
  framing-fold superinstruction (depth 15→~3), quantified in §4.

### Reproduce

```
# compact-band table + byte-exact spot-check (CPU, fast):
python examples/clever_compact_scoring_realtime.py --verify

# whole-step fps sweep + per-layer decomposition + REAL 2-GPU (both cards free):
python examples/clever_compact_scoring_realtime.py --bench --two-gpu --device cuda:0 --json out.json
```

Golden unchanged: this doc + `examples/clever_compact_scoring_realtime.py` touch **no build
files**; `CUDA_VISIBLE_DEVICES="" python -m c4_min._fingerprint_build` = `174ece66` before
and after.
