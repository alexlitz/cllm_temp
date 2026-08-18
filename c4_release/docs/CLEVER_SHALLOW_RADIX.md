# Does SHALLOW + wide-batch convert the clever VM's FLOP reduction to realtime?

**Date:** 2026-08-08 · **Scope:** the DEPTH lever — reduce sequential digit-layers
via larger RADIX, and MEASURE whether batching the shallower stack clears 30 fps
on the render-reduced Doom frame. Touches no build files — c4 golden `174ece66`
is unchanged (verified before and after).
Code: [`examples/clever_shallow_radix_realtime.py`](../examples/clever_shallow_radix_realtime.py).
All GPU numbers MEASURED on an idle RTX A5000 (device 0).

---

## TL;DR verdict

**The depth cap IS real and shrinking it DOES convert to wall-clock — near-perfectly
proportionally (3.0× less depth → 2.85–2.95× more fps) — but the shallower clever
*transformer* still does NOT clear 30 fps: it tops out at ≈ 11.9 fps (bf16, radix
4096, depth 15), ~2.5× short.** So "parallelism converts the FLOP reduction" is
CONFIRMED as a mechanism (fps tracks 1/depth), but the shallow radix VM as a full
attention+FFN transformer is per-layer too heavy to reach realtime — the remaining
gap is per-layer transformer cost (real GQA-softmax + SwiGLU), not depth.

Two things gate it, and both are now measured, not projected:

1. **Depth reduction converts (the crux, YES).** Holding the FFN intermediate FIXED
   to isolate the pure depth lever, fps scales ~linearly with 1/depth: radix 16
   (depth 45) → radix 4096/65536 (depth 15) gives **2.85× fps in fp32, 2.95× in bf16**
   for **3.0× less depth**. The batched-compute-bound stall the narrow VM hit at
   ~10 fps was genuinely the sequential-depth cap — lifting it raises fps
   proportionally.
2. **But the candidate-table growth can EAT the depth win (the radix↔nonzero tax).**
   With the *honest* radix-linear intermediate (the difference-min candidate LUT
   grows with radix), the per-layer FFN cost balloons 256→8192 (32×) while depth
   only shrinks 3×, so fps INVERTS past radix 256: best honest is **7.2 fps (bf16,
   radix 256)**, and radix 4096/65536 are SLOWER than radix 16. The depth win is
   real only if the candidate scoring stays compact (attention-hosted), not a
   radix-linear dense FFN.
3. **Precision ceiling caps the usable radix (byte-exactness).** DIV's r² accumulator
   is the binding op: bf16 (2⁸ ceiling) is byte-exact only at radix 16; fp32 (2²⁴)
   holds up to radix 4096 (r²=16.7M, exactly the boundary); **radix 65536 DIV
   (r²=4.29e9) overflows fp32 → needs fp64.** So the fastest *byte-exact* points
   are radix 256 (fp32) and radix 4096 (fp32); the radix-65536 fps is fp64-only for
   exactness (and the bf16 fps rows past radix 16 are throughput proxies, not exact).

---

## 1. The lever: DEPTH = summed-ISA digit-layers, shrunk by RADIX

The clever construction extracts one digit/limb per reused layer. For a 32-bit
operand at radix `r`, the number of limbs (= sequential layers) for an op is
`ceil(result_bits / log2 r)`. A c4 STEP = ~4 ADD-class framing passes (opcode-select
+ PC/SP/BP writeback) + 1 execute op (Doom is 63% ADD-class pointer walk). Larger
radix packs more bits/limb → fewer layers.

**MEASURED depth table** (`summed_isa_depth`):

| radix | log₂ | ADD limbs | DIV limbs | MUL limbs | CMP limbs | ADD-step | DIV-step | MUL-step | full-ALU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 9 | 8 | 16 | 8 | **45** | 44 | 52 | **96** |
| 256 | 8 | 5 | 4 | 8 | 4 | **25** | 24 | 28 | 52 |
| 4096 | 12 | 3 | 3 | 6 | 3 | **15** | 15 | 18 | 33 |
| 65536 | 16 | 3 | 2 | 4 | 2 | **15** | 14 | 16 | 30 |

The narrow clever VM's 42–51-layer wall (`CLEVER_OPTIMIZED_REALTIME.md`) is the
radix-16 anchor: the ADD-step is 45 and the full-ALU budget is ~51 (reported 96 for
the whole ADD+SUB+CMP+SHL+SHR+DIV+MOD budget unrolled). Radix 4096/65536 collapse the
ADD-step to depth 15 — the shallowness the narrow-VM verdict asked for.

## 2. Honest size + the radix↔nonzero↔precision tradeoff

Larger radix buys fewer layers but costs a wider difference-min candidate band and a
higher precision floor:

| radix | d_model | ffn_inter (honest) | candidate band | binding min-prec | fp32 exact-all | bf16 exact-all | DIV r² vs fp32 2²⁴ |
|---:|---:|---:|---:|:--|:--|:--|:--|
| 16 | 64 | 256 | 16 | fp16 | ✅ | ❌ | 256 ≤ fp32 |
| 256 | 64 | 256 | 256 | fp32 | ✅ | ❌ | 65,536 ≤ fp32 |
| 4096 | 64 | 4096 | 4,096 | fp64 (DIV) | ✅ | ❌ | 16,777,216 = fp32 boundary |
| 65536 | 64 | 8192 (capped) | 65,536 | fp64 (DIV) | ❌ | ❌ | 4,294,967,296 **> fp32** |

- **Residual d_model** grows only mildly (it carries the *selected* limb, not the
  whole band) → stays 64 across the sweep.
- **FFN intermediate** hosts the radix-wide candidate LUT and grows LINEARLY with
  radix (256 → 8192). This is the "bigger candidate table" tax.
- **DIV is the binding op** (accumulator r²). It sets the precision floor: fp32
  covers radix ≤ 4096; radix 65536 needs fp64.

## 3. MEASURED fps-vs-depth curve (render frame 358,058 steps, RTX A5000)

The clever stack is built at each radix's ADD-step depth as a genuine narrow
transformer (real ALiBi GQA-softmax + SwiGLU FFN, small random non-zero weights so
nothing dead-code-eliminates), batched to saturation (best batch shown), fp32 and
bf16, eager and CUDA-graph (best of the two shown).

### 3a. HONEST radix-linear intermediate (candidate table grows with radix)

| radix | depth | fp32 fps | fp32 ms/step | bf16 fps | bf16 ms/step |
|---:|---:|---:|---:|---:|---:|
| 16 | 45 | 2.27 | 322.1 | 4.03 | 181.7 |
| 256 | 25 | 3.98 | 184.0 | **7.21** | 101.5 |
| 4096 | 15 | 0.70 | 65.7 | 1.69 | 434.4 |
| 65536 | 15 | 0.35 | 527.4 | 0.87 | 209.6 |

fps rises 16→256 (depth 45→25) then **INVERTS**: the 32× FFN-intermediate blowup
(256→8192) swamps the 3× depth cut. **Honest best: 7.2 fps (bf16, radix 256).**

### 3b. PURE-DEPTH isolation (FFN intermediate held FIXED at 256)

Modelling the candidate scoring as a compact (attention-hosted / fixed-hidden) form
so ONLY the sequential-layer count changes with radix:

| radix | depth | fp32 fps | fp32 ms/step | bf16 fps | bf16 ms/step |
|---:|---:|---:|---:|---:|---:|
| 16 | 45 | 2.21 | 331.0 | 4.02 | 182.2 |
| 256 | 25 | 3.86 | 189.9 | 7.16 | 102.2 |
| 4096 | 15 | 6.30 | 116.2 | **11.86** | 61.7 |
| 65536 | 15 | 6.28 | 116.6 | 11.86 | 61.7 |

Now fps is MONOTONIC in 1/depth. **This is the crux measurement:**

> **fp32: depth 45 → 2.21 fps vs depth 15 → 6.30 fps = 2.85× fps for 3.0× less depth.**
> **bf16: depth 45 → 4.02 fps vs depth 15 → 11.86 fps = 2.95× fps for 3.0× less depth.**

The FLOP reduction converts to wall-clock ~proportionally to the depth cut once the
candidate table is held compact — confirming the sequential-depth cap was the wall
(radix 65536 and 4096 tie because the ADD-step floors at depth 15 for both).

## 4. The 30-fps verdict

**NO — no radix cleared 30 fps on this build.** Best measured:

- Honest radix-linear intermediate: **7.2 fps** (bf16, radix 256, depth 25) — 4.2× short.
- Pure-depth isolation: **11.9 fps** (bf16, radix 4096, depth 15) — 2.5× short.

The remaining 2.5× gap is NOT depth (that lever is now spent — depth 15 is the
ADD-step floor) and NOT batch (all rows saturate at batch 262,144). It is the
**per-layer full-transformer cost**: this stack pays a real ALiBi GQA-softmax + a
SwiGLU FFN per layer, whereas the `CLEVER_DOOM_REALTIME.md` 69-fps projection timed
the *bare* difference-min cell (a 10-wide abs+argmax, no attention softmax, no
SwiGLU). Reconciliation: the projection's ~69 fps is the arithmetic-core-only floor;
this doc's 12 fps is the same shallow depth wrapped in the honest transformer
layer that a real clever VM step must run. **Realtime needs the per-layer transformer
cost cut too** (e.g. collapse the framing softmax into a direct-CAM gather, or fuse
the 15 layers into one megakernel), not just fewer layers.

## 5. Byte-exactness (difference-min at larger radix)

Spot-checked ADD (limb ripple) + DIV (long-division trial) limb extraction on 4,000
random 32-bit operands per radix per dtype:

| radix | fp32 | bf16 | fp64 | why |
|---:|:--:|:--:|:--:|:--|
| 16 | ✅ | ✅ | ✅ | all accumulators ≤ ceiling |
| 256 | ✅ | ❌ | ✅ | bf16 DIV r²=65,536 > 2⁸ ceiling |
| 4096 | ✅ | ❌ | ✅ | fp32 DIV r²=16.7M = 2²⁴ boundary (holds); bf16 overflows |
| 65536 | ❌ | ❌ | ✅ | fp32 DIV r²=4.29e9 > 2²⁴ → **needs fp64** |

The difference-min digit extraction stays byte-EXACT at every radix *provided the
datapath dtype's exact-integer ceiling holds the op's accumulator* (the DIV r² bound
is binding). So the fast bf16 rows past radix 16 and the fp32 radix-65536 row are
**throughput proxies, not byte-exact** — an exact fast build must raise precision
where the radix demands it (bf16 only exact at radix 16; radix 65536 exact only in
fp64). This is the precision ceiling explicitly limiting how far the depth lever can
push under a given dtype.

## 6. Conclusion — does parallelism convert the reduction?

**Mechanistically YES; end-to-end NOT YET for realtime.**

- The depth cap that stalled the narrow clever VM at ~10 fps is real, and shrinking
  it via larger radix converts to fps **~1:1 with 1/depth** (measured 2.85–2.95× for
  3× less depth). Parallelism (batching) fills the card width-wise; radix removes the
  sequential-length that batching cannot shorten — the two are orthogonal and both
  needed.
- But it does NOT reach 30 fps because (a) the honest candidate-table growth inverts
  the win past radix 256 unless the scoring is kept compact, and (b) even compact,
  the shallow stack is still a full transformer whose per-layer softmax+SwiGLU cost
  floors it at ~12 fps at depth 15.
- The radix↔depth↔nonzero↔precision tradeoff is now fully quantified: **larger radix
  = fewer layers (faster, ~1:1) BUT a linearly-bigger candidate table (per-layer
  slower, can invert the win) AND a higher precision floor (DIV r² forces fp32 by
  radix 256 and fp64 by radix 65536, or byte-exactness breaks).** The realtime
  sweet spot on this hardware is radix 256–4096 in fp32 (byte-exact, shallowest
  without the fp64 penalty); pushing to 30 fps needs the per-layer transformer cost
  cut, not more radix.

### Reproduce

```
# depth + precision-ceiling + byte-exact tables (CPU, fast):
python examples/clever_shallow_radix_realtime.py --verify

# honest radix-linear intermediate fps sweep (idle GPU):
python examples/clever_shallow_radix_realtime.py --bench --device cuda:0 --json out.json

# pure-DEPTH isolation (fixed FFN intermediate) — the crux measurement:
python examples/clever_shallow_radix_realtime.py --bench --device cuda:0 --fixed-inter 256
```

Golden unchanged: this doc + `examples/clever_shallow_radix_realtime.py` touch **no
build files**; `python -m c4_min._fingerprint_build` = `174ece66` before and after.
```
```
