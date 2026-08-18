# Deep-serial minimal-param ALU cells + the Doom FLOP floor

**Date:** 2026-08-07 · **Scope:** analysis + standalone verified code, CPU-only.
**Touches no build files** — c4 golden (`174ece66` / fingerprint `7d4afe61`) is unchanged.
Code: [`examples/tiny_serial_alu.py`](../examples/tiny_serial_alu.py) (the exact ALU
cells) and [`examples/serial_doom_floor.py`](../examples/serial_doom_floor.py) (the
floor arithmetic).

---

## The question

The production doom-transformer costs **~5.68 MFLOP per VM step** (MEASURED, capstone
§6). Almost all of that is **width overhead**: `d_model ≈ 1440–1656`, so every gate
multiplies against a ~1440-wide residual even when the *useful* arithmetic is 1–100
FLOP. A gate that needs a handful of MACs still pays ~1440.

A **narrow + deep-serial** construction removes the width: keep `d_model` tiny, peel
one bit (or nibble) per layer, and share one hand-built "cell" across all
bit-positions (weight-tying). FLOP-per-op then collapses toward the op's true
gate-count — at the cost of **depth** (many strictly-sequential layer-applications).
This is a *FLOP* reduction (unlike pure weight-tying, which is params-only). Part 1
builds and VERIFIES-EXACT three such ALU cells; Part 2 estimates how low the Doom
frame FLOP goes if **all** c4 ops are built this way.

---

## Part 1 — three tiny serial ALU ops, hand-constructed, verified EXACT

Each op is a stack of tiny layers; the per-layer weight matrix is a fixed shared
cell. FLOP/op = `depth × (nonzero-MACs per layer) × 2` (MAC → 2 FLOP; count only
nonzero weights, matching the c4 `_flop_gauge` COO convention). No training —
weights are hand-set. Verified on **≥100 k random 32-bit pairs each** (re-run at
200 k, different seed: still exact; 600 k pairs total).

| op | nonzero params (shared cell) | d_model | depth (layer-apps) | precision | MAC/layer | **FLOP/op** | verified exact |
|---|---:|---:|---:|:--|---:|---:|:--:|
| **ADD** (32-bit) | 5 | 3 | 32 | fp32 | 5 | **320** | ✅ 200 016 pairs |
| **MUL** (32×32→64) | 5 | 16 | 32 | fp32 | 80 | **5 120** | ✅ 200 020 pairs |
| **DIV/MOD** (32-bit) | 8 | 18 | 32 | fp32 | 47 | **3 008** | ✅ 200 006 pairs |

Contrast — the **parallel fp64 companion adder** (the wide style): ~12 params, ~2
"layers", fp64, one shot, no depth — but it does *not* generalize past 2^53 (which is
exactly why a parallel 64-bit product/quotient needs fp128 or limbs; see below).

### How each cell works (and why it stays exact in fp32)

- **ADD — bit-serial full adder.** One layer per bit, LSB→MSB, carrying the single
  carry bit. `s3 = a_i + b_i + c ∈ {0,1,2,3}` (3-MAC accumulate); `carry = (s3 ≥ 2)`
  (1 threshold); `sum = s3 − 2·carry` (1 MAC). Single bits and a single carry are
  representable in **any** float → **fp32 exact**. Verified incl. full carry
  cascades (`0xFFFFFFFF + 1`, `0x80000000`, alternating-bit patterns). Covers
  SUB/AND/OR/XOR/SHL/SHR at ≤ ADD cost.

- **MUL — bit-serial shift-add, nibble-limb accumulator.**
  `product = Σ_{i: b_i=1} (a << i)`, 32 layers (one per bit of `b`). The 64-bit
  accumulator exceeds fp64's `2^53` exact range, so instead of fp128 it is carried
  as **16 base-16 nibble limbs** (each 0..15). A conditional limbed add (gate by
  `b_i`, ripple carry across 16 limbs) keeps every limb-sum ≤ ~31 → **fp32 exact**,
  full 64-bit product, no fp128. Verified exact on 200 020 pairs.
  - *fp128 parallel variant (also implemented + verified):* a single-shot
    `a × b` in numpy **`float128`** (80-bit, 64-bit mantissa — `2^64−1` is
    representable). Depth 1, but **fp128-required**. Confirmed byte-exact on the
    same inputs. This is the "parallel/short construction that overflows fp64" the
    task allows fp128 for; the limbed bit-serial form is cleaner and cheaper, so it
    is the headline MUL.

- **DIV/MOD — bit-serial restoring long division.** MSB-first, 32 iterations:
  `R = (R<<1) | a_i; if R ≥ D: R −= D, q_i = 1`. The 33-bit remainder is carried as
  9 nibble limbs; the compare and conditional subtract are small limbed ops →
  **fp32 exact** `(q, r)`. Verified on 200 006 pairs including **divisor = 1,
  powers of two, and a < b** (quotient 0), plus `0xFFFFFFFF / …` edges.

**Verification command:** `python examples/tiny_serial_alu.py --n 200000 --seed 7`
→ `RESULT: ALL EXACT (600,042 operand pairs total)`.

---

## Part 2 — the Doom FLOP floor under ALL-ops-deep-serial

### The narrow-serial VM-step model (grounded in Part 1)

A c4 VM step = **fetch** opcode + **decode** + **execute-one-op** + **writeback**
(PC/SP/BP). In the narrow-serial style every 32-bit register is processed
bit/nibble-serially:

- **framing** (fetch + decode + PC/SP/BP writeback) ≈ 4 ADD-class serial ops
  (a byte/address compare, a PC increment, an SP bump) + a ~40-way opcode-select
  one-hot ≈ **1 580 FLOP/step**.
- **execute**: pointer/stack/branch/compare ops (PSH/LEA/LI/JMP/BZ/CMP…) reduce to
  an address-ADD + a byte move ≈ **1 ADD (320 FLOP)**; ALU ops use their measured
  serial cell (ADD 320, MUL 5 120, DIV/MOD 3 008).

**Doom opcode mix** (ESTIMATE, anchored to the MEASURED fact that *63 % of the render
is PSH/LEA/LI pointer-walking* — capstone §6 / `PERF_LADDER_FINAL.md` §5; heavy
MUL/DIV are rare per-step because Doom's inner render divides fold into the render
superinstruction):

| bracket | FLOP/step |
|---|---:|
| PSH-only lower bound (every step the lightest pointer op) | 1 900 |
| ADD-class step | 1 900 |
| **Doom-mix weighted (63 % pointer-walk)** | **≈ 1 952** |
| a DIV step (rare) | 4 588 |

So the deep-serial c4 **VM-step floor ≈ 1 952 FLOP**, vs the current **5.68 MFLOP** —
a **~2 910× per-step reduction**. (Framing dominates the floor: pure *useful* c4 work
is ~1–100 FLOP, so even the serial floor is ~39× useful, versus ~113 600× useful
today. The remaining ~39× is the irreducible fetch/decode/writeback framing of a
byte-exact stack VM.)

### Doom frame FLOP — current vs deep-serial floor

Step counts are MEASURED: raw title-redraw frame **6 889 264** steps; render-reduced
steady frame **358 058** steps. Current = 5.68 MFLOP/step (verified: 6.89 M × 5.68 M
= 39.13 TFLOP; 358 058 × 5.68 M = 2.03 TFLOP).

| frame | steps | FLOP/step | total FLOP |
|---|---:|---:|---:|
| **RAW — current (wide)** | 6 889 264 | 5 680 000 | **39.13 TFLOP** |
| **RAW — deep-serial floor** | 6 889 264 | 1 952 | **13.45 GFLOP** |
| **RENDER — current (wide)** | 358 058 | 5 680 000 | **2.03 TFLOP** |
| **RENDER — deep-serial floor** | 358 058 | 1 952 | **698.9 MFLOP** |

**Reduction factor: ~2 910× at every level** (per-step, raw frame, render frame).

- RAW: **39.13 TFLOP → ~13.5 GFLOP**
- RENDER-reduced: **2.03 TFLOP → ~699 MFLOP**

Where the reduction comes from (honest decomposition of the 5.68 MFLOP current step:
~72 % FFN / ~28 % attention): the deep-serial form removes **both** (a) the ~1440-wide
per-gate width overhead (~144–480× per gate), and (b) the growing per-step re-embed
attention — a narrow serial VM keeps its whole state in a fixed 32-bit register file,
so there is no O(S²) score tensor and no growing KV. Both collapse to the op's real
gate-count.

### The cost: DEPTH explodes (occupancy is a SEPARATE question)

The FLOP win is real but it is bought with **strictly-sequential depth**:

| | current WIDE build | deep-serial floor |
|---|---|---|
| **params** (per op / shared cell) | ~thousands per block | **5–8** (shared, tied) |
| **FLOP / step** | 5.68 M | **~1 952** (~2 910× ↓) |
| **layer-apps / step** | ~264 blocks, each ~1440-wide (parallel *within* a block) | **~160**, each 3–18 wide, **strictly sequential** (bit *i* needs bit *i−1*'s carry) |

- Each serial layer is 3–18 wide and depends on the previous one's carry, so it
  **cannot be width-parallelized** onto a GPU's MMA units. The current wide build,
  though it costs 2 910× more FLOP, feeds those FLOP as parallel `[K, 1440]` GEMMs
  the hardware is built for.
- **Wall-clock ≠ FLOP.** The current build is already **occupancy-bound** (~11 % of
  FLOP peak — capstone §6): it is starved for parallel work, not for FLOP headroom.
  A 3–18-wide, ~160-deep dependency chain starves it *worse*. So the deep-serial
  floor is a **compute-lower-bound** statement, not a speed promise — on a GPU it may
  well be **slower** in wall-clock despite ~2 910× fewer FLOP.

The min-construction triple: **params ↓ (→ 5–8, tied)**, **FLOP ↓ (~2 910×)**,
**depth ↑ (strictly-sequential ~160 layer-apps/step)**.

---

## Headline

> With **all** c4 ops deep-serial + minimal-nonzero-param, a **byte-exact** Doom
> frame's arithmetic FLOP floor is **~13.5 GFLOP raw** / **~699 MFLOP
> render-reduced** — about **2 910× below** today's ~39 TFLOP raw / ~2 TFLOP render.
> The ~1440× per-gate **width overhead vanishes** (and with it the O(S²) attention);
> the price is **depth** — ~160 strictly-sequential layer-applications per step that
> a GPU cannot parallelize, so the FLOP floor is a lower bound, **not** a speed
> guarantee.

### Reproduce

```
python examples/tiny_serial_alu.py --n 200000 --seed 7   # verify the 3 exact cells
python examples/serial_doom_floor.py                     # the floor arithmetic
```
All numbers labelled MEASURED (from `_flop_gauge.py` / the capstone) or ESTIMATE
(the opcode mix, anchored to the measured 63 % pointer-walk). CPU-only; no build
files touched; c4 golden unchanged.
