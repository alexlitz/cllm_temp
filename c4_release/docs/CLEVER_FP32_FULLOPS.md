# A MIN-PARAM fp32 clever VM covers the FULL Doom op set byte-exact — NO fp64/fp128, ~7x edge holds

**Date:** 2026-08-09 · **Scope:** re-examine the prior precision NO-GO that assumed the
clever VM's **whole-value MUL** (32×32→64-bit product) needs **fp128** (and DIV at radix
65536 needs fp64). This builds a fp32-safe **8-bit-limb** MUL and re-measures whether the
whole Doom op set is fp32-exact with **no fp64/fp128 anywhere**, and whether the fp32
per-lane-step stays ~0.11 µs so the clever VM keeps its ~7× throughput edge over the wide
VM (0.788 µs/step).

Code: [`examples/clever_fp32_fullops.py`](../examples/clever_fp32_fullops.py). All GPU
numbers MEASURED on idle RTX A5000s (device 0; 2-GPU = a REAL concurrent two-device run,
both cards free), batched to saturation (batch 262,144). Golden `174ece66` untouched (no
build file — verified before and after).

---

## TL;DR verdict

**The precision NO-GO is OVERTURNED.** A min-param **fp32** clever VM is byte-exact for the
**entire Doom op set with NO fp64/fp128 anywhere**, and it keeps a **7.16× per-lane-step
edge** over the wide VM (measured).

- **fp32 byte-exact, ALL ops:** full 64-bit MUL, Doom FixedMul, Doom FixedDiv,
  ADD/SUB/DIV/MOD/CMP/SHL/SHR/LEA/bitwise/memory — **all PASS in fp32**, none needs > fp32.
- The MUL "needs fp128" claim was about **stuffing the 64-bit product into one fp32
  scalar** (2^64 ≫ 2^24). The **8-bit-limb decomposition** never forms that scalar: every
  partial product ≤ 65,025 and the worst column accumulator is **260,864 < 2^24 = 16.7M**
  (a ~64× margin), so the full 64-bit product is exact in fp32.
- **fp32 per-lane-step = 0.1100 µs** (Doom-mix weighted), **= 7.16× the wide VM's 0.788 µs**.
  The limb-MUL's extra depth (20 vs 15) costs only when MUL runs — and MUL is 1% of the mix,
  so the average is essentially unchanged from the ADD-class 0.1097 µs. **The ~7× edge
  survives.**
- 2-GPU (real, concurrent): **0.0548 µs/lane-step = 14.37× the wide VM.**

---

## 1. Why the NO-GO was about representation, not Doom's data

Doom is **100% integer + 16.16 fixed-point** (`fixed_t` is a 32-bit int, `FRACUNIT = 1<<16`)
— there are **no floats in Doom's data**. The "fp64/fp128" in the prior analysis is the
**clever VM's datapath dtype**: the clever cell holds a whole value in one float scalar and
extracts digits by a difference-min decode. The binding constraint is that every
intermediate accumulator must stay ≤ the dtype's **exact-integer ceiling** (fp32 = 2^24 =
16,777,216).

The measured per-op accumulator maxima (`opconfig.acc_max`) at each radix vs the fp32
ceiling:

| radix | MUL col-peak | fits fp32? | DIV r² | fits fp32? |
|---:|---:|:--:|---:|:--:|
| 16 | 1,904 | ✅ | 256 | ✅ |
| **256 (8-bit limbs)** | **260,864** | **✅ (64× margin)** | 65,536 | ✅ |
| 4096 | 50,315,264 | ❌ (needs > fp32) | 16,777,216 | ✅ (exact 2^24 boundary) |
| 65536 | 8,589,737,984 | ❌ | 4,294,967,296 | ❌ (needs fp64) |

So the fp128 claim is real **only for the near-whole-value MUL** (radix 4096: col-peak
50.3M > 2^24). **The 8-bit-limb (radix-256) MUL is the fix**: col-peak 260,864 fits fp32
outright. DIV at radix 4096 sits exactly at 2^24 → fp32-exact. The rest of the op set is
ADD-class (2r accumulator) → fp32-exact at radix 4096.

## 2. The fp32-safe limb MUL (and FixedMul / FixedDiv)

**Full 64-bit MUL.** Split each 32-bit operand into 4 base-256 limbs. Every partial product
`a_i·b_j ≤ 255·255 = 65,025`; the 16 partials land in 8 output-byte columns; the worst
column sum (with carry) ≤ 260,864 < 2^24. Carry-resolve each column mod 256 with the fp32
floor decode. **No 2^64 scalar ever exists** — the product is the 8 exact byte-limbs.

**FixedMul** = low-32 of `((int64)a·b) >> 16`: magnitude-multiply `|a|·|b|` through the limb
schoolbook (exact), form the two's-complement signed 64-bit product, arithmetic `>>16` (a
byte-slice of the 8 exact limbs), re-apply the sign — matching
`c4_min/doom_fixedpoint.py::fixed_mul` exactly.

**FixedDiv** = the overflow-guard wrapper around the 48-bit `FixedDiv2` long division. The
remainder/divisor/quotient reach ~2^31 (≫ 2^24) — the same precision cliff. **The limb fix
carries each 32-bit quantity as two 16-bit halves** (hi, lo), each < 2^16 ≪ 2^24, so
shift-left-1, unsigned compare-GE, and subtract-with-borrow are all exact in fp32. Nothing on
the datapath exceeds 2^17. Byte-exact vs `doom_fixedpoint.py::fixed_div` (the on-VM-exact
reference the byte-exact title frame depends on).

## 3. Per-op fp32 byte-exact pass rate (MEASURED, `--verify`)

Oracle for MUL = the exact python-int 64-bit product; for FixedMul/FixedDiv =
`c4_min.doom_fixedpoint.fixed_mul/fixed_div` over the doom battery (LCG operands + negatives
/ sign-magnitude + FRACUNIT boundaries + the overflow-guard regime); the rest reuse the
`clever_compact_scoring_realtime` ADD-ripple / DIV-long-division spot-check.

| op | form | fp32 byte-exact | fp128 used? |
|---|---|:--:|:--:|
| **MUL full 64-bit (32×32→64)** | 8-bit limbs, 20,000 random ops | **PASS** | **NO** |
| **FixedMul** ((a·b)>>16, signed) | limb product + sign, doom 16.16 battery (1,618) | **PASS** | **NO** |
| **FixedDiv** (48-bit long div + guard) | 16-bit-half datapath, doom 16.16 battery | **PASS** | **NO** |
| ADD / SUB / LEA / frame (ADD-class) | radix 4096 ripple | **PASS** | **NO** |
| CMP / SHL / SHR / bitwise / memory (ADD-class) | radix 4096 | **PASS** | **NO** |
| DIV / MOD | radix 4096 long division (r²=2^24 boundary) | **PASS** | **NO** |

**Every op is fp32-exact. NONE needs > fp32.** An additional 50,000 random full-range
FixedMul + FixedDiv cases (including negatives and the overflow regime) also pass 0 failures.

The negative control: MUL at radix 4096 as a near-whole-value col-peak (50.3M) does **not**
fit fp32 — that is the fp128-forcing form, and it is exactly what the 8-bit-limb MUL avoids.

## 4. fp32 full-op-set per-lane-step cost (MEASURED, A5000, batch 262,144)

Per-op-class ns/lane-step (fp32, whole clever STEP = T=1 direct framing attention + compact
SwiGLU FFN at the op's depth), and the Doom-mix-weighted average (MUL 1% / DIV 0.5% /
98.5% ADD-class, from `serial_doom_floor.DOOM_MIX`):

| op class | depth | µs/lane-step | vs wide VM (0.788 µs) |
|---|---:|---:|---:|
| ADD-class (framing / pointer-walk) | 15 | **0.1097** | **7.18×** |
| DIV (radix-4096 long div) | 15 | 0.1098 | 7.18× |
| **MUL (8-bit-limb schoolbook)** | **20** | **0.1457** | 5.41× |
| **DOOM-mix weighted** | — | **0.1100** | **7.16×** |

**The limb-MUL's extra depth (20 vs 15, +33%) raises the MUL step to 0.1457 µs (5.41× the
wide VM), but MUL is only 1% of the Doom mix**, so the weighted average is **0.1100 µs** —
essentially the ADD-class 0.1097 µs. Measured stable across runs (0.1097/0.1097 µs ADD).

- **fp32 per-lane-step = 0.1100 µs = 7.16× the wide VM's 0.788 µs → the ~7× edge survives.**
- **2-GPU (real concurrent): 0.0548 µs/lane-step = 14.37× the wide VM.**

This matches the prior compact-scoring ~0.11 µs measurement — the limb-MUL does **not** move
the average because it is depth-heavy on a rare op.

## 5. Corrected precision verdict

**Is a min-param fp32 clever VM byte-exact for ALL Doom ops with NO fp64/fp128, and does it
keep a per-lane-step advantage over the wide VM?** — **YES on both.**

- **Byte-exact, all Doom ops, fp32 only:** full 64-bit MUL, FixedMul, FixedDiv,
  ADD/SUB/DIV/MOD/CMP/SHL/SHR/LEA/bitwise/memory. **No op needs fp64 or fp128.** The
  fp128 requirement was an artifact of the whole-value (one-scalar-per-64-bit-product)
  representation; the **8-bit-limb** MUL (partials ≤ 65,025, column peak 260,864 < 2^24) and
  the **16-bit-half** FixedDiv datapath keep every fp32 intermediate below 2^24.
- **Per-lane-step advantage:** **0.1100 µs fp32 (Doom-mix weighted) = 7.16× the wide VM**
  (14.37× on 2 GPUs). The limb-MUL's depth is amortized to nothing by MUL's 1% mix share.

**The precision NO-GO ("whole-value MUL needs fp128") is overturned.** What remains open is
NOT precision but the runtime-build + single-stream-throughput-harness questions (this
measures a batched-to-saturation cell set, not the full built VM through a single-stream
autoregressive harness).

### Measured vs projected

- **MEASURED:** all §3 byte-exact pass rates (CPU, exact-int oracles); all §4 per-lane-step
  numbers and the 2-GPU run (idle A5000s, real concurrent two-device, batch 262,144).
- **PROJECTED:** nothing in §3–§4. The op-mix weighting uses the documented
  `serial_doom_floor.DOOM_MIX` fractions (an ESTIMATE grounded in the measured 63%
  pointer-walk fact); the per-op ns are measured, only their weighted blend uses the mix.

### Reproduce

```
# byte-exact fp32 full op set (CPU, fast), no fp64/fp128:
python examples/clever_fp32_fullops.py --verify

# fp32 per-lane-step + REAL 2-GPU (both cards free):
python examples/clever_fp32_fullops.py --bench --two-gpu --device cuda:0 --json out.json
```

Golden unchanged: this doc + `examples/clever_fp32_fullops.py` touch **no build files**;
`CUDA_VISIBLE_DEVICES="" python -m c4_min._fingerprint_build` = `174ece66` before and after.
