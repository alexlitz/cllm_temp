# Clever minimal-parameter ALU — precision × depth collapses the nibble param count

Hand-constructed (NOT trained), **digit-exact** transformer ALU cells for
**ADD/SUB**, **DIV/MOD**, and **MUL** on 32-bit unsigned integers, built by
composing two levers from the minimal-adder template
(`examples/minimal_10digit_adder.py`):

- **HIGH PRECISION** — hold the WHOLE operand values and whole intermediate
  results in ONE floating-point scalar each. **No nibble/bit decomposition.**
  Because a value is never split into 4-bit lanes, the model never stores the
  hundreds of per-nibble lookup-table / carry-corrector weights the production
  nibble-c4 VM carries.
- **CLEVER DEPTH** — extract the RESULT one decimal digit per layer, MSB-first,
  with the adder's difference-min selector `logit_d = -|value - (d + 0.5)|`
  (argmax over `d ∈ 0..9` picks `floor(value)`) plus a running-remainder update.
  The per-digit layer is a **single hand-set cell reused (recurrence)** once per
  output place, so unique params are constant while **depth = #output digits**.

Artifact: `examples/clever_minparam_alu.py` (model + census + verifier).
Run `python examples/clever_minparam_alu.py` (CPU only).

## The construction (identical template to the adder)

1. **Non-one-hot embedding**, `d_model = 4`: dim 0 = digit face value (0..9);
   dims 1/2/3 = flag bits for BOS / operator / `=`.
2. **ALiBi place-value ingest head**, slope `ln(10)`, read under **softmax1**
   (`e^{x}/(1+Σe^{x})`). One relative-position step scales the logit by `ln 10`,
   so `e^{logit}` is a power of ten — a descending place-value ladder. The
   softmax1 mean scaled back by its own denominator recovers the **whole operand
   value** as one exact scalar (`place_value_read`). Verified bit-exact on 50k
   random 10-digit operands (the `INGEST` check).
3. **Arithmetic core** (the only per-op difference):
   - **ADD/SUB**: `S = a + b` (or `a − b` with an unsigned wrap `+2^32`) — one
     whole fp64 value.
   - **DIV/MOD**: MSB-first long division *by digit-extraction*. Hold remainder
     `R` (= `a`) and divisor `b` whole; per place `p`: `q_p = floor(R/(b·10^p))`
     via the difference-min selector, then `R -= q_p·b·10^p`. Quotient = the
     assembled digits, MOD = final `R`.
   - **MUL**: hold the whole product `a·b` in one **fp128** value.
4. **Difference-min MSB-first digit decode** (`decode_digit`), the SAME reused
   cell for every op — one output digit per application, running remainder
   outside.

## Per-op results — nibble params vs clever params, precision, depth, reduction

Parameter schemes are the adder's four (measured by `census_for`):
**(a)** all non-zero dense entries · **(b)** minus the identity Q/K/V/O matrices
· **(c)** reuse the embedding value axis 0..9 as decode candidates · **(d)**
scalars only. The construction is depth-in-time (one reused cell), so the
**unique** param set is identical across ops; depth is reported separately.

| op | nibble-c4 (excl.) | (a) all | (b) −id | (c) reuse | **(d) scalars** | depth | precision |
|----|------------------:|--------:|--------:|----------:|----------------:|------:|:---------:|
| **ADD/SUB** | 6,459 | 42 | 26 | 16 | **4** | 11 | fp64 |
| **DIV/MOD** | 163,591 | 42 | 26 | 16 | **4** | 10 | fp64 |
| **MUL** | 7,911 | 42 | 26 | 16 | **4** | 20 | fp128 |

Reduction vs the nibble-c4 exclusive params (per scheme):

| op | vs (a)=42 | vs (c)=16 | **vs (d) scalars=4** |
|----|----------:|----------:|---------------------:|
| **ADD/SUB** (6,459) | 154× | 404× | **1,615×** |
| **DIV/MOD** (163,591) | **3,895×** | 10,224× | **40,898×** |
| **MUL** (7,911) | 188× | 494× | **1,978×** |

The four scalars in scheme (d) are the irreducible core, shared by every op:
**`ln(10)`** (ALiBi place-value slope), **`+1`** (softmax1 off-by-one that turns
the pooled mean into a sum), **`+0.5`** (floor-realizing half-shift), **`10.0`**
(place base).

### Verification (hand-set, no training)

`python examples/clever_minparam_alu.py --n 250000` → **ALL EXACT**, 800,050
operand pairs total (≥100k random per op + hard cases: `b=1`, `a<b`, powers of
two, `max·max`, full-cascade adds, borrow-wrap subs). Every output digit matches
by argmax.

## The precision × depth tradeoff surface

For each op, the minimal params achievable and the (precision, depth) it costs:

- **ADD/SUB — fp64, depth 11.** `a+b ≤ 2·(2^32−1) ≈ 8.59e9 < 2^53 ≈ 9.0e15`, so
  the whole sum is an fp64-exact integer. fp64 SUFFICES. Depth = 11 output
  digits. **6,459 → 4 scalars (1,615×).**

- **DIV/MOD — fp64, depth 10.** The headline. Long division by digit-extraction
  keeps every value ≤ `a < 2^32`. The only worry is the partial `q_p·b·10^p`,
  where `b·10^p` can exceed 2^53 for large `p` — but that only happens when
  `b·10^p > R`, i.e. the digit `q_p` is **0**, so the un-representable large
  product is **never subtracted**. Every USED partial satisfies
  `q_p·b·10^p ≤ 9·b·10^p ≤ 9·a < 2^53` (measured max ≈ 4.29e9), so fp64 is
  **exact**. fp64 SUFFICES — no fp128, despite division being the most
  param-heavy nibble op. **163,591 → 4 scalars (40,898×).** This is where holding
  whole values + recurrence pays off most: the nibble VM's ~164k div params are
  4-bit-lane quotient/remainder lookup tables and per-nibble restoring-division
  correctors; the whole-value long division needs none of them.

- **MUL — fp128, depth 20.** Here precision MUST go up. The 64-bit product
  `a·b ≤ (2^32−1)^2 ≈ 1.84e19 > 2^53`, so fp64 mis-floors a 64-bit product (the
  verifier's negative control shows fp64 gets 2/4 big products WRONG). x86-64
  `numpy.longdouble` (80-bit extended, 63-bit stored mantissa = 64 effective)
  holds every integer `< 2^64` exactly, so the whole product and every running
  remainder are exact. **fp128 REQUIRED.** Depth = 20 output digits (a 64-bit
  product is up to 20 decimal digits). **7,911 → 4 scalars (1,978×).**

**Platform note:** fp128 here is x86-64 `numpy.longdouble` / `numpy.float128`
(80-bit extended precision, 64-bit effective mantissa). On platforms where
`longdouble == float64` (e.g. some ARM/MSVC) the MUL construction would need a
true 128-bit type or a two-limb split; ADD/SUB and DIV/MOD stay fp64 everywhere.

### Reading the surface

| lever paid | ADD/SUB | DIV/MOD | MUL |
|------------|:-------:|:-------:|:---:|
| precision  | fp64    | fp64    | **fp128** |
| depth      | 11      | 10      | **20** |
| params (d) | 4       | 4       | 4   |

Params collapse to the same 4 scalars for all three ops; the cost of the
collapse is paid entirely in **precision** (fp64 suffices for add/div; mul needs
fp128) and **depth** (one reused layer per output digit: 10–11 for add/div, 20
for the wider mul product). The single largest win is **DIV/MOD: 163,591 → 4
(≈40,898×)** at fp64 with depth 10 — no precision cost at all, purely
whole-value long division + a reused digit-extraction cell.

## Contrast with the nibble-c4 VM

The nibble-c4 VM is the **opposite** strategy: it decomposes every value into
4-bit nibbles so each lane stays tiny and exact in fp32, at the cost of a wide
per-nibble lookup/corrector apparatus (6,459 ADD/SUB, 7,911 MUL, **163,591**
DIV/MOD exclusive params). This construction keeps the **whole value in one
high-precision scalar** and reads the result out one digit at a time — trading a
higher per-scalar precision (fp64, or fp128 for mul) and serial depth for a
constant handful of scalar parameters. Same "exact integer arithmetic in a
transformer" goal; inverted precision/params trade.
