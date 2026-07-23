# Constant-multiplier fast path — `a·b` with `b` a compile-time constant

`const_mul.py` builds `(a·b) & 0xFFFFFFFF` when **`b` is known at build time**, as a
standalone byte-exact bakeoff (mirroring the constant-divisor / general-multiply
bakeoff pattern). It compiles **two constructions**, measures depth / weights /
fp32-safety / byte-exactness, and reports how each construction shrinks vs the
general carry-share multiply.

**Baseline to beat** — the general carry-share multiply
(`nibble_alu32.compile_mul_blocks` + the shared carry round): **10 blocks,
11 430 nz**, on a *runtime* 32-bit `b`.

Everything is **composed** from the proven `nibble_alu32` primitives (`_ident`,
`_clear`, `_floor_div_pow`/`_floor_div_pow2`, `_byte_add_block`, `_truncate`,
`compile_mul_blocks`, `extend_layout_for_alu32`); this file only **reads /
imports** the shared files, never edits them. Run:

```
python -m c4_min.const_mul
```

## The two constructions

### A. Linear partial-product multiply (NO gated multiply)

`a·b = Σᵢ aᵢ·(b·16ⁱ) = Σᵢ Σⱼ aᵢ·Bⱼ·16^(i+j)`, where `aᵢ` are the **runtime**
input nibbles (0..15) and `Bⱼ` are the **baked** constant nibbles of `b`. Because
`b` is a constant, every partial `aᵢ·Bⱼ` is `runtime-nibble × constant-nibble` —
a **weighted copy** (a single `_ident` read of the `aᵢ` band scaled by the
*constant weight* `Bⱼ`), **not** a `_mul_gate`. No operand gather for `b`, no
gated products. Each raw partial `aᵢ·Bⱼ ≤ 15·15 = 225` is split low/high (one
shared `_floor_div_pow2` staircase, kmax=15) into product columns `c = i+j`
(kept `< 256`), then carry-resolved with the shared carry round.

Partials with `Bⱼ = 0` (or `aᵢ` statically zero) are **dropped** — the
constant-specific saving.

### B. Strength reduction (shift-add) via NAF

Decompose `b` into a minimal shift-add chain by its **non-adjacent form** (NAF /
canonical signed digit): `b = Σₖ dₖ·2ᵏ`, `dₖ ∈ {−1,0,+1}`, no two adjacent
nonzero digits → the fewest nonzero digits of any signed-binary form. Each nonzero
digit is a term `± (a << k)` = a **shift** (nibble/bit relayout) + an **add/sub**.
The terms fold left-to-right through the `nibble_alu32` 4-byte add-chain
(`_byte_add_block`), highest-shift first (the leading NAF digit of a positive `b`
is always `+1`, so the accumulator starts positive and every intermediate stays
in-range — two's-complement subtract handles the modular wrap regardless).

`×3 = (a<<2)−a` (1 sub), `×10 = (a<<3)+(a<<1)` (1 add), `×255 = (a<<8)−a` (1 sub).
For low-NAF-weight `b` this is 1–3 adds.

## Narrowing

* **Result width** — emit only `⌈log₁₆(b·2^in_width)⌉` result nibbles; provably-zero
  high nibbles are dropped (masked out by `& 0xFFFFFFFF`). Result nibbles fall from
  8 to as few as 3–4 when the input is bounded.
* **Input width** — if `a` is statically bounded to `in_width` bits, only its
  `⌈in_width/4⌉` low nibbles are nonzero → construction A emits fewer partials and a
  shorter carry-resolve. (Verified byte-exact over `a < 2^in_width`; table below.)

## Special cases

`b = 0` → result 0 (one clear block). `b = 1` → identity copy. **Powers of two**
→ pure nibble/bit **shift** (relayout): `b = 16ᵏ` is a 1-block positional copy
(zero staircase units), `b = 2ᵏ` with a sub-nibble bit shift is a 4-block relayout.

## Results — bakeoff table (byte-exact = 2023/2023 per (b, construction), fp32)

General carry-share multiply baseline = **depth 10 / nz 11 430**.

| b | A depth/nz | B depth/nz | pick | resnib | best vs gen | regime |
|---:|---:|---:|:--:|--:|--:|:--|
| 0 | 1 / 24 | 1 / 24 | A | 1 | 0.21 % | special/shift (pure relayout) |
| 1 | 1 / 48 | 1 / 48 | A | 8 | 0.42 % | special/shift (pure relayout) |
| 2 | 4 / 2052 | 4 / 2052 | A | 8 | 17.95 % | special/shift (bit shift) |
| 3 | 12 / 9876 | 14 / **5565** | **B** | 8 | 48.69 % | shift-add wins |
| 5 | 12 / 9876 | 13 / **5541** | **B** | 8 | 48.48 % | shift-add wins |
| 7 | 12 / 9876 | 14 / **5565** | **B** | 8 | 48.69 % | shift-add wins |
| 10 | 12 / 9876 | 16 / **7545** | **B** | 8 | 66.01 % | shift-add wins |
| 11 | 12 / 9876 | 22 / **8291** | **B** | 8 | 72.54 % | shift-add wins |
| 13 | 12 / 9876 | 21 / **8267** | **B** | 8 | 72.33 % | shift-add wins |
| 16 | 1 / 45 | 1 / 45 | A | 8 | 0.39 % | special/shift (nibble shift) |
| 17 | 12 / 10749 | 10 / **3534** | **B** | 8 | 30.92 % | shift-add wins |
| 97 | 12 / 10749 | 24 / **10028** | **B** | 8 | 87.73 % | shift-add wins |
| 100 | 12 / **10749** | 27 / 12032 | **A** | 8 | 94.04 % | linear-fold wins |
| 251 | 12 / 10749 | 22 / **8288** | **B** | 8 | 72.51 % | shift-add wins |
| 255 | 12 / 10749 | 11 / **3555** | **B** | 8 | 31.10 % | shift-add wins |
| 256 | 1 / 42 | 1 / 42 | A | 8 | 0.37 % | special/shift (nibble shift) |
| 1000 | 12 / **11493** | 27 / 11909 | **A** | 8 | 100.55 % | linear-fold wins |
| 65535 | 12 / 12108 | 11 / **3549** | **B** | 8 | 31.05 % | shift-add wins |
| 65537 | 12 / 10362 | 10 / **3525** | **B** | 8 | 30.84 % | shift-add wins |
| 4, 8 | 4 / 2052 | 4 / 2052 | A | 8 | 17.95 % | special/shift (bit shift) |
| 32 | 4 / 1929 | 4 / 1929 | A | 8 | 16.88 % | special/shift (bit shift) |
| 1024 | 4 / 1806 | 4 / 1806 | A | 8 | 15.80 % | special/shift (bit shift) |
| 65536 | 1 / 36 | 1 / 36 | A | 8 | 0.31 % | special/shift (nibble shift) |
| 2²⁰ | 1 / 33 | 1 / 33 | A | 8 | 0.29 % | special/shift (nibble shift) |
| 10⁶ | 12 / **11592** | 41 / 16923 | **A** | 8 | 101.42 % | linear-fold wins |
| 2²⁴ | 1 / 30 | 1 / 30 | A | 8 | 0.26 % | special/shift (nibble shift) |
| 2³²−1 | 12 / 13278 | 11 / **3537** | **B** | 8 | 30.94 % | shift-add wins |

`best vs gen` = the auto-picked construction's nz as a fraction of the general
multiply's 11 430 nz. `pick` uses the smaller build by **nz** (depth tiebreak).
Byte-exact = 2023/2023 for **every** (b, construction) over 2000 random full-32-bit
`a` + 23 structured edges; totals to **113 288 / 113 288**. A heavier 6000-operand
sweep on off-battery `b` (`12345, 999983, 0xABCDEF, 2³¹, 0xDEADBEEF, …`) also passes
**6006/6006** in both constructions.

## Input-width narrowing (construction A)

Bounding `a` to `in_width` bits shrinks A directly — fewer partials, fewer result
nibbles, shorter carry-resolve. Verified byte-exact over `a < 2^in_width`.

**b = 1000:**

| in_width | depth | nz | partials | resnib |
|---:|---:|---:|---:|---:|
| 4 | 7 | 2343 | 3 | 4 |
| 8 | 8 | 3837 | 6 | 5 |
| 12 | 9 | 5583 | 9 | 6 |
| 16 | 10 | 7581 | 12 | 7 |
| 24 | 12 | 11166 | 18 | 8 |
| 32 | 12 | 11493 | 21 | 8 |

**b = 255:**

| in_width | depth | nz | partials | resnib |
|---:|---:|---:|---:|---:|
| 4 | 6 | 1359 | 2 | 3 |
| 8 | 7 | 2472 | 4 | 4 |
| 12 | 8 | 3837 | 6 | 5 |
| 16 | 9 | 5454 | 8 | 6 |
| 24 | 11 | 9444 | 12 | 8 |
| 32 | 12 | 10749 | 15 | 8 |

A 4-bit-bounded `a·1000` costs **7 blocks / 2343 nz** — 20 % of the general
multiply — vs 12 / 11 493 for the unbounded case.

## fp32 discipline

**Fully fp32, zero fp64.** Every relu staircase argument is bounded:
a weighted-copy partial `aᵢ·Bⱼ ≤ 225`; a product column is kept `< 256`; the byte
add-chain sums `≤ 511`. The largest argument anywhere is **511** (the add-chain byte
sum), so `RELU_S·arg = 200·511 = 102 200 < 2²⁴ = 16 777 216` with a 164× margin.
No hidden unit ever exceeds fp32's integer-exactness ceiling — this is inherited
from `nibble_alu32`, which is the whole reason both constructions decompose to
nibble/byte granularity.

## Verdict — per-regime winner

* **Powers of two / `b ∈ {0,1}`** — a **pure shift (relayout)**, `0.2 %–18 %` of the
  general multiply's weights (1–4 blocks). Both constructions collapse to the same
  shift blocks. Overwhelming win; never use the general multiply for these.

* **Small / low-NAF-weight `b`** (3, 5, 7, 10, 17, 255, 65535, 65537, 2³²−1, …) —
  **strength reduction (construction B) wins**, `31 %–72 %` of the general
  multiply. The win tracks the NAF Hamming weight: weight-2 constants
  (`255 = 2⁸−1`, `65537 = 2¹⁶+1`) hit `~31 %` (one shift + one add-chain), while
  weight-4/5 constants (`10, 11, 13`) climb toward `66–72 %`. B is a few blocks
  **deeper** than A (each add is a 4-byte sequential chain), so if latency, not
  weight, is the budget, prefer A.

* **Dense / high-NAF-weight `b`** (100, 1000, 97, 10⁶, …) — **linear-fold
  (construction A) wins**, and here it is roughly **at parity** with the general
  multiply (`94 %–101 %`). A's weighted-copy partials are cheaper *per partial*
  than the general multiply's `_mul_gate` products (1 unit vs 2), but a dense `b`
  has ~all nibbles nonzero so it keeps ~all 36 partials **and** the full 8-column
  carry-resolve — the front-end saving is eaten by the shared carry back-end.
  A never dramatically beats the general multiply on an **unbounded 32-bit** input;
  its real leverage is the **narrowing** levers (result-width and input-width),
  where a bounded `a` cuts A to a small fraction (e.g. 4-bit `a·1000` → 20 %).

**Headline.** With `b` a compile-time constant, the general 10-block / 11 430-nz
multiply is almost never the right gadget: powers of two are a free shift
(`< 1 %`), low-NAF constants are a 1–3-add strength reduction (`~31 %`), and even
dense constants at parity shrink further under input/result-width narrowing. The
auto-picker (`build_const_mul(b, prefer="auto")`) chooses **strength reduction for
low-NAF `b`, linear-fold otherwise**, and shifts for special/power-of-two `b`.
Both constructions are byte-exact over the full 32-bit range and fully fp32.
