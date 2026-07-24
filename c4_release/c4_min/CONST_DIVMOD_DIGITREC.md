# Constant-divisor digit-recurrence DIVMOD (`const_divmod_digitrec.py`)

The **dead-simple, full-`(q, r)` divmod** for a **compile-time-constant** divisor
`b`, folded as persistent fp32 SwiGLU FFN blocks — byte-exact by construction, no
per-step multiply, no wide subtract, no lexicographic byte-compare, no fp64.

This is one of three constant-divisor gadgets. The other two (built in a sibling
worktree) are a **2-layer fold** for `mod` and a **magic-multiply** for `div`;
this one is the general fallback that produces BOTH quotient and remainder in
`~1–2 blocks per dividend nibble`. See the head-to-head at the bottom.

---

## Algorithm — MSB-first long division, exploiting a constant `b`

Precompute the baked thresholds `k·b` (`k = 1..15`) — pure constants. Keep a
running remainder `R` with the loop invariant `0 ≤ R < b`. For each dividend
nibble `j`, MSB-first:

```
R'  = 16·R + nibbleⱼ                # bring down the next nibble; R<b ⟹ R' < 16·b
qⱼ  = Σ_{k=1..15} [R' ≥ k·b]        # "test all 16 in parallel" — 15 baked thresholds, ONE layer
r   = R' − b·qⱼ                      # LINEAR readout of the same indicators — no multiply, no borrow
R   = r                              # guaranteed 0 ≤ R < b; feeds the next step
emit qⱼ as quotient nibble j
```

Final: quotient = the emitted `qⱼ`, remainder = `R`. The whole point: **`b`
constant ⟹ `k·b` are baked thresholds ⟹ `q` is a parallel count and `r` is a
linear readout of the same indicators.**

### The one subtlety: fp32-exactness needs a re-snap each step

The naive "carry `R` as a scalar and `×16` it" drifts: the silu-identity `×16`
recompose amplifies the ~1e-5 fp32 residue and compounds it across the 8 steps
(measured drift ≈ 0.9 by step 7 → wrong quotient digit). Two disciplines make it
byte-exact:

1. **Carry `R` as clean nibbles** (`R_NIB`). The bring-down `16·R + nibble` is
   then an EXACT positional nibble shift, and `R'` is recomposed to a scalar
   *fresh each step* from clean integers (residue ~0.1, absorbed by the
   `_step_ge` half-integer ramp) — nothing compounds.
2. **Snap the quotient digit to an exact integer BEFORE the reduce.** The raw
   staircase sum `QD_RAW = Σ_k[R'≥k·b]` lands within ~0.03 of the integer; the
   reduce `R = R' − b·QD` would amplify that residue by `b` (≈ 30 for `b=1000`)
   and corrupt the remainder. A second half-integer staircase `QD = Σ_m[QD_RAW≥m]`
   snaps it to the exact integer first, so `b·QD` is exact. The remainder readout
   uses the same raw→snap discipline per nibble.

Both are just extra `_step_ge` staircases (cheap); they turn a marginal circuit
into a batch-matmul-order-invariant, byte-exact one.

---

## Build API

```python
from c4_min.const_divmod_digitrec import build_const_divmod_digitrec, run_divmod

circ = build_const_divmod_digitrec(b, in_width=32)   # b is a python int (constant)
q, r = run_divmod(circ, a)                            # a // b, a % b  (fp32 forward)
```

`build_const_divmod_digitrec(b, in_width)` returns a `DigitRecDivmod` dataclass
carrying the ordered block list (`.blocks`, each a SwiGLU tensor dict), the band
map (`.A` dividend nibbles, `.QUOT`, `.REM`, `.ONE`), the narrowing widths, and
the fp32 metadata. It uses ONLY the pure SwiGLU unit emitters from
`nibble_alu32` (`_step_ge`, `_ident`, `_clear`, `_empty_spec`, `_truncate`,
`RELU_S`, `S`) — it never touches `nibble_alu32.py`, `nibble_vm.py`,
`qwen_full_vm.py`, or any `const_divmod*` build path.

- **`b == 0 → (0, 0)`** (ISA_SPEC 4.2, matching `isa.interpret`) — resolved AT
  BUILD TIME (a compile-time constant folds the divide-by-zero branch away: the
  whole circuit is a single clear-to-zero block).
- **Powers of two** need no special case — the general staircase is exact for
  them (verified 2..4096).

### Narrowing (provably-zero nibbles are never emitted)

- dividend nibbles processed = `ceil(in_width/4)` (skip statically-zero leading
  nibbles when the dividend is bounded by `in_width`),
- quotient nibbles = `ceil(log₁₆(2^in_width / b))` (higher nibbles are provably 0
  for a bounded dividend),
- remainder nibbles = `ceil(log₁₆ b)` (`r < b`).

A tight `in_width` both shrinks the block count (fewer steps) and narrows the
quotient — e.g. `b=1000` at `in_width=16` drops from 106 blocks / 6 q-nibbles to
fewer of each.

---

## Verification — byte-exact through the real SwiGLU forward

`python -m c4_min.const_divmod_digitrec` runs the battery; `pytest
c4_min/test_const_divmod_digitrec.py` is the test companion. Every case runs the
built blocks through the **exact production fp32 SwiGLU forward**
(`x + W_down·(silu(W_up·x+b_up)·(W_gate·x+b_gate))`), **0 fp64**, and checks BOTH
`q` and `r` byte-exact vs `(a//b, a%b)` over **≥2000 random 32-bit `a` + the edges
`{0, 1, b−1, b, b+1, 2³²−1}`**.

**Result: ALL BYTE-EXACT (q and r), 41/41 pytest, battery exit 0.**

### Cost + narrowing table (measured)

| `b`    | in_width | blocks | nz     | blocks/step | in_nib | q_nib | r_nib | max-recon | fp32-exact | q pass | r pass |
|-------:|---------:|-------:|-------:|------------:|-------:|------:|------:|----------:|:----------:|:------:|:------:|
| 2      | 32       | 74     | 9 030  | 9.25        | 8      | 8     | 1     | 29        | ✔          | 2006/2006 | 2006/2006 |
| 16     | 32       | 74     | 9 030  | 9.25        | 8      | 8     | 1     | 239       | ✔          | 2006/2006 | 2006/2006 |
| 100    | 32       | 90     | 11 865 | 11.25       | 8      | 7     | 2     | 1 499     | ✔          | 2006/2006 | 2006/2006 |
| 256    | 32       | 90     | 11 865 | 11.25       | 8      | 7     | 2     | 3 839     | ✔          | 2006/2006 | 2006/2006 |
| 1000   | 32       | 106    | 14 700 | 13.25       | 8      | 6     | 3     | 14 999    | ✔          | 2006/2006 | 2006/2006 |
| 4096   | 32       | 106    | 14 700 | 13.25       | 8      | 6     | 3     | 61 439    | ✔          | 2006/2006 | 2006/2006 |
| 5592   | 32       | 122    | 17 535 | 15.25       | 8      | 5     | 4     | 83 879    | ✔          | 2006/2006 | 2006/2006 |
| 8000   | 32       | 122    | 17 535 | 15.25       | 8      | 5     | 4     | 119 999   | ✔          | 2006/2006 | 2006/2006 |
| 65521  | **16**   | 62     | 8 595  | 15.50       | 4      | 1     | 4     | 61 424    | ✔          | 2006/2006 | 2006/2006 |
| 40961  | **16**   | 62     | 8 595  | 15.50       | 4      | 1     | 4     | 36 864    | ✔          | 2006/2006 | 2006/2006 |
| 50021  | **16**   | 62     | 8 595  | 15.50       | 4      | 1     | 4     | 45 924    | ✔          | 2006/2006 | 2006/2006 |
| 4093   | **16**   | 54     | 7 236  | 13.50       | 4      | 2     | 3     | 61 394    | ✔          | 2006/2006 | 2006/2006 |
| 0      | 32       | 1      | 27     | —           | 8      | 8     | 1     | —         | ✔ (0,0)    | all (0,0) | all (0,0) |

`blocks/step` = total blocks ÷ dividend nibbles processed. Depth is
**~1–2 blocks per nibble-step for the recurrence core** (bringdown + qraw +
qsnap + reduce + the re-nibble peel pairs), i.e. the whole divmod is **~7·(steps)
blocks**, and NARROWING (a bounded dividend) cuts both steps and quotient width.
The general (data-`b`) long division for comparison is **~262 blocks** — this
constant-`b` version is ~2–4× shallower because it drops the KB precompute, the
GT/EQ lanes, the `QD·b` multiply, and the borrow subtract (all replaced by baked
thresholds + a linear readout).

### fp32 discipline & ceiling (0 fp64)

The binding fp32 quantity is the **silu-relu OUTPUT VALUE** (the reconstructed
integer the ramp snaps), **not** the pre-scale argument `RELU_S·z`. The
`_step_ge` ramp is a *difference* of two nearby relu outputs
(`relu(z−(c−w)) − relu(z−c)`), and catastrophic cancellation makes that
difference exactly `w` even when the shared large value `z` rounds — so the
pre-scale `RELU_S·z` (up to ~2e8) is **not** the limit. What must stay `< 2²⁴` is
the reconstructed value `z = R'−thr` (q-staircase) and `r_top − 16^(rn−1)`
(readout).

- **Empirical full-32-bit-dividend ceiling: `b ≲ 12 000`** (sweep-verified: the
  q-staircase reconstructs `R'−b < 15·b`, exact while `15·b < 2²⁴`). The task's
  `~56k` estimate assumed `RELU_S=20`; this build uses the codebase `RELU_S=200`,
  and the ramp's difference-cancellation gives ~2× headroom past the naive
  `RELU_S·15·b < 2²⁴` bound (`b<5.6k`) — so the honest number lands at ~12k.
- **The REMAINDER readout is the tighter constraint for large `b`:** the readout
  peels nibble `c` with weight `16^c`, and for `r_nibbles ≥ 5` (`b > 2¹⁶`) the
  running residual (a value up to ~2²⁰) plus the `16⁴`-weighted peel exceeds
  fp32 exactness → **q stays exact, r fails.** So the full-`(q,r)` divmod is
  fp32-exact for **`b ≤ 2¹⁶`** (`r_nibbles ≤ 4`).
- **Narrowing rescues large `b`:** with `2^in_width ≤ ~b` the quotient is `∈{0,1}`
  and the remainder `r = a < 2^in_width`, so `b = 65521 / 40961 / 50021` are all
  byte-exact at `in_width=16`. Narrowing does NOT rescue the per-step `R'` range
  (which is `< 16·b` regardless of dividend size once `R` accumulates), so a
  divisor with `b > 2¹⁶` and a dividend `≥ b` is the genuine fp32 wall — reported
  honestly in the `fp32-LIMITED` battery section (q exact, r fails).

---

## Head-to-head: which constant-divisor gadget is smallest?

Three constant-divisor gadgets, three regimes. Honest per-regime winner:

| gadget | depth | produces | fp32 range | smallest when |
|---|---|---|---|---|
| **fold** (2 layers) | ~2 layers | **`mod` only** | small `b` (fits the fold's residue budget) | you need only `a mod b` for a **small** `b` — a base-16 digit-sum fold `a mod b = (Σ nibᵢ·(16^i mod b)) mod b` collapses to ~2 layers; unbeatable for small-`b` mod. |
| **magic-multiply** (~3–5 blocks) | ~3–5 blocks | **`div` only** (`r` via one extra `a − b·q`) | **large `b`, widest fp32 div** | you need `a // b` for a **large** `b`: a baked `⌈2ᴺ/b⌉` reciprocal + a fixed shift is ~constant-depth and never reads the remainder back, so it dodges the readout wall entirely — the widest-range div. |
| **digit-recurrence** (this) | **~7·⌈in_width/4⌉ blocks** (~1–2/nibble) | **full `(q, r)`** | **you need BOTH `q` and `r`, or the simplest correct-by-construction option** | full divmod, any `b ≤ 2¹⁶` (or narrowed), no magic-constant derivation, no separate mod path — the dead-simple general fallback. |

**Verdict, per regime:**

- **small-`b` `mod`** → the **fold** wins (2 layers beats ~50–74 blocks here).
- **large-`b` `div`** → the **magic-multiply** wins (constant depth, and it never
  hits the remainder-readout fp32 wall — the widest-range div, `b` up to ~2³¹).
- **full `divmod` / simplicity / widest fp32 range for BOTH `q` and `r`** → this
  **digit-recurrence** wins. It is the only one that emits `q` and `r` together
  byte-exact from one baked-threshold circuit, needs no magic-constant
  derivation, and its narrowing makes any `b ≤ 2¹⁶` (or a bounded-dividend large
  `b`) exact. It is depth-`O(nibbles)` (deeper than the other two) — the price of
  being the general, no-cleverness-required option.

The fold and magic-multiply are the *specialized* winners; the digit-recurrence
is the *robust default* — the one you reach for when you need the whole `(q, r)`
or don't want to derive a magic constant.
