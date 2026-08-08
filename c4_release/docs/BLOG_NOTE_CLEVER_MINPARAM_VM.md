# From a Minimal Adder to a Whole-ISA Clever VM: trading weights for depth

*Draft note for the blog post. Companion to the minimal-10-digit-adder post
(alexlitzenberger.com). All "measured" numbers are reproduced by the standalone
examples under `examples/`; none of this touches the byte-exact production VM
(golden `174ece66`).*

---

## 1. The adder hack, in one paragraph

The minimal-adder post held a whole multi-digit sum inside **one floating-point
value** and read the digits back out with attention, instead of storing an
arithmetic circuit. Three tricks do it:

- **Precision as storage.** A d-digit number lives in one fp scalar (exact
  because integers are exact in floats up to 2^53 for fp64).
- **Place value for free.** ALiBi attention with slope `ln(radix)` generates the
  descending powers of the radix as a *fixed bias* (one scalar), and `softmax1`
  (the `+1` in the denominator) turns attention's *mean* into a *sum*. The
  positional structure is **computed by the attention mechanism, not stored**.
- **Digit extraction by difference-min.** A decode head scores each candidate
  digit `d` with `logit_d = -|value - (d + 0.5)|`; `argmax` = `floor`. Ten
  candidate values (`0..9`) shared across every place.

Measured, as a real 1-layer transformer (`examples/minimal_10digit_adder.py`),
that whole machine is **42 total non-zero weights**:

| piece | total non-zeros |
|---|--:|
| embedding (9 digit-values + 3 flags) | 12 |
| attention Q/K/V/O identity diagonals (4 × d_model) | 16 |
| candidate digits `0..9` | 10 |
| ALiBi slope `ln(10)` · softmax1 `+1` · floor `+0.5` · base `10.0` | 4 |
| **total** | **42** |

**Counting convention (used everywhere below): TOTAL non-zeros = every position
in every weight tensor that holds a non-zero value, repeats included.** The 16
identity entries are all `1.0` but count as 16 separate positions; this is *not*
a count of distinct values. (The "4 scalars" figure some earlier notes quoted is
the distinct-*value* count — a different, smaller number that is **not** what a
checkpoint stores.)

---

## 2. The same machine is a whole ISA — what actually saves so much

The production byte-exact c4 VM does arithmetic the 8-bit way: it splits every
32-bit value into nibbles and stores explicit per-nibble lookup / carry / borrow
/ corrector tables. Measured total non-zeros in that build (golden `174ece66`):

| | total non-zeros |
|---|--:|
| full ISA (all ops) | **210,018** |
| — of which DIV/MOD alone | **163,591** |
| — bitwise OR/XOR/AND (each) | 5,438 |

The clever construction deletes almost all of that by moving arithmetic **out of
the weights and into the float**:

1. **Hold the whole value in one float** → the entire nibble-decomposition +
   carry/borrow corrector apparatus disappears. Add/sub/compare become a single
   float add. This is the bulk of the 210k.
2. **Place value via ALiBi + softmax1** → the positional tables disappear (one
   slope scalar generates them).
3. **One shared difference-min decode** (`0..radix-1` candidates) does the digit
   read-out for *every* arithmetic op.

The punchline: **arithmetic stops being stored *data* and becomes stored
*function*.** A single ~42-non-zero cell + ALiBi slope + candidate table serves
ADD, SUB, MUL, DIV, MOD, all six compares, and the shifts — because they are all
"hold the value in a float, extract digits." **DIV/MOD goes from 163,591
non-zeros to sharing that one cell (0 new stored weights).** The cost doesn't
vanish — it **moves from width (huge tables) to depth (one cheap layer per output
digit).**

**What does *not* collapse, and why:**

- **Bitwise (OR / XOR / AND).** Floats have no bit operations, so you genuinely
  need a per-nibble 16×16 lookup: **136 entries** (OR/AND, by symmetry), **256**
  (XOR). After arithmetic collapses, **the bitwise LUTs become the dominant
  non-zero chunk of the whole ISA.**
- **Memory (LI / LC / SI / SC).** One shared content-addressed attention (CAM),
  ~10 weights, reused by all four (per-op marginal ≈ 0).

---

## 3. Depth ↔ radix: can bigger steps avoid the depth?

Depth = one digit per layer = `ceil(bits / log₂ radix)`. Bigger steps (bigger
radix) mean fewer layers — but the difference-min decode needs a **`radix`-entry
candidate table**, and in a standard (unrolled) stack that table is replicated
per layer. So the decode non-zeros go as:

**total ≈ depth × radix = (bits / log₂ radix) × radix**

| radix | layers (32-bit) | candidate table | depth × radix |
|--:|--:|--:|--:|
| 2 | 32 | 2 | 64 |
| 16 | 8 | 16 | 128 |
| 256 | 4 | 256 | 1,024 |
| 65,536 | 2 | 65,536 | 131,072 |

The table grows **linearly** in radix while depth only falls **logarithmically**,
so the product *grows*: **bigger steps trade depth for width and net *more*
non-zeros.** Min-non-zeros is at *small* radix + deep. Two hard limits also bind
radix: the **precision ceiling** (you must resolve 1-part-in-radix in the
mantissa, and the op's accumulator — add `~2r`, cmp `~r+1`, div `~r²`, mul
`~r²·L` — must stay exactly representable, so DIV can't push radix as high as
ADD), and the absurdity of `radix = 2³²` (depth 1, a four-billion-entry table).

**Practical takeaway:** moderate radix (16–256) is the sweet spot — depth ~4–8 at
a 16–256-entry table. If *latency/depth* is your realtime worry, radix-256 roughly
**quarters the layers** for a modest non-zero increase: a good realtime trade even
though it is *not* the minimum-parameter corner.

---

## 4. Recurrence is only legitimate for a *looped* transformer

A tempting shortcut is to weight-tie the reused digit-cell and count it once.
That is only valid if the model **is** a looped / Universal-Transformer
implementation (the forward pass applies the same weights N times). A stock
feed-forward transformer (e.g. released Qwen2.5-0.5B: 24 *distinct* layers, each
applied once) **cannot** tie — it must **unroll** into distinct layers. So:

| | stored non-zeros | n_layers | note |
|---|--:|--:|---|
| **standard feed-forward (unrolled)** | full replicated total | summed per-op depth (≈ 51 for full clever-fp64 ISA) | every layer stores its own copy |
| **looped / Universal Transformer** | one copy of each cell | few stored, applied N | a *different architecture* than stock Qwen2 |

Consequence for "does it fit a stock 0.5B?": the unrolled clever-fp64 full ISA is
**narrower (hidden 896 vs 3008) and shallower (≈51 vs 123 layers) than the nibble
build** — a real win — but its ~51 distinct layers still exceed a stock 24-layer
budget, so it **fits the width, not the depth.** Only a *looped* variant fits a
0.5B-width checkpoint, and only as a Universal Transformer. **Recurrence reduces
*storage*, not *information*: the distinct cells are the same either way; looping
just avoids re-storing them.** (It also does *not* reduce the KV cache — see §6.)

---

## 5. Precision × total-non-zeros × realtime  *(measured — table pending)*

> Filled from the measurement build (`examples/`, `docs/CLEVER_REALTIME_MEASURED.md`).
> Columns: precision · radix · extraction · mode · n_layers · hidden · dense
> params · **total non-zeros** · ms/step (measured, GPU) · fps · realtime?
> Corners: min-NON-ZERO (fp64 whole-value + digit-extract) and min-WALLTIME
> (bf16 radix-16). Precision throughput anchors already measured:
> **bf16 ≈ 5.4× fp32 · int8 ≈ 2.7× (4× not realized on A5000) · fp64 ≈ 0.02× fp32.**

`<<INSERT MEASURED PRECISION TABLE>>`

**Realizability, stated plainly:** the unrolled clever full ISA *is* a genuine
vanilla feed-forward transformer — ~51 layers, hidden 896, ~217M dense params
(smaller than 0.5B's 494M but deeper), order-10³ total non-zeros, no recurrence,
no exotic ops (attention + FFN + ALiBi + softmax1 + difference-min head). Whether
it renders Doom in real time byte-exact is the measurement above; the current
*proven* byte-exact Doom is the wide nibble build at ~1 fps (memory-bound), and
the raw frame needs a render-macro step-fold independent of the cell cost.

---

## 6. Constraining the solver: precision + depth + width + KV together  *(pending)*

> Filled from the solver extension (`c4_min/qwen_fit_solver.py`,
> `docs/SOLVER_CONSTRAINTS.md`).

The network-size solver now takes **joint** hard constraints — precision, max
depth (layers), max width (hidden), and a **KV-cache byte budget** — and reports
the binding one. The KV formula:

**KV_bytes = 2 (K+V) × n_layers × n_heads × head_dim × seq_len × batch × bytes(precision)**

with one subtlety the note should stress: **a looped/UT model still pays KV for
its *applied* depth** (the loop unrolls into the cache at inference), so looping
saves parameters but **not** KV. This surfaces the real coupling —

**precision ↔ KV ↔ depth:** lower precision shrinks KV *bytes/element* but lowers
the radix ceiling → more digits → more depth → **more layers → more KV**. So the
KV-minimizing precision is not simply "the smallest one"; it's a frontier the
solver searches (`min_kv_precision`).

`<<INSERT MEASURED SOLVER CONSTRAINT TABLE>>`

---

## 7. One-line summary

Replace stored arithmetic tables with a float's own precision + attention's own
positional structure + one shared digit-extractor, and a 210,018-non-zero 8-bit
ISA becomes an order-10³-non-zero vanilla transformer — paying for it in **depth**
(not recurrence, unless you genuinely build a loop) and in a **bitwise-LUT floor**
that floats can't dissolve.
