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

## 5. Getting depth from forwards-per-step — the vanilla resolution

§4 left a tension: the digit-chain wants ~51 layers, but a *distinct-layer* stack
that deep doesn't fit a stock 24-layer model, and *weight-tying* it is only honest
for a Universal Transformer. There is a **third** place to put the depth — the
**autoregressive token loop.** Extract one digit per *emitted token* (per forward
pass) instead of per *layer*, threading the running remainder through the KV cache
/ token stream between forwards.

This is the natural fit for a transformer VM, which **already emits ~30 tokens per
VM step** — each token is already a forward pass. Spreading the digit-extraction
across those forwards keeps the network **shallow** (a couple of layers,
comfortably inside a stock 24-layer 0.5B) while realizing arbitrary effective
depth in the **sequence**. Crucially, **this recurrence is the ordinary
autoregressive loop — re-invoking the same weights per token, which every language
model already does — not an exotic tied-layer architecture.** So it is genuinely
*vanilla*, and it is the honest resolution of §4's recurrence question: you get the
store-once benefit without a Universal Transformer.

The three ways to realize a digit-extraction depth **D**:

| lever | where the depth lives | stored layers | forwards/step | KV cache | vanilla? |
|---|---|--:|--:|---|---|
| **distinct layers (unrolled)** | within one forward | D (deep) | 1 | ∝ D·seq | vanilla arch, but too deep for stock 24 |
| **looped layers (UT)** | within one forward, tied ×D | few | 1 | ∝ D·seq (applied) | ✗ Universal Transformer |
| **forwards / tokens (autoregressive)** | across D forwards, 1 digit/token | **few — fits stock 24** | **D** | **∝ D·seq (ties deep)** | ✓ standard AR loop |

Compute is conserved — ~D layer-applications per step in every row — so this does
**not** cut FLOPs; it **re-books the depth from parameter-storage / physical-depth
into sequence-length.** The trade-offs (as the `forwards_per_step` module makes
precise):

- **Fitting a vanilla checkpoint:** forwards-per-step *wins* — a shallow network
  of `ceil(D/F)` stored layers, stock architecture, no weight-tying.
- **KV / memory — essentially a *tie* (this corrects an earlier draft):** KV ∝
  n_layers × seq_len, so deep (`D` layers × base_seq) and shallow (`ceil(D/F)`
  layers × `F`·base_seq) both come out to **`D·base_seq`** at exact splits — the
  layers↔forwards trade is **KV-neutral**. *Measured:* full-frame KV is identical;
  under a **bounded eviction window** SHALLOW is **16× smaller** (only the layer
  ratio survives) — so windowed, the vanilla-fit config is the KV *winner*.
- **Realtime latency — the *actual* price:** distinct-layers is usually *faster* —
  `D` layers run in one forward (one kernel-launch chain, one KV read), whereas
  `D` forwards pay `D` launch-chains + `D` KV re-reads and batch worse. *Measured*
  (D=48, real 0.5B-shape blocks): the tax is **small** — DEEP is only **1.01×
  (fp32) / 1.07× (bf16)** faster than SHALLOW×16-forwards at the same effective
  depth. So **vanilla-fit does NOT cost you realtime relative to the deep form** —
  the per-digit-layer *width* (§6) gates absolute fps, not the depth-mode. (Launch
  floor ≈ 1.5 ms fp32 / 5.9 ms bf16 per forward, paid F× → 2.8% of the step in
  fp32 but 28% in bf16: the faster the datapath, the more the fixed launch tax
  shows.)

So the lever exists and it is the *right* one for "fits a real vanilla model" — but
it pays the depth back in KV and per-step forward count. That is exactly why the
solver (§7) must let the depth budget be satisfied by **layers OR forwards**, and
cost each accordingly.

---

## 6. Precision × total-non-zeros × realtime — MEASURED

All rows measured on an idle RTX A5000 (real torch cells — **14/14 op families
byte-exact** on 2–3k random 32-bit operands each: real embed/ALiBi/softmax1 ingest,
ADD/SUB/DIV/MOD/MUL, all six compares, shifts, the real 16×16-nibble-LUT bitwise
FFN, and the real attention-CAM memory ops). Doom step-counts: render-reduced
358,058, raw 6,889,264. `ms/step` = one forward of the **whole** layer stack — a
standard feed-forward transformer runs its entire stack every VM step.
(`examples/clever_realtime_*.py`, `docs/CLEVER_REALTIME_MEASURED.md`.)

| precision | radix | extraction | mode | n_lay | hidden | dense | **total nonzero** | ms/step | render fps | realtime? |
|---|--:|---|---|--:|--:|--:|--:|--:|--:|:--:|
| fp64 | 10 | whole_value | unrolled | 51 | 896 | 217M | **26,119** | 1.081 | 0.003 | ❌ |
| fp64 | 10 | whole_value | looped/UT | 6 | 896 | 26M | **3,183** | 1.081 | 0.003 | ❌ |
| fp32 | 10 | whole_value | unrolled | 51 | 896 | 217M | 26,119 | 0.039 | 0.073 | ❌ |
| **bf16** | 16 | digit_extract | unrolled | 42 | 896 | 179M | 25,624 | **0.009** | **0.312** | ❌ |
| bf16 | 16 | digit_extract | looped/UT | 6 | 896 | 26M | 3,183 | 0.009 | 0.312 | ❌ |
| int8 | 4 | digit_extract | unrolled | 92 | 896 | — | 28,374 | 0.020 | 0.137 | ❌ |
| fp128 | 10 | whole_value | unrolled | 51 | 896 | 217M | 26,119 | 1.081 | 0.003 | ❌ (fp64 floor) |

**Measured total-nonzero** (correcting my earlier ~1–4K estimate — it came in
higher, as flagged, once the FFN-realized LUT + projections are counted): **3,183
looped / 26,119 unrolled** for the full ISA (vs the nibble build's 210,018).

**Two Pareto corners:** min-NONZERO = fp64/fp128 whole_value (**3,183** looped),
slowest datapath; min-WALLTIME = bf16 radix-16 digit_extract (**0.009 ms/step**,
fastest), 25,624 nonzero. Opposite corners; neither realtime.

The **full 64-row `precision × radix × extraction × mode` census** is in
[`docs/CLEVER_NONZERO_CONFIG_TABLE.md`](CLEVER_NONZERO_CONFIG_TABLE.md)
(regen: `examples/clever_nonzero_table.py`). Its structure: the **bitwise LUT
dominates every unrolled total** (2,974 for the OR+AND+XOR triple per layer →
23,792 unrolled), arithmetic collapses to 2,091 unrolled / 153 looped, memory = 10,
and looped is ~8× smaller than unrolled (each cell stored once). 8 configs are
pruned for radix-over-precision-ceiling — int8 caps at radix 2, bf16 at radix 4 for
a *single-precision* whole ISA, so the min-walltime bf16-r16 corner is valid only
via a mixed-precision MUL→fp16 override.

**Realtime verdict — NO, and the honest why.** A realizable *full-width* vanilla
feed-forward clever transformer does **not** render Doom in real time at any
precision — best case ~0.31 fps (bf16), **~100× short** of 30 fps (~500× on the raw
frame). The wall is **not** the arithmetic: a standard feed-forward transformer
runs its **entire 42–51-layer × 896-wide stack every VM step** (9–1081 µs/step
measured), which is **~1000× the isolated-cell cost** (~0.04 µs/op). The earlier
`CLEVER_DOOM_REALTIME.md` ~69 fps projection used the *cell's* throughput, not the
realizable *model's* — **this measurement corrects that projection.**

**But the realtime lever is WIDTH, not depth-mode.** What gates absolute fps is the
**per-digit-layer width**: a full 896-wide Qwen block per digit costs ~1000× a
narrow difference-min cell. So a realizable vanilla transformer *could* reach
realtime **iff the per-digit cell is narrowed** to its few active dims — the
full-width realization is what's ~1000× too slow. That total-nonzero is tiny
(3,183) while walltime is dominated by the 896²-dense tensors the block carries is
exactly the **sparse-but-wide** gap: the information is small, the *realized* matmul
is not.

---

## 7. Constraining the solver: precision + depth + width + KV together

> `c4_min/qwen_fit_solver.py` (`FitConstraints` / `solve_opconfig`),
> `docs/SOLVER_CONSTRAINTS.md`; `forwards_per_step` in `c4_min/forwards_per_step.py`.

The network-size solver now takes **joint** hard constraints — precision, max
depth (layers), max width (hidden), and a **KV-cache byte budget** — and reports
the binding one. It also models the §5 trade: the required effective depth D can
be met by **layers** (deep network, `D` stored) **or by forwards/tokens** (shallow
network, `ceil(D/F)` stored — KV-neutral, same `D·seq`), and the solver costs each.
Wired as `FitConstraints.forwards_per_step`: the depth constraint is checked against
`ceil(D/F)` stored layers, so re-booking depth into the autoregressive loop rescues
a depth-bound config (e.g. the 51-layer clever-fp64 full ISA fits a 24-layer budget
at `F=3` → `ceil(51/3)=17`). The KV formula:

**KV_bytes = 2 (K+V) × n_layers × n_heads × head_dim × seq_len × batch × bytes(precision)**

with one subtlety the note should stress: **a looped/UT model still pays KV for
its *applied* depth** (the loop unrolls into the cache at inference), so looping
saves parameters but **not** KV. This surfaces the real coupling —

**precision ↔ KV ↔ depth:** lower precision shrinks KV *bytes/element* but lowers
the radix ceiling → more digits → more depth → **more layers → more KV**. So the
KV-minimizing precision is not simply "the smallest one"; it's a frontier the
solver searches (`min_kv_precision`).

**Worked constraint table** (`solve_opconfig`; caps: max_layers 24, hidden 896,
inter 4864, KV 256 MiB, seq 2048, batch 1; KV sized on the GQA 2×64 KV-heads):

| config | n_lay | hidden | applied (KV depth) | KV | fits? | binds |
|---|--:|--:|--:|--:|:--:|---|
| nibble-fp32 (default) | 123 | 3008 | 123 | 257.9M | ❌ | depth (+width) |
| clever-fp64 looped/UT | 6 | 896 | 20 | 167.8M | ✅ | — |
| clever-fp64 std-FF unrolled | 51 | 896 | 20 | 167.8M | ❌ | depth |
| clever-fp64 std-FF **+ forwards_per_step=3** | **17** | 896 | 20 | 167.8M | ✅ | — |
| bf16-r16 looped/UT | 6 | 896 | 16 | 16.8M | ✅ | — |

The looped and unrolled clever-fp64 twins pay **identical KV** (167.8M, applied
depth 20) but 26M vs 217M params — weight-tying cuts params, not KV. The
`forwards_per_step=3` row is the §5 lever: it re-books the 51-layer depth into
`ceil(51/3)=17` stored layers and clears the 24-layer cap **KV-neutrally**, as a
plain vanilla autoregressive loop.

**`min_kv_precision` frontier** (the coupling in action): tiny seq/batch →
**fp64 wins** (KV negligible, few-layer depth dominates); huge seq/batch →
**int8 wins** (KV dominates — 1 byte/elem beats fp64's 8 despite int8's lower radix
ceiling forcing more depth).

---

## 8. One-line summary

Replace stored arithmetic tables with a float's own precision + attention's own
positional structure + one shared digit-extractor, and a **210,018-non-zero** 8-bit
ISA becomes a **~3,200-non-zero (looped) / ~26,000 (unrolled)** vanilla transformer
— paying for it in **depth** (as distinct layers, a genuine loop, or
forwards-per-step in the autoregressive loop — all vanilla, all compute-conserved)
and in a **bitwise-LUT floor** floats can't dissolve. The catch the measurement
exposed: those few thousand non-zeros live inside full 896-wide blocks run every
step, so the *realized* model is **~100× off Doom realtime** — the fix is a
**narrow** per-digit cell, not the depth-mode.
