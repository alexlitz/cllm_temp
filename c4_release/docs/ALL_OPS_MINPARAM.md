# All c4 opcodes — clever minimal-param construction (fp32 / fp64 / fp128)

Extends the whole-value clever-minparam ALU (`examples/clever_minparam_alu.py` +
`examples/minimal_10digit_adder.py`) from ADD/SUB/MUL/DIV/MOD to **every c4
opcode**, adds an **fp32-clever** (nibble-serial recurrent) variant, and computes
the **network geometry** the clever op-set implies via the `c4_min` fit solver.

Standalone examples + a config/sizing analysis. Touches **no** c4_min build file
that defines golden `174ece66` (nibble_pure_forward_complete / nibble_alu32 /
compact_alloc). All numbers below are MEASURED by running the example on CPU
(x86-64; fp128 = numpy longdouble, 80-bit ext, 64-bit effective mantissa).

Reproduce:

```
python examples/clever_minparam_alu.py            # census + verify every op
python examples/clever_minparam_alu.py --op allops  # the per-op table only
python examples/clever_minparam_alu.py --op fp32    # fp32 variant + verify
python examples/clever_minparam_alu.py --op fitter  # network-size geometry
python examples/clever_minparam_alu.py --n 100000   # heavy exactness (765k cases)
```

## The two levers (recap)

1. **HIGH PRECISION** — hold the WHOLE operand/result value in ONE fp scalar (no
   nibble/bit split). fp64 holds every 32-bit value and the ~10^11 add/div range
   exactly (2^53); fp128 holds the 64-bit MUL product exactly (2^64).
2. **CLEVER DEPTH** — extract the result one decimal DIGIT per layer, MSB-first,
   with the difference-min selector `logit_d = -|value-(d+0.5)|` (argmax = floor),
   the per-digit cell REUSED (recurrence). Unique params are constant; DEPTH =
   #digits.

Every op routes into ONE of a handful of shared cells. The whole point: an op
that reuses the shared place-value ingest + difference-min decode adds **zero**
new stored weights — only the whole-value expression feeding the decode changes.

---

## Per-op table — which ops collapse to ~4, which have a floor

Class legend: **collapse** = reuses the shared ingest + difference-min decode cell
(0 new stored weights; scalars-only floor == the SHARED 4). **bitwise** = a
per-nibble 16×16 LUT floor (no ~4 collapse). **memory** = the shared-CAM floor
(per-op marginal ~0). **trivial** = register move / stack write / no-op.

`scal(d)` = scheme-(d) scalars-only unique-weight floor for that op. `all(a)` =
scheme-(a) all-nonzero (the shared ingest+decode is 42; collapse ops share it).

| op | class | prec | depth | scal(d) | all(a) | floor / note |
|----|-------|------|------:|--------:|-------:|--------------|
| ADD | collapse | fp64 | 11 | **4** | 42 | whole a+b, digit-extract |
| SUB | collapse | fp64 | 11 | **4** | 42 | whole a−b (wrap), digit-extract |
| MUL | collapse | fp128 | 20 | **4** | 42 | fp128 whole 64-bit product, digit-extract |
| DIV | collapse | fp64 | 10 | **4** | 42 | fp64 long-division digit-extract |
| MOD | collapse | fp64 | 10 | **4** | 42 | fp64 long-division remainder |
| EQ NE LT GT LE GE | collapse | fp64 | 1 | **4** | 42 | sign(a−b) held whole in fp64, +0.5 read; all 6 share ONE subtract |
| SHL | collapse | fp64 | 11 | **4** | 42 | ×2^n whole-value scale + 32-bit mask |
| SHR | collapse | fp64 | 11 | **4** | 42 | arithmetic /2^n whole-value scale (signed) |
| LEA | collapse | fp64 | 1 | **4** | 42 | bp+imm, whole-value add |
| JMP | collapse | fp64 | 1 | **4** | 42 | pc := imm (imm select) |
| BZ / BNZ | collapse | fp64 | 1 | **4** | 42 | CMP(ax,0) gates a PC add |
| ENT | collapse | fp64 | 1 | **4** | 42 | sp−imm add + 1 CAM stack write |
| ADJ | collapse | fp64 | 1 | **4** | 42 | sp+imm, whole-value add |
| LEV | collapse | fp64 | 1 | **4** | 42 | sp := bp + 2 CAM stack reads |
| JSR | collapse | fp64 | 1 | **4** | 42 | pc := imm + 1 CAM stack write |
| IMM PSH NOP HALT | trivial | fp32 | 1 | **0** | 0 | register move / stack write / no-op (no arithmetic) |
| OR | bitwise | fp32 | 8 | **136** | 42 | per-nibble 16×16 LUT floor = 256 (symmetric 136), depth 8 |
| AND | bitwise | fp32 | 8 | **136** | 42 | per-nibble 16×16 LUT floor = 256 (symmetric 136), depth 8 |
| XOR | bitwise | fp32 | 8 | **256** | 42 | per-nibble 16×16 LUT floor = 256, depth 8 |
| LI LC SI SC | memory | fp32 | 1 | **0** | 10 | **shared-CAM floor = 10** (shared by all 4 ops; per-op marginal 0) |

### Which collapse to ~4, which have a floor

- **COLLAPSE to ~4 scalars (zero new stored weights):** ADD, SUB, MUL, DIV, MOD,
  all six CMP (EQ/NE/LT/GT/LE/GE), SHL, SHR, LEA, JMP, BZ, BNZ, ENT, ADJ, LEV,
  JSR. All 25 reuse the same fp place-value ingest + difference-min decode cell;
  the scalars-only floor is the SAME 4 shared scalars — `ln(10)` slope, `+1`
  softmax1 off-by-one, `+0.5` floor shift, `10.0` place base. CMP is even leaner
  (no digit loop — one sign read, depth 1). The branch/frame ops are literally an
  address ADD plus at most one shared-CAM stack touch.
- **BITWISE FLOOR (hundreds, NOT ~4):** OR/XOR/AND. Floats have no bit ops, so the
  whole-value fp precision cannot help. The minimal EXACT realisation is a
  per-nibble 16×16 lookup applied 8× (depth 8): the honest floor is the **256**
  entry table (**136** by symmetry for OR/AND; XOR needs the full 256). This does
  not collapse to 4. (A bit-serial fallback trades the 256-table for depth 32.)
- **MEMORY FLOOR (shared CAM, NOT per-op arithmetic):** LI/LC/SI/SC. The address
  arithmetic is nil (an address is already a whole value). The real cost is the
  ONE shared content-addressed memory — an attention CAM keyed on the 32-bit
  address (~10 shared weights: 8 nibble-match key lanes + a match sharpness + a
  value read). All 4 ops reuse it, so the **per-op marginal is 0**; the floor is
  the shared CAM, counted once.

### Reduction vs the production nibble-c4 exclusive param counts

| op | nibble-c4 exclusive | clever scalars-only (d) | reduction |
|----|--------------------:|------------------------:|----------:|
| ADD/SUB | 6,459 | 4 | **1,615×** |
| DIV/MOD | 163,591 | 4 | **40,898×** |
| MUL | 7,911 | 4 | **1,978×** |

The clever core pays this back in DEPTH (the reused cell is applied `depth` times).

---

## fp32-clever variant — how small can fp32 get

fp32's 24-bit mantissa (2^24 ≈ 1.68e7) **cannot hold a 32-bit value**, so the
whole-value trick fails outright. But the CLEVER DEPTH lever still crushes the
param count: **nibble-decompose** and run a REUSED per-nibble cell (recurrence).
Every per-nibble value is < 16 → trivially fp32-exact; the cell is applied once
per nibble (depth = nibble count = 8 for 32-bit).

- **ADD/SUB** — nibble-serial carry ripple, LSB→MSB, 8 steps. One shared
  (add, floor(/16) carry, s−16·carry residue) cell; SUB is the same cell over the
  two's-complement of b (per-nibble `~b`, carry-in 1). Every `s < 32` is fp32-exact.
- **DIV/MOD** — nibble-serial restoring long division, base-16, MSB-first, 8
  steps. One shared (bring-down `R·16+a_k`, difference-min quotient nibble over
  0..15, restoring subtract) cell.

**Measured (VERIFIED EXACT over 32-bit, ADD/SUB + DIV/MOD):**

| op | fp32 scalars-only | fp32 all-nonzero | fp32 depth | fp64-clever scalars | fp64 depth | nibble-UNROLLED | reduction (scalars) |
|----|------------------:|-----------------:|-----------:|--------------------:|-----------:|----------------:|--------------------:|
| ADD | 4 | 28 | 8 | 4 | 11 | 6,459 | **1,615×** |
| DIV/MOD | 4 | 28 | 8 | 4 | 10 | 163,591 | **40,898×** |

The fp32-clever `all-nonzero` = 28 = 4 scalars + 16 nibble candidates (0..15,
reused axis) + 8 nibble-split embedding lanes. Same ~few-dozen unique-weight
footprint as fp64-clever — the point of the task: **even fp32 crashes the param
count via recurrence, just deeper** (nibble places instead of decimal places).
DEPTH is comparable (8 nibbles vs fp64's 10–11 decimal places) because base-16 is
denser than base-10.

Contrast: the production **nibble-UNROLLED** build stores every per-nibble
lookup/carry corrector as a distinct weight (ADD 6,459 / DIV·MOD 163,591). The
fp32-clever cell stores ONE per-nibble cell and applies it 8× → ~4 unique scalars
(28 all-nonzero), a **~1,600× / ~41,000×** unique-weight reduction, paid in depth.

---

## Fitter geometry — does clever-FULL fit stock 0.5B?

The fit solver (`c4_min/qwen_fit_solver.py`, `docs/HF_MODEL_FIT.md`) derives
`(n_layers, hidden, intermediate, params)` from the op-set's block specs. The
nibble FULL build needs ~3B-class WIDTH. The clever construction is the opposite:
a NARROW reused cell (depth-in-time), so its hidden/intermediate are
stock-trivial and its cost is APPLIED depth (the recurrence unroll).

Config axis added to the analysis: **nibble-fp32 | clever-fp64 | clever-fp128 |
fp32-clever**.

### NIBBLE build (MEASURED live on this branch via the real solver)

Stock Qwen2.5-0.5B budget: hidden ≤ 896, intermediate ≤ 4864, layers ≤ 24.

| subset | hidden | intermediate | STORED | D_used | fits 0.5B? |
|--------|-------:|-------------:|-------:|-------:|:----------:|
| base (ISA core) | 896 | 66 | 7 | 270 | **YES** |
| FULL (unrolled) | 3008 | 7920 | 123 | 2948 | no (hidden binding) |
| FULL (recurrent) | 3008 | 7920 | 60 | 2948 | no (hidden binding) |

Solver verdict: nibble-**FULL does NOT fit** stock 0.5B — **binding constraint =
hidden** (needs ~3B-class width). (These live numbers differ slightly from the
older `HF_MODEL_FIT.md` snapshot — 3008/123/60 vs 2944/107/44 — because this
worktree branch differs from `f5f31bd8`; the solver is the authority for the
current tree.)

### CLEVER op-set geometry (this construction) — TWO honest modes

**Recurrence requires a LOOPED / Universal-Transformer.** The stock released
**Qwen2.5-0.5B is a STANDARD feed-forward transformer** (24 distinct decoder layers,
each applied once), so it **cannot weight-tie** — it must UNROLL. The clever
geometry therefore has two honest modes; hidden is floored at the Qwen GQA head
partition (14 q-heads × 64 = **896**) in both.

**(a) STANDARD feed-forward (UNROLLED)** — the active op is data-dependent
(conditionally applied at run time), so the feed-forward network must CONTAIN every
op's machinery as DISTINCT stored layers. `n_layers` = the **summed unrolled depth**
across all machinery families (arith / div / mul / bitwise / memory / trivial).

| config | precision | hidden | intermediate | STORED n_layers (unrolled) | distinct params | fits stock 0.5B (24 layers)? |
|--------|-----------|-------:|-------------:|---------------------------:|----------------:|:----------------------------:|
| clever-fp64-FULL (whole-value) | fp64/fp128 | 896 | 896 | **51** = arith 11 + div 10 + mul 20 + bitwise 8 + mem 1 + trivial 1 | ~**217M** | **NO — depth** (51 > 24) |
| bf16-radix16-FULL (digit-extract) | fp16/bf16 | 896 | 896 | **42** = arith 8 + div 8 + mul 16 + bitwise 8 + mem 1 + trivial 1 | ~**179M** | **NO — depth** (42 > 24) |

clever-UNROLLED is **narrower AND shallower than nibble** (896 < 3008; 51 < 123 —
the digit-extraction depth is far less than the **189-block nibble long-division**),
but its **tens-to-hundreds of DISTINCT layers still exceed stock 0.5B's 24** → it
**fits WIDTH, NOT DEPTH → does NOT fit stock 0.5B as a standard transformer.**

**(b) LOOPED / Universal-Transformer (TIED)** — one reused cell re-applied `depth`
times per forward. STORED shrinks to a handful of cells; the cost is APPLIED depth.

| config | precision | hidden | intermediate | STORED (reused cells) | applied depth | fits a 0.5B-WIDTH UT checkpoint? |
|--------|-----------|-------:|-------------:|----------------------:|--------------:|:-------------------------------:|
| clever-fp64-FULL (whole-value, tied) | fp64/fp128 | 896 | 896 | ~**6** | 20 (MUL) | **YES — as a UT model, NOT stock feed-forward Qwen2** |
| bf16-radix16-FULL (digit-extract, tied) | fp16/bf16 | 896 | 896 | ~**6** | 16 (MUL) | **YES — UT-width checkpoint, not stock feed-forward** |

(+ the bitwise LUT block adds an intermediate-256 FFN table; the shared memory CAM
adds one attention head of ~10 weights. Neither perturbs the 896 hidden floor.)

### Does clever-FULL fit stock 0.5B? — NO as a standard transformer; only as a LOOPED/UT checkpoint

- **As a STANDARD feed-forward transformer (what stock Qwen2.5-0.5B is): NO.** The
  network must UNROLL (no loop to re-apply a tied cell), so `n_layers` = **51**
  (clever-fp64-FULL) / **42** (bf16-radix16-FULL) distinct stored layers, which
  **exceeds the 24 stock layers**. It fits the stock 0.5B **WIDTH** (hidden 896 ≤ 896,
  intermediate ≤ 4864) but **NOT the DEPTH** — the binding constraint is **depth**,
  not width.
- **As a LOOPED / Universal-Transformer: YES for the WIDTH,** as a **different
  architecture.** The tied variant stores ~6 reused cells (≤ 24) and re-applies them
  per forward (deepest single op = MUL depth 20), so it fits a 0.5B-**width** UT
  checkpoint — but a UT is **not** the released stock feed-forward Qwen2. The
  ~4-scalar / ~6-cell param win is a **UT-checkpoint claim only.**

This corrects the earlier "clever-FULL FITS stock 0.5B" phrasing, which counted
`tied` recurrence as a param-win for a stock feed-forward checkpoint — a model that
cannot actually use recurrence. Nibble FULL is blocked by **WIDTH** (hidden ~3008 >
896); clever-UNROLLED trades that width for DEPTH but its unrolled depth still
overshoots stock 0.5B's 24 layers. Recurrence buys the depth-fit **only** by
switching to a looped/UT architecture.

---

## Honesty notes

- All arithmetic (collapse) ops are VERIFIED EXACT over ≥100k random 32-bit
  operands + hard edge cases (765k cases total in the default `--n 100000` run),
  against the c4 reference semantics (32-bit signed word; SHR/CMP signed; SC/LC
  byte + sign-extend).
- MUL requires fp128: the negative control shows fp64 mis-floors 2/4 big 64-bit
  products (2^53 ceiling). Every other op is fp64-exact (fp32 for the nibble and
  bitwise cells whose per-step values are < 16).
- BITWISE and MEMORY are HONEST FLOORS, not ~4-scalar collapses: 256/136-entry
  nibble LUT and the ~10-weight shared CAM respectively. The table calls these out
  explicitly rather than pretending they collapse.
- The fitter clever geometry is an ACCOUNTING of what the clever op-set implies
  (narrow reused cell, hidden floored to the Qwen head partition), computed
  alongside the REAL nibble-build solver numbers. It is not a baked model — it is
  the sizing analysis the task asked for.
</content>
</invoke>
