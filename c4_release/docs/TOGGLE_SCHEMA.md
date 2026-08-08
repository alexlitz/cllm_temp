# Per-op TOGGLE schema — precision × radix × extraction × recurrence

A first-class, per-opcode config toggle system that consolidates the completed
min-param / precision / depth work
([`CLEVER_MINPARAM_ALU.md`](CLEVER_MINPARAM_ALU.md),
[`ALL_OPS_MINPARAM.md`](ALL_OPS_MINPARAM.md),
[`PRECISION_RADIX_SURFACE.md`](PRECISION_RADIX_SURFACE.md),
[`CLEVER_DOOM_REALTIME.md`](CLEVER_DOOM_REALTIME.md)) into ONE resolvable schema,
wired into the size-fitter + the `C4_*` flag registry.

**Golden safety:** the global DEFAULT config == the current nibble / fp32 build ==
golden **`174ece66`**. Building from DEFAULT is byte-identical to golden (the
resolver never touches the build unless a non-default `C4_OPCFG_*` flag is present).
Every non-default axis value is a documented **golden-MOVING** toggle.

**Scope (load-bearing honesty):** this is the **CONFIG / TOGGLE + FITTER + DOCS**
layer only. The byte-exact doom VM is **NOT** yet rebuilt with the clever ops —
that is separate future work (§7). The low-precision speedup is a **narrow-VM
lever**: today's wide VM is memory-bound, where bf16 nets ~1.0× (§6).

- Schema + validator + resolver: [`c4_min/opconfig.py`](../c4_min/opconfig.py)
- Fitter wiring: [`c4_min/qwen_fit_solver.py`](../c4_min/qwen_fit_solver.py)
  (`account_opconfig` / `opconfig_geometry_table`)
- Tests: [`c4_min/test_opconfig.py`](../c4_min/test_opconfig.py) (validator +
  default round-trip), `test_qwen_fit_solver.py` (the 3-config geometry)
- Standalone verified cells: `examples/{clever_minparam_alu,lowprec_radix_alu}.py`

---

## 1. The four axes (per opcode)

| axis | values | meaning |
|---|---|---|
| **precision** | `int8` · `fp16` · `bf16` · `fp32` · `fp64` · `fp128` | the datapath dtype; its exact-integer ceiling bounds the radix |
| **radix** | any int `r >= 2` | each value/limb lives in `[0, r)`; nibble = radix 16 |
| **extraction** | `nibble` · `digit_extract` · `whole_value` | how a value is read out |
| **recurrence** | `unrolled` · `tied` | one stored layer per place, or one reused cell applied `depth` × |

**extraction** in detail:
- `nibble` — the production 4-bit lanes (radix 16); every value split into lanes so
  each stays tiny + fp32-exact, at the cost of a wide per-nibble lookup/corrector
  apparatus (the golden build).
- `digit_extract` — MSB-first, one digit/limb per layer, via the difference-min
  selector `logit_d = -|value − (d+0.5)|` (argmax = floor); the clever-minparam form.
- `whole_value` — hold the WHOLE operand/result in ONE high-precision scalar, no
  decomposition (the min-params corner).

**recurrence**: `tied` weight-ties the per-place cell → STORED layers shrink while
APPLIED depth is unchanged (the Universal-Transformer fold). `unrolled` stores each
place as a distinct layer.

### exact-integer CEILING per precision

The largest integer `M` with every integer in `[0, M]` representable EXACTLY
(`2^(mantissa_bits+1)` for an IEEE float; int8 keys off its signed magnitude):

| precision | ceiling | note |
|---|---|---|
| int8 | 2^7−1 = 127 | signed magnitude |
| bf16 | 2^8 = 256 | 7 stored mantissa bits |
| fp16 | 2^11 = 2048 | 10 stored |
| fp32 | 2^24 ≈ 16.7M | 23 stored |
| fp64 | 2^53 ≈ 9.0e15 | 52 stored |
| fp128 | 2^64 ≈ 1.8e19 | x86-64 longdouble, 64-bit effective mantissa |

---

## 2. The precision ↔ radix COUPLING table (the validator)

Every value/limb is in `[0, r)`. The op's **accumulator max** must fit the
precision's ceiling. The accMax formulas (PRECISION_RADIX_SURFACE.md §1):

| op class | accumulator max at radix r | constraint |
|---|---|---|
| ADD/SUB | `a_d + b_d + carry ≈ 2r` | `2r ≤ ceiling` |
| CMP | one limb difference `≈ r` | `r+1 ≤ ceiling` |
| MUL (schoolbook column) | `Σ a_i·b_j + carry` (exact worst-case peak) | `≤ ceiling` |
| DIV/MOD | trial `q_p·b_limb ≈ r²` | `r² ≤ ceiling` |

`validate_axes(op, axes)` rejects a radix whose accMax exceeds the precision
ceiling (skipped for `whole_value`, which never limb-decomposes). The
`max_safe_radix(op, precision)` helper reports the largest power-of-two radix that
fits — the coupling table, verified equal to PRECISION_RADIX_SURFACE.md §2:

```
prec    ADD/SUB    CMP    MUL  DIV/MOD      (as max-safe radix 2^k)
int8        2^5    2^6    2^1      2^3
bf16        2^7    2^7    2^2      2^4
fp16       2^10   2^10    2^4      2^5
fp32       2^23   2^23   2^11     2^12
fp64       2^52   2^52   2^26     2^26
```

Reading it: **MUL is the worst-coupled** (its column accumulator forces the
smallest radix / deepest stack); **ADD/CMP couple weakly** (`2r` / `r`, fat radix
even at int8); **DIV sits in between** (`r²`). Low precision → small ceiling → small
max-safe radix → more limbs → deeper. The unit test `test_max_safe_radix_matches_
surface_table` pins this whole table.

---

## 3. The two PARETO corners

Two genuinely opposite corners of the same exact-integer-arithmetic frontier
(PRECISION_RADIX_SURFACE.md §6):

| corner | config | params (per op) | depth | wall-clock | Doom realtime? |
|---|---|---:|---:|---|---|
| **MIN-PARAMS** `min_params_config()` | fp64 / **fp128** (MUL), `whole_value`, `tied` | **~4 scalars** | 1–20 | **slow** (fp64 = 0.02× fp32; MUL 603 ns, DIV 302 ns/lane) | render: marginal · raw: no |
| **MIN-WALLTIME** `min_walltime_config()` | **bf16 radix-16** (fp16 MUL), `digit_extract`, `tied` | 5–8 scalars | 8–16 (deeper) | **fast** (bf16 = 5.4× fp32; MUL 46 ns, DIV 23 ns/lane, **~13× the fp64 cell**) | **render: YES (35–105 fps)** · raw: no (~5.5 fps) |

- **min-params** trades **precision (up)** for **params (down)** — the smallest
  weight count (~4 shared scalars: `ln(10)` slope, `+1` softmax1 off-by-one, `+0.5`
  floor shift, `10.0` place base), but the card's slowest datapath (fp64 = 1/42 of
  fp32 on the A5000).
- **min-walltime** trades **depth (up) + params (slightly up)** for **wall-clock
  (down)** — every layer on the tensor cores at 5.4× fp32, netting ~13× faster than
  the fp64 cell on MUL/DIV. `bf16` is the measured sweet spot (int8's theoretical 4×
  is **not** realized on the A5000 → its extra depth makes it *slower*). radix 16
  keeps MUL depth 16 / DIV depth 8 while staying exact (DIV boundary r²=256=ceiling;
  MUL's 1904 column peak needs fp16's 2^11 ceiling, hence fp16 for MUL there).

---

## 4. Per-config fitter geometry (the FITTER wiring)

`qwen_fit_solver.account_opconfig(config)` sizes ANY `OpConfig` into
`(hidden, intermediate, stored_layers, applied_depth, params, fits-stock-0.5B)`.
The **DEFAULT** (nibble) config routes through the REAL nibble fit solver (golden
build geometry); the clever configs are accounted from the completed min-param
construction (a NARROW reused cell floored to the Qwen GQA head partition 896, cost
in APPLIED depth). MEASURED live via `opconfig_geometry_table`:

```
config                              prec           extract        recur  hidden  inter stored applied  params  fit0.5B
nibble-fp32-FULL (DEFAULT/golden)   fp32 (nibble)  nibble         unroll   3008   7920    123     123    9.6B   no(hidden)
clever-fp64-tied-FULL               fp64/fp128     whole_value    tied      896    896      6      20   25.9M   YES
bf16-radix16-tied-FULL              fp16/bf16      digit_extract  tied      896    896      6      16   25.9M   YES
```

- **nibble-fp32-FULL** (the golden build): hidden **3008 > 896** → **does NOT fit
  stock 0.5B** (binding = hidden; needs ~3B-class width). This is the production
  nibble build's failure mode: blocked by WIDTH.
- **clever-fp64-tied-FULL** (min-params): hidden **896 ≤ 896**, intermediate 896,
  stored ≈ 6 cells → **FITS the stock 0.5B WIDTH + SHAPE**. The cost is APPLIED
  depth (deepest single op = MUL fp128 digit-extract depth **20**; the *summed*
  program depth over the op stream exceeds 24 — the recurrence tradeoff: fits the
  CHECKPOINT, not a single stock forward).
- **bf16-radix16-tied-FULL** (min-walltime): same narrow-cell shape → **FITS 0.5B**;
  deepest single op = MUL radix-16 over the 64-bit product = **16** base-16 limbs.
  This is the min-walltime realtime config.

The inversion is the whole point: **nibble FULL is blocked by WIDTH** (hidden 3008),
while the **clever configs trade all of that width for DEPTH-in-time** and so fit the
stock 0.5B width where nibble cannot — the whole-value + recurrence levers move the
cost off the width axis entirely.

---

## 5. How to set the toggles

### (a) As a config object

```python
from c4_min import opconfig as OC

cfg = OC.DEFAULT                    # nibble/fp32 == golden 174ece66
cfg = OC.min_params_config()       # fp64/fp128 whole-value tied (~4 scalars)
cfg = OC.min_walltime_config()     # bf16 radix-16 digit-extract tied (realtime)

# a hand-rolled per-op override:
cfg = OC.OpConfig(
    base=OC.AxisConfig(precision="fp32", radix=16, extraction="nibble",
                       recurrence="unrolled"),
    overrides={"DIV": dict(precision="fp64", extraction="whole_value",
                           recurrence="tied")})
OC.validate(cfg)                   # raises OpConfigError on a radix overflow
axes = cfg.for_op("DIV")           # the resolved AxisConfig for one op
```

### (b) As `C4_OPCFG_*` env flags (the `C4_*` convention)

Grammar: `C4_OPCFG_<OP>_<AXIS> = <value>`, where `<OP>` is an op name (`DIV`,
`MUL`, …) or `ALL` (sets the base for every op; per-op flags win), and `<AXIS>` is
`PRECISION` / `RADIX` / `EXTRACTION` / `RECURRENCE`:

```
C4_OPCFG_DIV_PRECISION=fp64
C4_OPCFG_DIV_RADIX=16
C4_OPCFG_DIV_EXTRACTION=whole_value
C4_OPCFG_DIV_RECURRENCE=tied
C4_OPCFG_ALL_PRECISION=bf16          # every op -> bf16 (individual flags override)
```

```python
cfg = OC.resolve()                 # reads os.environ; validated
```

**With NO `C4_OPCFG_*` flag set, `resolve()` returns `DEFAULT`** → the build is
byte-identical to golden `174ece66`.

---

## 6. The low-precision speedup is a NARROW-VM lever

The measured GPU throughput (RTX A5000, PRECISION_RADIX_SURFACE.md §4/5):

- **fp64 is ~40× slower than fp32, ~225× slower than bf16** — the min-params corner
  pays for its 4 scalars with the slowest datapath.
- **bf16 is the fastest realized dtype (5.4× fp32)**; **int8's theoretical 4× is
  NOT realized** (~2.7× on this card, occupancy-bound).
- Deep-low-precision-on-tensor-cores beats the fp64 whole-value cell on wall-clock
  for every op — **~13× for MUL and DIV** — because the extra depth is more than
  paid for by the 5.4× tensor-core throughput.

**But the speedup is real ONLY if the VM is narrow and compute-bound.** Today's
**wide** VM is memory/occupancy-bound (~11.5% HBM peak), where fp32→bf16 nets
**~1.0×** — no fps gain. Low precision is a **narrow-VM lever, not a drop-in
speedup of the current build**. On a narrow deep-serial bf16 VM the min-walltime
config clears **35 fps on the render-reduced Doom frame (75–105 fps at 1 ALU-op/
step)** but **not** the raw 6.89 M-step frame (~5.5 fps — that needs algorithmic
step-count reduction, not a faster kernel).

---

## 7. Honest note — this is the config/fitter/docs layer only

- The DEFAULT config == golden `174ece66`; the resolver + fitter add **no build /
  weight change** to the golden path (verified via `python -m
  c4_min._fingerprint_build` → `174ece66`).
- The clever cells (whole-value fp64/fp128, low-precision radix-limb bf16/fp16) are
  verified **byte-EXACT standalone** (≥100k random 32-bit pairs each + hard edges,
  `examples/{clever_minparam_alu,lowprec_radix_alu}.py`) — but a **full, byte-exact,
  doom-running VM built from them DOES NOT EXIST.** Rebuilding the byte-exact doom
  VM with the clever ops is a substantial re-architecture (fetch/decode/opcode-select
  + PC/SP/BP writeback + the live-CAM memory model in the same narrow whole-value/
  radix-limb style + the exact-fp32 operand-halving), **explicitly out of scope
  here** and left as separate future work.
- The fitter's clever geometry is an ACCOUNTING of what the clever op-set implies
  (narrow reused cell, hidden floored to the Qwen head partition), computed
  alongside the REAL nibble-build solver numbers — not a baked model.
