# Depth via FORWARDS-PER-STEP — the layers-OR-forwards depth lever

Companion to `docs/BLOG_NOTE_CLEVER_MINPARAM_VM.md` §5 ("Getting depth from
forwards-per-step — the vanilla resolution") and §7 (the joint solver). Module:
[`c4_min/forwards_per_step.py`](../c4_min/forwards_per_step.py); tests:
[`c4_min/test_forwards_per_step.py`](../c4_min/test_forwards_per_step.py).

This module is **self-contained and additive** — it supplies the `(L, F)` split
math and does NOT edit the network-size solver, so it merges cleanly with
concurrent work on `qwen_fit_solver.py` / `opconfig.py`. It never touches the build
path, so golden `174ece66` is unchanged.

## The split

The clever-fp64 construction realizes an effective digit-extraction depth `D`
(summed per-op digit-depth — `D ≈ 51` for the full ISA, `10` for DIV, `20` for MUL,
`11` for ADD/SUB) as a chain of `D` layer-applications. That depth can live in
**layers** or in **forwards/tokens**:

| lever | split | network | tokens/step | KV | vanilla? |
|---|---|---|--:|---|---|
| **distinct-layers (deep)** | `L = D`, `F = 1` | deep (`D` layers) | few | small seq | too deep for stock 24 |
| **forwards-per-step (shallow)** | `L` small (1–4), `F = ceil(D/L)` | **shallow — fits stock 24** | **+F/step** | **seq × F** | ✓ standard AR loop |

`L × F ≥ D` is the constraint; `F = ceil(D/L)` is the minimal forwards at a given
`L`. Forwards-per-step is the ordinary autoregressive loop (re-invoking the same
weights per token, which every LM already does) — genuinely *vanilla*, not a
tied-layer Universal Transformer.

## Compute is conserved

Every split does `L × F ≈ D` **layer-applications per VM step** (`D ≤ L·F ≤ D +
(L−1)`, the only slack being the `ceil` rounding). Forwards-per-step does **not**
cut FLOPs — it re-books the same `~D` layer-applications from physical-depth into
sequence-length. Verified for every split by `compute_conserved` /
`conserved_report` (test `test_compute_conserved_every_split_D51`).

## KV impact — and the honest first-order result

```
KV_bytes = 2 (K+V) × n_layers × n_heads × head_dim × seq_len × batch × bytes(precision)
```
(identical to `qwen_fit_solver.kv_cache_bytes`; cross-checked in
`test_kv_bytes_matches_solver_formula`.)

Forwards-per-step adds `F` tokens/step, so over a run the sequence grows by `~F ×
steps` → `effective_seq_len = seq_len × F`. The deep split keeps `seq_len` small but
`n_layers = D` large; the shallow split keeps `n_layers` small but `seq_len` large.

**The honest coupling:** both splits multiply the *same* `n_layers × seq_len` KV
core — deep = `D · base_seq`, shallow = `(L·F) · base_seq`. At an **exact-divisor**
split (`L·F == D`: 51×1 / 17×3 / 3×17 / 1×51) the pure deep and shallow corners
**tie on KV bytes**. So KV alone does *not* separate deep from shallow; the strict
differences are:

- **deep wins `launch_count`** (~`F`): 1 launch-chain + 1 KV re-read per VM step vs
  `F` of them → **min-latency**.
- **shallow wins `n_layers`**: fewest physical layers → fewest stored params → the
  **vanilla-fit** corner.

KV only rises *above* the tie at non-exact splits (`L·F > D`, where the `ceil`
rounding inflates the sequence a little). This is exactly why the solver must cost
each split explicitly rather than assume "deep = small KV, shallow = big KV".

## Worked table — D = 51 (full ISA)

`seq_len = 2048, batch = 1, precision = fp64, KV heads = 2, head_dim = 64.`
Selected splits (the full enumeration is 51 rows):

| L (layers) | F (fwd/step) | L·F | tokens/step | eff_seq | kv_bytes | launch | fits stock-24 |
|--:|--:|--:|--:|--:|--:|--:|:--:|
| **51** | **1** | 51 | 1 | 2048 | 213,909,504 | 1 | **no** (51 > 24) ← DEEP / min-latency |
| 26 | 2 | 52 | 2 | 4096 | 218,103,808 | 2 | no |
| 25 | 3 | 75 | 3 | 6144 | 314,572,800 | 3 | no |
| **24** | **3** | 72 | 3 | 6144 | 301,989,888 | 3 | **yes** ← fits-24 boundary |
| 17 | 3 | 51 | 3 | 6144 | 213,909,504 | 3 | yes (exact-divisor: KV = deep) |
| 13 | 4 | 52 | 4 | 8192 | 218,103,808 | 4 | yes |
| 4 | 13 | 52 | 13 | 26624 | 218,103,808 | 13 | yes |
| 3 | 17 | 51 | 17 | 34816 | 213,909,504 | 17 | yes (exact-divisor) |
| **1** | **51** | 51 | 51 | 104448 | 213,909,504 | 51 | **yes** ← SHALLOW / vanilla-fit / min-params |

- **Distinct-layers (L=51)** does **NOT** fit stock-24 → *fits width, not depth*.
- **Forwards-per-step (L≤24)** fits a stock vanilla 0.5B; the shallowest fit
  (L=1, F=51) is the min-stored-params corner.
- The min-KV value (213,909,504) is shared by all exact-divisor corners
  (51×1, 17×3, 3×17, 1×51) — the deep and shallow ends **tie** on KV; deep wins
  launches, shallow wins stored layers.

## Worked table — D = 10 (DIV)

Full enumeration (10 rows), same context:

| L | F | L·F | tokens/step | eff_seq | kv_bytes | launch | fits stock-24 |
|--:|--:|--:|--:|--:|--:|--:|:--:|
| **10** | **1** | 10 | 1 | 2048 | 41,943,040 | 1 | **yes** ← DEEP (D=10 ≤ 24) |
| 5 | 2 | 10 | 2 | 4096 | 41,943,040 | 2 | yes (exact-divisor) |
| 4 | 3 | 12 | 3 | 6144 | 50,331,648 | 3 | yes |
| 2 | 5 | 10 | 5 | 10240 | 41,943,040 | 5 | yes (exact-divisor) |
| **1** | **10** | 10 | 10 | 20480 | 41,943,040 | 10 | **yes** ← SHALLOW / vanilla-fit |

DIV's `D = 10 ≤ 24`, so even the **distinct-layers** stack fits stock-24 — DIV never
*needs* forwards-per-step to fit a vanilla model (only the full-ISA `D ≈ 51` does).

## API

```python
from c4_min import forwards_per_step as FPS

# enumerate + cost every (L, F) split of an effective depth D
r = FPS.depth_realizations(D=51, max_layers=24, seq_len=2048, batch=1,
                           precision="fp64", n_heads=2, head_dim=64)
r.deepest          # DepthSplit: L=max, F=1     (min-latency corner)
r.shallowest       # DepthSplit: L=min, F=D     (min-params corner)
r.min_kv()         # min-KV split (deep at exact-divisor)
r.vanilla_fit(24)  # shallowest split with L <= 24 (best vanilla-fit)
r.pareto           # non-dominated frontier over (kv_bytes, launch_count, n_layers)
r.fits_layers      # splits whose L <= max_layers
r.table()          # rendered worked table

# the §5 headline: shallow fits stock-24, deep (D=51) does not
v = FPS.fits_stock_vanilla(D=51, max_layers=24)
v.distinct_fits    # False  (L=51 > 24)
v.shallow_fits     # True   (L=1..24 with F forwards)
print(v.summary())

# the compute-conserved invariant  L*F ~= D
FPS.compute_conserved(D=51, L=4, F=13)   # True (52 covers 51, slack < L)

# KV formula (== qwen_fit_solver.kv_cache_bytes)
FPS.kv_bytes(n_layers, n_heads, head_dim, seq_len, batch, precision)
```

### `DepthSplit` fields

`D`, `n_layers` (L), `forwards_per_step` (F), `tokens_added_per_step` (=F),
`layer_applications` (=L·F, the conserved per-step compute), `effective_seq_len`
(=seq_len·F), `kv_bytes`, `launch_count` (~F), `fits_layers` (L ≤ max_layers).

## Integration with the solver (`qwen_fit_solver.py`)

The concurrent solver owns `FitConstraints` (fields `precision, max_layers,
max_hidden, max_intermediate, kv_budget_bytes, seq_len, batch, n_heads,
head_dim`), the `kv_cache_bytes(...)` formula, `min_kv_precision`, and
`account_opconfig` (whose `summed_unrolled_depth` yields the effective `D`). This
module is the **additive supplier** of the layers↔forwards split math.

`FitConstraints` gains **one** field:

```python
forwards_per_step: int = 1     # 1 == distinct-layers (deep); >1 == shallow
```

and the solver applies this module to the accounted geometry with the **exact
one-line call**:

```python
geom = apply_forwards_per_step(geom, constraints.forwards_per_step)
```

where `geom` is the `account_opconfig` geometry as a dict (carrying `n_layers` = the
effective depth `D`, plus the KV context). `apply_forwards_per_step` returns the dict
updated with `n_layers_physical = ceil(D/F)`, `forwards_per_step`,
`effective_seq_len = seq_len × F`, `layer_applications`, and re-sized `kv_bytes`.
The solver then checks `n_layers_physical` against `max_layers` (now met by the
shallow split) and `kv_bytes` against `kv_budget_bytes` (now paid on the F-inflated
sequence). `F = 1` is the identity (distinct-layers) split.

## Tests + golden

`c4_min/test_forwards_per_step.py` — 21 tests, all pass (CPU, no model built):
`(L,F)` enumeration; compute-conserved invariant; KV grows with F; shallow fits
stock-24 while deep (D=51) does not; the min-KV(deep) vs vanilla-fit(shallow) Pareto
example for D=51 and D=10; the `apply_forwards_per_step` hook.

Golden fingerprint `174ece66` is unchanged (this module is never on the build path).
