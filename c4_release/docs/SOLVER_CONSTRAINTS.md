# SOLVER_CONSTRAINTS.md — the JOINT hard-constraint fit solver

The c4 network-size solver (`c4_min/qwen_fit_solver.py`) sizes a c4 VM geometry
`(n_layers, hidden, intermediate)` from an op-set + a per-op
`opconfig.AxisConfig` (`{precision, radix, extraction, recurrence}`). This
document covers the **joint hard-constraint** layer added on top: it checks
**precision, depth (max layers), width (max hidden + max intermediate), AND the
KV-cache size budget — all at once** — and, on infeasibility, names the
**binding constraint** and the axis to relax.

The solver is OFF the model build path (pure CPU accounting, no model
materialised), so it never moves the golden `174ece66`.

Related docs: [`TOGGLE_SCHEMA.md`](TOGGLE_SCHEMA.md) (the per-op toggle system +
honesty on recurrence), [`PRECISION_RADIX_SURFACE.md`](PRECISION_RADIX_SURFACE.md)
(the precision↔radix accumulator coupling), [`CLEVER_MINPARAM_ALU.md`](CLEVER_MINPARAM_ALU.md).

---

## 1. The joint-constraint API

### `FitConstraints` (dataclass)

The hard-constraint box. Every cap is optional (`None` == no cap on that axis).

```python
FitConstraints(
    precision:          Optional[str] = None,   # int8/fp16/bf16/fp32/fp64/fp128
    max_layers:         Optional[int] = None,   # cap on n_layers (STORED cells)
    max_hidden:         Optional[int] = None,   # cap on hidden_size
    max_intermediate:   Optional[int] = None,   # cap on FFN width
    kv_budget_bytes:    Optional[int] = None,   # cap on the KV-cache footprint
    seq_len:            int = 2048,             # KV sizing context
    batch:              int = 1,                # KV sizing context
    n_heads:            Optional[int] = None,   # KV head count (default: GQA KV heads)
    head_dim:           Optional[int] = None,   # KV head dim  (default: arch head_dim)
)
```

`precision` does double duty: it **validates** the config's radix (via
`opconfig.max_safe_radix` — a too-low precision at a given radix is an
INVALID-RADIX bind) **and** sizes the KV bytes/elem. `None` accepts the config's
own per-op precisions and sizes KV on the config's deepest op precision (e.g.
MUL@fp128).

`n_heads` / `head_dim` default to the arch's **key-value** head geometry — the
honest GQA KV-cache head count (Qwen2.5-0.5B caches **2 KV heads × 64 dim**, NOT
the 14 query heads).

### `opconfig.precision_bytes(precision) -> int`

Storage bytes/elem: `int8=1, fp16=2, bf16=2, fp32=4, fp64=8, fp128=16`. This is
the datapath/cache footprint of one scalar (distinct from
`PRECISION_CEILING`, the exact-integer representability bound).

### `solve_opconfig(config, constraints, code_size=24, arch=QWEN2_5_ARCH) -> JointFitResult`

Jointly solve/validate an `opconfig.OpConfig` against ALL of `constraints` at
once. Returns a `JointFitResult`:

```python
JointFitResult(
    fits:               bool,
    geometry:           OpConfigGeometry,   # required (n_layers, hidden, inter, ...)
    kv_bytes:           int,                 # sized on the APPLIED depth (see §2)
    n_layers_kv:        int,                 # == applied depth (what KV sizes on)
    binding_constraint: Optional[str],       # None if fits; else the tightest bind
    relax_axis:         Optional[str],       # the axis to relax (None if fits)
    slack:              ConstraintSlack,     # per-constraint (cap - required)
    constraints:        FitConstraints,
    notes:              str,
)
```

`ConstraintSlack` carries `layers / hidden / intermediate / kv_bytes` (each
`cap - required`; **negative == violated**) plus `radix_valid: bool` and the
required `precision`.

---

## 2. The KV-cache size formula (and the applied-depth-not-stored nuance)

```
KV_bytes = 2 (K+V) × n_layers_kv × n_heads × head_dim × seq_len × batch
           × precision_bytes(precision)
```

`kv_cache_bytes(n_layers_kv, n_heads, head_dim, seq_len, batch, precision)`
computes it directly.

### CRITICAL nuance — `n_layers_kv` is the APPLIED depth, in BOTH modes

The KV cache grows with the number of **layer-APPLICATIONS at inference**, NOT
the number of **distinct STORED cells**.

A LOOPED / Universal-Transformer stores few cells (~6) but **re-applies** them
`applied_depth` times per forward; each application writes its **own** K and V
into the cache (the loop **unrolls into the cache** at run time). Therefore:

```
n_layers_kv = APPLIED depth   (in BOTH standard-FF and looped/UT modes)
```

**Weight-tying shrinks the PARAMETER footprint (fewer distinct cells) but does
NOT shrink the KV footprint** — the KV footprint is set by how many times a cell
is *applied*, which is unchanged. A looped model and its unrolled twin with the
same applied depth pay **identical KV** but **different stored params**:

| config (clever-fp64)      | stored cells | applied depth (`n_layers_kv`) | KV bytes | stored params |
|---------------------------|:------------:|:-----------------------------:|:--------:|:-------------:|
| LOOPED / UT               | 6            | 20                            | *X*      | ~25.9M        |
| STANDARD-FF UNROLLED      | 51           | 20                            | *X*      | ~217M         |

(both at `seq_len=1024, batch=4, fp64`: `X = 2·20·2·64·1024·4·8 = 167,772,160`
bytes = 160 MiB). Test: `test_looped_vs_unrolled_same_kv_diff_params`.

The KV head count defaults to the **GQA key-value heads** (2 for 0.5B), the
honest cache head count — overridable via `FitConstraints.n_heads`.

---

## 3. Binding-constraint reporting

`solve_opconfig` checks each constraint and collects the violations. When
**several** are violated, the **binding** constraint is the one with the
**largest relative overshoot** (`required / cap`) — the most-binding-first. It is
reported alongside the **axis to relax**:

| binding constraint            | relax axis (surfaced)                                                         |
|-------------------------------|-------------------------------------------------------------------------------|
| `depth (max_layers)`          | raise `max_layers` / `recurrence=tied`+`looped_transformer` (fewer stored cells) / shallower extraction |
| `width (max_hidden)`          | raise `max_hidden` / narrower extraction                                       |
| `width (max_intermediate)`    | raise `max_intermediate` / smaller radix/LUT extraction                        |
| `kv_cache (kv_budget_bytes)`  | raise `kv_budget_bytes` / **lower precision** (fewer bytes/elem) / shallower applied depth / smaller `seq_len,batch` |
| `precision (invalid radix)`   | raise `precision` / lower `radix` (with the exact overflow note)               |

Example (DEFAULT nibble against `max_hidden=896, max_layers=24`): overshoots
hidden `3008/896 = 3.4×` AND depth `123/24 = 5.1×`; depth is the larger relative
overshoot → `binding_constraint == "depth (max_layers)"`
(`test_multiple_binds_reports_most_binding_first`).

### A worked joint-constraint matrix

Box: `max_layers=24, max_hidden=896, max_intermediate=4864,
kv_budget_bytes=256MiB, seq_len=2048, batch=1`.

| config                        | n_layers | hidden | inter | appliedKV | KV      | fits? | binds                |
|-------------------------------|:--------:|:------:|:-----:|:---------:|:-------:|:-----:|----------------------|
| nibble-fp32 (DEFAULT)         | 123      | 3008   | 7920  | 123       | 257.9M  | no    | `depth (max_layers)` |
| clever-fp64 LOOPED/UT         | 6        | 896    | 896   | 20        | 167.8M  | **yes** | —                  |
| clever-fp64 std-FF UNROLLED   | 51       | 896    | 896   | 20        | 167.8M  | no    | `depth (max_layers)` |
| bf16-r16 LOOPED/UT            | 6        | 896    | 896   | 16        | 16.8M   | **yes** | —                  |
| bf16-r16 std-FF UNROLLED      | 42       | 896    | 896   | 16        | 16.8M   | no    | `depth (max_layers)` |

The DEFAULT nibble build is blocked on WIDTH+DEPTH (hidden 3008, depth 123).
The clever configs fit the 0.5B WIDTH (hidden 896) but only the **LOOPED/UT**
variants fit DEPTH (few stored cells); the honest **standard-feed-forward**
version must UNROLL (51 / 42 distinct stored layers) and binds on
`depth (max_layers)`. Note the LOOPED and UNROLLED twins pay the **same
appliedKV** (20 / 16) — the loop unrolls into the cache.

> Honesty (from `TOGGLE_SCHEMA.md` §honesty): `recurrence='tied'` is ONLY legal
> when `looped_transformer=True`. A stock feed-forward Qwen2 has no loop, so it
> must UNROLL: `n_layers = SUMMED unrolled per-op depth` (clever-fp64-FULL = 51
> layers). The looped/UT param win is a UT-checkpoint claim, NOT a stock
> feed-forward claim.

---

## 4. The precision ↔ KV ↔ depth coupling — `min_kv_precision`

Dropping precision is **not** a free KV win. It couples three ways:

* **precision ↓ → bytes/elem ↓** in the KV cache (int8=1 vs fp64=8), BUT
* **precision ↓ → radix ceiling ↓** (`opconfig.max_safe_radix`, the exact-int
  bound) → **more digits/limbs per value** → **more applied depth** → **more
  layer-applications** → **more KV** (and more stored params in an unrolled model).

So lower precision trades bytes/elem **against** applied depth. `min_kv_precision`
searches the precisions, builds the **honest** geometry for each (radix pinned to
that precision's max-safe value so the applied depth reflects the ceiling), and
ranks by total inference-memory footprint (`objective="total"`: stored-param
bytes + KV bytes) or by pure KV (`objective="kv"`).

```python
min_kv_precision(constraints, mode="natural", objective="total", ...) -> KvPrecisionResult
```

`mode` picks the model MODE per precision:
* `"natural"` (default) — each precision at its **honest natural mode**: the
  whole-value precisions (fp64/fp128) as **LOOPED/UT** (the min-params corner:
  FEW stored cells, big bytes/elem), the finite-radix precisions as
  **STANDARD-FF UNROLLED** (MANY distinct stored layers, small bytes/elem). This
  is the fp64(few-layers) vs int8(many-layers) tradeoff.
* `"looped"` / `"unrolled"` — force one mode for every precision.

### The worked example: fp64 (few layers, big bytes/elem) vs int8 (many layers, small bytes/elem)

Because KV scales with `seq_len·batch` but stored params do NOT, the
total-footprint winner **flips with the seq_len/batch regime**.

**TINY seq/batch (`seq_len=4, batch=1`) → fp64 wins.** KV is negligible, so the
(seq-independent) stored-param depth dominates, and fp64's 6 FEW-LAYER cells beat
int8's 54 MANY-LAYER stored layers:

```
  prec bytes/e radixDIV applied stored        KV    params     total
  fp64       8    whole      20      6    163.8K    207.6M    207.8M  <== WIN
  int8       1        8      22     54     22.5K    229.7M    229.7M
  fp16       2       32      13     37     26.6K    315.1M    315.1M
  bf16       2       16      16     42     32.8K    357.6M    357.6M
  fp32       4     4096       6     22     24.6K      1.1B      1.1B
```

**HUGE seq/batch (`seq_len=16384, batch=128`) → int8 wins.** KV dominates total
memory, so int8's tiny 1 byte/elem beats fp64's 8 — despite int8's deeper applied
depth (22 vs 20) and 54 stored layers:

```
  prec bytes/e radixDIV applied stored        KV    params     total
  int8       1        8      22     54     11.8B    229.7M     12.0B  <== WIN
  fp32       4     4096       6     22     12.9B      1.1B     14.0B
  fp16       2       32      13     37     14.0B    315.1M     14.3B
  bf16       2       16      16     42     17.2B    357.6M     17.5B
  fp64       8    whole      20      6     85.9B    207.6M     86.1B
```

Read the coupling off the columns: int8's `radixDIV=8` (far below fp32's 4096)
forces `applied=22` (more limbs) — the **depth cost** of dropping precision.
fp64 uses `whole` value (one scalar, `applied=20` fixed) at 8 bytes/elem. On
**pure KV** (`objective="kv"`, ranks by `applied × bytes/elem`) int8 always wins
(`22·1=22` < fp64 `20·8=160`); the flip above is a **total-memory** flip driven
by the stored-param term.

Tests: `test_min_kv_precision_fp64_wins_at_small_seq_batch`,
`test_min_kv_precision_int8_wins_at_large_seq_batch`,
`test_min_kv_precision_surfaces_the_depth_coupling`,
`test_min_kv_precision_pure_kv_objective_ranks_by_depth_times_bytes`.

---

## 5. Golden

The whole joint-constraint solver is CPU accounting off the build path. The
golden `state_dict` fingerprint is unchanged:

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=<repo>/c4_release python -m c4_min._fingerprint_build
# FINGERPRINT 174ece66...
```
