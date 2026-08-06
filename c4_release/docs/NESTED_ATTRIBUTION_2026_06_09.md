# `nested_*` 0 / 50 attribution (2026-06-09)

Date: 2026-06-09. Measured on this worktree at HEAD `98a34d08`
(`docs(1096): mul_ sub-cluster breakdown + rec_factorial partial sample`).

Driver:

```
cd c4_release
python -m pytest tests/test_suite_1096_pure_neural_pytest.py \
    -k "nested_" --runxfail --tb=line -q
```

Result: **50 failed, 1048 deselected in 273.35 s** — uniform 0 / 50 pass,
matching the `1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md` baseline.

## What "nested" means in this cluster

Per `tests/test_suite_1000.py` lines 461-480, the nested cluster is two
sub-clusters of **nested function calls** (not nested loops, not nested
if/else):

- **`nested_quad_*`** (25 tests, IDs 0950-0974):
  ```c
  int double_it(int x) { return x * 2; }
  int quad(int x)      { return double_it(double_it(x)); }
  int main()           { return quad(N); }
  ```
  Two levels of function call, plus a MUL inside each callee. Three
  frames total (main → quad → double_it).

- **`nested_sumsq_*`** (25 tests, IDs 0975-0999):
  ```c
  int square(int x)              { return x * x; }
  int sum_squares(int a, int b)  { return square(a) + square(b); }
  int main()                     { return sum_squares(A, B); }
  ```
  One sequential pair of nested calls plus an ADD at the outer frame.
  Three frames at peak, two simultaneously live (main → sum_squares →
  square, popped, then square again).

## Failure signature

Sampled (`-v --tb=short`):

| Test                              | Expected | Neural got    | Hex      | Comment |
|---|---:|---:|---:|---|
| `nested_quad_0  quad(10)`         | 40       | 10            | 0x000A   | input `n` passed through unmodified |
| `nested_quad_1  quad(8)`          | 32       | 8             | 0x0008   | input `n` passed through unmodified |
| `nested_quad_2  quad(10)`         | 40       | 10            | 0x000A   | input `n` passed through unmodified |
| `nested_sumsq_0  2^2 + 2^2`       | 8        | 65282         | 0xFF02   | low byte = `b`, high byte = 0xFF |
| `nested_sumsq_1  10^2 + 10^2`     | 200      | 65290         | 0xFF0A   | low byte = `b`, high byte = 0xFF |
| `nested_sumsq_2  9^2 + 7^2`       | 130      | 65287         | 0xFF07   | low byte = `b`, high byte = 0xFF |
| `nested_sumsq_3  8^2 + 10^2`      | 164      | 65290         | 0xFF0A   | low byte = `b`, high byte = 0xFF |

Two distinct sub-patterns:

1. **`nested_quad_*`**: the model emits the original `n` — identical
   pattern to `rec_factorial` (`5! → 5`, `8! → 8`) and to the
   `func_mul` outlier family. The first `double_it(x)` either never
   commits a MUL result, or LEV writes back the pre-MUL `STACK0` (`n`)
   into `AX_LO` rather than `2*n`.

2. **`nested_sumsq_*`**: the model emits `0xFF | b_low_byte`, i.e. the
   *second* argument's low byte ORed with `0xFF` in the high byte. The
   first `square(a)` result is lost, the second `square(b)` returns
   only `b` (same pass-through pattern as `quad`), and the high byte
   gets pinned to `0xFF` — strongly suggestive of an unmasked sign-bit
   write into `AX_CARRY_HI` on the second LEV, propagated to
   `OUTPUT_HI` via `lev_ax_byte_hi_0` / `lev_ax_passthrough_hi_0`.

The `0xFF | b` signature is structurally identical to the
`gcd_* → 0xFFE8 = 0xFF | 0xE8 = −24 s16`
sentinel documented in `1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md` §
Surprise findings #1.

## dim-flow audit (REG_AX_LO / OUTPUT_LO / OUTPUT_HI)

`AX_FULL_LO` / `AX_FULL_HI` (per `tools/dim_flow_audit.py`):

```
$ python tools/dim_flow_audit.py AX_FULL_LO --no-zeroing
WRITERS: L03 carry_forward_attn head 5 only.
READERS: L19 layer15_alu_high_byte_relay head 8 (gates: OP_MUL, OP_SHL).
OK: writes start at L03, reads start at L19.
```

`OUTPUT_LO` and `OUTPUT_HI` (relevant LEV writers):

```
$ python tools/dim_flow_audit.py OUTPUT_LO --writers-only | grep lev
  function_call_weights::lev_ax_passthrough_lo_N
      gates: OP_LEV+0*1, MARK_AX+0*1, MARK_PC+0*-15, IS_BYTE+0*-10,
             GATE:AX_CARRY_LO+N*1, thr=4
  function_call_weights::lev_ax_byte_lo_N
      gates: OP_LEV+0*1, IS_BYTE+0*1, H1+1*1, MARK_AX+0*-15,
             GATE:AX_CARRY_LO+N*1, thr=9
```

(symmetric `lev_ax_passthrough_hi_N` / `lev_ax_byte_hi_N` write
`OUTPUT_HI`.)

Two key observations:

1. **AX_FULL_LO is never read at L16/LEV.** The LEV-side copy reads
   `AX_CARRY_LO` and `AX_CARRY_HI` (the carry-stage registers), not
   the resolved `AX_FULL_*`. The MUL result from `double_it` /
   `square` is materialized into `AX_FULL_LO` at L19 — but the LEV
   path at the callee frame reads from `AX_CARRY_*`, so any MUL whose
   carry hasn't been committed back into `AX_CARRY_LO` is lost on
   return. This is the **same surface as Bug #33 (post-LEV AX
   corruption)** identified for `gcd_*` / `absdiff_*` / `rec_factorial`.

2. **`OUTPUT_HI = 0xFF` is consistent with the `lev_ax_byte_hi_N` byte
   gate firing unconditionally on a stale residual.** The
   `lev_ax_byte_hi_N` rule fires on `OP_LEV ∧ IS_BYTE ∧ H1+1`. If
   `AX_CARRY_HI` carries the MUL-stage upper byte from `square(b)` and
   that byte happens to be `0xFF` (e.g. a sign-extended residual from
   the failed pre-LEV mask), the write propagates `0xFF` to
   `OUTPUT_HI`. The pattern is identical across all sumsq tests
   regardless of `a`, `b` values — confirming the high byte is not a
   product of the multiplication but a stuck residual.

## Attribution: shared root with `gcd_*` / `rec_*` (Bug #33)

The 0/50 `nested_*` cluster is **not independent**. Three lines of
evidence converge on **Bug #33: Post-LEV AX corruption** (per
`BUG_CATALOG.md` Bug #33 and the closeout-plan **Wave C7 — post-LEV
AX recovery**):

1. **`nested_quad` ≡ `rec_factorial` pass-through pattern.** Both
   emit the input `n` instead of the recursive/nested product. Per
   `1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md` § Surprise findings #8,
   the rec_factorial pattern is "the same as the gcd 65512 sentinel:
   post-LEV the AX register loses the intermediate multiplication
   result." `nested_quad` exhibits this without the recursion
   complexity — pure nested calls, MUL inside the leaf.

2. **`nested_sumsq` ≡ `gcd_*` sentinel structure.** Both emit
   `0xFF | <low_byte>` — gcd produces `0xFFE8` (= −24 s16), sumsq
   produces `0xFFNN` where NN matches the second argument's low byte.
   Per `BUG_CATALOG.md` Bug #33 (and `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`),
   this is the L16 LEV routing miscopy — `STACK0_byte0 → AX_byte0` is
   either malformed or `AX_byte1` is not cleared on the byte-sized
   return value, allowing a stale high byte to leak.

3. **The `nested_quad_*` attribution to Bug #26 / Bug #33 was already
   recorded** in `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md` lines
   75-77 ("`nested_quad_*` rows now first-fatal on the post-LEV
   `step6:AX_byte0` / `step6:STACK0_byte0` slots that motivate bug
   #33"). This doc extends the attribution to `nested_sumsq` based on
   the 0xFF-leak signature and the missing AX_FULL_LO → LEV reader
   path.

### Specifically NOT the L10 PSH addr0_e0 surface

The L10 `PSH addr0=0xE0` OP_ENT-guard miss (per memory note
`project_l10_psh_addr_ent_bug.md`, `l10_ops.py:3888-3927`) is a known
upstream contributor for `var_*` / `func_add_*` / `func_identity_*`.
`nested_*` does cross PSH on every nested call entry, so a partial
contribution is plausible, but:

- `nested_quad`'s pass-through `quad(10) → 10` requires the LEV-side
  AX recovery to read the *original* `n` from `STACK0_byte0`. That
  pre-MUL `n` only survives to the LEV step if the PSH stage did push
  it correctly. If L10 PSH addr0_e0 were the dominant root cause, we
  would expect `var_mul`-style `→ 1` outputs, not `→ n`.

- The Bug #33 closeout doc explicitly bundles `absdiff_* +
  nested_quad_*` "downstream of the same L10 PSH addr0_e0 OP_ENT-guard
  miss" — meaning the L10 entry-side surface and the L16 LEV-side
  surface are *co-attributed* to Bug #33 as a single corrective
  wave. Fixing Wave C7 (post-LEV AX) is expected to close both
  partially.

## Recommended fix surface (Wave C7 candidate)

Per `BUG_CATALOG.md` Bug #33 cluster suspects and the
dim_flow_audit findings:

1. **`layer16_lev_routing` weight cluster** — `l16_lev_set_output_lo0_byte0`,
   `l16_lev_clear_output_lo10_byte0`, `l16_lev_*_ax_*`. The
   `lev_ax_byte_hi_N` rule must mask `OUTPUT_HI` to zero when the
   return value's `AX_CARRY_HI` is zero, not propagate the stale residual.

2. **`layer6_ax_load`** — callee return-value materialization
   fallback. Should route `AX_FULL_LO` (the MUL-resolved register) to
   `AX_CARRY_LO` before LEV reads it, not the pre-MUL `STACK0_byte0`.

3. **`tail_bit32_result_correction`** — post-LEV AX byte0 patch. Add a
   high-byte mask conditional on the LEV-stage IS_BYTE flag.

Expected closure: **`nested_quad_*` (25) + `nested_sumsq_*` (25) +
collateral on `rec_factorial` (≥ 20) + `gcd_*` (50) + `absdiff_*` (25)
= ≥ 145 rows on the same wave**. Largest single corrective surface at
HEAD.

## Smoke

```
$ pytest tests/test_smoke.py --tb=no -q
============ 6 failed, 45 passed, 1 deselected in <wall> ===========
```

Smoke is **45 / 51 PASS** — exactly the closeout-plan threshold,
unchanged by this doc-only delta. Six failures are the documented
Wave S1 memory cluster + S2 LEA.

## Cross-references

- [`1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md`](archive/1096_CUMULATIVE_PASS_COUNTS_2026_06_07.md)
  — base 50/50 fail measurement.
- [`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`](archive/EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md)
  — prior `nested_quad` attribution under Bug #26/#33; Wave C5 (BZ
  re-fire) + Wave C7 (post-LEV AX) definitions.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) — Bug #26 (`absdiff_` /
  `nested_quad_` dead categories), Bug #33 (post-LEV AX corruption,
  primary attribution for this cluster).
- [`1096_LOOP_RECURSION_SAMPLE_2026_06_07.md`](archive/1096_LOOP_RECURSION_SAMPLE_2026_06_07.md)
  — rec_factorial pass-through pattern (`n! → n`), structurally
  identical to `nested_quad(n) → n`.
- Memory note `project_l10_psh_addr_ent_bug.md` — L10 PSH addr0_e0
  OP_ENT guard; co-attributed but secondary for this cluster.
