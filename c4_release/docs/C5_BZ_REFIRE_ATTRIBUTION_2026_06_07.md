# Wave C5 — BZ branch-target re-fire attribution

Date: 2026-06-07 (H6 follow-up).
Base ref: `main` HEAD `674f9e21` (`docs(h6): correct bool_and survivor
id from _22 to _21`).
Task: attribute the "branch re-fire" surface called out in
`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md` Recommendation 1 and
`LOOP_RECUR_ATTRIBUTION_2026_06_07.md` §Step 4.

## TL;DR

`l6_ops.py:4509-4574` `_post_l9_bz_pc_override_rules` (and the BNZ
sibling at 4577-4640) bake the BZ-taken PC override into a post-L9
FFN block. The override has a **step-0 guard** (`HAS_SE +10` plus a
threshold bump) that blocks the rule on step 0, but has **no guard
against re-firing on subsequent BZ instances inside a loop or
function body**. The post-fix runner (`44709e13` retired the
Python BZ runner override; `763ff75c` made `_append_pc_byte0_imm_to_byte_addr_rules`
canonical at `l6_ops.py:440`) routes every BZ through the same
declarative path, so the moment a program contains two BZ
instructions executed in sequence the second one fires with stale
OUTPUT_LO.*.-1 residuals from the previous step.

| Cluster | Failing tests | Footprint estimate |
|---|---:|---:|
| `edge_if_hundred` / `edge_loop_never` / `edge_loop_once_skip` | 3 | 3 |
| `absdiff_*` | ~25/25 | 25 |
| `loop_sum_*`, `loop_countdown_*`, `loop_mul_*`, `loop_pow2_*` (Wave J overlap) | ~0/100 sampled in `LOOP_RECUR_ATTRIBUTION_2026_06_07.md` | 0-100 (overlaps Wave J) |
| **Direct C5 footprint (non-Wave-J)** | — | **~28 rows** |
| **Combined footprint with Wave J** | — | **~128 rows** |

## Bug shape

### Override rule structure (`l6_ops.py:4509-4574`)

```
target_conditions = (
    ("MARK_PC", 1.0),
    ("OP_BZ", 0.2),
    ("CMP+4", 1.0),
    ("CMP+5", 1.0),
    ("IS_BYTE", -10.0),
    ("HAS_SE", 10.0),         # step-0 guard
    ("MARK_STACK0", -10.0),
)
threshold = 3.5 + 10.0        # +10 from step-0 guard
```

The rule writes `OUTPUT_LO[target_lo]` and `OUTPUT_HI_THIS_STEP[target_hi]`
via `_append_pc_byte0_imm_to_byte_addr_rules` (`l6_ops.py:440-485`).

### Step-0 guard works; step-N≥2 guard missing

`HAS_SE` (`dim_registry_dynamic.py:129`, pin 137) is **0 on step 0,
1 on every step ≥ 1**. The +10 weight + +10 threshold bump makes the
rule dead on step 0 (`HAS_SE = 0` → contribution = 0 vs threshold
13.5) and live on every step ≥ 1 (`HAS_SE = 1` → contribution = 10
vs base threshold 3.5 → fires whenever the other 4 positive terms
sum ≥ 3.5). This is the fix landed in `L34_FFN_ATTRIBUTION_2026_06_04.md`
for the `var_simple` / `var_three` 0x00 leak at step 0.

The guard is **not BZ-instance-specific**. On step N ≥ 2 (the second
BZ inside a loop body), every gate term is reset fresh from the new
PC token:
- `MARK_PC = 1` — fresh PC marker on this step.
- `OP_BZ = 1` — opcode decoded from the new PC instruction.
- `CMP+4`, `CMP+5` — fresh from the post-L9 ALU (same-step variant by
  design; this is the whole point of running post-L9).
- `IS_BYTE = 0` — opcode token, not a byte.
- `MARK_STACK0 = 0` — PC marker, not a STACK0 marker.
- `HAS_SE = 1` — step is not step 0.

All seven gate terms align. The override fires with the **current
step's `FETCH_LO`** as source. If the loop body's preceding `JMP loop`
overwrote `FETCH_LO` with the loop-back target's nibble before the
new BZ re-fetch overwrites it back — and if the L10 FETCH_LO writer
runs *before* this override in layer order — the override will read
the wrong nibble.

### Cancel band reads `OUTPUT_LO.*.-1` (prev-step alias)

```
("OUTPUT_LO.*.-1", ...)      # cancel band gate
```

The OUTPUT_LO cancel gate is **intentionally cross-step** (per the
docstring at `l6_ops.py:4538-4540`) to subtract the previous step's
residual. On step 0 of a fresh BZ that's correct: clear the
default-increment PC. On step N ≥ 2 inside a loop where the previous
step was a `JMP` or arithmetic op, the cancel band subtracts an
**unrelated** OUTPUT_LO residual — which can leave dim values
non-zero in lanes the new target writes did not touch, producing the
`0xFF0A` / `0xFFE8` / `0xFF00..0xFF0A` symptoms catalogued in
`1096_LOOP_RECURSION_SAMPLE_2026_06_07.md`.

## Cross-cluster footprint

Per H6 `EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md` TL;DR table:

- `edge_if_hundred` (`if (100) return 1; return 0;`) — single BZ but
  on **step ≥ 2** because `ENT 0; IMM 100; BZ +else` puts the BZ at
  step 2. Neural emits the `else` literal (= 0) because the override
  re-fires with `IMM 100`'s leftover FETCH_LO instead of the BZ
  target offset.
- `edge_loop_never` (`while (i < 0) ...`) — first BZ inside `for/while`
  loop entry; same step-≥-2 pattern as `if_hundred`.
- `edge_loop_once_skip` (`while (i > 1) ...`) — same.
- `absdiff_*` (25/25) — `if (a > b) return a - b; return b - a;`
  inside `abs_diff()`. BZ at step 13 of the `abs_diff` body, well
  past step 0. Symptom: PC walks past bytecode end (per
  `ABSDIFF_BZ_REDIRECT_BUG.md` § Trace).
- Wave J loop family (`loop_sum_*`, `loop_countdown_*`, `loop_mul_*`,
  `loop_pow2_*`, 100 tests, ~0/100) — the BZ inside the loop body
  fires once per iteration; every iteration after the first has the
  same re-fire pathology because `HAS_SE = 1` from iteration 1
  onwards.

`bool_and_21` is NOT a member of C5: it fails on the
**var-MEM_addr1 baseline** (Wave C2, per H6 doc §TL;DR row 3), not
on BZ re-fire. Its outer-BZ-not-taken nested-`if` traverses the
override correctly because both inner predicates are TRUE so the
BZ-taken band is dead.

## Why this is hard to fix one-rule

The natural fix is a residual that distinguishes "first BZ in
program" from "second BZ in program" — e.g. a `BZ_FRESH` lifecycle
bit, or a step counter `STEP_GTE_2` flag. Neither exists:

1. **No step counter residual.** `HAS_SE` is the only step-discriminating
   flag and it saturates at step 1. Adding a true step counter
   requires a new dim, a new attention head to read it, and a
   per-step writer in the runner — Phase 7.B+ scope, not a single
   rule.
2. **No "branch-instance fresh" lifecycle bit.** Bug #14 (catalog)
   explicitly called out this gap: "No `IN_STEP_FRESH` lifecycle bit
   → tail rules can't distinguish fresh vs. residual". The same
   missing lifecycle bit applies here.
3. **Can't gate on the rule's own writes.** The override is the
   sole writer of `OUTPUT_LO[target]` on BZ-taken steps; gating it
   on its own previous-step output would deadlock on iteration 2.
4. **`OUTPUT_LO.*.-1` cancel gate is structurally wrong on
   re-fire.** The cancel band assumes the previous-step OUTPUT_LO
   carried the default-increment PC; on a JMP-taken or ENT step
   that's not true. Fixing this requires either (a) per-prev-step
   marker reads (`MARK_PC.*.-1`) plus `OP_BZ.*.-1` to detect "prev
   step was also a BZ-taken / JMP / control-flow change", or (b)
   moving the cancel band to a different residual that's known-zero
   pre-write. Both are multi-rule structural changes.

**Conclusion: C5 is a Phase-7-scope fix, not a one-rule fix.**
Following the closeout plan's "single-rule whack-a-mole is zero-sum"
guideline (`CLAUDE.md` § Workflow constraints), no fix is shipped
in this doc. Doc-only.

## Recommended next step

Add `BZ_FRESH` lifecycle bit at the dim-registry level (parallel
to the `IN_STEP_FRESH` plan in Bug #14):

1. Allocate a fresh dim `BZ_TARGET_FRESH` near `HAS_SE` (pin 137
   neighborhood).
2. Add an L4/L5 writer that sets `BZ_TARGET_FRESH = 1` only on the
   step where `OP_BZ` is decoded AND `FETCH_LO` was just written
   (gate on `MARK_PC` AND `OP_BZ` AND a new `FETCH_LO_FRESH` term
   from L5's `FETCH_LO` writer).
3. Add `("BZ_TARGET_FRESH", +10)` to the override conditions and
   bump threshold by +10 (mirror the `HAS_SE` step-0 guard
   pattern).
4. Verify byte-identity via `compare_symbolic_to_lowered_ffn` on
   `_post_l9_bz_bnz_pc_override_ir()`.

Expected impact: ~128 rows (28 direct + ~100 Wave J loop family).

## Cross-references

- [`EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md`](EDGE_ABSDIFF_BOOL_ATTRIBUTION_2026_06_07.md) — H6 cluster attribution; Recommendation 1.
- [`LOOP_RECUR_ATTRIBUTION_2026_06_07.md`](LOOP_RECUR_ATTRIBUTION_2026_06_07.md) — §Step 4 surface determination; `loop_pow2_0` trace at neural=0xFF0A.
- [`ABSDIFF_BZ_REDIRECT_BUG.md`](ABSDIFF_BZ_REDIRECT_BUG.md) — canonical
  per-step PC tracking on `absdiff_0`; PC walks past bytecode end.
- [`L34_FFN_ATTRIBUTION_2026_06_04.md`](L34_FFN_ATTRIBUTION_2026_06_04.md) — origin of the existing `HAS_SE +10` step-0 guard.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) §Bug #14 — missing `IN_STEP_FRESH` lifecycle bit; same root mechanism.
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:440-485`
  (`_append_pc_byte0_imm_to_byte_addr_rules`).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4509-4574`
  (`_post_l9_bz_pc_override_rules`).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:4577-4640`
  (`_post_l9_bnz_pc_override_rules`).
- `c4_release/neural_vm/dim_registry_dynamic.py:129` (`HAS_SE` pin).
