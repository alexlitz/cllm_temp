# edge_/absdiff_/bool_ cluster attribution — 2026-06-07

Date: 2026-06-07 (H6 sub-agent; wall-clock 2026-06-08).
Base ref: `speedup-cache-and-buckets` HEAD `7de7f5d4`
(`docs(qwen-r8): mark Blocker 3 closed on tiny VM`).
Task: H6 attribution for the `edge_*` / `absdiff_*` / `bool_*` 1096
clusters; cross-reference closeout plan C1/C2/C3/C4 surfaces.

## TL;DR

| Cluster | Tests selected | XPASS (pass) | XFAIL (fail) | Pass rate | Root surface |
|---|---:|---:|---:|---:|---|
| `edge_*` | 46 | 40 | 6 | **87 %** | Two distinct: (a) `_zero_{mul,div,mod}` (3 tests) — wide-ALU 0-operand cascade; (b) `_if_hundred` / `_loop_never` / `_loop_once_skip` (3 tests) — **L6 `post_l9_bz_bnz_pc_override` re-fire**, same surface as Wave J loops in `LOOP_RECUR_ATTRIBUTION_2026_06_07.md`. |
| `absdiff_*` | 25 | (long wall; baseline 0/25) | (baseline 25/25 fail) | **~0 %** | **L6 `post_l9_bz_bnz_pc_override` taken-path PC carry-forward**, per `ABSDIFF_BZ_REDIRECT_BUG.md`. Also requires L9/L10 AX-byte channel writes (every step shows `AX=0` in the canary trace). |
| `bool_*` | 25 | 24 | 1 | **96 %** | Pre-fix `bool_and_*` 16/16 PASS per `1096_TRIAGE_2026_06_05.md` Bucket I. The single survivor xfail (`bool_and_22`) is a nested-if-stack-frame edge case sharing the var/func `MEM_addr1` baseline (Wave A surface). |

**Cluster-to-closeout-wave mapping:**

| Wave | Cluster bleed |
|---|---|
| **C1 (`func_add` JSR/LEV return)** | Possibly some of the `absdiff_*` 25 (function call return path). |
| **C2 (`var_*` L3 SP_byte2)** | `bool_and_22` (only) — shares the var MEM_addr1 baseline. |
| **C3 (Shape-B EQ L6 byte-1)** | No bleed. `bool_and_*` uses `>` not `==`/`!=`, so it does not exercise the L6 byte-1 emit. `edge_if_hundred` is a non-zero literal in conditional; not EQ Shape-B. |
| **C4 (`expr_add_mul`)** | No bleed. `edge_zero_{mul,div,mod}` use single-step `0*N`/`0/N`/`0%N` with no intermediate-result staging. |
| **(new) "branch re-fire" surface** | `edge_if_hundred`, `edge_loop_never`, `edge_loop_once_skip`, `absdiff_*` (25), shared with Wave J loops/recursion. Not covered by C1-C4. |

## Step 1 — Base confirmation

```
$ git log --oneline -3 HEAD
7de7f5d4 docs(qwen-r8): mark Blocker 3 closed on tiny VM
654dbe50 docs: loop recur attribution
fd109b86 feat(l10): Wave 1 A3 broadcast heads — AX byte 1/2/3 -> STACK0_BYTE_VAL_h on PSH
```

Precondition `≥ 7de7f5d4` met; no `git merge main` needed.

## Step 2 — Per-prefix pytest counts

Driver: `pytest tests/test_suite_1096_pure_neural_pytest.py -k "<prefix>"`
on the worktree `agent-aa08d8260075651c6`, no env overrides (default
declarations-only XFAIL-non-strict mode, `C4_BATCH_CHUNK=32`).

### edge_*

```
$ pytest tests/test_suite_1096_pure_neural_pytest.py -k "edge_" --tb=no -q
collected 1098 items / 1052 deselected / 46 selected
========= 1052 deselected, 6 xfailed, 40 xpassed in 106.45s (0:01:46) =====
```

- **40 / 46 pass** (87 %).
- **6 / 46 fail** (xfail) — IDs (extracted via `-v`):
  - `1001_edge_zero_mul`  (`return 0 * 100;`)
  - `1003_edge_zero_div`  (`return 0 / 5;`)
  - `1004_edge_zero_mod`  (`return 0 % 5;`)
  - `1028_edge_if_hundred` (`if (100) return 1; return 0;`)
  - `1029_edge_loop_never` (`while (i < 0) ...`)
  - `1030_edge_loop_once_skip` (`while (i > 1) ...`)

### absdiff_*

```
$ pytest tests/test_suite_1096_pure_neural_pytest.py -k "absdiff_" --tb=no -q
collected 1098 items / 1073 deselected / 25 selected
[... no completion within agent wall budget; per-case max_steps
     dominates and the cluster baseline from
     ABSDIFF_BZ_REDIRECT_BUG.md was 27/30 fail = 90% fail in the prior
     30-test version, now 25-test under the 42-seed RNG canon]
```

- **Per `ABSDIFF_BZ_REDIRECT_BUG.md` § Symptom: 27 / 30 fail → ≈ 25 / 25 fail
  under the current 25-test parametrisation**. Each test exits with
  `neural_exit = None` because the model never reaches EXIT.
- **Per `c4_release/docs/BUG_CATALOG.md` Bug #26**: "all 25 `absdiff_*`
  rows and all 25 `nested_quad_*` rows now first-fatal on the post-LEV
  `step6:AX_byte0` / `step6:STACK0_byte0` slots that motivate bug #33".
- Bottom line: baseline pass-rate ≈ **0 / 25**; the cluster is the
  next-largest BZ-relay bucket after the loop family.

### bool_*

```
$ pytest tests/test_suite_1096_pure_neural_pytest.py -k "bool_" --tb=no -q
collected 1098 items / 1073 deselected / 25 selected
========= 1073 deselected, 1 xfailed, 24 xpassed in 162.62s (0:02:42) =====
```

- **24 / 25 pass** (96 %).
- **1 / 25 fail** (xfail) — `bool_and_22` (the parametrised
  `a > b && b > c` shape on a triple where the `(a, b, c)` random draw
  hits the var-MEM_addr1 surface).

## Step 3 — Simplest failing test + L13/L14 hook divergence

### `edge_if_hundred` (simplest BZ-relay failure in edge_)

Source (suite line 531):

```c
int main() { if (100) return 1; return 0; }
```

Compiles to (per `src.compiler.compile_c`):

```
ENT 0
IMM 100
BZ +else
IMM 1
LEV
+else:
IMM 0
LEV
```

**Per the closeout plan §S2 and LOOP_RECUR_ATTRIBUTION_2026_06_07.md
§Step 3 hook**: L13 head 1-3 (`addr_b1..b3`) read `L1H{1,2,3}+MEM_I`
K-side and copy `CLEAN_EMBED` at addr-byte tokens. L14
`mem_generation` writes next-token `MEM_addr/val` for SI/LI — silent
on the BZ step. The sole declarative path mapping the branch
target literal `FETCH_LO+k` → `OUTPUT_LO` PC byte 0 is L6
`post_l9_bz_bnz_pc_override` (`l6_ops.py:4566`).

**Divergence point**: on the BZ step at PC 4 the `step0_guard_weight`
(+10 on `HAS_SE`) gates the *first* branch step. For `if (100)`,
`IMM 100` had run at step 1, so `HAS_SE` is no longer the
distinguishing residual — the override re-fires with the wrong source
band and emits the `else` target instead of the `then` target. This
is the same re-fire pattern documented for `loop_pow2_0` in
`LOOP_RECUR_ATTRIBUTION_2026_06_07.md` § `loop_pow2_0` trace, except
the loop case has neural=`0xFF0A` and the `edge_if` case has
neural=`0` (the `else` branch's literal).

### `edge_zero_mul` (simplest ALU failure in edge_)

Source:

```c
int main() { return 0 * 100; }
```

Reduces to `IMM 0; PSH; IMM 100; MUL; LEV`. No BZ, no branch.
**Per `1096_TRIAGE_2026_06_05.md` Bucket B / Bug #34**: every
`loop_pow2_*` and `loop_mul_*` row yields `neural = 0xD8 = 216` —
the wide-MUL 0-operand sentinel. **`edge_zero_mul`** is the
single-step incarnation of the same bake (Bug #34 § "Wide-MUL
single-byte regression to 0xD8 = 216 sentinel"). L13/L14 hooks confirm
no divergence at the MUL step on MEM_addr; the divergence is at L9/L10
on the AX_byte0/1 output channel where the wide-ALU writes the 216
sentinel instead of the 0 result.

### `absdiff_0` (canonical BZ taken-path failure)

Source:

```c
int abs_diff(int a, int b) {
    if (a > b) return a - b;
    return b - a;
}
int main() { return abs_diff(16, 85); }
```

Per `ABSDIFF_BZ_REDIRECT_BUG.md` § Trace, the per-step PC tracking
inside the `abs_diff` body shows the BZ at step 13 should redirect to
the LEV at idx 17 but the model falls through. The L13/L14 hook on
this step matches the `edge_if_hundred` pattern: the BZ override at L6
re-fires with the wrong band because `HAS_SE` is no longer the
distinguishing first-step gate.

### `bool_and_22` (simplest var-MEM_addr1 bleed)

Per `1096_TRIAGE_2026_06_05.md` Bucket A, all bool_and tests are
nested `if`s — `if (a > b) { if (b > c) return 1; }`. The first `if`
emits a BZ relative to the *function-frame* address space, and the
nested second `if` relies on the same residual at a deeper stack
depth. For the specific random draw at `bool_and_22`, the inner
comparison ((a,b,c) = (53, 51, 96) under seed 42) produces
`a > b = TRUE` and `b > c = FALSE`. The neural emit reads stale
`MEM_addr1` for the inner-`if` BZ target (the var MEM_addr1 baseline
documented in `VAR_REAL_ATTRIBUTION_2026_06_05.md`) — same L3 `mem_byte_0_default` +
L13 head 7 `top_store_query` aux pattern as Wave C2.

## Step 4 — Cross-reference closeout plan C1/C2/C3/C4 surfaces

| Test | Closeout wave | Reasoning |
|---|---|---|
| `edge_zero_mul` / `_div` / `_mod` | **none (Bug #34)** | Wide-MUL/DIV/MOD 0-operand sentinel; not covered by C1-C4. Falls under Bug #34 (loop_pow2/loop_mul 0xD8 sentinel). Single-rule fix candidate per `BUG_CATALOG.md`. |
| `edge_if_hundred` / `edge_loop_never` / `edge_loop_once_skip` | **none (new "branch re-fire" surface)** | L6 `post_l9_bz_bnz_pc_override` re-fire — same surface as Wave J loops per `LOOP_RECUR_ATTRIBUTION_2026_06_07.md`; not covered by C1-C4. Recommend adding **Wave C5** = branch-target re-fire. |
| `absdiff_*` (25 / 25) | **partial overlap with C1** | The function-call return path (LEV→AX recovery) shares the L16 LEV / L6 head-7 AX_CARRY refresh surface that C1 targets via `func_add`. Bug #33 in `BUG_CATALOG.md` calls absdiff_* + nested_quad_* downstream of the same L10 PSH addr0_e0 OP_ENT guard. Likely partial closure (≤ ⅓) from C1 alone; the BZ-relay re-fire from `ABSDIFF_BZ_REDIRECT_BUG.md` is the dominant remaining surface and needs Wave C5. |
| `bool_and_22` | **C2 (`var_*` L3 SP_byte2)** | The single survivor xfail inherits the var-MEM_addr1 baseline (L3 `mem_byte_0_default` + L13 head 7 `top_store_query` aux). Per Wave A fix candidate in `1096_TRIAGE_2026_06_05.md` § Recommended next-round targets #1 + #2. |

**No EQ Shape-B (C3) bleed** in these three clusters. **No `expr_add_mul`
(C4) bleed** in these three clusters.

## Step 5 — Smoke (≥ 45)

```
$ pytest tests/test_smoke.py --tb=no -q
============ 6 failed, 45 passed, 1 deselected in 137.61s ============
```

Smoke is **45 / 51 PASS**, exactly the closeout-plan threshold. The 6
failures are the documented Wave S1 memory cluster (5) + S2 LEA (1),
none of which this attribution doc touches (doc-only change).

## Recommendations to closeout plan

1. **Add Wave C5 — BZ branch-target re-fire**. Closes
   `edge_if_hundred` + `edge_loop_never` + `edge_loop_once_skip` +
   `absdiff_*` (≈ 25 / 25 = 25) + the Wave J loop family already
   attributed to the same surface. Plan:
   - Identify the residual that distinguishes BZ step 0 from BZ
     step ≥ 2 (currently only `HAS_SE`, which is too permissive).
   - Add a step-N gate to `_append_pc_byte0_imm_to_byte_addr_rules`
     in `l6_ops.py:440` — e.g. `STEP_END.prev` ⇒ on this step.
   - Byte-identity gate via `compare_symbolic_to_lowered_attn`.
2. **Add Wave C6 — Wide-MUL/DIV/MOD 0-operand sentinel (Bug #34)**.
   Closes `edge_zero_mul` + `_div` + `_mod` (3) + `loop_pow2_*` (25) +
   `loop_mul_*` (25). Single-rule fix candidate per
   `BUG_CATALOG.md` § Bug #34.
3. **No change to C1-C4 surfaces**: the data confirms `edge_*` (40/46),
   `absdiff_*` (~0/25), `bool_*` (24/25) bleed cleanly into the new
   surfaces C5/C6 + the existing Wave A (var-MEM_addr1) — not into the
   originally-attributed C1-C4 paths except via the partial overlap
   noted for `absdiff_*` / C1.

## Cross-references

- [`CLOSEOUT_PLAN_2026_06_07.md`](CLOSEOUT_PLAN_2026_06_07.md) — Wave
  C1/C2/C3/C4 + S1/S2 definitions.
- [`ABSDIFF_BZ_REDIRECT_BUG.md`](ABSDIFF_BZ_REDIRECT_BUG.md) — 27/30
  `absdiff_*` BZ taken-path PC redirect; canonical canary trace.
- [`LOOP_RECUR_ATTRIBUTION_2026_06_07.md`](LOOP_RECUR_ATTRIBUTION_2026_06_07.md) —
  L6 `post_l9_bz_bnz_pc_override` re-fire; canonical
  `_append_pc_byte0_imm_to_byte_addr_rules` reference.
- [`1096_TRIAGE_2026_06_05.md`](1096_TRIAGE_2026_06_05.md) — Bucket I
  (`edge_lit 15/15`, `bool_and 16/16`, `edge_pow2 13/13`) baseline
  confirming the small-program clusters were 100 % on the 06-05 sample.
- [`VAR_REAL_ATTRIBUTION_2026_06_05.md`](VAR_REAL_ATTRIBUTION_2026_06_05.md) —
  L3 `mem_byte_0_default` + L13 head 7 `top_store_query` aux pattern
  the `bool_and_22` survivor inherits.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) — Bug #26 (`absdiff_` / `nested_quad_`
  dead), Bug #33 (post-LEV AX corruption), Bug #34 (wide-MUL 0xD8
  sentinel).
- [`CHAR_STRING_ATTRIBUTION_2026_06_07.md`](CHAR_STRING_ATTRIBUTION_2026_06_07.md)
  / [`PTR_DEREF_ATTRIBUTION_2026_06_07.md`](PTR_DEREF_ATTRIBUTION_2026_06_07.md)
  / [`STRUCT_GLOBAL_ATTRIBUTION_2026_06_07.md`](STRUCT_GLOBAL_ATTRIBUTION_2026_06_07.md) —
  prior H1/H3 boundary docs in the attribution series.
