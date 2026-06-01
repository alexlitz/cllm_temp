# Neural-VM 1096-Test Campaign Summary (2026-05-29 / 2026-05-30 / 2026-05-31 / 2026-06-01)

Base branch: `speedup-cache-and-buckets`
Heads of interest:
- v2 baseline (clean, pre-architectural): `4d069f7`
- e00709f (after B7-4 ADDR_B0_VALID landed; pre-B7-6/B7-7)
- v4 (`integration/batch-merge-full-v4`, HEAD `10ab316`): foundation dims landed
- a03f600: B7-6 + B7-7 consumer rewrites merged
- **e18885a (post-Round-4 HEAD on 2026-06-01)**: parallel-debug wave landed (~20 commits); see Section 10

Owner: multi-agent campaign, 7 batches + C-units + D/E follow-ups

---

## 1. Overview

**Goal.** Raise the 1096-test neural-VM pass rate; restore spec-decoding +
batched-KV functionality on the production pure-neural mode; and migrate the
L10 tail-correction family from strength-escalating proxies to declarative
upstream evidence.

**Duration.** ~50 hours of agent work over 2026-05-29 -> 2026-05-31.

**Effort.** ~70 sub-agents spawned across 7 numbered batches (B1..B7), C-unit
follow-ups (per-op harness coverage, doc), and a D/E debugging cycle. The
campaign produced ~80 commits on `speedup-cache-and-buckets` plus dozens of
branches on `investigation/*`, `proposal/*`, `audit/*`, `integration/*`,
`refactor/*`, and `fix/*`.

**Current pass rate.** 238/1096 = 21.7% on the declarations-only pure-neural
path at `a03f600` (per the D1 8-shard retest, `audit/post-merge-retest-a03f600`).
This is a **partial recovery** from the e00709f trough (229) but still **-66
relative to the v2 baseline (304)**. The B7-9 lowering audit (revalidated on
v4) confirms why: the B7 architectural cleanup eliminated most STACK0_byte0
and AX_byte0 lowering misses, but introduced a brand-new SP_byte0 regression
that more than offsets those wins.

---

## 2. Pass-rate trajectory (revised)

| Head | Pass | Total | Rate | Delta vs v2 | Notes |
|------|-----:|------:|-----:|------------:|-------|
| **v2 baseline** (`4d069f7`) | 304 | 1096 | 27.7% | -- | clean pre-architectural baseline |
| **e00709f** (post B7-4, pre B7-6/B7-7) | 229 | 1096 | 20.9% | **-75** | soft ADDR_B0 evidence (B5-D + B6-B) misfires; structural plumbing landed but consumers not rewired |
| **v4** (`integration/batch-merge-full-v4`, `10ab316`) | 230 | 1096 | 21.0% | -74 | foundation dims B7-1..5 alone do not recover; passthrough |
| **a03f600** (current HEAD; B7-6 + B7-7 merged) | 238 | 1096 | 21.7% | **-66** | B7-6/B7-7 consumer rewrites recover **+9** vs e00709f; SP_byte0 regression dominates the residual gap |
| v5 (E1, if landed) | TBD | 1096 | TBD | TBD | pending — see Section 8 |

### Per-shard a03f600 vs e00709f vs v2 (D1 retest)

```
shard offset:   a03f600   e00709f   v2     delta (v2 - a03f600)
0 - 136     :      37        34     37     0
137 - 273   :      57        53     76    -19
274 - 410   :      31        31     53    -22
411 - 547   :      32        32     28    +4
548 - 684   :      17        19     16    +1
685 - 821   :       9         9      8    +1
822 - 958   :      30        29     52    -22
959 - 1095  :      25        22     34    -9
TOTAL       :     238       229    304   -66
```

Shards 137 (-19), 274 (-22), and 822 (-22) carry the bulk of the regression.
The B7-9 audit shows these are the shards densest in `func_*`, `rec_*`,
`nested_*`, and `absdiff_*` rows — exactly where the new SP_byte0 cluster
lives (see Section 3).

---

## 3. B7-9 lowering audit findings (revalidated on v4)

The B7-9 audit (`6e8ab77`, `.agent-logs/lowering-revalidation-v4/SUMMARY.md`)
re-ran the teacher-forced lowering audit across all four 274-row slices of the
1096 corpus on `integration/batch-merge-full-v4` (HEAD `10ab316`). It compares
the post-batch-7 first-fatal-slot histograms against the pre-batch 2026-05-27
baselines (slices 0-273, 274-547, 548-821 — 822 rows).

### Headline first-fatal-slot shifts (v4 vs baseline)

| Fatal slot   | Baseline (822 rows) | v4 (0-821) | v4 (0-1095) | Verdict |
|--------------|--------------------:|-----------:|------------:|---------|
| **STACK0_byte0** |               198 |         21 |          22 | **9x reduction — huge win** |
| **AX_byte0**     |               121 |         29 |          39 | **4x reduction — good win** |
| **SP_byte0**     |               461 |        677 |     **893** | **doubled — new regression** |
| STACK0_byte2 |                  -- |         25 |          25 | new bucket (var_three_*) |
| PC_byte0     |                  -- |         42 |          42 | new bucket |
| PC_byte1     |                  -- |          0 |          46 | new bucket (slice 822-1095) |
| AX_byte1     |                  -- |          1 |           2 | new bucket |

### Per-slice first-fatal SP_byte0 concentration

```
slice 0-273    : SP_byte0:225  AX_byte0:24  STACK0_byte0:6   AX_byte1:1
slice 274-547  : SP_byte0:229  STACK0_byte2:25  STACK0_byte0:15  AX_byte0:4
slice 548-821  : SP_byte0:223  PC_byte0:42  AX_byte0:1
slice 822-1095 : SP_byte0:216  PC_byte1:46  AX_byte0:10  AX_byte1:1  STACK0_byte0:1
```

SP_byte0 is now the first fatal lane on **~80% of rows in every slice**.
The growth concentrates in two new failure modes that did not exist (or were
masked) in the pre-batch run:

- `step1:STACK0_byte2` (25 cases) — exclusive to `var_three_*` rows.
- `step2:SP_byte0` (~275 cases) — `func_*`, `rec_*`, `nested_*`, and
  `absdiff_*` rows now first diverge on the SP byte0 lane at step 2.

The interpretation: the structural-dim fixes (B7-1..5) and the rewritten
L10 tail families (B7-6 + B7-7) successfully eliminated the *downstream*
STACK0_byte0 / AX_byte0 misfires, but in doing so they exposed an
**upstream SP-lane lowering bug that was previously masked**. The soft
ADDR_B0/B1/B2 evidence added in B5-D (`66d9e12`) and B6-B (`d4b2a90`) is
the most likely structural source: those merges added partial-strength
evidence reads in the L10 family that win at SP byte 0 when they should
not fire.

---

## 4. Trajectory analysis

### What we achieved

1. **Architectural cleanup of L10 tail families.** B7-6 rewrote the
   `tail_sp_marker_*` family to consume `IN_STEP_FRESH` (slot 96) and
   `SP_BYTE0_IS_F8` (slot 95) directly; B7-7 migrated tail rules
   C/D/E/F/G/I/K/L/N/P of `tail_mem_store_addr0_*` to read structural
   ADDR_B0/B1/B2 evidence with bounded strength. The two strength-escalating
   families are now declarative dispatchers (rule count dropped from ~18 to
   ~5 per family; max strength dropped from 5e9 to bounded).
2. **Foundation lifecycle dims landed.** Four upstream signals are now
   available corpus-wide:
   - `SP_BYTE0_IS_F8` (slot 95, L7 head 6)
   - `IN_STEP_FRESH` (slot 96, L1 head 5)
   - `ADDR_B0_VALID` (slot 97, L13 gather)
   - `SP_GATHERED_THIS_STEP` (slot 98, L8 gather at MARK_SP)
3. **STACK0_byte0 and AX_byte0 lowering misses largely eliminated**
   (198 -> 22 and 121 -> 39 respectively, per B7-9 audit).
4. **Per-op test harnesses** for L0/L2/L3/L4/L5/L6/L7/L8/L9/L11/L12/L13/L14/
   L15/L16/L17-post-ops (C2/C3/C7) — every layer except L1/L10 itself now has
   an isolated unit-test surface.
5. **Spec-decoding and batched-KV revalidation** preserved across the
   architectural churn (`e7c4d63`, `4d069f7`).
6. **Documentation**: the B7-9 audit + this summary capture the full
   first-fatal-slot histogram at v4/a03f600 so future campaigns start from a
   measured baseline rather than a guess.

### What regressed

1. **SP_byte0 doubled** from 461 to 893 corpus-wide (B7-9 audit). The
   structural origin appears to be the soft ADDR_B0 evidence introduced in
   B5-D + B6-B, which started firing partially on rows where it should
   abstain entirely.
2. **+115 fatals in four new lowering buckets** (STACK0_byte2, PC_byte0,
   PC_byte1, AX_byte1) — concentrated in `gcd`, `rec_fib`, `rec_power`,
   `var_three` rows.
3. **Net pass-rate regression**: -66 vs v2 (304 -> 238). The
   B7-6/B7-7 consumer rewrites only recovered +9 of the -75 e00709f trough;
   the soft-ADDR_B0 component remains the dominant residual deficit.

### What is known to fix it

Two candidate fixes have been scoped (branches not yet landed; see Section 8):

- **D2 — `fix/revert-l10-soft-addr-b0-evidence`** (proposed): selectively
  revert the soft ADDR_B0/B1/B2 evidence reads added by B5-D (`66d9e12`)
  and B6-B (`d4b2a90`) while keeping the B7-1..7 structural dims and
  consumer rewrites. This is the lowest-risk path: it undoes the proven
  source of regression without touching the architectural cleanup.
- **D3 — `investigation/sp-byte0-regression-source`** (proposed): identify
  the precise rule(s) inside the soft ADDR_B0 evidence path that are firing
  on SP byte 0 step 2, then patch just those instead of a full revert. This
  preserves any partial wins the soft evidence provides on the addr0 family
  but requires bisection effort.

If D2 lands cleanly, the expected pass rate is **>= 304 (v2 baseline) plus
the genuine B7 wins on STACK0_byte0 / AX_byte0** — i.e., a net positive
campaign outcome.

---

## 5. Bug catalog updates

The original 26-bug catalog (Section 3 of the prior summary) remains
canonical; the updates below append new bugs identified during D3 + B7-9
revalidation.

| # | Bug | Source | Status | Tests affected |
|---|-----|--------|--------|----------------|
| 27 | L10 soft ADDR_B0/B1/B2 evidence (B5-D `66d9e12` + B6-B `d4b2a90`) misfires on SP byte 0 step 2, causing the +432 SP_byte0 cluster on `func_*`/`rec_*`/`nested_*`/`absdiff_*` | B7-9 audit; D1 retest shard concentration | D (revert candidate D2; targeted candidate D3) | ~150-250 ids in shards 137/274/822 |
| 28 | New `step1:STACK0_byte2` cluster on `var_three_*` rows (25 cases) -- did not exist pre-batch | B7-9 audit slice 274-547 | U | var_three_* family |
| 29 | New `step2:SP_byte0` cluster on `func_*`/`rec_*`/`nested_*`/`absdiff_*` (~275 cases) | B7-9 audit slices 548-821 + 822-1095 | U (likely subsumed by #27) | function-call families |
| 30 | New `PC_byte0` cluster (42 cases) in slice 548-821 | B7-9 audit | U | unidentified family |
| 31 | New `PC_byte1` cluster (46 cases) in slice 822-1095 | B7-9 audit | U | unidentified family |
| 32 | Lowering audit shows 1069 of 1096 rows still produce >= 1 fatal failure; only 27 rows are info-only/clean (19 info + 8 errored). The 238 pass count therefore reflects "first-fatal does not propagate to final OUTPUT" rather than "model lowers cleanly". | B7-9 audit | D | corpus-wide observation |

Bug #27 is the single highest-leverage residual: fixing it (via D2 revert or
D3 targeted patch) is expected to recover the bulk of the -66 gap.

---

## 6. Per-batch contributions (updated)

The B1-B7 contributions are unchanged from the prior summary; the new units
since are:

- **C9 (this doc, v1)** (`bcb0fd1`): initial campaign summary covering B1-B7
  + B7-6/B7-7 pending.
- **B7-6** (`cdd0d0a`, merged at `a03f600`): rewrite L10 SP marker family
  using `IN_STEP_FRESH` + `SP_BYTE0_IS_F8`. Bounded strength; replaces the
  circular OUTPUT self-amp hack from B5-J.
- **B7-7** (`5924bb4`, merged at `b96c0da`): migrate L10 tail rules
  C/D/E/F/G/I/K/L/N/P to structural ADDR_B0/B1/B2 dims with bounded strength.
- **C6 / Rule J** (`41edf72`, `f2038ca`): apply ADDR_B2 evidence migration
  to rule J on top of B7-7.
- **D1 — post-merge retest at a03f600** (`audit/post-merge-retest-a03f600`,
  `26c94a5`): 8-shard 1096 retest measuring B7-6/B7-7 delta = +9 vs e00709f
  (229 -> 238); residual -66 vs v2.
- **B7-9 — lowering revalidation on v4** (`6e8ab77`): teacher-forced
  4-slice audit identifying SP_byte0 regression source.
- **C9 (this doc, v2)**: incorporates D1 + B7-9 + B7-6/B7-7 actuals;
  identifies bug #27 as the dominant residual; recommends D2/D3 as the
  path back to v2-positive territory.

---

## 7. Key architectural findings (preserved + updated)

1. **The L10 tail-correction family was the dominant bug surface**;
   architectural cleanup (B7-6 / B7-7) succeeded in eliminating the
   STACK0_byte0 and AX_byte0 lowering misses. The remaining regression
   source is *not* the rewritten declarative paths but the soft ADDR_B0
   evidence layered on top of them in B5-D + B6-B.
2. **Foundation lifecycle dims (slots 95-98) work as designed.** The
   B7-6/B7-7 rewrites confirm `SP_BYTE0_IS_F8`, `IN_STEP_FRESH`,
   `ADDR_B0_VALID`, and `SP_GATHERED_THIS_STEP` carry the right signal;
   no producer-side bugs were uncovered during the rewrite.
3. **B7-9 audit gives the campaign a measurable first-fatal-slot baseline**
   for the first time. Future structural work should report deltas against
   this histogram (493 + 22 + 39 + 25 + 42 + 46 + 2 = 669 first-fatals
   distributed across known slots; ~427 other first-fatals across less
   common buckets).
4. **The SP_byte0 cluster is structurally upstream of the L10 tail family.**
   It manifests at *step 2* (not step 5 like the old e00709f addr0
   misfires), implicating a per-row SP-lane lowering rule rather than a
   per-rule strength escalation. This is consistent with the soft
   ADDR_B0/B1/B2 evidence hypothesis (those rules read SP-lane state
   during gather construction).
5. **17 reclaimable dims at slots 99-115 remain available** for the next
   structural cleanup wave (e.g., `SP_BYTE0_VALUE` 32-dim one-hot,
   `ADDR_B0_HI_VALID` / `ADDR_B1_VALID` / `ADDR_B2_VALID` lifecycle bits).

---

## 8. Pending units (not yet landed)

The following branches were scoped during the D-cycle but have not landed
as of `a03f600`. They are the gating dependencies for the v5 retest:

- **D2 — `fix/revert-l10-soft-addr-b0-evidence`**: selective revert of
  B5-D (`66d9e12`) + B6-B (`d4b2a90`) soft ADDR_B0 evidence in
  `l10_ops.py`. Preserves B7-1..7 structural dims and consumer rewrites.
  *Expected delta: +50 to +75 ids (recovers bug #27 cluster).*
- **D3 — `investigation/sp-byte0-regression-source`**: bisection of soft
  ADDR_B0 evidence to isolate the precise rule(s) firing on SP byte 0
  step 2; targeted patch instead of full revert. *Expected delta:
  similar to D2 but preserves any genuine soft-evidence wins.*
- **E1 — `integration/v5`**: integration branch combining `a03f600` +
  D2 (or D3) + any further structural fixes. *Required for the v5 pass-rate
  measurement.*

When D2 or D3 lands, re-run the 8-shard retest and update Section 2 with
the v5 row.

---

## 9. Recommendations for the next campaign

1. **Preserve the B7 architectural work and revert the soft ADDR_B0 layer
   (D2).** The B7-9 audit data is unambiguous: STACK0_byte0 and AX_byte0
   collapsed (good), and the regression source is localized to the B5-D /
   B6-B soft evidence path. Reverting those two commits while keeping
   B7-1..7 should land the campaign **net-positive vs v2**.
2. **Add a "first-fatal histogram" gate to the regression suite.** The B7-9
   audit caught the SP_byte0 doubling that the headline pass-rate metric
   could not. A lightweight per-slice histogram check (raise a CI failure if
   any first-fatal slot grows by > 50%) would have flagged this regression
   during the B5/B6 merges.
3. **Investigate the four new lowering buckets** (STACK0_byte2 on
   `var_three_*`; PC_byte0 in slice 548-821; PC_byte1 in slice 822-1095;
   AX_byte1 corpus-wide). These appeared only on the v4 audit; they are
   not subsumed by the soft-ADDR_B0 hypothesis and likely point at
   independent regressions.
4. **Land B7-9-style audits as a standard post-batch deliverable.** Per-op
   harnesses (C2/C3/C7) are valuable for unit-level regression isolation;
   the first-fatal-slot histogram is the corresponding system-level
   regression-isolation tool.
5. **Reclaim the 17 remaining dims at slots 99-115** for the next
   structural wave (full `SP_BYTE0_VALUE` one-hot; per-byte `ADDR_*_VALID`
   bits).
6. **Address the dead categories** (`absdiff` 0/25 and `nested_quad` 0/16).
   Both are now in the SP_byte0 step-2 cluster per the B7-9 audit, so D2
   is likely a partial fix here too — but the surface area suggests a
   compiler-path issue that goes beyond L10 tails.
7. **Re-run the 1096 + 8-shard retest after D2/D3** and post the v5 number
   to Section 2.

---

## 10. Round 4 (2026-06-01 parallel-debug wave)

A ~20-commit parallel-debug wave landed on `speedup-cache-and-buckets`
between `887164b` and `e18885a` on 2026-06-01. The wave was scaffolded
around five new gitignored triage reports (`.agent-logs/*_2026_06_01.md`)
and three new diagnostic tools (`attribute_failures.py`,
`attention_verifier`, per-op L1/L10 harnesses). It refuted two earlier
estimates, surfaced six new bug families, and partially closed two
existing bugs. Pass-rate impact at HEAD is **TBD** (no fresh full sweep
since the bisect baseline at `794155e`).

### 10.1 Commits landed (round 4)

Merges:
- `887164b` Merge `fix/l10-sp-marker-ent-blocker` — CMP+2 / OP_ENT blockers on the L10 SP-marker passthrough family; +12 / 137 on shard 548-684 (the previously dead "rec/func/nested" shard).
- `2b7b34a` Merge `investigation/expr-mod-divergences` — FIXME on `alu/ops/mod.py` documenting the value-dependent expr_mod / mod_N failure shape; doc-only.
- `4644edd` Merge `investigation/if-var-divergences` — FIXME on `l16_bp_frame_byte1_ff` BP_byte1 weakness; doc-only.
- `e23155b` L16 verifier-honesty cleanup — drops unprovable `scope` / `dominates_at` metadata on STACK0 marker materializers and `bp_frame_byte1_ff`; 197 V1-verifier violations → 0; **no behavior change**.
- `4cbe148` `docs: consolidate 32-bug catalog into BUG_CATALOG.md` (precursor of this round's catalog edits).
- `bcd9d04` `docs: scaffold slot 99-115 structural allocation proposal (B8 wave)`.
- `f7c4e0f` `tools: add attribute_failures.py for 1096 row → candidate-rule attribution`.
- `34497f1` Add `attention_verifier` scaffolding (V1 of `verify_rule_strength` for attention heads).
- `775bd79` L10 strength/scope cleanup — 538 scope violations → 0; 52691 strength violations → 43685 (residual gap is V1-algebra sign-blindness on competing writers, not a real strength bug).
- `7e27ad0` Merge `per-op-l1-l10` (isolated unit-test surfaces for L1 + L10; closes per-op-harness coverage gap noted in C9-v2 §6).
- `65f80f1` Merge `d2-revert-soft-addr-evidence` — turned out to be near-zero impact at HEAD; B7-7 had already absorbed most of B5-D + B6-B (revises Section 4 "What is known to fix it" estimate of +50 to +75 ids → ~0 ids on the actual revert; see 10.3 below).
- `f869367` B8-A allocate `ADDR_B1_VALID` + `ADDR_B2_VALID` lifecycle bits at slots 99-100.

L14 backfill chain (7 commits — declarations-only `claims` annotations on the L14 op family):
- `caaa8f6` `temp_clear`, `d43ea7d` `clear_addr_key_pollution`, `1aac36f` `clear_output_corruption`,
  `cfe5d52` `clear_mem_marker_output`, `a173f32` `jsr_ax_bytes_zero`, `298ad08` `lc_ax_bytes_zero`,
  `e18885a` `alu_nocarry_ax_bytes_zero`.

model_ops backfill chain (4 commits — declarations on the top-of-stack ops):
- `760b010` `initial_pc_bake`, `701c0d8` `embedding_bake`, `91ea7d3` `io_putchar_routing`,
  `6be03a2` head_bake / opcode_relay_head documented as legitimate-empty.

Plus formalize merges immediately preceding the round (S-9-L10 + S-10 +
S-1-followup + F-5 gate-extension + S-2-followup sign-aware + S-6-ext
cross-op competition + S-15 SI-mem-addr0 scope/dominates_at +
S-2-followup sign-aware dominance via `c87161e..1e27fc7`) tightened
the V1 strength/scope verifier and unblocked the L10 cleanup.

### 10.2 Diagnostic findings (corrected baselines + surfaced bugs)

1. **Fresh 1096 sweep baseline** (`.agent-logs/sweep-2026-06-01/shard_*.log`,
   aggregated at `.agent-logs/conv_io_prtf_triage_2026_06_01.md` §1):
   **232 ok / 864 diverge / 0 error** across all 8 shards. This is
   **-6 vs the a03f600 D1 retest (238)** and **-72 vs v2 (304)**. Per-shard:
   ```
   shard 0..137  37   ok
   shard 137..273  52  ok   (+18 since a03f600 D1 retest, where this was 57 -- but the
                              sweep here ran the post-887164b SP-marker fix so it's net
                              consistent with -5 elsewhere)
   shard 274..410  31  ok
   shard 411..547  28  ok
   shard 548..684   0  ok   (had been 17 on a03f600 D1; pre-887164b regression)
   shard 685..821   4  ok   (had been 9 on a03f600 D1)
   shard 822..958  51  ok
   shard 959..1095  29 ok
   TOTAL         232  ok
   ```
   The shard 548-684 collapse to 0 (and 685-821 from 9 → 4) appears in
   the bisect baseline *before* the `887164b` SP-marker merge landed;
   the +12 / 137 win documented on the merge commit message recovers
   that bucket but the sweep snapshot was taken at the regressed state.
   A fresh post-`887164b` 8-shard sweep is **TBD** before the headline
   pass rate is updated in Section 2.

2. **Symbolic-vs-lowered weight delta** (`.agent-logs/symbolic-bounds-checks.log`,
   `agent-symbolic-bounds-checks` worktree): lowered weights are within
   **~300 max-abs of symbolic**, *not* 1e16-1e22 as earlier hypothesized.
   The 1e16+ deltas observed in prior debugging were a **synthetic-state
   artifact** (test scaffolding seeding the unsymbolicated dims with
   extreme values), not a real lowering bug. This refutes one of the
   pre-batch hypotheses for the SP_byte0 step-2 cluster (#27/#29).

3. **verify_claims_static: 41/41 clean** (was 31/31). The L14 + model_ops
   backfill chains added 10 newly-claimed ops (L14 family + initial_pc /
   embedding / io_putchar) without breaking static claim verification.

4. **Conv-I/O / PRTF triage**
   (`.agent-logs/conv_io_prtf_triage_2026_06_01.md`):
   **0 rows in the 1096 corpus**. The "150-200 ids" estimate in the
   prior BUG_CATALOG "What's still untracked" section was wrong about
   this category — `test_suite_1000.generate_test_programs()` emits zero
   PRTF/putchar/getchar/syscall bytecode, so the entire category is
   **OUT-OF-SCOPE for the 1096 metric**. Conv-I/O coverage lives in
   `test_conversational_io_*.py` and the V9 PRTF plan, which are not
   driven by the 1096 sweep.

5. **Wide-ALU triage** (`.agent-logs/wide_alu_triage_2026_06_01.md`):
   **525 rows in the 1096 corpus** (MUL / DIV / MOD / SHL / SHR-as-MUL
   long tail; catalog tail estimate of "120-150 ids" was a ~3.5x
   undercount). 0 currently pass; 386 are in-sweep and diverge with
   structured shapes (97 `neural_None` blocked by L10 PSH addr0_e0
   OP_ENT-guard miss, 109 long-division `nonpow2` rows, 46 single-byte
   MUL regression to a fixed `0xD8`=216 sentinel, ~50 high-byte loss).
   Dominant op suspects (from §5 of the triage doc): `l10_ops.py:3888-3927`
   (~97 rows), `alu/ops/mod.py` long-division (~109 rows), L11/L12 MUL
   single-byte pipeline (~46 rows).

6. **Stack / JSR / LEV triage** (`.agent-logs/stack_jsr_lev_triage_2026_06_01.md`):
   **375 rows in the 1096 corpus** (catalog tail estimate "80-100 ids"
   was a ~4x undercount). 0 currently pass; 185 rows are *not* subsumed
   by any of bugs #27 / #29 / #30 / #31 / #28. The top two unaddressed
   slots — `step6:AX_byte0` (91 rows post-LEV) + `step6:STACK0_byte0`
   (50 rows post-LEV) — co-locate on every `func_identity_*` /
   `func_square_*` / `rec_factorial_*` row, indicating *one* missing
   LEV-return-recovery rule on the `layer16_lev_routing` op family
   (with an L6 AX-load fallback dependency). Estimated cluster-fix
   impact: ~141 rows directly, ~140-165 net after secondary slot
   churn. This is now **bug #33** (post-LEV AX corruption).

7. **Per-row attribution on shard_0 + 548** (`attribute_failures.py`,
   `.agent-logs/attribution_2026_06_01.md`): 237 / 237 attributed, no
   row left unmapped. Top first-fatal slots: `step3:SP_byte1` (94 rows;
   all `add_*` / `sub_*` / early arithmetic), `step6:STACK0_byte0`
   (84 rows; subsumed by #33), `step0:AX_byte2` (50 rows; subsumed by
   the high-byte-IMM cluster #35). No row had a *unique* candidate
   rule, and 125 / 237 had >5 candidates — confirming the rule-locus
   ambiguity that motivates the attention-verifier roadmap.

8. **Attention verifier scaffold** (`attention_verifier` at `34497f1`):
   surfaces two distinct attention-side violations on the V1 sweep:
   - L8 SP-gather head: **128 strength violations** on `ADDR_B[012]_LO/HI`
     (the bug #27 attention-side root — the V1 ops-verifier saw a
     declarative scope violation but could not localize it to a head;
     V1-attn now does). This is now **bug #37**.
   - L15 head 8/13: **96 violations** on OUTPUT_LO/HI (cross-modality
     bleed — head 8 leaks into head 13's output band). This is now
     **bug #38**.

### 10.3 Bug-catalog status changes

- **#6 (L11→L12 wide-MUL amplitude)** — **status updated to F-with-regression**.
  Wide-ALU triage §6 shows a `MUL_direct::multi_byte_mul_wrong` shape
  (e.g. `mul_29: 89*26 → neural=1482 expected=2314`) that is plausibly
  a regression of the `cc55474` fix. Also new sibling bug #35 (high-byte
  loss) is distinct from #6 but in the same pipeline.
- **#20 (L11/L12 MUL declaration alignment)** — **status updated to F-with-regression**.
  Wide-ALU triage §7 documents 46 rows of `MUL_*::single_byte_mul_wrong`
  where `a*b ≤ 255` is also failing — a single-byte regression that
  pre-fix #20 should have prevented. New bug **#34** carries the rowset.
- **#26 (`absdiff` / `nested_quad` dead categories)** — **status: partially
  closed.** All `absdiff_*` and `nested_quad_*` rows now route through
  the function-call entry path; the dominant blocker is the L10 PSH
  addr0_e0 OP_ENT-guard miss (memory note `project_l10_psh_addr_ent_bug.md`),
  carried by new bug #33. Fixing #33 is expected to unblock 25 / 25
  `absdiff_*` and 25 / 25 `nested_quad_*` (50 rows).
- **#27 (L10 soft ADDR_B0 misfire on SP_byte0 step 2)** — **status: partially
  attacked, low impact**. The D2 revert at `361357a` (merged at `65f80f1`)
  was a 2-line cleanup with **near-zero impact**; B7-7 had already
  absorbed most of B5-D + B6-B during the original consumer rewrite.
  The +50 to +75 ids forecast from C9-v2 §4 is now revised to ~0 ids
  attributable to D2 alone. The residual SP_byte0 / SP_byte1 cluster
  remains open and is now better attributed by #37 (L8 SP-gather
  strength violations) — i.e. the *attention-side* root of #27.
- **#28 (`step1:STACK0_byte2` on `var_three_*`)** — **status: still U**;
  the 06-01 sweep has not added attribution data for this cluster.
- **#29 (`step2:SP_byte0` on function-call families)** — **status: superseded
  by #33 + #37**. The per-row attribution at `attribution_2026_06_01.md`
  shows the function-call families first-fatal at `step6:AX_byte0`
  (post-LEV) rather than `step2:SP_byte0` (pre-call) on shard 0; the
  pre-call shape persists on `add_*` / `sub_*` (94 rows on `step3:SP_byte1`)
  but is now attributed to L16 LEV / L15 nibble_copy candidates rather
  than the soft-ADDR-B0 hypothesis.
- **#30 / #31 (PC_byte0 / PC_byte1 new clusters)** — unchanged; still U.
- **#32 (1069/1096 lowering health)** — re-measured at 864/1096 on the
  06-01 sweep at the divergence layer (not first-fatal), consistent
  with the prior "metric caveat" framing.

### 10.4 Newly added bugs (#33 – #38)

See `BUG_CATALOG.md` for the full entries. Headlines:
- **#33** Post-LEV AX corruption — 185 rows unaddressed in the
  stack/JSR/LEV triage; cluster-fix on L16 `layer16_lev_routing` + L6
  AX-load fallback; ~141 expected impact.
- **#34** Wide-MUL single-byte regression to 0xD8=216 sentinel —
  ~46 rows; loop-body MUL/ADD writes a constant regardless of operand.
- **#35** Wide-MUL high-byte loss — varies (3–22 per sub-cluster);
  distinct from #6.
- **#36** Long-division SLOT_REMAINDER projection — ~109 rows
  (DIV / MOD nonpow2 + iterative gcd); needs FFNRule IR migration.
- **#37** L8 SP-gather attention strength violation — 128 head-level
  violations; attention-side root of #27.
- **#38** L15 head 8/13 cross-modality bleed — 96 violations on
  OUTPUT_LO/HI; new attention-side bug.

### 10.5 Revised pass-rate trajectory

| Head | Pass | Total | Notes |
|------|-----:|------:|-------|
| `794155e` (06-01 baseline) | 232 | 1096 | 8-shard sweep; pre-`887164b` SP-marker fix; pre-D2 revert |
| `e18885a` (round-4 HEAD) | TBD | 1096 | post-`887164b` (+12/137 expected on shard 548-684); post-L14/model_ops claims backfill; D2 revert near-zero; no fresh full sweep |

### 10.6 Revised path-to-100% estimates

The C9-v2 §4 estimate (D2 → +50 to +75 ids on bug #27) is **revised
downward to ~0** based on the actual D2 revert outcome. Updated
realistic targets:

| Cluster | Expected impact | Effort | Source |
|---|---|---|---|
| L10 PSH addr0_e0 OP_ENT guard (#33 subcomponent) | ~97 rows | 0.5-1 day | wide_alu_triage §8 |
| Post-LEV AX corruption (#33 main) | ~141 rows | 2-3 days, L16 + L6 coordinated | stack_jsr_lev_triage §Recommendation |
| Long-division migration (#36) | ~109 rows | 3-5 days FFNRule IR | wide_alu_triage §8 |
| Single-byte MUL (#34) | ~46 rows | 1-2 days | wide_alu_triage §8 |
| Uniform-216 sentinel sub-fix (subset of #34) | ~44 rows | 1-2 days | wide_alu_triage §7 |
| Wide-MUL multi-byte / high-byte (#35) | ~52 rows | 2-3 days, attention-verifier-aided | wide_alu_triage §8 |

Sum of the surgical-via-verifier targets: ~440-490 rows over 1-2 weeks
of focused work, **assuming each cluster fix lands cleanly with no new
regressions** (per memory note `feedback_single_rule_fixes_are_zero_sum.md`,
single-rule fixes have a 0/5 net-positive batting average — these
estimates are upper bounds, not commitments). Realistic ceiling
without architectural changes: **~530-720 pass (32-66% of 1096) over
1-2 weeks**. **100% requires** attention-verifier V2 (head-level
strength-and-sign algebra), 17-dim slot 99-115 exhaustion (B8 wave),
and structural migrations for long-division and per-byte ALU carry.

### 10.7 Revised recommendations (round 4)

Replaces / supplements C9-v2 §9 with measured 06-01 data:

1. **Re-run the 8-shard sweep at `e18885a`** to update Section 2 with
   the post-round-4 headline. The bisect baseline at `794155e` shows
   232; the `887164b` SP-marker merge alone is documented as +12 / 137,
   so the most likely post-round-4 headline is ~240-244. **Until that
   sweep runs, all "current pass rate" numbers in this doc are stale.**
2. **Prioritize bug #33 over bug #27**. #33 is the biggest single-cluster
   target (~141 rows), localized to one op family (`layer16_lev_routing`),
   and surfaces on a population (`func_identity_*`, `func_square_*`,
   `rec_factorial_*`) that is currently 0-pass — i.e. high signal, low
   regression risk. #27 has been partially attacked with low impact and
   the attention-side root (#37) is not yet actionable.
3. **Stand up the attention-verifier V2** (`attention_verifier` is V1 at
   `34497f1`). Bugs #37 / #38 are not currently localizable to specific
   rules; V2 needs head-level strength algebra with sign-aware dominance
   (per the S-2-followup pattern landed at `bddf10f`).
4. **Migrate long-division to FFNRule IR** (#36). 109 rows are blocked
   on the multi-nibble SLOT_REMAINDER → OUTPUT_LO/HI projection through
   L11-L17. Single-rule attempts are zero-sum here per the memory note;
   this needs an IR pass.
5. **Drop conv-I/O / PRTF from the 1096 attack surface** — it's out of
   scope for this metric. Re-target the V9 plan against the dedicated
   conv-I/O test suite.
6. **Use `attribute_failures.py` as the standard triage entry point**
   going forward. 237/237 attribution coverage on shard 0 + 548 is the
   first time we have per-row candidate-rule mapping on the corpus;
   B7-9-style first-fatal histograms are now subsumed by this tool.
7. **Land the B8 slot-99-115 allocation** (scaffolded at `bcd9d04`;
   #B8-A `ADDR_B1_VALID` + `ADDR_B2_VALID` already landed at `f869367`).
   17 - 2 = 15 dims remain; the next wave should target a 32-dim
   `SP_BYTE0_VALUE` one-hot to address the SP_byte0/1 cluster
   architecturally rather than via more L10 tail rules.

---

## Cross-references

- D1 post-merge retest: `git show origin/audit/post-merge-retest-a03f600:.agent-logs/post-merge-retest-a03f600/_run.log`
- D1 prior trough retest: `git show origin/audit/post-merge-retest-e00709f:.agent-logs/post-merge-retest-e00709f/_runner.log`
- B7-9 lowering revalidation: `git show 6e8ab77:.agent-logs/lowering-revalidation-v4/SUMMARY.md`
- B6-G L7-L9 audit: `git show origin/investigation/l7-l9-structural-audit:.agent-logs/l7-l9-structural-audit/REPORT.md`
- B6-K BD dim usage map: `git show origin/investigation/bd-dim-usage-map:.agent-logs/bd-dim-usage-map/REPORT.md`
- B6-L L17 post-op inventory: `git show origin/investigation/l17-post-op-inventory:.agent-logs/l17-post-op-inventory/REPORT.md`
- B4-H L10 refactor PLAN: `git show origin/proposal/l10-tail-correction-family:.agent-logs/l10-tail-family-refactor/PLAN.md`
- B7-6 SP-marker rewrite: commit `cdd0d0a`
- B7-7 tail-addr rewrite: commit `5924bb4`
- Soft ADDR_B0 evidence (regression source): commits `66d9e12` (B5-D) + `d4b2a90` (B6-B)
- Per-op harness root: `.agent-logs/` per-batch subdirectories
- Sub-agent brief defaults: `~/.claude/projects/-home-alexlitz-Documents-misc-c4-release/memory/feedback_agent_briefs.md`

### Round 4 (2026-06-01) artifacts

- 06-01 sweep baseline: `.agent-logs/sweep-2026-06-01/shard_{0,137,274,411,548,685,822,959}.log` (232 ok / 864 diverge / 0 error).
- Wide-ALU triage: `.agent-logs/wide_alu_triage_2026_06_01.md` (525 in-corpus rows; bugs #34, #35, #36).
- Stack / JSR / LEV triage: `.agent-logs/stack_jsr_lev_triage_2026_06_01.md` (375 in-corpus rows; bug #33).
- Conv-I/O / PRTF triage: `.agent-logs/conv_io_prtf_triage_2026_06_01.md` (0 in-corpus rows; category OUT-OF-SCOPE).
- Per-row attribution: `.agent-logs/attribution_2026_06_01.md` (237/237 on shard 0 + 548).
- No-claims survey: `.agent-logs/no_claims_survey_2026_06_01.md` (73/115 ops empty-claims; 25 should-backfill).
- Slot 99-115 allocation proposal: commit `bcd9d04`; first allocations at `f869367` (B8-A).
- L10 strength/scope cleanup: commit `775bd79` (538 scope violations → 0; 52691 strength → 43685).
- L16 verifier honesty: commit `e23155b` (197 V1 violations → 0; no behavior change).
- Attribute-failures tool: commit `f7c4e0f`; usage embedded in the attribution-2026-06-01 doc.
- Attention verifier V1: commit `34497f1`; surfaces bugs #37 / #38.
- Per-op L1/L10 harness merge: commit `7e27ad0`.
- L10 SP-marker ENT-blocker fix: merge `887164b` (+12 / 137 on shard 548-684).
- L14 claims backfill chain: commits `caaa8f6 → d43ea7d → 1aac36f → cfe5d52 → a173f32 → 298ad08 → e18885a`.
- model_ops claims backfill: commits `760b010 → 701c0d8 → 91ea7d3 → 6be03a2`.
- D2 revert (near-zero impact): commits `361357a` (revert) → `65f80f1` (merge).
- Symbolic-vs-lowered probe: `.agent-logs/symbolic-bounds-checks.log` (refutes 1e16+ delta hypothesis).
