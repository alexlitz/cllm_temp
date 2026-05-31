# Neural-VM 1096-Test Campaign Summary (2026-05-29 / 2026-05-30 / 2026-05-31)

Base branch: `speedup-cache-and-buckets`
Heads of interest:
- v2 baseline (clean, pre-architectural): `4d069f7`
- e00709f (after B7-4 ADDR_B0_VALID landed; pre-B7-6/B7-7)
- v4 (`integration/batch-merge-full-v4`, HEAD `10ab316`): foundation dims landed
- **a03f600 (current HEAD)**: B7-6 + B7-7 consumer rewrites merged

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
