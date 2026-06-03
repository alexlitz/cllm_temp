# IR incremental improvements

Small concrete improvements to the IR, each shippable in 1-2 sessions
and delivering standalone value. No big-bang refactors. The unifying
goal: shrink the bug surface and make the next bug easier to find.

Status: **DRAFT — pending user review.**

---

## Ordering rationale

Stop trying to redesign the IR while the smoke tests are broken.
Get smoke green first; type safety and DSLs can follow once we have a
working baseline.

Order: smoke → cluster fixes → tooling → cleanup → small refactors.
Each item is independent.

---

## Step 1: Unblock smoke (priority 0)

Smoke has been 12 pass / 39 fail since the session started. Multiple
agents converged: **opcode flags ARE decoded at L5/L7 but ZEROED
between L7 and L14**. Something downstream over-clears OP_*.

**Action**: trace the residual stream at MARK_AX between L7 and L14 to
find the specific op/rule that zeros the opcode flags. Once found,
gate it correctly.

**Acceptance**: smoke recovers to ≥30/51 passing.
**Effort**: 1-2 sessions.

---

## Step 2: Strengthen one rule at a time (small + safe)

Today: rules describe "fire when conditions sum > threshold". The
L17 amplifier mass and the OP_IMM leak both showed how easily this
fires when not intended.

**Action**: for each rule with `strength > 10_000`, add an explicit
`dominates_at` scope. Audit corpus-wide via a tools script. Migrate
~20 rules.

**Acceptance**: 0 dominator writes without dominates_at.
**Effort**: 1 session.

---

## Step 3: Bare-literal lint (one-shot)

Today: `d_model = 512` and `n_heads = 8` literals scattered across
bake_fn bodies (5 HIGH-risk per the audit). The L10 compile blocker
was one such bomb.

**Action**: add a static linter check + migrate the 5 HIGH-risk sites
to receive d_model from a typed model_config object.

**Acceptance**: 0 bare-literal shape values in active bake bodies.
**Effort**: 1 session.

---

## Step 4: SSA value distinction (small refactor) — LANDED

Today: `DIM` and `DIM.*.-1` resolve to the same physical slot at
lowering. The OPCODE_BYTE_LO bug was an instance of this collapse.

**Action**: in dim_registry, distinguish step=0 and step=-1 as
separate logical values. Add a runtime check at compile: every
cross-step read has an explicit producer.

**Acceptance**: compiler-step0 safety (`ef6ef561`) becomes a hard
error in strict mode.

**Status**: shipped. `compile_full_vm_dynamic(strict=True)` now raises
`CrossStepReadError` for any `(consumer, ssa_read)` pair that finds a
same-step writer AND is not in the baseline allowlist. The bundled
`CROSS_STEP_BASELINE_ALLOWLIST` (82 entries at landing) preserves the
existing behaviour — every entry is a TODO to migrate the read away
from the cross-step alias. A regression ratchet
(`test_baseline_allowlist_size_matches_step4_landing`) caps the
allowlist at 82 so future commits must shrink, not grow it. Callers
can pass `cross_step_baseline_allowlist=[]` to promote every finding
to an error (new bake authors), pass a custom subset for incremental
migration, or pass `None` (default) to use the baseline.

**Effort**: 1-2 sessions.

---

## Step 5: Drop produces/consumes_fresh as separate fields

Today: 7 ops have both fields, 11 have produces only, 4 have
consumes only, 120 have neither. The fields duplicate what the rules
already declare.

**Action**: make `produces`/`consumes_fresh` computed properties on
`Operation` derived from the rules' contents. Delete all the
explicit declarations.

**Acceptance**: 0 explicit `produces=` / `consumes_fresh=` fields in
op declarations. All derive automatically.
**Effort**: 1 session.

---

## Step 6: Auto-attribute 1096 failures (tooling)

Today: 1096 failures get traced manually per-cluster. Each takes
hours.

**Action**: build `tools/attribute_1096_failure.py` that, given a
failing test, runs the model + the symbolic interpreter, and reports
the FIRST op whose `produces` claim doesn't match the runtime
residual. Auto-generates a fix brief.

**Acceptance**: any 1096 failure produces a one-page attribution doc.
**Effort**: 1-2 sessions.

---

## Step 7: Strict-mode compile gate (tooling)

Today: `strict=True` and `strict=False` modes exist but the strict
checks are scattered. The compiler-step0 safety warning is one such
check.

**Action**: collect all strict-mode checks into a single
`compile --strict --verify` mode that runs every check + reports a
unified manifest. Add to CI.

**Acceptance**: a single command verifies the entire IR against all
known invariants.
**Effort**: 1 session.

---

## Step 8: Smoke pass-count CI metric (tooling)

Today: smoke pass count is measured manually; no historical record.

**Action**: add `tools/smoke_track.py` that runs smoke + emits a JSON
manifest. Compare across commits to spot regressions instantly.

**Acceptance**: every commit auto-reports smoke pass delta.
**Effort**: 1 session.

---

## Step 9: Worktree + cache hygiene

Today: 590 stale worktrees from prior sessions. Compile cache is 30 GB
(LRU evictor just landed).

**Action**: mechanical cleanup of locked `.claude/worktrees/agent-*`
worktrees whose branches are merged into main.

**Acceptance**: worktree count drops to <50.
**Effort**: 1 session.

---

## Step 10: Document handoff for next session

Today: 30+ docs in `c4_release/docs/`. Hard to know which apply.

**Action**: write `c4_release/docs/STATUS.md` that's auto-updated per
commit and lists: what works, what's broken, what's in flight,
priority next steps.

**Acceptance**: a new session can start without reading 30 docs.
**Effort**: 1 session.

---

## Optional follow-ups (each independent, do later)

- **Spec DSL POC**: 1 opcode (IMM) compiled from a 2-line Python
  function. Validates the V3 vision is tractable. (2-3 sessions.)
- **Multi-step protocol library**: wide_add + wide_sub helpers used
  by hand-written rules today. (3-5 sessions.)
- **Single ALU group lowering**: AddSub from declarative IR →
  efficient composite. (3-5 sessions.)

These are bigger; defer until 1-10 are done.

---

## What this does NOT do

- No big architectural rewrite.
- No 18-session migration plan.
- No new type system shipped end-to-end before smoke works.

Each step is small. The compounding effect after 5-6 steps is the
"V2 type safety" payoff without the 23-session up-front commitment.

---

## Total effort

| Step | Sessions |
|---|--:|
| 1. Smoke unblock | 1-2 |
| 2. Strengthen rules | 1 |
| 3. Bare-literal lint | 1 |
| 4. SSA value distinction | 1-2 |
| 5. Drop produces fields | 1 |
| 6. Auto-attribute 1096 | 1-2 |
| 7. Strict compile gate | 1 |
| 8. Smoke tracker | 1 |
| 9. Worktree cleanup | 1 |
| 10. STATUS doc | 1 |
| **Total** | **10-13 sessions** |

Half the cost of the V2-only plan; more concrete value per step.
