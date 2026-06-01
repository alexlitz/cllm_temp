# Dynamic Scheduler Migration Plan (Phase B — Strict Path)

_Status: PLAN. Branch `speedup-cache-and-buckets` (HEAD ~`3650e01`)._
_Author: scheduler migration sub-agent, 2026-06-01._

This document is the canonical execution plan for retiring the hardcoded
`phase=N.M` ordering field from `Operation` and replacing it with a
fully-dep-derived layer scheduler (`compile_full_vm_dynamic`). It is
load-bearing for the next ~2 weeks of compiler work. Subsequent agents
should be able to start any unit (B9 … B15) below without re-doing the
Phase A analysis.

The strict variant (this plan) **actually deletes** the static-phase
fallback. Phase pruning in `LayerCompiler._topological_sort`
(`layer_compiler.py:1030+`) becomes a hard error: a cycle in the
declared-dep graph is treated as a missing declaration, not a tie to
break.

The hybrid alternative (keep `phase` as a tiebreaker forever) is rejected
because it leaves the 27 `phase_required_but_undeclared` ops in a state
where inserting a corrective op still forces a manual re-numbering pass
— the exact problem Phase A was scoped to fix.

---

## 1. Executive summary

What this migration delivers:

1. **`compile_full_vm_dynamic` as a public API** alongside the existing
   `compile_full_vm` (see `full_vm_compiler.py:389`). The dynamic variant
   derives layer assignment from the declared dependency DAG; the static
   path remains available for byte-identity regression testing during
   the rollout.
2. **Auto-pushing ops via declared deps.** Inserting a corrective op
   no longer requires choosing a phase number (and consequently no
   longer requires bumping every downstream phase to make room). The
   op declares what it reads/writes/produces/consumes and the scheduler
   places it.
3. **Layer count derived from DAG depth.** `n_layers =
   max(depth(op)) + 1` (computed during compile). Today the count is
   the `max(floor(phase))` of any block-kind op — a hand-set hand-off
   between bake author and compiler. Phase A confirms the dep DAG depth
   is **4** while the static layout uses **17** layers; the remainder
   is hidden ordering we must surface as explicit declarations or
   acknowledge as freely-placeable.
4. **No more manual `phase=N.M` renumbering** when inserting corrective
   ops. After B15 the field is deleted from `Operation` entirely.

Sources cited throughout:
- Phase A diagnostic: `.agent-logs/scheduler_phase_a_2026_06_01.md`
  (worktree path
  `.claude/worktrees/agent-a8b15867e5989a2f1/.agent-logs/scheduler_phase_a_2026_06_01.md`;
  CSV companion at same path with `.csv` extension).
- Phase A tool: `c4_release/tools/analyze_scheduler.py`
  (commit `103a481`).
- Slot 99-115 proposal:
  `c4_release/docs/SLOT_99_115_ALLOCATION_PROPOSAL.md`.
- Campaign summary: `c4_release/docs/CAMPAIGN_SUMMARY.md`.
- LayerCompiler internals: `layer_compiler.py:1030` (`_topological_sort`),
  `:739` (`consumes_fresh` check), `:786` (model-op phase sort).
- Postop attach phases:
  `c4_release/neural_vm/unified_compiler/ops/shared.py:191`
  (`phase = 1180 + layer_idx * 0.01`).

---

## 2. Goals and non-goals

### Goals

- G1. Replace `phase`-based topological tiebreaking with a strictly
  acyclic dep DAG over the ~113 production ops.
- G2. Land `compile_full_vm_dynamic` such that for the current op set
  it produces a layout that compiles, bakes, and passes the same
  per-op claim verification (Mode A) and 1096 corpus as
  `compile_full_vm`.
- G3. Surface every currently-hidden ordering constraint as an
  explicit declaration (`requires`, new `consumes_fresh`,
  `_THIS_STEP` / `_PREV_STEP` dim splits, or a documented
  `freely_placeable` exemption).
- G4. Make the DAG fully acyclic — break the single 57-op SCC reported
  by Phase A.
- G5. Retire `Operation.phase` and the `_topological_sort` phase-pruning
  branch (delete dead code; the field becomes unused).

### Non-goals

- N1. **Re-layouting trained weight cells.** This migration must keep
  every dim's `(layer, head, slot)` assignment byte-identical to the
  current static layout for the released bake. Layout drift is allowed
  only inside a controlled "strict bake-in" window (see B14) and only
  if it does not invalidate trained weights.
- N2. **Changing op semantics.** No corrective-op insertion, no rule
  rewrites, no new evidence channels beyond what the dim
  decomposition (B9) strictly requires. Bug-fix work continues on
  separate branches.
- N3. **Reworking the slot 99-115 allocation proposal.** B9 must
  coordinate with B8-A..D for any slot it newly consumes, but does
  not displace that proposal.
- N4. **Changing the dispatcher.** `build_model_from_layout` and
  `AutoregressiveVM.forward` are unchanged; the dispatcher is
  already dep-derived for `ops_per_layer`.

---

## 3. Acceptance criteria

| # | Criterion | Measurable today? | How verified |
|---|-----------|-------------------|--------------|
| AC1 | `compile_full_vm_dynamic` exists as a public API in `neural_vm.unified_compiler.full_vm_compiler` | NO — needs B11 | Import test + signature check |
| AC2 | Strict mode (no phase fallback) compiles the full canonical op set | NO — needs B11 + B9 + B12 | `compile_full_vm_dynamic(strict=True)` returns a valid `AutoregressiveVM` |
| AC3 | Strict mode passes per-op claim verification (Mode A) | YES (infra exists) — needs B11 to wire it | `python tools/verify_op.py` against the dynamic bake |
| AC4 | Strict mode passes `test_symbolic_declarative_runner_passes_full_1096_suite` (`tests/test_symbolic_declarative_program.py:55`) | YES (infra exists) — needs B11 | Pytest |
| AC5 | Strict-mode 1096 sweep is within **N=0 ids** of the static path | YES (infra: shard runner exists at `.agent-logs/fast-shards-*`) — decision pending: see open question Q1 | Compare per-shard pass lists; assert empty symmetric diff |
| AC6 | Every op declares enough deps to be either `phase_pinned_by_deps` or `freely_placeable` in the Phase A categorisation | YES — re-run `tools/analyze_scheduler.py` after B12 | `phase_required_but_undeclared == 0` AND `phase_inconsistent_with_deps == 0` |
| AC7 | Dep DAG is acyclic when phase pruning is disabled | YES — re-run analyzer after B9 | `dep_graph_cycle_member == 0` |
| AC8 | `Operation.phase` field has no remaining production reads outside the deprecated `compile_full_vm` static path | YES (grep) — see B15 | `grep -rn "\.phase" neural_vm/unified_compiler/` returns no live references after B15 |
| AC9 | Inserting a new op with declared deps but no `phase` value picks up a sensible layer assignment in dynamic mode | NO — needs B11 + B13 CI gate | Smoke test: register a stub op, observe layer assignment |
| AC10 | A new agent can read this doc and execute B9..B15 without re-doing Phase A | YES (this doc) | Editorial review |

AC1, AC2, AC8, AC9 require new tooling/code; AC3, AC4, AC5, AC6, AC7,
AC10 are measurable today (or with a single re-run of an existing tool).

---

## 4. Risk surface and mitigations

| Risk | Likelihood | Impact | Mitigation | Confidence |
|------|------------|--------|------------|------------|
| R1. Strict dep-based ordering changes the layout → trained weight cells move → released bake becomes invalid. | HIGH (Phase A says only 3/113 ops match dep-derived earliest; 28 non-cycle ops would float earlier) | HIGH | B11 lands with a `layer_pin` field on ops that records the current static layer. The dynamic compile then asserts cell-by-cell identity. After bake-in (B14), pins are removed one-at-a-time per op family with a 1096 retest gate. | MED |
| R2. Cycle-breaking via dim decomposition (B9) introduces new dims that compete with the slot 99-115 proposal (B8-A..D). | MED | MED | B9 explicitly enumerates which new dims it needs (see Section 6). Coordinate with B8 owners via Section 9 cross-reference; allow B9 to consume slots only if B8-B/C/D have not yet landed. | HIGH |
| R3. `decl_verifier` and per-op tests assume the current layer indices; many tests would break under strict mode. | HIGH | MED | Phased rollout per layer family (B14). Tests use `current_layer` from the compiled layout, NOT a hardcoded number. Audit tests in B11 prep. | MED |
| R4. The 27 `phase_required_but_undeclared` ops need new declarations; backfilling can introduce FALSE constraints (e.g. declaring a `requires` that is not actually load-bearing) that pin ops unnecessarily. | MED | MED | B12 declares one op at a time; each op's new dep set is reviewed against the corresponding `_set_layerN_*` bake helper to confirm the dep is real. The analyzer re-runs after each commit. | MED |
| R5. Strict mode requires every block-kind op with `layer_idx=N` pin to either justify the pin or convert to a `target_op_name` reference. Some pins are load-bearing (L0 is the embedding-aware boundary). | MED | LOW | B11 introduces a `layer_pin: int \| None` field documented as "this op MUST occupy this layer for structural reasons" (e.g. L0's `phase_a_ffn` lives at layer 0 because the embedding bake assumes it). The pin is opaque to the scheduler other than as an equality constraint. | HIGH |
| R6. Postop attach ops (`phase=1180.NN`) are currently `freely_placeable` per Phase A. Their actual constraint is "must run immediately after the corresponding ALU op". | HIGH | LOW | B10 (`requires` op-name references) gives them an explicit `requires={"after": "layerN_alu"}` instead of relying on the magic 1180 phase. | HIGH |
| R7. SCC #1 (57 ops) is dominated by `OUTPUT_HI` cycles (81 back-edges, per Phase A line 47). If the decomposition is wrong, every L3..L15 op shifts. | HIGH | HIGH | B9 lands `OUTPUT_HI_THIS_STEP` / `OUTPUT_HI_PREV_STEP` behind a flag first, runs the analyzer, then rewires consumers one at a time. Each rewire is its own commit and 1096-tested. | LOW-MED |
| R8. The hybrid window (B11 ships with phase as a tiebreaker; B14 turns it off) could regress silently — agents land code assuming dynamic mode equals static mode and only B14 surfaces the divergence. | MED | MED | CI gate in B13 runs strict-mode 1096 on every PR touching `ops/` or `compiler.py`. | MED |
| R9. Removing `phase` (B15) breaks downstream tools that introspect it (e.g. analyzers, the existing `tools/analyze_scheduler.py`). | LOW | LOW | Grep before B15; update tools to use `compiled_layer = layout.layer_for(op)` instead. | HIGH |

Top 3 by combined likelihood × impact:
- **R1** (layout drift invalidates weights) — confidence MED; the
  `layer_pin` mitigation is straightforward but the bake-in window
  (B14) is the highest-stakes hand-off.
- **R7** (OUTPUT_HI decomposition errors) — confidence LOW-MED;
  81 back-edges means the surface area is large.
- **R3** (test breakage) — confidence MED; the test audit is in scope
  for B11 but the actual breakage count won't be known until B14.

---

## 5. Phase plan

### Convention

Per the campaign convention (CAMPAIGN_SUMMARY.md §2, §6), structural
units are named with a single-letter prefix and a sequence number. B7
covered the lifecycle-bit additions (slots 95-98). B8 is the in-flight
slot-99-115 wave (SLOT_99_115_ALLOCATION_PROPOSAL.md). This migration
takes **B9..B15**.

Each unit lists: scope, dependencies, effort estimate (in
agent-days), and an explicit go/no-go signal that the next unit can
start.

### B9 — Dim decomposition wave (cycle breaking)

**Scope.** Introduce `_THIS_STEP` / `_PREV_STEP` (or equivalent
suffixes) for every dim family that produces back-edges inside the
largest SCC. Required families per Phase A line 47-56:

- `OUTPUT_HI` (81 back-edges) — split into
  `OUTPUT_HI_THIS_STEP` / `OUTPUT_HI_PREV_STEP`. See Section 6.1.
- `AX_CARRY_HI` (9 back-edges), `AX_CARRY_LO` (6 back-edges) —
  decomposition treats `AX_CARRY_*_THIS_STEP` (current ALU) vs
  `AX_CARRY_*_PREV_STEP` (carry-forward / residual). See 6.2.
- `ADDR_KEY` (8 back-edges) — decomposition treats memory-resolved
  addresses vs the fetch round's key. See 6.3.
- `TEMP` (7 back-edges) — likely needs renaming, not decomposing;
  `TEMP` consumers and producers may genuinely be in different
  ordering layers. See 6.4.
- `OP_LEV` (6 back-edges) — needs a `OP_LEV_DECODED_THIS_STEP` view
  vs the persistent register. See 6.5.
- `EMBED_HI` (3 back-edges), `ADDR_B0_HI` (3 back-edges),
  `CARRY` (3 back-edges), `OUTPUT_LO` (2 back-edges) — small enough
  to handle with single `requires` declarations rather than full
  decomposition (Section 6.6).

**Dependencies.** None (analyzer + dim_registry edits only).

**Effort estimate (8 agent-days):**
- 1.5d — `OUTPUT_HI` split (largest blast radius; ~30 producer/consumer
  edits across L3..L15)
- 0.5d — `AX_CARRY_HI` split
- 0.5d — `AX_CARRY_LO` split (couples to HI work)
- 0.5d — `ADDR_KEY` split
- 0.5d — `TEMP` audit + rename (may collapse into a `requires` edit if
  the rename suffices)
- 0.5d — `OP_LEV` split
- 1.0d — small-back-edge cleanup (EMBED_HI, ADDR_B0_HI, CARRY,
  OUTPUT_LO via `requires`)
- 1.5d — per-family per-op claim verification (Mode A) per dim
- 1.0d — analyzer re-runs + iteration; document any residual SCC

**Go/no-go signal.** `tools/analyze_scheduler.py` reports
`dep_graph_cycle_member == 0`. AC7 satisfied.

### B10 — `requires` field accepts op-name references

**Scope.** Today `Operation.requires` is a `Dict[str, str]` of residual
constraint strings (`layer_compiler.py:227`). Phase A's analyzer
opportunistically treats values that match an op name as a name
reference (`analyze_scheduler.py:137-143`), but the compiler ignores
this. Land an explicit, documented semantics:

- `requires["after"] = "<op_name>"` — this op must come strictly after
  the referenced op (treated as a dep edge by `_topological_sort`).
- `requires["same_layer_as"] = "<op_name>"` — this op must be assigned
  to the same `layer_idx` as the referenced op (used for postop attach
  ops; see R6).
- All other keys retain their current residual-constraint semantics.

Update `_topological_sort` and `compile_dynamic` (introduced in B11)
to honour `after` and `same_layer_as`. Backwards compatible: ops that
do not use these keys are unaffected.

**Dependencies.** None (independent of B9, but blocks B12 for the
postop attach migration).

**Effort estimate (1 agent-day):**
- 0.5d — schema + LayerCompiler edits
- 0.25d — unit tests
- 0.25d — analyzer update to consume the new schema natively
  (`analyze_scheduler.py:137-143` becomes load-bearing instead of
  a stub)

**Go/no-go signal.** A test op with
`requires={"after": "layer3_ffn"}` lands at a strictly-greater
`compiled_layer` than `layer3_ffn`.

### B11 — `compile_full_vm_dynamic` API (initial)

**Scope.** Land the API sketched in
`scheduler_phase_a_2026_06_01.md:631-721`. Initial implementation:

- Accepts a `strict: bool = False` parameter. Default mode keeps phase
  as a fallback tiebreaker (acyclic equivalence to today's static
  layout).
- When `strict=True`, phase pruning is disabled and cycles are hard
  errors.
- Adds `LayerCompiler.compile_dynamic(strict: bool)`.
- Adds an `Operation.layer_pin: Optional[int]` field for the
  load-bearing layer pins (R5 mitigation). Documented as "structural;
  not a tiebreaker."
- Asserts byte-identity to the static path when called with the
  current op set (the assert is gated by AC5's N=0 acceptance).

**Dependencies.** B10 (uses the new `requires` semantics for postop
attach ops). Soft dependency on B9 (strict mode requires no cycles —
without B9 the strict path is reachable only after a separate cycle
break).

**Effort estimate (2 agent-days):**
- 1d — API + compile_dynamic implementation
- 0.5d — `layer_pin` field + cell-identity assert
- 0.5d — byte-identity smoke test against the static path

**Go/no-go signal.** `compile_full_vm_dynamic(strict=False)` produces
the same `dim_positions` and `ops_per_layer` as `compile_full_vm`,
byte-identical. AC1 satisfied.

### B12 — Backfill declarations on the 27 `phase_required_but_undeclared` ops

**Scope.** For each op in Phase A's
`phase_required_but_undeclared` bucket (lines 64-126 of the report),
either:

(a) add the missing dep edge (a `reads`, `consumes_fresh`,
`requires["after"]`, or `requires["same_layer_as"]` declaration) so
the op becomes `phase_pinned_by_deps`; or

(b) explicitly mark it `freely_placeable` (no constraint declared and
none needed).

The 27 ops decompose into roughly four sub-buckets:

| Sub-bucket | Count | Pattern | Action |
|------------|-------|---------|--------|
| Postop attach (`lN_alu_postop_attach`, phase 1180.0N) | 6 (L8..L13) | `requires["same_layer_as"] = "layerN_alu"` via B10 | Mechanical |
| Convo-IO / tool-call / PRTF stubs (phase 3.x..7.x, no preds) | 11 | Either `requires["after"]` on the L_n upstream gather, or document as `freely_placeable` (most are gated by enable flags and produce nothing structural) | Per-op review |
| Layer1/2 threshold attns + layer2_lookback (no preds, current layer 1-2) | 3 | Add `requires["after"] = "layer0_threshold_attn"` or `produces`/`consumes_fresh` pair | Mechanical |
| L14 / L15 corrective ops (clear_addr_key_pollution, clear_output_corruption, etc., dep_depth=3 but pinned at 14-15) | 7 | These genuinely need to be near the end of the layer stack. Add `requires["after"]` on the latest op they overwrite (e.g. `layer10_alu`, `layer13_*`) | Per-op review |

**Dependencies.** B10 (so that `requires["after"]` is honoured).
Independent of B9 — the 27 ops are NOT cycle members.

**Effort estimate (2 agent-days):**
- 0.5d — postop attach sub-bucket (mechanical)
- 0.5d — convo-IO/tool-call/PRTF audit + decisions
- 0.5d — L14/L15 corrective ops per-op review
- 0.5d — re-run analyzer; iterate until
  `phase_required_but_undeclared == 0`

**Go/no-go signal.** `tools/analyze_scheduler.py` reports
`phase_required_but_undeclared == 0` AND
`phase_inconsistent_with_deps == 0`. AC6 satisfied.

### B13 — Freeze (CI gate; decline new ops without deps)

**Scope.** Add a CI test that runs `tools/analyze_scheduler.py` and
fails if either counter is non-zero after a future commit. Prevents
regression while B14 and B15 bake.

Also: extend `decl_verifier.py` (or add a sibling module) with a
"declarations-complete" check that runs in the per-op test suite.

**Dependencies.** B12 (the gate is meaningless until both counters are
already zero).

**Effort estimate (0.5 agent-day):**
- 0.25d — CI script + GitHub Actions wiring (or local pre-push hook)
- 0.25d — decl_verifier extension + per-op test integration

**Go/no-go signal.** A test PR that introduces an op with no deps and
no `freely_placeable` opt-in fails CI.

**How to flip the gate after B12 lands.** The CI test
`c4_release/tests/test_b13_dep_declaration_gate.py` is scaffolded but
defaults to `pytest.skip(...)` so it does not block today's CI while
the 21 `phase_required_but_undeclared` + 3
`phase_inconsistent_with_deps` + 66 `dep_graph_cycle_member` ops are
still on `main`. Once B12 backfill drives all three counters to zero,
flip the gate by either (a) exporting `B13_GATE_ENABLED=1` in the CI
workflow (one-line YAML change), or (b) committing an empty marker
file at `c4_release/docs/B13_GATE.flag`. Either signal lifts the skip;
the test then asserts every op landed in `freely_placeable` or
`phase_pinned_by_deps` and fails the build with the offending op names
grouped by bucket. There is intentionally no other knob — the flip is
auditable as a single commit.

### B14 — Strict-mode rollout (no phase fallback) + first dynamic 1096 sweep

**Scope.** Flip `compile_full_vm_dynamic` to `strict=True` for the
production path. Run the full 1096 sweep on the dynamic compile.
Compare to the static-path baseline (a03f600 = 238/1096 per CAMPAIGN
SUMMARY §2). Acceptable delta is N (see open question Q1).

Inside this unit:
- Remove `layer_pin` from any op where the pin was a tiebreaker
  rather than a structural requirement (re-test after each removal).
- Document any intentional layout drift (R1) in
  `BLOG_SPEC.md` or a new note.

**Dependencies.** B9, B10, B11, B12, B13 all landed.

**Effort estimate (2 agent-days):**
- 0.5d — flip the default + per-op claim verification
- 1d — full 1096 sweep + delta analysis (this is sharded; runtime is
  ~6h on `fast-shards-current`)
- 0.5d — `layer_pin` cleanup + re-test

**Go/no-go signal.** AC2, AC3, AC4, AC5 all satisfied. The 1096 delta
is within tolerance (Q1).

### B15 — Retire `phase` field

**Scope.** Delete `Operation.phase`, `Operation.layer_idx` (replaced
by `layer_pin` where load-bearing), and the static-phase code paths:

- `_topological_sort` phase-pruning branch (`layer_compiler.py:1030+`)
- The `model_ops.sort(key=lambda o: o.phase ...)` at line 786
  (replace with explicit `requires["after"]` declarations on each
  model op)
- The `consumes_fresh` check that compares phases at line 949+

Update `tools/analyze_scheduler.py` to use `compiled_layer` instead
of `phase`. Update `decl_verifier.py` similarly.

Delete the static `compile_full_vm`. (Optionally keep as a thin
wrapper that calls `compile_full_vm_dynamic`.)

**Dependencies.** B14 has shipped a strict-mode build to production
that has held for ≥1 sweep cycle.

**Effort estimate (1 agent-day):**
- 0.5d — code deletion + grep cleanup
- 0.25d — update `analyze_scheduler.py` and `decl_verifier.py` to
  the new schema
- 0.25d — final 1096 sweep on the cleaned-up code

**Go/no-go signal.** `grep -rn "\.phase" neural_vm/unified_compiler/`
returns no live references. AC8 satisfied. Migration complete.

### Dependency graph (units)

```
B9  (dim decomposition)  ─┐
                          ├──► B11 (dynamic API) ──► B14 (strict) ──► B15 (retire phase)
B10 (requires schema)  ──┤                       ▲
                          ├──► B12 (backfill 27 ops) ──┘
                          │
                          └──► B13 (CI gate) ──────────┘
```

B9 and B10 are independent and can run in parallel.

B11 needs B10. B12 needs B10. B11 and B12 can run in parallel after
B10. B13 needs B12 (gate is meaningless before B12 zeroes the
counters). B14 needs B9 ∧ B11 ∧ B12 ∧ B13. B15 needs B14.

### Total effort

8.0 (B9) + 1.0 (B10) + 2.0 (B11) + 2.0 (B12) + 0.5 (B13) + 2.0 (B14)
+ 1.0 (B15) = **16.5 agent-days**.

At a normal cadence of 1.5 agent-days per calendar day (parallel B9
+ B10, then parallel B11 + B12), this is achievable in **~10
calendar days** — within the 2-week budget.

---

## 6. Per-dim-family work (B9 detail)

For each cycle-causing dim, this section documents (a) the cross-step
readers, (b) the semantics, (c) the proposed decomposition, and (d)
slot impact.

Slot context: the slot 99-115 window has 17 free slots. B8-A claims
slots 99-100 (committed: `f869307`). B8-B/C/D have not yet landed.
The decomposition below tries to fit in slots **101-115** (15
remaining) without displacing B8-B's SP_BYTE0 sentinel family. If
B9 needs more than 15 slots, see open question Q2.

### 6.1 `OUTPUT_HI` (81 back-edges)

**Cross-step readers.** `layer3_carry_forward_attn` reads `OUTPUT_HI`
from the previous step's output (it is the residual of the prior
fwd pass). Producers like `nibble_copy_ffn`, `layer6_routing_ffn`,
and `layer8_multibyte_routing` write `OUTPUT_HI` during the current
step.

**Semantics.** The cross-step pattern is "the previous output is now
stale, this step is fresh." It is NOT genuinely persistent state —
the output of step N is consumed only by step N+1's L3
carry-forward, and only as a fast-zero target.

**Proposed decomposition.**
- `OUTPUT_HI_THIS_STEP` — written by current-step producers
  (L3..L15). Reset by the first FFN of the next step.
- `OUTPUT_HI_PREV_STEP` — written by `layer3_carry_forward_attn`
  from the previous step's `OUTPUT_HI_THIS_STEP`; read by any
  cross-step consumer that genuinely needs "what we output last
  time" (none today — the carry-forward is itself the only such
  reader, so this collapses into a no-op).

Net dim count: +1 if we keep both names; +0 if we rename
`OUTPUT_HI` → `OUTPUT_HI_THIS_STEP` and treat the residual as a
hidden register inside L3. Recommended: rename. The 81 back-edges
collapse because consumers now read `OUTPUT_HI_THIS_STEP`, and
`layer3_carry_forward_attn` reads it from the previous step's
register (not a declared dim) — which is exactly the case that
warrants a `requires["after"]` on every L3+ producer rather than a
data dep.

**Slot impact.** Rename is 0-slot. Adding a separate
`OUTPUT_HI_PREV_STEP` is 16 slots (mirror size). Not viable in
99-115. **Recommended: rename only.**

### 6.2 `AX_CARRY_HI` / `AX_CARRY_LO` (9 + 6 back-edges)

**Cross-step readers.** `layer8_head6_ax_carry_refresh`,
`layer8_multibyte_fetch` read AX_CARRY in the same step as
`layer6_relay_heads` / `layer6_attn` write it. The back-edge is
that earlier L8 ops also write to AX_CARRY through different lanes.

**Semantics.** AX_CARRY is the ALU carry pipe. The "back-edge" is
that L6 produces a refreshed carry from the prior step's ALU
output, while L8 ALU produces *this* step's carry. The two are
distinct concepts that share a name.

**Proposed decomposition.**
- `AX_CARRY_HI_THIS_STEP` / `AX_CARRY_LO_THIS_STEP` — produced by
  L8 ALU during the current step.
- `AX_CARRY_HI_PREV_STEP` / `AX_CARRY_LO_PREV_STEP` — produced by
  L6 from the prior step's L8 ALU output; consumed by `layer8_*`
  multibyte fetch as the carry-in.

This is a real two-register split (not a rename) — both registers
are live simultaneously inside L7/L8. Slot impact: +4 dims (each
AX_CARRY_* is 1 dim).

**Slot impact.** 4 slots. Fits in 101-104 if B8-B/C is deferred.

### 6.3 `ADDR_KEY` (8 back-edges)

**Cross-step readers.** `layer7_memory_heads`,
`layer14_addr_key_neural_decode` read; `layer5_fetch` and
`_layer5_fetch_dep_anchor` write.

**Semantics.** `ADDR_KEY` is the memory-fetch key used by L5+ heads.
The back-edge from L14 → L5 is because L14's "neural decode"
infers an `ADDR_KEY` that L5 will use **on the next step**.

**Proposed decomposition.**
- `ADDR_KEY_THIS_STEP` — written by L5 fetch heads in the current
  step. (Rename of today's `ADDR_KEY`.)
- `ADDR_KEY_FOR_NEXT_STEP` — written by L14 neural decode;
  consumed by L5 on the next step.

Two distinct registers. Slot impact: +1 dim (L14 already writes
into a residual; the new dim makes the residual explicit).

**Slot impact.** 1 slot. Fits in 105.

### 6.4 `TEMP` (7 back-edges)

**Cross-step readers.** `layer14_temp_clear` writes `TEMP`;
`layer6_routing_ffn`, `layer7_memory_heads` read.

**Semantics.** `TEMP` is a scratch register family. The L14 "clear"
is end-of-step hygiene; L6/L7 reads are next-step uses. This is
not a true cross-step dependency — L14 clearing TEMP means L6/L7
start with TEMP=0 next step, but the L6/L7 reads don't actually
depend on L14's write.

**Proposed decomposition.** No decomposition needed. The fix is
documentation + `requires` declarations: remove the
`reads={"TEMP"}` claim from `layer14_temp_clear` (it does not read,
it writes-to-zero) and confirm L6/L7 readers do not need a
`consumes_fresh` link to L14.

**Slot impact.** 0. This is a metadata fix, not a dim split.

### 6.5 `OP_LEV` (6 back-edges)

**Cross-step readers.** `opcode_decode_ffn` and
`_opcode_decode_ffn_dep_anchor` write; `layer3_carry_forward_attn`
and `layer3_ffn` read.

**Semantics.** `OP_LEV` is the LEV opcode flag. L3's carry-forward
reads the *previous step's* LEV (to decide whether to drop the
saved BP frame), but the L5 opcode decode produces the *current
step's* LEV.

**Proposed decomposition.**
- `OP_LEV_THIS_STEP` — written by L5 opcode decode; consumed by
  L7+ this-step rules.
- `OP_LEV_PREV_STEP` — written by an L0/L1 ALiBi relay (or by L3
  carry-forward itself); consumed by L3 next-step logic.

Slot impact: +1 dim. Fits in 106.

### 6.6 Small-back-edge cleanup (EMBED_HI, ADDR_B0_HI, CARRY, OUTPUT_LO)

These dims contribute 2-3 back-edges each. Per-dim decomposition is
not worth the slot cost. Instead, add a single explicit `requires`
declaration on the cycle-closing op for each.

Per Phase A lines 53-56:

- `EMBED_HI`: 3 back-edges from `layer4_pc_relay` to L3 ops. Fix
  with `requires["after"] = "layer3_ffn"` on `layer4_pc_relay`.
- `ADDR_B0_HI`: 3 back-edges from L9/L13 to `layer8_mem_to_alu`.
  Fix with `requires["after"] = "layer8_mem_to_alu"` on each
  reader.
- `CARRY`: 3 back-edges from L10 carry-relay to L9_alu. Fix with
  `requires["after"] = "layer9_alu"` on `layer10_carry_relay`,
  `layer10_carry_relay_bake`, `l10_post_ops_combined`.
- `OUTPUT_LO`: 2 back-edges from `layer8_alu`. Once OUTPUT_HI is
  renamed (6.1), these likely fall out — confirm by re-running the
  analyzer.

**Slot impact.** 0. These are dep declarations, not new dims.

### 6.7 Slot budget summary

| Family | New dims | Slots used |
|--------|---------:|-----------:|
| OUTPUT_HI (rename) | 0 | 0 |
| AX_CARRY split | 4 | 101-104 |
| ADDR_KEY split | 1 | 105 |
| OP_LEV split | 1 | 106 |
| TEMP (metadata) | 0 | 0 |
| EMBED_HI/ADDR_B0_HI/CARRY/OUTPUT_LO (requires) | 0 | 0 |
| **Total** | **6** | **101-106** |

This leaves **107-115** (9 slots) for B8-B/C/D. B8-B's SP_BYTE0_IS_*
family wants 3 slots (102-104 per the proposal); B9 takes 101-106;
B8-C wants 1 slot; B8-D wants 4-6 slots. With B9 displacing the
B8-B proposed slots, B8-B would need to land at **107-109** instead
of **102-104**. See open question Q2.

---

## 7. Test plan

Each unit (B9..B15) MUST pass the following checks before claiming
go/no-go signal.

### 7.1 Per-unit test gates

| Unit | Gate |
|------|------|
| B9 | Re-run `tools/analyze_scheduler.py`; assert `dep_graph_cycle_member == 0`. Per-op claim verification (`tools/verify_op.py`) passes for every op touched by the dim split. Per-rule strength verifier (`decl_verifier.py`) passes on the touched layers. |
| B10 | New unit test: an op with `requires={"after": "X"}` is scheduled strictly after X. |
| B11 | `compile_full_vm_dynamic(strict=False)` produces byte-identical `dim_positions` and `ops_per_layer` to `compile_full_vm`. Cell-by-cell layout compare across all baked weights. |
| B12 | Re-run analyzer; assert `phase_required_but_undeclared == 0` AND `phase_inconsistent_with_deps == 0`. |
| B13 | A test PR introducing an op with no deps fails CI. |
| B14 | All of: AC2, AC3, AC4. Full 1096 sweep on the dynamic strict path. Per-op symbolic-vs-lowered comparison. |
| B15 | AC8. Final 1096 sweep on the cleaned-up code; no regression vs B14's sweep. |

### 7.2 Per-op tests (Mode A claim verification)

Already exist in `tests/test_l*_per_op.py`. The strict-mode compile
should not change which claims are verifiable — only the layer index
the claim is verified at. Audit per-test asserts that hardcode a
specific `layer=N`; replace with `layout.layer_for("op_name")`.

### 7.3 Per-rule strength/scope verifier

`decl_verifier.py` reads from `(layer, head, slot)` triples. The
verifier should accept any layout that satisfies declared claims;
extend it to log when an op moves between layers across the
static-vs-dynamic compile.

### 7.4 Per-op symbolic-vs-lowered comparison

The symbolic declarative runner
(`tests/test_symbolic_declarative_program.py`) and the lowered
baked-VM runner must produce the same output token for every
sample in the 1096 corpus. Already gated by the existing test suite;
B14's go/no-go signal includes a re-run.

### 7.5 1096 spec-side test

`test_symbolic_declarative_runner_passes_full_1096_suite` at
`tests/test_symbolic_declarative_program.py:55`. Must pass under both
modes throughout the rollout.

### 7.6 Full 1096 sweep on the baked dynamic compile

Use the existing shard runner
(`.agent-logs/fast-shards-current` / `fast-shards-current-post-smoke`).
Compare per-id pass/fail lists; symmetric difference is the divergence
delta.

### 7.7 Cell-by-cell layout comparison vs static

For every `(layer, head, slot, weight_row, weight_col)` index, the
strict-mode bake must match the static bake exactly during the
hybrid window (B11..B13). Intentional divergences are allowed only
during B14 and must be documented.

---

## 8. Rollback plan

If B14 surfaces unacceptable regressions:

1. **First line of defence:** revert
   `compile_full_vm_dynamic` default from `strict=True` to
   `strict=False`. This restores the static-phase tiebreaker without
   touching ops, dims, or the dispatcher. ~1-line change.
2. **Second line:** revert the B12 declarations that caused the
   regression. Use the analyzer diff (before/after each B12 commit)
   to localise. Keep B9 dim splits (they are independent and the dim
   names persist).
3. **Third line:** revert B11 entirely. The static path
   (`compile_full_vm`) is unchanged through B11..B13, so the production
   bake is unaffected. B11 only ships a new function; deleting it
   has no production impact.
4. **Catastrophic line:** revert B9. Each dim split in B9 is a
   separate commit; revert in reverse order. The renames
   (`OUTPUT_HI` → `OUTPUT_HI_THIS_STEP`) require coordinated reverts
   of producer and consumer edits in the same commit.

The migration is designed such that B9 + B10 + B11 + B12 are all
shippable without flipping any default (B14 is the flip). Up to and
including B13, the production path is the static-phase compile. The
worst-case rollback before B14 is "delete the new code and we keep
moving."

After B14 ships strict-mode to production, weights are baked under
the strict layout. If a B14 regression escapes detection, the
release manager has two options:
- Rebuild the bake with `strict=False` (yields the pre-B14 layout
  and weights) and re-ship. ~6h.
- Hold the strict bake and fix forward via a B14.1 patch unit.

B15 is the point of no return — once `phase` is deleted, restoring
the static path requires reverting B15. Recommend ≥1 sweep cycle
between B14 ship and B15 land.

---

## 9. Cross-references

- **Phase A diagnostic.**
  `.agent-logs/scheduler_phase_a_2026_06_01.md` (counts at lines
  13-26; SCC list at lines 27-42; back-edge dims at lines 43-56;
  the 27 undeclared ops at lines 64-126).
- **Phase A tool.** `c4_release/tools/analyze_scheduler.py`
  (commit `103a481`).
- **Slot 99-115 proposal.**
  `c4_release/docs/SLOT_99_115_ALLOCATION_PROPOSAL.md` (B8-A
  landed at slots 99-100; B8-B claims 102-104 — coordinate with
  Section 6.7 above).
- **Campaign summary.** `c4_release/docs/CAMPAIGN_SUMMARY.md` (§6
  per-batch contributions; §10 round-4 wave; §10.5 pass-rate
  trajectory).
- **LayerCompiler.** `neural_vm/unified_compiler/layer_compiler.py`
  (`_topological_sort` at line 1030; `consumes_fresh` check at line
  739; model-op sort at line 786; `phase` field at line 136).
- **Static compile entrypoint.**
  `neural_vm/unified_compiler/full_vm_compiler.py:389`
  (`def compile_full_vm`).
- **Postop attach phase magic.**
  `neural_vm/unified_compiler/ops/shared.py:191` (`phase = 1180 + layer_idx * 0.01`).
- **Per-op test suite.** `tests/test_l*_per_op.py`.
- **1096 spec-side test.**
  `tests/test_symbolic_declarative_program.py:55`.
- **decl_verifier.** `neural_vm/unified_compiler/decl_verifier.py`.
- **Prior structural-migration patterns (B7-1..5).** Per Slot 99-115
  proposal §1.1, single producer + lifecycle bit + downstream
  consumer rewrite. B9 follows the same shape per dim family.

---

## 10. Open questions for the user

**Q1. 1096 delta tolerance for AC5.** Strict mode is allowed to
diverge from the static path by N ids on the 1096 sweep. What is N?

- Option A: **N = 0** (any divergence is a regression). Safest;
  forces every B9 / B12 edit to preserve behaviour exactly. May
  block B14 indefinitely if a benign reorder produces a different
  rule fire order in a rare sample.
- Option B: **N ≤ 5** (small drift acceptable as long as net pass
  rate ≥ static path). Pragmatic.
- Option C: **N ≤ 20**, with a per-id review of every divergence.
  Most permissive.

Recommendation: **N = 0 during B14 acceptance; allow N ≤ 5 with
documented justification for the final B15 sweep.**

**Q2. Slot displacement between B9 and B8-B/C/D.** Section 6.7 shows
B9 wants slots 101-106. B8-B (SLOT_99_115 proposal §2.2) wants
102-104. Both cannot have the same slots.

- Option A: B9 lands first (101-106), B8-B shifts to 107-109.
- Option B: B8-B lands first (102-104), B9's AX_CARRY split shifts
  to higher slots and `ADDR_KEY_FOR_NEXT_STEP` / `OP_LEV_PREV_STEP`
  land at 110-115.
- Option C: Halt B8-B/C/D until B9 is done, then re-evaluate slot
  pressure.

Recommendation: **B follow option C** — B8-B/C/D do not depend on
B9 but the slot accounting is cleaner if B9 commits first.

**Q3. Is `OUTPUT_HI_PREV_STEP` worth the 16 slots?** Section 6.1
recommends a rename rather than a true split because the only
cross-step reader is L3 carry-forward, which reads from the prior
step's register (not from a declared dim). Confirm: is there any
other consumer that genuinely needs "the previous step's output"
visible as a dim?

- If yes (16 slots): the proposal shifts AX_CARRY/ADDR_KEY/OP_LEV
  splits to the second wave.
- If no (recommended): rename only, no slot cost.

**Q4. Should `Operation.layer_pin` (R5 mitigation) become a permanent
field or get deleted in B15?** Some pins are load-bearing (L0's
embedding-aware boundary). After B15 these need to live somewhere.

- Option A: keep `layer_pin` as a permanent field; document the
  semantics.
- Option B: convert pins into `requires` declarations
  (`requires["same_layer_as"] = "embedding_bake"` etc.).
- Option C: delete; pinned ops become `freely_placeable` and rely
  on the bake helper's set_vm_weights ordering.

Recommendation: **Option A** — explicit is better than implicit;
load-bearing pins should be readable from the op declaration.

**Q5. Postop attach ops: is the `1180+layer_idx*0.01` phase scheme
load-bearing for any tooling outside `_topological_sort`?**
`shared.py:191` is the only definition site we found. Confirm there
are no analyzers, dashboards, or external scripts that grep for
`phase=1180` before B15 deletes the field.

---

## Appendix A: counts at a glance

(From `.agent-logs/scheduler_phase_a_2026_06_01.md` lines 13-26.)

- 113 ops analysed.
- 17 declared block-layer count vs **4** DAG-depth chain.
- 25 `freely_placeable`.
- 3 `phase_pinned_by_deps`.
- 27 `phase_required_but_undeclared`.
- 1 `phase_inconsistent_with_deps` (`phase_a_ffn` at layer 0,
  blocked by `layer0_threshold_attn`).
- 57 `dep_graph_cycle_member` (single SCC of size 56 + 1
  self-loop).

Top back-edge dims (line 47-56):

- `OUTPUT_HI`: 81 back-edges.
- `AX_CARRY_HI`: 9.
- `ADDR_KEY`: 8.
- `TEMP`: 7.
- `OP_LEV`: 6.
- `AX_CARRY_LO`: 6.
- `EMBED_HI`: 3.
- `ADDR_B0_HI`: 3.
- `CARRY`: 3.
- `OUTPUT_LO`: 2.

## Appendix B: glossary of categorisation buckets

(From `tools/analyze_scheduler.py:268-336`.)

- **`freely_placeable`** — no in-edges and no out-edges in the dep
  graph. Dynamic scheduler can place anywhere.
- **`phase_pinned_by_deps`** — current `floor(phase)` equals the
  dep-derived earliest layer. The static layout is already what a
  dep-derived scheduler would produce.
- **`phase_required_but_undeclared`** — current phase places the op
  strictly later than the DAG requires. Either (a) hidden ordering
  outside the declarations pins it, or (b) the dep declarations are
  incomplete and the dynamic scheduler would happily move it
  earlier (which may or may not be safe). The 27 ops in this bucket
  are the B12 work.
- **`phase_inconsistent_with_deps`** — current phase places the op
  strictly EARLIER than the dep graph allows. A real bug. Today: 1
  op (`phase_a_ffn`).
- **`dep_graph_cycle_member`** — in a strongly connected component;
  no topological order exists without breaking edges. Today: 57
  ops in a single SCC of size 56 + 1 self-loop. The B9 work.
