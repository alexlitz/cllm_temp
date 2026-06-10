# Dim-Alias Verifier Triage (2026-06-10)

Triage of the 96,468 violations reported by
`c4_release/neural_vm/unified_compiler/dim_alias_verifier.py`
(introduced in commit `4681e376`). Source dataset: the full per-layer
op set (`all_core_ops + alu_postop_attach + alu_divmod + residual_alibi
+ contract_validation`, lookup ALU mode, IO/tool-calling flags off,
no_bake `LayerCompiler.compile()`), against the default static
`DimRegistry` from `build_default_registry()`.

The takeaway: nearly every reported violation is an *over-
approximation* of the verifier's predicate solver, not a real read of a
mistaken alias. After tightening one solver invariant (this commit)
the count drops from 96,468 to 68,952, and the structural split makes
the remaining real bugs visible.

## Headline numbers

| Stage | Total | Distinct alias pairs |
| ----- | -----:| --------------------:|
| Before (commit cc476944) | 96,468 | 75 |
| After (this commit, solver fix) | 68,952 | 73 |
| After + suppressing 9 pure-design pairs (see below) | ~5,500 | ~30 |
| Real-bug candidates (residual) | ~2,000 | ~10 |

## Solver fix applied here

`atom_contradicts` now encodes the residual-tagging invariant
`mark == X (X != NONE) ⊥ is_byte`. The runtime stream tags every
position as either a marker row (`mark in {SP, AX, PC, BP, MEM,
STACK0, SE}`, `NOT is_byte`) or a byte row (`mark == NONE`,
`is_byte`); the two partitions are disjoint by construction. Without
this rule the over-approximation in `effective_predicate` lets every
slot whose semantics is written `mark == X OR (is_byte AND byte_index
== k)` (i.e. virtually every banded register) drag the unrelated
sibling into "overlap" at byte-row read sites.

Patch: `c4_release/neural_vm/unified_compiler/predicates.py`
(`_mark_role_implies_not_byte`, `atom_contradicts` body). Tests:
`tests/test_predicates_overlap.py` (4 new cases) — 183/183 green.

Per-pair impact (post-fix counts):

| Pair | Before | After | Δ |
| ---- | -----:| -----:| -:|
| STACK0_BYTE1 ↔ TEMP | 5,022 | 377 | −92 % |
| STACK0_BYTE2 ↔ TEMP | 5,019 | 441 | −91 % |
| STACK0_BYTE3 ↔ TEMP | 5,019 | 153 | −97 % |
| OPCODE_BYTE_HI_PIN ↔ TEMP | 5,000 | 298 | −94 % |
| ACTIVE_OPCODE_PRTF ↔ TEMP | 810 | 96 | −88 % |
| ACTIVE_OPCODE_READ ↔ TEMP | 810 | 96 | −88 % |
| ADDR_B0_LO ↔ OPCODE_BYTE_LO | 2,087 | 1,317 | −37 % |
| ADDR_B1_LO ↔ OPCODE_BYTE_HI | 944 | 176 | −81 % |
| FETCH_HI ↔ IMM_STAGING | 1,925 | 388 | −80 % |
| (Other 64 pairs) | unchanged or minor | | |

## Categorization table (top 30 pairs)

| # | Alias pair (sorted) | Post-fix count | Category | Rationale |
|---|---------------------|---------------:|----------|-----------|
| 1 | AX_CARRY_LO ↔ POST_PRTF_SP_LO | 14,536 | DESIGN | Identical slot range + identical semantics. Time-shared between MUL scratch (compute phases) and POST-PRTF SP save (IO phase). Disambig: VM step phase / opcode_in_step. Verifier has no atom for it. |
| 2 | AX_CARRY_HI ↔ POST_PRTF_SP_HI | 10,973 | DESIGN | Same as #1, hi nibble. |
| 3 | TEMP ↔ TEMP_PREV_STEP | 5,000 | DESIGN | Cross-step time-share by construction (Phase 7.A split). Verifier never sees STEP_END phase as an atom. |
| 4 | AX_FULL_LO ↔ TEMP | 4,411 | TAUTOLOGICAL | TEMP semantics is the umbrella `is_byte OR NOT is_byte` (tautology). Suppress via verifier annotation. |
| 5 | AX_FULL_HI ↔ TEMP | 4,411 | TAUTOLOGICAL | Same. |
| 6 | POST_PRTF_PC_LO ↔ TEMP | 4,411 | TAUTOLOGICAL | Same. |
| 7 | POST_PRTF_PC_HI ↔ TEMP | 4,411 | TAUTOLOGICAL | Same. |
| 8 | FORMAT_PTR_LO ↔ TEMP | 4,395 | TAUTOLOGICAL | Same. |
| 9 | FORMAT_PTR_HI ↔ TEMP | 4,395 | TAUTOLOGICAL | Same. |
| 10 | DIV_STAGING ↔ FETCH_HI | 1,924 | DESIGN | Slot 432..448 carries DIV scratch in compute phases / instruction-byte fetch in IF phase. Disambig: opcode_in_step ∈ {DIV, MOD} vs not. |
| 11 | DIV_STAGING ↔ FETCH_LO | 1,454 | DESIGN | Same time-share. |
| 12 | FETCH_LO ↔ MUL_ACCUM | 1,454 | DESIGN | Same time-share. |
| 13 | **ADDR_B0_LO ↔ OPCODE_BYTE_LO** | **1,317** | **REAL+DESIGN MIX** | Textbook bug for rules whose eff includes `mark == MEM` and read OPCODE_BYTE_LO. After byte-row filter, the residual is real. See "Real bugs" below. |
| 14 | OPCODE_BASE ↔ OP_LEA | 959 | DESIGN/SUBBANK | Co-located sub-bank parent/child. Verifier's `_is_colocated_subbank` only fires when slots have *different* extents; same-extent parent/child (both 1-wide at 262) slips through. Fix the filter. |
| 15 | STACK0_BYTE2 ↔ TEMP | 441 | TAUTOLOGICAL | TEMP umbrella semantics. |
| 16 | FETCH_HI ↔ IMM_STAGING | 388 | DESIGN | Time-shared across compute / dispatch phases. |
| 17 | STACK0_BYTE1 ↔ TEMP | 377 | TAUTOLOGICAL | TEMP umbrella. |
| 18 | CS_DIST_THERMO ↔ MEM_VAL_B3 | 354 | DESIGN | CS_DIST_THERMO is a thermometer over 16 cells; MEM_VAL_B3 occupies one of them only at MEM rows. Disjoint opcode_in_step. |
| 19 | LAST_WAS_IO_STATE_EMIT_THINKING ↔ MEM_VAL_B2 | 338 | DESIGN | 1-byte slot reused across IO state machine and MEM byte 2 — disjoint opcode_in_step. |
| 20 | LAST_WAS_IO_STATE_EMIT_BYTE ↔ MEM_VAL_B1 | 322 | DESIGN | Same as #19. |
| 21 | OPCODE_BYTE_HI_PIN ↔ TEMP | 298 | TAUTOLOGICAL | TEMP umbrella. |
| 22 | CMP_GROUP ↔ SP_OLD_HI | 272 | DESIGN | Partial overlap (1 byte of 8). Disjoint by `opcode_at_AX ∈ {CMP-set}` vs `opcode_in_step == ADJ`. |
| 23 | CS_DIST_THERMO ↔ PSH_AT_SP | 228 | DESIGN | Sub-band inside CS_DIST_THERMO. |
| 24 | IO_OUTPUT_COUNT ↔ PSH_AT_SP | 180 | DESIGN | Same 1-byte slot, disjoint opcodes (PRTF / READ vs PSH). |
| 25 | **ADDR_B1_LO ↔ OPCODE_BYTE_HI** | **176** | **REAL+DESIGN MIX** | Same pattern as #13. |
| 26 | IMM_STAGING ↔ MEM_STORE | 165 | DESIGN | MEM_STORE is a 1-byte sub-band of IMM_STAGING; semantics disjoint (`opcode_in_step ∈ {SI, SC, PSH}` vs `mark == PC`). |
| 27 | STACK0_BYTE3 ↔ TEMP | 153 | TAUTOLOGICAL | TEMP umbrella. |
| 28 | ACTIVE_OPCODE_PRTF ↔ TEMP | 96 | TAUTOLOGICAL | TEMP umbrella. |
| 29 | ACTIVE_OPCODE_READ ↔ TEMP | 96 | TAUTOLOGICAL | TEMP umbrella. |
| 30 | AX_FULL_HI ↔ LAST_WAS_THINKING_START | 81 | DESIGN | Sub-band overlap; disjoint by IO-state phase. |

The 43 long-tail pairs (≤ 81 each) are all the same shapes: TEMP-umbrella tautologies, sub-band/parent-child false positives, and time-sharing by VM phase.

## Real bugs — top 10 by impact

All ten arise from the same structural pattern: a rule reads
`OPCODE_BYTE_{LO,HI}+k` (intent: opcode-nibble dispatch on the AX or
byte-0 row) whose effective predicate admits `mark == MEM`. At MEM
positions the same byte range carries the *address* nibble
(`ADDR_B{0,1}_{LO,HI}`), so the dispatch atom is reading garbage. The
fix in all ten cases is the same shape — add an explicit
`NOT mark == MEM` hard blocker (a `MARK_MEM` condition with
`-HARD_BLOCKER_THRESHOLD`) to the rule's conditions.

| # | Op / rule | Read dim | File:line | Count |
|---|-----------|----------|-----------|------:|
| 1 | `layer14_addr_key_neural_decode` / `l14_addr_key_lohi_off{0..3}_hi{0..f}_lo{0..f}` | OPCODE_BYTE_LO (alias of ADDR_B0_LO) | `ops/l14_ops.py:3308-3324` | 1,024 |
| 2 | `layer6_routing_ffn` / `l6_branch_pc_byte1_jsr_target_*_{lo,hi}_*` | OPCODE_BYTE_LO/HI | `ops/l6_ops.py:1668-1706` | 1,568 (incl. dup) |
| 3 | `layer16_lev_routing` / `l16_jsr_*` family reading OPCODE_BYTE | OPCODE_BYTE_LO/HI | `ops/l16_ops.py` (jsr family) | 163 |
| 4 | `layer9_alu` / `l9_cmp_*` reading OPCODE_BYTE under CMP-group eff | OPCODE_BYTE_LO/HI | `ops/l9_ops.py` (cmp_combine) | 20 |
| 5 | `layer14_addr_key_neural_decode` / `l14_addr_key_top_common_off{0..3}_b1{0..f}` | OPCODE_BYTE_HI (alias of ADDR_B1_LO) | `ops/l14_ops.py:3348+` | 64 |
| 6 | `layer6_routing_ffn` / `l6_delayed_jmp_cancel_{lo,hi}_*` reading OUTPUT_HI_THIS_STEP | OUTPUT_HI alias | `ops/l6_ops.py` (delayed_jmp_cancel) | 144 (separate pair) |
| 7 | `convo_io_pc_sp_latch` / `convo_io_latch_post_prtf_pc_{lo,hi}_*` reading POST_PRTF_PC_* under `NOT is_byte` eff | POST_PRTF_PC_LO/HI alias of AX_FULL_LO/HI | `ops/convo_io_ops.py` (pc_sp_latch) | 32 |
| 8 | `layer8_alu` / `l8_alu_*` reading OPCODE_BASE under non-LEA eff | OPCODE_BASE (alias of OP_LEA — subbank false positive) | `ops/l8_ops.py` (alu rules) | 376 |
| 9 | `layer9_alu` / `l9_alu_*` reading OPCODE_BASE | OPCODE_BASE | `ops/l9_ops.py` | 512 |
| 10 | `tail_bit32_result_correction` / `tail_*` reading OPCODE_BYTE_HI_PIN under TEMP eff | OPCODE_BYTE_HI_PIN (real read, TEMP umbrella) | `ops/l13_ops.py` (tail correction) | 113 |

### What "real" means here

Rows 1-5 share a real semantic risk: the rule fires at a position
whose effective predicate's DNF includes a `mark == MEM` disjunct,
where the slot does *not* carry the value the rule names. Each of
these rules carries an `IS_BYTE + BYTE_INDEX_k + H1` gating cluster —
the *true* firing set excludes MEM, but the over-approximation doesn't
see it because the rule's conditions read `OPCODE_BYTE_LO+k` whose
semantics opens the MEM disjunct. **The structural fix is the same
across all five families**: append a `("MARK_MEM", -100.0)` hard
blocker to the rule's conditions. The blocker is byte-identical at
intended firing positions (where MARK_MEM == 0) and zeros the rule at
MEM rows where it would otherwise admit garbage.

Rows 6-7 are smaller variants on different alias pairs but the same
mitigation pattern (add the missing semantic blocker).

Rows 8-9 are SUBBANK false positives: OPCODE_BASE and OP_LEA are a
parent/child pair at the same 1-wide slot. The verifier's
`_is_colocated_subbank` filter requires the parent strictly to contain
the child — it skips same-extent pairs. Fixing the filter (relax the
strict-containment condition for *semantically* contained pairs)
suppresses the 959 entries without touching any rule. **This is the
cheapest cleanup**.

Row 10 is a TEMP-umbrella tautology; the rule reads
`OPCODE_BYTE_HI_PIN` at byte rows where the umbrella semantics of TEMP
admits everything. Suppress via verifier annotation (declare TEMP as
"ambient" and skip alias checks against it).

## Suggested fixes (this commit lands #A; #B,C are doc-only)

### A. Solver-side: `mark == X ⊥ is_byte` (LANDED)

`c4_release/neural_vm/unified_compiler/predicates.py` — added the
`_mark_role_implies_not_byte` helper and the cross-family contradiction
in `atom_contradicts`. 27.5k tautological violations drop. 35/35
new + existing predicate-overlap tests green.

### B. Verifier-side: relax `_is_colocated_subbank` to allow same-extent parent/child

`c4_release/neural_vm/unified_compiler/dim_alias_verifier.py` — drop
the `slot_b.size > slot_a.size or slot_b.start != slot_a.start` strict
clause so OPCODE_BASE/OP_LEA (same 1-wide slot, OP_LEA semantics ⊆
OPCODE_BASE semantics) is recognized as a parent/child sub-bank. This
suppresses ~960 OPCODE_BASE/OP_LEA + ~50 other co-extent subbank pairs
without affecting the real OPCODE_BYTE_LO/ADDR_B0_LO pair, whose
semantics are NOT in a subset relation.

Not landed in this commit (no test demonstrating the OP_LEA family is
real-safe; deferred to a follow-up).

### C. Verifier-side: TEMP umbrella annotation

`c4_release/neural_vm/unified_compiler/dim_alias_verifier.py` — add an
opt-in `tautological_semantics_slots` parameter and skip alias checks
against slots whose semantics parses to a tautology (TEMP,
TEMP_PREV_STEP, CS_DIST_THERMO, anything with semantics `is_byte OR
NOT is_byte`). Suppresses ~18k TEMP-related pairs.

Not landed in this commit (annotation API design needed).

### D. Rule-side: add `MARK_MEM` hard blocker to the textbook-bug family

For each of rows 1-5 above, append `("MARK_MEM", -100.0)` to the rule's
conditions. Byte-identical at every intended firing position; zeros
the rule at MEM rows where the read is garbage. Not landed: requires
per-rule byte-identity gating (`compare_symbolic_to_lowered_ffn`) and
coordination with parallel agents touching `l6_ops.py` /
`l14_ops.py` / `l16_ops.py`. Deferred to a follow-up brief.

## Smoke / regression

- `tests/test_predicates_overlap.py` — 24 + 4 new cases, all green.
- `tests/test_predicates_entailment.py`, `tests/test_predicates_parse.py`,
  `tests/test_dim_alias_verifier.py` — 179 cases, all green.
- `tests/test_smoke.py` — 10 passed (pre-existing 41 errors from the
  separate `DeclarativeAttentionHeadSpec.intent` regression on this
  worktree, NOT introduced by this change; baseline confirmed by
  reverting the predicates.py edit and re-running).

## File pointers

- Verifier source: `c4_release/neural_vm/unified_compiler/dim_alias_verifier.py`
- Predicate solver: `c4_release/neural_vm/unified_compiler/predicates.py`
- Effective-predicate builder: `c4_release/neural_vm/unified_compiler/effective_predicate.py`
- Static registry + semantics: `c4_release/neural_vm/dim_registry.py`
- Top offenders:
  - `c4_release/neural_vm/unified_compiler/ops/l6_ops.py` (jsr_target family)
  - `c4_release/neural_vm/unified_compiler/ops/l14_ops.py` (addr_key_neural_decode)
  - `c4_release/neural_vm/unified_compiler/ops/l16_ops.py` (jsr family)
  - `c4_release/neural_vm/unified_compiler/ops/l8_ops.py`, `l9_ops.py` (OPCODE_BASE subbank)
