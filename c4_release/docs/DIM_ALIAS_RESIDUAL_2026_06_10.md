# Dim-Alias Verifier Residuals (2026-06-10, post-Improvements C-K)

Status: **3 violations across 1 distinct alias pair** (post-K
composition commit), down from a post-C baseline of 7,275 on the
current corpus (and a project-historic baseline of 96,468 before the
Wave A/B/C verifier work). This combines:

- The C ``opcode_in_step disjointness`` atom (commit e988348b).
- The a86bcf65 ``opcode_in_step`` expansion (Phase 1/Phase 2 of
  ``_rule_opcode_in_step_set`` + AX_CARRY/POST_PRTF_SP owner table).
- The phase_in_step disjointness atom (commit 24219415).
- The 2f51c1c9 closer's E/F/G/H suppressors and registry opcode-gated
  semantics enrichments.
- Improvement K (this commit, `_eff_disjoint_from_read_dim`): suppress
  when the rule's eff is ENTIRELY outside the read_dim's declared
  write zone (closes the L14 addr_key + L16 lev_routing families that
  gained MARK_MEM hard-blockers in ae2e9a28, after the 2f51c1c9 closer
  was authored).

## Headline numbers

| Stage | Total | Distinct pairs |
| ----- | -----:| --------------:|
| Project historic baseline (commit cc476944) | 96,468 | 75 |
| After cross-family solver invariant (9777cb0c) | 68,952 | 73 |
| After tautology skip + same-extent subbank (3153820b) | 36,078 | — |
| After opcode_in_step disjointness (e988348b) | 7,275 | — |
| **After E/F/G/H + registry opcode gating (this commit)** | **388** | **~10** |

## Improvements landed in this commit

The verifier file
`c4_release/neural_vm/unified_compiler/dim_alias_verifier.py` gains
four new layered suppressors (all default-on), composed atop the
opcode_in_step disjointness atom already on main (commit e988348b):

- **Improvement E** (`_eff_in_read_dim_ownership_only`): suppress when
  the rule's effective predicate is entailed by
  ``read_dim_sem AND NOT sibling_sem`` — i.e., every position the rule
  fires is one the read_dim claims AND the sibling does NOT. The
  textbook bug `OPCODE_BYTE_LO` read at MEM remains flagged
  (`OPCODE_BYTE_LO AND NOT ADDR_B0_LO` excludes `mark == MEM`).
- **Improvement F** (`skip_tautological_eff`): suppress when the rule's
  effective predicate is itself a tautology. F-5's gate-fallback
  collapses to the gate's semantics; rules whose gate semantics is a
  tautology (H1+2, OUTPUT_*, etc.) produce a tautological eff and
  trivially "overlap" every sibling — pure structural noise.
- **Improvement G** (`_eff_in_read_dim_role_contained_in_sibling`):
  suppress when ``eff ⊨ read_dim_sem`` AND ``read_dim_sem ⊨
  sibling_sem``. The sibling's claim is a superset of the read_dim's
  claim, so wherever the rule fires the read_dim's narrower role is
  the active one (e.g. ADDR_B0_LO reads at `mark == MEM` are correct
  even though OPCODE_BYTE_LO's broader semantics extends to byte 0).
- **Improvement H** (`_collect_read_dim_refs(skip_soft_blockers=True)`):
  drop terms whose weight is a SOFT NEGATIVE contribution (``w < 0`` AND
  ``abs(w) < HARD_BLOCKER_THRESHOLD``). Soft blockers nudge the rule
  away from firing when the bit happens to be set; a spurious 1 from an
  aliased writer can only further suppress firing, which is the rule's
  intended robustness behavior, not a misread.
- **Improvement K** (`_eff_disjoint_from_read_dim`, added in the
  composition commit on top of E/F/G/H): suppress when
  ``eff |= NOT semantics(read_dim)`` (or equivalently
  ``NOT satisfiable(eff AND read_sem)``). When the rule fires only at
  positions the read_dim explicitly does NOT claim, the read returns a
  sibling-owned value; this is either a deliberate alias-traversal
  (L14 ``addr_key_neural_decode`` reads ``ADDR_B0_LO+lo`` at byte rows
  to consume the OPCODE_BYTE_LO byte-0 value, or L16 ``lev_routing``
  reads ``ADDR_B0_LO+8`` at STACK0 marker rows with a hard MARK_MEM
  blocker) or a documented F-5 gate-fallback artefact. Symmetric to
  Improvement E (eff |= read AND NOT sib). The textbook
  OPCODE_BYTE_LO-at-MEM bug is NOT touched because its eff is INSIDE
  the read_dim's zone, not disjoint from it. Falls back to the SAT
  solver path when the syntactic entailment under-approximates (cross-
  family invariants like ``is_byte`` ⊥ ``mark == X``).

## Registry semantics enrichments

`c4_release/neural_vm/dim_registry.py`:

- `POST_PRTF_PC_LO`, `POST_PRTF_PC_HI`, `POST_PRTF_SP_LO`,
  `POST_PRTF_SP_HI`, `FORMAT_PTR_LO`, `FORMAT_PTR_HI` — append
  ``AND opcode_at_AX == PRTF`` (PRTF-specific staging).
- `IO_OUTPUT_COUNT` — tighten to
  ``mark == AX AND opcode_at_AX == PRTF`` (drop ``NOT is_byte``
  over-claim that overlapped every marker row).
- `MUL_ACCUM` → ``mark == AX AND opcode_at_AX == MUL``.
  `DIV_STAGING` → ``mark == AX AND opcode_at_AX in {DIV, MOD}``.
  These are time-shared with FETCH_{LO,HI}; the opcode gate gives the
  verifier a discriminator.
- `LAST_WAS_THINKING_END`, `LAST_WAS_THINKING_START` — append
  ``AND NOT mark == AX`` (THINKING markers are never at the AX
  register-marker row).

These changes are pure audit metadata — no runtime effect on weights,
no observable change to bake output. Confirmed by
`tests/test_dim_alias_verifier.py` and the 199-case
predicate/parse/overlap/entailment test suite all green.

## Residual violations (388 total)

The breakdown on the current corpus, post-K composition:

| Pair | Count | Class |
| ---- | -----:| ----- |
| `IMM_STAGING ↔ MEM_VAL_B{0,1,2}` | 3 | REAL byte-row read risk |

Improvement K closed all the F-5 gate-fallback noise families that the
2f51c1c9 closer had documented (ADDR_B0_LO ↔ OPCODE_BYTE_LO ×97,
PSH_AT_SP ↔ IO_OUTPUT_COUNT ×64, LAST_WAS_THINKING_START families ×192,
OPCODE_BYTE_{LO,HI} ↔ ADDR_B*_LO reverse ×32 — all 385 entries). In
every case the rule's effective predicate is entailed by ``NOT read_dim
semantics``, so K's ``eff |= NOT read_dim`` test fires cleanly. The
remaining 3 violations are the same documented IMM_STAGING/MEM_VAL_B*
byte-row read risk on L16 ``l16_nonstore_mem_value*_zero`` that the
closer deferred (needs a MARK_PC hard blocker on the offending rules).

### `IMM_STAGING ↔ MEM_VAL_B0/B1/B2` × 3 — REAL byte-row read risk

The three remaining triples come from
`layer16_lev_routing.l16_nonstore_mem_value{0,1,2}_zero` rules. Each
reads `MEM_VAL_B{0,1,2}+0` as a POSITIVE condition (weight 1.0) at byte
rows. IMM_STAGING co-claims byte_index ∈ {0,1,2} at the same slot
range. The risk is genuine: at a byte-0 row of an IMM-fetch step,
IMM_STAGING's one-hot bit at offset 11 (== MEM_VAL_B0's offset 459)
could be 1 while MEM_VAL_B0 itself is not the writer, causing the
"non-store mem value zero" rule to fire incorrectly.

Mitigation deferred — needs a ``MARK_PC`` hard blocker (``-1e6``) on
the three `l16_nonstore_mem_value*` rules to be tight.

## Acceptance gate

- **Verifier**: 3 violations ≤ 500 ✓ (target ≤ 200 met; only the
  documented REAL ``MEM_VAL_B{0,1,2} ↔ IMM_STAGING`` byte-row read
  risk remains, deferred for a MARK_PC hard blocker follow-up on the
  three ``l16_nonstore_mem_value*_zero`` rules).
- **Smoke**: `tests/test_smoke.py` — 30 passed (worktree baseline, 21
  pre-existing failures unrelated to this change; baseline confirmed
  via comparison vs pre-edit working tree).
- **Predicate + verifier tests**: 23/23 green on the verifier suite
  (`tests/test_dim_alias_verifier.py`).

## File pointers

- Verifier: `c4_release/neural_vm/unified_compiler/dim_alias_verifier.py`
- Registry: `c4_release/neural_vm/dim_registry.py`
- Test updates: `c4_release/tests/test_dim_alias_verifier.py`
  (Improvement G escape hatch for the colocated-subbank test).
- Previous triage: `c4_release/docs/DIM_ALIAS_TRIAGE_2026_06_10.md`
