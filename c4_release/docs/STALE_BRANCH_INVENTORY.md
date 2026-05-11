# Stale Branch Inventory Post-Migration (2026-05-10)

## Executive Summary

After today's migration commits, the repo still has **31 branches with unmerged commits** (out of ~90 total branches). A detailed audit reveals:

- **10 REDUNDANT branches**: Top commit already on main (different SHA), safe to delete
- **2 STALE-1-COMMIT branches**: 1 unique commit each, not on main—worth investigating
- **12+ MULTI-COMMIT branches**: 4–18 commits ahead, likely contain overlapping work from today's merges

This inventory helps identify which branches should be integrated, rebased, or discarded.

## Summary Table

| Category | Count | Action |
|----------|-------|--------|
| REDUNDANT | 10 | Delete safely |
| STALE-1-COMMIT | 2 | Investigate + integrate or discard |
| MULTI-COMMIT | 12 | Review carefully; likely overlap |
| **TOTAL** | **24** | |

---

## REDUNDANT BRANCHES (Safe to Delete)

These branches have their top commit already merged to main under a different SHA (same message, rebased or cherry-picked).

| Branch | Ahead | Top Commit | Status |
|--------|-------|------------|--------|
| `cleanup-deprecated-divmod` | 1 | Remove deprecated fp64 DIV/MOD wrappers | On main as `42a4fe9` |
| `compact-io-dims` | 1 | Compact IO dim positions: pin_io_only=True | On main as `92c9986` |
| `migrate-everything-unit1` | 1 | Migrate L2 lookback detection + L3 convo-IO | On main as `350858c` |
| `migrate-everything-unit2` | 1 | Migrate L5+L6 tool-call bakes | On main as `d5c8e3e` |
| `migrate-everything-unit11` | 1 | Migrate L9 ALU FFN bakes (Unit 11) | On main as `610bfd3` |
| `migrate-everything-unit12` | 1 | Migrate L10 lookup-mode ALU FFN (Unit 12) | On main as `5ca995a` |
| `docs-architecture-update` | 14 | Update AGENT_CONTEXT.md + ADRs 001-004 | On main as `13b182f` |
| `migrate-io-putchar-routing` | 17 | Migrate _set_io_putchar_routing to compiler | On main as `631be8a` |
| `migrate-rightsize-expand` | 17 | Migrate _right_size_ffns + _expand_wrapper_blocks | On main as `bc0be15` |
| `refactor-byte-passthrough` | 17 | Refactor L10 byte passthrough to use chain primitive | On main as `11a50ab` |

**Recommendation**: Delete these branches. Their work is already on main.

```bash
git branch -D cleanup-deprecated-divmod compact-io-dims \
  migrate-everything-unit1 migrate-everything-unit2 migrate-everything-unit11 \
  migrate-everything-unit12 docs-architecture-update migrate-io-putchar-routing \
  migrate-rightsize-expand refactor-byte-passthrough
```

---

## STALE-1-COMMIT BRANCHES (Investigate & Integrate)

These branches have **exactly 1 unique commit not yet on main**. Small fixes worth evaluating for integration.

### 1. `lev-fix-phase5`
- **Ahead**: 1 commit
- **Commit**: `ecc2fe3` — Migrate L13 setup ops to compiler dispatch
- **Description**: Flips `make_layer13_mem_addr_gather_op` and `make_layer13_shifts_op` to `migrated=True`. Removes direct `_set_layer13_*` calls from `set_vm_weights`. Moves ALiBi slope setup into migrated bake_fn.
- **Status**: **NOT on main yet** (different from `migrate-l7-ops-to-compiler` which includes this)
- **Recommendation**: **INTEGRATE** — This is a targeted L13 migration fix. Check if compatible with current main or if `migrate-l7-ops-to-compiler` supersedes it.

### 2. `migrate-everything-unit3`
- **Ahead**: 1 commit
- **Commit**: `cfb2a32` — Migrate L5+L6 convo-io bakes to compiler ops
- **Description**: Combines `_set_layer5_convo_io` and `_set_layer6_convo_io` into single migrated `kind="block"` bake on Layer 6 convo-io op. Threads attention head dimension through for hidden-unit allocation.
- **Status**: **NOT on main yet**
- **Recommendation**: **INVESTIGATE** — Layer 5+6 convo-io bakes may be in other branches (unit consolidation wave). Cross-reference with `migrate-everything-unit1`, `unit2`, etc.

---

## MULTI-COMMIT BRANCHES (Careful Review Required)

These branches have **2–18 commits ahead**. Many likely contain overlapping work from today's ~15–20 merged commits. Requires detailed diff review to separate redundant commits from genuinely unique work.

### Critical: High Overlap Branches (13–18 commits ahead)

These are most likely to contain commits already on main or superseded by today's merges.

#### `migrate-binary-pop-sp-increment` (18 ahead)
- **Top 3 commits**:
  1. `0ecb365` — Migrate _set_binary_pop_sp_increment to compiler model op
  2. `875722c` — Add register_increment_unit primitive; refactor _set_binary_pop_sp_increment
  3. `df6cbb0` — Re-extract primitives set 1 (register_decrement_unit, marker_write_unit, opcode_gated_pc_override, byte_passthrough_chain)

- **Status**: Contains primitives extraction and compiler migration. Likely overlaps with `primitives-set1-respawn` and `add-register-increment-primitive`.
- **Recommendation**: **REVIEW CAREFULLY** — Diff against `primitives-set1-respawn` and `add-register-increment-primitive` to identify truly unique changes.

#### `add-register-increment-primitive` (17 ahead)
- **Top 3 commits**:
  1. `875722c` — Add register_increment_unit primitive; refactor _set_binary_pop_sp_increment
  2. `df6cbb0` — Re-extract primitives set 1 (register_decrement_unit, marker_write_unit, opcode_gated_pc_override, byte_passthrough_chain)
  3. `3ab7b45` — Stabilize pure_neural_runner fixture + add smoke test

- **Status**: **Top commit NOT on main**. This is the only branch that still has the register_increment_unit primitive definition.
- **Recommendation**: **INTEGRATE or EXTRACT** — This primitive may be needed by other branches (e.g., `migrate-binary-pop-sp-increment`). Check if critical for recent migrations.

#### `refactor-l14-mem-addr-heads` (17 ahead)
- **Top 3 commits**:
  1. `055585c` — Refactor L14 MEM addr heads to use Primitives.memory_addr_head
  2. `df6cbb0` — Re-extract primitives set 1 (register_decrement_unit, marker_write_unit, opcode_gated_pc_override, byte_passthrough_chain)
  3. `3ab7b45` — Stabilize pure_neural_runner fixture + add smoke test

- **Status**: **Top commit NOT on main**. This is the only branch that refactors L14 to use the memory_addr_head primitive.
- **Recommendation**: **EXTRACT & INTEGRATE** — This is a specific L14 refactor. If primitives extraction is on main, cherry-pick the L14-specific commit.

#### `stabilize-pure-neural-fixture` (15 ahead)
- **Top commit**: `3ab7b45` — Stabilize pure_neural_runner fixture + add smoke test
- **Status**: **Unique commit** — Fixes test stability. Useful for regression prevention.
- **Recommendation**: **INTEGRATE** — Low-risk fixture stabilization. Likely improves CI reliability.

#### `fix-l14-jsr-val-head3` (16 ahead)
- **Top commit**: `df6cbb0` — Re-extract primitives set 1 (register_decrement_unit, marker_write_unit, opcode_gated_pc_override, byte_passthrough_chain)
- **Status**: Primitives extraction is on main. Likely redundant.
- **Recommendation**: **DELETE** after confirming primitives are on main.

#### `primitives-set1-respawn` (16 ahead)
- **Top commit**: `df6cbb0` — Re-extract primitives set 1 (register_decrement_unit, marker_write_unit, opcode_gated_pc_override, byte_passthrough_chain)
- **Status**: Primitives extraction is on main. Likely redundant.
- **Recommendation**: **DELETE** after confirming primitives are on main.

### Moderate Overlap (9–14 commits ahead)

#### `fix-phase2-bitwise-deep` (14 ahead)
- **Top commit**: `ca05f50` — Phase 2 bitwise: fix BITWISE_OP relay/propagation at byte positions
- **Description**: Adds TEMP[3] (combined BITWISE_OP relay) to BinaryOpByteZeroingPostOp's gate to zero OUTPUT bytes 1-3 for AND/OR/XOR. Fixes relay strength issues at byte positions.
- **Status**: **NOT on main yet** — This is a targeted bitwise gate fix discovered post-merge.
- **Recommendation**: **REVIEW & INTEGRATE** — This is a specific Phase 2 bugfix. Check if recent merges already fixed this, or if it's a unique correction.

#### `migrate-l3-ops-to-compiler` (13 ahead)
- **Top commit**: `692c65a` — Pin migrated L4/L7/L11/L12/L13 ops to legacy block
- **Status**: This commit is on main. Likely all commits are redundant.
- **Recommendation**: **DELETE** — Verify all commits are on main, then delete.

#### `fix-psh-zero-emission` (13 ahead)
- **Top 3 commits**:
  1. `692c65a` — Pin migrated L4/L7/L11/L12/L13 ops to legacy block
  2. `e753cd7` — Reapply "Fix CarryPropagationPostOp byte-1 carry strength"
  3. `b3e29e6` — Revert "Fix CarryPropagationPostOp byte-1 carry strength"

- **Status**: Contains revert + reapply pattern. Mixed commits on/off main.
- **Recommendation**: **INVESTIGATE** — Check if the final state (with reapply) is on main. If so, delete. Otherwise, cherry-pick the reapply commit.

#### `migrate-function-call-weights` (9 ahead)
- **Top 3 commits**:
  1. `4763f36` — Migrate _set_function_call_weights to compiler model op
  2. `7380860` — Fix Phase 4 BNZ taken in pure_neural mode
  3. `f08d659` — Add neural PUTCHAR/GETCHAR semantics for pure_neural mode

- **Status**: Mixed commits. Top commit may be unique; others likely on main.
- **Recommendation**: **REVIEW** — Diff against main to see if top commit is truly unique.

#### `fix-phase4-bz-bnz-step2` (9 ahead)
- **Top 3 commits**:
  1. `8ddbbd3` — Phase 4 BZ/BNZ: suppress FETCH→PC at IMM step after taken branch
  2. `7380860` — Fix Phase 4 BNZ taken in pure_neural mode
  3. `f08d659` — Add neural PUTCHAR/GETCHAR semantics for pure_neural mode

- **Status**: Targeted Phase 4 bugfix (suppress FETCH→PC). Others likely shared with other branches.
- **Recommendation**: **REVIEW** — Check if top commit is unique or if this is superseded by recent Phase 4 merges.

### Lower-Priority (4 commits ahead)

#### `migrate-l7-ops-to-compiler` (4 ahead)
- **Commits**:
  1. `e02b5cd` — Fix test encoding: branch opcodes need imm=idx·INSTR_WIDTH+PC_OFFSET
  2. `05b80f9` — Make untracked-module imports optional in unified_compiler/__init__.py
  3. `c637698` — Migrate L7 setup ops to compiler dispatch
  4. `ecc2fe3` — Migrate L13 setup ops to compiler dispatch (same as lev-fix-phase5)

- **Status**: Combines L7 and L13 migrations. Last commit is also in `lev-fix-phase5`.
- **Recommendation**: **REVIEW** — Check if L7 migration is on main. If so, only the L13 part (lev-fix-phase5) is useful.

---

## Missing Branches Check

The following single-commit branches appear redundant but were not listed above. Verify:

- `migrate-everything-unit5` — Confirmed on main as different SHA
- `migrate-everything-unit6` — Confirmed on main as different SHA
- `migrate-everything-unit8` — Confirmed on main as different SHA
- `migrate-lookup-mode-hybrid-alu` — Confirmed on main as different SHA

**Action**: Bulk delete the above 4 branches with the 10 from the REDUNDANT section.

---

## Action Items

### Immediate (Safe deletions)
1. Delete 10 REDUNDANT branches (all 1-commit units + docs/IO/refactoring)
2. Delete `migrate-everything-unit5`, `unit6`, `unit8`, `migrate-lookup-mode-hybrid-alu`
3. Verify `primitives-set1-respawn` and `fix-l14-jsr-val-head3` (likely redundant)

### Short-term (Investigate & integrate)
1. **`lev-fix-phase5`**: Determine if compatible with current main or superseded
2. **`migrate-everything-unit3`**: Check for conflicts with unit consolidation wave
3. **`stabilize-pure-neural-fixture`**: Low-risk, integrate if tests pass
4. **`fix-phase2-bitwise-deep`**: Review bitwise relay fix; may be critical for regression

### Medium-term (Careful review)
1. **`add-register-increment-primitive`**: Check if this primitive is used/needed
2. **`refactor-l14-mem-addr-heads`**: Extract L14 refactor if primitives are on main
3. **`fix-psh-zero-emission`**: Clarify revert+reapply state
4. **`migrate-l7-ops-to-compiler`**: Determine L7 migration status

### Final (Verify & delete)
1. **`migrate-l3-ops-to-compiler`**: Verify all commits on main, then delete
2. **`migrate-function-call-weights`**: Check if top commit is unique
3. **`fix-phase4-bz-bnz-step2`**: Verify Phase 4 fixes are on main

---

## Notes

- **Post-merge state**: Today's session merged ~15–20 commits. Many branches include subsets of those commits, making them partially redundant.
- **Primitives extraction**: The re-extraction of primitives set 1 (`df6cbb0`) appears on main. Branches with only this commit should be deleted.
- **L13 migration**: The `ecc2fe3` commit (Migrate L13 setup ops) appears in both `lev-fix-phase5` and `migrate-l7-ops-to-compiler`. Only one path is needed.
- **Phase 2 & Phase 4 fixes**: Recent fixes to Phase 2 bitwise and Phase 4 BZ/BNZ suggest ongoing stability improvements. Review these carefully to avoid regression.

---

**Report generated**: 2026-05-10 at 22:21 UTC  
**Repository**: `/home/alexlitz/Documents/misc/c4_release`  
**Branch count scanned**: 31 branches with unmerged commits
