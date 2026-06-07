# Memory cluster fix plan - 2026-06-05

After 4 zero-sum attempts (V1-V4 documented in
`SMOKE_MEMORY_FIX_ATTEMPT_V{1,2,3,4}_20260605.md`), the memory cluster
is a **multi-session architectural refactor**, not a single-commit fix.
This plan breaks the work into 5 phases, each independently shippable.
The dependency chain is strict: Phase 1 unblocks Phase 2, etc.

## Status update - 2026-06-06

This document records the architectural plan that came out of the V1-V4
attempts. Some prerequisites have since moved forward: the slot registry
exists and targeted slot-registry tests pass, and current placement probes
resolve the main memory ops to the intended dynamic layers
(`layer13_mem_addr_gather -> L13`, `layer14_mem_generation -> L17`,
`layer15_memory_lookup -> L18`, with the shift composite at runtime B25).

The memory requirement is still open. Current dense smoke remains
`45 passed, 6 failed, 1 deselected`: `TestSmokeAddress::test_lea_basic`
plus five `TestSmokeMemory` cases. The immediate batched symptom is that
after the final address `IMM`, execution skips the following `LI`/`LC` and
returns AX (`0x200`/`0x300`) at `EXIT`. Store generation is also malformed
in focused probes (`SI` emitted `addr=0x110000`, `value=0x0200002a`
instead of `addr=0x200`, `value=0x2a`). Treat Phase 4 and the L10/L14
store/lookup path as the next investigation target, not as completed work.

## The bug, restated

5 smoke tests (`TestSmokeMemory::test_si_li_*`, `test_sc_lc_roundtrip`)
fail because `layer13_mem_addr_gather` resolves to L16 in baseline
instead of L13, severing the mem-addr -> mem-value chain. The
straightforward fix - pin it to L13 via `_layer13_attn_dep_anchor`
with `layer_idx=13` - fails because of a cascade of architectural
couplings:

1. **Slot conflict (V2)**: `l13_alu_shift_install` (sets `block.ffn =
   ALUShiftComposite`) and `efficient_l10_andorxor_wrap` (sets
   `block.ffn = PureFFN(bitwise_rules)`) both want `block[13].ffn`.
2. **Placement (V3)**: with A2 (`post_ops.append`) fixing the slot
   conflict, the composite lands at B15 instead of needed B25.
3. **Head_0 contest (V4)**: pulling `layer10_carry_relay_bake` and
   `layer13_mem_addr_gather` to the same `block[13].attn` makes them
   overwrite each other's head 0 weights silently.

The four blockers have to be solved in order. Each blocker has a
clean fix; the problem was that each attempt only solved one and
exposed the next.

## Phase 1 - Slot-conflict registry (1-2 sessions)

**Goal**: detect at compile time when two ops want the same slot
(`block.ffn`, `block.attn.head_N`, `block.post_ops[X]`).

**Why first**: prevents silent overwrites across the rest of the work.
Without this, every architectural change risks introducing a new
silent contest.

**Implementation**:

1. Add a `SlotRegistry` to `LayerCompiler` that records every bake's
   `(layer_idx, slot_kind, slot_id)` claim. Slot ids:
   - `("ffn",)` - the whole `block.ffn` module
   - `("attn", "head", k)` - one attention head k
   - `("attn", "Wq")` / `("attn", "Wk")` / `("attn", "Wv")` /
     `("attn", "Wo")` - per-matrix claim for non-head-aligned
     attention bakes
   - `("post_ops", i)` - explicit post_op slot (today's
     `post_ops.append` is implicitly idx=current_len)
2. Compile dispatcher records the claim before calling the bake_fn.
   When two ops claim the same slot at the same layer, compile fails
   with a clear error pointing at both ops.
3. Existing ops that legitimately co-bake a slot (e.g., L10's multiple
   tail-correction families sharing block.ffn) need a `slot_share=`
   flag on the Operation to whitelist them. Probably ~10 sites.

**Verification**: compile passes (no false positives), and intentionally
re-running V4's "memory-fix-v4 attempt" without the contest mitigations
produces a clear error instead of a silent regression.

**Files**: `layer_compiler.py` (the dispatcher), a few alu_ops sites,
plus tests in `tests/test_slot_registry.py`.

## Phase 2 - Move `layer10_carry_relay` off the L13 attn head 0 collision

**Goal**: free `block[13].attn.head_0` so `layer13_mem_addr_gather`
can own it cleanly.

**Why second**: this is the immediate decoupling that V4 found
necessary. Phase 1's registry will surface the collision; this phase
resolves it.

**Two options** (pick one based on disruption profile):

### Option A - Move carry_relay to a non-L13 layer

`layer10_carry_relay`'s name is misleading - it's named L10 but
currently lands at L13 due to the dep-graph anchor. The original
intent (per its name) is L10. Pin it back to L10.

This requires:
1. `layer_idx=10` on the op (instead of `target_op_name="layer10_alu"`)
2. Update everything that reads `target_op_name="layer10_carry_relay"`
   to use the new physical layer
3. Verify the rest of the L10 alu pipeline doesn't break - carry_relay
   was probably moved to L13 historically for some reason; revisit.

### Option B - Use a different head index for carry_relay

If carry_relay needs to stay at L13 for dep-graph reasons, give it a
different head index. The `attention_head_allocator` already supports
auto-fit; tell it carry_relay claims head 1 (or 4, whichever is free).

Then `layer13_mem_addr_gather` takes head 0 as it always wanted.

**Verification**: with Phase 1's registry, this compile cleanly with
both ops at L13 attn but different heads.

**Files**: 1 op factory (carry_relay) + maybe the attn_head_allocator.

## Phase 3 - Split the L13 anchor

**Goal**: `_layer13_attn_dep_anchor` currently couples to BOTH
`layer10_carry_relay` (via target_op_name) AND `layer13_mem_addr_gather`
(same). Splitting the anchor into per-purpose anchors lets each op
resolve independently.

**Implementation**:

1. Rename `_layer13_attn_dep_anchor` to `_layer13_mem_addr_anchor`.
   This anchor is the one mem_addr_gather targets.
2. Create a new `_layer10_carry_relay_anchor` (or similar) that
   carry_relay targets, with `layer_idx=10` (per Phase 2 Option A) or
   `layer_idx=13` with `slot=("attn", "head", 1)` (Phase 2 Option B).
3. Update every consumer of the old anchor name. Probably ~5-10 sites
   in alu_ops, l13_ops, possibly l10_ops.
4. Ensure the dep-graph still resolves correctly - no new cycles.

**Verification**: dep-graph builds cleanly; both `layer10_carry_relay`
and `layer13_mem_addr_gather` land where intended; no slot conflicts.

**Files**: l13_ops.py, l10_ops.py, alu_ops.py.

## Phase 4 - Re-apply the shift install + placement

**Goal**: now that Phases 1-3 have decoupled the L10/L13 ops, finally
apply the V4 fix (composite at B26 via post_ops.append + after=late_op).

**Implementation** (this is the V4 work, but now safe):

1. `make_alu_shift_composite_ops`:
   - `ffn_units_used=<N>` on each of 4 shift stages
   - `requires={"after": "layer16_lev_routing"}` on each stage
   - `target_op_name="layer16_lev_routing"` on the install op
   - Install via `block.post_ops.append(builder.composite)` (A2)
2. `ALU_LO` -> `ALU_LO.*.-1` SSA renames on stages
3. Add 4 `CROSS_STEP_DOCUMENTED_SAFE` entries

This is mechanical now. Phase 1 will catch any new slot conflicts
introduced; Phases 2-3 have decoupled the head_0 contest.

**Verification**: full smoke >= 46/52 baseline + 5 memory tests pass.

**Files**: alu_ops.py, full_vm_compiler_dynamic.py.

## Phase 5 - Composite idempotence audit

**Goal**: V3 originally suggested B3 ("op-gate composite forward for
idempotence"). After Phase 4 lands, audit whether the composite is
truly idempotent under various op_total / layer placement scenarios.
This protects against future regressions.

**Implementation**:

1. Add a per-composite forward-pass test that runs the composite at
   different physical block indices and asserts byte-identity to the
   expected reference.
2. If any composite fails this test, add a guard to its `forward`
   (e.g., skip if a sentinel residual dim is not set, indicating
   "this composite is being run at the wrong block").

**Verification**: composite-idempotence tests pass; smoke unchanged.

## Effort summary

| Phase | Sessions | Smoke gain | Notes |
|---|---|---|---|
| 1 | 1-2 | 0 (infra) | Catches the silent contest class of bug |
| 2 | 1 | 0 | Decoupling only; needed for 4 |
| 3 | 1 | 0 | Decoupling only; needed for 4 |
| 4 | 1 | **+5** | Memory cluster recovers |
| 5 | 1 | 0 (hardening) | Prevents future regressions |
| **Total** | **5-6 sessions** | **+5 smoke** | Reaches 51/52 (combined with lea_basic fix -> 52/52) |

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Phase 1's registry might trip on legitimately-shared slots in the existing codebase | Add `slot_share=True` flag; the audit should surface ~10 sites; expected ~1 hour to whitelist. |
| Phase 2 Option A might find carry_relay genuinely needs L13 for a non-obvious reason | If so, fall back to Option B (different head index). |
| Phase 3's anchor split might break dep-graph cycle detection | Add a test that compiles efficient mode + verifies the dep graph has no cycles before phase 3 lands. |
| Phase 4 might find a new blocker (V5) | If yes, the V1-V4 pattern continues; each new blocker is bounded and documented. |
| Memory cluster could be a symptom of something deeper | The 4 attempts all found different blockers - each a real architectural issue. After Phase 4, if smoke still doesn't recover, the cluster might need a 6th attempt with different scope. |

## Why this works when V1-V4 didn't

Each prior attempt tried to land everything in one commit. The
zero-sum pattern is because each commit had to solve every blocker
simultaneously, and any unsolved blocker silently regressed smoke. By
splitting into 5 phases:

- Phase 1 makes silent regressions impossible (registry errors loudly).
- Phases 2-3 decouple the ops first, BEFORE attempting the placement
  change.
- Phase 4 is the actual fix, but now safe because it can only succeed
  or hit a clear registry error.
- Phase 5 prevents future regressions.

The total work is similar to what V1-V4 attempted, but split so that
intermediate states are valid (smoke unchanged) rather than broken
(smoke regressed).

## Open questions for the next-session implementer

1. Is there a documented reason `layer10_carry_relay` lands at L13 in
   baseline? Search git log for the commit that introduced it.
2. Does the L10/L13 head_0 contest exist in lookup mode too, or only
   efficient mode? V4 ran efficient; lookup may behave differently.
3. Are there other slot conflicts hiding behind the silently-overwriting
   bakes that V1-V4 found? Phase 1 will surface them; need to know
   how many to budget for.
