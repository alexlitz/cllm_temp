# Deterministically safe KV eviction

## Goal

Evict KV cache entries only when they are **guaranteed dead** — i.e. the program (per the declarative IR) cannot read them on any later step. Byte-identity preserved on output logits vs no-eviction baseline.

## Liveness rules (what counts as dead)

An entry at `(layer, position, head, dim)` is **dead at step S** if every condition below holds:

1. **No future step reads it.** No `AttentionHeadIR` at any future step has Q-side conditions that would attend to this `(position, dim)`.
2. **The semantic value it represents was overwritten.** If the position wrote to a register (PC/AX/SP/BP/STACK0) or a memory cell, and a later position has overwritten that register or memory cell, the original KV entry's contribution is dead.
3. **The position didn't persist state.** Positions whose only writes are intra-step intermediates (TEMP, AX_FULL, CMP) with no cross-step reader.

Equivalent restatement: the entry is dead if every Q/K/V/O dependency edge into it from a future step is provably unreachable per the IR's static analysis.

## Why this is safe

The declarative IR (FFNRule + AttentionHeadIR) makes all reads/writes explicit. After Phase 6 lands, every weight write is declared. That means a static liveness analyzer can:

- Build a use-def graph per `(step, position, dim)`
- For each step S, mark entries with no path to a future Q-side read as dead
- Evict only dead entries

If any read-edge to an evicted entry materializes at runtime, the analysis was wrong — never a correctness issue, just an analyzer bug. The byte-identity gate catches it.

## Phasing (depends on Phase 6 declarative work)

### Phase 6.5 — Liveness analyzer (depends: Phase 6 Wave 5 validation green)

**6.5.1**: Build a static use-def graph over the declarative IR. For each FFNRule.constant_write / gated_write and AttentionHeadIR.Q/K/V/O, emit `(step, position, dim, role)` tuples.

**6.5.2**: Compute "live-out at step S" set: positions whose KV entries are read by any step > S.

**6.5.3**: Compute "overwritten by step S": registers/memory cells whose value at any prior position is dominated by a later position's write.

**6.5.4**: Combine into "evictable at step S" = entries written before step S whose `(position, dim)` is neither live-out nor not-yet-overwritten.

**Acceptance**: For the 1096 corpus, output a per-step evictable set. Spot-check 10 cases by hand against the IR.

### Phase 6.6 — Runtime eviction pass (depends: 6.5)

**6.6.1**: Add `--kv-eviction-policy=static_liveness` flag to `compile_full_vm` and the runtime. Default: off.

**6.6.2**: At each step boundary, the runtime consults the precomputed evictable set and zeroes (or marks-unused) the cache rows.

**6.6.3**: Determinism gate: same eviction decisions in spec-decode and main-decode; same decisions across runs given same input.

**Acceptance**:
- Spec-decode and main-decode produce identical eviction trajectories per step
- Byte-identical logits vs no-eviction baseline on the full 1096 corpus
- KV peak memory drop quantified (target: ≥30%)

### Phase 6.7 — CI gate (depends: 6.6)

**6.7.1**: Add `pytest c4_release/tests/test_kv_eviction_byte_identity.py` — runs 100 sampled corpus inputs in both modes, asserts byte-identical logits.

**6.7.2**: Add `analyze_kv_liveness.py` tool — reports per-op "produces / consumed by" KV entries to help auditing future ops.

## Acceptance criteria

1. Liveness analyzer derives evictable set from declarative IR alone (no runtime state)
2. Spec-decode and main-decode agree on eviction decisions at every step boundary
3. Byte-identical logits vs no-eviction baseline on full 1096 corpus
4. ≥30% peak KV memory reduction on representative workloads
5. Determinism: identical decisions across runs on same input
6. CI gate prevents regressions

## Risks

| Risk | Mitigation |
|---|---|
| Analyzer misses a read-edge (false-dead) | Byte-identity gate fails immediately; turn off eviction, fix analyzer |
| Imperative bakes still in tree → unknown reads | Phase 6 must be ≥95% complete first (no imperative attention IR holes) |
| Spec-decode/main-decode divergence on eviction | Decisions derived from IR position scope, not runtime; gate at step boundary |
| Cross-step dependencies (B9 PREV_STEP) confuse analyzer | Use the B9 PREV_STEP refs as explicit liveness extensions |

## Connection to today's work

Safe KV eviction is **the dividend** of the declarative-IR migration. Once every read/write is declared, we get a free liveness analyzer. Without Phase 6, conservative eviction (window-based) is the only option and degrades correctness; with Phase 6, semantic-aware eviction preserves byte-identity.

Sequence the work as: **Phase 6 complete → Phase 6.5 analyzer → Phase 6.6 runtime pass → Phase 6.7 CI gate**.
