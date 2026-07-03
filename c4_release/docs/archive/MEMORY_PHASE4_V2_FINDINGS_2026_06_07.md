# Memory Phase 4 v2 — paradigm shift finding (2026-06-07)

## TL;DR

**The Phase 1-4 architectural plan was chasing the wrong target.** Even with all topology fixed (`mem_addr_gather` at L13, `mem_generation` at L14, `memory_lookup` at L15), the 5 memory smoke tests STILL fail with identical wrong values. The bug is in the **L13→L14 residual handoff**, not in block placement.

## What was tried

Phase 4 v2 retry (agent `acdee94d`, 2026-06-07):
1. Pinned `layer16_lev_routing` to `layer_idx=16`
2. Pinned `layer14_mem_generation` to `layer_idx=14`
3. Pinned `layer15_memory_lookup` to `layer_idx=15`
4. Applied V4 composite changes (4 shift stages + install via `post_ops.append`)
5. Resolved cascading slot conflicts via additional pins (L11/L12/L13 anchors)

## Result

| Test | Baseline | After Phase 4 v2 |
|---|---|---|
| test_si_li_roundtrip (expects 42) | 512 | **512** |
| test_sc_lc_roundtrip | fail | **fail** |
| test_si_li_multiple_stores (expects 99) | 768 | **768** |
| test_si_li_overwrite (expects 55) | 512 | **512** |
| test_si_li_16bit_value (expects 4660) | 512 | **512** |
| test_shl_8bit | PASS | **FAIL (regression)** |

Memory tests: **unchanged values**. Topology is now correct end-to-end but the tests still produce the address (0x200=512) instead of the value at that address.

## The real bug

The "512" pattern means the model returns the address itself, not `mem[addr]`. This is a **residual handoff break** between:
- L13 `mem_addr_gather` (writes ADDR_B*_LO/HI correctly per the topology)
- L14 `mem_generation` (reads from those dims to drive memory_lookup)

Possible mechanisms:
1. L14 mem_generation reads `ADDR_B*.-1` (cross-step alias) instead of same-step
2. L14 reads `ADDR_KEY` (different dim) which is stale
3. L14 head 1's Q-K matches wrong MEM marker position
4. L15 memory_lookup's V-relay is broken, returning the address tensor itself

## Why this matters

The 4 prior memory cluster attempts (V1-V4) all assumed the bug was topological. Phase 1 (slot registry), Phase 2 (decouple carry_relay), Phase 3 (split L10 anchor), Phase 3b (split L13 anchor), Phase 3c (move L10 attn family) all landed cleanly and are correct, but **none of them fix the memory cluster** because the failure mode is at a different level.

## Recommendation

**Don't pursue Phase 4 further.** Investigate the actual L13→L14 residual handoff (agent `abc3adc0` is on this).

The Phase 1-3c work is still valuable as structural prep — slot conflict registry catches future bugs, anchor splits improve scalability — but they're not on the critical path for the memory cluster fix.

## test_shl_8bit regression diagnosis

Phase 4 v2 broke `test_shl_8bit` by moving the composite from B25 (`_layer13_attn_dep_anchor` host) to B26 (`layer16_lev_routing` host). The intervening B25 ops (now-late mem_addr/mem_gen) mutate AX_CARRY between the chain's producer (~B22) and `ALUShiftComposite.forward` reader. AX_CARRY chain is broken.

Per the `MEMORY_PHASE4_BLOCKER_2026_06_05.md` doc's Option B: a dedicated `_l13_alu_shift_anchor` with `requires={"after": "_layer12_ffn_dep_anchor", "before": "_layer14_attn_dep_anchor"}` would keep the composite at L13-L16 range where AX_CARRY is live. This is orthogonal to the memory cluster fix.
