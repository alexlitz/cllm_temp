# Close-out plan — remaining 6 smoke failures + 1096 corpus (2026-06-07)

## TL;DR

The 6 smoke failures split into 2 clusters (memory + LEA), each needing
a multi-rule structural fix. The 1096 corpus has 4 attributed clusters
(func_add, expr_add_mul, var, EQ Shape-B). Plan groups fixes into 5
waves ordered by risk × payoff.

## Smoke (45/51 → 51/51)

### Wave S1 — Memory cluster (5 tests) [3-4 sessions]

**Failing:** `test_si_li_roundtrip`, `test_sc_lc_roundtrip`,
`test_si_li_multiple_stores`, `test_si_li_overwrite`,
`test_si_li_16bit_value`.

**Root cause** (`docs/L8_SP_GATHER_STACK0_AUDIT_2026_06_07.md`):
no op broadcasts AX bytes 1/2/3 to `STACK0_BYTE1/2/3` rows during PSH.
PSH only writes byte 0. SI/LI then reads stale bytes 1-3.

**Why single-rule attempts fail:** the L10 PSH agent today extended head 3
and produced 39 failures (33-test regression). The 2 free L10 head slots
already collide with verifier magnitudes.

**Fix recipe** (audit-recommended):
1. **S1.1** Add dim family `STACK0_BYTE_VAL_h_LO/HI` (16 slots × 3 bytes = 48 dims).
2. **S1.2** Expand L10 head allocator budget +3 heads.
3. **S1.3** Add 3 new L10 heads that broadcast AX_BYTE1/2/3 → STACK0_BYTE_VAL slots during PSH.
4. **S1.4** Migrate L14 `mem_generation` reads to new dim family.
5. **S1.5** Re-run verifier magnitude collision audit.
6. **S1.6** Remove `_BINARY_POP_OPS` collapsed-step recovery in batched_pure_neural.

Acceptance per step: smoke ≥45/51 + memory test count monotone-non-decreasing.

### Wave S2 — `test_lea_basic` [2-3 sessions]

**Root cause** (`docs/TEST_LEA_BASIC_INVESTIGATION_2026_06_07.md`):
ENT-first-step STEP_END never emitted. Independent from memory cluster.

**Fix recipe:**
1. **S2.1** Probe at L1/L34 step-boundary detection during ENT step 0.
2. **S2.2** Identify the STEP_END writer that misses ENT step 0. Likely L34 `tail_step_end_emit` or L1 `step_boundary_detect`.
3. **S2.3** Add a model-level bake or L1 corrective: emit STEP_END at ENT step 0.
4. **S2.4** Verify `test_lea_basic` passes + no regression on `test_jsr_lev_roundtrip` (the bake-twin).

## 1096 corpus

Current attributable buckets:

| Bucket | Tests | Status |
|---|--:|---|
| Basic ALU (add/sub/cmp/shift) | ~150 | mostly passing |
| `func_add_*` (function call return) | 25 | **0/25** |
| `expr_add_mul_*` (multi-step expr) | 25 | ~5/25 |
| `var_*` (variable read/write) | 3 | **0/3** |
| Shape-B EQ_FALSE/NE_TRUE | ~20 | failing |
| `add_*`/`sub_*` random | ~150 | partial |
| Loops/recursion | varies | ~14% (Bucket J) |

### Wave C1 — `func_add_*` JSR/LEV return forwarding [3 sessions]

**Failure mode** (`docs/1096_ADD_HI_NIBBLE_PLUS_ONE_2026_06_07.md`):
`func_add(57,11) → 11`. Returns second argument instead of sum.

**Probable surface:** AX value at LEV step is overwritten by the
caller's second argument during the post-LEV PSH cascade. Likely
L13/L14 `mem_generation` reads stale `STACK0_BYTE_VAL_h` (shares root
with memory cluster).

**Plan:**
1. **C1.1** Run `func_add_0`. Capture AX at each layer of callee LEV step + caller post-LEV step.
2. **C1.2** Identify the layer where return value (sum) is overwritten by second-arg byte.
3. **C1.3** If it's the same `STACK0_BYTE_VAL_h` issue, wave S1 fixes it. Re-verify.
4. **C1.4** If independent, add JSR/LEV return-stage DSL.

### Wave C2 — `var_*` L3 SP_byte2 [2-3 sessions]

**Failure** (memory note `var_failure_mode_shifted`):
L3 block=3 gen=13 SP_byte2 OUTPUT_LO[1]-vs-[0] divergence.
The `_rewrite_initial_sp_marker_to_f8` op at L3 is dormant.

**Plan:**
1. **C2.1** Diff teacher-forced vs neural at L3 block 3 SP_byte2 OUTPUT_LO.
2. **C2.2** Find the op that wrote OUTPUT_LO[0] when [1] was expected.
3. **C2.3** DSL corrective at L3 OR L4 carry-forward chain.

### Wave C3 — Shape-B EQ_FALSE / NE_TRUE [2 sessions]

**Failure** (memory note `eq_byte1_l6_divergence`):
L6 byte-1 emit divergence on EQ(17, 17) and similar.

**Plan:**
1. **C3.1** Run `EQ_FALSE_0`. Diff L5/L6/L7 byte-1 OUTPUT.
2. **C3.2** Identify which L6 op (layer6_attn / layer6_routing_ffn / layer6_relay_heads) owns the byte-1 write.
3. **C3.3** DSL fix or attribution.

### Wave C4 — `expr_add_mul_*` multi-step expression [3 sessions]

**Failure mode:** heterogeneous off-by-{+1, +2, -}. Not a clean
single-byte error. Likely intermediate-result staging is overwritten
between sub-expressions.

**Plan:**
1. **C4.1** Pick simplest `expr_add_mul_0`. Decompose into instruction-by-instruction trace.
2. **C4.2** Identify the cross-instruction state corruption.
3. **C4.3** Probable shared root with collapsed-step (per
   `docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`).

## Cross-cluster dependencies

```
Wave S1 (memory cluster, new dim family)
  └─ unblocks Wave C1 (func_add return forwarding, same dim family)
      └─ unblocks ~partial Wave C4 (expr staging)

Wave S2 (LEA STEP_END) — independent

Wave C2 (var L3) — independent
Wave C3 (EQ L6 byte-1) — independent

Collapsed-step / L28+L34 fix (active agent af513417)
  └─ unblocks ~9 opcodes' override-removal path
      └─ unblocks Shape-A CMPs + remaining 1096 cluster bleed
```

## Effort summary

| Wave | Sessions | Target | Status |
|---|--:|---|---|
| S1 — Memory cluster | 3-4 | +5 smoke | not started |
| S2 — LEA STEP_END | 2-3 | +1 smoke | not started |
| C1 — func_add | 3 | +25 of 1096 | not started |
| C2 — var L3 | 2-3 | +3 of 1096 | not started |
| C3 — EQ L6 byte-1 | 2 | +~20 of 1096 | agent in flight |
| C4 — expr_add_mul | 3 | +~20 of 1096 | not started |
| Collapsed-step joint | 5-10 | unlocks override removal | agent in flight |

**Total to 51/51 smoke:** ~5-7 sessions across S1+S2.
**Total to substantial 1096 dent:** ~10-12 more sessions across C1-C4.
**Override-free smoke:** an additional 5-10 sessions on the L28+L34 cascade.

## Risks

- **S1.2 head allocator expansion** may cascade into dim_integrity
  regression. Mitigation: increment +1 head at a time, verify after each.
- **C1 may depend entirely on S1.** If func_add fails because of the
  same STACK0_BYTE_VAL issue, C1.4 is redundant — but we won't know
  until C1.1 probe.
- **C3 EQ fix may unblock the L34 cascade analysis** because L34
  `tail_bit32_result_correction` and L6 byte-1 may share an over-fire
  pattern. Worth running C3 in parallel with the L28+L34 cascade agent.

## Cherry-pick gates

Per recent session: every cherry-pick onto main must pass
`tests/test_smoke.py --tb=no -q` with ≥45/51 before the next commit
lands. The c6c89d07 incident (-15 smoke when stacked with override
changes) is the cautionary tale.
