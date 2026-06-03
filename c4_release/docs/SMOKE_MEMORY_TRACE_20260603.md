# Smoke `TestSmokeMemory` trace — 2026-06-03 worktree session

Brief: investigate 5/6 `TestSmokeMemory` failures in efficient mode
(`test_si_li_roundtrip`, `test_sc_lc_roundtrip`, `test_si_li_multiple_stores`,
`test_si_li_overwrite`, `test_si_li_16bit_value`). The one passing test is
`test_si_li_zero` which is the documented false-positive cluster
(expected==0 coincides with `exit_code=0` failure mode).

## Verdict: not a memory-specific bug

The 5 memory failures are **the same failure mode as the other 34 smoke
failures** characterized in `.agent-logs/smoke_drift_20260602.md`:

  - All 39 failing smoke tests return `exit_code = 0` regardless of program
  - Token stream is structurally correct (REG_PC/REG_AX/STEP_END grammar intact)
  - AX value-byte slots emit zero — the AX-write residual chain is broken
  - The 12 passing tests are the false-positive cluster (expected==0)

Per `smoke_drift_20260602.md`, prior bisect blamed commit `ae64239b`
(L13 `target_op_name="l13_alu_shift_getobd"`). The fix commit
`ff454052` retargeted to `_layer13_attn_dep_anchor`, but the doc
`HANDOFF_2026_06_02.md` notes smoke remained 12/39. This trace confirms
the fix did not propagate.

## Topology observation (single compile this session, alu_mode='efficient')

`compile_full_vm_dynamic(strict=False, alu_mode='efficient')` resolves
`n_layers=25`, `model.blocks=35`. Pre-expansion layer placement (from
`layout.ops_at(layer_idx)`):

```
L13 (real): layer10_carry_relay            (was intended for L13 mem-addr)
L14:        _layer11_ffn_dep_anchor         (was intended for L14)
L15:        _layer12_ffn_dep_anchor         (was intended for L15 memory_lookup)
L16:        _layer13_attn_dep_anchor + layer13_mem_addr_gather + layer13_shifts + l13_alu_shift_install
L17:        _layer14_attn_dep_anchor + layer14_mem_generation + L14 cleanups
L18:        layer15_memory_lookup + L15 ops
L19:        layer16_lev_routing
L20-23:     l13_alu_shift_{bdtoge,precompute,select,getobd}  (FFN composite stages stranded past LEV)
L24:        l10_post_ops_combined
```

The L13 dep anchor's contract is "Pin strictly after the L12 anchor so
the earliest landable layer is 13". In practice it lands at L16 because
upstream layers absorbed extra IR-anchor slots. The "L13" / "L14" /
"L15" / "L16" names are now nominal — the physical layer for each
anchor has drifted +3.

The drift is internally consistent: shift FFN stages bake into a
shared `_ALUShiftCompositeBuilder.composite` not into the per-layer
FFN, so the install op at L16 is the only place that actually touches
`model.blocks[16].ffn`. The fresh PureFFNs at L20-L23 take zero writes
and act as identity.

## Where AX writes go in efficient mode (audit)

For `test_si_li_roundtrip` (`IMM 0x200 PSH IMM 42 SI IMM 0x200 LI EXIT`),
the AX-write path on the `LI` step:

1. **L7 head 5**: `OP_LI` flag at AX marker → `OP_LI_RELAY` at AX value
   byte positions (`vm_step.py:5439, 7298`).
2. **L8 mem_to_alu** (`l8_ops.py:2114`): `enable=False` by default, so
   the per-step ADDR_KEY→mem[SP] lookup at AX never runs in efficient
   mode. NOT the source of the failure.
3. **L15 memory_lookup heads 0-3** (`vm_step.py:7193-7376`): heads 0-3
   gated on `OP_LI_RELAY` (`Q[base, BD.OP_LI_RELAY] = 2000.0`) read the
   MEM_STORE byte values matching ADDR. V/O copies
   `CLEAN_EMBED_LO/HI` → `OUTPUT_LO/HI` at AX byte positions.
4. **L16 lev_routing** (block 25-area): final OUTPUT routing.

The actual physical L15 (where layer15_memory_lookup lives) is layout
L18 = `model.blocks[18]`. Block 18 has `PureFFN W_up.shape=[512, 800]`.
The L15 attn weights are baked correctly by `_set_layer15_memory_lookup`
into `attn` at the resolved block — the bug is upstream.

## Suspect: the L13-named ops at L16 fire ADDR-write rules too late

`layer13_mem_addr_gather` writes `ADDR_B0_LO`, `ADDR_B0_HI`, etc., the
3-byte address used by L15 memory_lookup's address-matching. Since
mem_addr_gather now bakes into `model.blocks[16].attn` (post-expansion:
later — block expansion shifts again), the ADDR_B* dimensions for SI/SC/LI/LC
get written ONE LAYER AFTER layer15_memory_lookup at L18. But L18 > L16,
so the order is fine.

What's NOT fine: any consumer that historically reads ADDR_B0_LO at L14
or earlier now reads a stale value (the L13-anchor wrote it at L16,
after the L14 read). The `CrossStepReadWarning`s emitted during compile
confirm this: `layer14_addr_key_neural_decode` reads `ADDR_B0_HI.*.-1`
with same-step writers `_layer13_attn_dep_anchor, layer13_mem_addr_gather,
layer15_store_stack0_sp_byte0_addr, ...` — but those writers run AT L16-L18,
while the reader at `layer14_addr_key_neural_decode` runs AT L17. Cross-step
fallback returns 0 on step 1.

## Next step (not applied this session)

The proper fix is at the **scheduler / dep-graph** layer, not in any
single rule. Options ranked by risk:

1. **Increase the `phase=` on `_layer13_attn_dep_anchor`** from 12.5 →
   13.0 and verify it slots between L12-dep-anchor and L14-dep-anchor.
   The current 12.5 was chosen to be "between 12 and 13" but the
   landed L12 dep anchor is at L15 (phase 12), so the L13 anchor at
   phase 12.5 slots at L16 (next available after L15).
2. **Add explicit `layer_idx=13` back to the L13 anchor**, accepting
   the V4 ("no layer_idx") regression in exchange for runtime
   correctness. Per memory `feedback_single_rule_fixes_are_zero_sum.md`,
   this is the "revert and confirm" path that has actually netted
   positive in past sessions.
3. **Audit `_emit_cross_step_safety_warnings` output** (compile emits
   16+ warnings about same-step writers across L8/L10/L13/L14/L15)
   to pin which specific dim's same-step-writer mismatch breaks the
   AX-write chain end-to-end.

## Compile budget consumed: 1 / 1

Single `compile_full_vm_dynamic(strict=False, alu_mode='efficient')`
call inspected for layout. No code changes applied this session.

## Files referenced

- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:393-432` — `_layer13_attn_dep_anchor`
- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:435-518` — `layer13_mem_addr_gather`
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-179`   — shift composite ops + install
- `c4_release/neural_vm/vm_step.py:7193-7376`                      — L15 memory_lookup heads 0-3 bake
- `c4_release/docs/EFFICIENT_MODE_FIX_GAP.md`                       — alu_mode='efficient' inertness
- `.agent-logs/smoke_drift_20260602.md`                              — 39-failure triage
- `.agent-logs/smoke_bisect_20260602_1339.md`                        — original L13 bisect
- `c4_release/docs/HANDOFF_2026_06_02.md`                            — session handoff w/ verify steps
