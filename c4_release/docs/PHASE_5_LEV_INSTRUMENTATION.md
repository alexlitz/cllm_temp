# Phase 5 JSR/ENT/LEV Instrumentation: Root-Cause Findings

**Date:** 2026-05-11
**Branch:** `d4-phase5-lev-instrument`
**Test traced:** `TestPureNeuralJSRLEVSimple::test_jsr_then_lev_simple`
**Program:** `[IMM 7, JSR 3, EXIT, ENT 0, LEV]` — expected AX=7

## TL;DR

The Phase 5 xfails are **downstream of a compile-time bake-collision bug** in
`unified_compiler/migrated_ops.py`. Tracing `test_jsr_then_lev_simple`
revealed that the L1 threshold heads at block 1 were never being baked —
their thresholds, V values, and W_o writes were being silently overwritten
by L0's threshold bake collision at block 0. Layer 0 ended up with both L0's
and L1's W_o writes targeting different output dims but driven by L0's
coarse thresholds (3.5/4.5/7.5/8.5/...) instead of L1's fine thresholds
(0.5/1.5/2.5/6.5).

Consequence chain:
1. `L1H0[PC]` fires at *all* distances ≤ 3.5 from a PC marker instead of just
   at the marker itself.
2. L3 PC-carry-forward attention head 0 uses `K = L1H1[PC] - L1H0[PC]` to
   attend to the previous step's PC byte 0. Because both `L1H1[PC]` and
   `L1H0[PC]` fire at the same range, `K = 0` at every byte position. The
   head degenerates to self-attention.
3. EMBED at step 1's PC marker is wrong (zeros instead of prev PC=10).
4. L5 attention head 5 queries `ADDR_KEY` with wrong PC, fetches garbage
   OPCODE_BYTE.
5. L5 FFN all-step decode at PC marker spuriously fires `OP_LEV=+5` because
   the unit threshold (`-2.5*S`) is below the value it gets from
   `MARK_PC=1 + OPCODE_BYTE_HI[0]=2.0 = 3.0 > 2.5`.
6. With `OP_LEV=+5` injected into the residual stream at PC marker, every
   downstream LEV-gated op fires on every step of every program. All
   JSR/ENT/LEV-specific weights light up wrong-step.

## The fix

`unified_compiler/migrated_ops.py`:
- `make_layer1_threshold_attn_op`: change from `kind="attn"` to
  `kind="block", layer_idx=1`, wrapping the bake_fn to receive `block`.
- `make_layer2_threshold_attn_op`: same change with `layer_idx=2`.

This pins the threshold-attention bakes to their intended blocks. With
`kind="attn"`, `LayerCompiler._assign_layers` was placing them at the
"earliest layer with an unclaimed (layer, kind) slot" — and because
`make_layer0_threshold_attn_op` uses `kind="block"`, it doesn't reserve the
`(0, "attn")` slot. L1's attn op then took slot `(0, "attn")` and L0's
block-bake (which runs *after* attn bakes in the same block) overwrote
W_q/W_k while leaving L1's W_o writes pointing at L1H0/L1H1/L1H2 — driven
by L0's wrong thresholds.

### Verification of the fix

Before fix — block 0 attn (`Layer 0` in dump):
```
H0: Q[CONST]=80.0 K[IS_MARK]=3.50 → [(60, 1.0), (116, 1.0)]   ← H0 AND L1H0 (wrong)
H1: Q[CONST]=80.0 K[IS_MARK]=4.50 → [(67, 1.0), (123, 1.0)]   ← H1 AND L1H1 (wrong)
H2: Q[CONST]=80.0 K[IS_MARK]=7.50 → [(130, 1.0)]              ← L1H2 only (wrong)
```

After fix:
```
Layer 0  H0: K=3.50 → [(60,1.0)]                              ← L0 H0 only
Layer 1  H0: K=0.50 → [(116,1.0)]                             ← L1H0 only, fine threshold
Layer 1  H1: K=1.50 → [(123,1.0)]                             ← L1H1 only, fine threshold
Layer 1  H2: K=2.50 → [(130,1.0)]                             ← L1H2 only, fine threshold
Layer 1  H4: K=6.50 (L1H4 writes correctly)
Layer 2  H0: K=5.50 → [(452,1.0)]                             ← L2H0 only
```

### Test impact

- `tests/test_pure_neural_pc.py::test_imm_then_exit`: before fix
  `actual=67372036 (=0x04040404)`; after fix `actual=0`. The model no longer
  emits random garbage but does not yet produce the right answer (5). The
  L1 threshold bug was one of multiple cascading bugs in pure_neural mode.
- `tests/test_pure_neural_jsr_ent_lev.py`: still 7 xfails. The deeper PC
  arithmetic / register carry-forward chain is still broken — the fix only
  closes the bake-time hole, leaving the run-time semantics to a separate
  fix. With `--runxfail`, `TestPureNeuralJSRLEVSimple::test_lev_returns_to_caller`
  now passes for the wrong reason (the test expects 0 and gets 0 because
  the program reaches EXIT before the model can clobber AX further). Not a
  meaningful Phase 5 win.
- `tests/test_weight_setter.py`: 14 pass / 8 fail before and after fix —
  no regression.

## Investigation methodology

Trace script: `scripts/debug/trace_jsr_lev_simple.py`.

Procedure: teacher-force the post-step-0 state at PC marker pos 79
(PC=0x0A, AX=7, SP=0x10000, BP=0, STACK0=0), append a single `REG_PC` token
to start step 1, then run a full forward pass and inspect the residual
stream at the 5 marker positions (PC, AX, SP, BP, STACK0) layer by layer
with sub-step granularity (`block.attn` output, then `block.ffn` output).

### Observation chain

#### Layer-by-layer at PC marker (before fix)

```
PC[L00] OPLEV=+0.00 OPCODE_LO[3]=+0.00 EMBED(LO[0]=+0.00,HI[0]=+0.00)
PC[L01] OPLEV=+0.00                       EMBED(LO[0]=+0.00,HI[0]=+0.00)
PC[L02] OPLEV=+0.00                       EMBED(LO[0]=+1.00,HI[0]=+1.00)   ← wrong PC!
PC[L03] OPLEV=+0.00                       EMBED(LO[0]=+1.00,HI[0]=+1.00)   ← L3 attn doesn't fix it
PC[L04] OPLEV=+0.00                       EMBED(LO[0]=+1.00,HI[0]=+1.00)
PC[L05] OPLEV=+5.00 OPCODE_LO[8]=+0.00 OPC_HI[0]=+2.00                    ← L5 FFN fires LEV decode
PC[L06] OPLEV=+5.00 ... (propagates from L6 OP_LEV relay)
...
PC[L16] OPLEV=+5.00 ... (corrupts L16 LEV routing for SP=BP+16, BP→PC restore)
```

#### L3 head 0 attention scores (at step 1 PC marker, pos 79)

```
DIAGNOSTIC: L3 attn head 0 (PC carry-fwd) at step 1 PC marker (pos 79, ALiBi slope=0.5):
  pos=79 tok=257 w=0.3935 score=+0.00 L1H1[PC]=1.00 L1H0[PC]=1.00 EMBED_LO[0]=1.00  ← self
  pos=78 tok=262 w=0.2387 score=-0.50 L1H1[PC]=0.00 L1H0[PC]=0.00 EMBED_LO[0]=0.00  ← STEP_END
  pos=77 tok=0   w=0.1447 score=-1.00 L1H1[PC]=0.00 L1H0[PC]=0.00 EMBED_LO[0]=1.00  ← padding
  ...
```

The head never attends to the previous step's PC byte 0 (pos 45 — should
have `L1H1[PC]=1, L1H0[PC]=0`). Both `L1H0[PC]` and `L1H1[PC]` are 1.00 at
*every* PC marker / byte 0 / byte 1 position checked, even at distance 1
where `L1H0`'s 0.5 threshold should clearly block it.

#### Direct weight inspection (root cause)

```python
attn1 = model.blocks[1].attn
W_k[0, IS_MARK]  # → 5.50  (NOT 0.5 as the L1 op declares)
W_o[L1H0+0, 1]   # → 0.00  (NOT 1.0 — entirely missing)
```

Walking each block:
```
Layer 0 (block[0].attn):
  H0: K[IS_MARK]=3.50 (L0's threshold) → W_o writes to dim 60 (L0 H0) AND dim 116 (L1H0)
  H1: K[IS_MARK]=4.50 → W_o writes to dim 67 (L0 H1) AND dim 123 (L1H1)
  H2: K[IS_MARK]=7.50 → W_o writes to dim 130 (L1H2 only)
  H3-H7: L0's coarse thresholds, no L1 W_o writes

Layer 1 (block[1].attn):  pre-fix
  H0: K[IS_MARK]=5.50 → W_o writes to dim 452 (L2H0) — this is L2's bake!
                       (L1 should have been here, but ended up overwritten in Layer 0)
```

The compiler's `_assign_layers` placed:
- L0 threshold (kind="block", layer_idx=0) → block 0
- L1 threshold (kind="attn") → earliest unclaimed (layer, "attn") slot. Slot
  `(0, "attn")` was unclaimed (L0 uses `kind="block"`, not `kind="attn"`),
  so L1 took slot `(0, "attn")`.
- L2 threshold (kind="attn") → slot `(0, "attn")` now has phase=1 ≠ L2's
  phase=2, so L2 advanced to `(1, "attn")`.

Bake order (`build_model_from_layout`):
1. For each layer, attn ops bake first, then ffn ops.
2. Block ops bake AFTER all attn/ffn at that layer.

So at block 0:
1. L1's attn bake: sets W_q[0..2, CONST]=80, W_k[0..2, IS_MARK]={0.5, 1.5,
   2.5}, W_o writes to L1H0/L1H1/L1H2.
2. L0's block bake (which itself calls `_set_threshold_attn` on `block.attn`):
   sets W_q[0..7, CONST]=80 (no-op, same), W_k[0..7, IS_MARK]={3.5, 4.5,
   7.5, 8.5, 9.5, 14.5, 19.5, 24.5} — *overwrites* L1's 0.5/1.5/2.5 at
   heads 0/1/2 — and adds W_o writes to H0..H7 (additive, since different
   target dims).

Net: L1's W_o writes stay, but driven by L0's wrong thresholds.

## Recommended follow-up

This fix removes one major bake-time bug. Pure-neural execution is still
broken — the underlying multi-step register carry-forward + PC arithmetic
needs more work. Specifically:

1. After the fix, the model no longer emits random garbage (0xAFAFAFAF /
   0x04040404). It emits 0 — i.e. it's now producing structured but
   incorrect outputs. The fix unblocked one layer of the cascade.
2. Phase 1 (single-instruction `IMM` then `EXIT`) still fails, returning 0
   instead of 5. The next root cause is somewhere else — likely in the L4
   PC increment, L5 fetch with corrected EMBED, or the output head.
3. Phase 5-specific work cannot resume until Phase 1 baseline is green in
   pure_neural mode.

## Files changed

- `neural_vm/unified_compiler/migrated_ops.py` — repinned L1 and L2
  threshold attention ops to `kind="block"` with explicit `layer_idx`.
- `scripts/debug/trace_jsr_lev_simple.py` — new diagnostic script that
  produces the layer-by-layer trace and the L3 attention-score inspection
  used in this investigation.
- `docs/PHASE_5_LEV_INSTRUMENTATION.md` — this report.
