# var_three / var_update / var_mul cluster: L15 nibble_copy 0x0a leak

## Symptom

49 untraced rows in diag-1096 (var_update 25 + var_three 18 + var_mul 6).
All `decl_steps=17`, `suite_decl=match` (IR correct). Neural produces
"tiny integer ≈ one operand".

Examples:
- `var_mul_0` (23 * 47 = 1081): step 15 STACK0_byte0 expected 0x2f (47=b),
  neural emits **0x17 (23=a — operand a)**.
- `var_three_0` (29+6+20 = 55): step 22 STACK0_byte0 expected 0x23 (35),
  neural emits **0x06 (=b=6)**.
- `var_update_1` (50+28 = 78): step 14 STACK0_byte0 expected 0x4e (=78),
  neural argmax 0x101 (overflow >255, margin -3.6e8).

## Failure chain

Layer-by-layer trace of STACK0_byte0:

1. **Block 27 / layer 18 = `layer15_nibble_copy` (L15, width 42)** —
   first divergence point. Argmax flips to 0x0a (=10) with margin
   -0.87 → -199. This is the **bootstrap-JSR return-address marker**
   (PC=10 returns from main per C4 convention).
2. **Block 28 / layer 19 = `layer16_lev_routing` (L16, width 792)** —
   amplifies the 0x0a write 7-9 orders of magnitude.
3. **Block 34 / layer 34 = `tail_bit32_result_correction`
   (L17.post_ops[0], width 2059)** — final stage explodes another
   ~17 orders. Margin reaches ~-1.5e26.

(Block numbering is POST-L11-attach-removal. Earlier `L17_TAIL_MUL_DOUBLE_FIRE.md`
diagnosis was at the older 36-block layout — different block 27.)

## Root cause

`layer15_nibble_copy` rule in `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1576`
writes `OUTPUT_LO+10` (the 0x0a token) without a sufficient
`OP_LEV`-gated guard or `MARK_STACK0`-blocker, so the LEV PC-return
constant leaks into STACK0 emit rows.

`l16_lev_pc_top_return_0a`-family rules in
`c4_release/neural_vm/unified_compiler/ops/l16_ops.py:332-348` then
amplify because they ALSO write 0x0a unconditionally on rows the
LEV-detection isn't blocking properly.

The L17 tail (`tail_bit32_result_correction`) is just the final
amplifier — same end-stage explosion shape as in
`L17_TAIL_MUL_DOUBLE_FIRE.md`, but the upstream contaminator is
nibble_copy here, not the MUL composite.

## Proposed fixes

### Fix A: gate l16_lev_pc_top_return_0a on MARK_STACK0 absence

```diff
 # l16_ops.py l16_lev_pc_top_return_0a-family rules at line 332-348
 conditions=(
     ...
+    ("MARK_STACK0", -100.0),  # don't write 0x0a at STACK0 emit positions
 ),
```

### Fix B: gate l15_nibble_copy on OP-discrimination

The current `writes={OUTPUT_LO, OUTPUT_HI_THIS_STEP}` lacks an explicit
OP_LEV-only constraint. Add a positive `OP_LEV` predicate (or negative
ENT/IMM/ADD/SUB/MUL/etc.) so the rule only fires when an actual LEV is
being processed.

### Validation

After either fix, trace block 27 STACK0_byte0 at step 1 for var_mul_0.
Argmax should NOT be 0x0a. Final-step STACK0_byte0 should match
expected (0x2f = 47 = operand b).

## Cross-references

- `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1576` —
  `layer15_nibble_copy`
- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:332-348` —
  `l16_lev_pc_top_return_0a` family
- `c4_release/docs/L17_TAIL_MUL_DOUBLE_FIRE.md` — sibling L17 amplifier
  failure (different upstream contaminator)

## Status

Diagnosis only. 49 1096 tests blocked on this. Fix A is the smaller
first attempt.
