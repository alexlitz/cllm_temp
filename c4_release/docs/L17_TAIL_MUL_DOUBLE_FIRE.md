# L17 tail MUL double-fire — root cause

The largest 1096 failure mass at HEAD: **58 of 384 failures uniformly at
`L17.post_ops[0] block=27 layer=27 width=1620, expected-token-never-wins`**.
Logits dominated by `OUT_HI[0]` (~+9.4e6); `OUT_LO[expected]=-0.00`;
margin uniformly exactly **-16.00**.

## Diagnosis

**Block 27 is `FlattenedALUMul`** — the L12 MUL ALU composite, expanded
by `_expand_wrapper_blocks` (Phase 10.B, `neural_vm/vm_step.py:2578`)
into a standalone block with a fresh `attn.layer_idx = 27`.

The MUL pipeline is `nn.Sequential([_BDToGEStage, schoolbook, 3 carry,
genprop, lookahead, finalcorrection, _MulCombineStage, _GEToBDStage])`
(`neural_vm/efficient_alu_neural.py:1038-1229`). `GEToBDConverter.forward`
adds `+= 2.0` into `BD.OUTPUT_HI:BD.OUTPUT_HI+16` and
`BD.OUTPUT_LO:BD.OUTPUT_LO+16`, gated by
`opcode_mask = (OP_MUL > 0.1 at GE) * (MARK_AX > 0.5)` at
`efficient_alu_neural.py:740-750`.

## Why blocks 25 and 27 are both `FlattenedALUMul`

`_expand_wrapper_blocks` splits both `make_l11_alu_postop_attach_op`
(`unified_compiler/ops/alu_ops.py:layer11_mul_partial`) and
`make_l12_alu_postop_attach_op` (`alu_ops.py:274`) into adjacent
standalone blocks (25 and 27). Block 26 between them is a passthrough
PureFFN that does NOT clear `OP_MUL` or `MARK_AX` from the residual.

So when block 27 runs its MUL pipeline, the same AX-marker row is still
gated on `OP_MUL`. The MUL composite re-fires, adding another
`OUTPUT_HI += 2.0` per nibble. Both writes accumulate.

Margin **-16.00** = the `sum_k k * one_hot[k]` channel
(NIBBLE coefficient k=15 × 1.0) in `BDToGEConverter.NIB_A`
(`efficient_alu_neural.py:129-139`) projected through the head's
unembed. When `ALU_LO/HI` arrive with no active channel (`-0.00`) at a
non-AX/non-MUL row, `_clean_onehot` returns all-zero, `RESULT=0`,
and the `+= 2.0` from `_GEToBDStage` leaks into `OUTPUT_LO[0]`
through the wrapper-block layer-norm onto token 0xF0.

## Why this clobbers OUT_LO[expected]

After block 25 runs (L11 partial MUL), `OUTPUT_HI[expected_nibble]`
carries the correct partial product. Block 27 re-running the same
composite re-emits `OUTPUT_HI += 2.0` at the same nibble positions,
doubling the signal. But `OUT_LO[expected]` is also clobbered because
the second run's `_BDToGEStage` reads stale `ALU_HI` from block 25's
output, producing a different RESULT byte; the GEToBDStage's write
collides with the LO band.

## Proposed fixes (root vs. patch)

### Root-cause fix (preferred)

Make `make_l12_alu_postop_attach_op` (`alu_ops.py:274`) and
`make_l11_alu_postop_attach_op` share **one** baked
`FlattenedALUMul` instance. Both ops attach to the same wrapper block
post-expansion, with the L12 variant declared `requires={"after":
"layer11_mul_partial"}` so the scheduler doesn't duplicate.

Alternatively: `_expand_wrapper_blocks` should detect adjacent
identical composite types and collapse them into one block.

### Patch (more localized but band-aid-shaped)

In `_MulCombineStage.forward` (`neural_vm/efficient_alu_neural.py:730`),
add a single-fire guard: only fire when the residual indicates this
is the *first* MUL composite to write at this AX-marker row this
step. Track via a fresh dim bit `MUL_ALREADY_FIRED` cleared at step
boundary and set by `_MulCombineStage` after firing.

## Why this matters

This is the single largest 1096 failure mass — fixing it should recover
all 58 currently-observed failures plus likely the 49 var_update_*
/ var_three_* / var_mul_* rows that exhausted the trace cap (per
`diag-1096-scan-next/.agent-logs/1096_scan_next_20260602.md`).

## Cross-references

- `vm_step.py:2578-2700` — `_expand_wrapper_blocks`
- `efficient_alu_neural.py:730-778` — `_MulCombineStage`, `GEToBDConverter`
- `efficient_alu_neural.py:1038-1229` — `FlattenedALUMul` pipeline
- `unified_compiler/ops/alu_ops.py:263-293` — L11/L12 MUL postop attach
- `unified_compiler/ops/shared.py:166-262` — `_make_alu_postop_attach_op`
- `unified_compiler/ops/l10_ops.py:4132-4229` — `tail_ax_add_byte1_hi_zero`
  (writes OUTPUT_HI strength +50_000; ruled out as in-block-27 writer
  because lives at block 30, but is the upstream residual feeder)

## Status

Diagnosis only; no fix applied. Validation skipped (`decl_verifier.
verify_claims_static` requires a second compile that the 1-compile-budget
declined). Next agent should: instrument residual at block 25/26/27
boundaries to confirm the double-fire hypothesis numerically.
