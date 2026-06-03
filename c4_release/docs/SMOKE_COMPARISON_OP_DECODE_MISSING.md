# Smoke comparison failures — root cause is upstream of ComparisonCombine

## Reframe

`docs/EFFICIENT_MODE_FIX_GAP.md` and `docs/IF_EQ_CMP_DEFAULT_LEAK.md` framed
the 6 failing `TestSmokeComparison::test_*_true` tests as a
ComparisonCombine bug (default-leak / MARK_PC blocker). That reframe was
also wrong. The MARK_PC blocker at `vm_step.ComparisonCombine` (and the
parallel declarative `_l10_comparison_combine_rules`, lines 510/516
`MARK_PC_BLOCK=-50`) is already in place and would gate properly *if*
`OP_EQ` were ever asserted at the AX-marker position.

## Trace

Pure-neural runner, single-step (`AutoregressiveVMRunner(pure_neural=True,
trust_neural_alu=True, spec_k=0)`), bytecode
`IMM 42 ; PSH ; IMM 42 ; EQ ; EXIT`.

Captured residual stream at every transformer block over all forward
calls. Result:

* `OPCODE_BYTE_LO` / `OPCODE_BYTE_HI` ARE correctly produced at every PC
  marker (`computed opcode=0x11=17` at PC pos 149 — the EQ step).
* **`OP_EQ` (compact dim 201) is ZERO at every position in every block
  for the entire run.** Same for `OP_NE..OP_GE` and even `OP_IMM`,
  `OP_PSH`, `OP_EXIT`.

The `BatchedPureNeuralRunner` smoke wrapper shares this same compiled
model so the smoke and the AutoregressiveVMRunner traces match.

## What this means

The L5 opcode-decode FFN is not producing any `OP_<NAME>` flag. Sweep of
every `nn.Linear` / FFN `W_down` in the model finds **no** writer to any
`OP_*` dim in the family range `OP_LEA..OP_GETCHAR` (dims 187..217).
Only the embedding and unembedding heads touch that row range — neither
fires during a forward pass at the AX-marker.

`block.5.ffn.W_up` has 129 units that read `OPCODE_BYTE_LO+1` and
`OPCODE_BYTE_HI+1` with magnitude `1e3`, but they are *suppressors*
(weights `-1000`, threshold `-450`, writing to `OUTPUT_LO` only). The
expected per-opcode `+10/S` write to `OP_<NAME>` from
`_opcode_decode_main_rules` / `_opcode_decode_first_step_rules` is
absent.

## Why ComparisonCombine looks innocent

In the cmp combine FFN (`vm_step.py:649` and the parallel
declarative `_l10_comparison_combine_rules` lowered by
`make_l10_post_ops_combined`), both default and override units gate on
`OP_<NAME> > 0` (via either an explicit `W_gate[OP_<NAME>] = 1.0` or a
threshold that requires `MARK_AX + OP_<NAME> = 2S` to overcome
`b_up = -1.5S`). With `OP_<NAME>` stuck at zero, the gate never opens
and the FFN contribution at the AX marker is identically zero — exactly
what the captured per-unit math shows
(`up = -22.0`, `silu = -6e-9`, `act ≈ 0`).

The MARK_PC blocker at `-50S` only matters when the unit would
otherwise fire; here the unit is already silent for an unrelated
upstream reason.

## Repro

```python
runner = AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
# Hook each block.forward, capture residual; scan compact dim 201 (OP_EQ).
# All entries are < 1e-6 at every position across all forward calls.
```

## What this changes

* The single-line `_cmp_default` MARK_PC patch (lookup-mode) does not
  unblock smoke comparisons. It was already in place and is correct,
  but it sits downstream of an OP_EQ=0 stream.
* The next investigation belongs at L5: why does
  `_bake_opcode_decode_ffn` / `_opcode_decode_main_rules` /
  `_opcode_decode_first_step_rules` not produce W_down writers to the
  `OP_<NAME>` family in this build? Possible causes worth checking
  before guessing:
  1. The L5 FFN slot in the dependency-assigned model takes ~1514
     hidden units; only ~89 are claimed by the opcode decode allocator.
     The remaining ~1425 are owned by some later tenant whose bake may
     be zero-ing or relocating the opcode-decode rows.
  2. `Primitives.lower_ffn_rules` may be dropping rules whose
     `dim_ref("opcode_flag", NAME)` resolves to a slot the L5 unit
     allocator does not own.
  3. The Phase 8 `OPCODE_BYTE_LO.*.-1` SSA rename (l5_ops.py:527) is
     prev-step semantics; the decoder may be evaluating on step-1 zeros
     for the whole run rather than only on step 1.

## Status

Diagnosis only. The previously-shipped declarative `_cmp_default`
MARK_PC fix (in `_l10_comparison_combine_rules`) is preserved — it is
defensively correct, it just isn't the load-bearing fix.
