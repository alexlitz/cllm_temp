# DimContract audit failures — attribution

**Date**: 2026-06-09  
**Audit**: `python -m c4_release.tools.dim_contracts_audit`  
**Registry**: `c4_release/neural_vm/unified_compiler/dim_contracts.py`  
**Result**: 9/15 contract FAIL, 6/15 PASS.

This document attributes each FAIL into one of two structural buckets so
follow-up work can land per-bucket fixes instead of chasing nine
independent signals.

## Bucket A: declared-reads gap on `layer14_mem_generation`

Six contracts FAIL with the same signature:

```
[FAIL] stack0_byte_val_{1,2,3}_{lo,hi}_pshk2mem
    ERROR: consumer 'layer14_mem_generation' does not declare
           'STACK0_BYTE_VAL_{h}_{LO,HI}' in reads
    NOTE: consumer 'layer14_mem_generation' gate dims not in reads:
          ['OP_LC', 'OP_LI']
```

**Producer**: `layer10_psh_ax_broadcast` (l10_ops.py:2447-2463).  
Writes `STACK0_BYTE_VAL_{1,2,3}_{LO,HI}` (six dims) during PSH.

**Consumer**: `layer14_mem_generation` (l14_ops.py:789-837).  
The head spec at `_layer14_mem_generation_head_specs` heads 1/2/3 reads
all six dims at V slots 32..47 (LO) and 48..63 (HI), with matching O
writes to OUTPUT_LO/OUTPUT_HI nibbles (l14_ops.py:549-573, added by
commit `6c5be295` "Wave 1 A3.2"). The bake reads the dims, but the
declarative `reads={...}` set on the `Operation` does NOT list them.

**Mechanism**: the per-op decl_verifier passes because the declared
writes match the lowered writes; nothing inside the per-op verifier
correlates head-spec V-side AP entries against the op's `reads` set.
The cross-op `DimContract` verifier closes that gap.

**Fix sketch**: add the six STACK0_BYTE_VAL_h_LO/HI dims plus the
OP_LI / OP_LC gate dims to the `reads={}` set in
`make_layer14_mem_generation_op` (l14_ops.py:808-813). This is the
same edit history as commit `dfe729f9` ("A3.8") on master, which has
not yet landed on this branch. The reads are *not* runtime-load-bearing
(the weights already see them); the edit is structural-correctness only.

**Expected post-fix**: 6 FAIL -> 0 FAIL (Bucket A closes); plus the
NOTE about OP_LC/OP_LI clears.

## Bucket B: verifier needs SSA `.*.-1` alias awareness

Three contracts FAIL with the same root cause:

```
[FAIL] addr_b0_lo_prev_step_l15_to_l14   (dim='ADDR_B0_LO')
[FAIL] addr_b0_hi_prev_step_l15_to_l14   (dim='ADDR_B0_HI')
[FAIL] output_lo_post_ent_l16_to_l3_prev_step  (dim='OUTPUT_LO')

    ERROR: consumer does not declare '<DIM>' in reads
    ERROR: producer layer N is AFTER consumer layer M
           -- the consumer reads a stale value
```

**Mechanism**: Phase 8.A introduced SSA-style cross-step aliases:
when an op declares `"ADDR_B0_LO.*.-1"` in its `reads`, the static
scheduler treats it as "the previous step's ADDR_B0_LO residual",
sharing the same numeric residual slot but breaking the same-step
back-edge from L15 -> L14. The aliasing is honoured by the dim_flow
analyser and the dep-graph builder, but not by the
`DimContract.verify_dim_contract` body (line 308 in dim_contracts.py):
the check is `if contract.dim not in cons_op.reads`. A `.*.-1` alias
does not satisfy that predicate.

The producer-AFTER-consumer ordering error is the same artifact: the
contract names the base dim's producer (L15 for ADDR_B0_LO writes;
L16 for OUTPUT_LO on OP_ENT), but the consumer fires earlier in the
SAME step because the data flow is cross-step (the SSA alias).

**Fix sketch**: extend `verify_dim_contract` to treat
`<dim>.*.-1` in the consumer's `reads` as satisfying the
`contract.dim in reads` check, and to flip the ordering invariant
(producer-after-consumer is OK if the alias is cross-step). A
companion `Optional[bool]` field `cross_step: bool` on `DimContract`
would let the contract author opt into the relaxed check explicitly.

**Expected post-fix**: 3 FAIL -> 0 FAIL (Bucket B closes).

## Verifier-side TODO

Both buckets close with two small landings:

1. **Bucket A**: declarative edit on `layer14_mem_generation.reads`
   (~6 lines, byte-identical, no functional change).
2. **Bucket B**: alias-aware `DimContract` extension (~20 lines in
   `dim_contracts.py`; one new `cross_step` field, two predicate
   tweaks).

After both: the audit should report `15/15 PASS`.

## Cross-reference

- `docs/A3_5_L14_CONSUMER_DIAGNOSTIC_2026_06_09.md` — earlier write-up
  of the same Bucket A failure surfaced by the starter contracts.
- `docs/B9_OUTPUT_HI_SPLIT_SPEC.md` §2.1 — Phase 8.A G7 PREV_STEP
  alias design.
- `c4_release/neural_vm/unified_compiler/dim_flow.py` — alias-aware
  residual-slot analyser; the prior art for Bucket B's verifier
  extension.
