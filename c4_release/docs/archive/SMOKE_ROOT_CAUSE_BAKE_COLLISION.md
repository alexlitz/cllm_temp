# Smoke root cause: block-op FFN bake collision at block 5

## Discovery

All 39 smoke failures + the L5-FFN-slot-collision symptom from
`SMOKE_COMPARISON_OP_DECODE_MISSING.md` trace to a single architectural
flaw: **multiple block-ops bake into the same block's FFN starting at
`start_unit=0`, overwriting each other's biases**.

## Mechanism

At block 5, FOUR block-ops bind to the same physical FFN via dep
anchors:

- `layer5_fetch` (kind="block", target=`_layer5_fetch_dep_anchor`)
- `opcode_decode_ffn` (kind="block", target=`_opcode_decode_ffn_dep_anchor`)
- `layer6_routing_ffn` (kind="block", target=`_layer6_ffn_dep_anchor`)
- `convo_io_opcode_decode` (kind="block")

The L6 `_layer6_ffn_dep_anchor` is pinned to block 5 via
`requires={"same_layer_as": "_opcode_decode_ffn_dep_anchor"}` at
`c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2743`.

Each block-op calls `CompilerIR.lower_ffn(block.ffn, ..., start_unit=0)`
at `c4_release/neural_vm/unified_compiler/ir.py:1408`. The lowering:

```python
ffn.b_up.data[unit] = -S * rule.threshold          # ASSIGNMENT
ffn.W_up.data[unit, ...] += weight                 # additive
ffn.W_down.data[..., unit] += weight               # additive
```

The `b_up` assignment stomps any prior op's bias at the same unit
index. The `W_up`/`W_down` additivity isn't enough to compensate when
the gating bias is wrong — the unit's silu(W_up + b_up) goes negative
and the unit silently produces zero.

## Trace evidence

- After `opcode_decode_ffn.bake`: `W_down[OP_IMM=188, :].abs().max() = 0.1` ✓
- After `layer6_routing_ffn.bake` (same block, same unit range): `W_down[OP_IMM=188, :].abs().max() = 0.0` ✗

OP_IMM, OP_AND, OP_OR, OP_XOR, OP_EQ, OP_ADD — all opcode flags in
dim range 187..217 — get cleared. Downstream consumers (L6+ routing,
L7 head 5 relays to TEMP+4/5/6, L9 ALU, L10 cmp combine, etc.) see
zero opcode flags and never fire.

This is the **same evidence** the
`SMOKE_COMPARISON_OP_DECODE_MISSING.md` agent reported as "89 declared
units missing from `block.5.ffn.W_down` rows 187..217". They were
overwritten by L6's bake from unit=0.

## Why the bitwise/Address/32-bit agent traces were misleading

They observed `OP_LEA = 0` at L14, `OP_AND = 0` at every block,
test_imm_exit returning 0 — and concluded the bug was at L7-L14
(some over-firing clear rule). But the rule was firing correctly;
the OPCODE_BYTE_LO decode in L5 was wiped at L6 routing's bake
BEFORE the model ran. Tracing the runtime residual missed it because
the weights were already broken at compile time.

## Fix candidates

### Fix A — drop the same_layer_as pin (smallest possible change)

`c4_release/neural_vm/unified_compiler/ops/l6_ops.py:2743`:

```diff
-    requires={"same_layer_as": "_opcode_decode_ffn_dep_anchor"},
+    # Use a separate FFN slot from L5's opcode_decode_ffn so the L6
+    # routing FFN doesn't stomp the opcode-decode hidden units.
+    requires={"after": "_opcode_decode_ffn_dep_anchor"},
```

Risk: the scheduler may still co-place if both ops have the same
`phase=` setting. May need an explicit `phase=` bump on
`_layer6_ffn_dep_anchor`. Validate with a probe of which block
`layer6_routing_ffn` lands in after the change.

### Fix B — thread an FFN unit allocator through block-op dispatch (architectural)

`c4_release/neural_vm/unified_compiler/layer_compiler.py:1782-1796`
currently calls `_dispatch_operation_ir` with no shared unit
allocator across the block-ops that share a target FFN. The right
fix is to walk all block-ops at the same target FIRST, build an
`FFNUnitAllocator` covering their declared `ffn_units_used` ranges,
then dispatch each with a non-overlapping `start_unit` from the
allocator.

This is the V2 architectural fix; it eliminates the entire class of
"multiple ops on one FFN" stomp bugs.

## Status

Diagnosis only; no fix applied. Step 1 of `IR_INCREMENTAL_IMPROVEMENTS.md`
will land Fix A (smallest change) once the active agent finishes its
work. Fix B is a follow-up for Step 5 (architectural produces/consumes
derivation).

## Cross-references

- `c4_release/docs/SMOKE_COMPARISON_OP_DECODE_MISSING.md` — same evidence
  from a different angle (rows 187..217 in W_down are empty).
- `c4_release/docs/EFFICIENT_MODE_FIX_GAP.md` — the related observation
  that fixes targeting declarative IR don't reach efficient mode.
- `c4_release/neural_vm/unified_compiler/ir.py:1408` — the `lower_ffn`
  with assignment-not-add on `b_up`.
- `c4_release/neural_vm/unified_compiler/layer_compiler.py:1782-1796` —
  the block-op dispatch loop with no inter-op unit allocator.
