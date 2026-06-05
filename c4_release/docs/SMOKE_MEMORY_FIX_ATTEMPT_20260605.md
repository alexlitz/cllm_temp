# Smoke memory cluster fix attempt — 2026-06-05

Attempted Option A from `L13_ANCHOR_DOWNSTREAM_CHAIN.md` to recover the 5
`TestSmokeMemory` failures (`test_si_li_*` + `test_sc_lc_roundtrip`)
identified by `SMOKE_FAILURES_2026_06_05.md` as cluster #1 (P0,
"single L13/L14 dep-anchor retarget").

**Result: not viable as a 1-op edit. Reverted.** Documenting the gap.

## What was tried

Edits (worktree `/tmp/c4-memory-fix`, branch `memory-fix`):

1. `l13_ops.py` `_layer13_attn_dep_anchor`: added `layer_idx=13` next
   to the existing `phase=13.0`.
2. `alu_ops.py` `make_alu_shift_composite_ops`: added `layer_idx=13`
   to each of the 4 ffn stages (`l13_alu_shift_bdtoge`,
   `l13_alu_shift_precompute`, `l13_alu_shift_select`,
   `l13_alu_shift_getobd`).

Step (2) implements the "pin all 4 composite stages to L13" piece of
Option A in `L13_ANCHOR_DOWNSTREAM_CHAIN.md` (so `l13_alu_shift_install`
at L13 sees `builder.composite` fully populated).

## Why it fails (two distinct blockers, only the first one is documented in the audit)

### Blocker 1 (anticipated): dep validator rejects ALU_LO reads at L13

`l13_alu_shift_bdtoge` reads `ALU_LO`. `layer16_lev_routing` writes
`ALU_LO` at L16 (next-step PC staging — semantically a cross-step write
to the same residual slot). The validator at
`layer_compiler.py:1937` sees `writes_layer["ALU_LO"] = 16 > 13` and
raises:

```
ValueError: Op 'l13_alu_shift_bdtoge' pinned to layer 13 reads dim 'ALU_LO'
which is produced at layer 16
(must be < 13, or == with lower phase, or attn-then-ffn at same phase)
```

This is the same false back-edge that `_layer11_ffn_dep_anchor` solved
via the Phase 9.B SSA cross-step rename (see `l11_ops.py:240-246`):
`ALU_LO` → `ALU_LO.*.-1` (same numeric slot, byte-identical bake, dep
graph drops the back-edge).

Applied the rename to all 4 stages. Compile now passes.

### Blocker 2 (NOT in the audit): L13.ffn shape mismatch at runtime

With both edits applied, every smoke test fails at the FIRST forward
pass with a tensor shape mismatch — not in L13.ffn but downstream:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (264x800 and 512x1536)
```

(`mat2: 512x1536` is the c_attn projection in a standard transformer
attn — `1536 = 3 * d_model=512`. `mat1: 264x800` is residual-flattened
where the inner dim should be 512.)

Reproducer: any single smoke test (e.g. `test_imm_exit`) — failure is
at the model topology, not a per-test issue.

Root cause hypothesis: pinning the 4 ffn stages to L13 collapses the
L13 ffn slot. The composite builder ultimately swaps
`block[13].ffn = builder.composite` (an `ALUShiftComposite` with its own
512→…→512 internals), so block[13].ffn shape is fine. But the
right-sizing pass for L13.ffn — based on the (now collapsed) per-op
ffn unit allocations — expanded the residual width for downstream
blocks. The 800-wide tensor reaching block[14]'s attn is the symptom.

The audit doc anticipates this in step 3 of its fix sequencing:

> 3. Verify L13.ffn right-sizing honours the 4-stage composite's
>    declared units (`structural_model` ops). If not, add
>    `ffn_units_used` to each stage.

The 4 stages declare `claims=set()` and `produces={'__module_replacement': 'L13.ffn[ALUShiftComposite]'}`
— neither tells the FFN unit allocator what their actual unit count is,
so the allocator must be using a stale/default value that doesn't match
the composite's true shape. This is the `ffn_widths` plumbing
referenced in step 3.

## Files touched this session

- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py` — added a
  comment under `_layer13_attn_dep_anchor` documenting the 2026-06-05
  attempt (no behaviour change; the `layer_idx=13` edit and the SSA
  alias edits were reverted).

## Budget consumed

- 1 baseline smoke run on `TestSmokeMemory` only (confirmed 5/6 fail
  with exit_code=0, matching the audit).
- 1 full smoke run after applying the Option A edits (revealed
  Blocker 2 — the runtime shape mismatch).

## Recommended next step

A 1-op edit (the audit's stated effort estimate) is not enough. The
fix needs:

  1. Add `ffn_units_used=<N>` to each of the 4 `l13_alu_shift_*`
     factory `Operation(...)` constructions, where `<N>` matches the
     hidden width the corresponding `Shift*Stage` module declares.
     `efficient_alu_neural.ShiftBDToGEStage` etc. expose those in
     their `__init__`. Probably 256 each or similar.
  2. Verify the L13.ffn allocator now right-sizes block[13] to fit
     all 4 stages (or, since the install op replaces ffn outright,
     verify the allocator correctly elides L13.ffn when the install
     sentinel is present).
  3. Then re-apply the `layer_idx=13` pins on the anchor + 4 stages
     and the SSA alias on the reads.
  4. Re-run smoke. Expected: 5 memory tests recover; SHL/SHR
     unchanged (Shift tests are in the never-passed list under both
     pre- and post-edit configs); no other regressions.

This is a 2-3 file edit, not 1. Above the audit's "Effort: One agent"
estimate (which was based on the audit's view that pinning was the
only obstacle). The right-sizing plumbing is the gap.

## Refs

- `docs/SMOKE_FAILURES_2026_06_05.md` — cluster #1 (memory).
- `docs/SMOKE_MEMORY_TRACE_20260603.md` — fix options 1/2/3.
- `docs/L13_ANCHOR_DOWNSTREAM_CHAIN.md` — Option A spec (this attempt).
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-229` — the
  composite ops.
- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:393-449` —
  `_layer13_attn_dep_anchor`.
- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:1707` — L16's
  `ALU_LO` write (the validator's false back-edge).
- `c4_release/neural_vm/unified_compiler/ops/l11_ops.py:240-246` —
  Phase 9.B SSA alias precedent (`ALU_LO.*.-1`).
