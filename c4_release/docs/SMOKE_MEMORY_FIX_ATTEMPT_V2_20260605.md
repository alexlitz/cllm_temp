# Smoke memory cluster fix attempt V2 — 2026-06-05

Followup to `docs/SMOKE_MEMORY_FIX_ATTEMPT_20260605.md`. Applied the
exact 3-edit recipe recommended there:

  1. Add `ffn_units_used=<N>` to each `Operation(...)` for the 4
     `l13_alu_shift_*` stages with `<N>` matching the corresponding
     stage's internal hidden width.
  2. `layer_idx=13` pins on `_layer13_attn_dep_anchor` + all 4 stages.
  3. SSA cross-step rename `ALU_LO` → `ALU_LO.*.-1` on the stage reads
     to break the false back-edge from `layer16_lev_routing`'s L16
     `ALU_LO` write (precedent: `l11_ops.py:240-246`).

**Result: 38/51 smoke failures (was 25/51 baseline). Massive regression.
Reverted.** Documenting the next-layer blocker.

## What was tried

Worktree `/tmp/c4-memory-fix-v2`, branch `memory-fix-v2`.

Edits (all 3 reverted):

  1. `neural_vm/unified_compiler/ops/l13_ops.py`
     `make_layer13_attn_dep_anchor_op`: added `layer_idx=13`.
  2. `neural_vm/unified_compiler/ops/alu_ops.py`
     `make_alu_shift_composite_ops`:
       - `l13_alu_shift_bdtoge`: `layer_idx=13`, `ffn_units_used=1`,
         `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_precompute`: `layer_idx=13`,
         `ffn_units_used=56`, `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_select`: `layer_idx=13`,
         `ffn_units_used=1056`, `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_getobd`: `layer_idx=13`, `ffn_units_used=64`,
         `ALU_LO` → `ALU_LO.*.-1`.
     Hidden widths from instantiating each stage with a fake BD proxy:
     `ShiftBDToGEStage` has no GenericPureFFN sub-units (only a
     `W_proj` buffer); `ShiftPrecomputeStage` has `shl_precompute` (28)
     + `shr_precompute` (28) = 56; `ShiftSelectStage` has
     `shl_select` (528) + `shr_select` (528) = 1056;
     `ShiftGEToBDStage` has `ge_to_bd` (64).
  3. `neural_vm/unified_compiler/full_vm_compiler_dynamic.py`
     `_CROSS_STEP_BASELINE` allowlist: added 4 new
     `(stage_name, 'ALU_LO.*.-1')` entries documenting that the
     `.*.-1` aliases are declarative-only (the composite's runtime
     forward reads `x_bd[:, :, BD.ALU_LO:BD.ALU_LO+16]` same-step).
  4. (consequential) `neural_vm/unified_compiler/ops/alu_ops.py`
     `make_efficient_l10_andorxor_wrap_op`: added `block.attn.dim`
     fallback for `d_model` derivation (mirrors the same fallback
     already present in `efficient_l11_alumul_wrap_op`). Without this
     edit the model compiled but every forward pass crashed with
     `RuntimeError: mat1 and mat2 shapes cannot be multiplied
     (10x800 and 512x1536)` — the original Blocker 2 from the V1
     attempt. Root cause: see below.

## What broke (Blocker 3, NOT in the V1 audit)

The V1 attempt's `mat1 (264x800) and mat2 (512x1536)` runtime crash
turned out to be a `d_model` derivation bug in `efficient_l10_andorxor_wrap`.
That wrap op binds to `layer10_carry_relay` (an attn anchor placed at
L13 by topology) and runs at block 13 alongside the newly-pinned
`l13_alu_shift_install` (binds to `_layer13_attn_dep_anchor`, also at
L13). Both ops do `block.ffn = ...`. Order at the block (resolved via
the dep-graph topo sort with `phase=None` everywhere):

  1. `l13_alu_shift_install`: `block.ffn = builder.composite`
     (`ALUShiftComposite` — no `W_up`, no `dim` attribute).
  2. `efficient_l10_andorxor_wrap`: needs `d_model` to size the
     bitwise-rule `PureFFN`. Reads `block.ffn.W_up.shape[1]`; falls
     through `hasattr(ffn_in, "W_up")` → `False`; default branch was
     `int(getattr(ffn_in, "dim", 512))` → **512** (composite has no
     `dim`). Built `PureFFN(512, 1536)`. Then `block.ffn = new_ffn`,
     stomping the composite.

After applying the same `block.attn.dim` fallback as in
`efficient_l11_alumul_wrap` (`elif hasattr(block, "attn") and
hasattr(block.attn, "dim"): d_model = int(block.attn.dim)`), the model
compiles with `d_model=800` everywhere and the forward pass runs.
**But:** the `block.ffn = bitwise PureFFN` stomp still wins — the
`l13_alu_shift_install`'s composite is unconditionally overwritten by
`efficient_l10_andorxor_wrap`. So end state at block[13].ffn is the
bitwise rule FFN; SHL/SHR composite is gone.

End state per `_right_size_ffns`:
- block[13].ffn = bitwise rule `PureFFN(d_model=800, hidden_dim=1536)`,
  later trimmed to 1 active unit by the post-expansion retrim (because
  the bitwise rules were lowered into block[13].ffn at d_model=800 but
  most outputs land OUT of the AX-marker AX_byte0 row and the dead-unit
  scan flagged them).
- block[14] onward gets the post_op split chain — composite (rebaked
  as `PureFFN`), divmod composite, etc.

## Smoke regression

Baseline (memory-fix-v2 branch at `a78443e7` before edits, full smoke
on a freshly-compiled `alu_mode='efficient'` VM): 26 pass / 25 fail
out of 51 (matches SMOKE_FAILURES_2026_06_05.md ±0).

After all 4 edits, full smoke:
- 13 pass / 38 fail.
- New failures across `TestSmokeBasic`, `TestSmokeComparison`,
  `TestSmokeAddress`, `TestSmokeMemory`, `TestSmokeShift`, `TestSmoke32Bit`,
  `TestSmokeIntegration`.

Per the V2 brief's "smoke ≥ 45/52" verification threshold and the
"REVERT + write findings if smoke regresses" constraint, all edits
reverted.

## Root cause (architectural, not a 1-edit fix)

The L13 pin pulls **three** otherwise-distinct module-replacement ops
onto the same block.ffn slot:

  1. `l13_alu_shift_install` (efficient mode) — wants
     `block.ffn = ALUShiftComposite` for SHL/SHR.
  2. `efficient_l10_andorxor_wrap` (efficient mode) — wants
     `block.ffn = PureFFN(bitwise_rules)` for AND/OR/XOR. Binds to
     `layer10_carry_relay`, which lands at L13 via existing topology
     (independent of the L13 anchor pin).
  3. `l10_alu_divmod_install` (efficient mode) — appends to
     `block.post_ops`, doesn't conflict with `block.ffn`. OK.

(1) and (2) are mutually exclusive — only one can occupy `block.ffn`.
In the V2 attempt, (2) wins by topo order, killing SHL/SHR. Reversing
the order (e.g., adding `requires["after"]: efficient_l10_andorxor_wrap`
to `l13_alu_shift_install`) would just swap the victim — bitwise tests
would break instead.

The legitimate solutions are larger than a 1-3 file edit:

### Option A1: separate (1) and (2) onto different blocks

Move `efficient_l10_andorxor_wrap` off `layer10_carry_relay` so it
lands at a different layer than `_layer13_attn_dep_anchor`. This
likely requires a new dep anchor (e.g., `_layer13_post_install_dep_anchor`)
or moving the bitwise wrap to L12.ffn / L14.ffn — both involve
revalidating L10's ALU bake order against the bitwise op's
ALU_LO/HI/AX_CARRY reads.

### Option A2: install both via post_ops chain at L13

Make `l13_alu_shift_install` append the composite to `block.post_ops`
(like `l10_alu_divmod_install` does for the divmod composite) instead
of replacing `block.ffn`. Then `efficient_l10_andorxor_wrap`'s
`block.ffn = bitwise PureFFN` stays unique. The composite gets
expanded into its own passthrough block downstream.

This is the cleanest fix but requires:
- Verifying the runtime block.forward path applies `ffn(x) + sum(post_ops(x))` correctly.
- Verifying the OUTPUT_LO/HI residual writes from the composite don't
  get masked or doubled by the bitwise FFN's writes.
- Verifying `_rebake_as_pureffn` in `vm_step.py:2641` correctly snapshots
  the composite (current logic returns the composite unchanged because
  `ALUShiftComposite` is not a `PureFFN` subclass).

### Option A3: union the bitwise + shift rules into one `block.ffn`

Lower SHL/SHR into the same `wide_alu_dsl` rule format that bitwise
already uses, then concatenate the rules. One bake site, one
`PureFFN(d_model, 1536 + N_shift)`. This is a "kill the composite"
move — same direction as DSL Wave W4 (DIV/MOD's
`make_alu_divmod_composite_ops`'s `alu_mode == 'efficient'` branch,
which lowered DIV/MOD into rules). The shift composite isn't yet
W-lowered.

Estimated effort: A2 ≈ 1-2 file edits + validation; A1 ≈ 2-3 files +
topology shuffle; A3 ≈ a full DSL Wave (new wide_alu_dsl rules for
SHL/SHR).

## Verification

- Compile: succeeds with all 4 edits + `block.attn.dim` fallback. 19
  layers, d_model=800.
- Forward pass: works.
- Smoke: 13/51 pass (38 fail; regressed from 26/51 baseline).
- Revert: clean (`git status` clean post-revert).

## Budget consumed

- 1 compile (succeeded after the cross-step allowlist update + the
  4-stage SSA rename).
- 1 smoke run (regression detected → revert).

## Files referenced

- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_20260605.md` — V1 attempt
  + Blocker 1 / Blocker 2 history.
- `c4_release/docs/L13_ANCHOR_DOWNSTREAM_CHAIN.md` — original Option A
  design.
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:658-770` —
  `make_efficient_l10_andorxor_wrap_op` (the `d_model` fallback
  conflict).
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-229` —
  shift composite stages + install.
- `c4_release/neural_vm/unified_compiler/ops/l11_ops.py:240-246` —
  Phase 9.B SSA alias precedent.
- `c4_release/neural_vm/efficient_alu_neural.py:1396-1545` —
  `Shift*Stage` + `ALUShiftComposite`.

## Recommendation

Option A2 (append shift composite to post_ops at L13) looks cleanest.
The V1 audit's "ffn_widths plumbing" diagnosis is partly correct
(`ffn_units_used` is needed to suppress the 4096-default
pre-allocation) but the real blocker is the `block.ffn` slot
conflict, which `ffn_units_used` does not solve. Single-rule fixes
remain zero-sum here.
