# GE-format wide_div_rules — install-site staging blocker (2026-06-10)

> **RESOLVED 2026-06-11 (DSL Wave W6).** The byte-accurate GE-format
> lookup is now wired into the L10 efficient-mode install
> (`alu_ops.make_install`). `test_div_basic` (84/2=42) and
> `test_mod_basic` (43%10=3) PASS; the two `xfail` markers are removed.
> Zero regression on the smoke gate (24→26 pass, only div/mod flipped).
>
> **What unblocked it (the analysis below was correct but incomplete):**
> the @0 magnitude artifact persists on the **MARK_AX row** — the row
> the install gates on. Wall-1 (block-8 head-0 slope=0.1) cleaned the
> **SE** row only. Crucially, the artifact is a *constant* additive
> ~5.56 on cell 0 of ALU_LO/HI (verified across many dividends via
> `tools/probe_div_operand_clean.py`): cell-0 reads ~5.56 when the true
> nibble is elsewhere, and ~11.38 (= 5.56 + 5.82) when the true nibble
> IS 0. So a **constant subtraction of 5.56 from cell 0** (gated
> OP_DIV/OP_MOD + MARK_AX) cleans BOTH cases — leaving a clean
> per-nibble one-hot — without needing a 64-dim clean band (there are
> only 3 free residual dims; d_model=872). This is the fix the analysis
> below ("raise the threshold doesn't work") didn't consider.
>
> Two additional facts the original blocker missed:
>   1. The GE-format threshold (150) assumed operand cells of magnitude
>      1.0. The cleaned dividend cells are ~5.82, so the per-cell
>      condition weight is rescaled to 30/5.82 (helper now takes
>      `dividend_cond_weight` / `divisor_cond_weight` / `threshold`).
>      AX_CARRY (divisor) is already ~1.0 → default weight.
>   2. DIV decodes from the **MARK_AX row** (the pre-Wave-B AX-row decode
>      path), NOT the SE row — so it is NOT Wall-4-blocked like the
>      migrated CMP path. Verified via `tools/probe_div_decode_row.py`.
>
> Remaining edge cases (NOT targets, NOT regressions): dividends whose
> L8 operand-gather is itself corrupt (e.g. a=255 gathers to nibble 8,
> not 15) still mis-divide. That is an upstream operand-gather bug,
> independent of the GE-format install.
>
> The rest of this document is the original (now-superseded) blocker
> analysis, kept for provenance.

---

Status: **architectural blocker** (SUPERSEDED — see resolution above).
Companion to
[`DSL_W5_MULDIV_LIMIT.md`](DSL_W5_MULDIV_LIMIT.md),
[`LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md`](LONG_DIVISION_FFN_RULE_INFEASIBILITY_2026_06_09.md),
and [`DIV_22_FAILING_ATTRIBUTION_2026_06_09.md`](archive/DIV_22_FAILING_ATTRIBUTION_2026_06_09.md).

## What landed

`wide_alu_dsl.wide_div_rules_ge_format(width_bytes=1, ...)` is a new
DSL helper that emits a byte-accurate flat 8-bit cross-product lookup:
65,536 `FFNRule`s per opcode batch (256 dividend × 256 divisor pairs).
Each rule fires on a 5-way AND across `(MARK_AX, ALU_LO[a_lo],
ALU_HI[a_hi], AX_CARRY_LO[b_lo], AX_CARRY_HI[b_hi])` gated on
`OP_DIV`/`OP_MOD`, writing the byte-accurate quotient/remainder nibbles
to `OUTPUT_LO[out_lo]` / `OUTPUT_HI[out_hi]`. Total per call:
131,072 rules for the DIV + MOD pair.

Symbolic verification confirms correctness — at the rule level,
`(a=84, b=2)` writes `OUTPUT_LO+10` and `OUTPUT_HI+2`, reconstructing
`0x2A = 42`.

## Why the runtime install is **not** wired

The install op at
`unified_compiler/ops/alu_ops.py:make_alu_divmod_composite_ops:make_install`
appends a `PureFFN` post_op to `model.blocks[10].post_ops`. After
`_expand_wrapper_blocks` (Phase 0), that post_op becomes its own
TransformerBlock — measured at block 23 of the 37-block efficient-mode
model. At block 23's input, the residual is NOT clean one-hot:

```
Probed at L10 input, OP_DIV+MARK_AX hot row, dividend=84:
  ALU_LO band:
    [ 0]  6.2838       <-- noise, higher than active cell
    [ 4]  4.3444       <-- active cell (a_lo=4 for 84=0x54)
    [ 8]  0.4530
    other cells: 0
  ALU_HI band:
    [ 0]  6.2963       <-- noise
    [ 5]  4.3444       <-- active cell (a_hi=5)
```

`BDToGEConverter._clean_onehot` (`efficient_alu_neural.py:97-115`)
thresholds at `> 0.5` then sums `k * one_hot[k]` to extract the scalar
nibble. With cells [0]=6.28 and [4]=4.34 both surviving the threshold,
the sum yields `0 + 4 = 4` (correct), because the active "true" cell
contributes `k=4` and the noisy [0] cell contributes `k=0` (no harm).

The flat FFNRule lookup cannot replicate this trick. Each rule reads
raw cell values via a weighted sum (`W_up @ x + b_up`). With the noisy
ALU_LO[0]=6.28 cell, the rule for `(a=0, b=2)` fires harder
(score-from-ALU_LO = 30 × 6.28 = 188.4) than the rule for `(a=84, b=2)`
(score-from-ALU_LO = 30 × 4.34 = 130.2). Both fire above any threshold
that would let the `(a=84, b=2)` rule fire. The dominant `(a=0, b=2)`
rule writes `OUTPUT_LO+0` and `OUTPUT_HI+0` (because `0/2 = 0`),
yielding `result = 0` instead of `42`.

## Why "raise the threshold" does not work

For any threshold T:

- All-on score with clean one-hot: `40 + 4*30 = 160`. Pick `T < 160`.
- Noisy run-time score: each operand cell at 6.28 yields contribution
  `30 * 6.28 = 188.4`. Sum across 4 operand bands: up to `40 + 4*188.4 = 793.6`.
- A "missing match" run-time score: 3 operand cells at 6.28 + 1 at 0
  = `40 + 3*188.4 = 605`.

Any `T` in `[160, 793.6]` lets the correct-match rule fire AND any
3-out-of-4 noisy match also fire. The over-strong noisy contribution
guarantees that the rule for `(a=0, b=2)` (where ALU_LO[0]=6.28 and
ALU_HI[0]=6.28 both contribute the noise) outputs more than the
correct `(a=84, b=2)` rule.

The proportional `silu(score - T)` output makes this worse: higher
scores → larger writes. The noisy rule wins the argmax race in the
downstream OUTPUT decoding.

## Workarounds that would unblock

1. **Install upstream of the L10 ALU passthrough.** Per the L10 probe,
   the residual at block 6 already has `ALU_LO=0` (no dividend), at
   block 9 the dividend lands. The clean-onehot window may be at
   block 9 or earlier. Need to bake the 131,072-rule FFN at the
   `layer10_alu` kind="ffn" target instead of as a post_op. Requires
   re-routing the install op from kind="block" to kind="ffn" with
   `target_op_name="layer10_alu"`, and merging the rules into the
   existing 1,846-unit L10 layout via the `_l10_unit_allocator` (the
   merger needs a new sub-stage entry in `_L10_FFN_UNIT_LAYOUT_MAIN`).

2. **Add a `_clean_onehot` FFN stage before the lookup.** A 64-unit
   FFN that thresholds each of `ALU_LO[0..15]`, `ALU_HI[0..15]`,
   `AX_CARRY_LO[0..15]`, `AX_CARRY_HI[0..15]` at 0.5 and writes a
   clean 0/1 indicator to a fresh "clean band" (e.g.
   `ALU_LO_CLEAN+k`). Then point the GE-format rules at the clean
   band. Requires reserving 64 new residual dims and writing the
   cleanup ops.

3. **Use the existing `BDToGEConverter` pipeline.** This is what
   `FlattenedDivMod` does — extract scalar nibbles via the
   already-baked `BDToGEConverter`, then run the long-division
   pipeline. The byte-accurate `wide_div_rules_ge_format` does
   *not* re-use this; it bypasses the BD→GE conversion entirely.

## Current state

- `wide_div_rules_ge_format` exists in `wide_alu_dsl.py` and is
  symbolically correct. Tests in
  `tests/test_smoke.py::test_div_basic` and `::test_mod_basic` remain
  `xfail` for `MUL/DIV/MOD arch blocked` per memory
  `project_mul_div_mod_arch_blocked.md`.
- The L10 efficient-mode install op continues to use the legacy
  per-nibble `wide_div_rules(width_bytes=1)` (mathematically wrong for
  cross-nibble dividends — see `DSL_W5_MULDIV_LIMIT.md`) so the model
  produces *some* deterministic output for DIV/MOD in efficient mode.
- For byte-accurate DIV/MOD in efficient mode, the established
  `FlattenedDivMod` composite remains authoritative (used by lookup
  mode at block 24 of the lookup-mode model). The relevant memory
  note `project_mul_div_mod_arch_blocked.md` should be updated to
  reflect that the per-nibble bug now has a symbolically-correct
  replacement gated on staging.

## Cross-references

- `c4_release/neural_vm/unified_compiler/wide_alu_dsl.py:` —
  `wide_div_rules_ge_format` implementation.
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:` —
  `make_alu_divmod_composite_ops` install path (efficient mode).
- `c4_release/neural_vm/efficient_alu_neural.py:97-115` —
  `BDToGEConverter._clean_onehot` (the cleaner my install bypasses).
- `c4_release/neural_vm/efficient_alu_divmod_split.py:246` —
  `FlattenedDivMod` (current byte-accurate authority).
