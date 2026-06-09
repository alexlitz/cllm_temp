# Long-division Bug #36 attribution (2026-06-09, oracle-localized)

Date: 2026-06-09
Base ref: `main` HEAD `c0b02e7d` (`feat(l14): head-5 slot-2 HAS_SE guard
for JSR step-0`).
Brief: Bug #36 long-division projection — ~75 tests in the 1096
long-division cluster (basic `div_*` 22, plus `expr_mul_div_*`,
`mod_*`, `gcd_*`, `MOD_iterative` from `BUG_CATALOG.md` §Bug #36).

## TL;DR

Bug #36 is **structurally multi-rule** and **explicitly outside the
worktree's safe-edit perimeter** (constraint: do not touch L10 / L14 /
L15). The fault locus is `FlattenedDivMod` (an `nn.Module` post_op
appended at L10 phase=10.0–10.2) plus the L11–L17 OUTPUT_LO/HI
projection chain. The block-aware oracle was extended to DIV/MOD in
commits `4a354699` / `ec4ff967`, but the per-dim replay diff against
`IMM 1000; PSH; IMM 7; DIV; EXIT` does **not** localize a single
divergent block isolable to DIV — divergences appear at block 3
(generic warmup zero-propagation, same as every program) and block 12
(L10 PSH `STACK0_BYTE_VAL_1_LO` byte-1 staging, the same surface as
the open L10 PSH `MEM_addr0` bug). Both are L10 issues and both are
upstream of DIV semantics. No new fix landed; this doc complements the
existing
[`DIV_22_FAILING_ATTRIBUTION_2026_06_09.md`](DIV_22_FAILING_ATTRIBUTION_2026_06_09.md)
with the oracle-localization step the brief asked for and an
explicit structural-gap statement on `wide_div_rules`.

## Oracle replay-diff localization (Step 1–2 of the brief)

Program: `IMM 1000; PSH; IMM 7; DIV; EXIT`. Expected at EXIT: AX =
1000 / 7 = 142 (0x8E), so `AX_FULL_LO ≈ 0x8E`, `AX_FULL_HI = 0`.
Dividend 1000 = `0x03E8` requires byte-1 (`0x03`) and byte-0
(`0xE8`).

Note: `replay_expected_diff.py` does not include `AX_FULL_LO/HI` or
`TEMP_DIV_*` in `SUPPORTED_DIM_FAMILIES` — the brief's dims aren't
oracle-modeled yet. The closest covered dims are `OUTPUT_LO`,
`OUTPUT_HI`, `AX_CARRY_LO`, `AX_CARRY_HI`, and
`STACK0_BYTE_VAL_{1,2,3}_{LO,HI}`.

Per-dim first-divergent block (declarations-only, `--program "IMM
1000; PSH; IMM 7; DIV; EXIT"`):

| Dim | First block | Step | Suggested op | Notes |
|---|---|---|---|---|
| `OUTPUT_LO` | block 3 | 0 | `layer3_carry_forward_attn` | Generic step-0 zero (every program) |
| `OUTPUT_HI` | block 3 | 0 | `layer3_carry_forward_attn` | Same |
| `AX_CARRY_LO` | block 3 | 0 | `layer3_carry_forward_attn` | Same |
| `AX_CARRY_HI` | block 3 | 0 | `layer3_carry_forward_attn` | Same |
| `STACK0_BYTE_VAL_1_LO` | block 12 | 0 | `layer10_psh_ax_broadcast` | L10 PSH byte staging (the open `L10_PSH_addr0_e0` family) |

**Finding**: the oracle does **not** isolate a DIV-specific block.
The block-3 divergences are the standard initial-step
`layer3_carry_forward_attn` zero-propagation hit by every program at
step=0. The block-12 divergence is L10 PSH byte staging — the same
class as the open `L10 PSH addr0_e0 missing OP_ENT guard` bug pinned
at `l10_ops.py:3888-3927`. Neither block is downstream of the actual
DIV step (step=3 in this program). The DIV-step divergence in
`AX_FULL_LO/HI` is invisible because those dims have no oracle
projection.

This suggests the upcoming wave for oracle-projecting DIV should add
`AX_FULL_LO`/`AX_FULL_HI` to `SUPPORTED_DIM_FAMILIES` so a future agent
can localize the DIV step itself.

## Plumbing map (Step 3 of the brief)

DIV semantics in the unified compiler are **not** in `l11_ops.py /
l12_ops.py / l13_ops.py` — those files have **zero** references to
`DIV`, `FlattenedDivMod`, or `divmod_longdiv` (grep confirmed). DIV
lives entirely in:

1. **L10 post_op chain** (constrained — do not touch):
   - `c4_release/neural_vm/unified_compiler/ops/shared.py:424`
     `_FlattenedDivModBuilder` builds the 4-stage composite.
   - `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:1359-1425`
     wires phases 10.0 (BD→GE), 10.1 (long-division pipeline), 10.2
     (GE→BD writeback) into `block.post_ops`.
   - `c4_release/neural_vm/efficient_alu_divmod_split.py` —
     `FlattenedDivMod` composite (8 outer × 3 inner long-division
     loop).
   - `c4_release/neural_vm/alu/ops/divmod_longdiv.py:344-359` —
     `build_div_layers_longdiv` (`ClearDivSlotsFFN` +
     `LongDivisionModule` + `EmitDivResultModule`).
2. **L8 attention `AX_FULL_LO/HI` staging** (constrained-adjacent;
   not L10/14/15 but ties to L10 post_op input).
3. **L11–L17 OUTPUT_LO/HI projection chain**: same generic
   projection used by every ALU op; no DIV-specific code.

L17 ops file does **not** exist — the project's layer count tops out
at `l16_ops.py` in `unified_compiler/ops/`. The brief's `l17_ops.py`
reference is stale; L11–L13 are the only intermediate files between
the L10 post_op and L16. None of L11/L12/L13 carry DIV-specific
plumbing.

## Wide-DIV DSL gap (Step 4 of the brief)

`wide_alu_dsl.wide_div_rules` is **explicitly limited to
`width_bytes=1`**. From the source
(`c4_release/neural_vm/unified_compiler/wide_alu_dsl.py:866-882`):

```python
if width_bytes > 1:
    raise NotImplementedError(
        f"wide_div_rules: width_bytes={width_bytes} is deferred — "
        f"per-nibble independent division does not compose into "
        f"wide-operand division (e.g. 0xFF / 0x0F = 0x11, but "
        f"per-nibble would give 1 and a zero-divide guard). "
        f"Multi-byte DIV requires the long-division pipeline "
        f"implemented by FlattenedDivMod (see efficient_alu_"
        f"divmod_split.py) and is tracked under "
        f"docs/DSL_W5_MULDIV_LIMIT.md."
    )
```

This is the **structural gap**: there is no rule-set-level
multi-byte DIV path. All multi-byte DIV / MOD go through the
imperative `FlattenedDivMod` composite. Single-byte DIV
(`width_bytes=1` POC) covers the smoke `test_mul_basic` cluster but
not the 1096 long-division cluster, which routes multi-byte values
through `LongDivisionModule.forward` (an `nn.Module`, not in IR).

This matches `BUG_CATALOG.md` §Bug #36 ("needs FFNRule IR migration
of the projection chain"; 3-5 day estimate) and
`docs/DSL_W5_MULDIV_LIMIT.md`.

## Why no fix is shipped from this brief (Step 5)

The worktree constraint **"DO NOT touch L10/L14/L15 memory cluster"**
removes the entire structural fix surface from this brief's perimeter:

- The 4-stage `FlattenedDivMod` composite lives in **L10 post_ops**
  (phases 10.0 / 10.1 / 10.2 in `alu_ops.py:1359-1425`).
- The `LongDivisionModule.forward` imperative module lives in
  `alu/ops/divmod_longdiv.py` and is wrapped into L10's
  `block.post_ops`.
- The `AX_FULL_LO/HI` byte-1 staging implicated by the
  existing attribution doc as the leak source for `div_5`
  (176/4 → neural=0) and similar cases is at L8 attention reading
  `CLEAN_EMBED_LO/HI` for the `STACK0_BYTE1` cummax fallback — that's
  L8, but the consumer is L10's `BDToGEConverter`.

Any single-rule fix at L11–L13 / L16 OUTPUT_LO/HI projection is
documented as zero-sum per `feedback_single_rule_fixes_are_zero_sum.md`
(0/5 historical agent attempts net positive) and would not catch
the SLOT_QUOTIENT[2..7] drift that originates inside `FlattenedDivMod`
itself.

The existing attribution doc
[`DIV_22_FAILING_ATTRIBUTION_2026_06_09.md`](DIV_22_FAILING_ATTRIBUTION_2026_06_09.md)
already enumerates the 22 `div_*` rowset and the 14
`expr_mul_div_*` rowset, with the byte-1 dependence diagnosis and
`neural=0` / `neural=1` / `neural=16` failure-shape signatures. This
doc complements that one by adding (a) oracle-localization (per the
brief), (b) the `l17_ops.py` non-existence finding, (c) the explicit
`wide_div_rules` structural-gap quote, and (d) the worktree-constraint
rationale for no-fix.

## Next-wave entry points

For a future agent **without** the L10/L14/L15 constraint:

1. **Oracle projection for `AX_FULL_LO/HI`** —
   `c4_release/neural_vm/dim_oracle.py`. Add per-byte projection
   rules so the DIV step is visible to `replay_expected_diff.py`. This
   is read-only on the model; it just extends the oracle.
2. **Clear SLOT_QUOTIENT[2..7] / SLOT_REMAINDER[2..7] FFN rule** in
   `EmitDivResultModule` gated on `OP_DIV / OP_MOD`. Cheap to author
   via `FFNRule.gated_write`; needs byte-identity gate via
   `compare_symbolic_to_lowered_ffn`. (This is L10 surface — out of
   scope for this brief.)
3. **`LongDivisionModule.forward` → FFNRule migration** (the 3-5
   day effort estimate in `BUG_CATALOG.md` §Bug #36). 24 sub-ops
   (8 outer × 3 inner) lower regularly to FFNRule via the
   building-blocks DSL. Unlocks structural byte-1 clearing.
4. **Audit `prev_stack_lo/hi` fallback** at
   `efficient_alu_neural.py:174-205` for stale-STACK0 injection in
   single-byte dividend cases.

## Cross-references

- [`DIV_22_FAILING_ATTRIBUTION_2026_06_09.md`](DIV_22_FAILING_ATTRIBUTION_2026_06_09.md)
  — same-day per-row failure enumeration; this doc complements it
  with oracle-localization and structural-gap callouts.
- [`BUG_CATALOG.md`](BUG_CATALOG.md) §Bug #36 — root surface.
- [`DSL_W5_MULDIV_LIMIT.md`](DSL_W5_MULDIV_LIMIT.md) — `wide_div_rules`
  multi-byte deferral.
- `c4_release/neural_vm/unified_compiler/wide_alu_dsl.py:866-882` —
  the explicit `NotImplementedError` for `width_bytes > 1`.
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:1359-1425`
  — L10 post_op phase wiring.
- `c4_release/neural_vm/efficient_alu_divmod_split.py` —
  `FlattenedDivMod` composite.
- `c4_release/neural_vm/alu/ops/divmod_longdiv.py:344-359` —
  `build_div_layers_longdiv`.
- Memory note `feedback_single_rule_fixes_are_zero_sum.md` — 0/5
  historical record.
- Memory note `project_l10_psh_addr_ent_bug.md` — open L10 PSH
  staging bug that surfaces in the same block-12 oracle hit for this
  DIV program.
