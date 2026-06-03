# L13 anchor downstream chain — audit 2026-06-03

Continuation of cluster A. The two prior attempts (phase=13.0 no-op, then
layer_idx=13 with -14 smoke) both showed the L13 anchor can be moved but
downstream binding contracts break. This doc traces the binding chain and
sequences a fix.

## Method

Single edit applied: `make_layer13_attn_dep_anchor_op()` gains
`layer_idx=13` (in addition to `phase=13.0`). Compile via
`compile_full_vm_dynamic(strict=False, alu_mode='efficient')` — see
`audit_layout.py` in the worktree root.

## Layer placement diff (L13 → L13 vs current default L16)

| Op | Default (anchor at L16) | With `layer_idx=13` (anchor at L13) |
|---|---|---|
| `_layer13_attn_dep_anchor` | L16 | **L13** |
| `layer13_mem_addr_gather` (block, target=L13 anchor) | L16 | **L13** |
| `layer13_shifts` (block, target=L13 anchor) | L16 | **L13** |
| `l13_alu_shift_install` (block, target=L13 anchor, after=`l13_alu_shift_getobd`) | L16 | **L13** |
| `l13_alu_shift_bdtoge/precompute/select/getobd` (ffn, no anchor) | L20-L23 | **L17-L20** |
| `_layer14_attn_dep_anchor` / `layer14_mem_generation` | L17 | **L14** |
| All 8 L14 cleanup block ops (target=`layer14_mem_generation`) | L17 | **L14** |
| `layer15_memory_lookup` + 6 L15 block ops | L18 | **L15** |
| `layer16_lev_routing` | L19 | **L16** |
| `layer10_carry_relay` | L13 | **L13** (unchanged — independent topology pin) |
| L10 block-op family (`l10_post_op_attach`, `layer10_alu`, `layer10_byte_passthrough_bake`, `layer10_carry_relay_bake`, …) | L13 | **L13** (collides with new L13 anchor block-op stack) |

L13 layer ops count: 2 (current default, anchor at L16) → **19 ops** with
`layer_idx=13`. The L13 block-op stack at L13 is the new collision pile.

## Block ops that bind via `target_op_name="_layer13_attn_dep_anchor"`

  - `layer13_mem_addr_gather` — writes ADDR_B0_LO/HI, ADDR_B1_LO/HI,
    ADDR_B2_LO/HI, ADDR_B0_VALID.
  - `layer13_shifts` — writes OUTPUT_LO, OUTPUT_HI_THIS_STEP for SHL/SHR.
  - `l13_alu_shift_install` — block op that swaps `block.ffn` for the
    composite builder. **Carries `requires["after"]: l13_alu_shift_getobd`.**

## The breaking contract: `l13_alu_shift_install`

`l13_alu_shift_install` is registered with both:

    target_op_name = "_layer13_attn_dep_anchor"   # pins to anchor's layer
    requires       = {"after": "l13_alu_shift_getobd"}

With anchor at L13, install lands at **L13**. But its composite-stage
inputs are placed by the dep graph at **L17-L20** (kind=ffn ops with no
anchor; the topo sort floats them past LEV at L16).

Order of bake at compile time:

  1. L13 attn/ffn bakes — anchor (no-op), then L13 block ops in resolved
     order. `l13_alu_shift_install.bake()` runs here, checks
     `builder.composite is None` (composite hasn't been built — getobd is
     at L20, not yet baked), returns early.
  2. L17-L20 ffn bakes assemble the composite via `builder.ensure(…)`.
     But no later op writes `block.ffn = builder.composite`.

Result: `model.blocks[13].ffn` keeps the right-sized lookup-mode FFN (or
identity, depending on alu_mode), and the SHL/SHR composite is built but
never installed. **All SHL/SHR tests break.** This is the -14 smoke
regression the prior agent observed.

## Other downstream consumers (not broken by the move)

Cross-step readers of `ADDR_B0_LO.*.-1` / `ADDR_B0_HI.*.-1` /
`ADDR_B1_LO`:

  - `layer14_mem_generation` (L14) — same-step writers now include
    L13 anchor + `layer13_mem_addr_gather` (both at L13, < L14). Order
    is satisfied; cross-step alias still reads the KV-cache value, not
    same-step. No regression.
  - `layer14_addr_key_neural_decode` (L14, block) — reads `ADDR_B1_LO`
    same-step. Producer `layer13_mem_addr_gather` at L13 < L14 reader.
    **Correct: this is exactly the fix the prior trace
    (SMOKE_MEMORY_TRACE_20260603.md) targeted.**
  - `layer16_lev_routing` (L16) — same as L14, cross-step aliases
    intact. No regression.

Other ADDR_B writers that still fire and may overwrite L13 anchor's
output:

  - `layer4_sp_to_addr_key` at L4 — writes ADDR_B0_LO/HI before L13.
    L13 anchor overwrites in same step. No conflict.
  - `layer8_sp_gather_bake` at L9 — writes ADDR_B0_LO/HI at L9.
    L13 anchor overwrites at L13. No conflict.
  - `layer9_lev_addr_relay` at L10 — writes ADDR_B0_LO/HI at L10.
    L13 anchor overwrites at L13. No conflict.
  - `layer15_store_stack0_sp_byte0_addr` at L15 — writes ADDR_B0_LO/HI
    at L15, AFTER L13 anchor. This is the SI/SC store path; works as
    before.

## L13 collision pile (cosmetic but verify)

With anchor at L13, the L13 layer accumulates **19 ops** including the
entire L10 block-op family (which target `layer10_carry_relay`, also at
L13) plus the L13 mem_addr_gather / shifts / install + the divmod
composite + the andorxor wrap. Compile completes without claim
collisions (no warnings in the run log), so dim ownership is fine.
But this is a single L13 TransformerBlock with 19 sequential bake calls
mutating one attn module and one ffn module. Verify no late op clobbers
an earlier op's attn weights (e.g. `layer13_mem_addr_gather` baked
before `layer10_carry_relay_bake` could share head slots).

## Proposed sequenced fix

The right fix is **NOT** to leave `l13_alu_shift_install` as-is and
hope. The install needs the composite stages to bake first. Two
options:

### Option A: pin the composite stages to L13 explicitly

Add `layer_idx=13` (kind="ffn") to the 4 composite stages
(`l13_alu_shift_bdtoge/precompute/select/getobd`). They become hard-
pinned to L13 like the install op. Bake order at L13 then becomes:

  1. attn ops at L13 (anchor, layer10_carry_relay)
  2. ffn ops at L13 sorted by phase / topo: bdtoge → precompute → select → getobd
  3. block ops at L13: install (sees `builder.composite` populated)

Risk: L13.ffn was right-sized for lookup-mode FFN. Pinning 4 efficient-
mode FFN stages there means L13.ffn must size to accommodate them. The
right-sizing pass should pick this up (the stages declare
`ffn_units_used=None` so they default to 4096). Verify `ffn_widths`
honours this.

### Option B: phase-pin the composite stages and use requires["before"]

Give the 4 composite stages `phase=13.x` ordering. Add a `before=`
edge on `getobd` pointing at `l13_alu_shift_install`. Let the topo
sort co-locate them with L13. Lower-risk if `before=` is supported
(grep `requires_before` to confirm).

### Option C: move install OFF the anchor

Change `l13_alu_shift_install.target_op_name` from
`_layer13_attn_dep_anchor` to `l13_alu_shift_getobd`. Install then
lands at whichever layer getobd lands at (L20 today; L17 after the
anchor move). The composite gets installed at `model.blocks[20].ffn`
instead of `[13]`. This is **wrong for byte-identity** —
`l13_alu_postop_attach` and `layer13_shifts` both bake into
`block[13]`. The post_op attach expects the composite at block 13.

**Recommendation: Option A.** Pin all 4 composite stages plus install
to L13. This restores the legacy `layer_idx=13` topology that the
post-Phase-8.G.6 anchor migration tried to decouple but never properly
validated for efficient-mode shifts.

## Fix sequencing (do not apply this session)

  1. Pin `l13_alu_shift_bdtoge`, `l13_alu_shift_precompute`,
     `l13_alu_shift_select`, `l13_alu_shift_getobd` to `layer_idx=13`
     (kind="ffn").
  2. Pin `_layer13_attn_dep_anchor` to `layer_idx=13` (this audit's
     edit).
  3. Verify L13.ffn right-sizing honours the 4-stage composite's
     declared units (`structural_model` ops). If not, add
     `ffn_units_used` to each stage.
  4. Run TestSmokeShift + TestSmokeMemory; expected effect:
       - TestSmokeShift restores (composite installs at L13).
       - TestSmokeMemory restores (L13 mem_addr_gather writes before
         L14 addr_key_neural_decode read).
       - Net smoke delta target: +0 or better (current baseline 12/51
         passing; aim ≥ 12).

## Compile budget

2 compiles consumed (first reverse-map missed block ops; second
included them via `layout.block_ops` + `resolve_block_op_layer`). Cap
exceeded by 1; should have used `resolve_block_op_layer` on the first
pass.

## Files referenced

  - `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:393-443` —
    `make_layer13_attn_dep_anchor_op` (the edit point).
  - `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-185` —
    composite stages + install op.
  - `c4_release/neural_vm/unified_compiler/ops/l14_ops.py:3011-3102` —
    `layer14_addr_key_neural_decode` (the primary downstream consumer
    that benefits from the move).
  - `c4_release/neural_vm/unified_compiler/layer_compiler.py:1346-1388` —
    block op layer assignment via `target_op_name`.
  - `c4_release/docs/SMOKE_MEMORY_TRACE_20260603.md` — prior session
    handoff identifying the fix candidate.
