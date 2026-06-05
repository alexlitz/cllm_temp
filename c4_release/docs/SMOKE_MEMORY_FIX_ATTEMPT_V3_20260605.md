# Smoke memory cluster fix attempt V3 — 2026-06-05

Followup to `docs/SMOKE_MEMORY_FIX_ATTEMPT_V2_20260605.md`. Applied
**Option A2** from the V2 recommendation: install the SHL/SHR composite
via `block.post_ops.append(...)` instead of `block.ffn = ...`, modelled
on `l10_alu_divmod_install`'s post_op pattern.

**Result: 38/51 smoke failures (was 26/51 baseline). Identical
regression to V2. Reverted.** A2 alone does not fix the cluster.

## What was tried

Worktree `/tmp/c4-memory-fix-v3`, branch `memory-fix-v3`.

Edits (all reverted):

  1. `neural_vm/unified_compiler/ops/alu_ops.py`
     `make_alu_shift_composite_ops`:
       - `make_install()` bake: changed
         `block.ffn = builder.composite` →
         `block.post_ops.append(builder.composite)`.
         Module-replacement sentinel updated to
         `L13.post_ops[ALUShiftComposite]`.
       - `l13_alu_shift_bdtoge`: added `layer_idx=13`,
         `ffn_units_used=1`, renamed read `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_precompute`: added `layer_idx=13`,
         `ffn_units_used=56`, renamed read `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_select`: added `layer_idx=13`,
         `ffn_units_used=1056`, renamed read `ALU_LO` → `ALU_LO.*.-1`.
       - `l13_alu_shift_getobd`: added `layer_idx=13`,
         `ffn_units_used=64`, renamed read `ALU_LO` → `ALU_LO.*.-1`.
       - All 4 stage `produces` sentinels updated to
         `L13.post_ops[ALUShiftComposite]`.
  2. `neural_vm/unified_compiler/ops/l13_ops.py`
     `make_layer13_attn_dep_anchor_op`: added `layer_idx=13`.
  3. `neural_vm/unified_compiler/full_vm_compiler_dynamic.py`
     `CROSS_STEP_DOCUMENTED_SAFE`: added the 4
     `(l13_alu_shift_*, ALU_LO.*.-1)` entries (declarative-only —
     the composite's runtime forward reads ALU_LO same-step).

(Skipped the V2 `block.attn.dim` d_model fallback edit: A2 no longer has
the shift install racing for `block.ffn`, so when
`efficient_l10_andorxor_wrap` reads `block.ffn.W_up.shape[1]` to derive
`d_model`, the original vanilla `PureFFN(d_model=800, ...)` constructed
by `TransformerBlock.__init__` is still in place. Verified at compile
time — no shape crashes.)

## What broke (Blocker 4, NOT in the V2 audit)

The model compiles cleanly. The compile diagnostics confirm A2 took
effect structurally:

- Before edits: 36 blocks; `ALUShiftComposite` lands at `B25`
  (after `_expand_wrapper_blocks` splits L17-L23 worth of upstream
  post_ops into separate blocks); 1 install op consumes the L13
  `block.ffn` slot for the bitwise PureFFN via
  `efficient_l10_andorxor_wrap` (this is what the v2 doc described as
  the "win by topo order").
- After edits: 30 blocks; `ALUShiftComposite` lands at `B15`
  (right after `L13`'s post_op expansion); same bitwise PureFFN at
  `B13.ffn`; no slot conflict.

But every smoke test that exercises ALU output (38/51 fails — the
SAME breadth as the V2 regression, with the SAME nominal "expected X,
got 0" pattern) regresses. The shift composite now runs **10 layers
earlier than baseline** (B15 vs B25). Many AX-write smoke tests
(`test_add_basic`, `test_or_basic`, `test_eq_true`, …) need the L14-L24
setup ops to have already populated `ALU_LO/HI`, `AX_CARRY_LO/HI`,
`OUTPUT_LO/HI`, `MARK_AX` etc. at the AX-marker row BEFORE any
output-shaping post_op fires. With the composite jammed in at B15:

  - The composite fires before L14 `mem_generation`, before L14
    `addr_key_neural_decode`, before L15 `memory_lookup`, before L16
    `lev_routing`, etc.
  - Its `forward` reads `x_bd[:, :, BD.ALU_LO:BD.ALU_LO+16]`
    same-step. At B15 those residuals carry the L8/L9 ALU values
    (lookup-mode legacy carrying through efficient mode's wrap chain),
    so SHL/SHR run on stale data; AND/OR/XOR and other non-shift ops
    still pass through `op_total = op_shl + op_shr` gating which
    correctly drops their contribution. But the side-effect of the
    composite's `GEToBDConverter` is a **`MARK_AX`-gated write to
    OUTPUT_LO/HI for `op_total > 0`** rows. Non-shift opcodes have
    `op_shl = op_shr = 0` so they don't get the write directly — but
    the composite's full forward also clears the GE workspace and
    re-zeroes intermediate residuals, which downstream blocks (L16
    `lev_routing`, L24 `tail_bit32_result_correction`, etc.) then
    see as 0 when reading the same dims that the L14-L24 chain was
    supposed to set.
  - Net effect: every test that needs an AX value at exit returns 0
    (the post-composite residual zero-state), matching the V1 v2
    failure-mode signature exactly.

This is a "composite runs too early in the block sequence" bug, not a
slot-conflict bug. A2 successfully eliminated the slot conflict but
introduced an ordering bug instead.

## Smoke regression

Baseline (memory-fix-v3 branch at `77e8d51c` before edits, full smoke
on a freshly-compiled `alu_mode='efficient'` VM): 26 pass / 25 fail
out of 51 (matches V2 baseline ±0).

After all edits, full smoke:
- 13 pass / 38 fail.
- Same breadth as V2: `TestSmokeBasic`, `TestSmokeComparison`,
  `TestSmokeAddress`, `TestSmokeMemory`, `TestSmokeShift`,
  `TestSmoke32Bit`, `TestSmokeIntegration`.

Per the brief's "smoke ≥ 46/52 (current baseline)" verification
threshold and the "REVERT + write findings if smoke regresses"
constraint, all edits reverted.

## Why A2 alone doesn't work (architectural, deeper than A2)

The L13 pin still pulls the 4 composite stages onto block 13, which
forces the install op (post_ops.append) to land at L13 — and the
post_op expansion (`_expand_wrapper_blocks` at `vm_step.py:2675-2691`)
inserts the composite as a passthrough block **immediately after L13**.
That's B15 in the new layout. The composite was authored under the
implicit assumption that it runs LATE in the pipeline (post-L24), where
the L14-L24 chain has already populated the AX-marker residuals.
Moving it earlier breaks every test that depends on those late
residuals.

This is the same "where do block ops actually land?" gap the V1 doc
called out — the right-sizing pass + post_op expansion + dep-graph topo
combine in non-obvious ways. A2 was the **slot-conflict** fix per V2's
diagnosis; what it doesn't address is the **layer-placement** issue.

Phase 8.A.4's choice to bind the install to `_layer13_attn_dep_anchor`
(commit history visible in the bake_fn comments) was deliberately to
let the install float to wherever the anchor lands — pre-Option A2 that
was L16 (or B25 after post_op expansion), and SHL/SHR tests passed
fine. Pinning the anchor + 4 stages to L13 forces the composite into
B15, which is *too early*.

### Why A2 + L13 pin coexistence is broken

The original V2 brief said: "If the L13 anchor lands at L13 AND the 4
composite stages land at L13, the bitwise wrap also lands at L13 (via
`layer10_carry_relay` topology), so we have a 3-op contention on
`block[13].ffn`. A2 resolves it." That diagnosis is correct in
isolation. But removing the contention by routing the shift composite
to `block[13].post_ops` doesn't put it back where it executed
successfully before (B25); it puts it at B15.

The baseline `ALUShiftComposite at B25` placement only happens because
the 4 stages WITHOUT `layer_idx=13` were floating to L17-L20 (per the
dep-graph topo + their phase=None default), and the install op floated
along with them. The B25 location is incidental — it's "wherever the
late chain happened to drop the install op" — but it works because the
composite genuinely needs to run late.

## Legitimate next steps (none are 1-3 file edits)

### Option B1: don't pin the L13 anchor — fix memory routing some other way

The original 5-memory-test failure (cluster #1 in
`SMOKE_FAILURES_2026_06_05.md`) was diagnosed as "the L13 anchor lands
at L16 instead of L13, so `layer13_mem_addr_gather` (the memory
address gather attn) runs too late for L14/L15 to read its outputs".
The fix doesn't have to be "pin the anchor to L13" — it could be:

  - Move `layer13_mem_addr_gather` out from under the L13 anchor and
    pin it directly with `kind="block"` + `layer_idx=N`.
  - Add a separate `_layer13_mem_addr_anchor` distinct from the shift
    composite's anchor (which can keep floating).
  - Restructure L13 to have two anchors: one for the address gather
    (pinned to L13), one for the shift composite + bitwise wrap
    (floating, lands at L16).

### Option B2: pin shift composite to land at a known-good late block

Use `layer_idx=N` on the install op (or its anchor) where N is the
baseline location's logical layer (which is hard to compute — the
baseline lands at B25 *after post-op expansion*, but the dep-graph
operates on pre-expansion layer indices, so there isn't a stable
"layer 25" pre-expansion).

Tractable variant: add `before="layer14_mem_generation"` or
`after="layer16_lev_routing"` constraints to force the composite into
the right band. Requires checking which late ops depend on shift's
OUTPUT_LO/HI vs which ops write the residuals shift's forward reads,
and inserting the constraints accordingly.

### Option B3: gate the composite's effect inside its forward

Add a "shift opcode is active" guard to the composite's `forward`
(`ALUShiftComposite.forward`) so that for non-SHL/SHR opcodes the
composite is a complete passthrough (no writes to OUTPUT_LO/HI, no
clears, no intermediate residual mutation). The current code already
has `op_total = op_shl + op_shr` gating on the OUTPUT write, but the
sub-stage `bd_to_ge` / `ge_to_bd` may still mutate the workspace in
ways that affect downstream reads (TBD — needs verification by
diff-ing residuals before/after with `op_total == 0` for a non-shift
opcode).

This is the cleanest fix in principle but requires careful audit of
every stage's forward to ensure idempotence under
`op_total == 0`. Estimated effort: 2-3 file edits + a runtime
diff-verification harness.

## Verification

- Compile: succeeds (30 blocks, d_model=800,
  `ALUShiftComposite at B15`).
- Forward pass: works (no shape crashes — A2's slot conflict is
  genuinely resolved).
- Smoke: 13/51 pass (38 fail; regressed from 26/51 baseline).
- Revert: clean (`git status` clean post-revert).

## Budget consumed

- 1 compile (succeeded with all edits applied).
- 1 smoke run (regression detected → revert).

## Recommendation

Option A2's slot-conflict fix is structurally correct but
**insufficient** — it must be combined with one of B1/B2/B3 to
preserve the composite's late-block execution position. The V2 doc's
"A2 is the cleanest fix" claim assumed the install would naturally
re-find a late landing slot after the slot conflict was resolved; the
post_op expansion mechanism in `_expand_wrapper_blocks` does NOT
preserve that — it expands the post_op into the block immediately
after the host, not into a late floating slot.

Single-rule fix attempts remain zero-sum on this cluster. The right
next move is to:

  1. First confirm the failure-mode is "composite runs too early"
     (B3 verification — diff residuals after a no-op composite call vs
     baseline). If yes, B3 is the cleanest fix.
  2. If B3 reveals the composite is not idempotent under `op_total ==
     0`, fall back to B2 (`before=` / `after=` placement constraints)
     or B1 (split anchors).

This is no longer a "land Option A2" task — it's a topology refactor.

## Files referenced

- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V2_20260605.md` — V2
  attempt (slot-conflict diagnosis + Option A2 spec).
- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_20260605.md` — V1 attempt
  (right-sizing diagnosis + Blocker 1/2 history).
- `c4_release/neural_vm/efficient_alu_neural.py:1486-1554` —
  `ALUShiftComposite.forward` (the same-step ALU_LO reader + OUTPUT
  writer that depends on AX-marker residuals being populated).
- `c4_release/neural_vm/vm_step.py:2590-2700` —
  `_expand_wrapper_blocks` (post_op → block-immediately-after-host
  expansion).
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:172-228` —
  `l13_alu_shift_install` (the install op).
