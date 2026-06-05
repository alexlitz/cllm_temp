# Smoke memory cluster fix attempt V4 — 2026-06-05

Followup to `docs/SMOKE_MEMORY_FIX_ATTEMPT_V3_20260605.md`. Applied
**Option B2** from the V3 recommendation: keep V3's A2
`block.post_ops.append` install pattern, drop the `layer_idx=13` pin
on the 4 shift composite stages, and instead route their dep-graph
landing slot to a late op via `requires={"after": "layer16_lev_routing"}`
on each stage. Additionally re-bound the install op's `target_op_name`
from `_layer13_attn_dep_anchor` (L13) to `layer16_lev_routing` (late)
so the post_op expansion lands at a late physical block instead of
right after L13.

**Result: 38/51 smoke failures (was 26/51 baseline). Identical
regression to V2 and V3. Reverted.** B2 alone does not fix the cluster;
the failure mode is identical to V3 ("composite runs too early /
non-idempotent under op_total=0" pattern).

## What was tried

Worktree `/tmp/c4-memory-fix-v4`, branch `memory-fix-v4`.

Edits (all reverted):

  1. `neural_vm/unified_compiler/ops/l13_ops.py`
     `make_layer13_attn_dep_anchor_op`: added `layer_idx=13` pin
     (mirrors V2/V3) so `layer13_mem_addr_gather` (target_op_name=
     `_layer13_attn_dep_anchor`) lands at the physical L13 attn block
     instead of drifting to L16. Memory-cluster fix goal.

  2. `neural_vm/unified_compiler/ops/alu_ops.py`
     `make_alu_shift_composite_ops`:
       - `l13_alu_shift_bdtoge`: `ffn_units_used=1`,
         `requires={"after": "layer16_lev_routing"}`,
         `ALU_LO` → `ALU_LO.*.-1` SSA rename.
       - `l13_alu_shift_precompute`: `ffn_units_used=56`,
         `requires={"after": "layer16_lev_routing"}`, SSA rename.
       - `l13_alu_shift_select`: `ffn_units_used=1056`,
         `requires={"after": "layer16_lev_routing"}`, SSA rename.
       - `l13_alu_shift_getobd`: `ffn_units_used=64`,
         `requires={"after": "layer16_lev_routing"}`, SSA rename.
       - `make_install` (`l13_alu_shift_install`):
         * `block.ffn = builder.composite` →
           `block.post_ops.append(builder.composite)` (A2 from V3).
         * `target_op_name="_layer13_attn_dep_anchor"` →
           `target_op_name="layer16_lev_routing"` so the install op
           resolves to the late dep-graph layer (along with the 4
           stages' `after` constraint) and its post_op expansion lands
           at a late physical block.
       - All 5 ops' `produces` sentinel updated to
         `L13.post_ops[ALUShiftComposite]`.
  3. `neural_vm/unified_compiler/ops/alu_ops.py`
     `make_efficient_l10_andorxor_wrap_op`: added `block.attn.dim`
     fallback for `d_model` derivation (mirror of `efficient_l11_alumul_wrap`,
     defensive against ALUShiftComposite still being in `block.ffn` at
     bake time; turns out unnecessary with A2 + relocated install but
     left in for safety).
  4. `neural_vm/unified_compiler/full_vm_compiler_dynamic.py`
     `CROSS_STEP_DOCUMENTED_SAFE`: added the 4
     `(l13_alu_shift_*, ALU_LO.*.-1)` entries (declarative-only — the
     composite's runtime forward reads ALU_LO same-step).

## Topology verification (the structural objective was met)

Pre-expansion layer placement (`compile_full_vm_dynamic(alu_mode='efficient')`):

```
B0..B12: unchanged from baseline
B13: layer10_carry_relay (attn anchor) + _layer13_attn_dep_anchor (attn anchor)
B14: _layer11_ffn_dep_anchor + _layer14_attn_dep_anchor + layer14_mem_generation
B15: _layer12_ffn_dep_anchor + layer15_memory_lookup
B16: layer16_lev_routing
B17: l13_alu_shift_bdtoge       (post-lev_routing — late)
B18: l13_alu_shift_precompute
B19: l13_alu_shift_select
B20: l13_alu_shift_getobd
B21: l10_post_ops_combined
B22: post_l9_bz_bnz_pc_override
```

`layout.resolve_block_op_layer`:
- `layer13_mem_addr_gather`: target_layer=13 (memory fix WIN — bakes at
  the physical L13 attn block where L14-L15 expect to read ADDR_B*).
- `l13_alu_shift_install`: target_layer=16 (binds to layer16_lev_routing
  per the V4 edit).
- `efficient_l10_andorxor_wrap`: target_layer=13.

Post-expansion runtime block model: **34 blocks**, `ALUShiftComposite`
lands at **B26** (= layer 16's host expanded passthrough block). This
is the desired LATE landing slot. Per the V3 doc, B25 was the "known-
good" landing; B26 is one off and well past the L24 boundary the V3 doc
called out.

So the **structural objective was met**: composite is late, L13 anchor
is at L13, memory addr gather is at L13, no `block.ffn` slot conflict.

## What broke (the SAME failure pattern as V3)

Despite the correct structural placement, smoke regresses identically
to V3 — 38/51 fail (was 26/51 baseline), same breadth across every test
category that exercises AX-write residuals:

  - `TestSmokeBasic` (5 fail: add/sub/mul/div/mod basic)
  - `TestSmokeControlFlow` (3 fail: jmp_forward, bz_branch, bnz_branch)
  - `TestSmokeFunctionCall` (1 fail: simple_function)
  - `TestSmokeBitwise` (3 fail: or/and/xor basic)
  - `TestSmokeComparison` (6 fail: all the *_true tests)
  - `TestSmokeAddress` (2 fail: lea_basic, adj_sp)
  - `TestSmokeMemory` (5 fail: all SI/LI/SC/LC tests still fail)
  - `TestSmokeShift` (2 fail: shl, shr)
  - `TestSmoke32Bit` (10 fail: every 16-bit test)
  - `TestSmokeIntegration` (1 fail: cmp_and_branch)

Total: 13 pass / 38 fail. This is the same "AX-write residual chain
collapsed" mode V3 documented as Blocker 4.

## Why B2 didn't help (in spite of correct topology)

The V3 doc's Blocker 4 analysis was: "composite runs too early; its
sub-stages (BD→GE, precompute, select, GE→BD) clear the GE workspace
and re-zero intermediate residuals which downstream blocks then read
as 0". The diagnosis pinned this to the B15 composite landing.

**V4's B26 composite landing should have side-stepped this.** It
didn't, which suggests Blocker 4 is structural beyond just where the
composite lands. Possible deeper causes (all consistent with the
38-fail breadth):

  1. **L13 anchor pinning breaks something orthogonal.** With L13
     anchor at L13, `layer10_carry_relay_bake` (block op bound to
     `layer10_carry_relay`, lands at L13) and `layer13_mem_addr_gather`
     (block op bound to `_layer13_attn_dep_anchor`, also lands at L13)
     now both bake into `block[13].attn`. They contest head_0:
     `layer10_carry_relay_bake` writes CARRY relay on head 0 (V slots
     1, 2); `layer13_mem_addr_gather` writes ADDR gather on heads 0-2.
     Whichever runs last wins. In baseline this didn't happen because
     mem_addr_gather drifted to L16, away from L13's carry relay.
     Hypothesis: with carry relay overwritten, L8 ADD/SUB byte carry
     can't fire → AX writes get a garbage byte-1 → every AX-write
     test fails.
  2. **`efficient_l10_andorxor_wrap` co-located with the L13
     attn anchor.** Both `efficient_l10_andorxor_wrap` and the L13
     `mem_addr_gather` resolve to L13. The bitwise FFN replaces
     `block[13].ffn` (correct), but the attn at L13 is now doing
     mem-addr-gather instead of the L10 carry-relay duty the
     bitwise rule chain implicitly assumed.
  3. **`requires["after"]: layer16_lev_routing` pulls more than just
     the shift stages into the late band.** Anything in the
     transitive closure of "after l16_lev_routing" gets dragged with
     them. (The compile log shows 21 cross-step warnings vs ~12
     baseline; some are new edges from this constraint.)

## Smoke regression

Baseline (memory-fix-v4 branch at `01e63287` before edits, full smoke
on a freshly-compiled `alu_mode='efficient'` VM): 26 pass / 25 fail of
51 (matches V2/V3 baseline ±0).

After all V4 edits, full smoke:
- 13 pass / 38 fail / 1 deselected, 100.12s wall.
- Same breadth as V2/V3.

Per the brief's "smoke ≥ 50/52 (memory + others)" verification
threshold and the "REVERT + findings if smoke regresses" constraint,
all edits reverted.

## Why the V3 doc's B2 recommendation was insufficient

The V3 doc's B2 said: "Tractable variant: add `before="layer14_mem_generation"`
or `after="layer16_lev_routing"` constraints to force the composite into
the right band." V4 implemented exactly that: `after="layer16_lev_routing"`
on all 4 shift stages + re-binding the install's `target_op_name` to
the same late op. The composite landed at B26 (verified). The regression
still happened, identical to V3's B15-landing regression.

This means **moving the composite back to a late physical block does NOT
restore the baseline behaviour** — the failure mode is not "composite
running too early" but rather the broader topology mutation that comes
with pinning the L13 anchor.

Per the V3 doc's Option B3: "Add a 'shift opcode is active' guard to
the composite's `forward` so that for non-SHL/SHR opcodes the composite
is a complete passthrough (no writes, no clears, no intermediate
mutation)." V4 did not attempt B3; the brief explicitly said to try B2.
B3 remains the unfalsified path forward, plus the head_0 contest at
L13 attn needs an architectural decision (move carry_relay_bake to a
non-L13 layer, or split the L13 anchor in two).

## Composite landing (verified)

After all V4 edits, the runtime `model.blocks` post-`_expand_wrapper_blocks`:

```
B25.ffn: PureFFN (1400x800 attn)   ← the bitwise rule FFN host
B26.ffn: ALUShiftComposite         ← V4's intended late landing
B27.ffn: PureFFN
```

Composite at B26 (one block after V3's "known-good" B25 reference).
This is the correct band — late enough for the L14-L24 AX-marker
residuals to have populated ALU_LO/HI/AX_CARRY_LO/HI before the
composite reads them. Yet the AX-write tests still fail.

## Verification

- Compile: succeeds (34 runtime blocks post-expansion).
- Composite landing: B26 (late, per V4 design).
- `layer13_mem_addr_gather` landing: L13 (memory-cluster fix structural
  goal met).
- Forward pass: works (no shape crashes).
- Smoke: 13/51 pass (38 fail; regressed from 26/51 baseline).
- Revert: clean (`git status` clean post-revert).

## Budget consumed

- 1 compile (succeeded with all V4 edits applied, composite at B26).
- 1 smoke run (regression detected → revert).

## Recommendation

B2 (placement constraints to keep composite late) is structurally
correct but **insufficient** — the L13 anchor pin introduces an attn
head_0 contest between `layer10_carry_relay_bake` and
`layer13_mem_addr_gather` that breaks the AX-write residual chain end
to end. The next attempt must:

  1. **Decouple the L13 mem-addr-gather pin from the L10 carry-relay
     pin.** Possible approaches:
     - Move `layer10_carry_relay` to a non-L13 layer (a new
       `_layer10_attn_dep_anchor` pinned to L10).
     - Move `layer10_carry_relay_bake`'s head_0 carry write to a
       non-L13 layer.
     - Use a different head index for `layer13_mem_addr_gather` head_0
       so it doesn't contest head_0 with carry_relay_bake.
  2. **Audit the composite's idempotence under `op_total = 0`** (V3
     doc's B3) — even if B26 lands correctly, the composite's
     intermediate substages may still mutate residuals that downstream
     ops read.

Single-rule fix attempts remain zero-sum on this cluster. The memory
cluster is now a **3-attempt zero-sum streak** (V2/V3/V4 all regress
to 38 failures via different topology mutations). The next attempt
should be a **two-file refactor** (decouple L10/L13 anchors AND audit
composite idempotence) or be backed off entirely to the alternate
cluster (cascade-32bit) which the smoke audit identified as the same
size (5 tests) and independent.

## Files referenced

- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V3_20260605.md` — V3
  attempt (A2 + L13 pin → B15 landing, 38/51 failures).
- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_V2_20260605.md` — V2
  attempt (slot conflict diagnosis).
- `c4_release/docs/SMOKE_MEMORY_FIX_ATTEMPT_20260605.md` — V1 attempt
  (right-sizing diagnosis).
- `c4_release/docs/SMOKE_FAILURES_2026_06_05.md` — cluster A
  (memory) priority + recommendation.
- `c4_release/docs/SMOKE_MEMORY_TRACE_20260603.md` — original L13
  anchor drift diagnosis.
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:1902-1941` —
  `layer10_carry_relay` anchor + `_L10_HEAD_LAYOUT` (head_0 is
  carry relay).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:2063-2120` —
  `layer10_carry_relay_bake` (block op that contests B13.attn head_0
  with `layer13_mem_addr_gather` under the V4 pin).
- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:394-459` —
  `_layer13_attn_dep_anchor` (the L13 pin V2/V3/V4 added).
- `c4_release/neural_vm/unified_compiler/ops/l13_ops.py:462-545` —
  `layer13_mem_addr_gather` (block op heads 0-2 ADDR gather).
- `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:22-269` —
  shift composite stages + install (V4 edits live here).
- `c4_release/neural_vm/efficient_alu_neural.py:1486-1554` —
  `ALUShiftComposite.forward` (the same-step ALU_LO reader; idempotence
  audit needed per V3 doc's B3).
