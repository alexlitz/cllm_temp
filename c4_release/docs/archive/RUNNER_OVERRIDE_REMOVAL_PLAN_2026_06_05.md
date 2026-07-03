# Runner override removal plan — 2026-06-05

## The problem

Per `BLOG_SPEC.md`:

> I want it to be as much as possible a 100% pure transformer with no
> exotic architecture choices hacks only being clever not stretching the
> definitions

Five smoke gains this session were **runner overrides** — `batched_pure_neural.py` reads the bytecode, computes the answer in Python, and overwrites the model's emitted AX. That's tool-calling. It violates the thesis.

Counted as smoke wins but actually anti-progress:

| Commit | Runner override | What it patches over |
|---|---|---|
| `b6632137` | `_decode_bail_exit_code` forward-scan | Bail-path AX recovery (probably legit error recovery, not a bug-mask) |
| `f3342968` | Collapsed-step synth via `_compute_alu_legacy(IMM, binop)` | Model emits 2 register blocks under 1 STEP_END for binary ops; post-binop AX is wrong |
| `b5cf7099` | IMM override from bytecode imm field | L10/L16 tail rules write 0xE0/0xE8 instead of actual IMM byte |
| `ebb3f09a` | 32-bit cascade extension | Same root cause as f3342968 for multibyte ops |
| `69f77682` | CMP non-collapsed synth | Model's CMP head emits constant (per stack0 staleness chain) |

(Not in this list: `61a7e12c` LT polarity — that's a real model fix in `block.attn.W_gate`. Keep it.)

These overrides recovered 18 smoke tests. Removing them drops smoke back to 28/52. **That regression is OK** — it surfaces real model bugs that need real model fixes.

## What the model actually has to do

Per the blog: every opcode is implemented by hand-authored weights. The runner just drives autoregressive token emission. So:

- **For IMM N**: model should emit `REG_AX, byte0(N), byte1(N), byte2(N), byte3(N), STEP_END`. The runner is then done. Nothing else to compute.
- **For `IMM A, PSH, IMM B, SUB, EXIT`**: model emits 5 step blocks. Each step has its own STEP_END. After the SUB step, AX = A - B. The runner reads that AX. Nothing to synthesize.
- **For CMP**: model emits the comparison result in AX based on STACK0_BYTE0 and AX. Done.

If any of those don't work, **the bug is in the model weights**, not the runner.

## Removal plan

Six removals. Each starts with reverting the override and observing the failure mode, then identifying and implementing the model fix.

### Removal 1 — `b5cf7099` IMM 0xE0-0xFF override (smallest, clearest)

**The bug**: per `L5_BYTEDECODE_FIX_ATTEMPT_2026_06_05.md` (commit `d0f9d80b`), L10's `tail_lea_local_ax_marker_byte0_e8` (l10_ops.py:6400) writes 0xE8 at strength 1M and L16's `l16_psh_mem_addr0_e0_from_sp_no_addr_src` (l16_ops.py:768) writes 0xE0 at strength 200k. These fire spuriously on IMM dispatch rows when the operand has high bit set.

**Model fix**: strengthen the `OP_IMM` blocker on those tail rules from `-1e6` (or whatever it is) to `-1e9`. Mirror the `EDGE_POW2_OP_IMM_LEAK.md` fix pattern that was applied to a sibling rule.

**Expected smoke impact**: removes `b5cf7099` (-3 smoke: test_xor_basic, test_add_16bit, test_add_carry_cascade). The model fix recovers them (+3). Net 0 smoke. **Eliminates one runner override.**

### Removal 2 — `f3342968` collapsed-step synth (biggest)

**The bug**: per the agent that landed this, the model emits two register blocks under ONE STEP_END for SUB/DIV/MOD/SHL/SHR/MUL/etc. The post-binop AX is 0. The model's neural ALU for these ops emits a token sequence with broken inner step.

**Model fix candidates** (need investigation):
- L7's operand_gather attention isn't seeing the correct STACK0_BYTE0 by the time the binop AX is emitted — same as the EQ(17) chain
- OR the model's binary-op dispatch path consolidates two register blocks because it thinks the binop is a sub-step of the prior IMM
- OR the L14/L16 NEXT_SE flag isn't asserted between the IMM emit and the binop emit, causing them to share a STEP_END

**Expected smoke impact**: removes -9 smoke. The model fix needs all 5 binary ops + 3 CMP true ops + 1 mul_overflow to recover. This is the deepest model issue — likely 5-10 sessions to crack.

### Removal 3 — `ebb3f09a` 32-bit cascade

**Same root cause as #2**, just for multibyte. -3 smoke when reverted. If #2 lands, this comes for free.

### Removal 4 — `69f77682` CMP non-collapsed synth

**The bug**: per `CMP_DECODE_REWIRE_FINDINGS.md` (commit `f41923e3`), the model's L9/L10 CMP rules are correctly wired. The bug is **STACK0_BYTE0 staleness** in L7's operand_gather attention — for IMMs with both nibbles nonzero (17 = 0x11), the K-side STACK0_BYTE0 gate yields ALU=0.

**Model fix**: fix the L7 head 0 K-side gate per `CMP_POLARITY_INVESTIGATION_2026_06_03.md` §"Recommended fix sequence".

**Expected smoke impact**: -2 smoke (test_eq_false, test_ne_true). Likely recovers 4 of 6 CMP cluster tests when fixed (per the polarity doc).

### Removal 5 — `b6632137` bail-decode

Actually mostly legit. The forward-scan recovers AX on cap-hit elements which is a runner robustness feature, not a model bug mask. **Keep this one for now**; can revisit after #1-4 land.

### Removal 6 — `61a7e12c` LT polarity W_gate

Real model fix in `block.attn.W_gate`. **Keep.**

## Order to execute

Removal 1 first (smallest, narrow scope) to validate the approach + commit pattern. Then Removal 4 (stack0 staleness is the most common architecture-level bug source). Then Removal 2 (biggest, needs the most investigation). Removal 3 follows automatically once 2 lands.

## Process per removal

For each override:

1. Create a branch from current HEAD
2. Revert the override commit on that branch
3. Measure smoke regression (which tests broke?)
4. For each broken test:
   - Run it under hooks to capture the model's emitted token stream
   - Identify which step / which dim / which weight is producing the wrong value
   - Patch the model (a rule change, a missing blocker, a layer pin)
5. Re-measure smoke. Must be ≥ baseline pre-removal.
6. If smoke recovers via real model fix, land on main.
7. If smoke can't recover this session, document the findings + leave the override in place + commit only the new doc.

## Acceptance criteria

After Removals 1-4:
- `batched_pure_neural.py` has zero `_compute_alu_legacy` calls in the dispatch path
- Zero `_override_register_in_last_step` calls for opcode-specific computation
- The `_BINARY_POP_OPS`, `_NON_COLLAPSED_RECOVERY_OPS`, `_CMP_OPS` lists are deleted from the runner
- All AX values flow naturally through model.forward(x)
- The blog thesis is intact

## Honest accounting of cost

Smoke this session: 25 → 46. After removing the 4 runner overrides: 29 → ??/52 depending on which model fixes land.

Realistic outcome with this plan:
- Removal 1 (model fix tractable): 46 → 46 (no change after model fix)
- Removal 4 (stack0 staleness): 46 → 44 if model fix can't land; 46 → 46+ if it can
- Removal 2/3 (collapsed-step): 46 → 37 if model fix can't land in 1 session

Worst case if model fixes are all multi-session: 46/52 drops to ~32/52. But the **thesis is restored** — the model is a real transformer, not a transformer plus tool calls.

## Recommendation

This is the right thing to do. The session's smoke gains were measured but anti-thesis. Removing them costs visible progress but restores correctness.

Start with Removal 1 immediately — it's bounded and the fix is documented. Spawn an investigation agent on Removal 4 (stack0 staleness) since it's already heavily traced.
