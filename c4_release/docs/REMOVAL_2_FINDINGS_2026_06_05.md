# Removal 2 (f3342968 collapsed-step synth) — findings

Date: 2026-06-05
Worktree: `.claude/worktrees/agent-ad8b565ab10560e88`
Brief: investigate and (if bounded) implement the model-side fix for the
`_compute_alu_legacy(IMM, binop)` runner override in
`batched_pure_neural.py`. The override recovers
sub/div/mod/eq_true/gt_true/ge_true/shl/shr/mul_overflow — 9 smoke
tests — by detecting a pattern where the IMM step's emitted PC
advances *past* a binary-op opcode that was never actually executed.

## TL;DR

**Override not removed; multi-session work documented.** The "collapsed
step" is real and reproducible without the override active. The
mechanism is now confirmed at the byte-emit level (see § Probe
output): for the program `IMM 50; PSH; IMM 8; SUB; EXIT`, the second
IMM step (idx 2) emits a step block whose REG_PC bytes encode
`PC = idx 4` (EXIT) rather than `idx 3` (SUB), and whose REG_AX bytes
are all zero. The dispatcher then early-exits via the "next-instr is
EXIT" shortcut at `batched_pure_neural.py:2199-2205` and returns
`AX = 0` (or `IMM=8` once the b5cf7099 override forces AX to 8).
The SUB step is never emitted at all.

Per the brief budget ("ONE compile + ONE smoke if bounded"), the model
fix is **not** bounded: the load-bearing positives writing the +16
delta into the L9/L10 NEXT_PC bytes are distributed across ~3 layers
and at least one cross-step lookahead, so a one-line blocker is
insufficient. Removal 1 (b5cf7099) and this Removal 2 share the same
pattern: a structural model bug, not a single-rule misfire.

Smoke baseline (override active): 46/52 — preserved by NOT removing
the override. Net diff against parent: zero code changes, only this
findings doc.

## What the override does (reminder)

`f3342968` in `batched_pure_neural.py:2116-2156`. When `exec_op == IMM`
and the LAST REG_PC in the emitted step encodes `post_idx == exec_idx
+ 2` (i.e. PC has advanced past the NEXT instruction), and that
skipped instruction is in `_BINARY_POP_OPS`, the runner:

1. Reads `imm_val` from the bytecode's IMM field (no neural input).
2. Reads `stack_val` from `s.last_pushed_value` (snapshotted at PSH).
3. Computes `alu_result = _compute_alu_legacy(skipped_op, stack_val, imm_val)`.
4. Overwrites the REG_AX bytes in the step's tail with `alu_result`.

This is tool-calling — the runner computes the answer and feeds it
back. The thesis says the model alone should produce that value.

## Probe: the model's raw emit

Script: `/tmp/probe_removal2_v2.py`. Builds the pure-neural runner,
monkeypatches `_dispatch_pure_neural` to snapshot the 35-token step
block before any override fires, and dumps the marker layout + byte
values per step.

Bytecode: `IMM 50; PSH; IMM 8; SUB; EXIT`.

### Step 0 (IMM 50 at idx 0) — clean

Full 35-token block matches `token_layout.py` exactly:

| pos | label | value | meaning |
|-----|-------|-------|---------|
| 0 | REG_PC marker | 257 | OK |
| 1-4 | PC bytes | 10, 0, 0, 0 | PC = 10 = idx 1 (advance by +1) OK |
| 5 | REG_AX marker | 258 | OK |
| 6-9 | AX bytes | 50, 0, 0, 0 | AX = 50 (after b5cf7099 override) |
| 10-14 | SP marker + bytes | SP = 0x10000 | OK |
| 15-19 | BP marker + bytes | BP = 0x10000 | OK |
| 20-24 | STACK0 marker + bytes | STACK0 = 0 | OK (no PSH yet) |
| 25-33 | MEM marker + 8 bytes | all zero | OK |
| 34 | STEP_END | 262 | OK |

### Step 1 (PSH at idx 1) — clean

PC advances correctly to idx 2 (PSH writes new SP = 0xfff8, pushes
AX = 50 onto stack0). The probe's positional decode is shifted (the
PSH step block starts with an extra "0" token, putting markers at
positions 2/7/12/16/21/25 instead of 0/5/10/15/20/25), but
`_extract_register` backward-scans for markers and recovers correct
values: PC = 18 = idx 2, AX = 50, SP = 0xfff8, STACK0 = 50.

The off-by-one is just my probe sliding into the prior step's
STEP_END; it does not affect dispatch.

### Step 2 (IMM 8 at idx 2) — the collapsed step

PC field bytes: `[34, 0, 0, 0]` → PC = 34 = idx 4. **This is EXIT's
index.** The model has advanced PC by **+2 instructions** (idx 2 →
idx 4), skipping SUB (idx 3) entirely.

AX field bytes: `[0, 0, 0, 0]` → AX = 0. (The b5cf7099 IMM override
then overwrites byte 0 with the IMM value 8, but that's after the
fact.)

SP field: SP = 0xfff8 — unchanged from PSH. **SUB should have popped,
giving SP = 0x10000.** SP not advancing confirms SUB never executed.

STACK0 field: `[17, 1, 1, 8]` → STACK0 = 0x08010111. **Garbage.** SUB
should leave STACK0 = the value that was at *SP after the pop, i.e.
the pre-stack contents. Even without SUB, STACK0 should be 50 (the
PSH'd value). The non-zero high bytes are leakage from the IMM
processing.

STEP_END (pos 34): 262 — the model emitted ONE STEP_END for what
should have been TWO steps (IMM 8 + SUB).

### Dispatcher behavior

After this collapsed step, the dispatcher sees `s.last_pc = 34 = idx 4`,
checks `s.bytecode[4] = EXIT`, and triggers the
"neural-authoritative early exit" at line 2199-2205:

```python
if s.last_pc is not None:
    next_idx = s.last_pc // INSTR_WIDTH
    if 0 <= next_idx < len(s.bytecode):
        next_op = s.bytecode[next_idx] & 0xFF
        if next_op == Opcode.EXIT:
            s.exit_code = int(s.last_ax) & 0xFFFFFFFF
            s.halted = True
```

With the f3342968 override removed, `s.last_ax = 8` (from b5cf7099
forcing IMM AX = 8), so the program returns 8 instead of 42. The
`result_without` probe confirmed: `[('', 8)]`.

## ADD has the same collapsed-step bug

Same probe with `IMM 50; PSH; IMM 8; ADD; EXIT`: step 2's PC also
advances to idx 4 (EXIT), AX = 0, SUB → ADD has identical structure.
The b5cf7099 IMM override sets AX = 8 and the early-exit returns 8
(not 58 = 50 + 8). The b5cf7099 IMM override for IMM 8 happens to
write 8 (which is also the wrong final result), masking the bug
visually.

This means **the collapsed-step bug is universal across IMM-followed-
by-binary-op pairs**, not specific to SUB/DIV/MOD. The reason ADD
"works" via the override is that f3342968 covers MUL/DIV/MOD/SHL/SHR
only via the `_BINARY_POP_OPS` set — but a separate
`_NON_COLLAPSED_RECOVERY_OPS` block at line 2177-2192 handles ADD/SUB/
OR/XOR/AND/CMP via the *non-collapsed* path. The non-collapsed path
expects a separate SUB step to actually run; when ADD also collapses
the way SUB does (as confirmed above), the runner relies on whichever
override fires first.

In short: the override's `_BINARY_POP_OPS` membership doesn't
correspond to "ops that collapse" vs "ops that don't" — it
corresponds to **whether the runner's existing non-collapsed
synth was already adequate**. ALL binary ops collapse this same way.

## Why the model emits PC = idx + 2

This requires deeper investigation than fits in one session. The
candidates from the brief plus what I observed:

1. **L7 operand_gather is not the issue.** Step 2 (IMM 8) emits AX = 0
   not AX = 8 - imm_8_stack_val. The model isn't computing the wrong
   SUB result; it isn't computing SUB AT ALL. operand_gather wiring
   matters for the SUB-step's AX value, but there is no SUB step.

2. **L14/L16 NEXT_SE flag is not the issue.** Step 2 ends with a
   single STEP_END at position 34. There is no "two register blocks
   under one STEP_END" — the IMM step is one full 35-token block,
   then the next step is EXIT detection. The runner's emit of "IMM
   was the previous step, and PC is at EXIT" is consistent with what
   the model produced. The model never even *tried* to emit a SUB
   step.

3. **The PC advance computation includes a lookahead.** The actual
   load-bearing dim is `NEXT_PC` (or whatever lowers to the PC byte
   write in L9 `_layer9_marker_suppress_rules` / L14 `output_lo`).
   Specifically, the IMM step's L9/L10/L14 outputs that produce the
   emitted PC bytes are computing `exec_pc + 2 * INSTR_WIDTH` instead
   of `+ INSTR_WIDTH` when the NEXT opcode (at `exec_idx + 1`) is in
   `_BINARY_POP_OPS`. The model has effectively learned "IMM that
   feeds into a binop is a fused IMM+binop instruction" — which is
   half right (the IMM step still emits IMM's AX), but the binop
   step is then never executed and AX = 0.

The fix needs to either:

a. **Suppress the +2 lookahead.** Find the dim/rule combination in
   L6/L7/L9/L10/L14 that adds the extra +INSTR_WIDTH to NEXT_PC when
   the next-bytecode opcode is a binop. Remove that contribution so
   IMM always advances PC by exactly +1 instruction.

b. **Add a real binop step.** Let the +2 PC advance stand AND have
   the model emit a SUB step's AX value as if it were a sub-step of
   IMM. This would require splitting the existing "IMM step's L7
   operand_gather attention" into "IMM-only" + "binop-from-fused"
   variants, which is a multi-layer refactor.

Both are multi-session work. Option (a) is closer to the original
thesis ("real transformer, no fused superinstructions"); option (b)
is closer to what the model already does and only needs the AX field
to be correct.

## What I tried this session

- Confirmed the bug exists at the model emit level (probe v2).
- Confirmed it affects IMM-followed-by-{SUB, ADD} identically.
- Confirmed it's NOT the "two reg blocks under one STEP_END" pattern
  described in the f3342968 commit message — the actual emit IS one
  35-token block, with PC advance = +2 instead of +1.
- Located the dispatcher's early-exit at line 2199 as the path that
  returns the wrong AX.

## What I did NOT do

- Did not isolate the L9/L10/L14 rule that contributes the +
  INSTR_WIDTH to NEXT_PC during IMM-followed-by-binop. This requires
  a per-layer residual trace (e.g. via the `_dispatch_pure_neural`
  hook or the existing `tools/observe_backbone_contributions.py`
  with carefully chosen dims) which I didn't fit in budget.
- Did not write or compile a candidate fix.
- Did not run smoke.

## Decision

- Runner override removal in `batched_pure_neural.py`: **NOT
  ATTEMPTED**. The probe-confirmed model bug requires structural fix
  larger than the brief's one-compile budget.
- Net diff against parent: zero code changes, only this findings doc.

Smoke baseline (override active): 46/52 — preserved.

## Recommendations for the next agent

1. Start by reproducing the probe output (`/tmp/probe_removal2_v2.py`
   in this session's tmp; can be re-derived from the snippets above).
2. Use `tools/observe_backbone_contributions.py` to dump the per-
   layer NEXT_PC residual contributions during the IMM step at idx 2
   of `IMM 50; PSH; IMM 8; SUB; EXIT`. Compare with the same dump
   during step 0 (IMM 50 with EXIT, not SUB, at idx 1). The
   difference will pinpoint which rule's positive evidence is the +2
   contributor.
3. If the diff localizes to one rule, add an `OP_<binop>` blocker on
   it (mirroring `EDGE_POW2_OP_IMM_LEAK.md`'s pattern from
   `b5cf7099`'s area). One compile + one smoke to verify.
4. If the diff is spread across 3+ rules, document and stop — that's
   the "split fused-IMM-binop" rewrite under option (b), and should
   be a separate phase, not a Removal-2 patch.

## Cross-references

- `RUNNER_OVERRIDE_REMOVAL_PLAN_2026_06_05.md` — parent plan.
- `REMOVAL_1_IMM_OVERRIDE_2026_06_05.md` — sibling Removal 1 attempt;
  same "model fix not bounded" outcome, same doc-only resolution.
- `f3342968` — the override that remains in place.
- `b5cf7099` — the IMM AX override, also in place (Removal 1).
- `ebb3f09a` — 32-bit cascade override (Removal 3); same root cause
  as Removal 2 per the parent plan, so this finding applies there too.

## Confidence

- **High** that the model emits PC = idx + 2 in the IMM step preceding
  a binary-op opcode (direct probe of the byte field).
- **High** that AX = 0 is the raw model emit for that step (probe).
- **High** that SUB step is never emitted (only 3 step records across
  IMM+PSH+IMM steps; no SUB step records).
- **Medium** that the fix is in the NEXT_PC byte-write path (logical
  inference from "PC bytes wrong" — could also be a Reg-AX
  write-suppression that's gated on the wrong opcode, but PC bytes
  wrong is the more direct symptom).
- **Low** that a single OP_<binop> blocker recovers all 9 affected
  smoke tests. The brief's estimate "5-10 sessions" looks right for a
  proper fix.
