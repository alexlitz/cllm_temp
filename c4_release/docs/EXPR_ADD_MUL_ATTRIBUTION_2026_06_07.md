# `expr_add_mul_0` instruction-by-instruction attribution (2026-06-07)

Diagnostic-only doc for the Wave 1 Cluster C2 1096 corpus failure
`expr_add_mul_0: 22+24*22` (test_idx 0800). Companion to
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` and
`1096_ADD_HI_NIBBLE_PLUS_ONE_2026_06_07.md`. No fix attempted this
wave.

## TL;DR

`expr_add_mul_0` is a **downstream symptom** of the load-bearing
collapsed-step override at
`neural_vm/batched_pure_neural.py:2147-2192`. The override correctly
synthesizes `MUL(stack_val, ax_after_imm)` when an `IMM,<binop>` pair
collapses into a single neural step — but it sources `stack_val` from
`s.last_pushed_value`, which is **only refreshed when `exec_op ==
PSH`** (line 2017-2019). Programs with **multiple PSH/IMM pairs**
collapse the second `PSH;IMM` boundary too, so the second PSH never
gets its own dispatch and `last_pushed_value` stays stuck on the
first push's value. The MUL recovery then computes `MUL(first_push,
imm)` instead of `MUL(second_push, imm)`.

This is the same byte-emit-suppression cascade
(L27 `layer15_nibble_copy` → L28 `layer16_lev_routing` →
L34 `tail_bit32_result_correction`) described in
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`, but generalized to
multi-PSH programs where the shadow snapshot loses fidelity.

## Bytecode

`int main() { return 22 + 24 * 22; }` compiles to 8 instructions
(verified via `src/compiler.py`):

| idx | byte_pc | op   | arg |
|----:|--------:|------|----:|
|   0 |       0 | IMM  |  22 |
|   1 |       8 | PSH  |   — |
|   2 |      16 | IMM  |  24 |
|   3 |      24 | PSH  |   — |
|   4 |      32 | IMM  |  22 |
|   5 |      40 | MUL  |   — |
|   6 |      48 | ADD  |   — |
|   7 |      56 | EXIT |   — |

Declarative oracle: 8 steps, exit 550. Neural runner: **5 dispatches,
exit 506**. Diff = 44 = `(24-22)*22`.

## Step-by-step trace

Probe: `/tmp/probe_expr_add_mul_0.py` (monkey-patches
`BatchedPureNeuralRunner._dispatch_pure_neural` to snapshot
`last_pc / last_ax / stack0_shadow / last_pushed_value` pre/post each
dispatch).

| dispatch | exec_idx | exec_op | pre_PC | post_PC | post_AX |    post_pushed | notes                                              |
|---------:|---------:|---------|-------:|--------:|--------:|---------------:|----------------------------------------------------|
|        0 |        0 | IMM     |   None |      10 |      22 |           None | clean step                                          |
|        1 |        1 | PSH     |     10 |      18 |      22 |       **22**   | `last_pushed_value` set to first-PSH AX             |
|        2 |        2 | IMM     |     18 |      34 |      24 |       22       | PC jumps 18→34 (covers IMM+PSH idx 2-3); the 2nd PSH at idx 3 is **collapsed**; `last_pushed_value` NEVER refreshed |
|        3 |        4 | IMM     |     34 |      50 |     **484** | 22         | collapsed-step override fires: `MUL(last_pushed_value=22, ax_after_imm=22) = 484` ← **WRONG: should be `MUL(24, 22)=528`** |
|        4 |        6 | ADD     |     50 |      58 |     **506** | 22         | ADD(stack_top_carrying_22, 484) = 506 → exit       |

## First diverging step

**Dispatch 2 → 3 boundary** (the PSH at byte_idx 3 is the collapsed
opcode). The cross-instruction state corruption is in
`s.last_pushed_value`: the symbolic stack should have `[24, 22]`
(top-of-stack 24 after second PSH) at MUL time, but the runner's
shadow reads 22 (the first PSH's value) because the second PSH never
ran its `exec_op == PSH` branch at
`batched_pure_neural.py:2018-2019`.

The `tools/attribute_1096_failure.py` tool flags the first **token**
divergence at step 1 SP_byte3 with `expected_token=0x00,
neural_token=REG_BP` and suspect dims `OUTPUT_LO/HI, SP_CARRY_LO/HI`.
That is the same byte-emit-suppression-at-marker-slot symptom from
the collapsed-step doc (L34 `tail_bit32_result_correction` floor
crushes marker logits, REG_BP / REG_PC wins the tie-break) — so the
**model-emit** divergence starts at the very first PSH's STACK0
emission and continues across every subsequent PSH. The runner
masks the bad neural STACK0 via `last_pushed_value`, but that mask
only works for one PSH at a time.

## Suspected owner ops (downstream / symptom)

The op causing the heterogeneous `expr_add_mul_*` errors is **not** a
new declarative rule. It is the **`last_pushed_value` single-slot
shadow** at `batched_pure_neural.py:189` + the **PSH-only refresh
condition** at line 2018. The upstream root cause is the same as the
collapsed-step doc:

1. **L27** `layer15_nibble_copy` (`l15_ops.py:1720-1768`) — +40 spike
   at OUTPUT_LO/HI at the STACK0 marker
2. **L28** `layer16_lev_routing` /
   `lev_stack0_byte0_preserve_*` (`l16_ops.py:1646+`, `:267-300`) —
   ×97 self-amplification
3. **L34** `tail_bit32_result_correction` (`l10_ops.py:6958`) —
   ~-6.29e8 marker-logit suppressor

These produce byte-emit suppression at every STACK0 marker, forcing
**every** PSH in a `IMM,PSH,IMM,PSH,IMM,MUL,...` program to collapse
into its predecessor IMM step. The runner shadow happens to recover
the **first** collapsed `IMM,PSH,IMM,<binop>` pattern (smoke
`test_mul_overflow` etc.) but cannot recover a **chain** of them
because `last_pushed_value` is a scalar, not a stack mirror.

## Relationship to collapsed-step cascade

**Yes — downstream symptom.** Specifically:

- The single-PSH `IMM,PSH,IMM,<binop>,EXIT` shape is the
  collapsed-step doc's exemplar (smoke `test_sub_basic`,
  `test_mul_overflow`, etc.). The runner override at lines 2147-2192
  carries those tests.
- The double-PSH `IMM,PSH,IMM,PSH,IMM,MUL,ADD,EXIT` shape is the same
  cascade, applied twice. The override is structurally insufficient
  because `last_pushed_value` is a single slot.
- The 1096 `1096_ADD_HI_NIBBLE_PLUS_ONE_2026_06_07.md` doc's
  heterogeneous error catalog (`expr_add_mul_16` lo+1,
  `expr_add_mul_17` returns first operand, etc.) is consistent with
  this: the actual operand passed to the synth ALU depends on
  whether 1 or 2 PSH steps got collapsed and what value
  `last_pushed_value` happened to hold at the time. The sign and
  magnitude of the wrong byte vary because the wrong source byte
  varies.

The "real fix" is the same multi-rule structural cut documented in
`COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` (joint L28 OP_LEV gate +
L34 threshold raise on `stack0_pop_loaded_output_rules` /
`stack0_store_*` in `l10_ops.py:3884-3970`). With those landed, the
PSH dispatches would no longer collapse and the
`last_pushed_value` single-slot shadow would be irrelevant.

An **interim runner-side mitigation** (not attempted this wave, but
documented for record): widen `s.last_pushed_value` to a small ring
buffer indexed by `exec_idx` of the most recently seen PSH; have the
collapsed-step override at line 2168 scan backwards from `exec_idx -
1` for the nearest PSH and use its snapshot. Risk: any PSH whose
`exec_op` dispatch is skipped will still miss the snapshot, so the
ring buffer needs feeding from the **scheduled** bytecode walk, not
the actual `exec_op` stream. Effectively a serial shadow.

## Files reviewed (no edits this wave)

- `c4_release/neural_vm/batched_pure_neural.py:189` — single-slot
  `last_pushed_value`.
- `c4_release/neural_vm/batched_pure_neural.py:2013-2019` — PSH-only
  refresh of the shadow.
- `c4_release/neural_vm/batched_pure_neural.py:2147-2192` — the
  collapsed-step `_BINARY_POP_OPS` synth override.
- `c4_release/neural_vm/run_vm.py:123-141` — `_BINARY_POP_OPS`
  definition (includes MUL, ADD).
- `c4_release/src/compiler.py` — used to disassemble the test.
- `c4_release/tools/attribute_1096_failure.py` — first-token
  divergence + suspect-dim attribution (flagged SP_byte3 / REG_BP,
  consistent with marker suppression).
- `c4_release/.agent-logs/1096_fail_expr_add_mul_0.md` — auto-emitted
  brief from the attribution run.

## Probe artifact

`/tmp/probe_expr_add_mul_0.py` — monkey-patches the runner dispatch
loop to capture pre/post PC/AX/STACK0/pushed-value for each
opcode, then prints a step-by-step trace. Reproducible via
`python /tmp/probe_expr_add_mul_0.py` with the worktree's `c4_release`
on `sys.path`. Output included verbatim in this doc (the 5-row trace
above).

## Smoke / regression status

- No code edits this wave (diagnostic-only deliverable).
- Smoke baseline expected to remain at the pre-wave level (≥45). See
  `c4_release/docs/COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md` for the
  current 45/51 plateau and the override-removal status table.

## Confidence

- **High** that the cross-instruction state corruption is in
  `last_pushed_value` going stale across multiple PSH-collapse
  events.
- **High** that the upstream root cause is the same L27→L28→L34
  cascade as the single-PSH collapsed-step.
- **High** that a clean fix requires the multi-rule structural cut
  documented in `COLLAPSED_STEP_REAL_SURFACE_2026_06_07.md`, not a
  new declarative op or a single-slot runner tweak.
- **Medium** on the ring-buffer interim mitigation; needs
  per-`exec_idx` feeding from the bytecode walk, not the
  `exec_op` dispatch stream.
