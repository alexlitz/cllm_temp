# Collapsed-step override — real surface attribution (2026-06-07)

Attribution doc for the `f3342968` collapsed-step binary-ALU synth
override in `c4_release/neural_vm/batched_pure_neural.py:2147-2187`.
Companion to `IMM_OVERRIDE_REAL_SURFACE_2026_06_07.md` and
`32BIT_CASCADE_REAL_SURFACE_2026_06_07.md`.

## TL;DR

The `f3342968` override is **load-bearing for 9 binary-pop opcodes
under the `IMM N; PSH; IMM M; <binop>; EXIT` shape**:
MUL/DIV/MOD/SHL/SHR plus the Shape-A CMP family (EQ/GT/GE true).
BZ/BNZ no longer route through it (a sibling control-flow override now
handles those — see `44709e13`). The real surface is a **3-layer
amplifier cascade L27 → L28 → L34** at the STACK0 marker of step 2 of
any `IMM,PSH,IMM,<binop>,EXIT` program. The L28 amplifier was the
target of commit `1ed6b5b6`'s OP_LEV-gate fix attempt
(`l16_lev_stack0_byte0_preserve_*`) which **was reverted in `4f627491`
on 2026-06-07** because it did not actually let `f3342968` be removed
in smoke (the override removal regressed binop tests despite the gate
change). Override stays in place pending a multi-rule structural fix.

## Override surface (what the code does today)

`c4_release/neural_vm/batched_pure_neural.py:2147-2187`:

- Trigger: `exec_op == Opcode.IMM` and `s.last_pc // INSTR_WIDTH ==
  exec_idx + 2` (PC advanced by two instructions in a single step).
- Detects the skipped opcode at `exec_idx + 1`. If it is in
  `_BINARY_POP_OPS` (`MUL, DIV, MOD, SHL, SHR, EQ, NE, LT, GT, LE, GE`
  per `run_vm.py:123-141`), synthesizes the AX via
  `_compute_alu_legacy(skipped_op, last_pushed_value, ax_after_imm)`.
- `last_pushed_value` is a runner shadow snapshot taken pre-PSH (clean
  32-bit, unlike the noisy neural STACK0 emit) — see
  `batched_pure_neural.py:2013-2017`.

## Opcodes that still need it (current weights, 2026-06-07)

Per `OVERRIDE_REMOVAL_STATUS_2026_06_06.md` and re-confirmed against
HEAD `775cb19a`:

| Test | Opcode | Why it needs the override |
|---|---|---|
| `test_sub_basic` | SUB | Collapsed step; neural AX=0 |
| `test_div_basic` | DIV | Collapsed step; neural AX=0 |
| `test_mod_basic` | MOD | Collapsed step; neural AX=0 |
| `test_eq_true` | EQ (Shape A) | Collapsed step; neural AX=0 |
| `test_gt_true` | GT | Collapsed step; neural AX=0 |
| `test_ge_true` | GE | Collapsed step; neural AX=0 |
| `test_shl` | SHL | Collapsed step; neural AX=0 |
| `test_shr` | SHR | Collapsed step; neural AX=0 |
| `test_mul_overflow` | MUL | Collapsed step; neural AX=0 |
| (overlap) `test_xor_basic`, `test_add_16bit`, `test_add_carry_cascade` | XOR/ADD | Need both overrides 1+2 |

12 tests total; 9 unique to override 2.

BZ/BNZ are **no longer in this set** — `44709e13` "hoist BZ/BNZ
override above pure_neural early-return" gives those a dedicated
control-flow path that runs before any ALU recovery, so the
`_BINARY_POP_OPS` membership is now a historical detail for those two
opcodes (they never trigger `post_idx == exec_idx + 2` with binop
membership anymore).

## Missing model write the override compensates

The IMM-M step (second IMM in `IMM,PSH,IMM,<binop>,EXIT`) emits its
own well-formed Block A (`REG_PC=<binop_idx>`, `REG_AX=M`, etc.) up
through the `STACK0` marker, then — **instead of emitting STACK0
byte 0** — emits a second `REG_PC` marker and starts a Block B with
`REG_PC=<EXIT_idx>` and `REG_AX=0`. The `<binop>` step is **never
emitted by the model** at all. The runner's `_extract_register` scans
backwards for the last REG_PC marker and reads Block B's `REG_PC`, so
`s.last_pc` advances by 2 instructions in one model step.

The model's missing write is therefore: at step 2, position
**STACK0_byte0**, the model should emit the low byte of the pushed
operand (e.g. `0x32 = 50`) but instead emits the `REG_PC` marker
token. This is **byte-emit suppression at the STACK0 marker**, not a
PC arithmetic bug, per the byte-level probe in
`REMOVAL_2_DEEP_DIVE_2026_06_06.md`.

## Cascade attribution (L27 → L28 → L34)

Probed via `/tmp/probe_block_chain.py` (preserved in
`REMOVAL_2_DEEP_DIVE_2026_06_06.md`). Per-block residual at the
STACK0 marker position (token 132) over the 36 transformer blocks:

1. **L27** (`layer15_nibble_copy`, `l15_ops.py:1746-1768`): writes a
   `+40` spike to `OUTPUT_LO+2` / `OUTPUT_HI+3` at the STACK0 marker
   (encoding byte `0x32=50`, the value PSH'd one step ago). Semantic
   role: copy MEM nibbles into OUTPUT for store/load relays. **Root
   cause writer, but largest blast radius.**
2. **L28** (`layer16_lev_routing`, `l16_ops.py:285-300`): the
   `lev_stack0_byte0_preserve_{lo,hi}_{k}` family with
   `gate=OUTPUT_LO+k` / `gate=OUTPUT_HI_THIS_STEP+k` and
   `write_scale=50/S=0.5` self-amplifies the +40 spike to **+3,868**
   (×97 multiplier via gate self-feedback). **Proximate amplifier.**
3. **L34** (`tail_bit32_result_correction`, `l10_ops.py:3884-4150`):
   the `stack0_pop_loaded_output_rules` family has conditions like
   `("OUTPUT_LO+lo", 0.1)` + `("OUTPUT_HI_THIS_STEP+hi", 0.1)` with
   `gate=gate_mark_stack0`, threshold 10.5, `byte_writes(value,
   strength=500.0)`. With OUTPUT_LO+2 = 3,868 carried in, the
   0.1-weight condition contributes +387 alone, crossing threshold on
   hundreds of rules simultaneously. Sum: ~-6.29e8 dominant
   suppressor at every byte-logit cell. **Final crusher.**

Resulting marker-token logits at pos 132 sit at ~-10 (uniform floor);
the argmax tie-break selects token 257 = `REG_PC` because no marker
dim (`NEXT_PC`, `NEXT_STACK0`, ...) was set positive at this position.

## What was tried (and reverted)

| Date | Commit | Change | Result |
|---|---|---|---|
| 2026-06-07 04:15 | `1ed6b5b6` | `l16_lev_stack0_byte0_preserve_*`: OP_LEV weight 1.0→2.0, threshold 4.5→5.5 + remove `_BINARY_POP_OPS` block | Smoke unchanged at compile-time; runtime regressed when override deleted |
| 2026-06-07 13:37 | `4f627491` | Revert of `1ed6b5b6` | Restored baseline 45/51 |
| (prior session) | `cd53c076` | Revert of an earlier L16 `stack0_e8/f8` OP_LEV positive predicate | Same surface; same revert |
| (prior session) | `b91e3cd2` | KV-cache repair after L16 stack0 OP_LEV positive predicate | Reverted upstream |

Three independent attempts to tighten the L28 OP_LEV gate (the
proximate amplifier) have all been reverted. The gate strengthening
verifies under `compare_symbolic_to_lowered_ffn` (rule does not fire
at IMM step) but **does not produce a smoke-positive removal of the
override**. The amplifier chain has alternate paths:

- L27's +40 spike at OUTPUT_LO/HI does not require L28's amplifier to
  reach +387 at L34's 0.1-weight condition gate. The 1ed6b5b6 fix
  killed L28's ×97 but L27's +40 alone × 0.1 = +4 per condition,
  summed across the rule family, may still exceed L34's threshold.
- Without measurement of L34 residual after the L28 cut, the
  reverted attempt could not confirm where it broke.

## Recommended next step (not attempted this session)

Per `feedback_single_rule_fixes_are_zero_sum.md` (0/5 historical agent
attempts net positive on single-rule fixes), the proper fix is
**multi-rule structural**:

1. Re-attempt L28 OP_LEV gate strengthening, but pair with
   **L34 threshold raise** (10.5 → ~400 + 10.5) on
   `stack0_pop_loaded_output_rules` and `stack0_store_*` families in
   `l10_ops.py:3884-3970`. Cap the 0.1-weight match contribution OR
   raise the threshold so amplified OUTPUT_LO/HI residuals don't
   over-fire.
2. **Probe before commit**: run `/tmp/probe_block_chain.py` (or
   re-create equivalent) post-fix to confirm the per-block residual at
   token 132 stays below ±5e3 across all 36 blocks during the IMM-M
   step of `IMM 50; PSH; IMM 8; SUB; EXIT`.
3. Only then remove the `f3342968` override block (lines 2147-2187).

Estimated complexity: 5-10 sessions per `OVERRIDE_REMOVAL_STATUS_
2026_06_06.md` priority order.

## Attribution conclusion

- **Real surface**: byte-emit suppression at STACK0 marker (token 132)
  of step 2 in `IMM,PSH,IMM,<binop>,EXIT` programs.
- **Dominant writer**: L34 `tail_bit32_result_correction` /
  `stack0_pop_loaded_output_rules` (suppressor, -6.29e8).
- **Proximate amplifier**: L28 `layer16_lev_routing`
  `lev_stack0_byte0_preserve_*` (×97 via gate self-feedback).
- **Initial spike writer**: L27 `layer15_nibble_copy`.
- **Fix scope**: multi-rule (L28 + L34 jointly, not L28 alone).
- **Single-rule fix attempts**: 3 (1ed6b5b6, cd53c076, b91e3cd2 — all
  reverted).
- **Override status**: load-bearing for 9 unique tests; stays in
  place. The brief estimate of 5-10 sessions is accurate.

## Cross-references

- `c4_release/neural_vm/batched_pure_neural.py:2147-2187` — collapsed-
  step override.
- `c4_release/neural_vm/batched_pure_neural.py:2013-2017` — the
  `last_pushed_value` shadow.
- `c4_release/neural_vm/unified_compiler/ops/l16_ops.py:267-300` — L28
  amplifier (`lev_stack0_byte0_preserve_*`).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:3884-3970` —
  L34 suppressor (`stack0_pop_loaded_output_rules`).
- `c4_release/neural_vm/unified_compiler/ops/l15_ops.py:1746-1768` —
  L27 initial spike writer (`layer15_nibble_copy`).
- `c4_release/docs/REMOVAL_2_DEEP_DIVE_2026_06_06.md` — block-by-block
  byte-level probe + cascade identification.
- `c4_release/docs/REMOVAL_2_FINDINGS_2026_06_05.md` — original
  Removal-2 framing (refuted by deep dive).
- `c4_release/docs/OVERRIDE_REMOVAL_STATUS_2026_06_06.md` — current
  smoke-impact table.
- Reverted attempts: `1ed6b5b6` (reverted by `4f627491`), `cd53c076`,
  `b91e3cd2`.

## Confidence

- **High** that byte-emit suppression at STACK0 marker is the real
  surface (direct byte-level probe).
- **High** that L34 `tail_bit32_result_correction` is the dominant
  suppressor (per-block residual range).
- **High** that L28 `lev_stack0_byte0_preserve_*` is the proximate
  amplifier (gate self-feedback identified).
- **High** that single-rule L28-only fix is insufficient (3 reverted
  attempts).
- **Medium** that L34 threshold raise + L28 OP_LEV gate joint fix
  would succeed without further regressions; needs post-fix probe.
