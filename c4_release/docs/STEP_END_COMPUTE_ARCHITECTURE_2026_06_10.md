# STEP_END Compute Architecture (2026-06-10)

Architectural design note: where ALU / CMP / branch / dispatch compute
should fire inside the 35-token VM-step window.

Status: design + measurement. No rule migrations have landed yet.
The empirical probe (`c4_release/tools/probe_step_end_completeness.py`)
falsified the naive form of the principle; the doc records both the
principle and the prerequisite work that must land first.

## 1. Principle (user directive)

> "I think basically all the stuff that is not related to writing out
> the actual tokens should be happening in the one step at the end of
> the 35 token vm step."

Each VM op emits a 35-token block (`Token.STEP_TOKENS = 35`,
`token_layout.py`):

```
[ 0]      REG_PC marker
[ 1- 4]   PC bytes
[ 5]      REG_AX marker      <- MARK_AX
[ 6- 9]   AX bytes
[10]      REG_SP marker
[11-14]   SP bytes
[15]      REG_BP marker
[16-19]   BP bytes
[20]      STACK0 marker
[21-24]   STACK0 bytes
[25]      MEM marker
[26-29]   MEM addr bytes
[30-33]   MEM val  bytes
[34]      STEP_END (or HALT) <- MARK_SE
```

The principle says: byte emission (rows 1-4, 6-9, ...) belongs at byte
positions; everything else — opcode dispatch, ALU result computation,
CMP combine, branch decisions — belongs at the **STEP_END row** so it
fires once per step with the full state visible.

The current convention is the opposite: dispatch and ALU/CMP combine
rules in L8/L9/L10 gate on `MARK_AX + OP_<NAME>` (84 / 29 / 150 raw
`MARK_AX` references in `l8_ops.py` / `l9_ops.py` / `l10_ops.py`
respectively). MARK_AX is row 5 — only six tokens into a 35-token
window, with the rest of the register state still being relayed.

## 2. Why this matters

Several open bugs trace to MARK_AX racing inter-cycle relays. The L10
`cmp_combine` override needed `MARK_PC` blockers and CMP+0 -0.1 anti-
amplification because the residual at MARK_AX was not yet stable
(see the inline comment at `l10_ops.py:720` and the post-mortem in
`docs/IF_EQ_CMP_DEFAULT_LEAK.md`-style notes). The CMP "Shape B" issue
(memory note `project_eq_byte1_l6_divergence.md`) and the L15
LI lookup byte-attribution finding (memory `project_l15_li_stack0_byte_attribution.md`)
are downstream symptoms of compute landing while operand state is
mid-relay.

If the same compute fires at row 34 instead, every byte has already
been emitted to the residual, and every cross-row relay attention has
had 29 more rows to complete. The window for race conditions shrinks
to zero.

## 3. Empirical reality (probe, 2026-06-10)

`probe_step_end_completeness.py` runs `IMM 5; PSH; IMM 5; EQ; EXIT`
and compares the residual at MARK_AX vs. MARK_SE within the PSH step
(rows 84 and 111 on the captured sequence). Verdict at L8/L9:

| dim                | @MARK_AX | @MARK_SE | comment                       |
|--------------------|---------:|---------:|-------------------------------|
| MARK_AX            | 1.000    | 0.000    | identity                      |
| MARK_SE            | 0.000    | 1.000    | identity                      |
| MARK_SE_ONLY       | 0.000    | 1.000    | scheduler                     |
| HAS_SE             | 0.996    | 0.997    | global, broadcast already     |
| OP_PSH (this step) | 5.000    | 0.000    | **dispatch flag DEAD at SE**  |
| AX_CARRY_LO (slot) | nibble 5 | empty    | **AX carry DEAD at SE**       |
| AX_CARRY_HI (slot) | nibble 0 | empty    | **AX carry DEAD at SE**       |
| ALU_LO (slot)      | nibble 0 | empty    | **ALU result DEAD at SE**     |
| ALU_HI (slot)      | nibble 0 | empty    | **ALU result DEAD at SE**     |
| CMP (slot)         | empty    | empty    | only set during CMP step      |
| NEXT_PC            | 0.000    | 1.400    | scheduler hot at SE           |

**The user's hypothesis that "by STEP_END, AX/STACK0/OPCODE/etc are all
visible" is FALSIFIED.** STEP_END is currently a scheduler row carrying
only marker identity, NEXT_PC, MARK_SE_ONLY, HAS_SE, and CONST — not a
compute substrate. ALL operand and dispatch state lives at MARK_AX.

The OP_<NAME> dim itself is scoped to `mark == AX AND opcode_at_AX == NAME`
in the dim registry (`dim_registry_dynamic.py:277`), so it physically
cannot fire at STEP_END without an attention relay.

## 4. Migration plan

The principle is sound; the prerequisite is missing. Three waves:

### Wave A — STEP_END operand relay (NEW infra, blocking)

Add a declarative attention head, naming proposal
`step_end_operand_relay`, allocated at L7 or L8 (between fetch and
ALU). It must broadcast from MARK_AX into MARK_SE the following slots
of the same step:

  * `OP_<NAME>` (OPCODE_FLAGS 262..295, 34 dims)
  * `AX_CARRY_LO / AX_CARRY_HI` (328..359, 32 dims)
  * `ALU_LO / ALU_HI` (360..391, 32 dims)
  * `CMP` (the L9 cmp-flag slot)
  * `STACK0_BYTE0..3` (304, 508, 509, 510 alias bits) — already
    broadcast across step via existing relays; verify.

The Q must anchor on MARK_SE; the K must match MARK_AX within the
same step; the V must project the listed slots. Implementation pattern:
mirror `_layer1_has_se_broadcast_head_specs` in `l1_ops.py` which uses
an ALiBi-bounded head to broadcast `HAS_SE` to every row of the step.

Sweep gate: `compare_symbolic_to_lowered_attn` for the new head, then
`sweep_compare_attn.py` for no regressions.

### Wave B — Migrate scheduler-bound rules to STEP_END

Once Wave A lands, the following rules can be moved from MARK_AX gating
to MARK_SE gating. Replace each `("MARK_AX", w)` condition with
`("MARK_SE_ONLY", w)` and update the `scope` annotation. Order of
migration (by safety / value):

| #  | File           | Rule                                       | Reason to move                                                |
|---:|----------------|--------------------------------------------|---------------------------------------------------------------|
|  1 | `l10_ops.py`   | `_layer10_alu_cmp_combine_rules` overrides | Loudest race victim; CMP+0 -0.1 blocker is a MARK_AX hack     |
|  2 | `l10_ops.py`   | `_l10_comparison_combine_rules` cmp_default| Needs MARK_PC -50 blocker today; STEP_END has no PC conflict  |
|  3 | `l10_ops.py`   | `_layer10_alu_bitwise_or_rules`            | 512 units gated `MARK_AX*40`; fires after AX_CARRY relay      |
|  4 | `l10_ops.py`   | `_layer10_alu_bitwise_xor_rules`           | Same shape as OR                                              |
|  5 | `l10_ops.py`   | `_layer10_alu_bitwise_and_rules`           | Same shape as OR                                              |
|  6 | `l10_ops.py`   | `_layer10_alu_shl_shr_zero_rules`          | Race against AX_CARRY_HI                                      |
|  7 | `l9_ops.py`    | `_layer9_cmp_rules` (override 3way)        | The CMP-combine race begins here                              |
|  8 | `l9_ops.py`    | `_layer9_add_hi_nibble_rules`              | Stale `+0` channel diagnostic, see test_l9_collapsed_*        |
|  9 | `l9_ops.py`    | `_layer9_sub_hi_nibble_rules`              | Same as ADD                                                   |
| 10 | `l8_ops.py`    | ALU pre-stage 3-way ANDs gating MARK_AX*60 | Pre-stage doesn't write tokens; pure compute                  |
| 11 | `l8_ops.py`    | LEA/JMP/JSR pre-dispatch gates             | Branch dispatch is the canonical "not-token-emission" work    |
| 12 | `l8_ops.py`    | BZ/BNZ predicate gate                      | Same — depends on AX value, no byte-emission                  |
| 13 | `l16_ops.py`   | `LEV` pop-and-jump compute                 | Frame restore, no per-byte writes                             |
| 14 | `l11_ops.py`   | MEM addr-gather final combine              | The combine step itself, not the per-nibble broadcast         |
| 15 | `l13_ops.py`   | MEM route-to-AX compute                    | Routing decision; bytes already in slot                       |

(The 5 remaining slots are reserved for L14/L15 follow-ups once Wave A
proves out.)

Each migration must:

  1. Keep `compare_symbolic_to_lowered_ffn` green.
  2. Pass `tests/test_l9_collapsed_imm_input_isolated.py` on the
     migrated opcode.
  3. Hold 1096-corpus smoke (`tests/test_1096_*`) at or above current
     sentinel baseline (see memory note
     `project_1096_sentinel_baseline.md`).

### Wave C — Rules that STAY at MARK_AX or byte rows

These belong where they are; do NOT move:

  * Byte emission writes (`OUTPUT_LO+k` / `OUTPUT_HI+k`) at byte rows
    1-4, 6-9, 11-14, 16-19, 21-24, 26-29, 30-33. The byte must land at
    its own row.
  * Register marker bookkeeping (`MARK_PC`, `MARK_AX`, `MARK_SP`,
    `MARK_BP`, `MARK_MEM`, `MARK_STACK0`) — identity dims pinned at
    their own rows.
  * `AX_CARRY_*` and `STACK0_BYTE*` *writes* — these populate the
    slots that Wave A then broadcasts to MARK_SE.
  * NEXT_PC / NEXT_AX / NEXT_SP / NEXT_BP / NEXT_STACK0 / NEXT_MEM /
    NEXT_SE / NEXT_HALT — these are already at STEP_END; correct.
  * L5 fetch attention, L7 operand gather attention, L11/L12 memory
    lookup attention — these are the relays that POPULATE the slots
    Wave B then consumes.

## 5. Why a single-rule migration today would silently regress

The probe shows that migrating one L10 CMP rule from MARK_AX to
MARK_SE today produces a unit whose Q-gate score is dominated by
`MARK_SE`, with `OP_EQ * 1.0 = 0` and `CMP+i * 1.0 = 0` at the SE row.
The threshold is 4.0 with `MARK_SE * 1.0`, so the unit never fires —
the EQ result reverts to the default and the test_l9_collapsed_*
isolation tests break.

The migration MUST be paired with Wave A. We deliberately do NOT
prototype the single-rule migration in this commit; the architecture
note + the probe + the verification of the falsified hypothesis is
the deliverable. The next agent can:

  1. Read this doc.
  2. Re-run `probe_step_end_completeness.py` after landing
     `step_end_operand_relay`. Both `max|OP_<NAME>|` and
     `max|AX_CARRY_LO|` at MARK_SE must reach >= 0.9.
  3. Then migrate rule #1 from the Wave B table.

## 6. References

  * Probe: `c4_release/tools/probe_step_end_completeness.py`
  * Existing HAS_SE broadcast pattern: `l1_ops.py`
    `_layer1_has_se_broadcast_head_specs`
  * Dim ownership: `dim_registry_dynamic.py:91-296`
  * Token layout: `c4_release/neural_vm/token_layout.py`
  * Memory note on related races:
    `project_eq_byte1_l6_divergence.md`,
    `project_l15_li_stack0_byte_attribution.md`

