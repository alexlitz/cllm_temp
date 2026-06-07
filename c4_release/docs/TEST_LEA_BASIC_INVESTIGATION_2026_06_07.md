# test_lea_basic investigation — root cause is ENT-first-step token collapse, NOT L10 tail_lea (2026-06-07)

Base: `main` HEAD `6eaa6ef6`
Brief: investigate the L10 `tail_lea_local_ax_marker_byte0_e8` rule
(REMOVAL_1_RETRY_2026_06_06.md said residual sum ~1.0, threshold 7 — rule barely fires).

## TL;DR

`test_lea_basic` (`[ENT, IMM 0, LEA 2, EXIT]`, expect AX != 0) fails with
`exit_code = 0` because the **model never emits a complete first step**.
After `CODE_START ... DATA_END REG_PC 0x0a 0 0 0 REG_AX 0 0 0 0`, the
model degenerates into emitting `0x00` tokens forever (700+ tokens). No
`REG_SP`, no `REG_BP`, no `STACK0`, no `MEM`, no `STEP_END`. The cap is
reached; `_decode_bail_exit_code` finds the only `REG_AX` (value=0) and
returns 0.

**The L10 `tail_lea_local_ax_marker_byte0_e8` rule is a red herring for
this test.** Bumping its weights cannot fire the rule because
`_dispatch_pure_neural` is never invoked — STEP_END is never emitted.
The brief's hypothesis (rule-weight bump) is structurally wrong for
this failure mode.

## Diagnostic probe

Probe: per-element `_dispatch_pure_neural` hook + `_step_one` STEP_END
counter + full context dump (see `/tmp/lea_probe*.py`).

| Program (first-instr trigger) | exit | step_ends | dispatch calls | First-step token stream |
|---|---|---|---|---|
| `ENT, EXIT` | 0 | **0** | 0 | `REG_PC 0a 0 0 0 REG_AX 0 0 0 0 [0x00 × 700]` |
| `ENT, IMM 0, EXIT` | 0 | **0** | 0 | same collapse pattern |
| `ENT, IMM 0, LEA 2, EXIT` (test_lea_basic) | 0 | **0** | 0 | same collapse pattern |
| `ENT, LEA 2, EXIT` | 0 | **0** | 0 | same collapse pattern |
| `IMM 42, EXIT` | 42 | 1 | 1 | `REG_PC 0a 0 0 0 REG_AX 2a 0 0 0 REG_SP f8 ff 0 0 REG_BP ... STEP_END` |
| `IMM 42, LEA 2, EXIT` | 65512 (0xFFE8) | 14 | (multiple) | First STEP_END normal; LEA step emits AX=0xFFE8 from L10 tail rule. |

The collapse trigger is **ENT as the leading instruction**, not LEA. Any
program starting with ENT collapses identically. Any program NOT starting
with ENT emits a normal first STEP_END.

## Why this differs from REMOVAL_1_RETRY's diagnosis

`REMOVAL_1_RETRY_2026_06_06.md` reported the L10 residual probe showed
sum=1.0 (MARK_AX only) at every MARK_AX position for LEA_BASIC. That
measurement is correct but its interpretation needs revising: the
"MARK_AX positions" the probe captured are **partial residual snapshots
during the failed first-step generation**, not real AX-marker positions
of a completed LEA step. The rule "barely fires" because it has nothing
to fire on — no LEA AX marker is ever reached.

The probe at `tools/l10_tail_lea_residual_probe.py` should be augmented
with a STEP_END counter to flag this case.

## Why the brief's three fix paths don't apply

1. **"Bump L10 tail_lea weights"** — Cannot fire; dispatch never runs.
   Even with sum=100 the rule needs a MARK_AX position in a completed
   step, which never materializes for ENT-first programs.
2. **"LEA computed correctly but runner reads wrong dim"** —
   `_dispatch_pure_neural` never runs; no dim read happens.
3. **"Attention Q-side single-condition gate is silently no-op"
   (bf11d69d)** — Not applicable; the model collapse happens at the
   first-step REG_SP emission point, which is gated by ENT-first-step
   rules (`_layer6_ent_first_step_sp_byte0_rules` at
   `l6_ops.py:1277-1307`). These rules are FFN, not attention; the
   Q-side gate finding doesn't apply.

## Why the runner-side LEA AX override doesn't help

`run_vm.py:2387-2395` has a handler-mode LEA override (computes
`_last_bp + imm` and writes REG_AX). Porting it to
`batched_pure_neural._dispatch_pure_neural` (as
`LEA_BASIC_FINDINGS_2026_06_05.md` already documented) has no effect
because `_dispatch_pure_neural` is never called for an ENT-first program.

## Where the failure lives

The model collapses at the **REG_SP emission position** in the first
step. The expected token stream is:

```
REG_PC <pc:4> REG_AX <ax:4> REG_SP <sp:4> REG_BP <bp:4> STACK0 <st:4> MEM <addr:4> <val:4> STEP_END
```

The model emits everything up through `REG_AX <ax:4>` correctly, then
emits `0x00` instead of `REG_SP` (token id 259). This means the head
that decides "next token is REG_SP" is not converging on the right
output for ENT-first programs.

Candidate fix surfaces (NOT investigated in this bounded brief):
- **L6 ENT first-step SP rules** (`l6_ops.py:1277-1307`) write SP byte 0
  via `OP_ENT + MARK_SP + (HAS_SE=-10)` AND gated by FETCH nibbles.
  Verify these fire byte-identically with the legacy bake.
- **L16 ENT-main MEM marker relay** (`l16_ops.py:824`) — possibly
  doesn't route OP_ENT correctly at the MEM marker on first step.
- **L35 / L36 final-token head** — the head that decides the structural
  next-token (REG_PC/REG_AX/REG_SP/REG_BP/STACK0/MEM/STEP_END) may have
  a leak when OP_ENT is at PC=2 (no prior IMM to seed AX_CARRY).

## Recommended next step

A separate investigation: byte-identity probe the first-step output of
the simplest ENT-first program (`ENT, EXIT`) at the post-REG_AX position
and identify which dim is supposed to drive REG_SP token emission.
Compare against `IMM 42, EXIT` at the same position. The divergent dim
is the fix target.

This is NOT a one-rule-bump fix. It's a model-side bug deeper than the
L10 tail layer.

## Acceptance per brief

- **No fix commit landed** — the diagnosed surface (ENT-first-step
  token collapse) is out of scope for the L10 tail_lea path the brief
  scoped.
- **One compile used** — for the probes (no smoke run was needed because
  the dispatch trace alone disproves the brief's hypothesis).
- **Override at `b5cf7099`** unchanged — unrelated to this surface.
- **No regressions introduced** — no code changes.

## Files referenced

- `c4_release/tests/test_smoke.py:374-383` — `test_lea_basic` bytecode +
  `_ne(0)` check.
- `c4_release/neural_vm/batched_pure_neural.py:1941-1980` — `_step_one`
  (only dispatches on STEP_END/TOOL_CALL); 2310-2353 `_decode_exit_code`
  / `_decode_bail_exit_code`.
- `c4_release/neural_vm/run_vm.py:2387-2395` — handler-mode LEA override
  (not reached by pure_neural).
- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py:1277-1307` — L6
  ENT first-step SP byte 0 rules (candidate fix surface).
- `c4_release/neural_vm/unified_compiler/ops/l10_ops.py:6631-6673` —
  `tail_lea_local_ax_marker_byte0_e8` rule (the brief's hypothesis;
  NOT the actual failure surface).
- `c4_release/docs/LEA_BASIC_FINDINGS_2026_06_05.md` — prior
  investigation; documented the same dispatch-never-fires symptom on
  `f3342968`, ruled out runner-side shim.
- `c4_release/docs/SMOKE_FAILURES_2026_06_05.md:30,47` — categorized as
  "never-dispatches (test_lea_basic, ENT-main pin)" cluster.
