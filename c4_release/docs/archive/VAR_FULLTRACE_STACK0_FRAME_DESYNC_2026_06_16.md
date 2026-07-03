# var full_trace root is STACK0 OUTPUT-band crush → frame desync (2026-06-16)

HEAD `348d7ad9` + this session's commit `486ae2ae`. Path: spec_k=0, hook-free
(`tools/probe_var_full_chain.py`, `probe_var_raw_tokens.py`,
`probe_var_pc_incr.py`). GPU 0. Build: 52 physical blocks, d_model=1090.

## TL;DR — the complete, definitive root chain

**ALL prior var docs are superseded.** The step-0 JSR→BP leak, step-1 ENT
STACK0 L16 broadcast, and L3 SP_byte2 roots are all RESOLVED on current HEAD
(steps 0,1,2,3 now frame-clean). The var_simple full_trace blocker is now a
**STACK0 OUTPUT-band crush that emits a stray REGISTER-MARKER token mid-frame,
shifting the fixed-35-token decode frame and misreading PC**.

`run_1096_canonical --criterion full_trace --spec-k 0` on id 250
(`int x; x=990; return x;`, bytecode `JSR3; HALT; NOP; ENT8; LEA-8; PSH;
IMM990; SI; LEA-8; LI; EXIT`):

* **HEAD**: first divergence `step 4 (IMM) PC exp=58 got=66`, all 100 var
  programs (var_simple/mul/three/update) identical.
* **AFTER commit 486ae2ae**: advances to `step 5 (SI) PC exp=66 got=74`
  (all 100 var programs).

## The mechanism (why PC misreads)

The production full_trace decode (`_ff_check_new_steps`,
batched_pure_neural.py:2118+) reads each step from a **FIXED 35-token slice**
(`start = prefix + step*35`), NOT marker-re-anchored. `token_pos` advances +1
per emitted token and does NOT realign on STEP_END. So if any step emits a
STEP_END one token early (34 tokens not 35), every later step's slice is
misaligned and PC/AX read from the wrong offsets.

The early STEP_END is caused by a STACK0/MEM-region byte row whose OUTPUT band
is **dead (all cells negative)**: nothing scores positively, so a stray
register-MARKER token (257-262/268) wins the LM argmax. That marker ends the
step's data region early → 34-token step → frame shift.

`probe_var_raw_tokens.py`: on HEAD the step-3 (PSH) `STACK0[3]` row emits MEM-
marker token 261 (frame shifts at step 3 → step-4 PC=66). After 486ae2ae
step 3 is clean and the shift moves to the step-4 (IMM) `STACK0[0]` row, which
emits PC-marker 257 (frame shifts at step 4 → step-5 PC=74).

## Commit 486ae2ae (LANDED) — step-3 fix

`_layer10_psh_stack0_passthrough_head_spec` (block 16 / L11, head 3) fires on
the STACK0 byte-2/3 QUERY rows of a PSH step and crushes OUTPUT to ~-652.
Fix: flag-gated (`C4_PSH_STACK0_HIGHBYTE_DARKEN`, default ON) slot-7
`BYTE_INDEX_2/3 = -2e9` NOT-blocker. Byte 0 (value producer) and byte 1
(test_si_li_16bit round-trip) untouched. Smoke 51/0; exit_code 14/25
(unchanged vs flag-off); flag-off PARAM_HASH unchanged.

## The REMAINING root (step-4 IMM STACK0[0]) — DOCUMENTED WALL

Hook experiment (`probe_var_full_chain` + a block-41 OUTPUT-preserve hook on
MARK_STACK0 rows): **restoring the pre-L25 OUTPUT on STACK0 rows makes the
frame stay aligned through ALL 8 steps (step4 PC=58, step5 PC=66, step7 PC=82
— var_simple would PASS full_trace).** So the step-4 STACK0[0] crush is the
sole remaining blocker.

Genesis (`probe_var_pc_incr`-style per-block OUTPUT trace at row 252):

| block | logical | OUTPUT_LO[0] | OUTPUT_LO[8] (0xE8) | note |
|------:|--------:|-------------:|--------------------:|------|
| 34 | L20 FFN | **+619** | +3324 | L20 writes a 0x00 default at cell-0 |
| 41 | L25 FFN | **-2.3e8** | -8.4e6 | the crush (all cells negative) |

Two-attractor band: at the var IMM step the STACK0[0] OUTPUT band carries BOTH
a strong cell-0 = **+619** (the L19/L20 `STACK0=0x00` non-PSH default) AND the
relayed **0xE8** (cell 8 + cell 14, ~3324). Non-active cells sit at **-10.6**.

At L25 (block 41) the `stack0_pop_loaded_output_rules` family (255 rules,
`tail_stack0_pop_loaded_byte_XX`, one per byte value) fires. Each rule's gate
is `OUTPUT_LO[lo]*0.05 + OUTPUT_HI[hi]*0.05 + CMP+3*0.5` (threshold 12). With
TWO strong attractors (cell-0 619 AND value cell 3324), DOZENS of value-rules
clear threshold; each `byte_writes(value, 500)` lays -500 on its 15 non-target
nibbles, and the sum across all firing rules CRUSHES every OUTPUT_LO cell to
~-2.5e8 → dead row → stray PC-marker (257) wins → frame desync.

Contrast `test_si_li_16bit` (PASSES): its genuine pop row 224 has a CLEAN
single-attractor band (cell-0 + the 0x34 value cell, non-active cells ~0), so
far fewer rules fire and the 0x34 emits cleanly. The `pop_loaded` family is
**load-bearing** for the memory smoke (row 224 needs 0x34 preserved) — it
cannot be gated off.

## Why this is a WALL (not a single-rule fix)

1. The true enabler is the **two-attractor band**: the L19/L20 `STACK0=0x00`
   non-PSH default (cell-0 = 619) coexisting with the legitimately-relayed
   retained STACK0 value (0xE8). A clean fix removes the 0x00 default ONLY
   when STACK0 holds a retained push (after a PSH, before the next push), i.e.
   an L19/L20 STACK0-relay-vs-clear disambiguation — multi-rule, and the same
   surface the memory SI/LI/SC/LC defaults depend on.
2. The L25 `tail_bit32_result_correction` bank is the documented
   **width-sensitive, load-bearing hard wall** (memory note
   `project_l10_tail_bank_width_sensitive`): EXACTLY 2059 rules; the
   `pop_loaded` family is required by the memory smoke. Re-gating it risks the
   SI/LI/SC/LC round-trips.
3. Per `feedback_single_rule_fixes_are_zero_sum`, narrowing one rule here just
   relocates the dead-row crush to the next STACK0 region.

The robust fix is a deliberate two-part build: (a) at L19/L20, suppress the
`STACK0=0x00` default on rows where a retained push value is present (clean the
band to a single one-hot), so (b) only the matching `pop_loaded` rule fires at
L25 and the value emits cleanly. Both parts need byte-identity gating against
the full memory smoke + the cmp/branch byte-identity test the L25 bank guards.

## Tools added (read-only, spec_k=0)
- `tools/probe_var_pc_incr.py` — per-block EMBED/OUTPUT decode at a step's PC
  prediction row.
- `tools/probe_var_raw_tokens.py` — extended to 6 steps; the raw emitted
  token stream + 35-frame offset labels (shows the stray-marker frame shift).
