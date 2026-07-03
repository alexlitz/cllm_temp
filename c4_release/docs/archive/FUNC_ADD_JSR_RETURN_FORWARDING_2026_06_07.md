# func_add_* JSR/LEV return-value forwarding — partial recovery (2026-06-07)

Worktree: `agent-a5304350ae4dc113e` (base `68025246`).

## TL;DR

The 0/25 `func_add_*` failure cluster (`add(a,b) -> b` symptom) was
narrowed to the L6 head 7 **post-LEV AX_CARRY refresh** sub-pattern
introduced by commit `b882482e` (paired with L16's commit `3650e01`).
That refresh wrote `CLEAN_EMBED_LO/HI[STACK0_BYTE0] -> AX_CARRY_LO/HI`
at `MARK_AX` during `OP_LEV`, overwriting the AX value preserved through
LEV by the natural C calling convention.

Disabling the refresh restores 7/25 `func_add_*` passes (was 0/25)
without regressing `func_identity_*` (5/5 still pass) and without
breaking smoke (45/51 = pre-existing baseline).

The remaining 18/25 failures have *different* shapes (`func_add_11:
add(97,23) -> 240` instead of expected `120`, i.e. `0xF0` vs `0x78`)
and live downstream of the same step-0 `MEM_value1=0xff` corruption
that memory note `project_l10_psh_addr_ent_bug.md` attributes to the
L10 PSH `addr0_e0` missing `OP_ENT` guard. Those failures are
independent of return-value forwarding and require a separate fix.

## Failure shape — confirmed

`func_add_0: add(57, 11) expected 68 -> neural 11` (returns second arg).
Bytecode (from `src.compiler.compile_c`):

```
main: ENT 0; IMM 57; PUSH; IMM 11; PUSH; JSR add; ADJ 16; HALT
add:  ENT 0; LEA 24; LOAD; PUSH; LEA 16; LOAD; ADD; RET (=LEV)
```

`add` semantics: `ADD` leaves `AX = a + b = 68`, then `RET (=LEV)`
restores `SP`/`BP`/`PC`. C calling convention preserves AX through LEV
(symbolic_program.py:_op_lev only touches SP/BP/PC).

The failure pattern `add(57, 11) -> 11` matches `STACK0_byte0`
post-LEV: after LEV in `add()`, the freed stack slot at `SP` is the
most recently pushed caller argument, which is `b = 11`. Reading that
into AX explains the symptom *exactly*.

## Root cause — L6 head 7 LEV refresh

`c4_release/neural_vm/unified_compiler/ops/l6_ops.py` head 7
(`layer6_relay_heads_bake.psh_ax_carry_hi`) carried a sub-pattern
introduced in commit `b882482e` ("L6: materialize AX_CARRY_LO/HI from
STACK0_byte0 on post-LEV MARK_AX"):

- Slot 1 Q/K gate: `Q[MARK_AX] + Q[OP_LEV] - Q[CONST]`,
  `K[STACK0_BYTE0]`.
- Slots 2..17, 18, 49..63 V/O writes:
  `V[CLEAN_EMBED_LO/HI+k] -> O[AX_CARRY_LO/HI+k]`.

The original rationale (`docs/1096_ADD_HI_NIBBLE_PLUS_ONE_2026_06_07.md`
+ `.agent-logs/stack_jsr_lev_triage_2026_06_01.md`) was that the L8 ALU
contract requires `AX_CARRY` at `MARK_AX` to hold the "popped return
value, which lives on the freed STACK0 saved-AX slot". That mental
model is wrong: c4 does not save AX on the stack at JSR / pop it at
LEV. AX is preserved through LEV by L3 head 1's carry_forward of the
previous step's `AX byte 0 EMBED` -- which during `add()`'s LEV is the
callee's just-computed return value.

`func_identity(x)` superficially worked because the post-LEV freed
stack slot happens to be the only pushed argument `x`. For
`func_add(a, b)` the freed slot is `b` (the last push), so the
"refresh" forces `AX = b`.

## Fix shipped

Removed the LEV refresh from L6 head 7 spec
(`_layer6_relay_head_specs`):

- Dropped slot 1 Q/K gate (lines 3461-3469).
- Dropped slot 2..17 / 18 / 49..63 V/O CLEAN_EMBED -> AX_CARRY writes
  (lines 3472-3483).
- Kept slot 0 (PSH STACK0_BYTE0 -> ALU_LO relay) and slots 33..48
  (AX_CARRY_HI -> ALU_HI relay) unchanged.

Also dropped the matching cells from `layer6_relay_heads_bake.claims`
and tightened `layer6_relay_heads` (dep anchor) reads/writes to drop
`STACK0_BYTE0`, `CLEAN_EMBED_LO/HI`, `OP_LEV`, `AX_CARRY_LO`,
`AX_CARRY_HI` -- they are no longer programmed by this bake.

L16's `l16_lev_ax_carry_lo/hi_{k}` rules
(`l16_ops.py:228-256`) are kept: they read `AX_CARRY` at `MARK_AX`
during LEV and write `OUTPUT`. With L3 head 1 as the sole `AX_CARRY`
producer they now propagate the *correct* preserved AX.

## Test deltas

Measured on this worktree at commit `68025246`:

| Suite                             | Before | After |
|---|---:|---:|
| `tests/test_smoke.py`             | 45/51 | **45/51** (same 6 LEA + SI/LI/SC/LC failures; no regression) |
| `func_identity_*` (550..574)      | 5/5*  | **5/5** |
| `func_add_*` (575..599)           | 0/25  | **7/25** (passing: 3, 4, 5, 6, 7, 8, 9; failing: 0-2, 10-24) |

*`func_identity_*` baseline at the prior cohort was attributed to
commits `b882482e + 3650e01`; my fix removes the `b882482e`-side
refresh but `func_identity` continues to pass because L3 head 1 alone
carries the single returned arg through LEV.

## What still fails (18/25 func_add_*)

`func_add_11: add(97, 23)` returns 240 instead of 120 (`0xF0` vs
`0x78`). That is byte-0 fully wrong, NOT the "return second arg"
pattern. The pre-existing `step 0 MEM_value1 = 0xff` corruption
(`project_l10_psh_addr_ent_bug.md`: L10 PSH `addr0_e0` missing OP_ENT
guard, `l10_ops.py:3877-3911` `tail_stack0_pushed_addr_byte1_store_ff_after_e0`)
cascades into `main: ENT`'s saved-BP, corrupting the frame for the
entire program. Symptoms vary by `(a, b)`: some compose with my fix to
produce the right answer; others combine to produce wrong byte-0 or
byte-1 (e.g. `0xF0`).

The 18 remaining failures should be attacked by fixing the L10 PSH
`OP_ENT` guard, not by further L6/L16 LEV surgery.

## Files changed

- `c4_release/neural_vm/unified_compiler/ops/l6_ops.py`:
  - `_layer6_relay_head_specs`: drop head 7 LEV refresh slot 1 + V/O
    writes (slots 2..17, 18, 49..63).
  - `make_layer6_relay_heads_bake_op`: drop matching claim cells and
    tighten reads/writes set.
  - `make_layer6_relay_heads_op` (dep anchor): drop
    STACK0_BYTE0/CLEAN_EMBED_LO/HI/OP_LEV reads,
    AX_CARRY_LO/HI writes.
  - Update head-spec + docstring to record the disablement.

## Cross-references

- `docs/1096_ADD_HI_NIBBLE_PLUS_ONE_2026_06_07.md` — prior refutation of
  the L8/L9 hi-nibble hypothesis; identifies `func_add_*` cluster as
  JSR/LEV return-value forwarding (this doc closes that pointer).
- `docs/BUG_CATALOG.md#bug-33` — original misattributed root cause; the
  STACK0_byte0->AX_byte0 mental model encoded there is the bug, not
  the fix.
- Commits `b882482e` (L6 refresh) and `3650e01` (L16 rules) — `b882482e`
  is the active source of this regression; `3650e01` stays in place
  because its `l16_lev_ax_carry_*` materializer is now correctly driven
  by L3 head 1.
- `project_l10_psh_addr_ent_bug.md` (memory note) — the upstream issue
  blocking the remaining 18/25 `func_add_*` and many `var_*` rows.
