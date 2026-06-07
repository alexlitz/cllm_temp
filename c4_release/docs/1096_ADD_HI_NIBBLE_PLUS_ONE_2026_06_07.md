# 1096 corpus: add_* hi-nibble +1 investigation — REFUTED (2026-06-07)

Date: 2026-06-07
Base ref: worktree `agent-a26c35b7073ed3623` at `cd53c076`
(Revert "fix(removal-2): L16 stack0_e8/f8 marker OP_LEV positive predicate").
Mode: pure-neural, `--runxfail`, `tests/test_suite_1096_pure_neural_pytest.py`.

## TL;DR

The hypothesis from the prior agent brief — "every basic `add_*` test
fails with a `+1` hi-nibble drift owned by L8 `AddSub5StageBlock` /
L9 lookup ADD" — does not reproduce. **All 50 basic `add_0..add_49`
tests PASS** in pure-neural mode at this base. The 45 surviving
`add_*` failures live in **`func_add_*` (25/25)** and a subset of
**`expr_add_mul_*`**, both of which exercise machinery far beyond the
L8/L9 ADD path.

A fix targeted at the L8/L9 ADD lookup or `AddSub5StageBlock`
post-op would touch the wrong surface and risk regressing the 50
passing basic adds.

## Measurement

`pytest tests/test_suite_1096_pure_neural_pytest.py -k add_ --runxfail`
on the worktree base `cd53c076`:

| Cluster | IDs | Pass | Fail |
|---|---:|---:|---:|
| basic `add_0..add_49` | 0..49 | **50** | 0 |
| `func_add_0..func_add_24` | 575..599 | 0 | **25** |
| `expr_add_mul_*` (subset) | 800..824 | ~5 | ~20 |

Total: **45 failed / 56 passed** across all `add_` ids. Sample concrete
divergences (declarative-expected vs neural-observed):

- `func_add_0: add(57, 11) exp 68 → neural 11` (returns second arg)
- `expr_add_mul_16: 4+5*2 exp 14 (0x0E) → neural 15 (0x0F)` (lo +1)
- `expr_add_mul_20: 13+15*12 exp 193 (0xC1) → neural 195 (0xC3)` (lo +2)
- `expr_add_mul_14: 15+30*23 exp 705 (0x2C1) → neural 720 (0x2D0)`
  (byte0_lo -1, byte0_hi +1; not a clean "+1" pattern)
- `expr_add_mul_17: 2+10*16 exp 162 → neural 10` (returns first operand)
- `expr_add_mul_19: 29+18*5 exp 119 (0x77) → neural 174 (0xAE)`
  (both nibbles wrong; not "+1" drift)

The error magnitudes and signs are **heterogeneous** — they do not
form a consistent "hi nibble +1" pattern. Several failures (`func_add_0`,
`expr_add_mul_17`) return one of the operands unchanged, which is a
symptom of the binary-op routing (PSH→ALU→AX cascade) collapsing,
not of the L8/L9 hi-nibble cell being off by one.

## Why the prior hypothesis seemed plausible

The brief described two co-located writers to `BD.OUTPUT_HI` (= dim 190,
aliased `OUTPUT_HI_THIS_STEP`):

1. **L9 lookup `_layer9_add_hi_nibble_rules`**
   (`c4_release/neural_vm/unified_compiler/ops/l9_ops.py:200-260`)
   — writes `OUTPUT_HI_THIS_STEP+((a_hi + b_hi + carry_in) % 16) +=
   2.0/S` at MARK_AX, gated by OP_ADD. `carry_in` comes from `CARRY+0`.

2. **L8 + L9 post_op `AddSub5StageBlock`**
   (`c4_release/neural_vm/efficient_alu_addsub_split.py:166-205`,
   installed via `_make_alu_postop_attach_op` in both
   `make_l8_alu_postop_attach_op` and `make_l9_alu_postop_attach_op` —
   `c4_release/neural_vm/unified_compiler/ops/alu_ops.py:312-347`,
   `c4_release/neural_vm/unified_compiler/ops/shared.py:191-291`)
   — clears `OUTPUT_LO/HI/CARRY` at active OP_ADD/SUB MARK_AX, then
   re-writes via `GEToBDConverter` (`efficient_alu_neural.py:286-415`)
   with `+= indicator_lo * 2.0` and `+= indicator_hi * 2.0`.

In the production `alu_mode='lookup'` default, **both** the L9 lookup
hi-nibble rules and the L8 and L9 post-op installs all run.
`_suppress_l9_legacy_addsub_writes` (l9_ops.py:1260-1283) only runs in
`alu_mode='efficient'`, so the legacy `W_down[OUTPUT_HI:, add_units]`
cells are NOT zeroed.

The post_op clears OUTPUT_HI/LO/CARRY via `x_bd_clean *= (1.0 - active)`
before `ge_to_bd` rewrites them. At a full `active=1.0` mask the clear
should wipe whatever L9 lookup deposited. So even though the writers
double up declaratively, the runtime ordering — L8 FFN → L8 post_op
(clears+rewrites) → L9 FFN (writes hi-nibble using `CARRY+0`, which
was cleared by L8 post_op to 0) → L9 post_op (clears+rewrites
correctly from BD operands) — should leave the correct byte at the
output.

The basic `add_0..49` measurement confirms this: every basic ADD —
including `add_13` (203+733=936=0x3A8), `add_22` (348+284=632=0x278),
`add_29` (618+270=888=0x378) which all require a lo→hi carry — passes
cleanly.

## What the actual failures look like

Both `func_add_*` and `expr_add_mul_*` programs interleave multiple ALU
ops with PSH/LEV/JSR/RET frames. The diverse error signatures
(operand-pass-through, off-by-1, off-by-N, completely wrong byte1)
point at the **multi-byte / multi-step PSH→ALU→AX cascade**, the
**JSR/LEV return-value forwarding**, and the **L11/L12 MUL
double-fire** chain — not at the single-byte ADD lookup.

The L17 tail MUL double-fire doc (`docs/L17_TAIL_MUL_DOUBLE_FIRE.md`)
called out exactly this kind of cascade for MUL. The L11 MUL post-op
attach was removed in response. An analogous audit for the
ADD post-op chain is plausible — but the symptom would manifest in
the basic `add_*` tests if the L8/L9 ADD post_op double-fire were
producing a hi-nibble drift, and it does not.

## Recommendation

1. **Do not patch L8/L9 ADD lookup or `AddSub5StageBlock`** based on
   the "hi nibble +1" hypothesis. Basic `add_*` is clean; touching
   these surfaces is more likely to regress than to recover.
2. **Re-scope** the investigation:
   - `func_add_*` cluster (25 failures) — bucket as
     **JSR/LEV return-value forwarding** (the ALU ran correctly but
     the result never reached the caller's AX). Cross-reference
     `func_identity_*` failures and the L10 PSH `MEM_addr0`
     memory note (`project_l10_psh_addr_ent_bug.md`).
   - `expr_add_mul_*` cluster — bucket as
     **multi-step expression / partial-result staging**. The
     intermediate `5*2`-class subresult must survive the next PSH and
     re-enter the ADD ALU at the AX marker. Likely shares the
     `MEM_STORE`/`STACK0_BYTE1` cascade that the L17 MUL doc described.
3. **The `+1` drift in some `expr_add_mul_*` failures** (e.g. _16,
   _20) is byte-0-lo, not hi-nibble, suggesting an `OUTPUT_LO` re-fire
   surface — possibly the same L8/L9 post_op double-fire path but
   manifesting under the multi-step (not single-step) input
   distribution. A block-by-block residual probe at the
   `(L8.post_ops, L9.ffn, L9.post_ops)` boundary for one failing
   `expr_add_mul_*` step is the right next diagnostic — see the same
   pattern as `VAR_REAL_ATTRIBUTION_2026_06_05.md`.

## Files reviewed (no edits)

- `c4_release/neural_vm/unified_compiler/ops/l9_ops.py:200-260`
  (L9 lookup ADD hi-nibble) and `:1260-1283`
  (`_suppress_l9_legacy_addsub_writes`, efficient-only).
- `c4_release/neural_vm/unified_compiler/ops/l8_ops.py:365-517`
  (L8 ALU add_lo + add_carry rules).
- `c4_release/neural_vm/efficient_alu_addsub_split.py:166-260`
  (`_AddSubGEToBD`, `AddSub5StageBlock`).
- `c4_release/neural_vm/efficient_alu_neural.py:27-415`
  (`BDToGEConverter`, `GEToBDConverter`).
- `c4_release/neural_vm/unified_compiler/ops/shared.py:191-291`
  (`_make_alu_postop_attach_op`).
- `c4_release/neural_vm/unified_compiler/ops/all_core_ops.py:543-575`
  (`all_alu_postop_attach_ops`).
- `c4_release/docs/L17_TAIL_MUL_DOUBLE_FIRE.md` — closely-related
  prior diagnosis for the MUL post_op chain.

## Smoke baseline preserved

`pytest tests/test_smoke.py`: **45 passed / 6 failed / 1 deselected**
(LEA + SI/LI/SC/LC memory cluster failures, pre-existing per
`STATUS_1096_2026_06_05.md` and unrelated to ADD). No regression from
this investigation since no code was changed.
