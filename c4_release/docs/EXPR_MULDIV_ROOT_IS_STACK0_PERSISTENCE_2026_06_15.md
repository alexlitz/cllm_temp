# expr-with-mul/div clusters: root is STACK0 cross-step persistence (#221), NOT the high-byte handoff

2026-06-15. Comprehensive probe lane on `expr_mul_div` (850-874),
`expr_add_mul` (800-824), `expr_mod` (875-899), `expr_paren` (825-849).
Verified spec_k=0, BUILT `model.dim_positions` (d_model=1090, n_heads=10),
hook-free, on HEAD `272e5c40` (flags at production defaults:
`C4_MUL_WIDTH2=1`, `C4_DIV_MULTIBYTE=1`, `C4_STACK0_B0_DUMP=1`).

## Full-trace baseline (HEAD 272e5c40, `tools/run_1096_canonical.py --criterion full_trace`)

| cluster        | ids     | pass |
|----------------|---------|------|
| expr_add_mul   | 800-824 | 0/25 |
| expr_paren     | 825-849 | 19/25|
| expr_mul_div   | 850-874 | 0/25 |
| expr_mod       | 875-899 | 3/25 |

## The hypothesis that was REFUTED

The brief hypothesised the break was the **intermediate's HIGH byte** not
reaching the 2nd op's operand: "MUL=784 computed correctly but DIV emits
769 = undivided", to be fixed by mirroring the DIV/MOD `STACK0_BYTE_VAL_1`
relay template.

Direct per-step register-trace + per-frame residual probes
(`tools/probe_expr_regtrace.py`, `tools/probe_stack0_b0_nibble.py`,
`tools/probe_expr_frame_select.py`) **refute this**:

* The FIRST op (MUL/ADD) computes the correct multi-byte result and the PSH
  stores it correctly. For `14*56/8`: per-step AX = `[14,14,56,784,784,8,769]`
  — MUL=**784 correct**; the STACK0 frame created by the PSH-of-784 holds
  byte0=**16 = 0x10 (correct)** at its `STACK0_BYTE0` row.
* The break is that the **SECOND op reads a DIFFERENT, STALE STACK0 frame**.
  For `14*56/8` the DIV reads a re-materialised STACK0 frame whose byte0 =
  **14** (the ORIGINAL first operand `a`), not 16. The intervening `IMM 8`
  step (which does NOT touch the stack) re-emits top-of-stack as a value
  carried from TWO frames back, not the just-pushed intermediate.
* This is a **byte-0** failure, not a high-byte one. The DIV/MOD
  `STACK0_BYTE_VAL_1` high-byte relay (the brief's template) therefore
  cannot help — even the low byte the 2nd op reads is wrong.

### Quantified (`tools/probe_expr_frame_select.py`, all 50 mul/div+add_mul)

> **45/50** programs: the 2nd op reads a STALE/WRONG `STACK0_BYTE0` frame
> (it reads the original `a`, not the pushed intermediate). The 5 "OK_b0"
> are byte-0 coincidences (stale value == intermediate byte0) that still
> fail on a later step/byte.

`C4_DIV_MULTIBYTE=1` (already the default) does NOT move expr_mul_div
(0/25 → 0/25) or expr_mod, confirming the DIV relay is not the lever.

## The actual root = STACK0 byte-0 cross-step persistence wall (#221)

This is exactly [[project_if_bool_expr_is_stack0_highnibble_framing_drift]]
(memory) / `STACK0_BYTE0_DUMP_CARRY_ROOT_2_2026_06_13.md`: the pushed
top-of-stack value emits fine at the PSH step (freshly computed) but is
LOST/replaced at the NEXT step where it must be persisted across the step
boundary. Surface: `layer10_stack0_persistence_head` /
`layer10_stack0_byte_relay` heads 4/5/6 (`l10_ops.py`), and the L3 byte-0
marker carry. The persistence head re-selects an OLDER STACK0 frame (the
first-operand push) over the most-recent (the intermediate push).

`expr_mod` (`a%b+c`, single-byte intermediates) fails the SAME way: the
MOD result is dropped and the ADD operates on the ORIGINAL dividend
(`46%6+6` → neural got 52 = 46+6, not 4+6=10). Same cross-step
top-of-stack persistence corruption.

`expr_paren` (`(a+b)*c`) mostly PASSES (19/25) because its ADD intermediate
`a+b ≤ 40` is single-byte AND the structure (`IMM, IMM, ADD, IMM, MUL`)
puts the 2nd op closer; the persistence still drifts on the both-nibbles-
nonzero values (the 6 fails), matching the #221 value-dependence.

## Why no single-rule expr-only fix exists

* The corruption is upstream of every operand-gather: the model's
  top-of-stack residual itself holds the wrong value by the 2nd-op step.
  Re-pointing `layer7_operand_gather` (ALiBi recency → most-recent
  `STACK0_BYTE0`) can't help — the most-recent frame IS the corrupt one,
  and there's no residual signal distinguishing a corrupt re-materialised
  frame from a legitimate one.
* The fix lives in the `layer10_stack0_persistence` head (load-bearing for
  var/store/SI/SC; single-rule edits here are documented zero-sum, 0/5
  historical agents) and the L16/block-38 `tail_bit32_result_correction`
  H1/H3 nuke — the SAME surface as the AX byte-1 dump wall. The
  `C4_STACK0_B0_DUMP` infra (now default-ON on main) fixes the byte-0
  EMISSION (slicer reads 35 tokens) but NOT the cross-step VALUE
  corruption probed here; and the discriminator between framing-drift rows
  and healthy carried-arithmetic rows is the open blocker (2 prior agents
  bounced — see ROOT_2 doc).

## Conclusion

The expr-with-mul/div clusters are blocked by the **cross-step STACK0
byte-0 value-persistence wall (#221)**, the dominant ~760-fail root, NOT by
an intermediate high-byte handoff. No model edit landed (smoke held 51/0;
HEAD byte-identical). The lever is the coordinated multi-session
persistence/AX-byte build, not an expr-local relay.

Reusable probes added (all spec_k=0, BUILT `dim_positions`, compile real
corpus bytecode via `compile_c`):
`tools/probe_expr_regtrace.py` (per-step AX trace vs oracle),
`tools/probe_expr_frame_select.py` (bulk stale-frame classifier — the 45/50
number), `tools/probe_stack0_b0_nibble.py` (per-frame STACK0_BYTE0 value),
`tools/probe_expr_intermediate_relay.py`,
`tools/probe_mul_psh_allbands.py` (locates the high byte across all bands),
`tools/probe_psh_axrows.py`, `tools/probe_div_operand_select.py`.
