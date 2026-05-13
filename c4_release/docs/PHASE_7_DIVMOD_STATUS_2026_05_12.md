# Phase 7 DIV/MOD status — 2026-05-12

## TL;DR

`TestSmokeBasic::test_div_basic` and `TestSmokeBasic::test_mod_basic` both
return `0`. Root cause is **not** in the L10 `FlattenedDivMod` composite
itself; the composite computes the correct result when given correct
operands. The bug is in the **L7 operand-gather** path (shared with ADD/SUB)
that fails to deliver operand A from STACK0 byte 0 into `ALU_LO/HI` at the
AX-marker position.

This is Phase 2/3 scope, not Phase 7. Per the task brief: *"div/mod depend
on the same L7 operand-gather path that Phase 2/3 fixes. If `test_add_basic`
is also currently failing on your worktree, your div/mod fix may need
Phase 2/3 to land first."*

## Evidence

### Smoke-test failure pattern (HEAD = `ecdbe67`)

```
test_add_basic   FAIL: 10 + 32 -> 32  (expected 42; got just operand B)
test_sub_basic   FAIL: 50 - 8  -> 0x0F0F0F07 (operand A = 0 with borrow chain)
test_mul_basic   FAIL: 6  * 7  -> 0 (0 * 7 = 0)
test_div_basic   FAIL: 84 / 2  -> 0 (0 / 2 = 0)
test_mod_basic   FAIL: 43 % 10 -> 0 (0 % 10 = 0)
```

All five failure values are consistent with **operand A = 0** at the
ALU — the ALU correctly computes f(0, B) for each opcode.

### FlattenedDivMod composite is structurally correct

A standalone synthetic harness
(`c4_release/debug_archive/diag_divmod_synthetic.py`) constructs the
composite directly via `install_bdtoge / install_longdiv / install_getobd`
and feeds it a hand-crafted BD residual at an AX-marker position with
operand A in `ALU_LO/HI` and operand B in `AX_CARRY_LO/HI`:

```
PASS: DIV 10/2          -> 5  (expect 5)
PASS: DIV 84/2          -> 42 (expect 42, matches test_div_basic)
PASS: MOD 43%10         -> 3  (expect 3,  matches test_mod_basic)
PASS: DIV 0/2 (L7-fail) -> 0  (reproduces smoke-test failure exactly)
```

The fourth case (`opA = 0`) reproduces the observed smoke failure
byte-for-byte, confirming the L7 operand-gather hypothesis.

## Why this is blocked on Phase 2/3

The L7 operand-gather path is owned by:

- `c4_release/neural_vm/unified_compiler/ops/l7_ops.py::make_layer7_operand_gather_op`
  (head 0: prev STACK0 byte 0 -> ALU_LO/HI at AX marker)
- `c4_release/neural_vm/setup_helpers.py::_set_layer7_operand_gather`
  (the bake function it calls)

The L10 DIV/MOD composite reads `BD.ALU_LO/HI` for operand A:

- `c4_release/neural_vm/efficient_alu_neural.py::BDToGEConverter.forward`
  lines 115-123: `x_ge[:, :, 0, NIB_A] = (alu_lo * k_coeffs).sum(-1)`.

If `ALU_LO[0..15] == 0` at the AX-marker position when `BDToGEConverter`
runs (because L7 head 0 failed to gather STACK0 byte 0), then `NIB_A = 0`
and the long-division pipeline computes `0 / B` or `0 % B`, both `0`. No
amount of fixing inside `FlattenedDivMod` can recover the operand from a
zeroed input.

## What I did not change

- `efficient_alu_divmod_split.py`: pipeline is correct, no edits.
- `unified_compiler/ops/alu_ops.py::make_alu_divmod_composite_ops`: bake
  registration is correct, no edits.
- `unified_compiler/ops/shared.py::_FlattenedDivModBuilder`: correct, no
  edits.

## What needs to happen for Phase 7 smoke-DIV/MOD to pass

Phase 2/3 (L7 operand-gather) needs to land. Once `test_add_basic` returns
42 (handler-mode parity), `test_div_basic` and `test_mod_basic` should
automatically pass without any further changes to the DIV/MOD pipeline,
because the synthetic harness already confirms the composite produces the
correct outputs given correct operands.

## Reproduction

```
python c4_release/debug_archive/diag_divmod_synthetic.py
```

Expect all four PASS lines (including the L7-fail case which is
intentional).
