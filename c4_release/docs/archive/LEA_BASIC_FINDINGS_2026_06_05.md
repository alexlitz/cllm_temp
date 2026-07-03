# test_lea_basic — fix attempt findings (2026-06-05)

## Symptom

`test_lea_basic` smoke fails with `expected != 0, got 0` on the address
group. The bytecode is `ENT(0), IMM(0), LEA(2), EXIT` and the test only
asks for any non-zero AX (== `BP + 2`).

## Baseline (this worktree, branch `lea-basic-fix` from
`speedup-cache-and-buckets` HEAD `f3342968`)

```
37 passed / 14 failed / 1 deselected — 121.71s
```

`test_lea_basic` is in the failing set. The 13 other failures match the
`SMOKE_TRIAGE_POST_L5.md` matrix unchanged.

## Failure mode is NOT "no STEP_END"

The highbit-imm commit message (`f3342968`) describes
`test_lea_basic` as an "ENT/LEA model stall". The actual observed
failure is just `exit_code = 0`, which puts it in the same
"A — AX-zero / `exit_code=0`" cluster called out in
`docs/SMOKE_TRIAGE_POST_L5.md` for memory + shift + comparison + bit32
tests. There is no batch-run error or "got None" — the runner halts
cleanly and reports `0`.

## Runner-side LEA shim does not move the result

The serial runner already carries a handler-mode LEA override at
`c4_release/neural_vm/run_vm.py:2310-2318`:

```python
if ax is not None and exec_op == Opcode.LEA:
    # TODO(phase-5): remove once ENT establishes BP correctly
    # in pure_neural (test_lea_basic xfails). Required.
    imm = bytecode[exec_idx] >> 8
    if imm >= 0x800000:
        imm -= 0x1000000
    alu_result = (self._last_bp + imm) & 0xFFFFFFFF
    self._last_ax = alu_result
    self._override_register_in_last_step(context, Token.REG_AX, alu_result)
```

This override is inside the **handler-mode** branch — the pure-neural
branch at `run_vm.py:1972-2122` returns at line 2122 before reaching
it, and the batched runner
(`c4_release/neural_vm/batched_pure_neural.py::_dispatch_pure_neural`)
has no equivalent block.

I ported the override to the batched runner in two forms, mirroring the
BZ/BNZ pattern at `batched_pure_neural.py:1944-1961`:

* `exec_op == Opcode.LEA`: override REG_AX in the just-emitted LEA step
  and (in the second iteration) also force `s.last_pc =
  exec_pc + INSTR_WIDTH` and rewrite REG_PC, so the "Neural-authoritative
  early exit" check at `batched_pure_neural.py:2020-2026` sees the
  trailing EXIT and halts with the synthesized AX.
* `exec_op == Opcode.IMM` AND `last_pc` skipped one byte forward past a
  LEA: mirrors the collapsed-step binary-ALU recovery at
  `batched_pure_neural.py:1990-2013` for the IMM→LEA collapse case.

**Neither form moved the test.** Smoke after the two-trigger shim was
again `37 passed / 14 failed`, with `test_lea_basic` still reporting
`expected != 0, got 0`. The shim was reverted (working tree clean at
commit time).

The runner-shim non-result means the failure cannot be patched at the
override layer the BZ/BNZ + IMM-binop overrides operate at. The most
likely explanations:

1. The LEA step never reaches `_dispatch_pure_neural` — either the
   model stalls inside the LEA step until the per-element token budget
   (`max(expected_steps) * STEP_TOKENS` = 4 × 35 = 140 tokens) is
   exhausted, and `_decode_bail_exit_code` returns the in-progress
   REG_AX (= 0); or the model collapses LEA into a prior step in a
   shape neither of my two triggers matches.
2. The model emits STEP_END for LEA, my override fires, **but** the
   subsequent EXIT step's token stream emits its own REG_AX = 0 over
   the top, and the early-exit fallthrough never fires because the
   model's emitted REG_PC for LEA does not point at the trailing EXIT
   (the override of REG_PC into the LEA step is masked by the model's
   own emission, so the next STEP_END's `_extract_register` returns
   the stale model value).

Distinguishing (1) from (2) requires inspecting the token stream the
runner buffers for the LEA program — outside the one-compile + one-smoke
budget for this brief.

## Why this is the right place to stop

`feedback_single_rule_fixes_are_zero_sum.md` documents that single-rule
fix attempts are zero-sum (0/5 net positive in prior session). This
fix attempt extends that pattern: a runner-side shim with the same
shape as the working BZ/BNZ / IMM-binop overrides did **not** move the
test. The hop from "the override-shape works for BZ/BNZ" to "the same
override-shape works for LEA" is broken by something specific to LEA's
token-emission shape that needs a token-trace, not another rule.

## Recommended next step

Capture the LEA program's token stream once (e.g. via
`batched_pure_neural.BatchedPureNeuralRunner._step_one` with a
print-on-STEP_END instrumented) to confirm whether the LEA step ever
emits STEP_END at all. That single observation will tell whether the
fix surface is:

* the cap-hit bail decode (`_decode_bail_exit_code` — synthesize LEA
  during bail when the in-progress step is LEA), or
* the EXIT-step AX override (force the EXIT step's REG_AX to whatever
  the prior LEA step's overridden AX was).

## Files referenced

* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/batched_pure_neural.py`
  (lines 1838-2027: `_dispatch_pure_neural` + early-exit + bail)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/run_vm.py:2310-2318`
  (serial-runner LEA shim, handler-mode only — never reached by
  pure-neural)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/unified_compiler/ops/l5_ops.py:493`
  (L5 opcode decode for OP_LEA — opcode value 0, decode rule
  `("37", "OP_LEA+0")` is present)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/unified_compiler/ops/l8_ops.py:406-441`
  (L8 LEA lo-nibble ALU rules — gates on OP_LEA, reads ALU_LO + FETCH_LO)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/unified_compiler/ops/l9_ops.py:263-314`
  (L9 LEA hi-nibble ALU rules)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/unified_compiler/ops/model_ops.py:197-224`
  (LEA first-step ALU init — only fires when `HAS_SE=0`)
* `/tmp/c4-lea-basic-fix/c4_release/neural_vm/unified_compiler/ops/l16_ops.py:1514-1579`
  (L16 LEA writeback — `BP-8` shape only, gated by `CMP+7 + HAS_SE`)
* `/tmp/c4-lea-basic-fix/c4_release/docs/SMOKE_TRIAGE_POST_L5.md`
  (does not list `test_lea_basic` — it was passing at HEAD
  `5f51cecb`; this means the regression to FAIL happened between
  `5f51cecb` and `f3342968`; the highbit-imm commit message also
  acknowledges LEA as broken at `f3342968`)
