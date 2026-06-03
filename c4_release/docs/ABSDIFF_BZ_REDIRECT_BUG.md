# absdiff cluster: BZ taken-path PC redirect bug

## Symptom

absdiff_* 1096 tests: 27/30 fail with `neural_exit=None` — model never
reaches EXIT. Probed test: absdiff_0 (`|16 - 85|`, expected 69).

## Trace

absdiff_0 compiles cleanly and dispatches into `abs_diff` via JSR at PC 26.
Per-step PC tracking (canary in lookup mode, pure_neural=True):

| Step | PC  | Instruction       | Note |
|-----:|-----|-------------------|------|
|   6  |  26 | ENT 0             | abs_diff entry — correct |
|   7  |     | LEA, LI, PSH, ... | sequential — correct    |
|  13  |     | BZ 18             | should redirect to PC 146 (idx 18) |
|  14  |  90 | LEA 24            | **fell through — bug**  |
|  15-33 |   | walks idx 12-17, then off the end | PC increments forever by 8 |

PC walks past the bytecode and never reaches LEV at idx 17. STEP_END
emits forever, AX stays 0, `_decode_exit_code` returns 0.

## Root cause

L4 FFN BZ redirect block (`_set_layer4_ffn`, `vm_step.py:5030-5095`)
doesn't carry-forward the taken-path PC. Documented in
`run_vm.py:2193-2196` comment: "Phase-4 BZ taken-path: `_set_layer4_ffn`
PC carry-forward bug. All `test_pure_neural_jmp_bz` BZ taken/not-taken
cases xfail."

A Python BZ override exists at `run_vm.py:2192-2215` — it resolves
`target_pc` from CMP flags and overrides REG_PC in the last step. But
this override sits **AFTER** the `pure_neural` early-return at
`run_vm.py:2094`. So:
- `pure_neural=True` runs (canary, 1096 driver) skip the override.
- `pure_neural=False` handler-mode runs hit the override and work.

## Why this affects the full absdiff cluster

`BatchedPureNeuralRunner` (the 1096 corpus driver) is also pure-neural
and has its own dispatch loop. Both surfaces miss the override.

Additionally, the canary trace shows AX=0 for every step including
LEA/LI/IMM. The post-step AX byte channel is not being populated by
pure-neural ALU writes for arithmetic ops. So even on a working BZ
relay, CMP[4]/CMP[5] would read AX=0 unconditionally — making BZ
*always* taken (a separate symptom).

## Proposed fixes (priority order)

### 1. Tactical: hoist BZ override (unblocks 27 absdiffs)

Move the Python BZ/BNZ override block (`run_vm.py:2192-2215`) ABOVE
the `pure_neural` early-return at line 2094. Gate with
`not self.trust_neural_alu` so true pure-neural test cases
(`test_pure_neural_jmp_bz`) are unaffected. Mirror this hoist in
`BatchedPureNeuralRunner`.

### 2. Strategic: real neural fix

Audit L6 head 4 (`_set_bz_bnz_relay`, `vm_step.py:4112-4139`) — confirm
CMP[4]=AX_LO_IS_ZERO and CMP[5]=AX_HI_IS_ZERO are written at the BZ's
PC position from the previous step's AX. Also verify L9/L10 ALU
populates AX byte slots for GT (`AX=0` everywhere suggests the byte
channel is silent post-step for non-IMM ops).

### 3. Diagnostic next-step

Re-run canary with `pure_neural=False, trust_neural_alu=False`
(handler-mode + lookup ALU). If absdiff_0 returns 69, that confirms
the BZ override does the right thing and the only gap is the early-
return at line 2094.

## Cross-references

- `c4_release/neural_vm/run_vm.py:2094` — `pure_neural` early-return
- `c4_release/neural_vm/run_vm.py:2192-2215` — Python BZ/BNZ override
- `c4_release/neural_vm/vm_step.py:5030-5095` — `_set_layer4_ffn`
- `c4_release/neural_vm/vm_step.py:4112-4139` — L6 head 4
  `_set_bz_bnz_relay`
- `c4_release/tests/test_pure_neural_jmp_bz.py` — xfail BZ tests

## Status

Diagnosis only; no fix applied. Estimated recovery: 27/30 absdiff
tests via the tactical hoist, plus likely cascade fixes to other
branch-heavy 1096 tests (if_eq probably similar since CMP→BZ→EXIT is
the same path).
