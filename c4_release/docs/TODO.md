# Neural VM TODO

## In Progress

### AX Byte 1-3 Passthrough Across Steps
- L10 attention head 1 code written and wired up (ALiBi slope=1.0, W_o=2.0)
- `_set_layer10_byte_passthrough()` in `vm_step.py`
- SUB 0-1 = 0xFFFFFFFF confirmed PASS in quick test
- **Remaining**: Run full carry test suite (ADD 255+1, ADD 128+128, SUB 0-16) and full 19-test suite to verify no regressions

## Pending

### 1. Enable Function Call Weights: JSR, ENT, LEV, LEA
- `_set_function_call_weights()` exists but disabled at `vm_step.py:~1068`
- **Blocker**: L5 heads 2-3 and L6 head 7 fire unconditionally (not gated by opcode)
  - L5 heads 2-3: relay BP EMBED→TEMP at STACK0, SP EMBED→TEMP at BP every step
  - L6 head 7: copies PC OUTPUT to AX_CARRY at STACK0 every step, not just JSR
  - Likely benign but needs verification before enabling
- Runner handlers work today: `_handler_lea`, `_handler_jsr`, `_handler_ent`, `_handler_lev`
- Neural weights add byte-0 computation; runner corrects bytes 1-3

### 2. BZ/BNZ Neural Weight Correctness
- Branch-not-taken works. Branch-taken doesn't.
- BZ PC override uses 4-way AND: `MARK_PC + CMP[2](OP_BZ) + CMP[4](AX_LO_IS_ZERO) + CMP[5](AX_HI_IS_ZERO)`
- Possible CMP[2] overload: head 4 writes OP_BZ, head 6 writes ENT flag
- Zero detection relay may not have correct values when AX was just set by IMM
- Workaround: add runner-side `_handler_bz` / `_handler_bnz`

### 3. OR/XOR/AND Neural Weight Correctness
- `IMM 0xF0; PSH; IMM 0x0F; OR; EXIT` returns 15 instead of 255
- Pre-existing issue in L10 bitwise FFN weights
- Likely operand gather (L7/L8) not correctly relaying STACK0→ALU for bitwise ops

### 4. JMP Neural Weight Correctness
- JMP doesn't override PC. `JMP 2; IMM 99; IMM 42; EXIT` outputs PC=0 (should be 10)
- Root cause: L6 relay writes OP_JMP to CMP[0] but threshold `T_jmp=5.5` may be too high
- Fix: lower `T_jmp` in `_set_layer6_routing_ffn` or add runner-side `_handler_jmp`

### 5. MoE Runner Routing for Branches
- MoE routing assumes sequential flow (`_last_pc + 5`)
- After branch, routes to wrong next-step expert
- Low impact (full residual still passes through), but matters once BZ/BNZ fully work
- Fix: detect branch ops at STEP_END, use `_last_pc` directly for next opcode lookup

### 6. Test Suite Improvements
- Add runner-based tests exercising full `AutoregressiveVMRunner.run()` path (MoE, context pruning, multi-byte corrections)
- MoE provides 1.5-1.7x speedup via runner, but test suite uses `run_program()` which bypasses MoE

## Completed

- [x] Neural DIV/MOD integration (DivModModule at L10.5, delta replacement for OUTPUT_LO/HI)
- [x] Fix PSH STACK0, binary pop SP, and SUB bugs
- [x] Fix PSH: STACK0 not receiving AX value, SP not decrementing
- [x] Fix BZ taken: branch target not applied when AX=0
- [x] Fix ADD 15+1 (hi nibble carry) — normalized L8 carry W_down, corrected L9 threshold
- [x] Inter-byte carry/borrow within single step (L9 detect + L10 relay + L10 FFN override)

## Key Context

### Test Status
- 18/19 tests passing before byte passthrough fix
- SUB 0-1 was the remaining failure (now fixed, needs full regression test)

### Carry/Borrow Architecture
- CARRY[0]: nibble carry (L8→L9)
- CARRY[1]: ADD byte carry-out (L9, 256 units)
- CARRY[2]: SUB byte borrow-out (L9, 256 units)
- L10 head 0: carry relay (AX marker → AX byte positions, ALiBi=5.0)
- L10 head 1: byte passthrough (prev step AX bytes 1-3 → current step, ALiBi=1.0, W_o=2.0)
- L10 FFN: inter-byte carry override (4 units, gated by H1[AX_IDX])

### Diagnostic Scripts (/private/tmp/)
- `test_carry_fix.py` — ADD/SUB carry/borrow test cases
- `test_summary.py` — Full 19-test suite
- `diag_carry6.py` — Layer-by-layer trace at AX marker
- `diag_carry_relay.py` — Verify L10 attention relay
- `diag_tokens.py` — Check generated token values
- `diag_byte_flags.py` — Compare flags at AX vs SP byte positions
