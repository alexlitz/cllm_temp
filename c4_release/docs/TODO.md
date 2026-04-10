# Neural VM TODO

## Pending

### 1. Enable Function Call Weights: JSR, ENT, LEV, LEA
- `_set_function_call_weights()` exists but disabled at `vm_step.py:~1068`
- **Blocker**: L5 heads 2-3 and L6 head 7 fire unconditionally (not gated by opcode)
  - L5 heads 2-3: relay BP EMBED→TEMP at STACK0, SP EMBED→TEMP at BP every step
  - L6 head 7: copies PC OUTPUT to AX_CARRY at STACK0 every step, not just JSR
  - Likely benign but needs verification before enabling
- Runner handlers work today: `_handler_lea`, `_handler_jsr`, `_handler_ent`, `_handler_lev`
- Neural weights add byte-0 computation; runner corrects bytes 1-3

### 2. MoE Runner Routing for Branches
- MoE routing assumes sequential flow (`_last_pc + 5`)
- After branch, routes to wrong next-step expert
- Low impact (full residual still passes through), but matters for optimal MoE routing
- Fix: detect branch ops at STEP_END, use `_last_pc` directly for next opcode lookup

### 3. Test Suite Improvements
- Add runner-based tests exercising full `AutoregressiveVMRunner.run()` path (MoE, context pruning, multi-byte corrections)
- MoE provides 1.5-1.7x speedup via runner, but test suite uses `run_program()` which bypasses MoE

### 4. Contract Documentation
- Contract validation warnings for incomplete layer documentation in `neural_vm/dim_registry.py`
- Only 8 of 16 layers documented in `build_default_contracts()`
- Layers 2-5 and 7-14 missing from contract list
- Non-critical: all tests pass, purely a documentation issue

## Completed

### Core Operations
- [x] Neural DIV/MOD integration (DivModModule at L10.5, delta replacement for OUTPUT_LO/HI)
- [x] Fix PSH STACK0, binary pop SP, and SUB bugs
- [x] Fix PSH: STACK0 not receiving AX value, SP not decrementing
- [x] Fix ADD 15+1 (hi nibble carry) — normalized L8 carry W_down, corrected L9 threshold
- [x] Inter-byte carry/borrow within single step (L9 detect + L10 relay + L10 FFN override)
- [x] AX Byte 1-3 Passthrough Across Steps (L10 head 1, ALiBi=1.0, W_o=2.0)

### Control Flow
- [x] BZ/BNZ branch operations (both branch-taken and branch-not-taken work correctly)
- [x] JMP operation (2026-04-09: fixed with IS_BYTE blockers in L6/L10)
- [x] NOP after IMM (2026-04-09: fixed with IS_BYTE blockers in L6)
- [x] EXIT with return value 0 (2026-04-09: fixed with L6 crossfire patch)

### Bitwise Operations
- [x] OR/XOR/AND operations (verified working: 240|15=255, 15^15=0, 255&15=15)

### Architecture
- [x] n_heads architecture fix (reverted from 16→8 to restore HD=64, fixing attention score budgets)

## Key Context

### Test Status (Updated 2026-04-09)
- **Quick suite**: 100/100 tests passing (100%)
- **Full suite**: 1096/1096 tests passing (100%)
- **Specific bug tests**: JMP 16, NOP after IMM, EXIT 0, IMM 123 all passing
- All core operations (arithmetic, bitwise, control flow, memory) verified working

### Carry/Borrow Architecture
- CARRY[0]: nibble carry (L8→L9)
- CARRY[1]: ADD byte carry-out (L9, 256 units)
- CARRY[2]: SUB byte borrow-out (L9, 256 units)
- L10 head 0: carry relay (AX marker → AX byte positions, ALiBi=5.0)
- L10 head 1: byte passthrough (prev step AX bytes 1-3 → current step, ALiBi=1.0, W_o=2.0)
- L10 FFN: inter-byte carry override (4 units, gated by H1[AX_IDX])

### Bug Fix History (2026-04-09)

**Problem**: JMP, NOP after IMM, EXIT 0, and IMM 123 all failing in pure neural mode

**Root Causes**:
1. Architecture regression: n_heads changed from 8→16, breaking attention score budgets
2. Residual activation: FFN units firing at wrong positions due to residual values in ALU/AX_CARRY/OUTPUT_BYTE dimensions
3. Crossfire: Unprogrammed L6 units with OUTPUT_BYTE weights writing to OUTPUT

**Fixes Applied**:
- Reverted n_heads from 16→8 (line 731)
- Added IS_BYTE blockers (-S*10 to -S*20) in L6 NOP/JMP routing and L10 bitwise ops
- Added OP_JMP blocker (-S*20) in L6 IMM routing
- L6 crossfire patch: zero out units with OUTPUT_BYTE weights writing to OUTPUT

**Documentation**: See `BUG_FIXES_2026-04-09.md` for complete technical details
