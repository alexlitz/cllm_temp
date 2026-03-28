# Addressing Fixes Summary

## Critical Bugs Fixed

### 1. INSTR_WIDTH Mismatch (FIXED)

**Problem**: `constants.py` had `INSTR_WIDTH=5`, but:
- Model was trained with 8-byte aligned instructions
- Compiler uses `* 8` and `// 8` for all addressing
- Stack operations use 8-byte words

**Fix**: Set `INSTR_WIDTH=8` in `neural_vm/constants.py`
```python
INSTR_WIDTH = 8  # 5 bytes instruction + 3 bytes padding
```

**Impact**: PC now increments by 8 (not 5), matching model training

---

### 2. PC_OFFSET Mismatch (FIXED - MOST CRITICAL)

**Problem**: PC addressing convention mismatch
- **Model trained with**: `PC_OFFSET=2` (PC points to immediate byte, not opcode)
- **Constants file had**: `PC_OFFSET=0` (PC points to opcode)
- **Compiler used**: Implicit `PC_OFFSET=0` (hardcoded addressing)

**Example of Mismatch**:
```
With PC_OFFSET=0:
  idx=0 PC=0:  JSR 16
  idx=2 PC=16: ENT 0    ← Target

With PC_OFFSET=2 (correct):
  idx=0 PC=2:  JSR 18
  idx=2 PC=18: ENT 0    ← Target
```

**Fixes Applied**:

1. **constants.py**: Set `PC_OFFSET=2`
   ```python
   PC_OFFSET = 2  # Legacy addressing: PC points to immediate byte
   ```

2. **src/compiler.py**: Import and use PC_OFFSET
   ```python
   from neural_vm.constants import INSTR_WIDTH, PC_OFFSET

   def current_addr(self) -> int:
       return len(self.code) * INSTR_WIDTH + PC_OFFSET

   def patch(self, addr: int, target: int):
       idx = (addr - PC_OFFSET) // INSTR_WIDTH
       op = self.code[idx] & 0xFF
       self.code[idx] = op + (target << 8)
   ```

**Impact**: Compiler now generates addresses matching model's expectations

---

### 3. ENT Handler Missing PC Override (FIXED)

**Problem**:
- JSR and LEV handlers override PC after execution
- ENT handler did NOT override PC
- Result: PC stuck at ENT instruction, infinite loop

**Fix**: Added PC advancement in ENT handler
```python
def _handler_ent(self, context, output):
    # ... existing ENT logic ...

    # ENT advances PC by INSTR_WIDTH
    next_pc = (exec_pc + INSTR_WIDTH) & 0xFFFFFFFF
    self._override_register_in_last_step(context, Token.REG_PC, next_pc)
```

**Impact**: ENT now advances PC like other function-call handlers

---

## Addressing Convention

With the fixes, the addressing scheme is:

```
Instruction Format: 1 opcode byte + 4 immediate bytes + 3 padding = 8 total
PC Convention: PC points to immediate byte (offset +2 from instruction start)

Memory Layout:
  Addr 0-7:   Instruction 0 (opcode at 0, immediate at 1-4, padding at 5-7)
  Addr 8-15:  Instruction 1 (opcode at 8, immediate at 9-12, padding at 13-15)
  Addr 16-23: Instruction 2 (opcode at 16, immediate at 17-20, padding at 21-23)

PC Values:
  Instruction 0: PC = 2  (points to immediate at addr 1)
  Instruction 1: PC = 10 (points to immediate at addr 9)
  Instruction 2: PC = 18 (points to immediate at addr 17)

Conversion:
  PC → index: idx = (pc - 2) // 8
  index → PC: pc = idx * 8 + 2
```

---

## Test Verification

Example program: `int main() { return 42; }`

**Compiled Bytecode** (with PC_OFFSET=2):
```
idx=0 PC=2:  JSR 18    # Jump to main
idx=1 PC=10: EXIT 0    # Exit if main returns
idx=2 PC=18: ENT 0     # Main function entry
idx=3 PC=26: IMM 42    # Load return value
idx=4 PC=34: LEV       # Return from main
```

**Expected Execution**:
1. Step 0: Execute JSR at PC=2 → Jump to PC=18
2. Step 1: Execute ENT at PC=18 → PC advances to 26
3. Step 2: Execute IMM at PC=26 → Load 42 into AX
4. Step 3: Execute LEV at PC=34 → Return
5. Result: AX = 42

---

## Remaining Issues

### Issue: Model PC Advancement for Non-Handler Opcodes

**Observation**: After ENT handler overrides PC to 26, model executes IMM but may output wrong PC.

**Evidence** (from debug trace):
```
Step 1: ENT at PC=18 → handler overrides to 26
Step 2: IMM at PC=26 (no handler) → model outputs PC=18 (wrong!)
```

**Possible Causes**:
1. Model not trained to advance PC for all opcodes
2. Model expects additional augmentations in embedding
3. PC relay weights incorrect for non-handler opcodes

**Status**: Under investigation

---

## Files Modified

1. `neural_vm/constants.py`: INSTR_WIDTH=8, PC_OFFSET=2
2. `src/compiler.py`: Import constants, use PC_OFFSET in addressing
3. `neural_vm/run_vm.py`: Add PC override to ENT handler
4. `src/transformer_vm.py`: Use INSTR_WIDTH constant (previous session)

---

## Commits

1. `66c22f4`: "Fix INSTR_WIDTH and add PC override to ENT handler"
2. `bba9c33`: "Fix PC_OFFSET mismatch between compiler and model"

---

## Next Steps

1. **Test Execution**: Verify programs run correctly with all fixes
2. **Investigate Model**: Why doesn't model advance PC for IMM and other opcodes?
3. **Handler Strategy**: May need generic PC advancement for all opcodes
4. **Test Suite**: Update tests to use `load_bytecode()` instead of `load()`
