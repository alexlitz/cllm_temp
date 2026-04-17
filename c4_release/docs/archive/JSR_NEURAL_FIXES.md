# JSR Neural Implementation Fixes

## Summary

Added OP_JSR flag relays to three critical attention heads to enable fully neural JSR execution.

## Changes Made

### 1. L5 Head 7 - First-Step OP Flag Relay (vm_step.py:3272, 3292)

**Purpose:** Relay OP flags from CODE section to PC marker on first step.

**Changes:**
```python
# Line 3272 - V matrix
attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR

# Line 3292 - O matrix
attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR
```

**Effect:** On the first step, when executing a JSR instruction, L5 head 7 copies the OP_JSR flag from the opcode byte (ADDR_KEY=2) to the PC marker.

### 2. L5 Head 6 - Non-First-Step OP Flag Relay (vm_step.py:3211, 3231)

**Purpose:** Relay OP flags from CODE section to PC marker on subsequent steps.

**Changes:**
```python
# Line 3211 - V matrix
attn.W_v[base + v_offset, BD.OP_JSR] = 1.0; v_offset += 1  # ADDED for neural JSR (non-first steps)

# Line 3231 - O matrix
attn.W_o[BD.OP_JSR, base + o_offset] = 1.0; o_offset += 1  # ADDED for neural JSR (non-first steps)
```

**Effect:** For multi-step JSR execution (e.g., if MEM token generation requires multiple steps), subsequent steps maintain the OP_JSR flag at the PC marker.

### 3. L6 Head 5 - PC→AX OP Flag Relay (vm_step.py:3714, 3735) [PREVIOUSLY ADDED]

**Purpose:** Relay OP flags from PC marker to AX marker for routing.

**Changes:**
```python
# Line 3714 - V matrix
attn.W_v[base + 3, BD.OP_JSR] = 1.0  # ADDED: relay JSR flag

# Line 3735 - O matrix
attn.W_o[BD.OP_JSR, base + 3] = 1.0  # ADDED: write JSR flag to AX marker

# Lines 3752-3759 - FETCH position shift
# Shifted FETCH from position 17 → 18 to accommodate OP_JSR insertion
```

**Effect:** L6 head 5 relays OP_JSR from PC marker to AX marker, enabling L6 head 3 to copy it to TEMP[0] for PC override triggering.

## Neural JSR Execution Flow

1. **Embedding** (confirmed working):
   - JSR opcode byte (value=3) has OP_JSR=1.0 at ADDR_KEY=2
   - Immediate byte 0 has target address at ADDR_KEY=3

2. **L5 Head 7** (first step) or **L5 Head 6** (subsequent steps):
   - Queries PC marker (MARK_PC=1, HAS_SE=0 for head 7 or HAS_SE=1 for head 6)
   - Matches CODE byte at ADDR_KEY=2 (opcode)
   - Copies OP_JSR flag to PC marker

3. **L5 Head 3** (first step):
   - Queries PC marker
   - Matches CODE byte at ADDR_KEY=3 (immediate byte 0)
   - Copies immediate value (jump target) to FETCH

4. **L6 Head 5**:
   - Relays OP_JSR from PC marker to AX marker
   - Relays FETCH (jump target) from PC to AX

5. **L6 Head 3**:
   - Copies OP_JSR from AX to TEMP[0] at PC marker
   - TEMP[0] ≈ 5.0 signals JSR is active

6. **L6 FFN** (units 7449-7479):
   - Detects TEMP[0] > 4.0 at PC marker
   - Overrides PC NEXT_* nibbles with FETCH value
   - PC = jump target (e.g., 25)

7. **L14 Attention**:
   - Generates MEM token with old PC value (PC+5) stored to STACK0

8. **Register Updates**:
   - PC gets new value from OUTPUT (jump target)
   - Execution continues at target address

## Testing

**Test Program:**
```c
// Bytecode representation:
JSR 25;      // Jump to address 25 (instr 3)
EXIT;        // Should not execute
... (padding to address 25)
IMM 42;      // Address 25: Set AX = 42
EXIT;        // Exit with code 42
```

**Expected Result:** Exit code 42

**Test Script:** `test_jsr_final.py`

## Verification Steps

1. **Check OP_JSR is in embeddings:**
   ```bash
   python check_jsr_addr_keys.py
   ```
   Expected: Position 1 (opcode) has OP_JSR=1.0, ADDR_KEY=2

2. **Check OP_JSR is in L5 head 7 weights:**
   ```bash
   python inspect_l5_head7_weights.py
   ```
   Expected: OP_JSR at V[448+4] = 1.000

3. **Run JSR test:**
   ```bash
   python test_jsr_final.py
   ```
   Expected: "JSR works: True"

## Known Issues

- JSR handler is currently commented out in `run_vm.py:242` for testing
- Once neural JSR is verified working, remove the `_handler_jsr` function entirely

## Next Steps

1. Wait for `test_jsr_final.py` to complete (running on CPU, slow)
2. If successful, remove JSR handler from `run_vm.py`
3. Run full test suite to ensure no regressions
4. Update documentation to reflect 100% neural JSR

## Files Modified

- `neural_vm/vm_step.py`: Lines 3211, 3231, 3272, 3292, 3714, 3735, 3752-3759
