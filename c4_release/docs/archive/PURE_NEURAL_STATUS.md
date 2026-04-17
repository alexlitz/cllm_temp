# Pure Neural Execution Status (use_speculator=False)

## Current State: **NOT WORKING**

The neural transformer model **cannot execute programs autonomously** without the Draft VM speculator.

## Test Results

```bash
python test_pure_neural.py
```

**Result**: `('', 0)` instead of expected `300`

### Detailed Analysis

Using `test_token_generation.py`, the model:
- Generates only **2 steps** (105 tokens) before emitting HALT
- Returns exit code **0** instead of correct value
- Never properly executes the program

Expected behavior for `int main() { return 42; }`:
1. Execute JSR 16 (jump to main)
2. Execute ENT (enter function)
3. Execute IMM 42 (load 42 into AX)
4. Execute LEV (leave function)
5. Execute EXIT with AX=42

Actual behavior:
1. Model halts after 2 steps
2. AX remains 0
3. Exit code is 0

## Root Cause

The hand-crafted weights in `neural_vm/vm_step.py` have **multiple fundamental issues**:

### 1. Incorrect Initial State Generation
- Model generates PC=0x1000, AX=0x1000, SP=0x1000, BP=0x1000
- Should be PC=0, AX=0, SP=0x10000, BP=0x10000
- The weights don't encode proper initial state generation logic

### 2. Broken Instruction Execution
- Even with correct initial state provided, model can't execute instructions
- JSR, IMM, EXIT, LEV all fail to produce correct next states
- Model generates all zeros or random values for registers

### 3. Premature HALT Generation
- Model emits HALT after only 2 steps
- NEXT_HALT flag is being set incorrectly by L6_ffn or other layers

### 4. Missing/Incomplete Weight Configurations
- Contract violations show layers reading dimensions that aren't written:
  - `OPCODE_FLAGS` (actually written by L5_ffn, contract outdated)
  - `AX_CARRY_LO/HI` (actually written by L3_attn, contract outdated)
  - `ADDR_KEY` (actually written by L7/L8/L13_attn, contract outdated)
- Weight inspection shows many layers have incomplete contracts

## Why Tests Pass with use_speculator=True

When `use_speculator=True` (default):
- `FastLogicalVM` executes the actual program logic
- Returns correct results (42, 300, etc.)
- Neural model is only used for optional validation (validate_ratio=0.0 by default)
- **Draft VM does all the real work**

This is why `tests/run_1000_tests.py` passes 100% - it's testing Draft VM, not the neural model!

## What Would Be Needed to Fix

To make pure neural execution work, we would need to:

1. **Fix Initial State Generation**
   - Add FFN weights to generate PC=0, AX=0, SP=0x10000, BP=0x10000
   - Or add initial state to context (attempted but model still fails)

2. **Fix All Instruction Execution Logic**
   - Debug and fix weights for JSR, ENT, IMM, LEV, EXIT, ADD, SUB, etc.
   - Ensure proper register arithmetic (PC updates, SP updates, AX computation)
   - Fix multi-byte value handling (currently only byte 0 works)

3. **Fix HALT Logic**
   - Prevent premature HALT generation
   - Ensure HALT only triggers on EXIT instruction
   - Fix NEXT_HALT weight configuration in L6_ffn

4. **Remove Shadow State Corrections Dependency**
   - Current code in `run_vm.py` lines 296-361 applies corrections:
     - SP byte corrections for PSH, ADJ, binary ops
     - AX byte preservation (only byte 0 from model, bytes 1-3 from shadow)
   - These corrections mask the model's inability to compute correctly
   - Would need to ensure model produces correct values without corrections

5. **Comprehensive Testing**
   - Test each opcode individually
   - Verify register arithmetic at each step
   - Compare against FastLogicalVM ground truth

## Estimated Effort

This is a **major undertaking** requiring:
- Deep understanding of the 16-layer transformer weight structure
- Debugging thousands of hand-crafted weight values across multiple layers
- Potentially weeks of work to fix all instruction execution paths

## Recommendation

**Current state is "works as designed":**
- Use `use_speculator=True` (default) for fast, correct execution via Draft VM
- Neural model serves as optional validation/verification
- This architecture is intentional and functional

**To pursue pure neural execution:**
- Start with fixing single opcodes (e.g., IMM, EXIT)
- Build up layer-by-layer verification
- Consider whether the effort is worth it vs. current Draft VM approach

## Files Referenced

- `neural_vm/run_vm.py` - Runner with shadow state corrections
- `neural_vm/vm_step.py` - Hand-crafted weight configuration
- `src/speculator.py` - FastLogicalVM (working reference implementation)
- `tests/run_1000_tests.py` - Test suite (tests Draft VM, not neural model)

## Testing Commands

```bash
# Test pure neural (currently fails)
python test_pure_neural.py

# Test with speculator (works)
python -c "from src.baked_c4 import BakedC4Transformer; c4 = BakedC4Transformer(use_speculator=True); print(c4.run_c('int main() { return 42; }'))"

# Debug token generation
python test_token_generation.py

# Check corrections being applied
python test_check_corrections.py
```

## Status: 2026-03-26

Pure neural execution is **not functional** and would require significant work to fix. The speculative execution path (Draft VM) works correctly and is the recommended approach.
