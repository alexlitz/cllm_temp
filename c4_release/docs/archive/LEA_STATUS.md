# LEA Opcode Status

## Current State
- **Accuracy**: ~88.6% (strict validation)
- **Status**: Not passing simple test (predicts 1 instead of 8)

## Root Causes Identified

### 1. Dimension Amplification
- OP_LEA amplified from ~2.4 to ~7.4 through layers
- FETCH amplified from 1.0 to 2.0 (multiple attention heads writing)
- ALU amplified from ~1.0 to ~14.0 (Layer 6 FFN writes)

### 2. Architecture Complexity
LEA requires coordination across 7+ layers:
- Layer 3: Would initialize ALU (but OP_LEA doesn't exist yet)
- Layer 5: Decodes OP_LEA at PC marker
- Layer 5 Head 3: Dual-writes FETCH to PC and AX markers
- Layer 6 Head 5: Relays OP_LEA from PC to AX
- Layer 6 FFN: Initializes ALU with BP default for first step
- Layer 7 Head 1: Copies BP OUTPUT to ALU (subsequent steps)
- Layer 8/9: LEA-specific ADD units (read FETCH instead of AX_CARRY)

### 3. First-Step Special Casing
- First step has no previous BP marker to attend to
- Requires separate code path to initialize ALU with BP default
- Timing issues: OP_LEA created in L5 but needed in L3

## Changes Made

### Successful Changes
1. ✓ Created separate LEA ADD units in L8/9 that read from FETCH
2. ✓ Layer 5 head 3 dual-writes FETCH to PC and AX markers
3. ✓ Layer 6 FFN initializes ALU for first-step LEA

### Failed Approaches
1. ✗ Copying FETCH to AX_CARRY (sigmoid gating causes 0.5 leakage)
2. ✗ Cancellation pattern (amplification overwhelms cancellation)
3. ✗ Sharp sigmoid gating (dimension amplification breaks thresholds)
4. ✗ Threshold adjustment (fragile, breaks with small value changes)

## Next Steps Options

### Option A: Continue Neural Debugging
- Normalize ALU/FETCH/OP_LEA values to prevent amplification
- Add explicit dimension clearing before writes
- Estimated time: 4-8 hours
- Risk: High (deep architectural issues)

### Option B: Runner Correction
- Add LEA special handling in batch_runner.py
- Similar to existing JMP/BZ corrections
- Estimated time: 30-60 minutes
- Risk: Low (well-understood pattern)

### Option C: Document and Move On
- Test other opcodes (ADD, SUB, MUL, DIV, MOD, bitwise, shifts)
- Return to LEA if other opcodes reveal patterns
- Estimated time: Immediate
- Risk: None (deferred work)

## Recommendation
**Option C**: Test other opcodes first. LEA's 88.6% accuracy suggests the architecture mostly works, but edge cases and amplification issues make it fragile. Other opcodes may pass at higher rates and reveal patterns that help fix LEA more efficiently.

## Test Command
```bash
python test_lea_simple.py  # Currently fails: predicts 1, expects 8
```
