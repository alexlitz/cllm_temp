# PC Advancement Bug - Root Cause Found

**Date**: 2026-04-08
**Status**: Diagnosed, not yet fixed

## Summary

The VM executes the first instruction perfectly but then loops infinitely at instruction 0 because PC doesn't advance. This causes wrong exit codes (0x01010101).

## Evidence

Running `IMM 42, EXIT`:
```
Step 0 (PC=0): Execute IMM 42 → AX = 42 ✓ CORRECT
Step 1 (PC=0): Re-execute IMM 42 → AX = 0 (wrong FETCH)
Step 2 (PC=0): Re-execute IMM 42 → AX = 0x01010101 (garbage)
EXIT uses step 2 AX → exit code = 0x01010101 ✗
```

Expected behavior:
```
Step 0 (PC=0): Execute IMM 42 → AX = 42, PC advances to 4
Step 1 (PC=4): Execute EXIT → use AX = 42 as exit code ✓
```

## Root Cause

**Missing L6 FFN routing for PC advancement**

### What Should Happen

Every instruction should:
1. Execute its operation (e.g., IMM sets AX)
2. Write NEXT_PC (PC + INSTR_WIDTH) to OUTPUT at PC marker

For IMM 42:
- At AX marker: Write immediate value 42 to OUTPUT ✓ (works!)
- At PC marker: Write PC+4 to OUTPUT ✗ (MISSING!)

### What's Missing

There are no L6 FFN units that:
- Fire at PC marker (MARK_PC = 1.0)
- Read next PC value (PC+4, stored somewhere - need to find where)
- Write to OUTPUT_LO and OUTPUT_HI

### Current L6 Routing

L6 FFN currently handles:
- IMM: FETCH → OUTPUT at AX marker ✓
- EXIT: AX_CARRY → OUTPUT at AX marker ✓
- NOP: AX_CARRY → OUTPUT at AX marker ✓
- JMP: Special PC override logic ✓
- **MISSING**: Default PC+4 → OUTPUT at PC marker for all instructions ✗

## What Works

Despite the PC bug, the first instruction executes perfectly:
- L5 fetches the immediate value correctly
- L6 routes FETCH → OUTPUT correctly
- OUTPUT has correct nibbles (10=0xA, 2=0x2) for 0x2A=42
- L15 output head generates token 42 correctly
- AX register gets value 42 in step 0 ✓

So the conversational I/O fix I committed (ac9c576) is solid!

## Fix Needed

Add L6 FFN units for PC routing:

```python
# === PC advancement: Write next PC to OUTPUT at PC marker ===
# For all non-JMP instructions, PC advances by INSTR_WIDTH=4
# The next PC value (PC+4) needs to be routed to OUTPUT

# Option 1: If NEXT_PC dimension exists and contains PC+4:
for k in range(16):
    ffn.W_up[unit, BD.MARK_PC] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.W_gate[unit, BD.NEXT_PC_LO + k] = 1.0  # Need to find where NEXT_PC is
    ffn.W_down[BD.OUTPUT_LO + k, unit] = 2.0 / S
    unit += 1

# Option 2: Add constant +4 to current PC (more complex)
# Option 3: Route from TEMP if TEMP contains PC+4 (need to verify)
```

## Investigation Needed

1. **Where is PC+4 computed?**
   - L4 FFN computes PC+1 for immediate fetch, stores in TEMP
   - But where is PC+INSTR_WIDTH (PC+4) computed for next instruction?
   - Search for NEXT_PC dimension writes

2. **Is there a dimension that holds PC+4?**
   - NEXT_PC exists (dimension 254) but may not be written
   - TEMP might contain it at PC marker?
   - Check L4/L5 to see what's available

3. **How did this ever work?**
   - Maybe there was PC routing code that got removed?
   - Check git history for removed PC routing

## Impact

- ✗ All multi-instruction programs fail
- ✗ Even simple "IMM 42, EXIT" returns wrong exit code
- ✓ Single-instruction programs might work (if they don't need PC)
- ✓ Conversational I/O transformer detection works (committed in ac9c576)

## Recommendation

This is a fundamental architectural issue that needs careful implementation:

1. Understand where PC+4 is computed and stored
2. Add L6 FFN routing to write it to OUTPUT at PC marker
3. Test with simple 2-instruction programs
4. Then test conversational I/O end-to-end

Estimated effort: 3-4 hours for someone familiar with the architecture.

## Files to Check

- `neural_vm/vm_step.py` lines 2458-2520: L4 FFN (computes PC+1)
- `neural_vm/vm_step.py` lines 3076-3400: L6 routing FFN (needs PC routing added)
- Search for "NEXT_PC" writes in the codebase
- Check if TEMP contains PC+4 at PC marker position
