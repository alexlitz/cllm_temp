# IMM Neural Implementation - Complete Analysis & Status

## Summary

**Goal**: Make IMM instruction execute neurally (without runner fallback)

**Current Status**: PARTIALLY WORKING
- Exit code 42: ✓ (via runner fallback)
- Neural execution: ✗ (model generates AX=0)

## Fixes Applied

### Fix 1: OP_* Flags in Opcode Byte Embeddings ✓

**Problem**: Opcode bytes (0-255) lacked OP_* flags in their embeddings
**Location**: `neural_vm/vm_step.py` lines 1519-1559
**Fix**: Added code to set OP_IMM=1.0 for byte 1 (IMM opcode), OP_EXIT=1.0 for byte 38 (EXIT), etc.

```python
from .embedding import Opcode
for opcode_value, op_dim in [
    (Opcode.IMM, BD.OP_IMM),
    (Opcode.EXIT, BD.OP_EXIT),
    ...
]:
    embed[opcode_value, op_dim] = 1.0
```

**Verification**: ✓ Confirmed byte 1 now has OP_IMM=1.0 in embedding

### Fix 2: SP/BP Default Unit Marker Gates ✓

**Problem**: Layer 3 FFN units 4 & 5 (SP/BP carry propagation) fired at ALL register positions, corrupting AX
**Location**: `neural_vm/vm_step.py` lines 2434-2454
**Fix**: Added marker gates to restrict units to SP/BP positions only

```python
# Unit 4 (SP default)
ffn.W_gate[unit, BD.MARK_SP] = 1.0  # Only fire at SP positions

# Unit 5 (BP default)
ffn.W_gate[unit, BD.MARK_BP] = 1.0  # Only fire at BP positions
```

**Verification**: ✓ Confirmed gates present, OUTPUT_LO[1]=0 at AX positions

## Remaining Issue: Layer 5 Opcode Fetch Broken ✗

**Problem**: Layer 5 attention not populating OPCODE_BYTE_LO/HI at PC marker

### Expected Flow (from code comments)

1. **Layer 5 Attention Head 1** (`_set_layer5_fetch`, line 2680-2718):
   - Query: PC marker looks for ADDR_KEY matching current PC value
   - Key: CODE section bytes with ADDR_KEY
   - Value: Copies EMBED_LO/HI (opcode nibbles) from matched CODE byte
   - Output: Writes to OPCODE_BYTE_LO/HI at PC marker

2. **Layer 5 FFN Opcode Decode** (`_set_opcode_decode_ffn`, line 2930-2939):
   - Reads: OPCODE_BYTE_LO[1] AND OPCODE_BYTE_HI[0] (for IMM)
   - Writes: OP_IMM=5.0 at PC marker (if both nibbles match)

3. **Layer 5 Attention Head 5** (line 3187-3221):
   - Relays OP_IMM from PC marker → AX marker

4. **Layer 6 FFN IMM Routing** (line 3262-3274):
   - Condition: OP_IMM AND MARK_AX
   - Action: Route FETCH → OUTPUT

### Current Observations

**ADDR_KEY Injection**: ✓ Working
- IMM opcode at position 1 has ADDR_KEY[2]=1.0 (correct for PC=2 with PC_OFFSET=2)

**OPCODE_BYTE Population**: ✗ Broken
- OPCODE_BYTE_LO/HI remain 0.000 at PC marker through all layers
- This breaks the entire chain

**OP_IMM Appearance**: ⚠️ Appears in Layer 5 but value is wrong
- OP_IMM jumps from 0.000 → 5.000 in Layer 5
- But this might be a different mechanism (first-step decode)

## Root Cause Hypothesis

Layer 5 attention Head 1 is not successfully fetching the opcode byte from CODE to PC marker. Possible reasons:

1. **ADDR_KEY matching not working**: Attention query/key match fails
2. **PC value wrong at PC marker**: Query doesn't have correct PC to match ADDR_KEY=2
3. **VALUE weights wrong**: V doesn't copy EMBED_LO/HI correctly
4. **OUTPUT weights wrong**: W_o doesn't write to OPCODE_BYTE_LO/HI correctly
5. **HEAD routing issue**: Wrong head index or MoE routing problem

## Diagnostic Tests Run

1. ✓ `verify_imm_fix.py`: Confirmed marker gates, found AX=0 issue
2. ✓ `debug_imm_neural_pathway.py`: Traced layers, found OP_IMM=0 at AX
3. ✓ `debug_opcode_relay.py`: Found OP_IMM appears in Layer 5 (but value 5.0)
4. ✓ `debug_opcode_byte.py`: Confirmed OPCODE_BYTE_LO/HI not populated

## Next Steps

### Option A: Debug Layer 5 Attention Fetch (Complex)
- Trace Layer 5 attention Head 1 Q/K/V/O matrices
- Check ADDR_KEY matching mechanism
- Verify PC value propagation to PC marker
- Fix attention weights if broken

### Option B: Simplify to Direct Embedding Approach (Simpler)
- Remove opcode fetch mechanism entirely
- Rely on OP_* flags in embeddings propagating naturally
- Adjust Layer 5 FFN to read from OP_* dims at PC marker (already set in embeddings)
- Skip OPCODE_BYTE intermediate representation

### Option C: Use First-Step Decode Path (Medium)
- Layer 5 already has "first-step decode at PC marker" (line 2930-2939)
- This writes OP_IMM=5.0 directly to PC marker
- Verify this path works
- Extend it to handle all steps (not just first step)

## Recommendation

**Option B: Direct Embedding Approach** is cleanest:

1. Remove dependency on OPCODE_BYTE_LO/HI
2. OP_* flags already set in opcode byte embeddings (Fix 1)
3. Layer 5 attention should relay OP_* from CODE bytes to PC marker
4. Then existing relay (PC → AX) and routing (AX) will work

**Implementation**:
- Add Layer 4 or Layer 5 attention head that copies OP_* flags from CODE to PC marker
- Use ADDR_KEY matching (same mechanism as immediate fetch)
- Query at PC marker, match ADDR_KEY, copy OP_IMM/OP_EXIT/etc. from CODE byte

This bypasses the broken OPCODE_BYTE mechanism and uses the embeddings we just fixed.

## Files Modified

1. `neural_vm/vm_step.py`:
   - Lines 1519-1559: Added OP_* flags to opcode byte embeddings
   - Lines 2434-2454: Added marker gates to SP/BP default units

## Files Created

1. `IMM_RUNNER_FALLBACK_ANALYSIS.md`: Documents runner fallback mechanism
2. `verify_imm_fix.py`: Verification test script
3. `debug_imm_neural_pathway.py`: Layer-by-layer diagnostic
4. `debug_opcode_relay.py`: Opcode flag propagation check
5. `debug_opcode_byte.py`: OPCODE_BYTE population check
6. `IMM_NEURAL_IMPLEMENTATION_STATUS.md`: This file

## Timeline

- **Session Start**: Found IMM instruction produces wrong exit code (65536 vs 42)
- **Root Cause 1**: Layer 3 SP/BP units corrupting AX → Fixed with marker gates
- **Root Cause 2**: Opcode bytes lack OP_* flags → Fixed in embeddings
- **Current Block**: Layer 5 opcode fetch not working → Needs fix (Option B recommended)
