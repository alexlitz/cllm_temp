# PC Addressing Issue Summary

## Date
2026-03-26

## Current Status

✅ **L4 Attention**: Working correctly - relays OUTPUT from PC marker to AX marker within same step
✅ **L5 Fetch**: Working correctly with PC=0 - fetches opcode from bytecode address 0
✅ **Opcode Decode**: OP_IMM = 5.0 (correct!)
✗ **IMM Value Fetch**: FETCH_LO has wrong value (not fetching immediate 42)
✗ **PC Output**: Neural VM outputs PC=5, but DraftVM expects PC=7

## Root Cause: Two Different Address Spaces

### 1. Bytecode Address Space (for fetching instructions)
Used by neural VM's L5 attention to fetch from code section:
- First opcode: **address 0** (context position 1)
- First immediate byte: **address 1** (context position 2)
- ...

ADDR_KEY encoding: nibble-based
- Address 0: ADDR_KEY[0]=1, ADDR_KEY[16]=1 (nibbles: lo=0, hi=0)
- Address 1: ADDR_KEY[1]=1, ADDR_KEY[16]=1 (nibbles: lo=1, hi=0)

### 2. Token Position Value (for output)
Used by DraftVM as PC value in output tokens:
- Formula: `PC = idx * 5 + 2`
- Instruction 0: PC goes from 2 → 7 (before → after execution)
- Instruction 1: PC goes from 7 → 12

This is what appears in draft tokens as a **byte value**, not an address.

## The Mismatch

**Neural VM currently**:
1. Fetches from bytecode address 0 ✓
2. Increments by 5 → bytecode address 5
3. Outputs nibble-encoded value 5

**DraftVM expects**:
1. (Fetches from internal bytecode at instruction index 0)
2. (Increments instruction index 0 → 1)
3. **Outputs PC = idx * 5 + 2 = 1 * 5 + 2 = 7**

## Possible Solutions

### Option A: Separate Fetch Address and Output PC
Neural VM maintains TWO PC values:
- **Fetch address**: Used by L5 to fetch instructions (0, 5, 10, ...)
- **Output PC**: Computed from instruction index for output (2, 7, 12, ...)

This requires:
- L5 fetch uses one address space (current implementation)
- Output layer converts to DraftVM's PC formula

### Option B: Use DraftVM PC Formula Throughout
Neural VM uses DraftVM's PC formula everywhere:
- Initial PC = 2 (not 0)
- Fetch at address `(PC - 2) / 5 * [bytes_per_instruction]`
- Increment PC by 5

This matches DraftVM but requires complex address translation in L5.

### Option C: Fix Context Layout
Change `build_context` to match DraftVM's expectations:
- Each instruction is 5 bytes (not 8)
- Remove padding
- Addresses align with DraftVM's PC formula

This requires changing context building and may break existing code.

## Recommendation

**Option A** seems cleanest: keep bytecode addressing for fetch (working now), but add a conversion layer to compute output PC.

The neural VM should:
1. Track instruction index internally
2. Use fetch address = `idx * [bytes_per_instruction]` for L5
3. Compute output PC = `idx * 5 + 2` for final output

Or simpler: Since the neural VM currently has fetch address = idx * 5, we just need to add 2:
- Fetch address: 0, 5, 10, 15, ...
- Output PC: 2, 7, 12, 17, ... (add 2)

## Next Steps

1. Understand how many bytes per instruction in the context (check if padding is included in ADDR_KEY)
2. Determine correct increment value for next instruction fetch
3. Add offset of +2 to final PC output to match DraftVM formula
4. Debug why FETCH_LO isn't getting immediate value 42

---

**Key Insight**: DraftVM's PC is not a bytecode address - it's a computed token position value using formula `idx * 5 + 2`. The neural VM must output this value, not the bytecode fetch address.
