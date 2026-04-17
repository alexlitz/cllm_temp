# 100% Neural Execution Status

## Fixes Applied This Session

### 1. L5 ALiBi Fix (vm_step.py line 1700)
- **Issue**: ALiBi distance penalty (-0.1 × distance) was destroying bytecode attention
- **Example**: For bytecode at position 2, distance ~1690, penalty = -169 overwhelming +150 address match score
- **Fix**: Set `attn5.alibi_slopes.fill_(0.0)` to disable ALiBi for L5
- **Result**: L5 can now attend to distant bytecode positions

### 2. Context Window Fix (run_vm.py line 339)
- **Issue**: 512-token window excluded bytecode prefix
- **Fix**: Increased `max_context_window = 2048`
- **Result**: Bytecode (positions 0-N) now included in attention window

### 3. generate_next Window Fix (run_vm.py line 348)
- **Issue**: `generate_next()` had internal 512 window override
- **Fix**: Pass `max_context_window=2048` parameter
- **Result**: Full context used for token generation

## Opcode Status

### ✅ Working (100% Neural)
| Opcode | Test | Notes |
|--------|------|-------|
| IMM | `IMM 42; EXIT` → 42 | Load immediate works |
| IMM 0 | `IMM 0; EXIT` → 0 | Zero case works |
| IMM 255 | `IMM 255; EXIT` → 255 | Large values work |
| Sequential IMM | Multiple IMM; EXIT | Last value wins |

### ❌ Broken
| Opcode | Issue | Root Cause |
|--------|-------|------------|
| JMP | PC doesn't jump | Address conversion uses nibble+2 instead of INDEX×8+2 |
| JSR | Same as JMP | Same multiplication issue |
| PSH | Stack value not written | Memory write system broken |
| ADD | Returns just AX, not pop+AX | Depends on broken stack read |
| SUB | Wrong result | Depends on broken stack read |
| MUL | Returns 0 | Depends on broken stack read |
| LI | Doesn't load from address | Memory read not working |

## Root Cause Analysis

### JMP/JSR Address Conversion Bug
The JMP target address should be: `PC = INDEX × 8 + PC_OFFSET`

Current implementation (vm_step.py ~line 3988):
```python
new_k = (k + 2) % 16  # Add PC_OFFSET=2 to nibble
```

This just adds 2 to the nibble index, which is NOT the same as multiplying by 8!

Example:
- JMP 4: INDEX=4
- Expected PC: 4×8+2 = 34 = 0x22
- Actual: nibble 4+2=6 → 0x06 (WRONG!)

### Memory Operation Bug
PSH/LI/ADD depend on the memory subsystem:
1. PSH should write AX to *SP
2. STACK0 should contain the value at *SP
3. ADD should read from stack via STACK0 or L15 memory lookup

Current behavior:
- PSH decrements SP correctly (0x10000 → 0x0FFF8)
- But STACK0 shows garbage (154) instead of pushed value (10)
- ADD just keeps current AX (32), doesn't add stack value

The memory write/read attention (L8 for STACK0, L15 for general memory) may have issues with:
1. Address matching in long autoregressive contexts
2. Memory value propagation across steps

## Recommendations

### Short-term (Handlers)
Keep handlers for opcodes that require memory or control flow:
- JMP, JSR, BZ, BNZ: Control flow needs address conversion
- PSH, LI, SI, LC, SC: Memory operations need fixes
- ADD, SUB, MUL, etc.: Binary ops that pop from stack

### Medium-term (Fixes Needed)
1. **JMP/JSR**: Implement proper INDEX×8+2 conversion in L6 FFN
   - Requires cross-nibble bit shifting (shift left 3 bits)
   - Complex but doable with more FFN units

2. **Memory Operations**: Debug L8 (STACK0) and L15 (memory lookup)
   - Check if memory tokens are being written correctly
   - Verify address matching works for stack addresses
   - May need ALiBi adjustments for memory attention layers

### What Works Now
Simple programs using only IMM and EXIT can run 100% neurally:
```
IMM 42
EXIT
```
Returns 42 correctly!
