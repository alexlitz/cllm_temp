# Nibble-Based Execution: Findings and Solutions

## Problem Discovery

When testing the nibble-based weight compiler, we discovered that operations failed when carry propagation was needed:
- ✅ 2 + 3 = 5 (no carry) - worked
- ❌ 100 + 200 = 166 (with carry) - failed, expected 300

### Root Cause

The `nibble_weight_compiler.py` only generates **Layer 1** weights, which compute raw sums without carry propagation. The comment in the code explicitly states:

```python
"""This is simplified version - full version needs 3-layer pipeline
with carry lookahead (see alu/ops/add.py).
"""
```

## Architecture Requirements

Multi-nibble arithmetic operations (ADD, SUB, MUL, DIV) require a **3-layer pipeline**:

### Layer 1: Raw Computation + Generate
- Computes `RAW_SUM = A + B` per nibble
- Generates `CARRY_OUT` flags: `step(A + B >= base)`
- Generates `PROPAGATE` flags: `step(A + B == base - 1)`
- **6 hidden units per position** (3 cancel pairs)

### Layer 2: Carry Lookahead
- Parallel prefix computation: `C[i] = G[i-1] OR (P[i-1] AND G[i-2]) OR ...`
- Propagates `CARRY_OUT[i]` → `CARRY_IN[i+1]` across all positions
- Uses **FlattenedFFN** (cross-position FFN), NOT attention
- **~60 hidden units total** for 8 positions

### Layer 3: Finalize
- Computes `RESULT = (RAW_SUM + CARRY_IN) mod base`
- **4 hidden units per position** (2 cancel pairs)

## Solution: ALUWeightExtractor

Created `alu_weight_extractor.py` to bridge the real ALU implementations (`alu/ops/add.py`) with AutoregressiveVM:

### Key Features
1. **Extracts weights from GenericPureFFN** (per-position operations)
2. **Extracts weights from GenericFlattenedFFN** (cross-position operations)
3. **Transforms to AutoregressiveVM format**:
   - GenericPureFFN: `[batch, num_pos, dim]` → `[batch, seq, d_model]`
   - Per-position weights expanded to flattened 1280-dim space

### Test Results
All tests pass with extracted weights:
```
✅ 2 + 3 = 5 (no carry)
✅ 10 + 20 = 30 (no carry)
✅ 100 + 200 = 300 (WITH carry) ← Previously failed!
✅ 255 + 1 = 256 (edge case)
```

## Layer Allocation Strategy

### Current Allocation
```
Layers 0-8:  Instruction fetch, decode, setup
Layer 9:     PRIMARY_ALU (Layer 1 of 3-layer pipeline)
Layer 10:    Carry lookahead (Layer 2 of 3-layer pipeline)
Layer 11:    Finalize (Layer 3 of 3-layer pipeline)
Layers 12-13: Memory operations
Layers 14-15: Output generation
```

### Implications
- **All arithmetic ops** (ADD, SUB, MUL, DIV, MOD) share the same 3-layer pipeline
- Opcode gating ensures only the correct operation executes
- No unit offset needed within the 3-layer pipeline (each layer processes all positions)

## Carry Propagation: FFN vs Attention

**IMPORTANT**: The original `carry_ffn.py` (in archive) used attention for carry propagation.
The current implementation uses **FlattenedFFN with parallel prefix computation**, which is more efficient.

### Why FFN Instead of Attention?
1. **Parallel prefix algorithm**: Computes all carries in O(log N) depth
2. **Fixed routing**: Position i always depends on positions i-1, i-2, ..., 0
3. **No softmax needed**: Direct combinational logic via step functions
4. **Fewer parameters**: ~60 hidden units vs attention's Q/K/V matrices

## Next Steps

### 1. Update Weight Loader
Modify `weight_loader.py` to:
- Use `ALUWeightExtractor` for ADD, SUB, and other arithmetic operations
- Load weights into layers 9-11 (3-layer pipeline)
- Remove the simplified nibble_weight_compiler for arithmetic ops

### 2. Extend to Other Operations
Extract weights for:
- **SUB**: Similar 3-layer structure with borrow propagation
- **MUL**: More complex, may need different layer count
- **DIV/MOD**: Complex, uses iterative refinement

### 3. Comparison Operations
Test if comparison ops (EQ, NE, LT, GT, LE, GE) work correctly:
- These may not need carry propagation
- Single-layer execution might suffice

### 4. Bytecode Runner
Create end-to-end executor that:
- Runs full VM forward pass through all 16 layers
- Handles multi-step programs
- Integrates with existing test suite

### 5. Test Suite Integration
Run the 1000+ functional test suite:
- Verify all opcodes work correctly
- Compare with FastLogicalVM baseline
- Measure accuracy and performance

## Key Insights

1. **Single FFN layer is insufficient** for multi-nibble arithmetic with carries
2. **Carry propagation requires cross-position communication** (FlattenedFFN)
3. **Real ALU implementations** (alu/ops/*.py) are production-ready and tested
4. **Nibble-based architecture** works when properly structured as 3-layer pipeline
5. **Unit offset strategy** is still valuable for later layers with multiple opcodes

## Files Modified/Created

### Created
- `alu_weight_extractor.py`: Weight extraction from chunk-generic ALU
- `test_real_add.py`: Test 3-layer ADD with GenericFFN directly
- `test_extracted_add.py`: Test extracted weights in AutoregressiveVM
- `test_multilayer_add.py`: Test multi-layer execution (revealed issues)
- `test_multi_opcode.py`: Test multiple opcodes with unit offsets
- `test_isolated_add.py`: Test ADD in isolation (revealed carry issue)
- `test_unit_allocation.py`: Debug unit offset allocation

### Modified
- `weight_loader.py`: Added unit offset allocation strategy
- `nibble_weight_compiler.py`: Added unit_offset parameter support
- `multi_operation_compiler.py`: Added unit_offset parameter support
- `opcode_nibble_integration.py`: Added unit_offset parameter support

## References

- `alu/ops/add.py`: Real 3-layer ADD implementation
- `alu/chunk_config.py`: Chunk-generic configuration (NIBBLE, BYTE, etc.)
- `alu/ops/common.py`: GenericPureFFN and GenericFlattenedFFN base classes
- `archive/carry_ffn.py`: Old attention-based carry propagation (deprecated)
