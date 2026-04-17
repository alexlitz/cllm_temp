# Path B: Multi-Layer Integration - Implementation Plan

## Goal
Reduce parameters from 141K → ~50K (64% reduction) by using efficient multi-layer ops from `neural_vm/alu/ops/`.

## Strategy
Spread each efficient operation across consecutive transformer layers, using residual connections to pass intermediate results.

## Layer Reorganization Plan

### Current Layer Allocation
```
L0-L7:   Setup, fetch, operand gather
L8-L9:   ADD/SUB (lookup tables)
L10:     Bitwise + CMP + DIV/MOD
L11-L12: MUL (lookup tables)
L13:     SHIFT (lookup tables) + MEM addr gather
L14-L15: MEM generation
```

### Proposed Reorganization
```
L0-L7:   [Unchanged] Setup, fetch, operand gather

L8:      ADD stage 1: Raw sum (30 params)
L9:      ADD stage 2: Carry lookahead (264 params)
         SUB stage 1: Raw diff (31 params)
L10:     ADD stage 3: Final correction (34 params)
         SUB stages 2-3: Lookahead + correction (297 params)
         Bitwise stage 1: Bit extraction (414 params each)

L11:     Bitwise stage 2: Combine + cleanup (96 params each)
         MUL stage 1: Schoolbook partial products (264 params)
L12:     MUL stage 2: Carry extract pass 1 (8960 params)
L13:     MUL stage 3: Carry extract pass 2 (748 params)
         SHIFT stage 1: Precompute (128 params SHL+SHR)
L14:     MUL stage 4: Carry extract pass 3 (184 params)
         SHIFT stage 2: Select (2684 params SHL+SHR)
         MEM addr gather [keep existing]
L15:     MUL stages 5-7: G/P + Lookahead + Correction (690 params)
         MEM generation [adjust as needed]
```

### Key Changes
1. **L8-L10**: ADD/SUB spread across 3 layers (current: 2 layers)
2. **L11-L15**: MUL spread across 5 layers (current: 2 layers)
3. **L13-L14**: SHIFT spread across 2 layers (current: 1 layer)
4. **L14-L15**: Keep MEM operations, share space with tail end of MUL/SHIFT

## Implementation Order

### Phase 1: SHIFT (Proof-of-Concept)
**Simplest multi-layer integration**

**L13 FFN**: Precompute stage
- Input: ALU_LO[0-15], ALU_HI[0-15] (one-hot nibbles)
- Process: Convert to scalar, compute sub-shifts for r=1,2,3
- Output: Write to temp dimensions (SHIFT_S1, SHIFT_C1, etc.)
- Units: 28 (SHL) + 28 (SHR) = 56 units

**L14 FFN**: Select stage
- Input: Read temp dimensions from L13, shift amount from AX_CARRY_LO
- Process: Select and route based on shift amount
- Output: Write to OUTPUT_LO/HI
- Units: 528 (SHL) + 528 (SHR) = 1056 units

**Total SHIFT**: 1,112 units vs current 4,096 (73% reduction, saves 31K params)

**Success Criteria**:
- All SHIFT tests pass
- Parameter count reduces by ~31K
- No regression in other operations

### Phase 2: ADD/SUB (3-layer pipeline)
**After SHIFT proves pattern works**

**L8**: Raw sum/diff + initial carry detection
**L9**: Carry lookahead propagation
**L10**: Final correction with carry application

**Total**: ~660 units vs current 1,024 (35% reduction, saves ~400 params)

### Phase 3: MUL (5-7 layer pipeline)
**Biggest savings potential**

**L11**: Schoolbook multiplication partial products
**L12-L14**: Three passes of carry extraction
**L15**: Generate/propagate, lookahead, final correction

**Total**: ~10,846 units vs current 8,192... wait, this is MORE units!

**Issue**: The efficient MUL saves params through better organization, not fewer units.
Need to recheck the actual parameter count calculation.

### Phase 4: Bitwise (2-layer)
**L10**: Bit extraction
**L11**: Boolean combine + cleanup

**Total**: ~500 units each for AND/OR/XOR vs current 1,536 total

## Temporary Dimension Allocation

We need temporary dimensions to pass data between layers:

```
Current BD dimensions used: 0-449
Available for temps: 450-511 (62 dimensions)

Allocations:
SHIFT_TEMPS: 450-461 (12 dims for s_r and c_r slots)
MUL_TEMPS: 462-490 (29 dims for partial products)
ADD_TEMPS: 491-495 (5 dims for carry propagation)
BITWISE_TEMPS: 496-505 (10 dims for bit extraction)
```

## Integration Approach

For each operation:

1. **Build efficient layers** using existing `neural_vm/alu/ops/` code
2. **Extract weight patterns** from GenericE format
3. **Transform to BD format**:
   - Map GenericE slot indices to BD dimension indices
   - Convert scalar operations to one-hot operations where needed
   - Preserve the computational flow
4. **Bake weights** into target FFN modules
5. **Test** with existing opcode tests

## Code Structure

Create `neural_vm/path_b_integration.py`:
```python
def integrate_shift_stage1(ffn, S, BD):
    """Set L13 FFN weights for SHIFT precompute."""
    # Build GenericE version
    shl_layers = build_shl_layers(NIBBLE, 23)
    stage1 = shl_layers[0]

    # Extract and transform weights
    transform_precompute_to_bd(stage1, ffn, S, BD)

def integrate_shift_stage2(ffn, S, BD):
    """Set L14 FFN weights for SHIFT select."""
    # Build GenericE version
    shl_layers = build_shl_layers(NIBBLE, 23)
    stage2 = shl_layers[1]

    # Extract and transform weights
    transform_select_to_bd(stage2, ffn, S, BD)
```

## Testing Strategy

1. **Unit tests** for each stage independently
2. **Integration tests** across layer boundaries
3. **Opcode tests** to verify correctness
4. **Full program suite** (1000 programs)
5. **Parameter count verification**

## Risk Mitigation

1. **Start simple**: SHIFT first (2 layers only)
2. **Incremental**: One operation at a time
3. **Reversible**: Keep current implementation alongside for comparison
4. **Well-tested**: Verify each integration before moving to next

## Expected Outcome

| Operation | Current Params | Efficient Params | Savings |
|-----------|---------------|------------------|---------|
| SHIFT     | 36,864        | 5,624           | 31,240  |
| ADD/SUB   | ~1,024        | 656             | 368     |
| MUL       | 57,344        | 10,846          | 46,498  |
| Bitwise   | ~1,536        | 1,506           | 30      |
| **TOTAL** | **96,768**    | **18,632**      | **78,136 (81%)** |

Combined with other ops: **141K → ~50K (64% reduction)**

## Next Steps

1. Implement SHIFT integration infrastructure
2. Test SHIFT proof-of-concept
3. Expand to ADD/SUB
4. Expand to MUL
5. Final integration and testing
