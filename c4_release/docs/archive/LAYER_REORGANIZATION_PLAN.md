# Layer Reorganization Plan for Option D

## Current Understanding

The C4 VM operates on **32-bit values**. The existing efficient implementations in `neural_vm/alu/ops/` are correctly sized for 32-bit operations using the NIBBLE config (8 positions of 4-bit chunks).

## Efficient Operation Requirements

| Operation | Layers | Parameter Count |
|-----------|--------|-----------------|
| MUL | 7 | 10,846 |
| ADD | 3 | ~700 (estimated) |
| SUB | 3 | ~700 (estimated) |
| SHIFT | 2 | 5,624 (integrated ✓) |
| Bitwise (AND) | 2 | 486 |
| Bitwise (OR) | 2 | 510 |
| Bitwise (XOR) | 2 | 510 |

**Note:** Bitwise ops share Layer 0 (BitExtractFFN, 414 params), only Layer 1 differs.

## Current Layer Allocation

| Layer | Current Function | Units Used |
|-------|-----------------|------------|
| L0-L7 | Infrastructure (step, PC, registers, addressing) | Various |
| L8 | ADD/SUB lo nibble + carry/borrow detect | 753 |
| L9 | ADD/SUB hi nibble + comparisons | 1,296 |
| L10 | Carry apply + bitwise (AND/OR/XOR) | ~2,000 |
| L11 | MUL partial products | 4,096 |
| L12 | MUL combine | 4,096 |
| L13 | SHIFT (SHL/SHR) | ~~4,096~~ 5,624 (efficient ✓) |
| L14 | Memory address generation | ~500 |
| L15 | Memory lookup | ~500 |

## Challenge

We need to fit:
- ADD (3 layers) + SUB (3 layers) + MUL (7 layers) + Bitwise (2 layers) + SHIFT (2 layers) = **17 ALU layers**
- Memory operations: 2 layers
- Infrastructure: 8 layers (L0-L7)

**Total needed: 27 layers, but we only have 16!**

## Integration Approaches

### Approach A: Wrapper Method (SHIFT Model)

**What worked for SHIFT:**
- Run both layers sequentially in single FFN's forward() method
- Wrapper converts BD ↔ GenericE format
- Drop-in replacement for single transformer layer

**Can we apply this to other ops?**

✅ **Bitwise (2 layers):** Yes, similar to SHIFT
- Create `EfficientBitwiseFFN` wrapper
- Integrate into L10
- Savings: ~1,000 params

⚠️ **ADD/SUB (3 layers each):** Challenging
- Need to coordinate across L8-L9
- Could create wrapper that runs all 3 layers in forward()
- But adds complexity to single layer

❌ **MUL (7 layers):** Very difficult
- Running 7 layers sequentially in one forward() pass is inefficient
- Loses parallelism benefits of transformer architecture
- May not actually save parameters due to overhead

### Approach B: Layer Reallocation

**Idea:** Reorganize L8-L15 to accommodate multi-layer pipelines

**Option B1: Sequential Allocation**
```
L8:  ADD Layer 1 (raw sum)
L9:  ADD Layer 2 (carry lookahead)
L10: ADD Layer 3 (final) + SUB Layer 1 (raw diff)
L11: SUB Layer 2 (borrow lookahead)
L12: SUB Layer 3 (final) + Bitwise Layer 1 (extract)
L13: Bitwise Layer 2 (combine) + SHIFT (efficient wrapper)
L14: MUL Layer 1-2 (schoolbook + carry pass 1)
L15: MUL Layer 3-4 (carry pass 2-3)
L16: MUL Layer 5-7 (GenProp + Lookahead + Final) + Memory
```

**Problem:** We only have 16 layers, this needs 17+

**Option B2: Hybrid Wrapper + Reallocation**
```
L8:  EfficientADDSUBFFN (runs 3-layer ADD/SUB pipeline in wrapper)
L9:  Bitwise Layer 1 (extract) + Layer 2 (combine) in wrapper
L10: MUL Layer 1 (schoolbook)
L11: MUL Layer 2 (carry pass 1)
L12: MUL Layer 3 (carry pass 2)
L13: MUL Layer 4 (carry pass 3) + SHIFT (efficient, in wrapper)
L14: MUL Layer 5-6 (GenProp + Lookahead) in wrapper
L15: MUL Layer 7 (Final) + Memory operations
```

**Problem:** Still complex, mixing wrapped and non-wrapped approaches

### Approach C: Aggressive Wrapper Consolidation

**Idea:** Use wrapper approach for ALL multi-layer operations

```
L8:  EfficientADDSUBFFN (3-layer ADD, 3-layer SUB pipelines)
L9:  [Available for other ops or merged into L8]
L10: EfficientBitwiseFFN (2-layer AND/OR/XOR pipelines)
L11: EfficientMULFFN_Part1 (layers 1-3 of MUL)
L12: EfficientMULFFN_Part2 (layers 4-7 of MUL)
L13: EfficientShiftFFN (2-layer SHL/SHR) ✓ Done
L14: Memory address generation
L15: Memory lookup
```

**Pros:**
- Clean architecture
- All efficient ops integrated
- Fits in existing 16 layers

**Cons:**
- Running many layers in single forward() may be inefficient
- Loses some parallelism
- Complex implementation

## Recommended Strategy

**Phase 1: Low-hanging fruit (Wrapper Approach)**

1. ✅ **SHIFT** - Done, 31K params saved
2. **Bitwise** - 2 layers, similar to SHIFT
   - Create `EfficientBitwiseFFN` wrapper
   - Replace L10 bitwise ops
   - Estimated savings: ~1,000 params
   - Low risk

**Phase 2: Medium effort (ADD/SUB Wrapper)**

3. **ADD/SUB** - 3 layers each
   - Create `EfficientAddSubFFN` wrapper that runs both pipelines
   - Replace L8-L9 (may need to merge functionality)
   - Estimated savings: ~1,300 params
   - Medium risk - L8-L9 also handle comparisons

**Phase 3: High effort (MUL Reorganization)**

4. **MUL** - 7 layers
   - Options:
     a) Split into 2 wrappers (L11: layers 1-3, L12: layers 4-7)
     b) Reallocate L11-L15 to give MUL more space
     c) Run all 7 layers in single mega-wrapper (inefficient?)
   - Estimated savings: ~47K params
   - High risk - most complex operation

## Parameter Savings Breakdown

| Phase | Operation | Savings | Cumulative | Effort | Risk |
|-------|-----------|---------|------------|--------|------|
| Done | SHIFT | 31,240 | 31,240 (78%) | 4h | Low |
| 1 | Bitwise | ~1,000 | 32,240 (81%) | 2-4h | Low |
| 2 | ADD/SUB | ~1,300 | 33,540 (84%) | 4-8h | Med |
| 3 | MUL | ~47,000 | 80,540 (>100%?) | 8-16h | High |

**Note:** Phase 3 savings seem too high - need to verify current MUL param count.

## Next Steps

1. Verify current parameter counts for all operations
2. Decide on approach for each operation
3. Implement Phase 1 (Bitwise) as proof of concept
4. Assess results before proceeding to Phase 2-3

## Open Questions

1. Can we run 7 MUL layers efficiently in a wrapper?
2. What's the performance impact of sequential layer execution?
3. Should we use transformer parallelism or wrapper efficiency?
4. Can ADD/SUB share a wrapper without breaking comparisons?
