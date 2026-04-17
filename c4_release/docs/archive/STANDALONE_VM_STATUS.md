# Standalone Neural VM - Implementation Status

## Summary

Work has begun on making the neural VM fully standalone (Option 2 from VANILLA_TRANSFORMER_PLAN.md). Helper functions have been created and partially integrated, but full implementation requires careful completion.

## Completed Work

### 1. Helper Functions Created ✅
**Location**: vm_step.py lines 1819-1865

Two helper functions were added:

```python
def _route_marker_passthrough_16(ffn, marker_dims, source_lo, source_hi, unit, S, BD, T=0.5):
    """Identity carry: at marker_dim, pass source_lo/hi through to OUTPUT."""
    # Routes EMBED → OUTPUT at specified markers
    # Supports multiple markers (list/tuple)
    # Returns updated unit index

def _route_ax_default_passthrough_16(ffn, unit, S, BD, T=0.5):
    """Default AX identity passthrough: AX_CARRY → OUTPUT at MARK_AX."""
    # Routes AX_CARRY → OUTPUT at AX marker
    # Always active (T=0.5), opcodes override with T=4.0
    # Returns updated unit index
```

**Status**: ✅ Functions defined with correct FFN attribute names (W_up, b_up, W_gate, W_down)

### 2. VANILLA_TRANSFORMER_PLAN.md ✅
**Location**: Root directory

Comprehensive plan document created covering:
- Architecture audit (vanilla vs non-vanilla components)
- Required changes for standalone operation
- 4-phase implementation plan
- Success criteria
- Testing strategy
- Open questions

## Remaining Work

### Phase 1: Core Standalone Functionality

#### Task 1.1: Integrate Default Passthroughs in L6 FFN
**File**: vm_step.py, function `_set_layer6_routing_ffn` (starts around line 1981)
**Status**: ⚠️ Partially implemented, needs completion

**Required code** (add at beginning of function, after `unit = 0`):

```python
def _set_layer6_routing_ffn(ffn, S, BD):
    """..."""
    unit = 0

    # === Default register identity passthroughs (T=0.5, always active) ===
    # These preserve register values across steps by default.
    # Opcode-specific routing (T=4.0) overrides these when an opcode modifies a register.

    # AX default passthrough: AX_CARRY → OUTPUT at MARK_AX
    unit = _route_ax_default_passthrough_16(ffn, unit, S, BD, T=0.5)

    # PC default passthrough: EMBED → OUTPUT at MARK_PC
    unit = _route_marker_passthrough_16(ffn, BD.MARK_PC, BD.EMBED_LO, BD.EMBED_HI, unit, S, BD, T=0.5)

    # SP/BP default passthroughs: EMBED → OUTPUT at respective markers
    unit = _route_marker_passthrough_16(ffn, BD.MARK_SP, BD.EMBED_LO, BD.EMBED_HI, unit, S, BD, T=0.5)
    unit = _route_marker_passthrough_16(ffn, BD.MARK_BP, BD.EMBED_LO, BD.EMBED_HI, unit, S, BD, T=0.5)

    # STACK0 passthrough with PSH suppression
    stack0_unit_start = unit
    unit = _route_marker_passthrough_16(ffn, BD.MARK_STACK0, BD.EMBED_LO, BD.EMBED_HI, unit, S, BD, T=0.5)
    # Suppress during PSH (CMP[0]~1): S*1 - 2*S*1 - S*0.5 = -S*1.5 < 0 (blocked)
    # Without PSH (CMP[0]=0): S*1 - 0 - S*0.5 = S*0.5 > 0 (fires)
    for u in range(stack0_unit_start, unit):
        ffn.W_up[u, BD.CMP + 0] = -S * 2  # suppress when PSH active

    T = 4.0  # opcode threshold: correct OP ≈ 5 + MARK_AX 1 = 6 > T

    # [Rest of function continues with opcode-specific routing...]
```

**Why this works**:
- Default passthroughs use T=0.5 (low threshold, always active)
- Opcode-specific routing uses T=4.0 (high threshold, requires opcode + marker)
- When opcode present: both fire, but residual sums (default + opcode override)
- When no opcode: only default fires, preserving register values

#### Task 1.2: Remove Obsolete Comment
**File**: vm_step.py around line 3372
**Action**: Delete or update the comment that says "SP/BP/STACK0 byte computations removed - DraftVM handles these"

#### Task 1.3: Handle Step 0 Initial State
**Issue**: Step 0 has no previous step, so AX_CARRY = 0
**Options**:
- A: Prepend dummy "step -1" with zero tokens to context
- B: Special handling in embedding layer for first step
- C: Accept that step 0 starts with registers = 0 (simplest)

**Recommended**: Option C - it's the natural initial state

### Phase 2: Testing & Validation

#### Test 1: Model Loading
```bash
python -c "from neural_vm.vm_step import AutoregressiveVM, set_vm_weights; \
           model = AutoregressiveVM(); set_vm_weights(model); \
           print('Model loaded successfully')"
```

#### Test 2: Simple IMM Test
Create test file to verify IMM opcode works:
```python
from neural_vm.vm_step import AutoregressiveVM, set_vm_weights
from neural_vm.speculative import DraftVM
from neural_vm.embedding import Opcode

model = AutoregressiveVM()
set_vm_weights(model)
model.eval()

bytecode = [Opcode.IMM | (42 << 8), Opcode.EXIT]
draft = DraftVM(bytecode)

# Test that neural VM can predict DraftVM tokens
```

#### Test 3: Full Test Suite
```bash
python tests/test_vm.py
```

### Phase 3: Remove Non-Vanilla Operations

#### ADDR_KEY Injection
**Current**: Manual injection after embedding
**Target**: AddressEmbedding layer (see VANILLA_TRANSFORMER_PLAN.md)

#### OUTPUT Clamping
**Current**: Manual clamping after each layer
**Target**: Remove or replace with LayerNorm (see VANILLA_TRANSFORMER_PLAN.md)

## Known Issues

### Issue 1: FFN Attribute Names
**Problem**: Early edits used `ffn.up.weight.data` instead of `ffn.W_up`
**Status**: ✅ FIXED in helper functions
**Remaining**: Check for any other occurrences in passthroughs section

### Issue 2: File Modification Conflicts
**Problem**: File kept getting modified during editing (linter/formatter?)
**Workaround**: Clear Python cache before testing:
```bash
find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Issue 3: Test Files Not Found
**Problem**: test_multi_opcode_consistency.py doesn't exist
**Solution**: Use existing test suite in tests/test_vm.py

## Quick Start Guide

### To Complete Implementation:

1. **Edit vm_step.py**:
   - Find `_set_layer6_routing_ffn` function
   - Add default passthroughs section (see Task 1.1 above)
   - Verify unit counter increments correctly

2. **Clear cache and test**:
   ```bash
   find . -name "*.pyc" -delete
   python tests/test_vm.py 2>&1 | head -50
   ```

3. **Debug failures**:
   - Check which tests fail
   - Create minimal reproduction script
   - Trace through layers to find issue

4. **Iterate**: Fix issues, test, repeat

## Architecture Decisions

### Default Passthrough Strategy
**Chosen approach**: Always-active default passthroughs (T=0.5) + opcode overrides (T=4.0)

**Why this works**:
- SwiGLU activation: `out = silu(W_up * x + b_up) * gate(W_gate * x + b_gate)`
- Default units: `silu(S*MARK - S*0.5)` ≈ small positive when marker present
- Opcode units: `silu(S*OP + S*MARK - S*4.0)` ≈ large positive when both present
- Residual connection adds both contributions
- Opcode contribution dominates when present, default takes over when absent

### PSH Suppression for STACK0
**Implementation**: Add negative weight on CMP[0] to STACK0 passthrough units
- When PSH active: CMP[0] = 1, suppresses passthrough
- When PSH inactive: CMP[0] = 0, passthrough fires normally

## References

- **STACK0_PASSTHROUGH_FINDINGS.md**: Earlier investigation results
- **VANILLA_TRANSFORMER_PLAN.md**: Comprehensive implementation plan
- **vm_step.py**: Main weight initialization file
- **tests/test_vm.py**: Existing test suite

## Critical Finding: Architecture Limitation

**The neural VM was designed for SPECULATIVE DECODING, not pure autoregressive generation.**

### What Works:
- **Speculative mode**: DraftVM generates all 35 tokens of a step → neural VM validates them in one forward pass
- L3 attention can see the complete previous step (35 tokens) to carry forward register values

### What Doesn't Work:
- **Pure autoregressive mode**: Generating tokens one-by-one from scratch
- When predicting token N+1, the model needs register values from previous step
- But L3 attention expects to see a complete step (35 tokens), not partial tokens

### Why Pure Autoregressive Fails:
1. At position 54 (STEP_END of initial step), predicting position 55 (PC marker of step 0)
2. Need PC=2 from initial step to fetch instruction from bytecode
3. L3 should copy PC to EMBED, but L3 expects complete steps, not single tokens
4. Result: EMBED=0, no instruction fetch, OP_IMM=0, no execution

### Solution Options:
**Option A**: Keep speculative decoding (requires DraftVM)
- Default passthroughs still useful for validation
- Tests need to use DraftVM + neural VM together

**Option B**: Redesign for pure autoregressive (major architectural change)
- Modify L3 attention to work with partial steps
- OR use different mechanism to carry forward register state
- Much more complex than adding default passthroughs

### Testing Results (2025-03-26):

**Test 1: Speculative mode with IMM instruction**
- Context: [CODE, DATA, initial_step(PC=2, AX=0), draft_step_0(PC=7, AX=42)]
- Result: 33/35 tokens match, 2 mismatches (PC_B0, AX_B0)
- Neural VM predicts: PC=2 (not advancing), AX=0 (not executing IMM)

**Layer-by-layer trace**:
1. After L2: EMBED_LO = [0, 0, 0, 0] ✗ (should be [2, 0, 0, 0] from initial_step PC)
2. After L3: AX_CARRY_LO ≈ [1.0, 0, 0, 0] ✗ (relaying PC=0, not PC=2)
3. After L4: FETCH_LO = 0 ✗ (no instruction fetched)
4. After L5: OP_IMM = 0 ✗ (no opcode decoded)
5. After L6: OUTPUT preserves PC=2, AX=0 via default passthroughs ✓

**Root cause**: L3 attention isn't copying PC bytes from initial_step to EMBED in draft_step_0.

### Why L3 Attention Fails:
- L3 should attend from draft_step_0 positions back to initial_step PC marker
- But causal masking might prevent this, OR
- L3 attention weights assume a different context structure, OR
- The original architecture never actually used draft tokens in the context this way

### Recommendation:
- **INVESTIGATE**: Check how batch_runner actually uses the model - does it validate tokens differently?
- **CHECK**: Review L3 attention weight setup to understand expected context structure
- **CONSIDER**: The system might only work with external DraftVM execution, not neural validation

## Next Session TODO

1. ✅ Complete Task 1.1 (integrate default passthroughs) - DONE
2. Test default passthroughs with speculative decoding (batch_runner + DraftVM)
3. Verify existing test suite passes with default passthroughs
4. Document speculative vs autoregressive trade-offs
5. Consider whether pure autoregressive generation is actually needed

## Notes

- Helper functions use correct FFN attribute names (W_up, b_up, W_gate, W_down)
- Default passthroughs must come BEFORE opcode-specific routing in unit order
- STACK0 passthrough needs PSH suppression via CMP[0] negative weight
- All changes are vanilla transformer operations (no custom forward pass modifications)
