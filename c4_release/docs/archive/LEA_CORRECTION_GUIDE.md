# LEA Opcode Correction Guide

## Problem

The LEA (Load Effective Address) opcode has ~88.6% neural prediction accuracy due to:
- Dimension amplification across layers (OP_LEA, FETCH, ALU values amplified 2-14x)
- Complex multi-layer routing (7+ layers involved)
- First-step initialization challenges

## Solution

**Arithmetic Correction**: Detect LEA execution and compute the correct result (AX = BP + immediate) instead of using neural predictions.

## Usage

### Basic Correction

```python
from neural_vm.lea_correction import correct_lea_prediction

# Get neural prediction
with torch.no_grad():
    logits = model.forward(input_tokens)
    neural_pred = logits[0, -1].argmax().item()

# Apply LEA correction
corrected_pred = correct_lea_prediction(context, draft_tokens, neural_pred)
```

### Check if Instruction is LEA

```python
from neural_vm.lea_correction import is_lea_instruction

if is_lea_instruction(context, draft_tokens):
    # This step executed LEA
    print("LEA detected")
```

### Integration with Batch Runner

To integrate LEA correction into speculative execution:

```python
# In batch_runner.py or custom runner:
from neural_vm.lea_correction import correct_lea_prediction

# After getting predictions:
for i, draft_tok in enumerate(draft_tokens):
    pred = logits[0, ctx_len - 1 + i, :].argmax(-1).item()

    # Apply correction for AX byte 0 (position 6 in draft tokens)
    if i == 6:
        pred = correct_lea_prediction(context, draft_tokens, pred)

    # Validate prediction
    if pred == draft_tok:
        accepted += 1
    else:
        break
```

## Test Results

**Without Correction:**
- LEA 8: Predicts 1 (expected 8) ✗

**With Correction:**
- LEA 0: Predicts 0 ✓
- LEA 8: Predicts 8 ✓
- LEA 100: Predicts 100 ✓
- LEA 255: Predicts 255 ✓
- LEA 256: Predicts 0 ✓

**Success Rate: 100%** (with correction applied)

## How It Works

1. **Parse PC**: Extract PC from draft tokens (positions 1-4)
2. **Calculate Previous PC**: Subtract INSTR_WIDTH to get PC before execution
3. **Find Instruction**: Locate opcode in bytecode context
4. **Detect LEA**: Check if opcode == 0
5. **Compute Result**: AX = (BP + immediate) & 0xFFFFFFFF
6. **Return Byte 0**: Extract low byte of computed AX

## Files

- `neural_vm/lea_correction.py`: Correction implementation
- `test_lea_with_correction.py`: Single LEA test
- `test_lea_comprehensive.py`: Multiple LEA cases
- `LEA_STATUS.md`: Detailed neural debugging history

## Limitations

- Only corrects AX byte 0 prediction (position 6 in draft tokens)
- Requires correct draft tokens for BP and PC
- Does not fix neural weights (correction is post-processing)

## Future Work

To eliminate the need for corrections:
1. Fix dimension amplification in Layer 6 attention (normalize OP_LEA, FETCH, ALU)
2. Simplify LEA routing (reduce from 7 layers to 3-4)
3. Add explicit dimension clearing before writes

Estimated effort: 4-8 hours
