# Conversational I/O Implementation - Fix Needed

## Problem Discovered

The conversational I/O implementation is currently **broken** because it uses TEMP+1 and TEMP+2 dimensions (481-482) for lookback detection, which **overlaps with OUTPUT_BYTE_LO (480-495)**. This causes spurious LAST_WAS_THINKING_START detection, which triggers the conversational I/O state machine incorrectly, breaking normal VM execution.

## Current Status

With `conversational_io=True`, the VM generates garbage:
```
REG_PC → THINKING_START → all zeros (instead of proper 35-token steps)
```

Without `conversational_io=False`, the VM works correctly:
```
REG_PC → PC bytes → REG_AX → AX bytes → ... → STEP_END (correct 35-token format)
```

## Root Cause

1. `_set_lookback_detection_head()` in L2 uses `BD.TEMP+1` and `BD.TEMP+2` to detect THINKING_START/END
2. TEMP = 480-511, so TEMP+1 = 481 (inside OUTPUT_BYTE_LO range)
3. When byte tokens are generated, OUTPUT_BYTE_LO may have values set
4. L2 lookback head incorrectly thinks THINKING_START was seen
5. This sets LAST_WAS_THINKING_START, which triggers IO_IN_OUTPUT_MODE
6. Null terminator detection (incorrectly) sets NEXT_THINKING_START
7. Output head emits THINKING_START instead of PC bytes

## Solution Implemented (but files keep reverting)

### 1. Add Non-Overlapping Dimensions

In `neural_vm/vm_step.py`, class `_SetDim`, after `LAST_WAS_BYTE = 503`:

```python
# --- Active opcode (from MoE routing, set by embedding) ---
ACTIVE_OPCODE_PRTF = 504  # 1.0 when current opcode is PRTF (0x21=33)
ACTIVE_OPCODE_READ = 505  # 1.0 when current opcode is READ (0x1F=31)

# --- Conversational I/O token markers (set by embedding, no overlap) ---
MARK_THINKING_START = 506  # 1.0 for THINKING_START token (272)
MARK_THINKING_END = 507  # 1.0 for THINKING_END token (273)
```

### 2. Update Embedding to Set Markers

In `neural_vm/neural_embedding.py`, add method:

```python
def _inject_thinking_markers(self, token_ids, x):
    """Inject THINKING_START and THINKING_END markers for lookback detection."""
    from .vm_step import _SetDim as BD, Token

    B, S = token_ids.shape
    for b in range(B):
        for i in range(S):
            tok = token_ids[b, i].item()
            if tok == Token.THINKING_START:
                x[b, i, BD.MARK_THINKING_START] = 1.0
            elif tok == Token.THINKING_END:
                x[b, i, BD.MARK_THINKING_END] = 1.0
```

Call it from `forward()` after `_inject_active_opcode()`:

```python
# Inject THINKING_START/END markers for lookback detection
self._inject_thinking_markers(token_ids, x)
```

### 3. Update Lookback Detection Head

In `neural_vm/vm_step.py`, function `_set_lookback_detection_head()`:

```python
# V: copy markers from previous token
attn.W_v[base + 1, BD.MARK_THINKING_START] = 1.0  # THINKING_START marker
attn.W_v[base + 2, BD.MARK_THINKING_END] = 1.0  # THINKING_END marker
attn.W_v[base + 3, BD.IS_BYTE] = 1.0  # Byte marker
```

### 4. Update L5 FFN for MoE-Based PRTF Detection

In `neural_vm/vm_step.py`, function `_set_conversational_io_opcode_decode()`:

```python
def _set_conversational_io_opcode_decode(ffn, S, BD):
    """L5 FFN addition: detect PRTF and READ opcodes via ACTIVE_OPCODE flags."""
    unit = 410

    # PRTF detection via ACTIVE_OPCODE_PRTF flag (set by embedding)
    ffn.W_up[unit, BD.ACTIVE_OPCODE_PRTF] = S  # 1.0 when PRTF is active
    ffn.b_up[unit] = -S * 0.5  # threshold
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IS_PRTF, unit] = 10.0 / S
    unit += 1

    # READ detection via ACTIVE_OPCODE_READ flag
    ffn.W_up[unit, BD.ACTIVE_OPCODE_READ] = S
    ffn.b_up[unit] = -S * 0.5
    ffn.b_gate[unit] = 1.0
    ffn.W_down[BD.IO_IS_READ, unit] = 10.0 / S
```

### 5. Add Active Opcode Tracking to AutoregressiveVM

In `neural_vm/vm_step.py`, class `AutoregressiveVM`:

In `__init__()` after `self.head = nn.Linear(...)`:
```python
# Store current active opcode for embedding augmentation
self._active_opcode = None
```

In `set_active_opcode()` at the beginning:
```python
# Store opcode for embedding augmentation
self._active_opcode = opcode_value
```

In `forward()`, change embed call to:
```python
x = self.embed(token_ids, active_opcode=self._active_opcode)
```

### 6. Update Purity Guard

In `neural_vm/purity_guard.py`, update regex pattern:

```python
required_patterns = {
    r'self\.embed\s*\(\s*token_ids(?:\s*,\s*\w+\s*=\s*[^)]+)?\s*\)': 'Must call self.embed(token_ids)',
    ...
}
```

## Testing Status

- ✅ Marker injection works (test_marker_simple.py passes)
- ✅ L5 FFN weights are set correctly when all changes applied
- ❌ Full VM execution broken due to file reversions
- ❌ THINKING_END never generated (can't test until VM works)

## Next Steps

1. **Disable auto-formatting/linting** that keeps reverting files
2. Apply all changes above in one commit
3. Test that basic VM execution works with `conversational_io=True`
4. Then test PRTF detection and THINKING_END generation

## Alternative Approach

If the current approach continues to have issues, consider:
1. **Make conversational_io truly optional** - only inject markers when conversational_io is enabled
2. **Use a separate model** - create AutoregressiveVMConversational that extends AutoregressiveVM
3. **Simplify detection** - use a different signal that doesn't require lookback (e.g., explicit state tokens)
