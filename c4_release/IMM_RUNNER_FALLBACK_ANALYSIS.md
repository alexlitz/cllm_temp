# IMM Instruction: Runner Fallback Analysis

## Summary

The IMM instruction produces correct exit codes (42) **despite the neural model generating incorrect AX values ([0,0,0,0])** because the runner has a fallback handler that extracts the immediate value directly from bytecode and overrides the model's output.

## Discovery Timeline

1. **Observed**: IMM 42, EXIT returns exit_code=42 ✓
2. **Puzzling**: Step output shows AX=[0,0,0,0] ✗
3. **Root Cause**: Runner has `_handler_imm` fallback that bypasses neural weights entirely

## How It Works

### Neural Model Flow (Broken)

```
Step 0: IMM 42
├─ Model generates: REG_AX [0, 0, 0, 0]  ← WRONG (should be [42, 0, 0, 0])
├─ Reason: Layer 3 SP/BP default units were firing at AX positions
└─ Fix applied: Added marker gates to units 4 & 5 (vm_step.py:2385-2400)
```

### Runner Override Flow (Working)

```python
# neural_vm/run_vm.py:408-410
exec_op = bytecode[exec_idx] & 0xFF  # exec_op = IMM
func_handler = self._func_call_handlers.get(exec_op)  # _handler_imm
func_handler(context, output)  # Call IMM handler

# neural_vm/run_vm.py:1253-1267
def _handler_imm(self, context, output):
    """IMM -- Load immediate value into AX."""
    exec_pc = self._exec_pc()
    exec_idx = exec_pc // INSTR_WIDTH
    instr = self._bytecode[exec_idx]
    imm = instr >> 8  # Extract immediate (42) from bytecode
    if imm >= 0x800000:
        imm -= 0x1000000  # Sign-extend 24-bit immediate
    self._override_ax_in_last_step(context, imm & 0xFFFFFFFF)  # Override to 42
    next_pc = (exec_pc + INSTR_WIDTH) & 0xFFFFFFFF
    self._override_register_in_last_step(context, Token.REG_PC, next_pc)

# Result: AX in context is overridden to [42, 0, 0, 0]
```

### Exit Code Extraction (Working)

```python
# neural_vm/run_vm.py:461-463
if exec_op in AX_MODIFYING_OPS:  # IMM is in this set
    self._last_ax = ax  # Extract overridden AX (42)

# neural_vm/run_vm.py:604-609 (on HALT)
if next_token == Token.HALT:
    self._override_register_in_last_step(context, Token.REG_AX, self._last_ax)
    break

# neural_vm/run_vm.py:611
return "".join(output), self._decode_exit_code(context)  # Returns 42

# neural_vm/run_vm.py:1540-1548
def _decode_exit_code(self, context):
    """Extract exit code from the last REG_AX before HALT."""
    for i in range(len(context) - 1, -1, -1):
        if context[i] == Token.REG_AX and i + 4 < len(context):
            val = 0
            for j in range(4):
                val |= (context[i + 1 + j] & 0xFF) << (j * 8)
            return val  # Returns 42
    return 0
```

## Timeline of Context Modifications

```
1. Model generates:     REG_AX [0, 0, 0, 0]
                             ↓
2. _handler_imm():      REG_AX [42, 0, 0, 0]  ← Override from bytecode
                             ↓
3. Extract (line 463):  self._last_ax = 42   ← Save overridden value
                             ↓
4. HALT detected:       (preserve _last_ax)
                             ↓
5. _decode_exit_code(): return 42            ← Extract from context
```

## Why Step Output Shows AX=0

The step output (printed representation) shows the **original model generation** before the runner's `_handler_imm` override is applied. But internally, the context is modified:

```python
# What the model generated (visible in step output)
REG_AX [0, 0, 0, 0]

# What the runner writes to context (used for exit_code extraction)
REG_AX [42, 0, 0, 0]  ← Overridden by _override_ax_in_last_step()
```

## Runner Fallback Philosophy

From `neural_vm/run_vm.py:213-214`:

> Function-call handlers dispatched by exec_pc (not output PC).
> These are runner-side compatibility shims for full 32-bit correctness
> while the corresponding neural memory paths are being completed.

**Translation**: The neural weights for IMM are not yet trained/set. The runner provides a fallback that extracts the immediate value directly from bytecode instructions to ensure correct execution.

## Implications

### For Testing
- ✅ Exit codes are correct (thanks to runner fallback)
- ✗ Model outputs are wrong (neural weights not set)
- ⚠️ Tests that check `runner.run()` output will pass
- ⚠️ Tests that check raw model generation will fail

### For Training
- The model is not learning to execute IMM neurally
- Layer 6 FFN routing logic for IMM is bypassed
- FETCH → OUTPUT path is not being used

### For My Fix
- ✅ My fix to Layer 3 FFN (marker gates) is correct
- ✅ Stopped SP/BP defaults from corrupting AX positions
- ⚠️ But the IMM instruction still needs neural weights to be set

## What's Actually Broken

The full IMM execution path should be:

```
Layer 5: Fetch immediate from code → FETCH_LO/HI
Layer 6: Route FETCH → OUTPUT (gated by OP_IMM)
Output:  AX bytes = FETCH bytes
```

**Current state**:
- Layer 5: FETCH is likely set (code addressing works)
- Layer 6: OP_IMM flag may not be set, or routing weights missing
- Layer 6: Fallback to runner handler compensates

## Next Steps (If Neural IMM is desired)

1. **Verify OP_IMM flag**: Check if Layer 2 sets OP_IMM=1 for IMM opcode
2. **Check Layer 6 routing**: Verify FFN units route FETCH → OUTPUT when OP_IMM=1
3. **Test without runner**: Disable `_handler_imm` fallback and see what model produces
4. **Set missing weights**: If routing units don't exist, add them to `set_vm_weights()`

## Conclusion

My fix to Layer 3 FFN (adding marker gates to units 4 and 5) **solved the immediate problem** of preventing SP/BP default values from corrupting AX register outputs.

However, the IMM instruction **does not execute neurally** - it relies entirely on a runner-side fallback that extracts immediate values from bytecode and overrides the model's output.

**Exit code 42 is correct, but the neural VM is not doing the work.**

For a truly neural implementation, the Layer 6 FFN routing logic for IMM needs to be implemented/debugged. The runner fallback is a temporary compatibility shim that masks the fact that the neural weights are incomplete.
