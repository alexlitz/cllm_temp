# ADD Operation Neural Weight Investigation

## Investigation Goal

Determine why the neural ADD operation returns the first operand (10) instead of the sum (42) when the handler is disabled.

## Test Case

```c
int main() { return 10 + 32; }
```

**Expected**: exit_code = 42
**Actual (without handler)**: exit_code = 10 (first operand only)
**Actual (with handler)**: exit_code = 42 (handler performs addition in Python)

## Current Findings

### 1. Handler Dependency Confirmed

From `neural_vm/run_vm.py:214-235`, the handler configuration explicitly states:
```python
# Arithmetic operations (neural weights broken, using fallback)
Opcode.ADD: self._handler_add,
Opcode.SUB: self._handler_sub,
Opcode.MUL: self._handler_mul,
Opcode.DIV: self._handler_div,
Opcode.MOD: self._handler_mod,
```

### 2. Handler Implementation

From `neural_vm/run_vm.py:1241`:
```python
def _handler_add(self, context, output):
    """ADD -- pop stack, AX = stack_val + AX."""
    stack_val = self._last_sp_value()  # Get top of stack
    ax = self._last_ax
    result = (stack_val + ax) & 0xFFFFFFFF
    self._override_ax_in_last_step(context, result)
```

The handler performs the actual addition in Python and overrides the neural output.

### 3. Neural Architecture (from vm_step.py:3878-3915)

**Layer 8 FFN** implements ADD via 3-way AND gates:

```python
# ADD: lo nibble (256 units)
for a in range(16):
    for b in range(16):
        result = (a + b) % 16
        ffn.W_up[unit, BD.MARK_AX] = S
        ffn.W_up[unit, BD.ALU_LO + a] = S
        ffn.W_up[unit, BD.AX_CARRY_LO + b] = S
        ffn.b_up[unit] = -S * 2.5  # 3-way AND
        ffn.W_gate[unit, BD.OP_ADD] = 1.0
        ffn.W_down[BD.OUTPUT_LO + result, unit] = 2.0 / S
```

**Data flow**:
1. **Layer 7 attention**: Gather first operand from previous step's STACK0 → ALU_LO
2. **Layer 3 attention**: Carry second operand (AX) → AX_CARRY_LO
3. **Layer 8 FFN**: Compute sum via AND gates → OUTPUT_LO
4. **Layer 10+**: Relay output to next step's AX

### 4. Symptom Analysis

**Pattern**: Returns 10 (first operand only)

**Hypotheses**:
1. ALU_LO gathering (Layer 7) might not be working - operand not reaching ALU
2. AX_CARRY relay (Layer 3) might not be working - second operand not available
3. Layer 8 ADD circuit might not be activating - weights not firing
4. Layer 10 might be overriding ALU output with passthrough - bypassing computation

## Investigation Attempts

### Attempt 1: Activation Capture

Created `debug_add_neural_weights.py` to hook transformer layers and capture intermediate activations.

**Issues encountered**:
- CUDA OOM when hooking all 16 layers
- Reduced to hooking only key layers (0, 3, 7, 8, 10, 15)
- Successfully ran but found no OP_ADD activation in early layers

### Attempt 2: Opcode Flag Detection

Created `debug_add_simple.py` to check when OP_ADD flag becomes active.

**Findings**:
- Hooked Layer 6 FFN output (after opcode decoding)
- Only 3 VM steps captured (136 tokens / 35 tokens per step)
- No OP_ADD flag found in any step
- Only saw OP_0, OP_35-39 (unknown opcodes)

**Issue**: Either:
- ADD instruction not being executed
- OP_ADD flag set in different way than expected
- Capturing wrong layer/position

## Next Steps

### 1. Verify Bytecode Execution

Need to confirm that ADD instruction is actually being executed:
- Disassemble the bytecode to see instruction sequence
- Trace which instructions are executed in which steps
- Verify that we're looking at the right step for ADD

### 2. Check Data Flow at ADD Step

Once we identify the correct step:
- Check Layer 7 attention: Is ALU_LO populated with first operand?
- Check Layer 3 attention: Is AX_CARRY_LO populated with second operand?
- Check Layer 8 FFN: Are ADD units activating?
- Check Layer 10: Is OUTPUT being overridden?

### 3. Compare with Working Opcode

Compare ADD data flow with a working opcode (like IMM or JMP):
- What's different about the activation patterns?
- Are the input dimensions populated correctly?
- Are the weights configured correctly?

### 4. Weight Inspection

Examine the actual weight values:
- Are Layer 7 attention weights configured to gather STACK0 → ALU_LO?
- Are Layer 3 attention weights configured to carry AX → AX_CARRY_LO?
- Are Layer 8 FFN weights actually implementing the ADD circuit?

## Technical Challenges

1. **Large sequence length**: 35 tokens per step makes analysis verbose
2. **Memory constraints**: Cannot hook all layers simultaneously
3. **Opcode detection**: Opcode flags set by Layer 5, need to hook correct layer
4. **Sparse activations**: Most dimensions are zero, hard to find signal
5. **Complex data flow**: Multiple layers involved in single operation

## Questions to Answer

1. **Is ADD instruction being executed?** Need bytecode trace
2. **Where is OP_ADD flag set?** Need to hook Layer 5 FFN output
3. **Are operands reaching ALU?** Need Layer 7 attention analysis
4. **Is AX being carried?** Need Layer 3 attention analysis
5. **Are ADD units activating?** Need Layer 8 FFN activation analysis
6. **Is output being overridden?** Need Layer 10 comparison

## Preliminary Conclusion

The neural ADD implementation appears structurally correct based on weight initialization code, but runtime activation analysis shows that either:

1. The data routing is broken (operands not reaching ALU)
2. The ADD circuit is not activating (weights not firing)
3. The output is being overridden (passthrough bypassing ALU)

Further investigation requires:
- Bytecode execution trace
- Layer-by-layer activation analysis at the exact ADD execution step
- Comparison with working opcodes to identify differences

---

**Date**: 2026-04-07
**Status**: Investigation in progress
**Next**: Verify bytecode execution and identify ADD step
