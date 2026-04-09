# Neural Arithmetic Investigation - Final Summary

**Date**: 2026-04-07
**Status**: Mechanisms working, but full ADD still incorrect
**Exit Code**: 32 (with max_steps=5) or 10 (with max_steps=10) instead of expected 42

## Critical Discovery

**The original investigation was COMPLETELY WRONG.**

The initial diagnosis (docs/FIX_ATTEMPT_SUMMARY.md) concluded that:
- IS_MARK flags were missing ✗ WRONG
- Layer 1 threshold heads were broken ✗ WRONG
- Issue was autoregressive format ✗ WRONG

**Reality**: All the neural mechanisms ARE working, but the investigation only examined the **initial context prefix** (first forward pass), not the **generated tokens** where actual VM execution happens!

## What Actually Works ✓

### 1. IS_MARK Flag Mechanism
**Status**: WORKING ✓

**Evidence** (`check_is_mark_all_passes.py`):
```
Pass 6: REG_AX generated (position 54)
IS_MARK value: 1.000
```

- Marker tokens (REG_PC, REG_AX, REG_SP, etc.) ARE generated during autoregressive execution
- Each marker token HAS IS_MARK=1.0 in its embedding
- The mechanism works exactly as designed

**Previous mistake**: Only captured first forward pass (prefix), missed generated tokens

### 2. Layer 1 Threshold Heads
**Status**: WORKING ✓

**Evidence** (`debug_l1_mechanism.py`):
```
Layer 1 Attention Configuration:
  ALiBi slopes: [10. 10. 10.  0. 10. 10. 10. 10.]

Threshold Head Configuration:
  Head 0 (threshold 0.5):
    W_q[0, CONST=8] = 80.00
    W_k[0, IS_MARK=7] = 0.50
  Head 1 (threshold 1.5):
    W_q[64, CONST=8] = 80.00
    W_k[64, IS_MARK=7] = 1.50

Last Layer 1 attention output:
  Max L1H0[AX]: 0.993 ✓
  Max L1H1[AX]: 1.000 ✓

Positions with L1H1[AX] > 0.5:
  Position 54: L1H1[AX]=1.00, L1H0[AX]=0.99 ← byte 0
  Position 55: L1H1[AX]=0.99, L1H0[AX]=0.00 ← byte 0
```

- Threshold attention computes distances correctly
- L1H1/L1H0 byte index encoding works
- ALiBi slopes are configured correctly

**Previous mistake**: Only examined first forward pass, found no markers

### 3. Layer 3 AX Carry-Forward
**Status**: WORKING (in specific forward passes) ✓

**Evidence** (`debug_all_l3_passes.py`):
```
Found 4 passes with AX_CARRY_LO > 0.5

Passes with AX_CARRY set:
  Pass 41 (step 1, seq_len=99): AX_CARRY_LO max=1.000
  Pass 76 (step 2, seq_len=99): AX_CARRY_LO max=1.000
  Pass 111 (step 3, seq_len=99): AX_CARRY_LO max=1.000
  Pass 146 (step 4, seq_len=108): AX_CARRY_LO max=1.000
```

- Layer 3 attention DOES populate AX_CARRY_LO/HI
- Uses L1H1/L1H0 byte index patterns correctly
- Mechanism works as designed

## What's Still Broken ✗

### Exit Code: 32 or 10, Not 42

**Observed behavior**:
- With `max_steps=5`: exit code = 32 (second operand)
- With `max_steps=10`: exit code = 10 (first operand)
- Expected: exit code = 42 (10 + 32)

**Evidence** (`debug_layer8_add.py`):
```
Exit code: 10
Expected: 42 (10 + 32)

Layer 7 output (gather operands):
  ALU_LO (first operand): nibble 0 (max=0.71)
  AX_CARRY_LO (second op): nibble 0 (max=0.00) ← ZERO!
```

### Hypothesis: Token-Level vs Step-Level State

**Key insight**: Layer 3 sets AX_CARRY during **specific forward passes** (seq_len=99, 108), but this may not persist to when Layer 7/8 need it for ADD computation.

**Autoregressive generation structure**:
1. Each token generation is a separate forward pass
2. Context grows by 1 token each pass
3. Layer outputs are NOT persistent across passes
4. A value set at seq_len=99 may not be visible at seq_len=150

**The problem**: AX_CARRY is set during one forward pass but may be lost by the time ADD needs it in a different forward pass!

## Root Cause Analysis

### Multi-Pass Consistency Issue

**Theory**: The residual stream doesn't "remember" values across forward passes in autoregressive generation.

When generating token N:
1. Input: tokens [0...N-1]
2. Embedding: tokens [0...N-1] → embeddings
3. Transformer: processes sequence
4. Output: logits for token N

When generating token N+1:
1. Input: tokens [0...N] (includes newly generated token N)
2. Embedding: **FRESH** embedding of all tokens
3. Transformer: **FRESH** forward pass
4. Output: logits for token N+1

**Critical point**: AX_CARRY set during pass N is in the residual stream during pass N, but pass N+1 starts fresh!

**However**: The attention mechanism CAN look back at previous tokens, so AX_CARRY should be visible in the context...

### Alternative Theory: Positional Dependency

**Hypothesis**: AX_CARRY is only set at specific sequence positions (e.g., position 99 in a seq_len=99 context), but when the sequence grows (seq_len=150), that position is in the middle of the context, and the mechanism expects it at the end?

## Why Handlers Work

Python handlers bypass the multi-pass issue by:
1. Directly reading from context (previous step's AX value)
2. Modifying the generated step's AX value
3. Operating at step granularity, not token granularity

## Diagnostic Questions

### Q1: Is AX_CARRY visible during ADD forward pass?

Need to check: During the forward pass that generates the ADD step, is AX_CARRY present in the context at the positions Layer 7 is looking at?

### Q2: Is Layer 7 attention looking at the right positions?

Need to verify: Layer 7 attention scores during ADD step - is it attending to positions where AX_CARRY was set?

### Q3: Is this a KV cache or context pruning issue?

The runner may be pruning context or using KV cache in a way that loses AX_CARRY values.

## Next Steps (Ordered by Priority)

### Option 1: Check Layer 7 Attention During ADD ⭐ RECOMMENDED

**Goal**: Verify if Layer 7 can "see" AX_CARRY values in context during ADD step

**Script**: Debug Layer 7 attention scores during the specific forward pass that computes ADD
- Which positions is Layer 7 attending to?
- Do those positions have AX_CARRY set?
- Is the attention mechanism working correctly?

### Option 2: Check Multi-Step State Propagation

**Goal**: Verify how values propagate through multiple steps

**Script**: Trace AX value from IMM(10) → stack → IMM(32) → stack → ADD
- Does AX=10 persist after IMM?
- Does stack contain 10 and 32 before ADD?
- Is Layer 3 looking at the right stack positions?

### Option 3: Accept Handlers as Permanent Solution

**Rationale**:
- All individual mechanisms work correctly
- The issue is architectural (multi-pass autoregressive generation)
- Fixing may require redesigning the attention-memory system
- Handlers work reliably and correctly

**Action**:
- Document why handlers are needed (autoregressive state persistence)
- Update comments from "neural weights broken" to "requires step-level state"
- Mark as architectural limitation, not bug

## Investigation Scripts Summary

### Corrected Scripts (Valid Findings)
| Script | Purpose | Key Finding |
|--------|---------|-------------|
| `check_is_mark_all_passes.py` | Check IS_MARK in all forward passes | IS_MARK=1.0 on marker tokens ✓ |
| `debug_l1_mechanism.py` | Check Layer 1 threshold heads | L1H1/L1H0 working correctly ✓ |
| `debug_add_final_check.py` | Check Layer 3 AX carry-forward | AX_CARRY populated in some passes ✓ |
| `debug_all_l3_passes.py` | Scan ALL Layer 3 passes | 4 passes have AX_CARRY=1.000 ✓ |
| `debug_layer8_add.py` | Check ADD computation in Layer 8 | AX_CARRY=0 at Layer 7 output ✗ |

### Invalid Scripts (Wrong Methodology)
| Script | Problem | Why Invalid |
|--------|---------|-------------|
| `check_is_mark.py` | Only captured first forward pass | Missed generated tokens |
| `debug_l1_threshold.py` | Only captured first forward pass | Missed threshold head activity |
| `debug_l1_attention.py` | Only captured first forward pass | Missed marker tokens |

**Critical methodological lesson**: In autoregressive generation, examining only the first forward pass (prefix) misses all the action! Must capture ALL forward passes or at least the final ones where VM execution happens.

## Conclusion

The investigation journey:
1. **Initial belief**: Layer 1 threshold heads broken (wrong)
2. **First correction**: IS_MARK missing (wrong methodology)
3. **Second correction**: All mechanisms work! (current understanding)
4. **Current mystery**: Why does ADD still return wrong result?

**Most likely answer**: Token-level autoregressive generation doesn't preserve step-level state (like AX_CARRY) in a way that downstream layers can reliably access across multiple forward passes.

**Recommended action**: Option 1 (debug Layer 7 attention) to confirm, then likely Option 3 (accept handlers).

---

**Total Investigation Time**: ~6 hours
**Methodological Errors**: 2 major (examined wrong data)
**Correct Understanding**: All mechanisms work individually
**Remaining Mystery**: Why doesn't it work end-to-end?
**Next**: Debug Layer 7 attention during ADD step
