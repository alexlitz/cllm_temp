# L3 Head 5 Fix Analysis: Why AX OUTPUT → AX_CARRY Failed

**Date**: 2026-04-09
**Investigation**: Analysis of failed attempt to populate AX_CARRY with current AX value

---

## The Failed Fix (Found in commit 10071e2 and earlier)

### Implementation

```python
# Head 5: AX marker OUTPUT preservation (prev AX marker OUTPUT → current AX marker AX_CARRY)
# FIX (2026-04-09): Add dedicated head to copy previous AX marker's OUTPUT to AX_CARRY.
# This fixes JMP/NOP/EXIT AX corruption where AX_CARRY had wrong values.
# Attends from current AX marker to previous AX marker (not byte 0, but the marker itself).
# Only fires on subsequent steps (HAS_SE=1) to avoid issues on first step.
base = 5 * HD
L = 15.0

# Q: Fire at AX marker on subsequent steps only (HAS_SE=1)
attn3.W_q[base, BD.MARK_AX] = L
attn3.W_q[base, BD.HAS_SE] = L
attn3.W_q[base, BD.CONST] = -L * 1.5  # Threshold: need both MARK_AX and HAS_SE

# K: Match previous step's AX marker
attn3.W_k[base, BD.MARK_AX] = L

# V: Copy OUTPUT_LO/HI from previous AX marker (the final register value)
for k in range(16):
    attn3.W_v[base + 1 + k, BD.OUTPUT_LO + k] = 1.0
    attn3.W_v[base + 17 + k, BD.OUTPUT_HI + k] = 1.0

# O: Write to AX_CARRY_LO/HI at current AX marker (higher priority than head 1)
for k in range(16):
    attn3.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 2.0  # Higher weight to override head 1
    attn3.W_o[BD.AX_CARRY_HI + k, base + 17 + k] = 2.0

# Anti-leakage gate
GATE = 33
attn3.W_q[base + GATE, BD.MARK_AX] = L
attn3.W_q[base + GATE, BD.CONST] = -L / 2
attn3.W_k[base + GATE, BD.CONST] = L
```

### What It Tried to Do

1. **Attend**: Current step's AX marker → Previous step's AX marker
2. **Copy**: Previous AX marker's OUTPUT_LO/HI (the final computed AX value)
3. **Write**: To AX_CARRY_LO/HI at current AX marker
4. **Override**: Use weight 2.0 (vs head 1's 1.0) to take priority

### Why It Should Have Worked

- **Correct source**: OUTPUT contains the final AX value (not EMBED)
- **Correct target**: AX_CARRY is what PSH needs
- **Proper gating**: Only fires on subsequent steps (HAS_SE=1)
- **Anti-leakage**: Gate dim 33 to prevent spurious attention

---

## The Failure

**Result**: "Predictions became all 1's"

This catastrophic failure suggests the fix caused extreme values somewhere in the network, likely saturating activations or attention weights.

---

## Potential Root Causes

### Hypothesis 1: Conflict with Head 1 (Probable)

**Head 1** (existing):
```python
# L3 head 1: AX carry (prev step AX byte 0 → AX_CARRY staging)
_set_carry_forward_attn(
    attn3, 1, BD.MARK_AX, AX_I, AX_I, HD, BD.AX_CARRY_LO, BD.AX_CARRY_HI
)
```

Writes to AX_CARRY with weight 1.0.

**Head 5** (new):
Writes to AX_CARRY with weight 2.0.

**Problem**: Both heads write to the same dims (AX_CARRY_LO/HI). Attention outputs are ADDITIVE!

**Result**:
```
AX_CARRY = head1_output * 1.0 + head5_output * 2.0
```

If both heads fire simultaneously:
- Head 1: Copies byte 0 EMBED (~16 one-hot value)
- Head 5: Copies full OUTPUT (~16 one-hot value for each nibble)
- Sum: Could create values > 1.0 in multiple dims
- If multiple dims activated: argmax unpredictable, possibly all 1's (nibble 0xF)

### Hypothesis 2: OUTPUT Feedback Loop (Possible)

L3 reads OUTPUT from previous step, but OUTPUT is computed AFTER L16 in the previous step.

**Potential issue**:
1. Step N-1: L16 FFN writes to OUTPUT
2. Step N: L3 head 5 copies OUTPUT → AX_CARRY
3. L6 FFN reads AX_CARRY → writes to OUTPUT
4. Step N+1: L3 head 5 copies that OUTPUT → AX_CARRY
5. **Amplification**: If any normalization is off, values could grow

This is unlikely with proper layer norm, but possible if:
- Layer norm not applied to OUTPUT dims
- FFN routing has gain > 1.0
- Multi-step accumulation

### Hypothesis 3: First-Step Edge Case (Less Likely)

Despite the `HAS_SE=1` gate, something might fire on first step:
- CONST dim might not be exactly 1.0
- Threshold -L*1.5 might allow leakage
- First step might have unexpected HAS_SE value

**Result**: Reading from uninitialized OUTPUT on step 0 → undefined behavior

### Hypothesis 4: Attention Score Explosion (Possible)

With L=15.0 and ALiBi slope unset (defaults to 0?):

```
Score = Q·K / sqrt(HD) - slope * distance
```

For self-attention at AX marker (d=0):
```
Score = (15 + 15) * 15 / sqrt(64) - 0 * 0 = 450 / 8 = 56.25
```

`exp(56.25)` is astronomically large (>10^24), which could cause:
- Softmax overflow
- Numerical instability
- NaN propagation → all 1's as fallback

**Check**: If ALiBi slope wasn't set for L3 head 5.

### Hypothesis 5: V Slot Conflict (Less Likely)

Head 1 uses V slots [1-16, 17-32] for EMBED_LO/HI.
Head 5 uses V slots [1-16, 17-32] for OUTPUT_LO/HI.

**Potential issue**: Different heads reading different source dims but writing to overlapping output dims could cause interference if attention computation has bugs.

---

## How to Test Each Hypothesis

### Test 1: Check Additivity

Add head 5 and log AX_CARRY values before/after. If values > 1.0 or negative, confirms additivity issue.

### Test 2: Check Layer Norm

Verify layer norm is applied to OUTPUT dims. If not, values could accumulate unboundedly.

### Test 3: Check ALiBi Slopes

```python
if hasattr(attn3, 'alibi_slopes') and attn3.alibi_slopes is not None:
    print(f"L3 alibi slopes: {attn3.alibi_slopes}")
```

If slope[5] is 0 or unset, attention scores explode.

### Test 4: Single-Step Test

Run a single step with head 5 enabled. If it fails on step 0, confirms edge case. If it works on step 0 but fails on step 1, confirms feedback loop or additivity.

### Test 5: Reduce Weight

Try weight 0.5 instead of 2.0 for head 5. If this works, confirms additivity issue (total weight would be 1.5 instead of 3.0).

---

## Recommended Fix Approach

### Option A: Use Different Dimension (BEST)

Instead of writing to AX_CARRY (which head 1 uses), write to a new dimension:

```python
# New dimension
BD.AX_FULL = 466  # 32 dims for full AX value from OUTPUT

# Head 5: Write to AX_FULL instead of AX_CARRY
for k in range(16):
    attn3.W_o[BD.AX_FULL + k, base + 1 + k] = 1.0
    attn3.W_o[BD.AX_FULL + 16 + k, base + 17 + k] = 1.0
```

Then L6 head 6 reads from AX_FULL instead of AX_CARRY.

**Pros**: No conflict with head 1, clean separation
**Cons**: Uses 32 dims (but we have ~79 available per dimension registry)

### Option B: Disable Head 1 When Head 5 Fires (COMPLEX)

Add negative weight from HAS_SE to head 1:

```python
# In head 1 setup:
attn3.W_q[base_head1, BD.HAS_SE] = -L  # Suppress on subsequent steps
```

**Pros**: Reuses existing dims
**Cons**: Complex interaction, might break byte 0 carry for other operations

### Option C: Use Subtraction Instead of Addition (HACKY)

Make head 5 write with negative weight to cancel head 1:

```python
# Head 5: Write with weight that cancels head 1's contribution
for k in range(16):
    attn3.W_o[BD.AX_CARRY_LO + k, base + 1 + k] = 1.0  # New value
    attn3.W_o[BD.AX_CARRY_LO + k, base_head1 + 1 + k] = -1.0  # Cancel head 1
```

**Pros**: Mathematically correct
**Cons**: Very hacky, fragile, hard to debug

### Option D: Move to Different Layer (MODERATE)

Use L4 or L5 attention instead of L3:

```python
# L4 head X: Copy AX OUTPUT → AX_CARRY
# No conflict because L4 doesn't have head 1 carry forward
```

**Pros**: No conflict, cleaner
**Cons**: Requires reworking L4/L5 head allocation, changes architecture

---

## Next Steps

1. **Test current setup** with head 5 to reproduce "all 1's" failure
2. **Log intermediate values** to confirm which hypothesis is correct
3. **Implement Option A** (new dimension) as the cleanest fix
4. **Verify** with 1-var and 2-var programs

---

## Implementation Timeline

1. Create test script that adds head 5 and runs simple program
2. Capture predictions to see if they become "all 1's"
3. Add logging to identify root cause
4. Implement Option A (new BD.AX_FULL dimension)
5. Test with full program suite

---

**Status**: Ready to test. Implementation found in git history (commit 10071e2). Failure mode documented but root cause unknown without testing.
