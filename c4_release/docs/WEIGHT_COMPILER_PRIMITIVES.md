# Weight Compiler Primitives

This document defines all primitive operations supported by the graph-based weight compiler.

Each primitive specifies:
- **Semantics**: What the operation does
- **Inputs**: Number and type of inputs
- **Outputs**: Number and type of outputs
- **Units**: Hidden units required
- **Pattern**: Weight setting pattern
- **Cost**: Computational cost

---

## Arithmetic Primitives

### ADD
**Semantics**: `out = a + b`

**Inputs**: 2 scalars (a, b)
**Outputs**: 1 scalar (out)
**Units**: 2 (cancel pair)

**Pattern**:
```
Unit 0: +a +b → out (weight: 1/S²)
Unit 1: -a -b → out (weight: 1/S²)
```

**Weights**:
```python
W_up[0, a_reg] = S
W_up[0, b_reg] = S
W_down[out_reg, 0] = 1/(S*S)

W_up[1, a_reg] = -S
W_up[1, b_reg] = -S
W_down[out_reg, 1] = 1/(S*S)
```

**Cost**: 2 units, 4 non-zero W_up, 2 non-zero W_down

---

### SUB
**Semantics**: `out = a - b`

**Inputs**: 2 scalars (a, b)
**Outputs**: 1 scalar (out)
**Units**: 2 (cancel pair)

**Pattern**:
```
Unit 0: +a -b → out
Unit 1: -a +b → out
```

**Weights**:
```python
W_up[0, a_reg] = S
W_up[0, b_reg] = -S
W_down[out_reg, 0] = 1/(S*S)

W_up[1, a_reg] = -S
W_up[1, b_reg] = S
W_down[out_reg, 1] = 1/(S*S)
```

**Cost**: 2 units, 4 non-zero W_up, 2 non-zero W_down

---

### MUL
**Semantics**: `out = a * b` (mod base)

**Inputs**: 2 one-hot vectors (a, b) of width N
**Outputs**: 1 one-hot vector (out) of width N
**Units**: 4*N (for N=16)

**Pattern**: Outer product lookup table
```
For each output value k:
  Unit 2k:   sum over (i,j) where i*j=k of: +a[i] +b[j]
  Unit 2k+1: sum over (i,j) where i*j=k of: -a[i] -b[j]
```

**Weights**:
```python
# For each output value k:
for k in range(base):
    unit_pos = 2*k
    unit_neg = 2*k + 1
    for i in range(base):
        for j in range(base):
            if (i * j) % base == k:
                W_up[unit_pos, a_reg + i] = S
                W_up[unit_pos, b_reg + j] = S
                W_up[unit_neg, a_reg + i] = -S
                W_up[unit_neg, b_reg + j] = -S

    W_down[out_reg + k, unit_pos] = 1/(S*S)
    W_down[out_reg + k, unit_neg] = 1/(S*S)
```

**Cost**: 2*base units, ~base² non-zero W_up entries per output value

---

### DIV
**Semantics**: `out = a / b` (integer division)

**Inputs**: 2 one-hot vectors (a, b) of width N
**Outputs**: 1 one-hot vector (out) of width N
**Units**: 4*N

**Pattern**: Similar to MUL, lookup table for division
```
For each output value q:
  sum over (a,b) where floor(a/b)=q
```

**Cost**: 2*base units

---

### MOD
**Semantics**: `out = a mod b`

**Inputs**: 2 scalars or one-hot vectors
**Outputs**: 1 scalar or one-hot vector
**Units**: 6 (for scalar), 4*N (for one-hot)

**Pattern** (scalar):
```
Unit 0-1: raw_sum = a + carry_in (cancel pair)
Unit 2-3: result = raw_sum - base*step(raw_sum >= base)
Unit 4-5: carry_out = step(raw_sum >= base)
```

**Cost**: 6 units (scalar), 2*base units (one-hot)

---

## Comparison Primitives

### CMP_EQ
**Semantics**: `out = (a == b)` → 1 if equal, 0 otherwise

**Inputs**: 2 scalars (a, b)
**Outputs**: 1 scalar (out)
**Units**: 4

**Pattern**:
```
Unit 0: step(a - b >= 0)    → contributes +1 if a >= b
Unit 1: step(b - a >= 0)    → contributes +1 if b >= a
Unit 2: -step(a - b >= 1)   → contributes -1 if a > b
Unit 3: -step(b - a >= 1)   → contributes -1 if b > a

Result: (a >= b) + (b >= a) - (a > b) - (b > a)
      = 1 if a == b, 0 otherwise
```

**Weights**:
```python
# Unit 0: a >= b
W_up[0, a_reg] = S
W_up[0, b_reg] = -S
b_up[0] = S
W_down[out_reg, 0] = 1/S

# Unit 1: b >= a
W_up[1, a_reg] = -S
W_up[1, b_reg] = S
b_up[1] = S
W_down[out_reg, 1] = 1/S

# Unit 2: -(a > b)
W_up[2, a_reg] = S
W_up[2, b_reg] = -S
b_up[2] = 0
W_down[out_reg, 2] = -1/S

# Unit 3: -(b > a)
W_up[3, a_reg] = -S
W_up[3, b_reg] = S
b_up[3] = 0
W_down[out_reg, 3] = -1/S
```

**Cost**: 4 units, 8 non-zero W_up, 4 non-zero W_down

---

### CMP_LT
**Semantics**: `out = (a < b)` → 1 if a < b, 0 otherwise

**Inputs**: 2 scalars (a, b)
**Outputs**: 1 scalar (out)
**Units**: 2

**Pattern**:
```
Unit 0: step(b - a >= 1)   → 1 if b > a (i.e., a < b)
Unit 1: -step(b - a >= 0)  → -1 if b >= a
```

**Weights**:
```python
# Unit 0: b - a >= 1
W_up[0, a_reg] = -S
W_up[0, b_reg] = S
b_up[0] = 0
W_down[out_reg, 0] = 1/S

# Unit 1: -(b - a >= 0)
W_up[1, a_reg] = -S
W_up[1, b_reg] = S
b_up[1] = S
W_down[out_reg, 1] = -1/S
```

**Cost**: 2 units, 4 non-zero W_up, 2 non-zero W_down

---

### CMP_LE
**Semantics**: `out = (a <= b)` → 1 if a <= b, 0 otherwise

**Inputs**: 2 scalars
**Outputs**: 1 scalar
**Units**: 2

**Pattern**: Equivalent to `!(a > b)` or `b >= a`
```
Unit 0: step(b - a >= 0)
Unit 1: -step(b - a >= 1) [optional for robustness]
```

**Cost**: 2 units

---

### CMP_GT
**Semantics**: `out = (a > b)`

**Inputs**: 2 scalars
**Outputs**: 1 scalar
**Units**: 2

**Pattern**: Equivalent to `b < a`
```
Unit 0: step(a - b >= 1)
Unit 1: -step(a - b >= 0)
```

**Cost**: 2 units

---

### CMP_GE
**Semantics**: `out = (a >= b)`

**Inputs**: 2 scalars
**Outputs**: 1 scalar
**Units**: 2

**Pattern**:
```
Unit 0: step(a - b >= 0)
Unit 1: -step(a - b >= 1)
```

**Cost**: 2 units

---

### CMP_NE
**Semantics**: `out = (a != b)`

**Inputs**: 2 scalars
**Outputs**: 1 scalar
**Units**: 2 (or 1 via NOT(CMP_EQ))

**Pattern**: Equivalent to `NOT(CMP_EQ(a, b))`

**Cost**: 2 units (if fused), 6 units (if composed with NOT)

---

## Logical Primitives

### AND
**Semantics**: `out = a && b` (logical AND)

**Inputs**: 2 scalars (0 or 1)
**Outputs**: 1 scalar (0 or 1)
**Units**: 2

**Pattern**: Step function on sum
```
Unit 0: step(a + b >= 2)  → 1 if both inputs are 1
Unit 1: -step(a + b >= 3) → never triggers (impossible)
```

**Weights**:
```python
# Unit 0: a + b >= 2
W_up[0, a_reg] = S
W_up[0, b_reg] = S
b_up[0] = -S  # threshold = 2, bias = -S*(2-1) = -S
W_down[out_reg, 0] = 1/S

# Unit 1: -(a + b >= 3)
W_up[1, a_reg] = S
W_up[1, b_reg] = S
b_up[1] = -2*S
W_down[out_reg, 1] = -1/S
```

**Cost**: 2 units

---

### OR
**Semantics**: `out = a || b` (logical OR)

**Inputs**: 2 scalars (0 or 1)
**Outputs**: 1 scalar (0 or 1)
**Units**: 2

**Pattern**: Step function on sum
```
Unit 0: step(a + b >= 1)  → 1 if at least one input is 1
Unit 1: -step(a + b >= 2) → 0 if both inputs are 1 (correction)
```

**Weights**:
```python
# Unit 0: a + b >= 1
W_up[0, a_reg] = S
W_up[0, b_reg] = S
b_up[0] = 0
W_down[out_reg, 0] = 1/S

# Unit 1: -(a + b >= 2)
W_up[1, a_reg] = S
W_up[1, b_reg] = S
b_up[1] = -S
W_down[out_reg, 1] = -1/S
```

**Cost**: 2 units

---

### NOT
**Semantics**: `out = !a` (logical NOT)

**Inputs**: 1 scalar (0 or 1)
**Outputs**: 1 scalar (0 or 1)
**Units**: 2

**Pattern**: `out = 1 - a`
```
Unit 0: const(1) via bias
Unit 1: -a via W_up
```

**Weights**:
```python
# Unit 0: constant 1
b_up[0] = S
W_down[out_reg, 0] = 1/S

# Unit 1: -a
W_up[1, a_reg] = -S
W_down[out_reg, 1] = 1/S
```

**Cost**: 2 units

---

### XOR
**Semantics**: `out = a ^ b` (logical XOR)

**Inputs**: 2 scalars (0 or 1)
**Outputs**: 1 scalar (0 or 1)
**Units**: 4 (or compose: `(a || b) && !(a && b)`)

**Pattern**: `(a + b) - 2*(a && b)`
```
Unit 0-1: sum = a + b (cancel pair)
Unit 2-3: both = a && b (step >= 2)
Result: sum - 2*both
```

**Cost**: 4 units (direct), 6 units (composed)

---

## Conditional Primitives

### SELECT (MUX)
**Semantics**: `out = cond ? a : b`

**Inputs**: 3 scalars (cond, a, b)
**Outputs**: 1 scalar (out)
**Units**: 4

**Pattern**:
```
Unit 0-1: cond * a (gated by cond)
Unit 2-3: (1 - cond) * b (gated by NOT(cond))
Result: sum of both branches
```

**Weights**:
```python
# Branch A: cond * a
W_up[0, a_reg] = S
W_gate[0, cond_reg] = S  # Gate by condition
W_down[out_reg, 0] = 1/(S*S)

W_up[1, a_reg] = -S
W_gate[1, cond_reg] = -S
W_down[out_reg, 1] = 1/(S*S)

# Branch B: (1-cond) * b
W_up[2, b_reg] = S
W_gate[2, not_cond_reg] = S  # Gate by NOT(cond)
W_down[out_reg, 2] = 1/(S*S)

W_up[3, b_reg] = -S
W_gate[3, not_cond_reg] = -S
W_down[out_reg, 3] = 1/(S*S)
```

**Cost**: 4 units (requires NOT(cond) as input, add 2 units if not available)

---

### IF_THEN
**Semantics**: `if cond: out = a` (conditional assignment)

**Inputs**: 2 scalars (cond, a)
**Outputs**: 1 scalar (out)
**Units**: 2

**Pattern**: Gated relay
```
Unit 0-1: relay a, gated by cond
```

**Cost**: 2 units

---

## Memory Primitives

### LOOKUP
**Semantics**: `out = memory[addr]` (read from attention-based memory)

**Inputs**: 1 address (one-hot or scalar)
**Outputs**: 1 value vector
**Units**: 0 (uses attention, not FFN)

**Pattern**: Attention-based memory read
- Query: addr embedding
- Keys: stored address embeddings
- Values: stored value vectors
- Output: Σ attention_weight[i] * value[i]

**Implementation**: Not FFN-based, uses attention layer

**Cost**: 0 FFN units

---

### STORE
**Semantics**: `memory[addr] = value` (write to attention-based memory)

**Inputs**: 1 address, 1 value vector
**Outputs**: None (side effect: adds to KV cache)
**Units**: 0 (uses attention mechanism)

**Pattern**: Store (addr, value) pair in KV cache

**Implementation**:
- Emit token sequence: [STORE_MARKER, addr, value_byte0, value_byte1, ...]
- Attention mechanism stores as KV pair

**Cost**: 0 FFN units, O(value_size) tokens

---

## Register Operations

### MOVE (RELAY)
**Semantics**: `out = in` (copy register)

**Inputs**: 1 scalar or vector
**Outputs**: 1 scalar or vector (same type)
**Units**: 2 (scalar), 2*N (vector of width N)

**Pattern** (scalar):
```
Unit 0: +in
Unit 1: -(-in)
Result: in
```

**Pattern** (vector):
```
For each dimension i:
  Unit 2i:   +in[i] → out[i]
  Unit 2i+1: -in[i] → out[i]
```

**Weights** (scalar):
```python
W_up[0, in_reg] = S
W_down[out_reg, 0] = 1/S

W_up[1, in_reg] = -S
W_down[out_reg, 1] = 1/S
```

**Cost**: 2 units (scalar), 2*width units (vector)

---

### CLEAR
**Semantics**: `reg = 0` (zero register)

**Inputs**: None (implicit: current register value)
**Outputs**: 1 scalar or vector (zeroed)
**Units**: 2 (scalar), 2*N (vector)

**Pattern**: Subtract register from itself
```
Unit 0: +reg → -reg (negative contribution)
Unit 1: -reg → -reg (negative contribution)
Result: reg = reg - reg = 0
```

**Weights**:
```python
W_up[0, reg] = S
W_down[reg, 0] = -1/S

W_up[1, reg] = -S
W_down[reg, 1] = -1/S
```

**Cost**: 2 units (scalar), 2*width units (vector)

---

### SWAP
**Semantics**: `(a, b) = (b, a)` (exchange registers)

**Inputs**: 2 registers (a, b)
**Outputs**: 2 registers (swapped)
**Units**: 8 (4 for each direction)

**Pattern**: Compose with 3 MOVE operations
```
temp = a
a = b
b = temp
```

**Cost**: 3 MOVE operations = 6 units

---

## Constant Operations

### CONST
**Semantics**: `out = constant` (load immediate value)

**Inputs**: None (constant baked in bias)
**Outputs**: 1 scalar
**Units**: 2

**Pattern**: Bias-only activation
```
Unit 0: bias = S * constant
Unit 1: bias = -S * constant (for stability)
Result: constant
```

**Weights**:
```python
b_up[0] = S * constant
W_down[out_reg, 0] = 1/S

b_up[1] = -S * constant
W_down[out_reg, 1] = 1/S
```

**Cost**: 2 units

---

## Bit Operations (One-Hot Encoding)

### BIT_AND
**Semantics**: `out = a & b` (bitwise AND)

**Inputs**: 2 one-hot vectors (a, b) of width N
**Outputs**: 1 one-hot vector (out) of width N
**Units**: 2*N

**Pattern**: Lookup table
```
For each output bit k:
  sum over (i,j) where (i & j) = k
```

**Cost**: 2*N units

---

### BIT_OR
**Semantics**: `out = a | b` (bitwise OR)

**Inputs**: 2 one-hot vectors
**Outputs**: 1 one-hot vector
**Units**: 2*N

**Pattern**: Lookup table similar to BIT_AND

**Cost**: 2*N units

---

### BIT_XOR
**Semantics**: `out = a ^ b` (bitwise XOR)

**Inputs**: 2 one-hot vectors
**Outputs**: 1 one-hot vector
**Units**: 2*N

**Cost**: 2*N units

---

### SHL (Shift Left)
**Semantics**: `out = a << n` (shift left by n bits)

**Inputs**: 1 one-hot vector (a), 1 scalar (n)
**Outputs**: 1 one-hot vector (out)
**Units**: 2*N*N (all possible shifts)

**Pattern**: Lookup table indexed by (a, n)

**Cost**: 2*N*N units (expensive!)

**Optimization**: If shift amount is constant, use 2*N units

---

### SHR (Shift Right)
**Semantics**: `out = a >> n` (shift right by n bits)

**Inputs**: 1 one-hot vector (a), 1 scalar (n)
**Outputs**: 1 one-hot vector (out)
**Units**: 2*N*N

**Cost**: 2*N*N units (or 2*N for constant shift)

---

## Summary Table

| Primitive | Inputs | Outputs | Units | Cost |
|-----------|--------|---------|-------|------|
| ADD | 2 scalars | 1 scalar | 2 | Low |
| SUB | 2 scalars | 1 scalar | 2 | Low |
| MUL | 2 one-hot | 1 one-hot | 2*N | Medium |
| DIV | 2 one-hot | 1 one-hot | 2*N | Medium |
| MOD | 2 scalars/one-hot | 1 scalar/one-hot | 6 / 2*N | Medium |
| CMP_EQ | 2 scalars | 1 scalar | 4 | Low |
| CMP_LT | 2 scalars | 1 scalar | 2 | Low |
| CMP_LE | 2 scalars | 1 scalar | 2 | Low |
| CMP_GT | 2 scalars | 1 scalar | 2 | Low |
| CMP_GE | 2 scalars | 1 scalar | 2 | Low |
| AND | 2 scalars | 1 scalar | 2 | Low |
| OR | 2 scalars | 1 scalar | 2 | Low |
| NOT | 1 scalar | 1 scalar | 2 | Low |
| XOR | 2 scalars | 1 scalar | 4 | Low |
| SELECT | 3 scalars | 1 scalar | 4-6 | Low-Medium |
| IF_THEN | 2 scalars | 1 scalar | 2 | Low |
| LOOKUP | 1 addr | 1 vector | 0 | Attention |
| STORE | 1 addr, 1 value | none | 0 | Attention |
| MOVE | 1 scalar/vector | 1 scalar/vector | 2 / 2*N | Low |
| CLEAR | none | 1 scalar/vector | 2 / 2*N | Low |
| SWAP | 2 registers | 2 registers | 6 | Low |
| CONST | none | 1 scalar | 2 | Low |
| BIT_AND | 2 one-hot | 1 one-hot | 2*N | Medium |
| BIT_OR | 2 one-hot | 1 one-hot | 2*N | Medium |
| BIT_XOR | 2 one-hot | 1 one-hot | 2*N | Medium |
| SHL | 1 one-hot, 1 scalar | 1 one-hot | 2*N² | High |
| SHR | 1 one-hot, 1 scalar | 1 one-hot | 2*N² | High |

## Design Principles

1. **Cancel Pair Pattern**: Most operations use paired units (+/-) for numerical stability
2. **Step Function Pattern**: Comparisons use step(x >= threshold) with bias adjustment
3. **Lookup Tables**: One-hot operations use precomputed lookup tables
4. **Gating**: Conditional execution via W_gate matrix
5. **Attention for Memory**: LOOKUP/STORE use attention mechanism, not FFN
6. **Cost Model**: Each primitive has predictable unit cost for allocation

## Usage in Compiler

```python
# Arithmetic
graph.add("a", "b", "sum")           # sum = a + b
graph.sub("a", "b", "diff")          # diff = a - b
graph.mul("a", "b", "product")       # product = a * b

# Comparison
graph.cmp_eq("a", "b", "is_equal")   # is_equal = (a == b)
graph.cmp_lt("a", "b", "is_less")    # is_less = (a < b)

# Logical
graph.logical_and("x", "y", "both")  # both = x && y
graph.logical_or("x", "y", "either") # either = x || y
graph.logical_not("x", "not_x")      # not_x = !x

# Conditional
graph.select("cond", "a", "b", "result")  # result = cond ? a : b

# Memory
graph.lookup("addr", "value")        # value = memory[addr]
graph.store("addr", "value")         # memory[addr] = value

# Register ops
graph.move("src", "dst")             # dst = src
graph.clear("reg")                   # reg = 0
graph.const(42, "fortytwo")          # fortytwo = 42
```

## Next Steps

1. Implement all primitives in WeightEmitter
2. Add cost model for register allocation
3. Test each primitive independently
4. Build composite operations (e.g., nibble addition using primitives)
5. Benchmark hidden unit usage vs manual weight setting
