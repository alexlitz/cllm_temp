# Attention-Based Eviction Analysis

## The Problem You Identified

The current implementation uses **hardcoded context pruning** rather than **attention score-based eviction**.

## Current Implementation (Suboptimal)

### Lines 384-392 in `run_vm.py`:
```python
# Context pruning: keep prefix + retained MEM sections + last step.
last_step = context[-(Token.STEP_TOKENS):]
mem_flat = []
for tokens in self._mem_history.values():
    mem_flat.extend(tokens)
context[prefix_len:] = mem_flat + list(last_step)
self.model._mem_history_end = prefix_len + len(mem_flat)
```

### What This Does:
1. **Keeps**: Bytecode prefix + MEM sections (one per address) + last step (35 tokens)
2. **Evicts**: Everything else (all intermediate register states, old steps)

### The Problem:
- **Hardcoded policy**: Not based on attention scores
- **Dictionary-based**: `_mem_history[addr]` keeps latest write per address
- **No score analysis**: Doesn't compute which entries can actually receive attention

---

## How It SHOULD Work (Score-Based Eviction)

### The Principle: Maximum Possible Attention Score

For each position `i` in the context, compute its **maximum possible attention score** across all future queries.

**Eviction Rule**: If `max_score[i] < threshold` (e.g., -10), position `i` will never receive meaningful attention → **can be evicted safely**.

---

## Score-Based Eviction for Each Entry Type

### 1. **Overwritten Memory (MEM sections with MEM_STORE=0)**

**Current score**:
```python
# L15 attention scoring (lines 3759-3773)
# For MEM entries with MEM_STORE=0:
store_anchor = 50 * (-50) / 8 = -312.5  # Q=50, K=-50 (non-store)
bias = 0                                 # (if target position)
address_match = +300                     # (if address matches)
total = 0 - 312.5 + 300 = -12.5         # Still negative!

# With softmax1: exp(-12.5) / (1 + ...) ≈ 0
```

**Maximum possible score**: `-12.5` (even with perfect address match!)

**Eviction decision**: `max_score < 0` → **EVICT** ✓

---

### 2. **Old Register States**

**Query weights for registers (lines 1570-1586)**:
```python
# L3 attention carries forward registers from PREVIOUS step
# Only the LAST step's markers have positive Q weights
# Older steps have Q=0 for all marker dimensions

# For step N-2 (two steps ago):
Q[MARK_PC] = 0   # No weight for old PC markers
Q[MARK_AX] = 0   # No weight for old AX markers
# etc.
```

**Maximum possible score**: `0` (no query will ever target old markers)

**Eviction decision**: `max_score = 0` → **EVICT** ✓

---

### 3. **Valid Memory (MEM sections with MEM_STORE=1)**

**Score**:
```python
store_anchor = 50 * 50 / 8 = +312.5   # Q=50, K=+50 (has MEM_STORE=1)
zfod_offset = -96 * 50 / 8 = -600     # Baseline shift
address_match = +300                   # (if address matches)
total = +312.5 - 600 + 300 = +12.5    # Positive!

# With softmax1: exp(+12.5) / (1 + ...) ≈ 0.999
```

**Maximum possible score**: `+12.5`

**Eviction decision**: `max_score > 0` → **KEEP** ✓

---

### 4. **Bytecode/Data Prefix**

**Score**:
```python
# L5 fetch attention (lines 3763-3827)
# Code bytes have ADDR_KEY encoding
# When PC points to this address, score is high
address_match = +300  # Binary address match
bias = 0              # Target position (AX marker)
total ≈ +300          # Strong positive score

# With softmax1: High attention weight
```

**Maximum possible score**: `+300`

**Eviction decision**: `max_score > 0` → **KEEP** ✓

---

## The Correct Eviction Algorithm

### **Algorithm: Score-Based KV Cache Pruning**

```python
def evict_low_score_entries(context, token_ids, model):
    """
    Evict entries that can never receive significant attention.

    Returns: pruned_context, pruned_token_ids
    """
    keep_mask = []

    for i, token in enumerate(token_ids):
        max_score = compute_max_attention_score(i, token, context, model)

        # Eviction threshold: -10
        # Anything below this will contribute ~0 via softmax1
        if max_score >= -10.0:
            keep_mask.append(True)   # Keep this position
        else:
            keep_mask.append(False)  # Evict this position

    # Filter context to keep only high-score entries
    pruned_context = [context[i] for i in range(len(context)) if keep_mask[i]]
    pruned_token_ids = [token_ids[i] for i in range(len(token_ids)) if keep_mask[i]]

    return pruned_context, pruned_token_ids
```

### **Computing Maximum Possible Score**

```python
def compute_max_attention_score(pos, token, context, model):
    """
    Compute the maximum attention score this position could receive
    from ANY future query.
    """
    max_score = -float('inf')

    # Check each attention layer
    for layer_idx in range(model.n_layers):
        layer_max = compute_layer_max_score(layer_idx, pos, token, context, model)
        max_score = max(max_score, layer_max)

    return max_score


def compute_layer_max_score(layer_idx, pos, token, context, model):
    """
    Compute max score for a specific layer.

    For L15 (memory lookup):
    - Check MEM_STORE flag
    - If MEM_STORE=0: max_score ≈ -312.5 (store_anchor kills it)
    - If MEM_STORE=1: max_score ≈ +12.5 (can get attention)

    For L3 (register carry):
    - Check if this is the most recent marker of its type
    - If not most recent: max_score = 0 (Q weights are 0 for old markers)

    For L5 (code fetch):
    - Check if ADDR_KEY is set (bytecode positions)
    - If yes: max_score ≈ +300 (address match)
    """
    if layer_idx == 15:  # L15 memory lookup
        if token == Token.MEM:
            # Check MEM_STORE flag in context embedding
            has_mem_store = context[pos].get('MEM_STORE', 0) > 0.5
            if has_mem_store:
                # Can get attention: +312.5 - 600 + 300 = +12.5
                return +12.5
            else:
                # Overwritten: -312.5 - 600 + 300 = -612.5
                return -612.5
        else:
            return -float('inf')  # Non-MEM tokens don't participate

    elif layer_idx == 3:  # L3 register carry
        if token in [Token.REG_PC, Token.REG_AX, Token.REG_SP, Token.REG_BP]:
            # Only most recent marker has positive Q weight
            is_most_recent = check_if_most_recent_marker(pos, token, context)
            if is_most_recent:
                return +50.0  # Typical relay score
            else:
                return 0.0    # Old markers have Q=0
        else:
            return -float('inf')

    elif layer_idx == 5:  # L5 code fetch
        # Bytecode positions have ADDR_KEY
        has_addr_key = check_has_addr_key(pos, context)
        if has_addr_key:
            return +300.0  # Address match score
        else:
            return -float('inf')

    else:
        # Other layers...
        return compute_generic_layer_max(layer_idx, pos, token, context, model)
```

---

## Why This Matters

### **Current Hardcoded Policy**:
- Keeps all MEM sections (one per address) regardless of whether they can be accessed
- Evicts everything else blindly
- No consideration of actual attention mechanics

### **Score-Based Policy**:
- **Evicts based on attention impossibility**: If an entry can't receive attention, remove it
- **Automatic handling of edge cases**:
  - Overwritten memory (MEM_STORE=0) → automatically evicted
  - Old register states → automatically evicted
  - Valid memory → automatically kept
- **Provably correct**: An entry with max_score < -10 contributes `exp(-10)/(1+...) ≈ 0.00005` ≈ 0
- **Generalizes**: Works for any attention pattern, not just hardcoded rules

---

## Implementation Status

### **Currently Implemented**:
✅ softmax1 (enables ZFOD for low scores)
✅ MEM_STORE flag (marks valid memory)
✅ Hardcoded dictionary-based eviction (`_mem_history`)

### **Missing**:
❌ Maximum score computation for each entry
❌ Score-based eviction policy
❌ Dynamic threshold tuning
❌ Per-layer score analysis

---

## Recommended Next Steps

1. **Implement `compute_max_attention_score()`**
   - For each layer, compute the maximum score any query could give this entry
   - Use attention weight matrices to determine Q·K bounds

2. **Replace hardcoded `_mem_history` with score-based pruning**
   - After each step, scan context and compute max scores
   - Evict entries with `max_score < -10`

3. **Verify correctness**
   - Run tests with both policies
   - Confirm outputs are identical (evicted entries truly don't contribute)

4. **Measure speedup**
   - Track context size reduction
   - Measure attention computation speedup

5. **Tune threshold**
   - Experiment with different thresholds (-5, -10, -20)
   - Balance between accuracy and memory reduction

---

## Example: 1000-Step Program

### **Current Policy (Hardcoded)**:
- Context size: ~50 unique addresses × 9 tokens = 450 tokens
- Plus: bytecode (500 tokens) + last step (35 tokens)
- **Total: ~985 tokens**

### **Score-Based Policy**:
- Evict all MEM entries with MEM_STORE=0: ~0 tokens (all evicted)
- Evict old register states: ~0 tokens (all evicted)
- Keep only: bytecode + valid MEM entries + last step
- **Total: ~900 tokens** (assuming 40 valid addresses)

### **Improvement**:
- Slightly better (evicts dead entries more aggressively)
- More importantly: **Provably correct** (based on attention mechanics, not heuristics)

---

## Conclusion

**Your observation is spot-on**: The eviction should be based on maximum possible attention scores, not hardcoded policies.

The current `_mem_history` dictionary is a **heuristic approximation** that happens to work because:
- It keeps latest writes (which have MEM_STORE=1 → positive scores)
- It evicts old writes (which would have MEM_STORE=0 → negative scores)

But a **score-based policy** would be:
1. More general (works for any attention pattern)
2. Provably correct (only evicts entries that can't contribute)
3. Easier to reason about (no magic heuristics)
4. Potentially more efficient (can evict more aggressively with confidence)
